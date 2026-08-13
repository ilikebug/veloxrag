from __future__ import annotations

import json
from typing import Any
from uuid import UUID, uuid4

import httpx
import pytest

from rag_service.mcp.server import (
    DEFAULT_BASE_URL,
    RagMcpConfigError,
    _Settings,
    build_server,
)

_KNOWLEDGE_BASE_ID = UUID("00000000-0000-4000-8000-0000000000b1")


def _environ(**overrides: str) -> dict[str, str]:
    values = {
        "RAG_MCP_TOKEN": "agent-token-sentinel",
        "RAG_MCP_KNOWLEDGE_BASE": str(_KNOWLEDGE_BASE_ID),
    }
    values.update(overrides)
    return values


def _settings(**overrides: str) -> _Settings:
    return _Settings(_environ(**overrides))


@pytest.mark.parametrize(
    "overrides",
    [
        {"RAG_MCP_KNOWLEDGE_BASE": "not-a-uuid"},
        {"RAG_MCP_BASE_URL": "ftp://elsewhere"},
    ],
)
def test_settings_refuse_unusable_configuration(overrides: dict[str, str]) -> None:
    # Failing at startup is the only chance to say what is missing: an MCP client
    # shows stderr when a server exits, and nothing after that.
    with pytest.raises(RagMcpConfigError):
        _settings(**overrides)


def test_settings_accept_no_token_and_no_knowledge_base() -> None:
    # Both are optional so a local-trusted service with one knowledge base needs
    # no environment at all — the client config is just the command.
    settings = _Settings({})

    assert settings.token == ""
    assert settings.knowledge_base_id is None


def test_settings_default_to_the_local_service() -> None:
    settings = _settings()

    assert settings.base_url == DEFAULT_BASE_URL
    assert settings.knowledge_base_id == _KNOWLEDGE_BASE_ID


def test_settings_strip_a_trailing_slash_from_the_base_url() -> None:
    # Paths are joined with a leading slash, so a trailing one would double it.
    assert _settings(RAG_MCP_BASE_URL="http://127.0.0.1:8000/").base_url == "http://127.0.0.1:8000"


@pytest.mark.asyncio
async def test_exposed_tools_are_read_only() -> None:
    server = build_server(_settings())

    tools = await server.list_tools()

    # Deliberately no ingest, no key minting and no knowledge base creation: an
    # agent that can provision storage can also destroy it, and deletion here is
    # real.
    assert {tool.name for tool in tools} == {
        "search_memory",
        "list_documents",
        "memory_status",
    }


class _Recorder:
    def __init__(self, response: httpx.Response) -> None:
        self.response = response
        self.requests: list[httpx.Request] = []

    async def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.response


def _patched_client(monkeypatch: pytest.MonkeyPatch, recorder: _Recorder) -> None:
    def factory(settings: Any) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=settings.base_url,
            transport=httpx.MockTransport(recorder.handle),
        )

    monkeypatch.setattr("rag_service.mcp.server._client", factory)


async def _call(server: Any, name: str, arguments: dict[str, object]) -> str:
    result = await server.call_tool(name, arguments)
    content = result.content if hasattr(result, "content") else result
    first = content[0] if isinstance(content, (list, tuple)) else content
    return str(getattr(first, "text", first))


@pytest.mark.asyncio
async def test_search_sends_the_query_and_returns_passages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = _Recorder(
        httpx.Response(
            200,
            json={
                "results": [
                    {
                        "text": "the chunk size diluted the embedding",
                        "score": 0.82,
                        "document_id": str(uuid4()),
                        "title_path": ["Turn 12"],
                        "source": {"filename": "session.md", "start_offset": 0, "end_offset": 36},
                        "metadata": {"source_type": "chat"},
                    }
                ]
            },
        )
    )
    _patched_client(monkeypatch, recorder)
    server = build_server(_settings())

    rendered = await _call(server, "search_memory", {"query": "why did recall drop", "top_k": 3})

    request = recorder.requests[0]
    assert request.url.path == f"/v1/knowledge-bases/{_KNOWLEDGE_BASE_ID}/search"
    assert json.loads(request.content) == {
        "query": "why did recall drop",
        "top_k": 3,
        "rerank": False,
    }
    payload = json.loads(rendered)
    # The offsets travel with the passage: a hit cut mid-sentence is only
    # useful if the agent can tell where it sits in the document.
    assert payload[0]["source"]["end_offset"] == 36
    assert payload[0]["metadata"]["source_type"] == "chat"


@pytest.mark.asyncio
async def test_search_passes_rerank_and_a_source_filter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = _Recorder(httpx.Response(200, json={"results": []}))
    _patched_client(monkeypatch, recorder)
    server = build_server(_settings())

    await _call(
        server,
        "search_memory",
        {"query": "q", "rerank": True, "source_type": "doc"},
    )

    assert json.loads(recorder.requests[0].content) == {
        "query": "q",
        "top_k": 5,
        "rerank": True,
        "filters": {"metadata": {"source_type": "doc"}},
    }


@pytest.mark.asyncio
async def test_search_reports_an_empty_result_as_words(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patched_client(monkeypatch, _Recorder(httpx.Response(200, json={"results": []})))
    server = build_server(_settings())

    # Not "[]": an agent reading an empty array tends to retry the same query,
    # while a sentence tells it the memory simply has nothing.
    assert "No matching passages" in await _call(server, "search_memory", {"query": "q"})


@pytest.mark.asyncio
async def test_failures_surface_the_service_error_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patched_client(
        monkeypatch,
        _Recorder(
            httpx.Response(
                409,
                json={"error": {"code": "RERANK_NOT_CONFIGURED", "message": "..."}},
            )
        ),
    )
    server = build_server(_settings())

    rendered = await _call(server, "search_memory", {"query": "q", "rerank": True})

    # The code is what lets an agent distinguish a missing reranker from an
    # outage and decide whether retrying without rerank is worth it.
    assert "RERANK_NOT_CONFIGURED" in rendered


@pytest.mark.asyncio
async def test_status_reports_the_bound_knowledge_base(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patched_client(monkeypatch, _Recorder(httpx.Response(200, json={})))
    server = build_server(_settings())

    payload = json.loads(await _call(server, "memory_status", {}))

    assert payload["knowledge_base_id"] == str(_KNOWLEDGE_BASE_ID)
    assert payload["retrieval_ready"] is True


@pytest.mark.asyncio
async def test_status_explains_an_unreachable_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Broken(_Recorder):
        async def handle(self, request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused")

    _patched_client(monkeypatch, _Broken(httpx.Response(200)))
    server = build_server(_settings())

    rendered = await _call(server, "memory_status", {})

    # An agent cannot start a docker stack, but it can tell the human which
    # command to run.
    assert "make start" in rendered


@pytest.mark.asyncio
async def test_no_token_sends_no_authorization_header(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = _Recorder(httpx.Response(200, json={"results": []}))
    _patched_client(monkeypatch, recorder)
    monkeypatch.setattr(
        "rag_service.mcp.server._client",
        lambda settings: httpx.AsyncClient(
            base_url=settings.base_url, transport=httpx.MockTransport(recorder.handle)
        ),
    )
    server = build_server(_settings(RAG_MCP_TOKEN=""))

    await _call(server, "search_memory", {"query": "q"})

    # An empty bearer would be rejected outright; the service's local-trusted
    # mode only covers requests that carry no credential at all.
    assert "authorization" not in {k.lower() for k in recorder.requests[0].headers}


class _ListingRecorder:
    def __init__(self, items: list[dict[str, object]]) -> None:
        self.items = items
        self.paths: list[str] = []

    async def handle(self, request: httpx.Request) -> httpx.Response:
        self.paths.append(request.url.path)
        if request.url.path == "/v1/knowledge-bases":
            return httpx.Response(200, json={"items": self.items})
        return httpx.Response(200, json={"results": []})


@pytest.mark.asyncio
async def test_a_single_knowledge_base_is_resolved_without_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    only = str(uuid4())
    recorder = _ListingRecorder([{"id": only, "status": "active"}])
    monkeypatch.setattr(
        "rag_service.mcp.server._client",
        lambda settings: httpx.AsyncClient(
            base_url=settings.base_url, transport=httpx.MockTransport(recorder.handle)
        ),
    )
    server = build_server(_Settings({}))

    await _call(server, "search_memory", {"query": "q"})
    await _call(server, "search_memory", {"query": "again"})

    assert f"/v1/knowledge-bases/{only}/search" in recorder.paths
    # Resolved once and remembered: a lookup per search would add a round trip
    # for an answer that cannot change while the server runs.
    assert recorder.paths.count("/v1/knowledge-bases") == 1


@pytest.mark.asyncio
async def test_several_knowledge_bases_refuse_to_be_guessed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = _ListingRecorder(
        [{"id": str(uuid4()), "status": "active"}, {"id": str(uuid4()), "status": "active"}]
    )
    monkeypatch.setattr(
        "rag_service.mcp.server._client",
        lambda settings: httpx.AsyncClient(
            base_url=settings.base_url, transport=httpx.MockTransport(recorder.handle)
        ),
    )
    server = build_server(_Settings({}))

    rendered = await _call(server, "search_memory", {"query": "q"})

    # Picking one would silently search the wrong memory.
    assert "RAG_MCP_KNOWLEDGE_BASE" in rendered
