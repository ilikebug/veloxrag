"""MCP surface over the retrieval API, for local agents to use as memory.

The HTTP API is complete but not usable by an agent on its own: reaching it
means minting an admin token from a container CLI, signing an agent key,
remembering a knowledge base id, and hand-writing requests. That ceremony
protects a multi-tenant deployment; for one person running this next to their
editor it is only friction, and an agent cannot perform it at all.

So this server holds the credential and the knowledge base, and exposes what an
agent actually needs: search, see what is searchable, and check the service is
up. Nothing here mints keys, creates knowledge bases or ingests — an agent that
can provision its own storage can also destroy it, and deletion in this service
is now real.

Configured entirely by environment so the whole thing is one command in an MCP
client's config:

    RAG_MCP_BASE_URL       default http://127.0.0.1:8000
    RAG_MCP_TOKEN          optional; omit it when the service runs with
                           RAG_LOCAL_TRUSTED_AUTH=true (which `make start` sets)
    RAG_MCP_KNOWLEDGE_BASE optional; resolved automatically when the service has
                           exactly one knowledge base

With a local-trusted service and a single knowledge base, both are unnecessary
and the client config is just the command.
"""

from __future__ import annotations

import json
import os
from typing import Annotated, Any
from uuid import UUID

import httpx
from mcp.server import MCPServer
from pydantic import Field

DEFAULT_BASE_URL = "http://127.0.0.1:8000"
_REQUEST_TIMEOUT_SECONDS = 120.0
# Reranking adds a provider round trip per query. Left to the agent to ask for,
# because only the agent knows whether this question is worth the latency.
_MAX_TOP_K = 50
_DEFAULT_READ_END = 2000
_MAX_READ_CODEPOINTS = 8000


class RagMcpConfigError(Exception):
    """Startup configuration failure, safe to print."""


class _Settings:
    __slots__ = ("base_url", "knowledge_base_id", "token")

    base_url: str
    knowledge_base_id: UUID | None
    token: str

    def __init__(self, environ: dict[str, str]) -> None:
        base_url = environ.get("RAG_MCP_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
        raw_knowledge_base = environ.get("RAG_MCP_KNOWLEDGE_BASE", "").strip()
        if not base_url.startswith(("http://", "https://")):
            raise RagMcpConfigError("RAG_MCP_BASE_URL must be an http or https URL")
        knowledge_base_id: UUID | None = None
        if raw_knowledge_base:
            try:
                knowledge_base_id = UUID(raw_knowledge_base)
            except ValueError:
                raise RagMcpConfigError(
                    "RAG_MCP_KNOWLEDGE_BASE must be a knowledge base UUID"
                ) from None
        self.base_url = base_url
        self.token = environ.get("RAG_MCP_TOKEN", "").strip()
        self.knowledge_base_id = knowledge_base_id


def _client(settings: _Settings) -> httpx.AsyncClient:
    # No Authorization header at all when no token is configured: the service's
    # local-trusted mode only applies to unauthenticated requests, so sending an
    # empty bearer would be rejected rather than falling through.
    headers = {"Authorization": f"Bearer {settings.token}"} if settings.token else {}
    return httpx.AsyncClient(
        base_url=settings.base_url,
        headers=headers,
        timeout=_REQUEST_TIMEOUT_SECONDS,
    )


async def _resolve_knowledge_base(settings: _Settings, client: httpx.AsyncClient) -> UUID:
    """Find the knowledge base when none was configured.

    Only unambiguous when there is exactly one. Guessing among several would
    silently search the wrong memory, which is worse than refusing.
    """
    if settings.knowledge_base_id is not None:
        return settings.knowledge_base_id
    response = await client.get("/v1/knowledge-bases", params={"limit": 2})
    if response.status_code != 200:
        raise RagMcpConfigError(_failure(response))
    items = [item for item in response.json().get("items", ()) if item.get("status") == "active"]
    if not items:
        raise RagMcpConfigError("The service has no knowledge base yet")
    if len(items) > 1:
        raise RagMcpConfigError(
            "The service has more than one knowledge base; set RAG_MCP_KNOWLEDGE_BASE"
        )
    resolved = UUID(str(items[0]["id"]))
    # Cached for the process: resolving per call would add a round trip to every
    # search for an answer that does not change while the server runs.
    settings.knowledge_base_id = resolved
    return resolved


def _failure(response: httpx.Response) -> str:
    """Turn a service error into something an agent can act on.

    The service returns a machine code and no detail by design. Passing that
    code through beats a generic failure, because the agent can tell "this
    knowledge base has no reranker configured" from "the provider is down".
    """
    try:
        body = response.json()
        code = body["error"]["code"]
    except Exception:
        code = f"HTTP_{response.status_code}"
    return f"Request failed: {code} (status {response.status_code})"


def build_server(settings: _Settings) -> MCPServer:
    server = MCPServer(
        name="rag-memory",
        instructions=(
            "Searchable memory over documents and past sessions. Use search_memory "
            "before answering questions about this project's history, decisions, or "
            "documentation; the answer is often already recorded. Results are "
            "passages, not whole documents, and each carries the document it came "
            "from with the character range it occupies there.\n\n"
            "Judging relevance is your job, not the search's. Ask for more "
            "passages than you expect to use, keep the ones that look right, and "
            "call read_document on those to see the text around them before "
            "deciding. Measured on this corpus, the answer sits just outside the "
            "matched passage often enough that reading around a hit takes the "
            "chance of having it in hand from 0.89 to 1.00; a passage that looks "
            "truncated is worth widening rather than discarding."
        ),
    )

    @server.tool(
        description=(
            "Search indexed memory and return the passages that match. Prefer a "
            "specific question over keywords. Retrieving more than you need and "
            "picking among them yourself works better than trusting the first "
            "result: the ranking is vector similarity, which cannot tell that a "
            "passage merely repeats the question. Leave rerank off unless this "
            "service has a rerank profile configured — the default setup has "
            "none, and asking for it fails with RERANK_NOT_CONFIGURED."
        )
    )
    async def search_memory(
        query: Annotated[str, Field(description="What to look for, phrased as a question")],
        top_k: Annotated[int, Field(default=5, ge=1, le=_MAX_TOP_K)] = 5,
        rerank: Annotated[bool, Field(default=False)] = False,
        source_type: Annotated[
            str | None,
            Field(default=None, description="Restrict to one kind, e.g. 'doc' or 'chat'"),
        ] = None,
    ) -> str:
        payload: dict[str, Any] = {"query": query, "top_k": top_k, "rerank": rerank}
        if source_type:
            payload["filters"] = {"metadata": {"source_type": source_type}}
        async with _client(settings) as client:
            try:
                knowledge_base_id = await _resolve_knowledge_base(settings, client)
            except RagMcpConfigError as error:
                return str(error)
            response = await client.post(
                f"/v1/knowledge-bases/{knowledge_base_id}/search",
                json=payload,
            )
        if response.status_code != 200:
            return _failure(response)
        results = response.json().get("results", ())
        if not results:
            return "No matching passages."
        return json.dumps(
            [
                {
                    "text": item.get("text", ""),
                    "score": item.get("score"),
                    "document_id": item.get("document_id"),
                    "title_path": item.get("title_path", []),
                    "source": item.get("source", {}),
                    "metadata": item.get("metadata", {}),
                }
                for item in results
            ],
            ensure_ascii=False,
            indent=1,
        )

    @server.tool(
        description=(
            "Read a document's text around a character range, to see what a "
            "search result was cut off from. Pass the document_id and the "
            "source.start_offset / source.end_offset of a passage, widened by a "
            "few hundred characters on each side. Offsets past either end are "
            "clamped rather than refused, and total_codepoints tells a clamped "
            "range from an exhausted one."
        )
    )
    async def read_document(
        document_id: Annotated[str, Field(description="From a search result")],
        start: Annotated[int, Field(default=0, ge=0)] = 0,
        end: Annotated[int, Field(default=_DEFAULT_READ_END, ge=1)] = _DEFAULT_READ_END,
    ) -> str:
        try:
            identifier = UUID(document_id)
        except (AttributeError, TypeError, ValueError):
            return "document_id is not a valid id; take it from a search result."
        if end <= start:
            return "end must be greater than start."
        # Bounded here rather than left to the caller: an agent widening a hit
        # wants context, and a request for the whole document would spend its
        # window on text it did not ask about.
        if end - start > _MAX_READ_CODEPOINTS:
            end = start + _MAX_READ_CODEPOINTS
        async with _client(settings) as client:
            response = await client.get(
                f"/v1/documents/{identifier}/content",
                params={"start": start, "end": end},
            )
        if response.status_code != 200:
            return _failure(response)
        body = response.json()
        return json.dumps(
            {
                "document_id": body.get("document_id"),
                "start_offset": body.get("start_offset"),
                "end_offset": body.get("end_offset"),
                "total_codepoints": body.get("total_codepoints"),
                "text": body.get("text", ""),
            },
            ensure_ascii=False,
            indent=1,
        )

    @server.tool(
        description=(
            "List what is indexed, so a search can be narrowed or a gap in the "
            "memory can be recognised rather than guessed at."
        )
    )
    async def list_documents(
        limit: Annotated[int, Field(default=20, ge=1, le=100)] = 20,
    ) -> str:
        async with _client(settings) as client:
            try:
                knowledge_base_id = await _resolve_knowledge_base(settings, client)
            except RagMcpConfigError as error:
                return str(error)
            response = await client.get(
                f"/v1/knowledge-bases/{knowledge_base_id}/documents",
                params={"limit": limit},
            )
        if response.status_code != 200:
            return _failure(response)
        items = response.json().get("items", ())
        if not items:
            return "This knowledge base has no documents yet."
        return json.dumps(
            [
                {
                    "document_id": item.get("id"),
                    "display_name": item.get("display_name"),
                    "status": item.get("status"),
                }
                for item in items
            ],
            ensure_ascii=False,
            indent=1,
        )

    @server.tool(
        description=(
            "Report which knowledge base this server is bound to and whether the "
            "service can answer searches right now."
        )
    )
    async def memory_status() -> str:
        async with _client(settings) as client:
            try:
                response = await client.get("/ready/retrieve")
            except httpx.HTTPError:
                return (
                    f"Cannot reach the RAG service at {settings.base_url}. "
                    "Start it with `make start`."
                )
            ready = response.status_code == 200
            try:
                knowledge_base = str(await _resolve_knowledge_base(settings, client))
            except RagMcpConfigError as error:
                knowledge_base = str(error)
        return json.dumps(
            {
                "base_url": settings.base_url,
                "knowledge_base_id": knowledge_base,
                "retrieval_ready": ready,
            },
            ensure_ascii=False,
        )

    return server


def main() -> int:
    try:
        settings = _Settings(dict(os.environ))
    except RagMcpConfigError as error:
        # Printed rather than logged: an MCP client shows stderr when a server
        # fails to start, and this is the only chance to say what is missing.
        print(f"velox-mcp: {error}", flush=True)
        return 1
    build_server(settings).run("stdio")
    return 0


__all__ = ["DEFAULT_BASE_URL", "RagMcpConfigError", "build_server", "main"]
