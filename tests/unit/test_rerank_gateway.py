from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from decimal import Decimal
from uuid import UUID, uuid4

import httpx
import pytest

from rag_service.providers.credentials import (
    EncryptedProviderCredential,
    ProviderCredentialKeyring,
)
from rag_service.providers.rerank import (
    MAX_RERANK_DOCUMENTS,
    RerankConfigSnapshot,
    RerankGateway,
    RerankGatewayError,
    RerankOperationalConfig,
)
from rag_service.providers.transport import ProviderHttpResponse

_KEY = b"r" * 32
_SECRET = "rerank-secret-sentinel"
_CREDENTIAL_ID = UUID("00000000-0000-4000-8000-0000000000a1")


def _keyring() -> ProviderCredentialKeyring:
    return ProviderCredentialKeyring(keys={"v1": _KEY}, active_key_version="v1")


@dataclass(slots=True)
class _CredentialReader:
    values: dict[UUID, EncryptedProviderCredential] = field(default_factory=dict)
    error: BaseException | None = None

    async def get_encrypted(self, credential_id: UUID) -> EncryptedProviderCredential | None:
        if self.error is not None:
            raise self.error
        return self.values.get(credential_id)


@dataclass(slots=True)
class _Transport:
    responses: list[ProviderHttpResponse | BaseException]
    requests: list[dict[str, object]] = field(default_factory=list)
    authorization_hashes: list[str] = field(default_factory=list)
    header_references: list[dict[str, str]] = field(default_factory=list)

    async def post_json(
        self,
        *,
        endpoint: object,
        path: str,
        headers: dict[str, str],
        payload: dict[str, object],
        timeout_seconds: float,
    ) -> ProviderHttpResponse:
        self.authorization_hashes.append(
            hashlib.sha256(headers.get("Authorization", "").encode()).hexdigest()
        )
        self.header_references.append(headers)
        self.requests.append(
            {
                "path": path,
                "payload": json.loads(json.dumps(payload)),
                "timeout_seconds": timeout_seconds,
            }
        )
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response

    async def aclose(self) -> None:
        return None


def _snapshot(**updates: object) -> RerankConfigSnapshot:
    values: dict[str, object] = {
        "provider_type": "openai_compatible",
        "base_url": "https://provider.example/v1",
        "credential_id": _CREDENTIAL_ID,
        "model_name": "bge-reranker-v2-m3",
        "default_headers": {},
    }
    values.update(updates)
    return RerankConfigSnapshot(**values)  # type: ignore[arg-type]


def _operational(**updates: object) -> RerankOperationalConfig:
    values: dict[str, object] = {
        "timeout_seconds": Decimal("30.000"),
        "provider_enabled": True,
        "profile_enabled": True,
    }
    values.update(updates)
    return RerankOperationalConfig(**values)  # type: ignore[arg-type]


def _gateway(transport: _Transport, reader: _CredentialReader | None = None) -> RerankGateway:
    keyring = _keyring()
    if reader is None:
        reader = _CredentialReader(
            values={_CREDENTIAL_ID: keyring.encrypt(_CREDENTIAL_ID, _SECRET.encode("utf-8"))}
        )
    return RerankGateway(keyring=keyring, credential_reader=reader, transport=transport)


def _response(body: object, status_code: int = 200) -> ProviderHttpResponse:
    return ProviderHttpResponse(
        status_code=status_code,
        headers={},
        body=json.dumps(body).encode("utf-8"),
    )


@pytest.mark.asyncio
async def test_rerank_sends_the_query_and_texts_and_returns_scored_order() -> None:
    transport = _Transport(
        responses=[_response([{"index": 2, "score": 0.1}, {"index": 0, "score": 0.9}])]
    )

    ordered = await _gateway(transport).rerank(
        snapshot=_snapshot(),
        operational=_operational(),
        query="why did retrieval regress",
        documents=("first", "second", "third"),
    )

    assert transport.requests[0]["path"] == "/rerank"
    assert transport.requests[0]["payload"] == {
        "model": "bge-reranker-v2-m3",
        "query": "why did retrieval regress",
        "texts": ["first", "second", "third"],
    }
    # Highest score first regardless of the order the provider replied in.
    assert [(item.index, item.score) for item in ordered] == [(0, 0.9), (2, 0.1)]


@pytest.mark.asyncio
async def test_rerank_accepts_the_cohere_response_shape() -> None:
    transport = _Transport(
        responses=[_response({"results": [{"index": 1, "relevance_score": 0.4}]})]
    )

    ordered = await _gateway(transport).rerank(
        snapshot=_snapshot(),
        operational=_operational(),
        query="q",
        documents=("a", "b"),
    )

    assert [(item.index, item.score) for item in ordered] == [(1, 0.4)]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body",
    [
        [{"index": 5, "score": 0.5}],
        [{"index": -1, "score": 0.5}],
        [{"index": 0, "score": 0.5}, {"index": 0, "score": 0.2}],
        [{"index": 0}],
        [{"index": 0, "score": "high"}],
        [{"index": True, "score": 0.5}],
        [{"index": 0, "score": float("inf")}],
        [],
        {"results": []},
        {"unexpected": "shape"},
    ],
)
async def test_rerank_rejects_a_response_that_cannot_address_the_candidates(body: object) -> None:
    transport = _Transport(responses=[_response(body)])

    # A returned index selects one of the caller's candidates, so a malformed or
    # duplicated one would reorder the wrong documents rather than fail loudly.
    with pytest.raises(RerankGatewayError) as raised:
        await _gateway(transport).rerank(
            snapshot=_snapshot(),
            operational=_operational(),
            query="q",
            documents=("a", "b"),
        )

    assert raised.value.code == "PROVIDER_RESPONSE_INVALID"
    assert raised.value.retryable is False


@pytest.mark.asyncio
async def test_rerank_rejects_a_non_json_body() -> None:
    transport = _Transport(
        responses=[ProviderHttpResponse(status_code=200, headers={}, body=b"<html>")]
    )

    with pytest.raises(RerankGatewayError) as raised:
        await _gateway(transport).rerank(
            snapshot=_snapshot(),
            operational=_operational(),
            query="q",
            documents=("a",),
        )

    assert raised.value.code == "PROVIDER_RESPONSE_INVALID"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status_code", "code", "retryable"),
    [
        (429, "PROVIDER_RATE_LIMITED", True),
        (503, "PROVIDER_UNAVAILABLE", True),
        (401, "PROVIDER_AUTHENTICATION_FAILED", False),
        (404, "PROVIDER_MODEL_NOT_FOUND", False),
        (422, "PROVIDER_INPUT_REJECTED", False),
        (302, "PROVIDER_REDIRECT_REJECTED", False),
    ],
)
async def test_rerank_maps_status_codes_the_same_way_embedding_does(
    status_code: int, code: str, retryable: bool
) -> None:
    transport = _Transport(responses=[_response([], status_code=status_code)])

    with pytest.raises(RerankGatewayError) as raised:
        await _gateway(transport).rerank(
            snapshot=_snapshot(),
            operational=_operational(),
            query="q",
            documents=("a",),
        )

    # Shared with the embedding gateway on purpose: retryable decides whether a
    # caller backs off or gives up, and the two must not disagree.
    assert (raised.value.code, raised.value.retryable) == (code, retryable)


@pytest.mark.asyncio
async def test_rerank_classifies_a_timeout_as_retryable() -> None:
    transport = _Transport(responses=[httpx.ReadTimeout("slow")])

    with pytest.raises(RerankGatewayError) as raised:
        await _gateway(transport).rerank(
            snapshot=_snapshot(),
            operational=_operational(),
            query="q",
            documents=("a",),
        )

    assert (raised.value.code, raised.value.retryable) == ("PROVIDER_TIMEOUT", True)


@pytest.mark.asyncio
async def test_rerank_reports_a_missing_credential_without_calling_the_provider() -> None:
    transport = _Transport(responses=[])

    with pytest.raises(RerankGatewayError) as raised:
        await _gateway(transport, _CredentialReader()).rerank(
            snapshot=_snapshot(),
            operational=_operational(),
            query="q",
            documents=("a",),
        )

    assert raised.value.code == "PROVIDER_CREDENTIAL_UNAVAILABLE"
    assert transport.requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("operational_update", "code"),
    [
        ({"provider_enabled": False}, "PROVIDER_DISABLED"),
        ({"profile_enabled": False}, "MODEL_PROFILE_DISABLED"),
    ],
)
async def test_rerank_refuses_disabled_configuration_before_the_call(
    operational_update: dict[str, object], code: str
) -> None:
    transport = _Transport(responses=[])

    with pytest.raises(RerankGatewayError) as raised:
        await _gateway(transport).rerank(
            snapshot=_snapshot(),
            operational=_operational(**operational_update),
            query="q",
            documents=("a",),
        )

    assert raised.value.code == code
    assert transport.requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "documents",
    [(), ("",), ("a\x00b",), tuple(f"doc {index}" for index in range(MAX_RERANK_DOCUMENTS + 1))],
)
async def test_rerank_rejects_unusable_documents_before_the_call(
    documents: tuple[str, ...],
) -> None:
    transport = _Transport(responses=[])

    with pytest.raises(RerankGatewayError) as raised:
        await _gateway(transport).rerank(
            snapshot=_snapshot(),
            operational=_operational(),
            query="q",
            documents=documents,
        )

    assert raised.value.code == "RERANK_INPUT_INVALID"
    assert transport.requests == []


@pytest.mark.asyncio
async def test_rerank_clears_the_decrypted_authorization_header_after_the_call() -> None:
    transport = _Transport(responses=[_response([{"index": 0, "score": 1.0}])])

    await _gateway(transport).rerank(
        snapshot=_snapshot(),
        operational=_operational(),
        query="q",
        documents=("a",),
    )

    # The header dict holds the decrypted secret; the keyring only zeroes the
    # buffer it owns, so the gateway has to drop its own copy.
    assert transport.authorization_hashes[0] != hashlib.sha256(b"").hexdigest()
    assert transport.header_references[0] == {}


def test_rerank_snapshot_repr_hides_the_endpoint() -> None:
    snapshot = _snapshot(base_url="https://tenant-42.internal.example/v1")

    # Reprs reach logs through exception context, and a base_url routinely names
    # the tenant.
    assert "tenant-42" not in repr(snapshot)


@pytest.mark.parametrize(
    "updates",
    [
        {"provider_type": "unsupported"},
        {"model_name": ""},
        {"credential_id": "not-a-uuid"},
        {"default_headers": {"Authorization": "Bearer leaked"}},
    ],
)
def test_rerank_snapshot_rejects_unusable_configuration(updates: dict[str, object]) -> None:
    with pytest.raises(RerankGatewayError):
        _snapshot(**updates)


def test_rerank_gateway_rejects_invalid_dependencies() -> None:
    with pytest.raises(ValueError):
        RerankGateway(
            keyring=_keyring(),
            credential_reader=_CredentialReader(),
            transport=_Transport(responses=[]),
            max_concurrency=0,
        )


@pytest.mark.asyncio
async def test_rerank_rejects_an_oversized_query_before_the_call() -> None:
    transport = _Transport(responses=[])

    with pytest.raises(RerankGatewayError) as raised:
        await _gateway(transport).rerank(
            snapshot=_snapshot(),
            operational=_operational(),
            query="q" * 32_001,
            documents=("a",),
        )

    assert raised.value.code == "RERANK_INPUT_INVALID"
    assert transport.requests == []


@pytest.mark.asyncio
async def test_rerank_uses_a_fresh_credential_read_per_call() -> None:
    keyring = _keyring()
    reader = _CredentialReader(
        values={_CREDENTIAL_ID: keyring.encrypt(_CREDENTIAL_ID, _SECRET.encode("utf-8"))}
    )
    transport = _Transport(
        responses=[_response([{"index": 0, "score": 1.0}]), _response([{"index": 0, "score": 1.0}])]
    )
    gateway = RerankGateway(keyring=keyring, credential_reader=reader, transport=transport)

    for _ in range(2):
        await gateway.rerank(
            snapshot=_snapshot(),
            operational=_operational(),
            query="q",
            documents=("a",),
        )

    # Not cached: a rotated or revoked credential has to take effect on the next
    # call rather than living for the process lifetime.
    assert len(transport.requests) == 2


def test_rerank_snapshot_freezes_supplied_headers() -> None:
    headers = {"X-Title": "acme"}
    snapshot = _snapshot(default_headers=headers)

    headers["X-Title"] = "mutated"

    assert isinstance(snapshot.default_headers, Mapping)
    assert snapshot.default_headers["X-Title"] == "acme"


def test_rerank_snapshot_rejects_headers_outside_the_allowlist() -> None:
    # The same allowlist the embedding path uses; an arbitrary header would let
    # provider configuration smuggle content onto an outbound request.
    with pytest.raises(RerankGatewayError):
        _snapshot(default_headers={"X-Tenant": "acme"})


@pytest.mark.asyncio
async def test_rerank_rejects_a_snapshot_of_the_wrong_type() -> None:
    transport = _Transport(responses=[])

    with pytest.raises(RerankGatewayError) as raised:
        await _gateway(transport).rerank(
            snapshot=uuid4(),  # type: ignore[arg-type]
            operational=_operational(),
            query="q",
            documents=("a",),
        )

    assert raised.value.code == "RERANK_CONFIGURATION_INVALID"
