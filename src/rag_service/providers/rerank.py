"""Cross-encoder reranking over already-retrieved candidates.

Separate from the embedding gateway rather than folded into it. Embedding is a
batched, restartable pipeline stage whose gateway carries admission control,
token accounting and per-attempt telemetry to match; reranking is one call on
the query path that either sharpens an answer already in hand or is skipped.
Reusing that machinery would have meant reshaping a concurrency-sensitive
module around a caller with none of the same needs.

Two things are shared deliberately: failure classification (see
`gateway_failures`), so a 429 means the same thing to both, and the transport,
so both go through the same endpoint policy and TLS trust.

The request follows text-embeddings-inference's `/rerank`: `{query, texts}` in,
`[{index, score}]` out. That shape is what this service has been verified
against. Cohere and Jina name the same fields differently, so the response
parser accepts their spelling too, but a provider needing a different *request*
belongs behind `vendor_specific` rather than being guessed at here.
"""

from __future__ import annotations

import asyncio
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from types import MappingProxyType
from typing import cast
from uuid import UUID

from rag_service.providers.credentials import (
    EncryptedProviderCredential,
    ProviderCredentialKeyring,
)
from rag_service.providers.embeddings import (
    ProviderCredentialReader,
    ProviderJsonTransport,
)
from rag_service.providers.gateway_failures import status_failure, transport_failure
from rag_service.providers.network_policy import (
    validate_provider_endpoint_url,
    validate_provider_headers,
)
from rag_service.providers.transport import ProviderHttpResponse

MAX_RERANK_DOCUMENTS = 200
MAX_RERANK_TEXT_CODEPOINTS = 32_000


class RerankGatewayError(Exception):
    """Safe rerank failure, carrying nothing the provider sent back."""

    def __init__(self, code: str, message: str, *, retryable: bool) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable


def _error(code: str, message: str, *, retryable: bool = False) -> RerankGatewayError:
    return RerankGatewayError(code, message, retryable=retryable)


@dataclass(frozen=True, slots=True)
class RerankConfigSnapshot:
    """Everything needed to address one rerank provider, frozen at read time."""

    provider_type: str
    base_url: str
    credential_id: UUID
    model_name: str
    default_headers: Mapping[str, str]

    def __post_init__(self) -> None:
        if (
            type(self.provider_type) is not str
            or self.provider_type not in {"openai_compatible", "openrouter", "vendor_specific"}
            or type(self.base_url) is not str
            or type(self.credential_id) is not UUID
            or type(self.model_name) is not str
            or not 1 <= len(self.model_name) <= 255
            or not isinstance(self.default_headers, Mapping)
        ):
            raise _error("RERANK_CONFIGURATION_INVALID", "Rerank configuration is invalid")
        try:
            headers = validate_provider_headers(self.default_headers)
        except Exception:
            raise _error(
                "RERANK_CONFIGURATION_INVALID", "Rerank configuration is invalid"
            ) from None
        object.__setattr__(self, "default_headers", MappingProxyType(dict(headers)))

    def __repr__(self) -> str:
        # No base_url or headers: both routinely carry tenant identifiers, and
        # this object reaches logs through exception context.
        return f"RerankConfigSnapshot(provider_type={self.provider_type!r})"


@dataclass(frozen=True, slots=True)
class RerankOperationalConfig:
    """Runtime limits read from the provider config and model profile."""

    timeout_seconds: Decimal
    provider_enabled: bool
    profile_enabled: bool

    def __post_init__(self) -> None:
        if (
            type(self.timeout_seconds) is not Decimal
            or not self.timeout_seconds.is_finite()
            or not Decimal("0") < self.timeout_seconds <= Decimal("600")
            or type(self.provider_enabled) is not bool
            or type(self.profile_enabled) is not bool
        ):
            raise _error("RERANK_CONFIGURATION_INVALID", "Rerank configuration is invalid")


@dataclass(frozen=True, slots=True)
class RerankedDocument:
    """One candidate's position in the reranked order."""

    index: int
    score: float


def _validate_documents(documents: Sequence[str]) -> tuple[str, ...]:
    if type(documents) not in {tuple, list} or not documents:
        raise _error("RERANK_INPUT_INVALID", "Rerank input is invalid")
    if len(documents) > MAX_RERANK_DOCUMENTS:
        raise _error("RERANK_INPUT_INVALID", "Rerank input is invalid")
    for document in documents:
        if (
            type(document) is not str
            or not document
            or len(document) > MAX_RERANK_TEXT_CODEPOINTS
            or "\x00" in document
        ):
            raise _error("RERANK_INPUT_INVALID", "Rerank input is invalid")
    return tuple(documents)


def _score_of(entry: Mapping[object, object]) -> float:
    # `score` is text-embeddings-inference, `relevance_score` is Cohere and Jina.
    for key in ("score", "relevance_score"):
        value = entry.get(key)
        if type(value) in {int, float}:
            numeric = float(cast(float, value))
            if math.isfinite(numeric):
                return numeric
    raise _error("PROVIDER_RESPONSE_INVALID", "Provider response is invalid")


def _parse_response(body: bytes, *, document_count: int) -> tuple[RerankedDocument, ...]:
    try:
        document = json.loads(body)
    except ValueError:
        raise _error("PROVIDER_RESPONSE_INVALID", "Provider response is invalid") from None
    entries: object = document
    if isinstance(document, Mapping):
        entries = document.get("results")
    if type(entries) is not list or not entries:
        raise _error("PROVIDER_RESPONSE_INVALID", "Provider response is invalid")
    seen: set[int] = set()
    parsed: list[RerankedDocument] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise _error("PROVIDER_RESPONSE_INVALID", "Provider response is invalid")
        index = entry.get("index")
        # A returned index addresses the caller's candidate list, so anything
        # out of range or repeated would silently reorder the wrong documents.
        if type(index) is not int or isinstance(index, bool) or not 0 <= index < document_count:
            raise _error("PROVIDER_RESPONSE_INVALID", "Provider response is invalid")
        if index in seen:
            raise _error("PROVIDER_RESPONSE_INVALID", "Provider response is invalid")
        seen.add(index)
        parsed.append(RerankedDocument(index=index, score=_score_of(entry)))
    parsed.sort(key=lambda item: (-item.score, item.index))
    return tuple(parsed)


class RerankGateway:
    """One call boundary for reranking retrieved candidates."""

    def __init__(
        self,
        *,
        keyring: ProviderCredentialKeyring,
        credential_reader: ProviderCredentialReader,
        transport: ProviderJsonTransport,
        max_concurrency: int = 4,
    ) -> None:
        if (
            type(keyring) is not ProviderCredentialKeyring
            or not callable(getattr(credential_reader, "get_encrypted", None))
            or not callable(getattr(transport, "post_json", None))
            or type(max_concurrency) is not int
            or not 1 <= max_concurrency <= 1024
        ):
            raise ValueError("Rerank gateway dependencies are invalid")
        self._keyring = keyring
        self._credential_reader = credential_reader
        self._transport = transport
        self._max_concurrency = max_concurrency
        self._semaphore: asyncio.Semaphore | None = None

    def _limit(self) -> asyncio.Semaphore:
        # Built lazily so the gateway can be constructed outside a running loop,
        # matching how the application wires its dependencies at import time.
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self._max_concurrency)
        return self._semaphore

    async def rerank(
        self,
        *,
        snapshot: RerankConfigSnapshot,
        operational: RerankOperationalConfig,
        query: str,
        documents: Sequence[str],
    ) -> tuple[RerankedDocument, ...]:
        if (
            type(snapshot) is not RerankConfigSnapshot
            or type(operational) is not RerankOperationalConfig
        ):
            raise _error("RERANK_CONFIGURATION_INVALID", "Rerank configuration is invalid")
        if not operational.provider_enabled:
            raise _error("PROVIDER_DISABLED", "Provider is disabled")
        if not operational.profile_enabled:
            raise _error("MODEL_PROFILE_DISABLED", "Model profile is disabled")
        if type(query) is not str or not query or len(query) > MAX_RERANK_TEXT_CODEPOINTS:
            raise _error("RERANK_INPUT_INVALID", "Rerank input is invalid")
        validated = _validate_documents(documents)

        async with self._limit():
            try:
                encrypted = await self._credential_reader.get_encrypted(snapshot.credential_id)
            except Exception as error:
                raise self._classified(error) from None
            if type(encrypted) is not EncryptedProviderCredential:
                raise _error("PROVIDER_CREDENTIAL_UNAVAILABLE", "Provider credential unavailable")
            try:
                response = await self._post(
                    snapshot=snapshot,
                    operational=operational,
                    encrypted=encrypted,
                    query=query,
                    documents=validated,
                )
            except asyncio.CancelledError:
                raise
            except Exception as error:
                raise self._classified(error) from None

        failure = status_failure(
            response.status_code, input_rejected_message="Provider rejected rerank input"
        )
        if failure is not None:
            raise _error(failure.code, failure.message, retryable=failure.retryable)
        return _parse_response(response.body, document_count=len(validated))

    @staticmethod
    def _classified(error: Exception) -> RerankGatewayError:
        if isinstance(error, RerankGatewayError):
            return error
        failure = transport_failure(error)
        return _error(failure.code, failure.message, retryable=failure.retryable)

    async def _post(
        self,
        *,
        snapshot: RerankConfigSnapshot,
        operational: RerankOperationalConfig,
        encrypted: EncryptedProviderCredential,
        query: str,
        documents: tuple[str, ...],
    ) -> ProviderHttpResponse:
        endpoint = validate_provider_endpoint_url(snapshot.base_url)

        async def invoke(secret_buffer: bytearray) -> ProviderHttpResponse:
            headers: dict[str, str] = {}
            payload: dict[str, object] = {}
            authorization = "<redacted>"
            try:
                if (
                    not secret_buffer
                    or len(secret_buffer) > 8192
                    or any(byte < 33 or byte > 126 for byte in secret_buffer)
                ):
                    raise _error("PROVIDER_CREDENTIAL_INVALID", "Provider credential is invalid")
                authorization = f"Bearer {secret_buffer.decode('ascii', errors='strict')}"
                headers = {
                    "Accept": "application/json",
                    "Authorization": authorization,
                    "Content-Type": "application/json",
                    **dict(snapshot.default_headers),
                }
                payload = {
                    "model": snapshot.model_name,
                    "query": query,
                    "texts": list(documents),
                }
                return await self._transport.post_json(
                    endpoint=endpoint,
                    path="/rerank",
                    headers=headers,
                    payload=payload,
                    timeout_seconds=float(operational.timeout_seconds),
                )
            finally:
                # Cleared rather than left to the garbage collector: the header
                # dict holds the decrypted secret, and the keyring zeroes only
                # the buffer it owns.
                headers.clear()
                payload.clear()
                authorization = "<redacted>"

        return await self._keyring.use_decrypted_async(snapshot.credential_id, encrypted, invoke)


__all__ = [
    "MAX_RERANK_DOCUMENTS",
    "MAX_RERANK_TEXT_CODEPOINTS",
    "RerankConfigSnapshot",
    "RerankGateway",
    "RerankGatewayError",
    "RerankOperationalConfig",
    "RerankedDocument",
]
