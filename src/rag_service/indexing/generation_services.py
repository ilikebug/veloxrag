"""Recoverable three-phase saga for an initial empty index generation."""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import logging
import re
from collections.abc import Awaitable, Callable, Sequence
from contextlib import AbstractAsyncContextManager
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Literal, Protocol, cast
from uuid import UUID, uuid4, uuid5

from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from rag_service.api.cursors import CursorPosition, decode_cursor, encode_cursor
from rag_service.api.errors import BusinessError
from rag_service.auth.policies import AdminPrincipal
from rag_service.db.models.knowledge_bases import (
    IndexGenerationCreationRequest,
    KnowledgeBase,
    KnowledgeBaseIndexGeneration,
    KnowledgeBaseMutation,
)
from rag_service.indexing.generation_repositories import (
    CollectionCleanupClaim,
    GenerationRepository,
    SqlAlchemyGenerationRepository,
)
from rag_service.indexing.generation_schemas import (
    IndexGenerationCreate,
    IndexGenerationPage,
    SafeIndexGeneration,
)
from rag_service.indexing.identities import canonical_json_bytes, canonical_sha256, collection_name
from rag_service.indexing.qdrant import (
    CollectionSpec,
    ManagedCollection,
    PayloadIndex,
    QdrantClient,
    QdrantConfigurationError,
    QdrantTransientError,
    managed_collection_identity,
)
from rag_service.observability.logging import emit_safe_log
from rag_service.providers.embeddings import (
    EmbeddingConfigSnapshot,
    EmbeddingGatewayError,
    EmbeddingOperationalConfig,
    EmbeddingResult,
)
from rag_service.providers.repositories import (
    ModelProfileRecord,
    ProviderConfigRecord,
    ProviderCredentialRecord,
)

_GENERATION_NAMESPACE = UUID("72fb80bb-a22e-57a7-ac94-6b94caa14a5c")
_REQUEST_NAMESPACE = UUID("6fb85c10-72ae-56c8-9359-920ce6dd86c4")
_PROBE_TEXT = "x"
_IDEMPOTENCY_PATTERN = re.compile(r"^[!-~]+$")
_DISTANCES = frozenset({"cosine", "dot", "euclid", "manhattan"})
_SYSTEM_PAYLOAD_INDEXES = (
    PayloadIndex("knowledge_base_id", "keyword"),
    PayloadIndex("document_id", "keyword"),
    PayloadIndex("version_id", "keyword"),
    PayloadIndex("chunk_index", "integer"),
    PayloadIndex("chunk_hash", "keyword"),
)
_FILTER_INDEX_SCHEMAS = {
    "keyword": "keyword",
    "integer": "integer",
    "float": "float",
    "boolean": "bool",
    "datetime": "datetime",
}

logger = logging.getLogger(__name__)

SessionFactory = Callable[[], AbstractAsyncContextManager[AsyncSession]]
RepositoryFactory = Callable[[AsyncSession], GenerationRepository]
Clock = Callable[[], datetime]
Checkpoint = Callable[[str], Awaitable[None]]


class EmbeddingGateway(Protocol):
    async def embed(
        self,
        *,
        snapshot: EmbeddingConfigSnapshot,
        operational: EmbeddingOperationalConfig,
        inputs: Sequence[str],
    ) -> EmbeddingResult: ...


class GenerationConfigurationError(ValueError):
    __slots__ = ("code", "safe_message")

    def __init__(self, code: str, safe_message: str) -> None:
        self.code = code
        self.safe_message = safe_message
        super().__init__(safe_message)


async def _noop_checkpoint(_name: str) -> None:
    return None


class GenerationSagaHooks:
    """Internal injectable crash/checkpoint hook; never exposed as a public route."""

    def __init__(self, checkpoint: Checkpoint = _noop_checkpoint) -> None:
        if not callable(checkpoint):
            raise ValueError("generation hook is invalid")
        self._checkpoint = checkpoint

    async def reached(self, name: str) -> None:
        await self._checkpoint(name)


@dataclass(frozen=True, slots=True)
class BuiltEmbeddingConfiguration:
    snapshot: dict[str, object]
    semantic_hash: str
    gateway_snapshot: EmbeddingConfigSnapshot
    operational: EmbeddingOperationalConfig


@dataclass(frozen=True, slots=True)
class _ValidatedEmbeddingSnapshot:
    snapshot: dict[str, object]
    semantic_hash: str
    gateway_snapshot: EmbeddingConfigSnapshot
    provider_config_id: UUID


@dataclass(frozen=True, slots=True)
class _Reservation:
    request_id: UUID
    generation_id: UUID
    knowledge_base_id: UUID
    filter_schema_revision: int
    generation: SafeIndexGeneration
    embedding: BuiltEmbeddingConfiguration
    filter_snapshot: dict[str, object]
    collection_spec: CollectionSpec


@dataclass(frozen=True, slots=True)
class _TerminalFailure:
    status_code: int
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class OrphanReconciliationResult:
    deleted: tuple[str, ...]
    next_cursor: str | None


_ORPHAN_CURSOR_V1_PREFIX = "orphan-cleanup-v1:"
_ORPHAN_CURSOR_V2_PREFIX = "orphan-cleanup-v2:"


@dataclass(frozen=True, slots=True)
class _ExpiredCleanupPosition:
    lease_expires_at: datetime
    collection_name: str


@dataclass(frozen=True, slots=True)
class _OrphanCursorState:
    qdrant_cursor: str | None = None
    expired_after: _ExpiredCleanupPosition | None = None
    expired_scan_exhausted: bool = False
    turn: str = "expired"


def _validated_qdrant_cursor(value: object) -> str | None:
    if value is None:
        return None
    if type(value) is not str or not value or len(value) > 255 or "\x00" in value:
        raise ValueError
    return value


def _decode_orphan_cursor(cursor: str | None) -> _OrphanCursorState:
    if cursor is None:
        return _OrphanCursorState()
    if cursor.startswith(_ORPHAN_CURSOR_V1_PREFIX):
        try:
            turn, encoded = cursor.removeprefix(_ORPHAN_CURSOR_V1_PREFIX).split(":", 1)
            if turn not in {"expired", "qdrant"}:
                raise ValueError
            qdrant_cursor = _validated_qdrant_cursor(
                None
                if encoded == "-"
                else base64.b64decode(
                    encoded.encode("ascii"),
                    altchars=b"-_",
                    validate=True,
                ).decode("utf-8")
            )
        except (binascii.Error, UnicodeError, ValueError):
            raise ValueError("orphan reconciliation cursor is invalid") from None
        return _OrphanCursorState(qdrant_cursor=qdrant_cursor, turn=turn)
    if not cursor.startswith(_ORPHAN_CURSOR_V2_PREFIX):
        return _OrphanCursorState(qdrant_cursor=_validated_qdrant_cursor(cursor))
    try:
        encoded = cursor.removeprefix(_ORPHAN_CURSOR_V2_PREFIX)
        payload = json.loads(
            base64.b64decode(
                encoded.encode("ascii"),
                altchars=b"-_",
                validate=True,
            ).decode("utf-8")
        )
        if type(payload) is not dict or set(payload) != {"e", "q", "t", "x"}:
            raise ValueError
        turn = payload["t"]
        exhausted = payload["x"]
        if turn not in {"expired", "qdrant"} or type(exhausted) is not bool:
            raise ValueError
        qdrant_cursor = _validated_qdrant_cursor(payload["q"])
        encoded_expired = payload["e"]
        expired_after = None
        if encoded_expired is not None:
            if (
                type(encoded_expired) is not list
                or len(encoded_expired) != 2
                or any(type(value) is not str for value in encoded_expired)
            ):
                raise ValueError
            lease_expires_at = datetime.fromisoformat(encoded_expired[0].replace("Z", "+00:00"))
            collection_name = encoded_expired[1]
            if (
                lease_expires_at.tzinfo is None
                or lease_expires_at.utcoffset() != UTC.utcoffset(lease_expires_at)
                or not collection_name
                or len(collection_name) > 255
                or "\x00" in collection_name
            ):
                raise ValueError
            expired_after = _ExpiredCleanupPosition(
                lease_expires_at.astimezone(UTC),
                collection_name,
            )
    except (binascii.Error, json.JSONDecodeError, UnicodeError, ValueError):
        raise ValueError("orphan reconciliation cursor is invalid") from None
    return _OrphanCursorState(qdrant_cursor, expired_after, exhausted, turn)


def _encode_orphan_cursor(state: _OrphanCursorState) -> str:
    if state.turn not in {"expired", "qdrant"}:
        raise ValueError("orphan reconciliation cursor is invalid")
    _validated_qdrant_cursor(state.qdrant_cursor)
    expired = None
    if state.expired_after is not None:
        expired = [
            state.expired_after.lease_expires_at.astimezone(UTC)
            .isoformat(timespec="microseconds")
            .replace("+00:00", "Z"),
            state.expired_after.collection_name,
        ]
    payload = json.dumps(
        {
            "e": expired,
            "q": state.qdrant_cursor,
            "t": state.turn,
            "x": state.expired_scan_exhausted,
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return _ORPHAN_CURSOR_V2_PREFIX + base64.urlsafe_b64encode(payload).decode("ascii")


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _configuration_error(code: str, message: str) -> GenerationConfigurationError:
    return GenerationConfigurationError(code, message)


def build_embedding_configuration(
    profile: ModelProfileRecord,
    provider: ProviderConfigRecord,
    credential: ProviderCredentialRecord | None,
    *,
    distance: str,
    _require_enabled: bool = True,
    _snapshot_credential_id: UUID | None = None,
) -> BuiltEmbeddingConfiguration:
    if type(profile) is not ModelProfileRecord or profile.capability != "embedding":
        raise _configuration_error("INVALID_EMBEDDING_PROFILE", "Embedding profile is invalid")
    if _require_enabled and not profile.enabled:
        raise _configuration_error("MODEL_PROFILE_DISABLED", "Model profile is disabled")
    if type(provider) is not ProviderConfigRecord or provider.id != profile.provider_config_id:
        raise _configuration_error("INVALID_PROVIDER_CONFIG", "Provider configuration is invalid")
    if _require_enabled and not provider.enabled:
        raise _configuration_error("PROVIDER_CONFIG_DISABLED", "Provider configuration is disabled")
    credential_id = (
        provider.credential_id if _snapshot_credential_id is None else _snapshot_credential_id
    )
    if credential_id is None or credential is None or credential.id != credential_id:
        raise _configuration_error(
            "PROVIDER_CREDENTIAL_UNAVAILABLE",
            "Provider credential is unavailable",
        )
    if type(profile.dimension) is not int or profile.dimension <= 0:
        raise _configuration_error("INVALID_EMBEDDING_PROFILE", "Embedding profile is invalid")
    if distance not in _DISTANCES:
        raise _configuration_error("INVALID_DISTANCE", "Index distance is invalid")
    try:
        gateway_snapshot = EmbeddingConfigSnapshot(
            adapter_schema_version="openai-embeddings-v1",
            provider_type=provider.provider_type,
            base_url=provider.base_url,
            credential_id=credential_id,
            default_headers=provider.default_headers,
            routing_options=provider.routing_options,
            model_name=profile.model_name,
            dimension=profile.dimension,
            distance=distance,
            max_input_tokens=profile.max_input_tokens,
            vector_config=profile.vector_config,
        )
        operational = EmbeddingOperationalConfig(
            provider_config_id=provider.id,
            provider_enabled=provider.enabled,
            profile_enabled=profile.enabled,
            timeout_seconds=profile.timeout_seconds,
            max_concurrency=provider.max_concurrency,
            requests_per_minute=provider.requests_per_minute,
            batch_size=profile.batch_size,
        )
    except ValueError:
        raise _configuration_error(
            "INVALID_EMBEDDING_CONFIGURATION",
            "Embedding configuration is invalid",
        ) from None

    semantic_document: dict[str, object] = {
        "adapter_schema_version": gateway_snapshot.adapter_schema_version,
        "provider_type": gateway_snapshot.provider_type,
        "base_url": gateway_snapshot.base_url,
        "default_headers": dict(gateway_snapshot.default_headers),
        "routing_options": _json_copy(dict(gateway_snapshot.routing_options)),
        "model_name": gateway_snapshot.model_name,
        "dimension": gateway_snapshot.dimension,
        "distance": gateway_snapshot.distance,
        "max_input_tokens": gateway_snapshot.max_input_tokens,
        "vector_config": _json_copy(dict(gateway_snapshot.vector_config)),
    }
    snapshot = {
        **semantic_document,
        "provider_config_id": str(provider.id),
        "credential_id": str(gateway_snapshot.credential_id),
    }
    return BuiltEmbeddingConfiguration(
        snapshot=snapshot,
        semantic_hash=canonical_sha256(semantic_document),
        gateway_snapshot=gateway_snapshot,
        operational=operational,
    )


def _json_copy(value: object) -> object:
    return json.loads(canonical_json_bytes(value))


def build_filter_snapshot(filter_schema: dict[str, object]) -> dict[str, object]:
    try:
        copied = _json_copy(filter_schema)
        if type(copied) is not dict or set(copied) != {"fields"}:
            raise ValueError
        fields = copied["fields"]
        if type(fields) is not list or len(fields) > 64:
            raise ValueError
        seen_paths: set[str] = set()
        for field in fields:
            if type(field) is not dict or set(field) != {
                "name",
                "source_path",
                "type",
                "operators",
                "field_id",
                "payload_path",
            }:
                raise ValueError
            field_type = field["type"]
            payload_path = field["payload_path"]
            if (
                type(field_type) is not str
                or field_type not in _FILTER_INDEX_SCHEMAS
                or type(payload_path) is not str
                or not payload_path.startswith("metadata.f_")
                or len(payload_path) != len("metadata.f_") + 32
                or any(character not in "0123456789abcdef" for character in payload_path[11:])
                or payload_path in seen_paths
            ):
                raise ValueError
            seen_paths.add(payload_path)
        return cast(dict[str, object], copied)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        raise ValueError("filter schema snapshot is invalid") from None


def payload_indexes_for_filter_snapshot(
    snapshot: dict[str, object],
) -> tuple[PayloadIndex, ...]:
    validated = build_filter_snapshot(snapshot)
    fields = cast(list[dict[str, object]], validated["fields"])
    dynamic = tuple(
        PayloadIndex(
            cast(str, field["payload_path"]),
            _FILTER_INDEX_SCHEMAS[cast(str, field["type"])],
        )
        for field in fields
    )
    return (*_SYSTEM_PAYLOAD_INDEXES, *dynamic)


def canonical_empty_validation(
    *,
    knowledge_base_id: UUID,
    generation_id: UUID,
    collection: str,
    revision: int,
    actual_point_count: int,
) -> tuple[dict[str, object], str]:
    if (
        type(knowledge_base_id) is not UUID
        or type(generation_id) is not UUID
        or type(collection) is not str
        or type(revision) is not int
        or revision < 0
        or type(actual_point_count) is not int
        or actual_point_count < 0
    ):
        raise ValueError("empty validation manifest is invalid")
    manifest: dict[str, object] = {
        "schema_version": "empty-generation-validation-v1",
        "knowledge_base_id": str(knowledge_base_id),
        "generation_id": str(generation_id),
        "collection": collection,
        "revision": revision,
        "expected_point_count": 0,
        "actual_point_count": actual_point_count,
        "point_ids": [],
    }
    return manifest, canonical_sha256(manifest)


def _validate_persisted_embedding_snapshot(
    snapshot: dict[str, object],
    embedding_config_hash: str,
    index_profile_hash: str,
) -> _ValidatedEmbeddingSnapshot:
    try:
        expected_keys = {
            "adapter_schema_version",
            "provider_type",
            "base_url",
            "provider_config_id",
            "credential_id",
            "default_headers",
            "routing_options",
            "model_name",
            "dimension",
            "distance",
            "max_input_tokens",
            "vector_config",
        }
        if type(snapshot) is not dict or set(snapshot) != expected_keys:
            raise ValueError
        provider_config_id = UUID(cast(str, snapshot["provider_config_id"]))
        gateway_snapshot = EmbeddingConfigSnapshot(
            adapter_schema_version=cast(str, snapshot["adapter_schema_version"]),
            provider_type=cast(str, snapshot["provider_type"]),
            base_url=cast(str, snapshot["base_url"]),
            credential_id=UUID(cast(str, snapshot["credential_id"])),
            default_headers=cast(dict[str, str], snapshot["default_headers"]),
            routing_options=cast(dict[str, object], snapshot["routing_options"]),
            model_name=cast(str, snapshot["model_name"]),
            dimension=cast(int, snapshot["dimension"]),
            distance=cast(str, snapshot["distance"]),
            max_input_tokens=cast(int, snapshot["max_input_tokens"]),
            vector_config=cast(dict[str, object], snapshot["vector_config"]),
        )
        semantic_document: dict[str, object] = {
            "adapter_schema_version": gateway_snapshot.adapter_schema_version,
            "provider_type": gateway_snapshot.provider_type,
            "base_url": gateway_snapshot.base_url,
            "default_headers": dict(gateway_snapshot.default_headers),
            "routing_options": _json_copy(dict(gateway_snapshot.routing_options)),
            "model_name": gateway_snapshot.model_name,
            "dimension": gateway_snapshot.dimension,
            "distance": gateway_snapshot.distance,
            "max_input_tokens": gateway_snapshot.max_input_tokens,
            "vector_config": _json_copy(dict(gateway_snapshot.vector_config)),
        }
        semantic_hash = canonical_sha256(semantic_document)
        if (
            type(embedding_config_hash) is not str
            or type(index_profile_hash) is not str
            or semantic_hash != embedding_config_hash
            or semantic_hash != index_profile_hash
        ):
            raise ValueError
        return _ValidatedEmbeddingSnapshot(
            snapshot=cast(dict[str, object], _json_copy(snapshot)),
            semantic_hash=semantic_hash,
            gateway_snapshot=gateway_snapshot,
            provider_config_id=provider_config_id,
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        raise GenerationConfigurationError(
            "GENERATION_CONFIGURATION_CONFLICT",
            "Generation configuration changed during creation",
        ) from None


def _safe_generation(row: KnowledgeBaseIndexGeneration) -> SafeIndexGeneration:
    try:
        if row.embedding_profile_id is None or row.distance is None:
            raise ValueError
        return SafeIndexGeneration(
            id=row.id,
            knowledge_base_id=row.knowledge_base_id,
            embedding_profile_id=row.embedding_profile_id,
            status=cast(
                Literal["building", "active", "retiring", "failed"],
                row.status,
            ),
            distance=cast(
                Literal["cosine", "dot", "euclid", "manhattan"],
                row.distance,
            ),
            created_at=row.created_at,
            validated_at=row.validated_at,
            activated_at=row.activated_at,
        )
    except (AttributeError, TypeError, ValueError, ValidationError):
        raise BusinessError(500, "INTERNAL_ERROR", "Internal server error") from None


def _request_fingerprint(command: IndexGenerationCreate) -> bytes:
    document = {
        "schema": "initial-index-generation.create:v1",
        "embedding_profile_id": str(command.embedding_profile_id),
        "distance": command.distance,
    }
    return bytes.fromhex(canonical_sha256(document))


def _stable_id(namespace: UUID, actor_id: UUID, knowledge_base_id: UUID, key: str) -> UUID:
    return uuid5(namespace, f"{actor_id}:{knowledge_base_id}:{key}")


def _terminal_error(request: IndexGenerationCreationRequest) -> BusinessError:
    if (
        request.state != "failed"
        or request.final_http_status is None
        or request.safe_error_code is None
        or request.safe_error_message is None
    ):
        return BusinessError(500, "INTERNAL_ERROR", "Internal server error")
    return BusinessError(
        request.final_http_status,
        request.safe_error_code,
        request.safe_error_message,
    )


class GenerationQueryService:
    """Database-only query service for safe generation state."""

    def __init__(
        self,
        *,
        session_factory: SessionFactory,
        default_page_size: int = 20,
        max_page_size: int = 100,
        repository_factory: RepositoryFactory = SqlAlchemyGenerationRepository,
    ) -> None:
        if (
            not callable(session_factory)
            or type(default_page_size) is not int
            or type(max_page_size) is not int
            or not 1 <= default_page_size <= max_page_size <= 100
            or not callable(repository_factory)
        ):
            raise ValueError("generation query service dependencies are invalid")
        self._session_factory = session_factory
        self._default_page_size = default_page_size
        self._max_page_size = max_page_size
        self._repository_factory = repository_factory

    async def list_generations(
        self,
        knowledge_base_id: UUID,
        *,
        cursor: str | None,
        limit: int | None,
    ) -> IndexGenerationPage:
        if type(knowledge_base_id) is not UUID:
            raise BusinessError(422, "VALIDATION_ERROR", "Invalid generation request")
        page_limit = self._default_page_size if limit is None else limit
        if type(page_limit) is not int or not 1 <= page_limit <= self._max_page_size:
            raise BusinessError(422, "VALIDATION_ERROR", "Invalid generation request")
        position = None if cursor is None else decode_cursor(cursor)
        async with self._session_factory() as session:
            repository = self._repository_factory(session)
            knowledge_base = await session.get(KnowledgeBase, knowledge_base_id)
            if knowledge_base is None:
                raise BusinessError(404, "RESOURCE_NOT_FOUND", "Resource not found")
            rows = await repository.list_generations(
                knowledge_base_id,
                position,
                page_limit + 1,
            )
            has_more = len(rows) > page_limit
            page_rows = rows[:page_limit]
            next_cursor = None
            if has_more and page_rows:
                last = page_rows[-1]
                next_cursor = encode_cursor(CursorPosition(created_at=last.created_at, id=last.id))
            return IndexGenerationPage(
                items=tuple(_safe_generation(row) for row in page_rows),
                next_cursor=next_cursor,
            )


class GenerationService:
    def __init__(
        self,
        *,
        session_factory: SessionFactory,
        qdrant: QdrantClient,
        embedding_gateway: EmbeddingGateway,
        repository_factory: RepositoryFactory = SqlAlchemyGenerationRepository,
        clock: Clock = _utc_now,
        hooks: GenerationSagaHooks | None = None,
        max_idempotency_key_length: int = 128,
        cleanup_lease_duration: timedelta = timedelta(minutes=15),
    ) -> None:
        if (
            not callable(session_factory)
            or not callable(getattr(qdrant, "ensure_collection", None))
            or not callable(getattr(embedding_gateway, "embed", None))
            or not callable(repository_factory)
            or not callable(clock)
            or type(max_idempotency_key_length) is not int
            or not 1 <= max_idempotency_key_length <= 128
            or type(cleanup_lease_duration) is not timedelta
            or not timedelta(0) < cleanup_lease_duration <= timedelta(days=1)
        ):
            raise ValueError("generation service dependencies are invalid")
        self._session_factory = session_factory
        self._qdrant = qdrant
        self._embedding_gateway = embedding_gateway
        self._repository_factory = repository_factory
        self._clock = clock
        self._hooks = GenerationSagaHooks() if hooks is None else hooks
        self._max_idempotency_key_length = max_idempotency_key_length
        self._cleanup_lease_duration = cleanup_lease_duration

    def _now(self) -> datetime:
        now = self._clock()
        if now.tzinfo is None or now.utcoffset() != UTC.utcoffset(now):
            raise BusinessError(500, "INTERNAL_ERROR", "Internal server error")
        return now

    def _idempotency_key(self, value: str) -> str:
        if (
            type(value) is not str
            or not 1 <= len(value) <= self._max_idempotency_key_length
            or _IDEMPOTENCY_PATTERN.fullmatch(value) is None
        ):
            raise BusinessError(422, "VALIDATION_ERROR", "Invalid generation request")
        return value

    @staticmethod
    def _cleanup_in_progress() -> BusinessError:
        return BusinessError(
            503,
            "GENERATION_CLEANUP_IN_PROGRESS",
            "Generation collection cleanup is in progress",
            retryable=True,
        )

    @staticmethod
    def _collection_retired() -> BusinessError:
        return BusinessError(
            409,
            "GENERATION_COLLECTION_RETIRED",
            "Generation collection was retired; use a new idempotency key",
        )

    async def _load_current_embedding(
        self,
        repository: GenerationRepository,
        generation: KnowledgeBaseIndexGeneration,
    ) -> BuiltEmbeddingConfiguration:
        if (
            generation.embedding_profile_id is None
            or generation.distance is None
            or generation.embedding_config_snapshot is None
            or generation.embedding_config_hash is None
            or generation.index_profile_hash is None
        ):
            raise GenerationConfigurationError(
                "GENERATION_CONFIGURATION_CONFLICT",
                "Generation configuration changed during creation",
            )
        snapshot = cast(dict[str, object], generation.embedding_config_snapshot)
        stored = _validate_persisted_embedding_snapshot(
            snapshot,
            generation.embedding_config_hash,
            generation.index_profile_hash,
        )
        source = await repository.load_embedding_source(generation.embedding_profile_id)
        if source is None or source.provider.id != stored.provider_config_id:
            raise GenerationConfigurationError(
                "GENERATION_CONFIGURATION_CONFLICT",
                "Generation configuration changed during creation",
            )
        snapshot_credential_id = stored.gateway_snapshot.credential_id
        snapshot_credential = await repository.load_credential(snapshot_credential_id)
        if snapshot_credential is None:
            raise GenerationConfigurationError(
                "PROVIDER_CREDENTIAL_UNAVAILABLE",
                "Provider credential is unavailable",
            )
        current = build_embedding_configuration(
            source.profile,
            source.provider,
            snapshot_credential,
            distance=generation.distance,
            _require_enabled=False,
            _snapshot_credential_id=snapshot_credential_id,
        )
        if current.semantic_hash != stored.semantic_hash:
            raise GenerationConfigurationError(
                "GENERATION_CONFIGURATION_CONFLICT",
                "Generation configuration changed during creation",
            )
        return BuiltEmbeddingConfiguration(
            snapshot=stored.snapshot,
            semantic_hash=stored.semantic_hash,
            gateway_snapshot=stored.gateway_snapshot,
            operational=current.operational,
        )

    async def _reservation_from_rows(
        self,
        repository: GenerationRepository,
        request: IndexGenerationCreationRequest,
        generation: KnowledgeBaseIndexGeneration,
    ) -> _Reservation:
        if (
            generation.embedding_config_snapshot is None
            or generation.embedding_config_hash is None
            or generation.filter_schema_snapshot is None
            or generation.applied_filter_schema_revision is None
            or generation.distance is None
        ):
            raise GenerationConfigurationError(
                "GENERATION_CONFIGURATION_CONFLICT",
                "Generation configuration changed during creation",
            )
        try:
            expected_collection = collection_name(
                generation.knowledge_base_id,
                generation.id,
            )
            if generation.qdrant_collection_name != expected_collection:
                raise ValueError
            embedding = await self._load_current_embedding(repository, generation)
            filter_snapshot = build_filter_snapshot(
                cast(dict[str, object], generation.filter_schema_snapshot)
            )
            collection_spec = CollectionSpec(
                generation.qdrant_collection_name,
                embedding.gateway_snapshot.dimension,
                generation.distance,
                payload_indexes_for_filter_snapshot(filter_snapshot),
            )
            canonical_empty_validation(
                knowledge_base_id=generation.knowledge_base_id,
                generation_id=generation.id,
                collection=generation.qdrant_collection_name,
                revision=0,
                actual_point_count=0,
            )
            safe_generation = _safe_generation(generation)
        except GenerationConfigurationError:
            raise
        except (BusinessError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            raise GenerationConfigurationError(
                "GENERATION_CONFIGURATION_CONFLICT",
                "Generation configuration changed during creation",
            ) from None
        return _Reservation(
            request_id=request.id,
            generation_id=generation.id,
            knowledge_base_id=generation.knowledge_base_id,
            filter_schema_revision=generation.applied_filter_schema_revision,
            generation=safe_generation,
            embedding=embedding,
            filter_snapshot=filter_snapshot,
            collection_spec=collection_spec,
        )

    async def _reserve(
        self,
        knowledge_base_id: UUID,
        command: IndexGenerationCreate,
        *,
        actor: AdminPrincipal,
        idempotency_key: str,
    ) -> _Reservation | SafeIndexGeneration | _TerminalFailure:
        fingerprint = _request_fingerprint(command)
        async with self._session_factory() as session, session.begin():
            repository = self._repository_factory(session)
            knowledge_base = await repository.get_knowledge_base_for_update(knowledge_base_id)
            if knowledge_base is None:
                raise BusinessError(404, "RESOURCE_NOT_FOUND", "Resource not found")
            existing = await repository.get_request(
                actor.key_id,
                knowledge_base_id,
                idempotency_key,
            )
            if existing is not None:
                if existing.request_fingerprint != fingerprint:
                    raise BusinessError(
                        409,
                        "IDEMPOTENCY_KEY_REUSED",
                        "Idempotency key was reused with a different request",
                    )
                generation = await repository.get_generation(existing.generation_id)
                if generation is None:
                    raise BusinessError(500, "INTERNAL_ERROR", "Internal server error")
                await repository.acquire_collection_fence(generation.qdrant_collection_name)
                cleanup_claim = await repository.get_collection_cleanup_claim(
                    generation.qdrant_collection_name
                )
                if cleanup_claim is not None:
                    if cleanup_claim.completed_at is None:
                        raise self._cleanup_in_progress()
                    if existing.state == "succeeded" or generation.status == "active":
                        raise BusinessError(500, "INTERNAL_ERROR", "Internal server error")
                    retired = self._collection_retired()
                    if existing.state == "failed":
                        raise retired
                    generation.status = "failed"
                    generation.safe_error_code = retired.code
                    generation.safe_error_message = retired.message
                    if knowledge_base.pending_index_generation_id == generation.id:
                        knowledge_base.pending_index_generation_id = None
                    existing.state = "failed"
                    existing.final_http_status = retired.status_code
                    existing.safe_result = None
                    existing.safe_error_code = retired.code
                    existing.safe_error_message = retired.message
                    await repository.flush()
                    return _TerminalFailure(
                        retired.status_code,
                        retired.code,
                        retired.message,
                    )
                if existing.state == "failed":
                    raise _terminal_error(existing)
                if existing.state == "succeeded":
                    if generation.status != "active":
                        raise BusinessError(500, "INTERNAL_ERROR", "Internal server error")
                    return _safe_generation(generation)
                if knowledge_base.status != "active":
                    code = "GENERATION_CONFIGURATION_CONFLICT"
                    message = "Generation configuration changed during creation"
                    generation.status = "failed"
                    generation.safe_error_code = code
                    generation.safe_error_message = message
                    if knowledge_base.pending_index_generation_id == generation.id:
                        knowledge_base.pending_index_generation_id = None
                    existing.state = "failed"
                    existing.final_http_status = 409
                    existing.safe_result = None
                    existing.safe_error_code = code
                    existing.safe_error_message = message
                    await repository.flush()
                    return _TerminalFailure(409, code, message)
                try:
                    return await self._reservation_from_rows(
                        repository,
                        existing,
                        generation,
                    )
                except GenerationConfigurationError as error:
                    status_code = 409 if error.code == "GENERATION_CONFIGURATION_CONFLICT" else 422
                    generation.status = "failed"
                    generation.safe_error_code = error.code
                    generation.safe_error_message = error.safe_message
                    if knowledge_base.pending_index_generation_id == generation.id:
                        knowledge_base.pending_index_generation_id = None
                    existing.state = "failed"
                    existing.final_http_status = status_code
                    existing.safe_result = None
                    existing.safe_error_code = error.code
                    existing.safe_error_message = error.safe_message
                    await repository.flush()
                    return _TerminalFailure(
                        status_code,
                        error.code,
                        error.safe_message,
                    )

            if knowledge_base.status != "active":
                raise BusinessError(
                    409,
                    "GENERATION_CONFIGURATION_CONFLICT",
                    "Generation configuration changed during creation",
                )
            # An active generation no longer disqualifies the request: creating a
            # second one alongside it is how a cutover starts, which is the only
            # way to change an embedding model, a chunk size or a filter schema
            # without abandoning the knowledge base. What still disqualifies it is
            # another build already in flight — two would race for the same pending
            # pointer, and the loser would leave a collection nothing points at.
            if (
                knowledge_base.pending_index_generation_id is not None
                or await repository.building_generation_exists(knowledge_base_id)
            ):
                raise BusinessError(
                    409,
                    "INDEX_GENERATION_ALREADY_CONFIGURED",
                    "An index generation is already configured",
                )
            source = await repository.load_embedding_source(command.embedding_profile_id)
            if source is None:
                raise BusinessError(
                    422,
                    "INVALID_EMBEDDING_PROFILE",
                    "Embedding profile is invalid",
                )
            if source.legacy_secret_ref is not None:
                raise BusinessError(
                    422,
                    "PROVIDER_CREDENTIAL_UNAVAILABLE",
                    "Provider credential is unavailable",
                )
            try:
                embedding = build_embedding_configuration(
                    source.profile,
                    source.provider,
                    source.credential,
                    distance=command.distance,
                )
                filter_snapshot = build_filter_snapshot(
                    cast(dict[str, object], knowledge_base.filter_schema)
                )
            except GenerationConfigurationError as error:
                raise BusinessError(422, error.code, error.safe_message) from None
            except ValueError:
                raise BusinessError(500, "INTERNAL_ERROR", "Internal server error") from None

            generation_id = _stable_id(
                _GENERATION_NAMESPACE,
                actor.key_id,
                knowledge_base_id,
                idempotency_key,
            )
            creation_request_id = _stable_id(
                _REQUEST_NAMESPACE,
                actor.key_id,
                knowledge_base_id,
                idempotency_key,
            )
            qdrant_collection = collection_name(knowledge_base_id, generation_id)
            await repository.acquire_collection_fence(qdrant_collection)
            cleanup_claim = await repository.get_collection_cleanup_claim(qdrant_collection)
            if cleanup_claim is not None:
                if cleanup_claim.completed_at is not None:
                    raise self._collection_retired()
                raise self._cleanup_in_progress()
            request = IndexGenerationCreationRequest(
                id=creation_request_id,
                actor_api_key_id=actor.key_id,
                knowledge_base_id=knowledge_base_id,
                idempotency_key=idempotency_key,
                request_fingerprint=fingerprint,
                generation_id=generation_id,
                state="building",
            )
            await repository.add_request(request)
            await self._hooks.reached("after_creation_request")
            generation = KnowledgeBaseIndexGeneration(
                id=generation_id,
                knowledge_base_id=knowledge_base_id,
                embedding_profile_id=command.embedding_profile_id,
                sparse_profile_id=None,
                index_profile_hash=embedding.semantic_hash,
                qdrant_collection_name=qdrant_collection,
                status="building",
                rebuild_snapshot_at=self._now(),
                caught_up_revision=knowledge_base.mutation_revision,
                distance=command.distance,
                embedding_config_snapshot=deepcopy(embedding.snapshot),
                filter_schema_snapshot=deepcopy(filter_snapshot),
                applied_filter_schema_revision=knowledge_base.filter_schema_revision,
                embedding_config_hash=embedding.semantic_hash,
            )
            await repository.add_generation(generation)
            await self._hooks.reached("after_generation_insert")
            knowledge_base.pending_index_generation_id = generation_id
            await repository.flush()
            collection_spec = CollectionSpec(
                qdrant_collection,
                embedding.gateway_snapshot.dimension,
                command.distance,
                payload_indexes_for_filter_snapshot(filter_snapshot),
            )
            return _Reservation(
                request_id=request.id,
                generation_id=generation.id,
                knowledge_base_id=knowledge_base_id,
                filter_schema_revision=knowledge_base.filter_schema_revision,
                generation=_safe_generation(generation),
                embedding=embedding,
                filter_snapshot=filter_snapshot,
                collection_spec=collection_spec,
            )

    async def _mark_failed(
        self,
        reservation: _Reservation,
        *,
        status_code: int,
        code: str,
        message: str,
        actual_point_count: int | None = None,
    ) -> None:
        async with self._session_factory() as session, session.begin():
            repository = self._repository_factory(session)
            knowledge_base = await repository.get_knowledge_base_for_update(
                reservation.knowledge_base_id
            )
            generation = await repository.get_generation(
                reservation.generation_id,
                for_update=True,
            )
            creation_request = await session.get(
                IndexGenerationCreationRequest,
                reservation.request_id,
                with_for_update=True,
            )
            if knowledge_base is None or generation is None or creation_request is None:
                raise BusinessError(500, "INTERNAL_ERROR", "Internal server error")
            if creation_request.state == "succeeded" or generation.status == "active":
                return
            generation.status = "failed"
            generation.safe_error_code = code
            generation.safe_error_message = message
            if actual_point_count is not None:
                generation.expected_point_count = 0
                generation.actual_point_count = actual_point_count
            if knowledge_base.pending_index_generation_id == generation.id:
                knowledge_base.pending_index_generation_id = None
            creation_request.state = "failed"
            creation_request.final_http_status = status_code
            creation_request.safe_error_code = code
            creation_request.safe_error_message = message
            creation_request.safe_result = None
            await repository.flush()

    async def _reload_current_embedding(
        self,
        reservation: _Reservation,
    ) -> BuiltEmbeddingConfiguration:
        async with self._session_factory() as session:
            repository = self._repository_factory(session)
            generation = await repository.get_generation(reservation.generation_id)
            if generation is None:
                raise BusinessError(500, "INTERNAL_ERROR", "Internal server error")
            return await self._load_current_embedding(repository, generation)

    async def _activate(
        self,
        reservation: _Reservation,
        *,
        actual_point_count: int,
    ) -> SafeIndexGeneration:
        if actual_point_count != 0:
            raise BusinessError(500, "INTERNAL_ERROR", "Internal server error")
        conflict: BusinessError | None = None
        safe: SafeIndexGeneration | None = None
        async with self._session_factory() as session, session.begin():
            repository = self._repository_factory(session)
            knowledge_base = await repository.get_knowledge_base_for_update(
                reservation.knowledge_base_id
            )
            await repository.acquire_collection_fence(reservation.collection_spec.name)
            cleanup_claim = await repository.get_collection_cleanup_claim(
                reservation.collection_spec.name
            )
            generation = await repository.get_generation(
                reservation.generation_id,
                for_update=True,
            )
            request = await session.get(
                IndexGenerationCreationRequest,
                reservation.request_id,
                with_for_update=True,
            )
            if knowledge_base is None or generation is None or request is None:
                raise BusinessError(500, "INTERNAL_ERROR", "Internal server error")
            if cleanup_claim is not None and cleanup_claim.completed_at is None:
                raise self._cleanup_in_progress()
            if cleanup_claim is not None:
                if request.state == "succeeded" or generation.status == "active":
                    raise BusinessError(500, "INTERNAL_ERROR", "Internal server error")
                retired = self._collection_retired()
                generation.status = "failed"
                generation.safe_error_code = retired.code
                generation.safe_error_message = retired.message
                if knowledge_base.pending_index_generation_id == generation.id:
                    knowledge_base.pending_index_generation_id = None
                request.state = "failed"
                request.final_http_status = retired.status_code
                request.safe_result = None
                request.safe_error_code = retired.code
                request.safe_error_message = retired.message
                conflict = retired
                await repository.flush()
            elif request.state == "succeeded" and generation.status == "active":
                return _safe_generation(generation)
            elif request.state == "failed":
                raise _terminal_error(request)
            elif (
                generation.status != "building"
                or knowledge_base.status != "active"
                or knowledge_base.pending_index_generation_id != generation.id
                or knowledge_base.active_index_generation_id == generation.id
                or knowledge_base.filter_schema_revision != reservation.filter_schema_revision
            ):
                generation.status = "failed"
                generation.safe_error_code = "GENERATION_CONFIGURATION_CONFLICT"
                generation.safe_error_message = "Generation configuration changed during creation"
                if knowledge_base.pending_index_generation_id == generation.id:
                    knowledge_base.pending_index_generation_id = None
                request.state = "failed"
                request.final_http_status = 409
                request.safe_error_code = "GENERATION_CONFIGURATION_CONFLICT"
                request.safe_error_message = "Generation configuration changed during creation"
                conflict = BusinessError(
                    409,
                    "GENERATION_CONFIGURATION_CONFLICT",
                    "Generation configuration changed during creation",
                )
                await repository.flush()
            else:
                now = self._now()
                new_revision = knowledge_base.mutation_revision + 1
                manifest, manifest_hash = canonical_empty_validation(
                    knowledge_base_id=knowledge_base.id,
                    generation_id=generation.id,
                    collection=generation.qdrant_collection_name,
                    revision=new_revision,
                    actual_point_count=actual_point_count,
                )
                generation.caught_up_revision = new_revision
                generation.validated_revision = new_revision
                generation.expected_point_count = 0
                generation.actual_point_count = actual_point_count
                generation.validation_manifest_hash = manifest_hash
                generation.validated_at = now
                generation.activated_at = now
                generation.safe_error_code = None
                generation.safe_error_message = None
                knowledge_base.mutation_revision = new_revision
                await repository.add_mutation(
                    KnowledgeBaseMutation(
                        id=uuid5(
                            _REQUEST_NAMESPACE,
                            f"mutation:{generation.id}:{new_revision}",
                        ),
                        knowledge_base_id=knowledge_base.id,
                        revision=new_revision,
                        mutation_type="index_config_changed",
                        target_type="index_generation",
                        target_id=generation.id,
                        payload={
                            "generation_id": str(generation.id),
                            "embedding_profile_id": str(generation.embedding_profile_id),
                            "index_profile_hash": generation.index_profile_hash,
                            "embedding_config_hash": generation.embedding_config_hash,
                            "applied_filter_schema_revision": (
                                generation.applied_filter_schema_revision
                            ),
                            "validation_manifest": manifest,
                            "validation_manifest_hash": manifest_hash,
                        },
                    )
                )
                # The swap and the retirement share this transaction on purpose. A
                # window where neither generation is active would make every
                # indexed document unsearchable, and one where both are would let
                # a reader see either.
                superseded = (
                    await repository.get_generation(
                        knowledge_base.active_index_generation_id,
                        for_update=True,
                    )
                    if knowledge_base.active_index_generation_id is not None
                    else None
                )
                if superseded is not None:
                    # Retired and flushed before the successor is promoted, because
                    # uq_kb_index_generations_one_active is a partial unique index
                    # over status = 'active'. Assigning both in one flush presents
                    # the index with two active rows for this knowledge base and it
                    # rejects the transaction, which is the constraint doing its job.
                    superseded.status = "retiring"
                    await repository.flush()
                generation.status = "active"
                knowledge_base.pending_index_generation_id = None
                knowledge_base.active_index_generation_id = generation.id
                safe = _safe_generation(generation)
                request.state = "succeeded"
                request.final_http_status = 201
                request.safe_result = safe.model_dump(mode="json")
                request.safe_error_code = None
                request.safe_error_message = None
                await repository.flush()
                await self._hooks.reached("before_activation_commit")
        if conflict is not None:
            raise conflict
        if safe is None:
            raise BusinessError(500, "INTERNAL_ERROR", "Internal server error")
        return safe

    async def abandon_generation(
        self,
        knowledge_base_id: UUID,
        generation_id: UUID,
        *,
        actor: AdminPrincipal,
    ) -> SafeIndexGeneration:
        """Fail a generation stuck in `building` so the knowledge base is usable again.

        Creating a generation calls the embedding provider inside the request. A
        provider outage there leaves the generation `building`, and because a
        `building` generation blocks creating another one, the knowledge base has
        no active generation and no way to get one: retrying with a fresh
        idempotency key returns 409 forever, and only replaying the original key
        resumes it. This is the way out when that key is gone.

        Safe against a creation that is still running: activation re-checks that
        the generation is still `building` and fails itself with
        GENERATION_CONFIGURATION_CONFLICT when it is not.
        """
        if type(knowledge_base_id) is not UUID or type(generation_id) is not UUID:
            raise BusinessError(422, "VALIDATION_ERROR", "Invalid generation request")
        if type(actor) is not AdminPrincipal:
            raise BusinessError(422, "VALIDATION_ERROR", "Invalid generation request")
        async with self._session_factory() as session, session.begin():
            repository = self._repository_factory(session)
            knowledge_base = await repository.get_knowledge_base_for_update(knowledge_base_id)
            if knowledge_base is None:
                raise BusinessError(404, "RESOURCE_NOT_FOUND", "Resource not found")
            generation = await repository.get_generation(generation_id, for_update=True)
            if generation is None or generation.knowledge_base_id != knowledge_base_id:
                raise BusinessError(404, "RESOURCE_NOT_FOUND", "Resource not found")
            if generation.status != "building":
                # Never an active one: abandoning that would leave a knowledge
                # base whose documents are indexed but unsearchable.
                raise BusinessError(
                    409,
                    "GENERATION_NOT_ABANDONABLE",
                    "Only a building index generation can be abandoned",
                )
            now = self._now()
            generation.status = "failed"
            generation.safe_error_code = "GENERATION_ABANDONED"
            generation.safe_error_message = "Index generation was abandoned"
            if knowledge_base.pending_index_generation_id == generation.id:
                knowledge_base.pending_index_generation_id = None
            knowledge_base.updated_at = now
            return _safe_generation(generation)

    async def create_initial_generation(
        self,
        knowledge_base_id: UUID,
        command: IndexGenerationCreate,
        *,
        actor: AdminPrincipal,
        idempotency_key: str,
    ) -> SafeIndexGeneration:
        if type(knowledge_base_id) is not UUID or type(actor) is not AdminPrincipal:
            raise BusinessError(422, "VALIDATION_ERROR", "Invalid generation request")
        if type(command) is not IndexGenerationCreate:
            raise BusinessError(422, "VALIDATION_ERROR", "Invalid generation request")
        key = self._idempotency_key(idempotency_key)
        reservation_or_safe = await self._reserve(
            knowledge_base_id,
            command,
            actor=actor,
            idempotency_key=key,
        )
        if isinstance(reservation_or_safe, SafeIndexGeneration):
            return reservation_or_safe
        if isinstance(reservation_or_safe, _TerminalFailure):
            raise BusinessError(
                reservation_or_safe.status_code,
                reservation_or_safe.code,
                reservation_or_safe.message,
            )
        reservation = reservation_or_safe
        await self._hooks.reached("after_reservation")
        try:
            await self._qdrant.ensure_collection(reservation.collection_spec.vector_only())
            await self._hooks.reached("after_collection")
            await self._qdrant.ensure_payload_indexes(
                reservation.collection_spec.name,
                reservation.collection_spec.payload_indexes,
            )
            await self._hooks.reached("after_payload_indexes")
            await self._qdrant.verify_collection(reservation.collection_spec)
            actual_point_count = await self._qdrant.count_points(reservation.collection_spec.name)
        except QdrantTransientError:
            raise BusinessError(
                503,
                "QDRANT_UNAVAILABLE",
                "Qdrant is unavailable",
                retryable=True,
            ) from None
        except QdrantConfigurationError:
            await self._mark_failed(
                reservation,
                status_code=409,
                code="QDRANT_COLLECTION_MISMATCH",
                message="Qdrant collection does not match generation configuration",
            )
            raise BusinessError(
                409,
                "QDRANT_COLLECTION_MISMATCH",
                "Qdrant collection does not match generation configuration",
            ) from None

        if actual_point_count != 0:
            await self._mark_failed(
                reservation,
                status_code=409,
                code="QDRANT_COLLECTION_NOT_EMPTY",
                message="Qdrant collection is not empty",
                actual_point_count=actual_point_count,
            )
            raise BusinessError(
                409,
                "QDRANT_COLLECTION_NOT_EMPTY",
                "Qdrant collection is not empty",
            )

        try:
            current_embedding = await self._reload_current_embedding(reservation)
        except GenerationConfigurationError as error:
            status_code = 409 if error.code == "GENERATION_CONFIGURATION_CONFLICT" else 422
            await self._mark_failed(
                reservation,
                status_code=status_code,
                code=error.code,
                message=error.safe_message,
            )
            raise BusinessError(status_code, error.code, error.safe_message) from None

        try:
            await self._embedding_gateway.embed(
                snapshot=current_embedding.gateway_snapshot,
                operational=current_embedding.operational,
                inputs=(_PROBE_TEXT,),
            )
            await self._hooks.reached("after_probe")
        except EmbeddingGatewayError as error:
            if error.retryable:
                raise BusinessError(
                    503,
                    error.code,
                    str(error),
                    retryable=True,
                ) from None
            await self._mark_failed(
                reservation,
                status_code=422,
                code=error.code,
                message=str(error),
            )
            raise BusinessError(422, error.code, str(error)) from None
        return await self._activate(
            reservation,
            actual_point_count=actual_point_count,
        )

    async def _claim_orphan_cleanup(
        self,
        collection: str,
    ) -> CollectionCleanupClaim | None:
        knowledge_base_id, generation_id = managed_collection_identity(collection)
        now = self._now()
        async with self._session_factory() as session, session.begin():
            repository = self._repository_factory(session)
            await repository.get_knowledge_base_for_update(knowledge_base_id)
            await repository.acquire_collection_fence(collection)
            current = await repository.collection_statuses((collection,))
            if current.get(collection) not in {None, "failed"}:
                return None
            return await repository.claim_collection_cleanup(
                collection_name=collection,
                knowledge_base_id=knowledge_base_id,
                generation_id=generation_id,
                lease_owner=uuid4(),
                now=now,
                lease_duration=self._cleanup_lease_duration,
            )

    async def _complete_orphan_cleanup(self, claim: CollectionCleanupClaim) -> bool:
        async with self._session_factory() as session, session.begin():
            repository = self._repository_factory(session)
            await repository.get_knowledge_base_for_update(claim.knowledge_base_id)
            await repository.acquire_collection_fence(claim.collection_name)
            return await repository.complete_collection_cleanup(
                claim.collection_name,
                lease_owner=claim.lease_owner,
                lease_epoch=claim.lease_epoch,
                now=self._now(),
            )

    async def _authorize_orphan_cleanup(self, claim: CollectionCleanupClaim) -> bool:
        await self._hooks.reached("before_orphan_authoritative_recheck")
        now = self._now()
        async with self._session_factory() as session, session.begin():
            repository = self._repository_factory(session)
            await repository.acquire_collection_fence(claim.collection_name)
            current = await repository.get_collection_cleanup_claim_for_update(
                claim.collection_name
            )
            if (
                current is None
                or current.completed_at is not None
                or current.lease_owner != claim.lease_owner
                or current.lease_epoch != claim.lease_epoch
                or current.lease_expires_at <= now
            ):
                return False
            status = (await repository.collection_statuses((claim.collection_name,))).get(
                claim.collection_name
            )
            if status in {None, "failed"}:
                return True
            await repository.release_collection_cleanup(
                claim.collection_name,
                lease_owner=claim.lease_owner,
                lease_epoch=claim.lease_epoch,
            )
            return False

    async def _expire_orphan_cleanup(self, claim: CollectionCleanupClaim) -> bool:
        async with self._session_factory() as session, session.begin():
            repository = self._repository_factory(session)
            await repository.get_knowledge_base_for_update(claim.knowledge_base_id)
            await repository.acquire_collection_fence(claim.collection_name)
            return await repository.expire_collection_cleanup(
                claim.collection_name,
                lease_owner=claim.lease_owner,
                lease_epoch=claim.lease_epoch,
                now=self._now(),
            )

    @staticmethod
    def _cleanup_claim_has_canonical_identity(claim: CollectionCleanupClaim) -> bool:
        try:
            knowledge_base_id, generation_id = managed_collection_identity(claim.collection_name)
        except ValueError:
            return False
        return knowledge_base_id == claim.knowledge_base_id and generation_id == claim.generation_id

    async def reconcile_orphan_collections(
        self,
        *,
        grace_period: timedelta,
        limit: int,
        cursor: str | None,
    ) -> OrphanReconciliationResult:
        if (
            type(grace_period) is not timedelta
            or grace_period <= timedelta(0)
            or type(limit) is not int
            or not 1 <= limit <= 100
            or (
                cursor is not None
                and (
                    type(cursor) is not str or not cursor or len(cursor) > 1024 or "\x00" in cursor
                )
            )
        ):
            raise ValueError("orphan grace period is invalid")
        now = self._now()
        cutoff = now - grace_period
        state = _decode_orphan_cursor(cursor)
        expired_scan_budget = 1 if limit == 1 and state.turn == "expired" else max(1, limit // 2)
        if limit == 1 and state.turn == "qdrant":
            expired_scan_budget = 0
        expired_after = state.expired_after
        expired_scan_exhausted = state.expired_scan_exhausted
        if expired_scan_budget and expired_scan_exhausted:
            expired_after = None
            expired_scan_exhausted = False
        scanned_expired: tuple[CollectionCleanupClaim, ...] = ()
        eligible_expired: tuple[str, ...] = ()
        if expired_scan_budget:
            async with self._session_factory() as session:
                repository = self._repository_factory(session)
                expired_rows = await repository.list_expired_collection_cleanup_claims(
                    now=now,
                    limit=expired_scan_budget + 1,
                    after_lease_expires_at=(
                        None if expired_after is None else expired_after.lease_expires_at
                    ),
                    after_collection_name=(
                        None if expired_after is None else expired_after.collection_name
                    ),
                )
                scanned_expired = expired_rows[:expired_scan_budget]
                canonical_expired = tuple(
                    claim
                    for claim in scanned_expired
                    if self._cleanup_claim_has_canonical_identity(claim)
                )
                expired_statuses = await repository.collection_statuses(
                    tuple(claim.collection_name for claim in canonical_expired)
                )
            eligible_expired = tuple(
                claim.collection_name
                for claim in canonical_expired
                if expired_statuses.get(claim.collection_name) in {None, "failed"}
            )
            expired_scan_exhausted = len(expired_rows) <= expired_scan_budget
            if scanned_expired:
                last_expired = scanned_expired[-1]
                expired_after = _ExpiredCleanupPosition(
                    last_expired.lease_expires_at,
                    last_expired.collection_name,
                )

        page_budget = limit - len(eligible_expired)

        page = None
        page_statuses: dict[str, str] = {}
        if page_budget:
            page = await self._qdrant.list_managed_collections(
                limit=page_budget,
                cursor=state.qdrant_cursor,
            )
            if any(type(item) is not ManagedCollection for item in page.items):
                raise BusinessError(500, "INTERNAL_ERROR", "Internal server error")
            async with self._session_factory() as session:
                repository = self._repository_factory(session)
                page_statuses = await repository.collection_statuses(
                    tuple(collection.name for collection in page.items)
                )

        candidates: list[str] = []
        deleted: list[str] = []
        failed_count = 0
        candidates.extend(eligible_expired)
        if page is not None:
            for collection in page.items:
                if collection.created_at >= cutoff:
                    continue
                if page_statuses.get(collection.name) not in {None, "failed"}:
                    continue
                if collection.name not in candidates:
                    candidates.append(collection.name)

        for candidate_name in candidates:
            try:
                cleanup_claim = await self._claim_orphan_cleanup(candidate_name)
            except asyncio.CancelledError:
                raise
            except BaseException:
                failed_count += 1
                continue
            if cleanup_claim is None:
                continue
            try:
                authorized = await self._authorize_orphan_cleanup(cleanup_claim)
            except asyncio.CancelledError:
                raise
            except BaseException:
                failed_count += 1
                continue
            if not authorized:
                continue
            try:
                await self._hooks.reached("before_orphan_delete")
            except asyncio.CancelledError:
                raise
            except BaseException:
                failed_count += 1
                continue
            try:
                await self._qdrant.delete_collection(candidate_name)
            except asyncio.CancelledError:
                raise
            except BaseException:
                failed_count += 1
                try:
                    await self._expire_orphan_cleanup(cleanup_claim)
                except asyncio.CancelledError:
                    raise
                except BaseException:
                    pass
                continue
            try:
                await self._hooks.reached("after_orphan_delete")
            except asyncio.CancelledError:
                raise
            except BaseException:
                failed_count += 1
                continue
            try:
                completed = await self._complete_orphan_cleanup(cleanup_claim)
            except asyncio.CancelledError:
                raise
            except BaseException:
                failed_count += 1
                continue
            if completed:
                deleted.append(candidate_name)
        emit_safe_log(
            logger,
            logging.WARNING if failed_count else logging.INFO,
            "cleanup.action.completed",
            operation="orphan_cleanup",
            phase="qdrant",
            outcome="failed" if failed_count else "succeeded",
            count=failed_count if failed_count else len(deleted),
            candidate_count=len(candidates),
        )
        qdrant_cursor = state.qdrant_cursor if page is None else page.next_cursor
        next_turn = "qdrant" if limit == 1 and eligible_expired and page is None else "expired"
        qdrant_has_more = page is not None and page.next_cursor is not None
        qdrant_not_scanned = limit == 1 and eligible_expired and page is None
        expired_has_more = not expired_scan_exhausted
        if qdrant_has_more or qdrant_not_scanned or expired_has_more:
            next_cursor = _encode_orphan_cursor(
                _OrphanCursorState(
                    qdrant_cursor=qdrant_cursor,
                    expired_after=expired_after,
                    expired_scan_exhausted=expired_scan_exhausted,
                    turn=next_turn,
                )
            )
        else:
            next_cursor = None
        return OrphanReconciliationResult(tuple(deleted), next_cursor)


__all__ = [
    "BuiltEmbeddingConfiguration",
    "EmbeddingGateway",
    "GenerationConfigurationError",
    "GenerationQueryService",
    "GenerationSagaHooks",
    "GenerationService",
    "OrphanReconciliationResult",
    "build_embedding_configuration",
    "build_filter_snapshot",
    "canonical_empty_validation",
    "payload_indexes_for_filter_snapshot",
]
