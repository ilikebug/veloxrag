"""Atomic PostgreSQL reservation for durable document ingestion."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, cast
from uuid import UUID, uuid4

from sqlalchemy import and_, func, or_, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from rag_service.api.errors import BusinessError
from rag_service.api.validation import JSONValue
from rag_service.auth.policies import AgentPrincipal, Capability
from rag_service.db.models.auth import ApiKey, ApiKeyKnowledgeBaseScope
from rag_service.db.models.documents import (
    Document,
    DocumentIndexState,
    DocumentUploadIdempotency,
    DocumentVersion,
    Job,
)
from rag_service.db.models.knowledge_bases import (
    KnowledgeBase,
    KnowledgeBaseIndexGeneration,
    KnowledgeBaseMutation,
)
from rag_service.db.models.providers import ModelProfile, ProviderConfig
from rag_service.indexing.generation_services import build_filter_snapshot
from rag_service.indexing.identities import canonical_json_bytes, canonical_sha256, collection_name
from rag_service.ingestion.artifacts import canonical_document_artifact_identity
from rag_service.ingestion.schemas import validate_metadata_against_filter_snapshot
from rag_service.jobs.repositories import (
    ExhaustedJob,
    JobLease,
    LostLeaseError,
    SqlAlchemyJobRepository,
)
from rag_service.providers.embeddings import EmbeddingConfigSnapshot, EmbeddingOperationalConfig


class DocumentActivationConflictError(RuntimeError):
    """The lease is current, but activation business predicates changed."""


@dataclass(frozen=True, slots=True)
class UploadPreflight:
    knowledge_base_id: UUID
    generation_id: UUID
    filter_schema_snapshot: dict[str, object]
    applied_filter_schema_revision: int
    current_filter_schema_revision: int


@dataclass(frozen=True, slots=True)
class UploadReservation:
    actor: AgentPrincipal
    knowledge_base_id: UUID
    preflight: UploadPreflight | None
    idempotency_key: str | None
    request_fingerprint: bytes
    document_id: UUID
    version_id: UUID
    job_id: UUID
    source_object_key: str
    source_size: int
    source_checksum_sha256: str
    display_name: str
    tags: tuple[str, ...]
    metadata: dict[str, JSONValue]
    declared_mime_type: str
    detected_mime_type: str
    source_extension: str
    parser_name: str


@dataclass(frozen=True, slots=True)
class UploadReservationResult:
    document_id: UUID
    version_id: UUID
    job_id: UUID
    replay: bool

    @classmethod
    def created(cls, document_id: UUID, version_id: UUID, job_id: UUID) -> UploadReservationResult:
        return cls(document_id, version_id, job_id, False)

    @classmethod
    def replayed(cls, document_id: UUID, version_id: UUID, job_id: UUID) -> UploadReservationResult:
        return cls(document_id, version_id, job_id, True)


@dataclass(frozen=True, slots=True, repr=False)
class ParseStageInput:
    knowledge_base_id: UUID
    generation_id: UUID
    document_id: UUID
    version_id: UUID
    source_object_key: str
    source_checksum_sha256: str
    source_extension: str
    parser_name: str
    parser_version: str
    parser_config: dict[str, object]
    version_status: str


@dataclass(frozen=True, slots=True, repr=False)
class ChunkStageInput:
    knowledge_base_id: UUID
    generation_id: UUID
    document_id: UUID
    version_id: UUID
    source_checksum_sha256: str
    source_extension: str
    parsed_object_key: str
    parsed_object_checksum_sha256: str
    parser_name: str
    parser_version: str
    parser_config: dict[str, object]
    chunker_name: str | None
    chunker_version: str | None
    chunker_config: dict[str, object]
    version_status: str
    version_created_at: datetime


@dataclass(frozen=True, slots=True, repr=False)
class EmbedIndexStageInput:
    knowledge_base_id: UUID
    generation_id: UUID
    document_id: UUID
    version_id: UUID
    actor_api_key_id: UUID | None
    model_profile_id: UUID
    provider_config_id: UUID
    manifest_object_key: str
    manifest_checksum_sha256: str
    source_checksum_sha256: str
    parsed_checksum_sha256: str
    parser_name: str
    parser_version: str
    parser_config: dict[str, object]
    chunker_name: str
    chunker_version: str
    chunker_config: dict[str, object]
    chunk_config_hash: str
    chunk_count: int
    version_created_at: datetime
    version_status: str
    index_state_status: str
    index_state_embedding_config_hash: str | None
    next_chunk_index: int
    qdrant_collection_name: str
    embedding_config_hash: str
    embedding_snapshot_canonical: bytes
    filter_snapshot: dict[str, object]
    filter_snapshot_canonical: bytes
    applied_filter_schema_revision: int
    document_metadata: dict[str, JSONValue]
    document_metadata_canonical: bytes
    gateway_snapshot: EmbeddingConfigSnapshot
    operational: EmbeddingOperationalConfig


@dataclass(frozen=True, slots=True, repr=False)
class ActivationStageInput:
    knowledge_base_id: UUID
    generation_id: UUID
    document_id: UUID
    version_id: UUID
    job_id: UUID
    source_checksum_sha256: str
    detected_mime_type: str
    chunk_count: int
    expected_point_count: int
    actual_point_count: int


def _embedding_configuration(
    generation: KnowledgeBaseIndexGeneration,
    profile: ModelProfile,
    provider: ProviderConfig,
) -> tuple[UUID, EmbeddingConfigSnapshot, EmbeddingOperationalConfig, bytes]:
    try:
        raw = generation.embedding_config_snapshot
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
        if type(raw) is not dict or set(raw) != expected_keys:
            raise ValueError
        provider_config_id = UUID(cast(str, raw["provider_config_id"]))
        gateway = EmbeddingConfigSnapshot(
            adapter_schema_version=cast(str, raw["adapter_schema_version"]),
            provider_type=cast(str, raw["provider_type"]),
            base_url=cast(str, raw["base_url"]),
            credential_id=UUID(cast(str, raw["credential_id"])),
            default_headers=cast(dict[str, str], raw["default_headers"]),
            routing_options=cast(dict[str, object], raw["routing_options"]),
            model_name=cast(str, raw["model_name"]),
            dimension=cast(int, raw["dimension"]),
            distance=cast(str, raw["distance"]),
            max_input_tokens=cast(int, raw["max_input_tokens"]),
            vector_config=cast(dict[str, object], raw["vector_config"]),
        )
        semantic = {
            "adapter_schema_version": gateway.adapter_schema_version,
            "provider_type": gateway.provider_type,
            "base_url": gateway.base_url,
            "default_headers": dict(gateway.default_headers),
            "routing_options": json.loads(canonical_json_bytes(dict(gateway.routing_options))),
            "model_name": gateway.model_name,
            "dimension": gateway.dimension,
            "distance": gateway.distance,
            "max_input_tokens": gateway.max_input_tokens,
            "vector_config": json.loads(canonical_json_bytes(dict(gateway.vector_config))),
        }
        semantic_hash = canonical_sha256(semantic)
        if (
            generation.embedding_profile_id != profile.id
            or generation.distance != gateway.distance
            or generation.embedding_config_hash != semantic_hash
            or generation.index_profile_hash != semantic_hash
            or provider_config_id != provider.id
            or profile.provider_config_id != provider.id
            or profile.capability != "embedding"
            or provider.credential_id != gateway.credential_id
        ):
            raise ValueError
        operational = EmbeddingOperationalConfig(
            provider_config_id=provider.id,
            provider_enabled=provider.enabled,
            profile_enabled=profile.enabled,
            timeout_seconds=profile.timeout_seconds,
            max_concurrency=provider.max_concurrency,
            requests_per_minute=provider.requests_per_minute,
            batch_size=profile.batch_size,
        )
        return provider_config_id, gateway, operational, canonical_json_bytes(raw)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        raise ValueError("ingestion stage state is invalid") from None


class SqlAlchemyIngestionPipelineRepository:
    """Read immutable stage inputs and persist stage facts without committing."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def object_key_is_referenced(self, object_key: str) -> bool:
        if type(object_key) is not str or not object_key or len(object_key) > 1024:
            raise ValueError("object key reference check is invalid")
        referenced = await self._session.scalar(
            select(DocumentVersion.id)
            .where(
                or_(
                    DocumentVersion.source_object_key == object_key,
                    DocumentVersion.parsed_object_key == object_key,
                    DocumentVersion.chunk_manifest_object_key == object_key,
                )
            )
            .limit(1)
        )
        return referenced is not None

    async def object_key_cleanup_is_allowed(self, object_key: str) -> bool:
        if type(object_key) is not str or not object_key or len(object_key) > 1024:
            raise ValueError("object key reference check is invalid")
        identity = canonical_document_artifact_identity(object_key)
        if identity is None:
            return not await self.object_key_is_referenced(object_key)
        await self._session.scalar(
            select(KnowledgeBase.id)
            .where(KnowledgeBase.id == identity.knowledge_base_id)
            .with_for_update()
        )
        version = await self._session.scalar(
            select(DocumentVersion)
            .join(Document, Document.id == DocumentVersion.document_id)
            .where(
                DocumentVersion.id == identity.version_id,
                Document.id == identity.document_id,
                Document.knowledge_base_id == identity.knowledge_base_id,
            )
            .with_for_update()
        )
        if version is None:
            return not await self.object_key_is_referenced(object_key)
        jobs = (
            await self._session.scalars(
                select(Job)
                .where(
                    Job.target_type == "document_version",
                    Job.target_id == identity.version_id,
                    Job.operation == "ingest_document",
                )
                .order_by(Job.created_at, Job.id)
                .with_for_update()
            )
        ).all()
        if object_key in {
            version.source_object_key,
            version.parsed_object_key,
            version.chunk_manifest_object_key,
        } or await self.object_key_is_referenced(object_key):
            return False
        if identity.artifact_type == "source":
            return True
        expected_stage = "parse" if identity.artifact_type == "parsed" else "chunk"
        recoverable_version_statuses = (
            {"uploaded", "parsing", "failed"}
            if identity.artifact_type == "parsed"
            else {"chunking", "failed"}
        )
        if version.status in recoverable_version_statuses:
            for job in jobs:
                recoverable_job = job.status in {"queued", "running", "retry_wait"} or (
                    job.status == "failed" and job.retryable is True
                )
                if recoverable_job and expected_stage in {job.stage, job.resume_stage}:
                    return False
        return True

    async def _load_graph(
        self,
        version_id: UUID,
        generation_id: UUID,
    ) -> tuple[DocumentVersion, Document, DocumentIndexState]:
        row = (
            await self._session.execute(
                select(DocumentVersion, Document, DocumentIndexState)
                .join(Document, Document.id == DocumentVersion.document_id)
                .join(
                    DocumentIndexState,
                    DocumentIndexState.document_version_id == DocumentVersion.id,
                )
                .where(
                    DocumentVersion.id == version_id,
                    DocumentIndexState.index_generation_id == generation_id,
                )
            )
        ).one_or_none()
        if row is None:
            raise ValueError("ingestion stage state is invalid")
        return row[0], row[1], row[2]

    async def load_parse_stage(
        self,
        version_id: UUID,
        generation_id: UUID,
    ) -> ParseStageInput:
        version, document, _state = await self._load_graph(version_id, generation_id)
        if (
            version.source_extension is None
            or version.parser_name is None
            or version.parser_version is None
        ):
            raise ValueError("ingestion stage state is invalid")
        return ParseStageInput(
            document.knowledge_base_id,
            generation_id,
            document.id,
            version.id,
            version.source_object_key,
            version.source_checksum_sha256,
            version.source_extension,
            version.parser_name,
            version.parser_version,
            dict(version.parser_config),
            version.status,
        )

    async def load_chunk_stage(
        self,
        version_id: UUID,
        generation_id: UUID,
    ) -> ChunkStageInput:
        version, document, _state = await self._load_graph(version_id, generation_id)
        if (
            version.source_extension is None
            or version.parsed_object_key is None
            or version.parsed_object_checksum_sha256 is None
            or version.parser_name is None
            or version.parser_version is None
        ):
            raise ValueError("ingestion stage state is invalid")
        return ChunkStageInput(
            document.knowledge_base_id,
            generation_id,
            document.id,
            version.id,
            version.source_checksum_sha256,
            version.source_extension,
            version.parsed_object_key,
            version.parsed_object_checksum_sha256,
            version.parser_name,
            version.parser_version,
            dict(version.parser_config),
            version.chunker_name,
            version.chunker_version,
            dict(version.chunker_config),
            version.status,
            version.created_at,
        )

    async def load_embed_index_stage(
        self,
        version_id: UUID,
        generation_id: UUID,
        job_id: UUID,
    ) -> EmbedIndexStageInput:
        version, document, state = await self._load_graph(version_id, generation_id)
        knowledge_base = await self._session.get(KnowledgeBase, document.knowledge_base_id)
        generation = await self._session.get(KnowledgeBaseIndexGeneration, generation_id)
        job = await self._session.get(Job, job_id)
        if (
            knowledge_base is None
            or generation is None
            or job is None
            or knowledge_base.status != "active"
            or knowledge_base.active_index_generation_id != generation.id
            or generation.status != "active"
            or job.knowledge_base_id != knowledge_base.id
            or job.target_type != "document_version"
            or job.target_id != version.id
            or job.index_generation_id != generation.id
            or job.operation != "ingest_document"
            or generation.embedding_profile_id is None
            or generation.filter_schema_snapshot is None
            or generation.applied_filter_schema_revision is None
            or generation.embedding_config_hash is None
        ):
            raise ValueError("ingestion stage state is invalid")
        profile = await self._session.get(ModelProfile, generation.embedding_profile_id)
        if profile is None:
            raise ValueError("ingestion stage state is invalid")
        provider = await self._session.get(ProviderConfig, profile.provider_config_id)
        if provider is None:
            raise ValueError("ingestion stage state is invalid")
        provider_config_id, gateway, operational, snapshot_canonical = _embedding_configuration(
            generation, profile, provider
        )
        try:
            filter_snapshot = build_filter_snapshot(
                cast(dict[str, object], generation.filter_schema_snapshot)
            )
            document_metadata = cast(
                dict[str, JSONValue],
                json.loads(canonical_json_bytes(document.metadata_)),
            )
            actor_api_key_id = job.actor_api_key_id
            if actor_api_key_id is None:
                legacy_idempotency = await self._session.scalar(
                    select(DocumentUploadIdempotency)
                    .where(DocumentUploadIdempotency.job_id == job.id)
                    .order_by(DocumentUploadIdempotency.created_at, DocumentUploadIdempotency.id)
                    .limit(1)
                )
                if (
                    legacy_idempotency is None
                    or legacy_idempotency.knowledge_base_id != knowledge_base.id
                    or legacy_idempotency.document_id != document.id
                    or legacy_idempotency.document_version_id != version.id
                ):
                    raise ValueError("ingestion stage state is invalid")
                actor_api_key_id = legacy_idempotency.actor_api_key_id
            if (
                version.chunk_manifest_object_key is None
                or version.chunk_manifest_checksum_sha256 is None
                or version.chunk_config_hash is None
                or version.parsed_object_checksum_sha256 is None
                or version.parser_name is None
                or version.parser_version is None
                or version.chunker_name is None
                or version.chunker_version is None
                or version.chunk_count is None
                or version.chunk_count < 1
                or version.status not in {"embedding", "indexing"}
                or state.status not in {"embedding", "indexing"}
                or state.expected_point_count != version.chunk_count
                or state.chunk_manifest_checksum_sha256 != version.chunk_manifest_checksum_sha256
                or not 0 <= state.next_chunk_index <= version.chunk_count
                or (
                    state.next_chunk_index == 0
                    and state.embedding_config_hash not in {None, generation.embedding_config_hash}
                )
                or (
                    state.next_chunk_index > 0
                    and state.embedding_config_hash != generation.embedding_config_hash
                )
                or generation.qdrant_collection_name
                != collection_name(knowledge_base.id, generation.id)
            ):
                raise ValueError
            return EmbedIndexStageInput(
                knowledge_base_id=knowledge_base.id,
                generation_id=generation.id,
                document_id=document.id,
                version_id=version.id,
                actor_api_key_id=actor_api_key_id,
                model_profile_id=profile.id,
                provider_config_id=provider_config_id,
                manifest_object_key=version.chunk_manifest_object_key,
                manifest_checksum_sha256=version.chunk_manifest_checksum_sha256,
                source_checksum_sha256=version.source_checksum_sha256,
                parsed_checksum_sha256=version.parsed_object_checksum_sha256,
                parser_name=version.parser_name,
                parser_version=version.parser_version,
                parser_config=cast(
                    dict[str, object], json.loads(canonical_json_bytes(version.parser_config))
                ),
                chunker_name=version.chunker_name,
                chunker_version=version.chunker_version,
                chunker_config=cast(
                    dict[str, object], json.loads(canonical_json_bytes(version.chunker_config))
                ),
                chunk_config_hash=version.chunk_config_hash,
                chunk_count=version.chunk_count,
                version_created_at=version.created_at,
                version_status=version.status,
                index_state_status=state.status,
                index_state_embedding_config_hash=state.embedding_config_hash,
                next_chunk_index=state.next_chunk_index,
                qdrant_collection_name=generation.qdrant_collection_name,
                embedding_config_hash=generation.embedding_config_hash,
                embedding_snapshot_canonical=snapshot_canonical,
                filter_snapshot=filter_snapshot,
                filter_snapshot_canonical=canonical_json_bytes(filter_snapshot),
                applied_filter_schema_revision=generation.applied_filter_schema_revision,
                document_metadata=document_metadata,
                document_metadata_canonical=canonical_json_bytes(document_metadata),
                gateway_snapshot=gateway,
                operational=operational,
            )
        except (TypeError, ValueError, json.JSONDecodeError):
            raise ValueError("ingestion stage state is invalid") from None

    async def load_validate_stage(
        self,
        version_id: UUID,
        generation_id: UUID,
        job_id: UUID,
    ) -> EmbedIndexStageInput:
        return await self.load_embed_index_stage(version_id, generation_id, job_id)

    async def load_activation_stage(
        self,
        version_id: UUID,
        generation_id: UUID,
        job_id: UUID,
    ) -> ActivationStageInput:
        version, document, state = await self._load_graph(version_id, generation_id)
        job = await self._session.get(Job, job_id)
        if (
            job is None
            or job.knowledge_base_id != document.knowledge_base_id
            or job.operation != "ingest_document"
            or job.target_type != "document_version"
            or job.target_id != version.id
            or job.index_generation_id != generation_id
            or version.detected_mime_type is None
            or version.chunk_count is None
            or state.expected_point_count is None
            or state.actual_point_count is None
        ):
            raise ValueError("ingestion stage state is invalid")
        return ActivationStageInput(
            knowledge_base_id=document.knowledge_base_id,
            generation_id=generation_id,
            document_id=document.id,
            version_id=version.id,
            job_id=job.id,
            source_checksum_sha256=version.source_checksum_sha256,
            detected_mime_type=version.detected_mime_type,
            chunk_count=version.chunk_count,
            expected_point_count=state.expected_point_count,
            actual_point_count=state.actual_point_count,
        )

    async def _lock_stage_graph(
        self,
        *,
        knowledge_base_id: UUID,
        generation_id: UUID,
        document_id: UUID,
        version_id: UUID,
    ) -> tuple[
        KnowledgeBase,
        KnowledgeBaseIndexGeneration,
        Document,
        DocumentVersion,
        DocumentIndexState,
    ]:
        knowledge_base = await self._session.scalar(
            select(KnowledgeBase).where(KnowledgeBase.id == knowledge_base_id).with_for_update()
        )
        generation = await self._session.scalar(
            select(KnowledgeBaseIndexGeneration)
            .where(KnowledgeBaseIndexGeneration.id == generation_id)
            .with_for_update()
        )
        document = await self._session.scalar(
            select(Document).where(Document.id == document_id).with_for_update()
        )
        version = await self._session.scalar(
            select(DocumentVersion).where(DocumentVersion.id == version_id).with_for_update()
        )
        state = await self._session.scalar(
            select(DocumentIndexState)
            .where(
                DocumentIndexState.document_version_id == version_id,
                DocumentIndexState.index_generation_id == generation_id,
            )
            .with_for_update()
        )
        if (
            knowledge_base is None
            or generation is None
            or document is None
            or version is None
            or state is None
            or knowledge_base.status != "active"
            or knowledge_base.active_index_generation_id != generation.id
            or generation.knowledge_base_id != knowledge_base.id
            or generation.status != "active"
            or document.knowledge_base_id != knowledge_base.id
            or document.current_version_id is not None
            or document.pending_version_id != version.id
            or document.status != "processing"
            or version.document_id != document.id
        ):
            raise ValueError("ingestion stage state is invalid")
        return knowledge_base, generation, document, version, state

    async def commit_parse_stage(
        self,
        expected: ParseStageInput,
        *,
        parsed_object_key: str,
        parsed_checksum_sha256: str,
        parser_name: str,
        parser_version: str,
        parser_config: dict[str, object],
    ) -> None:
        _kb, _generation, document, version, _state = await self._lock_stage_graph(
            knowledge_base_id=expected.knowledge_base_id,
            generation_id=expected.generation_id,
            document_id=expected.document_id,
            version_id=expected.version_id,
        )
        existing_parsed = (
            version.parsed_object_key,
            version.parsed_object_checksum_sha256,
        )
        if (
            document.checksum_sha256 != expected.source_checksum_sha256
            or version.source_object_key != expected.source_object_key
            or version.source_checksum_sha256 != expected.source_checksum_sha256
            or version.source_extension != expected.source_extension
            or version.parser_name != expected.parser_name
            or version.parser_version != expected.parser_version
            or version.parser_config != expected.parser_config
            or expected.version_status != "parsing"
            or version.status != expected.version_status
            or existing_parsed
            not in {
                (None, None),
                (parsed_object_key, parsed_checksum_sha256),
            }
        ):
            raise ValueError("ingestion stage state is invalid")
        version.parsed_object_key = parsed_object_key
        version.parsed_object_checksum_sha256 = parsed_checksum_sha256
        version.parser_name = parser_name
        version.parser_version = parser_version
        version.parser_config = parser_config
        version.status = "chunking"

    async def commit_parse_started(self, expected: ParseStageInput) -> None:
        _kb, _generation, document, version, _state = await self._lock_stage_graph(
            knowledge_base_id=expected.knowledge_base_id,
            generation_id=expected.generation_id,
            document_id=expected.document_id,
            version_id=expected.version_id,
        )
        if (
            document.checksum_sha256 != expected.source_checksum_sha256
            or version.source_object_key != expected.source_object_key
            or version.source_checksum_sha256 != expected.source_checksum_sha256
            or version.source_extension != expected.source_extension
            or version.parser_name != expected.parser_name
            or version.parser_version != expected.parser_version
            or version.parser_config != expected.parser_config
            or version.status not in {"uploaded", "parsing"}
            or version.parsed_object_key is not None
            or version.parsed_object_checksum_sha256 is not None
        ):
            raise ValueError("ingestion stage state is invalid")
        version.status = "parsing"

    async def commit_chunk_stage(
        self,
        expected: ChunkStageInput,
        *,
        manifest_object_key: str,
        manifest_checksum_sha256: str,
        chunk_config_hash: str,
        chunker_name: str,
        chunker_version: str,
        chunker_config: dict[str, object],
        chunk_count: int,
    ) -> None:
        _kb, _generation, document, version, state = await self._lock_stage_graph(
            knowledge_base_id=expected.knowledge_base_id,
            generation_id=expected.generation_id,
            document_id=expected.document_id,
            version_id=expected.version_id,
        )
        existing_manifest = (
            version.chunk_manifest_object_key,
            version.chunk_manifest_checksum_sha256,
            version.chunk_config_hash,
        )
        if (
            document.checksum_sha256 != expected.source_checksum_sha256
            or version.source_checksum_sha256 != expected.source_checksum_sha256
            or version.source_extension != expected.source_extension
            or version.parsed_object_key != expected.parsed_object_key
            or version.parsed_object_checksum_sha256 != expected.parsed_object_checksum_sha256
            or version.parser_name != expected.parser_name
            or version.parser_version != expected.parser_version
            or version.parser_config != expected.parser_config
            or version.chunker_name != expected.chunker_name
            or version.chunker_version != expected.chunker_version
            or version.chunker_config != expected.chunker_config
            or version.status != "chunking"
            or existing_manifest
            not in {
                (None, None, None),
                (manifest_object_key, manifest_checksum_sha256, chunk_config_hash),
            }
            or state.status != "queued"
            or state.expected_point_count is not None
            or state.actual_point_count is not None
            or state.error_code is not None
            or state.validated_at is not None
            or state.chunk_manifest_checksum_sha256 is not None
            or state.embedding_config_hash is not None
            or state.next_chunk_index != 0
            or state.safe_error_message is not None
            or chunk_count < 1
        ):
            raise ValueError("ingestion stage state is invalid")
        version.chunk_manifest_object_key = manifest_object_key
        version.chunk_manifest_checksum_sha256 = manifest_checksum_sha256
        version.chunk_config_hash = chunk_config_hash
        version.chunker_name = chunker_name
        version.chunker_version = chunker_version
        version.chunker_config = chunker_config
        version.chunk_count = chunk_count
        version.status = "embedding"
        state.status = "embedding"
        state.expected_point_count = chunk_count
        state.chunk_manifest_checksum_sha256 = manifest_checksum_sha256
        state.next_chunk_index = 0

    async def commit_embed_index_batch(
        self,
        expected: EmbedIndexStageInput,
        *,
        next_chunk_index: int,
    ) -> None:
        _knowledge_base, generation, document, version, state = await self._lock_stage_graph(
            knowledge_base_id=expected.knowledge_base_id,
            generation_id=expected.generation_id,
            document_id=expected.document_id,
            version_id=expected.version_id,
        )
        if (
            type(next_chunk_index) is not int
            or not (
                expected.next_chunk_index < next_chunk_index <= expected.chunk_count
                or expected.next_chunk_index == next_chunk_index == expected.chunk_count
            )
            or generation.embedding_profile_id != expected.model_profile_id
            or generation.embedding_config_hash != expected.embedding_config_hash
            or generation.index_profile_hash != expected.embedding_config_hash
            or generation.qdrant_collection_name != expected.qdrant_collection_name
            or canonical_json_bytes(generation.embedding_config_snapshot)
            != expected.embedding_snapshot_canonical
            or canonical_json_bytes(generation.filter_schema_snapshot)
            != expected.filter_snapshot_canonical
            or generation.applied_filter_schema_revision != expected.applied_filter_schema_revision
            or canonical_json_bytes(document.metadata_) != expected.document_metadata_canonical
            or document.checksum_sha256 != expected.source_checksum_sha256
            or version.source_checksum_sha256 != expected.source_checksum_sha256
            or version.parsed_object_checksum_sha256 != expected.parsed_checksum_sha256
            or version.parser_name != expected.parser_name
            or version.parser_version != expected.parser_version
            or version.parser_config != expected.parser_config
            or version.chunker_name != expected.chunker_name
            or version.chunker_version != expected.chunker_version
            or version.chunker_config != expected.chunker_config
            or version.chunk_config_hash != expected.chunk_config_hash
            or version.chunk_manifest_object_key != expected.manifest_object_key
            or version.chunk_manifest_checksum_sha256 != expected.manifest_checksum_sha256
            or version.chunk_count != expected.chunk_count
            or version.status != expected.version_status
            or state.status != expected.index_state_status
            or state.expected_point_count != expected.chunk_count
            or state.chunk_manifest_checksum_sha256 != expected.manifest_checksum_sha256
            or state.embedding_config_hash != expected.index_state_embedding_config_hash
            or state.next_chunk_index != expected.next_chunk_index
        ):
            raise ValueError("ingestion stage state is invalid")
        version.status = "indexing"
        state.status = "indexing"
        state.embedding_config_hash = expected.embedding_config_hash
        state.next_chunk_index = next_chunk_index

    @staticmethod
    def _validate_indexed_snapshot(
        expected: EmbedIndexStageInput,
        generation: KnowledgeBaseIndexGeneration,
        document: Document,
        version: DocumentVersion,
        state: DocumentIndexState,
    ) -> None:
        if (
            generation.embedding_profile_id != expected.model_profile_id
            or generation.embedding_config_hash != expected.embedding_config_hash
            or generation.index_profile_hash != expected.embedding_config_hash
            or generation.qdrant_collection_name != expected.qdrant_collection_name
            or canonical_json_bytes(generation.embedding_config_snapshot)
            != expected.embedding_snapshot_canonical
            or canonical_json_bytes(generation.filter_schema_snapshot)
            != expected.filter_snapshot_canonical
            or generation.applied_filter_schema_revision != expected.applied_filter_schema_revision
            or canonical_json_bytes(document.metadata_) != expected.document_metadata_canonical
            or document.checksum_sha256 != expected.source_checksum_sha256
            or version.source_checksum_sha256 != expected.source_checksum_sha256
            or version.parsed_object_checksum_sha256 != expected.parsed_checksum_sha256
            or version.parser_name != expected.parser_name
            or version.parser_version != expected.parser_version
            or version.parser_config != expected.parser_config
            or version.chunker_name != expected.chunker_name
            or version.chunker_version != expected.chunker_version
            or version.chunker_config != expected.chunker_config
            or version.chunk_config_hash != expected.chunk_config_hash
            or version.chunk_manifest_object_key != expected.manifest_object_key
            or version.chunk_manifest_checksum_sha256 != expected.manifest_checksum_sha256
            or version.chunk_count != expected.chunk_count
            or version.status != "indexing"
            or state.status != "indexing"
            or state.expected_point_count != expected.chunk_count
            or state.chunk_manifest_checksum_sha256 != expected.manifest_checksum_sha256
            or state.embedding_config_hash != expected.embedding_config_hash
            or state.next_chunk_index != expected.chunk_count
            or state.error_code is not None
            or state.safe_error_message is not None
            or state.validated_at is not None
        ):
            raise ValueError("ingestion stage state is invalid")

    async def commit_validate_count(
        self,
        expected: EmbedIndexStageInput,
        *,
        actual_count: int,
    ) -> None:
        _knowledge_base, generation, document, version, state = await self._lock_stage_graph(
            knowledge_base_id=expected.knowledge_base_id,
            generation_id=expected.generation_id,
            document_id=expected.document_id,
            version_id=expected.version_id,
        )
        self._validate_indexed_snapshot(expected, generation, document, version, state)
        if (
            type(actual_count) is not int
            or actual_count < 0
            or state.actual_point_count not in {None, actual_count}
        ):
            raise ValueError("ingestion stage state is invalid")
        state.actual_point_count = actual_count

    async def commit_validate_stage(
        self,
        expected: EmbedIndexStageInput,
        *,
        actual_count: int,
    ) -> None:
        _knowledge_base, generation, document, version, state = await self._lock_stage_graph(
            knowledge_base_id=expected.knowledge_base_id,
            generation_id=expected.generation_id,
            document_id=expected.document_id,
            version_id=expected.version_id,
        )
        self._validate_indexed_snapshot(expected, generation, document, version, state)
        if (
            type(actual_count) is not int
            or actual_count != expected.chunk_count
            or state.actual_point_count != actual_count
        ):
            raise ValueError("ingestion stage state is invalid")
        validated_at = cast(datetime, await self._session.scalar(select(func.clock_timestamp())))
        state.status = "validated"
        state.validated_at = validated_at

    async def commit_terminal_failure(
        self,
        lease: JobLease,
        *,
        retryable: bool,
        error_code: str,
        safe_error_message: str,
    ) -> None:
        self._validate_terminal_failure(
            retryable=retryable,
            error_code=error_code,
            safe_error_message=safe_error_message,
        )
        if lease.target_type != "document_version" or lease.index_generation_id is None:
            raise ValueError("terminal ingestion failure is invalid")

        identity = (
            await self._session.execute(
                select(DocumentVersion.document_id, Document.knowledge_base_id)
                .join(Document, Document.id == DocumentVersion.document_id)
                .where(DocumentVersion.id == lease.target_id)
            )
        ).one_or_none()
        if identity is None:
            raise LostLeaseError
        document_id, knowledge_base_id = identity

        knowledge_base = await self._session.scalar(
            select(KnowledgeBase).where(KnowledgeBase.id == knowledge_base_id).with_for_update()
        )
        generation = await self._session.scalar(
            select(KnowledgeBaseIndexGeneration)
            .where(KnowledgeBaseIndexGeneration.id == lease.index_generation_id)
            .with_for_update()
        )
        document = await self._session.scalar(
            select(Document).where(Document.id == document_id).with_for_update()
        )
        version = await self._session.scalar(
            select(DocumentVersion).where(DocumentVersion.id == lease.target_id).with_for_update()
        )
        state = await self._session.scalar(
            select(DocumentIndexState)
            .where(
                DocumentIndexState.document_version_id == lease.target_id,
                DocumentIndexState.index_generation_id == lease.index_generation_id,
            )
            .with_for_update()
        )
        job = await self._session.scalar(
            select(Job)
            .where(
                Job.id == lease.id,
                Job.knowledge_base_id == knowledge_base_id,
                Job.operation == lease.operation,
                Job.target_type == lease.target_type,
                Job.target_id == lease.target_id,
                Job.target_revision.is_not_distinct_from(lease.target_revision),
                Job.index_generation_id.is_not_distinct_from(lease.index_generation_id),
                Job.stage.is_not_distinct_from(lease.stage),
                Job.status == "running",
                Job.lease_owner == lease.lease_owner,
                Job.lease_epoch == lease.lease_epoch,
                Job.lease_expires_at.is_not(None),
                Job.lease_expires_at > func.clock_timestamp(),
                Job.cancel_requested_at.is_(None),
            )
            .with_for_update()
        )
        if (
            knowledge_base is None
            or generation is None
            or document is None
            or version is None
            or state is None
            or job is None
        ):
            raise LostLeaseError

        finished_at = cast(datetime, await self._session.scalar(select(func.clock_timestamp())))
        self._apply_terminal_failure(
            knowledge_base=knowledge_base,
            generation=generation,
            document=document,
            version=version,
            state=state,
            job=job,
            retryable=retryable,
            error_code=error_code,
            safe_error_message=safe_error_message,
            finished_at=finished_at,
        )

    @staticmethod
    def _validate_terminal_failure(
        *,
        retryable: bool,
        error_code: str,
        safe_error_message: str,
    ) -> None:
        if (
            type(retryable) is not bool
            or type(error_code) is not str
            or not error_code
            or len(error_code) > 64
            or type(safe_error_message) is not str
            or not safe_error_message
            or len(safe_error_message) > 500
        ):
            raise ValueError("terminal ingestion failure is invalid")

    @staticmethod
    def _apply_terminal_failure(
        *,
        knowledge_base: KnowledgeBase,
        generation: KnowledgeBaseIndexGeneration,
        document: Document,
        version: DocumentVersion,
        state: DocumentIndexState,
        job: Job,
        retryable: bool,
        error_code: str,
        safe_error_message: str,
        finished_at: datetime,
    ) -> None:
        safe_version = (
            version.document_id == document.id
            and document.knowledge_base_id == knowledge_base.id
            and generation.knowledge_base_id == knowledge_base.id
            and version.base_version_id is None
        )
        safe_unactivated_version = safe_version and document.current_version_id != version.id
        if (
            safe_unactivated_version
            and document.current_version_id is None
            and document.pending_version_id == version.id
            and document.status == "processing"
            and document.deleted_at is None
        ):
            document.status = "failed"
        elif (
            safe_unactivated_version
            and document.current_version_id is not None
            and document.pending_version_id == version.id
        ):
            document.pending_version_id = None
        if safe_unactivated_version and version.status in {
            "uploaded",
            "parsing",
            "chunking",
            "embedding",
            "indexing",
        }:
            version.status = "failed"
        if safe_unactivated_version and state.status in {
            "queued",
            "embedding",
            "indexing",
            "validated",
        }:
            state.status = "failed"
            state.error_code = error_code
            state.safe_error_message = safe_error_message
        job.status = "failed"
        job.retryable = retryable
        job.next_retry_at = None
        job.error_code = error_code
        job.error_message_sanitized = safe_error_message
        job.lease_owner = None
        job.lease_expires_at = None
        job.worker_heartbeat_at = None
        job.finished_at = finished_at

    async def commit_exhausted_failure(self, candidate: ExhaustedJob) -> None:
        if (
            candidate.operation != "ingest_document"
            or candidate.target_type != "document_version"
            or candidate.index_generation_id is None
            or candidate.attempt_count != candidate.max_attempts
        ):
            raise ValueError("exhausted ingestion failure is invalid")
        job_fence = SqlAlchemyJobRepository.exhausted_fence(candidate)

        identity = (
            await self._session.execute(
                select(DocumentVersion.document_id, Document.knowledge_base_id)
                .join(Document, Document.id == DocumentVersion.document_id)
                .where(DocumentVersion.id == candidate.target_id)
            )
        ).one_or_none()
        if identity is None:
            raise LostLeaseError
        document_id, knowledge_base_id = identity
        if candidate.knowledge_base_id != knowledge_base_id:
            raise LostLeaseError

        knowledge_base = await self._session.scalar(
            select(KnowledgeBase).where(KnowledgeBase.id == knowledge_base_id).with_for_update()
        )
        generation = await self._session.scalar(
            select(KnowledgeBaseIndexGeneration)
            .where(KnowledgeBaseIndexGeneration.id == candidate.index_generation_id)
            .with_for_update()
        )
        document = await self._session.scalar(
            select(Document).where(Document.id == document_id).with_for_update()
        )
        version = await self._session.scalar(
            select(DocumentVersion)
            .where(DocumentVersion.id == candidate.target_id)
            .with_for_update()
        )
        state = await self._session.scalar(
            select(DocumentIndexState)
            .where(
                DocumentIndexState.document_version_id == candidate.target_id,
                DocumentIndexState.index_generation_id == candidate.index_generation_id,
            )
            .with_for_update()
        )
        job = await self._session.scalar(select(Job).where(*job_fence).with_for_update())
        if (
            knowledge_base is None
            or generation is None
            or document is None
            or version is None
            or state is None
            or job is None
        ):
            raise LostLeaseError

        finished_at = cast(datetime, await self._session.scalar(select(func.clock_timestamp())))
        self._apply_terminal_failure(
            knowledge_base=knowledge_base,
            generation=generation,
            document=document,
            version=version,
            state=state,
            job=job,
            retryable=False,
            error_code="JOB_ATTEMPTS_EXHAUSTED",
            safe_error_message="Job attempts exhausted",
            finished_at=finished_at,
        )

    async def commit_activation(
        self,
        expected: ActivationStageInput,
        lease: JobLease,
    ) -> None:
        knowledge_base = await self._session.scalar(
            select(KnowledgeBase)
            .where(KnowledgeBase.id == expected.knowledge_base_id)
            .with_for_update()
        )
        generation = await self._session.scalar(
            select(KnowledgeBaseIndexGeneration)
            .where(KnowledgeBaseIndexGeneration.id == expected.generation_id)
            .with_for_update()
        )
        document = await self._session.scalar(
            select(Document).where(Document.id == expected.document_id).with_for_update()
        )
        version = await self._session.scalar(
            select(DocumentVersion)
            .where(DocumentVersion.id == expected.version_id)
            .with_for_update()
        )
        state = await self._session.scalar(
            select(DocumentIndexState)
            .where(
                DocumentIndexState.document_version_id == expected.version_id,
                DocumentIndexState.index_generation_id == expected.generation_id,
            )
            .with_for_update()
        )
        job = await self._session.scalar(
            select(Job)
            .where(
                Job.id == expected.job_id,
                Job.id == lease.id,
                Job.knowledge_base_id == expected.knowledge_base_id,
                Job.operation == "ingest_document",
                Job.target_type == "document_version",
                Job.target_id == expected.version_id,
                Job.target_revision.is_not_distinct_from(lease.target_revision),
                Job.index_generation_id.is_not_distinct_from(expected.generation_id),
                Job.stage.is_not_distinct_from("activate"),
                Job.status == "running",
                Job.lease_owner == lease.lease_owner,
                Job.lease_epoch == lease.lease_epoch,
                Job.lease_expires_at.is_not(None),
                Job.lease_expires_at > func.clock_timestamp(),
                Job.cancel_requested_at.is_(None),
            )
            .with_for_update()
        )
        if job is None:
            raise LostLeaseError

        document_matches = (
            document is not None
            and await self._session.scalar(
                select(Document.id).where(
                    Document.id == expected.document_id,
                    Document.knowledge_base_id == expected.knowledge_base_id,
                    Document.status == "processing",
                    Document.current_version_id.is_(None),
                    Document.pending_version_id == expected.version_id,
                    Document.checksum_sha256 == expected.source_checksum_sha256,
                    Document.deleted_at.is_(None),
                )
            )
            is not None
        )
        version_matches = (
            version is not None
            and await self._session.scalar(
                select(DocumentVersion.id).where(
                    DocumentVersion.id == expected.version_id,
                    DocumentVersion.document_id == expected.document_id,
                    DocumentVersion.base_version_id.is_(None),
                    DocumentVersion.source_checksum_sha256 == expected.source_checksum_sha256,
                    DocumentVersion.detected_mime_type == expected.detected_mime_type,
                    DocumentVersion.chunk_count == expected.chunk_count,
                    DocumentVersion.status == "indexing",
                    DocumentVersion.activated_at.is_(None),
                )
            )
            is not None
        )
        state_matches = (
            state is not None
            and await self._session.scalar(
                select(DocumentIndexState.document_version_id).where(
                    DocumentIndexState.document_version_id == expected.version_id,
                    DocumentIndexState.index_generation_id == expected.generation_id,
                    DocumentIndexState.status == "validated",
                    DocumentIndexState.expected_point_count == expected.expected_point_count,
                    DocumentIndexState.actual_point_count == expected.actual_point_count,
                    DocumentIndexState.expected_point_count
                    == DocumentIndexState.actual_point_count,
                    DocumentIndexState.next_chunk_index == expected.chunk_count,
                    DocumentIndexState.validated_at.is_not(None),
                )
            )
            is not None
        )
        if (
            knowledge_base is None
            or generation is None
            or document is None
            or version is None
            or state is None
            or knowledge_base.status != "active"
            or knowledge_base.active_index_generation_id != expected.generation_id
            or generation.knowledge_base_id != expected.knowledge_base_id
            or generation.status != "active"
            or not document_matches
            or not version_matches
            or not state_matches
            or expected.expected_point_count != expected.actual_point_count
            or expected.chunk_count != expected.expected_point_count
        ):
            raise DocumentActivationConflictError("Document activation conflict")

        activated_at = cast(datetime, await self._session.scalar(select(func.clock_timestamp())))
        version.status = "ready"
        version.activated_at = activated_at
        document.current_version_id = version.id
        document.pending_version_id = None
        document.mime_type = version.detected_mime_type
        document.status = "active"
        knowledge_base.mutation_revision += 1
        self._session.add(
            KnowledgeBaseMutation(
                id=uuid4(),
                knowledge_base_id=knowledge_base.id,
                revision=knowledge_base.mutation_revision,
                mutation_type="document_activated",
                target_type="document_version",
                target_id=version.id,
                payload={"document_id": str(document.id)},
            )
        )
        job.status = "succeeded"
        job.retryable = False
        job.next_retry_at = None
        job.error_code = None
        job.error_message_sanitized = None
        job.lease_owner = None
        job.lease_expires_at = None
        job.worker_heartbeat_at = None
        job.finished_at = activated_at


class SourceObjectVerifier(Protocol):
    async def verify_object(
        self,
        object_key: str,
        *,
        expected_size: int,
        expected_checksum: str,
    ) -> object: ...


class SqlAlchemyUploadRepository:
    def __init__(self, session: AsyncSession, object_store: SourceObjectVerifier) -> None:
        self._session = session
        self._object_store = object_store

    async def preflight(self, knowledge_base_id: UUID) -> UploadPreflight | None:
        row = (
            await self._session.execute(
                select(KnowledgeBase, KnowledgeBaseIndexGeneration)
                .join(
                    KnowledgeBaseIndexGeneration,
                    KnowledgeBaseIndexGeneration.id == KnowledgeBase.active_index_generation_id,
                )
                .where(
                    KnowledgeBase.id == knowledge_base_id,
                    KnowledgeBase.status == "active",
                    KnowledgeBaseIndexGeneration.status == "active",
                )
            )
        ).one_or_none()
        if row is None:
            return None
        kb, generation = row
        if (
            not isinstance(generation.filter_schema_snapshot, dict)
            or generation.applied_filter_schema_revision is None
        ):
            return None
        return UploadPreflight(
            kb.id,
            generation.id,
            generation.filter_schema_snapshot,
            generation.applied_filter_schema_revision,
            kb.filter_schema_revision,
        )

    async def _existing_idempotency(
        self, reservation: UploadReservation
    ) -> DocumentUploadIdempotency | None:
        if reservation.idempotency_key is None:
            return None
        return (
            await self._session.execute(
                select(DocumentUploadIdempotency)
                .where(
                    DocumentUploadIdempotency.actor_api_key_id == reservation.actor.key_id,
                    DocumentUploadIdempotency.knowledge_base_id == reservation.knowledge_base_id,
                    DocumentUploadIdempotency.idempotency_key == reservation.idempotency_key,
                )
                .with_for_update()
            )
        ).scalar_one_or_none()

    @staticmethod
    def _replay(existing: DocumentUploadIdempotency, fingerprint: bytes) -> UploadReservationResult:
        if existing.request_fingerprint != fingerprint:
            raise BusinessError(409, "IDEMPOTENCY_KEY_REUSED", "Idempotency key reused")
        return UploadReservationResult.replayed(
            existing.document_id, existing.document_version_id, existing.job_id
        )

    async def reserve(self, reservation: UploadReservation) -> UploadReservationResult:
        actor = (
            await self._session.execute(
                select(ApiKey)
                .where(
                    ApiKey.id == reservation.actor.key_id,
                    ApiKey.public_id == reservation.actor.public_id,
                    ApiKey.key_type == "agent",
                    ApiKey.status == "active",
                    ApiKey.revoked_at.is_(None),
                    or_(ApiKey.not_before.is_(None), ApiKey.not_before <= func.now()),
                    or_(ApiKey.expires_at.is_(None), ApiKey.expires_at > func.now()),
                )
                .with_for_update()
            )
        ).scalar_one_or_none()
        if actor is None:
            raise BusinessError(401, "INVALID_API_KEY", "Invalid API key")
        scope = (
            await self._session.execute(
                select(ApiKeyKnowledgeBaseScope).where(
                    ApiKeyKnowledgeBaseScope.api_key_id == reservation.actor.key_id,
                    ApiKeyKnowledgeBaseScope.knowledge_base_id == reservation.knowledge_base_id,
                )
            )
        ).scalar_one_or_none()
        if scope is None:
            raise BusinessError(404, "RESOURCE_NOT_FOUND", "Resource not found")
        if Capability.INGEST.value not in actor.capabilities:
            raise BusinessError(403, "INSUFFICIENT_CAPABILITY", "Insufficient capability")

        kb = (
            await self._session.execute(
                select(KnowledgeBase)
                .where(KnowledgeBase.id == reservation.knowledge_base_id)
                .with_for_update()
            )
        ).scalar_one_or_none()
        if kb is None or kb.status != "active":
            raise BusinessError(
                409,
                "GENERATION_CONFIGURATION_CONFLICT",
                "Index generation configuration conflict",
            )

        existing = await self._existing_idempotency(reservation)
        if existing is not None:
            return self._replay(existing, reservation.request_fingerprint)

        active_repair = await self._session.scalar(
            select(Job.id)
            .where(
                Job.knowledge_base_id == kb.id,
                Job.operation == "rebuild_generation",
                Job.target_type == "index_generation",
                Job.index_generation_id == kb.active_index_generation_id,
                Job.status.in_(("queued", "running", "retry_wait")),
            )
            .order_by(Job.created_at, Job.id)
            .limit(1)
        )
        if active_repair is not None:
            raise BusinessError(
                409,
                "GENERATION_REPAIR_IN_PROGRESS",
                "Generation repair is already in progress",
                retryable=True,
            )

        preflight = reservation.preflight
        if preflight is None:
            raise BusinessError(
                409,
                "KNOWLEDGE_BASE_NOT_INDEX_CONFIGURED",
                "Knowledge base has no active index generation",
            )

        generation = (
            await self._session.execute(
                select(KnowledgeBaseIndexGeneration)
                .where(KnowledgeBaseIndexGeneration.id == preflight.generation_id)
                .with_for_update()
            )
        ).scalar_one_or_none()
        if (
            generation is None
            or generation.status != "active"
            or kb.active_index_generation_id != generation.id
            or kb.filter_schema_revision != preflight.current_filter_schema_revision
            or generation.applied_filter_schema_revision != preflight.applied_filter_schema_revision
            or generation.filter_schema_snapshot != preflight.filter_schema_snapshot
        ):
            raise BusinessError(
                409,
                "GENERATION_CONFIGURATION_CONFLICT",
                "Index generation configuration conflict",
            )
        validate_metadata_against_filter_snapshot(reservation.metadata, preflight)
        await self._object_store.verify_object(
            reservation.source_object_key,
            expected_size=reservation.source_size,
            expected_checksum=reservation.source_checksum_sha256,
        )
        if reservation.idempotency_key is not None:
            inserted_id = (
                await self._session.execute(
                    insert(DocumentUploadIdempotency)
                    .values(
                        id=uuid4(),
                        actor_api_key_id=reservation.actor.key_id,
                        knowledge_base_id=kb.id,
                        idempotency_key=reservation.idempotency_key,
                        request_fingerprint=reservation.request_fingerprint,
                        document_id=reservation.document_id,
                        document_version_id=reservation.version_id,
                        job_id=reservation.job_id,
                        result_status="accepted",
                    )
                    .on_conflict_do_nothing(
                        constraint="uq_document_upload_idempotency_actor_kb_key"
                    )
                    .returning(DocumentUploadIdempotency.id)
                )
            ).scalar_one_or_none()
            if inserted_id is None:
                existing = await self._existing_idempotency(reservation)
                if existing is None:
                    raise RuntimeError("idempotency conflict was not observable")
                return self._replay(existing, reservation.request_fingerprint)

        document_id = (
            await self._session.execute(
                insert(Document)
                .values(
                    id=reservation.document_id,
                    knowledge_base_id=kb.id,
                    display_name=reservation.display_name,
                    mime_type=reservation.detected_mime_type,
                    checksum_sha256=reservation.source_checksum_sha256,
                    pending_version_id=reservation.version_id,
                    status="processing",
                    tags=list(reservation.tags),
                    metadata_=reservation.metadata,
                )
                .on_conflict_do_nothing(
                    index_elements=[Document.knowledge_base_id, Document.checksum_sha256],
                    index_where=and_(
                        Document.deleted_at.is_(None),
                        Document.checksum_sha256.is_not(None),
                    ),
                )
                .returning(Document.id)
            )
        ).scalar_one_or_none()
        if document_id is None:
            raise BusinessError(409, "DUPLICATE_DOCUMENT", "Document content already exists")

        kb.mutation_revision += 1
        mutation = KnowledgeBaseMutation(
            id=uuid4(),
            knowledge_base_id=kb.id,
            revision=kb.mutation_revision,
            mutation_type="document_version_created",
            target_type="document_version",
            target_id=reservation.version_id,
            payload={"document_id": str(reservation.document_id)},
        )
        version = DocumentVersion(
            id=reservation.version_id,
            document_id=reservation.document_id,
            version_number=1,
            source_object_key=reservation.source_object_key,
            source_checksum_sha256=reservation.source_checksum_sha256,
            declared_mime_type=reservation.declared_mime_type,
            detected_mime_type=reservation.detected_mime_type,
            source_extension=reservation.source_extension,
            parser_name=reservation.parser_name,
            parser_version="1",
            status="uploaded",
        )
        state = DocumentIndexState(
            document_version_id=reservation.version_id,
            index_generation_id=generation.id,
            status="queued",
        )
        job = Job(
            id=reservation.job_id,
            knowledge_base_id=kb.id,
            actor_api_key_id=reservation.actor.key_id,
            target_type="document_version",
            target_id=reservation.version_id,
            target_revision=kb.mutation_revision,
            index_generation_id=generation.id,
            mutation_id=mutation.id,
            operation="ingest_document",
            stage="parse",
            status="queued",
        )
        self._session.add_all([version, mutation])
        await self._session.flush()
        self._session.add_all([state, job])
        await self._session.flush()
        return UploadReservationResult.created(
            reservation.document_id, reservation.version_id, reservation.job_id
        )


__all__ = [
    "ActivationStageInput",
    "ChunkStageInput",
    "DocumentActivationConflictError",
    "ParseStageInput",
    "SqlAlchemyIngestionPipelineRepository",
    "SqlAlchemyUploadRepository",
    "UploadPreflight",
    "UploadReservation",
    "UploadReservationResult",
]
