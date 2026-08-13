"""Fail-closed reservation for same-generation Qdrant disaster repair."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AbstractAsyncContextManager, suppress
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, TypedDict, cast
from uuid import UUID, uuid4

from sqlalchemy import and_, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from rag_service.api.errors import BusinessError
from rag_service.api.validation import JSONValue
from rag_service.db.models.documents import Document, DocumentIndexState, DocumentVersion, Job
from rag_service.db.models.knowledge_bases import (
    KnowledgeBase,
    KnowledgeBaseIndexGeneration,
    KnowledgeBaseMutation,
)
from rag_service.db.models.providers import ModelProfile, ProviderConfig, ProviderCredential
from rag_service.indexing.generation_repositories import SqlAlchemyGenerationRepository
from rag_service.indexing.generation_services import (
    BuiltEmbeddingConfiguration,
    GenerationConfigurationError,
    build_embedding_configuration,
    build_filter_snapshot,
    payload_indexes_for_filter_snapshot,
)
from rag_service.indexing.identities import (
    canonical_json_bytes,
    canonical_sha256,
    collection_name,
    point_id,
)
from rag_service.indexing.qdrant import (
    CollectionSpec,
    QdrantClient,
    QdrantConfigurationError,
    QdrantTransientError,
)
from rag_service.infrastructure.minio_store import ObjectStoreError
from rag_service.ingestion.chunkers import Chunk
from rag_service.ingestion.pipeline import (
    MAX_CHUNK_MANIFEST_BYTES,
    PipelineEmbeddingGateway,
    PipelineObjectStore,
    PipelineProviderUsageSink,
    approved_filter_metadata,
    close_async_iterators_without_masking_primary,
    iter_verified_manifest_batches,
    manifest_expectation,
    provider_usage_observer,
    qdrant_point_for,
    qdrant_point_payload,
)
from rag_service.ingestion.repositories import EmbedIndexStageInput
from rag_service.jobs.repositories import JobLease, LostLeaseError, RebuildTargetRefresh
from rag_service.jobs.runner import (
    JobExecutionContext,
    JobHandlerOutcome,
    PermanentJobError,
    RetryableJobError,
)
from rag_service.observability.logging import SafeLogContext, emit_safe_log
from rag_service.observability.metrics import METRICS, OperationalMetrics
from rag_service.observability.repositories import ProviderUsageContext
from rag_service.providers.embeddings import EmbeddingGatewayError

type SessionFactory = Callable[[], AbstractAsyncContextManager[AsyncSession]]

_ACTIVE_JOB_STATUSES = ("queued", "running", "retry_wait")
logger = logging.getLogger(__name__)


class GenerationRepairReservation(TypedDict):
    generation_id: str
    job_id: str
    status: Literal["queued"]


@dataclass(frozen=True, slots=True)
class _RepairTarget:
    document_id: UUID
    version_id: UUID
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
    document_metadata: dict[str, JSONValue]

    def fingerprint_document(self) -> dict[str, object]:
        return {
            "document_id": str(self.document_id),
            "version_id": str(self.version_id),
            "source_checksum_sha256": self.source_checksum_sha256,
            "parsed_checksum_sha256": self.parsed_checksum_sha256,
            "parser_name": self.parser_name,
            "parser_version": self.parser_version,
            "parser_config": self.parser_config,
            "chunker_name": self.chunker_name,
            "chunker_version": self.chunker_version,
            "chunker_config": self.chunker_config,
            "chunk_config_hash": self.chunk_config_hash,
            "manifest_object_key": self.manifest_object_key,
            "manifest_checksum_sha256": self.manifest_checksum_sha256,
            "chunk_count": self.chunk_count,
            "version_created_at": self.version_created_at.isoformat(),
            "document_metadata": self.document_metadata,
        }


@dataclass(frozen=True, slots=True)
class _RepairSnapshot:
    knowledge_base_id: UUID
    generation_id: UUID
    anchor_mutation_id: UUID
    mutation_revision: int
    collection_spec: CollectionSpec
    point_total: int
    fingerprint: str
    embedding: BuiltEmbeddingConfiguration = field(compare=False, repr=False)
    filter_snapshot: dict[str, object] = field(compare=False, repr=False)
    applied_filter_schema_revision: int
    embedding_profile_id: UUID
    targets: tuple[_RepairTarget, ...] = field(compare=False, repr=False)


def _not_active() -> BusinessError:
    return BusinessError(
        409,
        "GENERATION_REPAIR_NOT_ACTIVE",
        "Only the active index generation can be repaired",
    )


def _configuration_conflict() -> BusinessError:
    return BusinessError(
        409,
        "GENERATION_REPAIR_CONFIGURATION_CONFLICT",
        "Generation configuration no longer matches its immutable snapshot",
    )


def _manifest_unavailable() -> BusinessError:
    return BusinessError(
        409,
        "GENERATION_REPAIR_MANIFEST_UNAVAILABLE",
        "Canonical chunk manifests are unavailable for generation repair",
    )


def _repair_in_progress() -> BusinessError:
    return BusinessError(
        409,
        "GENERATION_REPAIR_IN_PROGRESS",
        "Generation repair is already in progress",
    )


def _ingestion_in_progress() -> BusinessError:
    return BusinessError(
        409,
        "GENERATION_REPAIR_INGESTION_IN_PROGRESS",
        "Document ingestion is in progress for the active generation",
        retryable=True,
    )


def _collection_conflict() -> BusinessError:
    return BusinessError(
        409,
        "GENERATION_REPAIR_COLLECTION_CONFLICT",
        "The generation collection does not match its immutable specification",
    )


def _collection_not_empty() -> BusinessError:
    return BusinessError(
        409,
        "GENERATION_REPAIR_COLLECTION_NOT_EMPTY",
        "A preexisting generation collection must be empty before repair",
    )


def _qdrant_unavailable() -> BusinessError:
    return BusinessError(
        503,
        "QDRANT_UNAVAILABLE",
        "Qdrant is unavailable",
        retryable=True,
    )


def _state_changed() -> BusinessError:
    return BusinessError(
        409,
        "GENERATION_REPAIR_STATE_CHANGED",
        "Generation repair inputs changed; retry the reservation",
        retryable=True,
    )


class GenerationRepairService:
    """Reserve a fenced repair job without mutating visible generation facts."""

    def __init__(
        self,
        *,
        session_factory: SessionFactory,
        qdrant: QdrantClient,
        metrics: OperationalMetrics = METRICS,
    ) -> None:
        if not callable(session_factory) or not callable(
            getattr(qdrant, "collection_exists", None)
        ):
            raise ValueError("generation repair dependencies are invalid")
        self._session_factory = session_factory
        self._qdrant = qdrant
        self._metrics = metrics

    def _observe_queued_job(
        self,
        *,
        knowledge_base_id: UUID,
        generation_id: UUID,
        job_id: UUID,
    ) -> None:
        with suppress(BaseException):
            self._metrics.record_job_state(state="queued")
        try:
            emit_safe_log(
                logger,
                logging.INFO,
                "job.state.changed",
                context=SafeLogContext(
                    knowledge_base_id=knowledge_base_id,
                    job_id=job_id,
                    generation_id=generation_id,
                ),
                operation="rebuild_generation",
                state="queued",
            )
        except BaseException:
            return

    async def reserve(self, generation_id: UUID) -> GenerationRepairReservation:
        if type(generation_id) is not UUID:
            raise BusinessError(422, "VALIDATION_ERROR", "Invalid generation repair request")

        async with self._session_factory() as session:
            initial = await self._read_snapshot(session, generation_id)
            if await self._active_repair_exists(session, generation_id):
                raise _repair_in_progress()

        await self._probe_collection(initial.collection_spec)

        job_id = uuid4()
        try:
            async with self._session_factory() as session, session.begin():
                knowledge_base = cast(
                    KnowledgeBase | None,
                    await session.scalar(
                        select(KnowledgeBase)
                        .where(KnowledgeBase.id == initial.knowledge_base_id)
                        .with_for_update()
                    ),
                )
                generation = cast(
                    KnowledgeBaseIndexGeneration | None,
                    await session.scalar(
                        select(KnowledgeBaseIndexGeneration)
                        .where(KnowledgeBaseIndexGeneration.id == generation_id)
                        .with_for_update()
                    ),
                )
                if knowledge_base is None or generation is None:
                    raise BusinessError(404, "RESOURCE_NOT_FOUND", "Resource not found")
                repository = SqlAlchemyGenerationRepository(session)
                await repository.acquire_collection_fence(initial.collection_spec.name)
                if await self._active_repair_exists(session, generation_id):
                    raise _repair_in_progress()
                if await self._active_ingestion_exists(
                    session,
                    initial.knowledge_base_id,
                    generation_id,
                ):
                    raise _ingestion_in_progress()
                current = await self._snapshot_from_rows(
                    session,
                    knowledge_base,
                    generation,
                    lock_configuration=True,
                )
                if current != initial:
                    raise _state_changed()
                # The first probe avoids holding database locks across the usual
                # network path. This second probe is the ownership boundary: an
                # ingestion that wrote points and became terminal after the first
                # probe must not leave a polluted collection for repair to claim.
                await self._probe_collection(current.collection_spec)
                session.add(
                    Job(
                        id=job_id,
                        knowledge_base_id=current.knowledge_base_id,
                        target_type="index_generation",
                        target_id=current.generation_id,
                        target_revision=current.mutation_revision,
                        index_generation_id=current.generation_id,
                        mutation_id=current.anchor_mutation_id,
                        parent_job_id=None,
                        root_job_id=None,
                        idempotency_key=None,
                        operation="rebuild_generation",
                        stage="indexing",
                        status="queued",
                        progress_current=0,
                        progress_total=current.point_total,
                        attempt_count=0,
                        max_attempts=5,
                        next_retry_at=None,
                        worker_heartbeat_at=None,
                        cancel_requested_at=None,
                        error_code=None,
                        error_message_sanitized=None,
                        started_at=None,
                        finished_at=None,
                        lease_owner=None,
                        lease_epoch=0,
                        lease_expires_at=None,
                        retryable=True,
                        resume_stage="indexing",
                        actor_api_key_id=None,
                    )
                )
                await session.flush()
        except IntegrityError:
            async with self._session_factory() as session:
                if await self._active_repair_exists(session, generation_id):
                    raise _repair_in_progress() from None
            raise BusinessError(500, "INTERNAL_ERROR", "Internal server error") from None

        self._observe_queued_job(
            knowledge_base_id=initial.knowledge_base_id,
            generation_id=generation_id,
            job_id=job_id,
        )

        return {
            "generation_id": str(generation_id),
            "job_id": str(job_id),
            "status": "queued",
        }

    async def _read_snapshot(
        self,
        session: AsyncSession,
        generation_id: UUID,
    ) -> _RepairSnapshot:
        generation = await session.get(KnowledgeBaseIndexGeneration, generation_id)
        if generation is None:
            raise BusinessError(404, "RESOURCE_NOT_FOUND", "Resource not found")
        knowledge_base = await session.get(KnowledgeBase, generation.knowledge_base_id)
        if knowledge_base is None:
            raise BusinessError(404, "RESOURCE_NOT_FOUND", "Resource not found")
        return await self._snapshot_from_rows(
            session,
            knowledge_base,
            generation,
            lock_configuration=False,
        )

    async def _snapshot_from_rows(
        self,
        session: AsyncSession,
        knowledge_base: KnowledgeBase,
        generation: KnowledgeBaseIndexGeneration,
        *,
        lock_configuration: bool,
    ) -> _RepairSnapshot:
        if (
            generation.knowledge_base_id != knowledge_base.id
            or generation.status != "active"
            or knowledge_base.active_index_generation_id != generation.id
        ):
            raise _not_active()
        try:
            expected_collection = collection_name(knowledge_base.id, generation.id)
            if (
                generation.qdrant_collection_name != expected_collection
                or generation.embedding_profile_id is None
                or generation.distance is None
                or generation.embedding_config_snapshot is None
                or generation.embedding_config_hash is None
                or generation.index_profile_hash is None
                or generation.filter_schema_snapshot is None
                or generation.applied_filter_schema_revision is None
            ):
                raise ValueError
            embedding = await self._embedding_configuration(
                session,
                generation,
                lock_configuration=lock_configuration,
            )
            filter_snapshot = build_filter_snapshot(
                cast(dict[str, object], generation.filter_schema_snapshot)
            )
            if filter_snapshot != generation.filter_schema_snapshot:
                raise ValueError
            spec = CollectionSpec(
                expected_collection,
                embedding.gateway_snapshot.dimension,
                generation.distance,
                payload_indexes_for_filter_snapshot(filter_snapshot),
            )
        except (
            GenerationConfigurationError,
            KeyError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ):
            raise _configuration_conflict() from None

        anchor_mutation_id = await self._configuration_anchor(
            session,
            knowledge_base.id,
            generation,
        )
        targets, point_total = await self._canonical_targets(session, knowledge_base.id, generation)
        fingerprint = canonical_sha256(
            {
                "knowledge_base_id": str(knowledge_base.id),
                "generation_id": str(generation.id),
                "anchor_mutation_id": str(anchor_mutation_id),
                "mutation_revision": knowledge_base.mutation_revision,
                "collection_spec": {
                    "name": spec.name,
                    "dimension": spec.dimension,
                    "distance": spec.distance,
                    "payload_indexes": [
                        {
                            "path": index.path,
                            "schema": index.schema,
                            "params": [[name, value] for name, value in index.params],
                        }
                        for index in spec.payload_indexes
                    ],
                },
                "embedding_snapshot": embedding.snapshot,
                "embedding_profile_id": str(generation.embedding_profile_id),
                "filter_snapshot": filter_snapshot,
                "applied_filter_schema_revision": generation.applied_filter_schema_revision,
                "targets": [target.fingerprint_document() for target in targets],
                "point_total": point_total,
            }
        )
        return _RepairSnapshot(
            knowledge_base_id=knowledge_base.id,
            generation_id=generation.id,
            anchor_mutation_id=anchor_mutation_id,
            mutation_revision=knowledge_base.mutation_revision,
            collection_spec=spec,
            point_total=point_total,
            fingerprint=fingerprint,
            embedding=embedding,
            filter_snapshot=filter_snapshot,
            applied_filter_schema_revision=generation.applied_filter_schema_revision,
            embedding_profile_id=generation.embedding_profile_id,
            targets=targets,
        )

    @staticmethod
    async def _configuration_anchor(
        session: AsyncSession,
        knowledge_base_id: UUID,
        generation: KnowledgeBaseIndexGeneration,
    ) -> UUID:
        if generation.caught_up_revision is None:
            raise _configuration_conflict()
        anchor = cast(
            KnowledgeBaseMutation | None,
            await session.scalar(
                select(KnowledgeBaseMutation).where(
                    KnowledgeBaseMutation.knowledge_base_id == knowledge_base_id,
                    KnowledgeBaseMutation.revision == generation.caught_up_revision,
                    KnowledgeBaseMutation.mutation_type == "index_config_changed",
                    KnowledgeBaseMutation.target_type == "index_generation",
                    KnowledgeBaseMutation.target_id == generation.id,
                )
            ),
        )
        payload = None if anchor is None else anchor.payload
        if (
            anchor is None
            or type(payload) is not dict
            or payload.get("generation_id") != str(generation.id)
            or payload.get("embedding_profile_id") != str(generation.embedding_profile_id)
            or payload.get("index_profile_hash") != generation.index_profile_hash
            or payload.get("embedding_config_hash") != generation.embedding_config_hash
            or payload.get("applied_filter_schema_revision")
            != generation.applied_filter_schema_revision
            or payload.get("validation_manifest_hash") != generation.validation_manifest_hash
        ):
            raise _configuration_conflict()
        return anchor.id

    async def _embedding_configuration(
        self,
        session: AsyncSession,
        generation: KnowledgeBaseIndexGeneration,
        *,
        lock_configuration: bool,
    ) -> BuiltEmbeddingConfiguration:
        profile_id = generation.embedding_profile_id
        snapshot = generation.embedding_config_snapshot
        if profile_id is None or type(snapshot) is not dict:
            raise GenerationConfigurationError(
                "GENERATION_CONFIGURATION_CONFLICT",
                "Generation configuration changed",
            )
        try:
            credential_id = UUID(cast(str, snapshot["credential_id"]))
        except (KeyError, TypeError, ValueError):
            raise GenerationConfigurationError(
                "GENERATION_CONFIGURATION_CONFLICT",
                "Generation configuration changed",
            ) from None

        if lock_configuration:
            profile = cast(
                ModelProfile | None,
                await session.scalar(
                    select(ModelProfile).where(ModelProfile.id == profile_id).with_for_update()
                ),
            )
            if profile is None:
                raise GenerationConfigurationError(
                    "GENERATION_CONFIGURATION_CONFLICT",
                    "Generation configuration changed",
                )
            provider = cast(
                ProviderConfig | None,
                await session.scalar(
                    select(ProviderConfig)
                    .where(ProviderConfig.id == profile.provider_config_id)
                    .with_for_update()
                ),
            )
            credential = cast(
                ProviderCredential | None,
                await session.scalar(
                    select(ProviderCredential)
                    .where(ProviderCredential.id == credential_id)
                    .with_for_update()
                ),
            )
            if provider is None or credential is None:
                raise GenerationConfigurationError(
                    "GENERATION_CONFIGURATION_CONFLICT",
                    "Generation configuration changed",
                )

        repository = SqlAlchemyGenerationRepository(session)
        source = await repository.load_embedding_source(profile_id)
        snapshot_credential = await repository.load_credential(credential_id)
        if source is None or snapshot_credential is None or generation.distance is None:
            raise GenerationConfigurationError(
                "GENERATION_CONFIGURATION_CONFLICT",
                "Generation configuration changed",
            )
        current = build_embedding_configuration(
            source.profile,
            source.provider,
            snapshot_credential,
            distance=generation.distance,
            _require_enabled=False,
            _snapshot_credential_id=credential_id,
        )
        if (
            current.snapshot != snapshot
            or current.semantic_hash != generation.embedding_config_hash
            or current.semantic_hash != generation.index_profile_hash
        ):
            raise GenerationConfigurationError(
                "GENERATION_CONFIGURATION_CONFLICT",
                "Generation configuration changed",
            )
        return current

    async def _canonical_targets(
        self,
        session: AsyncSession,
        knowledge_base_id: UUID,
        generation: KnowledgeBaseIndexGeneration,
    ) -> tuple[tuple[_RepairTarget, ...], int]:
        rows = (
            await session.execute(
                select(Document, DocumentVersion, DocumentIndexState)
                .outerjoin(
                    DocumentVersion,
                    and_(
                        DocumentVersion.id == Document.current_version_id,
                        DocumentVersion.document_id == Document.id,
                    ),
                )
                .outerjoin(
                    DocumentIndexState,
                    and_(
                        DocumentIndexState.document_version_id == DocumentVersion.id,
                        DocumentIndexState.index_generation_id == generation.id,
                    ),
                )
                .where(
                    Document.knowledge_base_id == knowledge_base_id,
                    Document.status == "active",
                    Document.deleted_at.is_(None),
                )
                .order_by(Document.id)
            )
        ).all()
        targets: list[_RepairTarget] = []
        point_total = 0
        for document, version, state in rows:
            if (
                version is None
                or state is None
                or document.current_version_id != version.id
                or version.document_id != document.id
                or document.checksum_sha256 != version.source_checksum_sha256
                or version.status != "ready"
                or version.activated_at is None
                or version.parsed_object_checksum_sha256 is None
                or version.parser_name is None
                or version.parser_version is None
                or version.chunker_name is None
                or version.chunker_version is None
                or version.chunk_count is None
                or version.chunk_count < 1
                or version.chunk_manifest_object_key is None
                or version.chunk_manifest_checksum_sha256 is None
                or version.chunk_config_hash is None
                or state.status != "validated"
                or state.validated_at is None
                or state.expected_point_count != version.chunk_count
                or state.actual_point_count != version.chunk_count
                or state.chunk_manifest_checksum_sha256 != version.chunk_manifest_checksum_sha256
                or state.embedding_config_hash != generation.embedding_config_hash
                or state.next_chunk_index != version.chunk_count
            ):
                raise _manifest_unavailable()
            try:
                parser_config = json.loads(canonical_json_bytes(version.parser_config))
                chunker_config = json.loads(canonical_json_bytes(version.chunker_config))
                document_metadata = json.loads(canonical_json_bytes(document.metadata_))
                if (
                    type(parser_config) is not dict
                    or type(chunker_config) is not dict
                    or type(document_metadata) is not dict
                    or version.chunk_config_hash
                    != canonical_sha256(
                        {
                            "config": chunker_config,
                            "name": version.chunker_name,
                            "version": version.chunker_version,
                        }
                    )
                ):
                    raise ValueError
            except (TypeError, ValueError, json.JSONDecodeError):
                raise _manifest_unavailable() from None
            targets.append(
                _RepairTarget(
                    document_id=document.id,
                    version_id=version.id,
                    source_checksum_sha256=version.source_checksum_sha256,
                    parsed_checksum_sha256=version.parsed_object_checksum_sha256,
                    parser_name=version.parser_name,
                    parser_version=version.parser_version,
                    parser_config=parser_config,
                    chunker_name=version.chunker_name,
                    chunker_version=version.chunker_version,
                    chunker_config=chunker_config,
                    chunk_config_hash=version.chunk_config_hash,
                    manifest_object_key=version.chunk_manifest_object_key,
                    manifest_checksum_sha256=version.chunk_manifest_checksum_sha256,
                    chunk_count=version.chunk_count,
                    version_created_at=version.created_at,
                    document_metadata=cast(dict[str, JSONValue], document_metadata),
                )
            )
            point_total += version.chunk_count
        return tuple(targets), point_total

    async def _active_repair_exists(
        self,
        session: AsyncSession,
        generation_id: UUID,
    ) -> bool:
        statement = (
            select(Job.id)
            .where(
                Job.operation == "rebuild_generation",
                Job.target_type == "index_generation",
                Job.target_id == generation_id,
                Job.index_generation_id == generation_id,
                Job.status.in_(_ACTIVE_JOB_STATUSES),
            )
            .order_by(Job.created_at, Job.id)
            .limit(1)
        )
        return (await session.scalar(statement)) is not None

    @staticmethod
    async def _active_ingestion_exists(
        session: AsyncSession,
        knowledge_base_id: UUID,
        generation_id: UUID,
    ) -> bool:
        statement = (
            select(Job.id)
            .where(
                Job.knowledge_base_id == knowledge_base_id,
                Job.operation == "ingest_document",
                Job.index_generation_id == generation_id,
                Job.status.in_(_ACTIVE_JOB_STATUSES),
            )
            .order_by(Job.created_at, Job.id)
            .limit(1)
        )
        return (await session.scalar(statement)) is not None

    async def _probe_collection(self, spec: CollectionSpec) -> None:
        try:
            if not await self._qdrant.collection_exists(spec.name):
                return
            if await self._qdrant.describe_collection(spec.name) != spec:
                raise _collection_conflict()
            if await self._qdrant.count_points(spec.name) != 0:
                raise _collection_not_empty()
        except BusinessError:
            raise
        except QdrantTransientError:
            raise _qdrant_unavailable() from None
        except QdrantConfigurationError:
            raise _collection_conflict() from None


class _RepairTargetChanged(RuntimeError):
    pass


async def _noop_repair_checkpoint(_name: str) -> None:
    return None


class GenerationRepairHooks:
    """Deterministic crash and race seams for repair recovery tests."""

    def __init__(self, checkpoint: Callable[[str], Awaitable[None]] = _noop_repair_checkpoint):
        if not callable(checkpoint):
            raise ValueError("generation repair hook is invalid")
        self._checkpoint = checkpoint

    async def reached(self, name: str) -> None:
        await self._checkpoint(name)


class GenerationRepairPipeline:
    """Recreate and exactly validate one active generation under a Job lease."""

    def __init__(
        self,
        *,
        session_factory: SessionFactory,
        object_store: PipelineObjectStore,
        embedding_gateway: PipelineEmbeddingGateway,
        qdrant: QdrantClient,
        provider_usage_sink: PipelineProviderUsageSink,
        max_manifest_bytes: int = MAX_CHUNK_MANIFEST_BYTES,
        hooks: GenerationRepairHooks | None = None,
    ) -> None:
        if (
            not callable(session_factory)
            or not callable(getattr(object_store, "read_stream", None))
            or not callable(getattr(embedding_gateway, "embed", None))
            or not callable(getattr(qdrant, "ensure_collection", None))
            or not callable(getattr(provider_usage_sink, "record", None))
            or type(max_manifest_bytes) is not int
            or not 1 <= max_manifest_bytes <= 5 * 1024 * 1024 * 1024
        ):
            raise ValueError("generation repair pipeline dependencies are invalid")
        self._session_factory = session_factory
        self._object_store = object_store
        self._embedding_gateway = embedding_gateway
        self._qdrant = qdrant
        self._provider_usage_sink = provider_usage_sink
        self._max_manifest_bytes = max_manifest_bytes
        self._hooks = GenerationRepairHooks() if hooks is None else hooks
        self._state = GenerationRepairService(session_factory=session_factory, qdrant=qdrant)

    async def _load_snapshot(self, generation_id: UUID) -> _RepairSnapshot:
        async with self._session_factory() as session:
            return await self._state._read_snapshot(session, generation_id)

    @staticmethod
    def _validate_lease(context: JobExecutionContext) -> UUID:
        lease = context.lease
        if (
            lease.operation != "rebuild_generation"
            or lease.target_type != "index_generation"
            or lease.index_generation_id != lease.target_id
            or lease.target_revision is None
            or type(lease.mutation_id) is not UUID
            or lease.stage not in {"indexing", "validating", "complete"}
        ):
            raise PermanentJobError(
                "GENERATION_REPAIR_STAGE_CONFLICT",
                "Generation repair stage conflict",
            )
        return lease.target_id

    async def _expected_snapshot(
        self,
        context: JobExecutionContext,
    ) -> _RepairSnapshot:
        generation_id = self._validate_lease(context)
        snapshot = await self._load_snapshot(generation_id)
        if snapshot.anchor_mutation_id != context.lease.mutation_id:
            raise PermanentJobError(
                "GENERATION_REPAIR_CONFIGURATION_CONFLICT",
                "Generation repair configuration changed",
            )
        if snapshot.mutation_revision != context.lease.target_revision:
            await self._refresh_target(context, snapshot.knowledge_base_id, generation_id)
            raise RetryableJobError(
                "GENERATION_REPAIR_TARGET_CHANGED",
                "Generation repair target changed",
            )
        if context.lease.progress_total != snapshot.point_total:
            raise PermanentJobError(
                "GENERATION_REPAIR_STAGE_CONFLICT",
                "Generation repair stage conflict",
            )
        return snapshot

    async def _locked_snapshot(
        self,
        session: AsyncSession,
        knowledge_base_id: UUID,
        generation_id: UUID,
        *,
        lock_job: JobLease | None = None,
    ) -> tuple[_RepairSnapshot, Job | None]:
        knowledge_base = cast(
            KnowledgeBase | None,
            await session.scalar(
                select(KnowledgeBase).where(KnowledgeBase.id == knowledge_base_id).with_for_update()
            ),
        )
        generation = cast(
            KnowledgeBaseIndexGeneration | None,
            await session.scalar(
                select(KnowledgeBaseIndexGeneration)
                .where(KnowledgeBaseIndexGeneration.id == generation_id)
                .with_for_update()
            ),
        )
        if knowledge_base is None or generation is None:
            raise PermanentJobError(
                "GENERATION_REPAIR_NOT_ACTIVE",
                "Generation repair target is not active",
            )
        try:
            snapshot = await self._state._snapshot_from_rows(
                session,
                knowledge_base,
                generation,
                lock_configuration=True,
            )
        except BusinessError as error:
            raise PermanentJobError(error.code, error.message) from None
        job: Job | None = None
        if lock_job is not None:
            job = cast(
                Job | None,
                await session.scalar(
                    select(Job)
                    .where(
                        Job.id == lock_job.id,
                        Job.operation == lock_job.operation,
                        Job.target_type == lock_job.target_type,
                        Job.target_id == lock_job.target_id,
                        Job.target_revision.is_not_distinct_from(lock_job.target_revision),
                        Job.index_generation_id.is_not_distinct_from(lock_job.index_generation_id),
                        Job.mutation_id.is_not_distinct_from(lock_job.mutation_id),
                        Job.stage.is_not_distinct_from(lock_job.stage),
                        Job.status == "running",
                        Job.lease_owner == lock_job.lease_owner,
                        Job.lease_epoch == lock_job.lease_epoch,
                        Job.lease_expires_at.is_not(None),
                        Job.lease_expires_at > func.clock_timestamp(),
                        Job.cancel_requested_at.is_(None),
                    )
                    .with_for_update()
                ),
            )
            if job is None:
                raise LostLeaseError
        return snapshot, job

    async def _refresh_target(
        self,
        context: JobExecutionContext,
        knowledge_base_id: UUID,
        generation_id: UUID,
    ) -> _RepairSnapshot:
        async def refresh(session: AsyncSession) -> RebuildTargetRefresh:
            snapshot, _job = await self._locked_snapshot(
                session,
                knowledge_base_id,
                generation_id,
            )
            return RebuildTargetRefresh(
                target_revision=snapshot.mutation_revision,
                progress_total=snapshot.point_total,
                fingerprint=snapshot.fingerprint,
            )

        await self._hooks.reached("target.before_reset")
        refreshed = await context.reset_rebuild_target(refresh)
        snapshot = await self._load_snapshot(generation_id)
        if (
            snapshot.mutation_revision != refreshed.target_revision
            or snapshot.point_total != refreshed.progress_total
            or snapshot.fingerprint != refreshed.fingerprint
        ):
            raise RetryableJobError(
                "GENERATION_REPAIR_TARGET_CHANGED",
                "Generation repair target changed",
            )
        await self._hooks.reached("target.after_reset")
        return snapshot

    @staticmethod
    def _stage_input(snapshot: _RepairSnapshot, target: _RepairTarget) -> EmbedIndexStageInput:
        return EmbedIndexStageInput(
            knowledge_base_id=snapshot.knowledge_base_id,
            generation_id=snapshot.generation_id,
            document_id=target.document_id,
            version_id=target.version_id,
            actor_api_key_id=None,
            model_profile_id=snapshot.embedding_profile_id,
            provider_config_id=snapshot.embedding.operational.provider_config_id,
            manifest_object_key=target.manifest_object_key,
            manifest_checksum_sha256=target.manifest_checksum_sha256,
            source_checksum_sha256=target.source_checksum_sha256,
            parsed_checksum_sha256=target.parsed_checksum_sha256,
            parser_name=target.parser_name,
            parser_version=target.parser_version,
            parser_config=target.parser_config,
            chunker_name=target.chunker_name,
            chunker_version=target.chunker_version,
            chunker_config=target.chunker_config,
            chunk_config_hash=target.chunk_config_hash,
            chunk_count=target.chunk_count,
            version_created_at=target.version_created_at,
            version_status="ready",
            index_state_status="validated",
            index_state_embedding_config_hash=snapshot.embedding.semantic_hash,
            next_chunk_index=target.chunk_count,
            qdrant_collection_name=snapshot.collection_spec.name,
            embedding_config_hash=snapshot.embedding.semantic_hash,
            embedding_snapshot_canonical=canonical_json_bytes(snapshot.embedding.snapshot),
            filter_snapshot=snapshot.filter_snapshot,
            filter_snapshot_canonical=canonical_json_bytes(snapshot.filter_snapshot),
            applied_filter_schema_revision=snapshot.applied_filter_schema_revision,
            document_metadata=target.document_metadata,
            document_metadata_canonical=canonical_json_bytes(target.document_metadata),
            gateway_snapshot=snapshot.embedding.gateway_snapshot,
            operational=snapshot.embedding.operational,
        )

    async def _verified_batches(
        self,
        expected: EmbedIndexStageInput,
        *,
        next_chunk_index: int,
    ) -> AsyncIterator[tuple[Chunk, ...]]:
        stream = self._object_store.read_stream(
            expected.manifest_object_key,
            expected_checksum=expected.manifest_checksum_sha256,
            max_bytes=self._max_manifest_bytes,
        )
        batches = iter_verified_manifest_batches(
            stream,
            expected=manifest_expectation(expected),
            next_chunk_index=next_chunk_index,
            batch_size=expected.operational.batch_size,
        )
        try:
            async for batch in batches:
                yield batch
        finally:
            await close_async_iterators_without_masking_primary(batches, stream)

    async def _assert_control_plane_current(
        self,
        session: AsyncSession,
        expected: _RepairSnapshot,
    ) -> None:
        knowledge_base = await session.get(KnowledgeBase, expected.knowledge_base_id)
        generation = await session.get(
            KnowledgeBaseIndexGeneration,
            expected.generation_id,
        )
        if (
            knowledge_base is None
            or generation is None
            or generation.knowledge_base_id != expected.knowledge_base_id
            or generation.status != "active"
            or knowledge_base.active_index_generation_id != expected.generation_id
            or knowledge_base.mutation_revision != expected.mutation_revision
            or generation.embedding_profile_id != expected.embedding_profile_id
            or generation.qdrant_collection_name != expected.collection_spec.name
            or generation.applied_filter_schema_revision != expected.applied_filter_schema_revision
        ):
            raise _RepairTargetChanged
        try:
            embedding = await self._state._embedding_configuration(
                session,
                generation,
                lock_configuration=False,
            )
            anchor_mutation_id = await self._state._configuration_anchor(
                session,
                expected.knowledge_base_id,
                generation,
            )
            filter_snapshot = build_filter_snapshot(
                cast(dict[str, object], generation.filter_schema_snapshot)
            )
            spec = CollectionSpec(
                collection_name(expected.knowledge_base_id, expected.generation_id),
                embedding.gateway_snapshot.dimension,
                cast(str, generation.distance),
                payload_indexes_for_filter_snapshot(filter_snapshot),
            )
        except (
            GenerationConfigurationError,
            KeyError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ):
            raise _RepairTargetChanged from None
        if (
            anchor_mutation_id != expected.anchor_mutation_id
            or spec != expected.collection_spec
            or embedding.snapshot != expected.embedding.snapshot
            or embedding.semantic_hash != expected.embedding.semantic_hash
            or filter_snapshot != expected.filter_snapshot
        ):
            raise _RepairTargetChanged

    async def _reset_after_drift(
        self,
        context: JobExecutionContext,
        expected: _RepairSnapshot,
    ) -> None:
        await self._refresh_target(
            context,
            expected.knowledge_base_id,
            expected.generation_id,
        )
        raise RetryableJobError(
            "GENERATION_REPAIR_TARGET_CHANGED",
            "Generation repair target changed",
        )

    async def _prepare_collection(self, snapshot: _RepairSnapshot) -> None:
        await self._qdrant.ensure_collection(snapshot.collection_spec.vector_only())
        await self._qdrant.ensure_payload_indexes(
            snapshot.collection_spec.name,
            snapshot.collection_spec.payload_indexes,
        )
        await self._qdrant.verify_collection(snapshot.collection_spec)

    async def _advance_to_validating(
        self,
        context: JobExecutionContext,
        snapshot: _RepairSnapshot,
    ) -> None:
        async def commit(session: AsyncSession) -> None:
            await self._assert_control_plane_current(session, snapshot)

        try:
            await context.advance_stage(
                commit,
                stage="validating",
                resume_stage="validating",
                progress_current=snapshot.point_total,
                progress_total=snapshot.point_total,
            )
        except _RepairTargetChanged:
            await self._reset_after_drift(context, snapshot)

    async def _indexing_stage(self, context: JobExecutionContext) -> None:
        snapshot = await self._expected_snapshot(context)
        await self._prepare_collection(snapshot)
        progress = context.lease.progress_current
        if not 0 <= progress <= snapshot.point_total:
            raise PermanentJobError(
                "GENERATION_REPAIR_STAGE_CONFLICT",
                "Generation repair stage conflict",
            )
        if progress == snapshot.point_total:
            await self._advance_to_validating(context, snapshot)
            return

        global_offset = 0
        for target in snapshot.targets:
            target_end = global_offset + target.chunk_count
            if progress >= target_end:
                global_offset = target_end
                continue
            expected = self._stage_input(snapshot, target)
            approved_metadata = approved_filter_metadata(expected)
            local_start = max(0, progress - global_offset)
            batches = self._verified_batches(expected, next_chunk_index=local_start)
            try:
                async for chunk_batch in batches:
                    batch_start = global_offset + chunk_batch[0].chunk_index
                    usage_context = ProviderUsageContext(
                        request_id=f"{context.lease.id}:repair:{batch_start}",
                        actor_api_key_id=None,
                        provider_config_id=expected.provider_config_id,
                        model_profile_id=expected.model_profile_id,
                    )
                    result = await self._embedding_gateway.embed(
                        snapshot=expected.gateway_snapshot,
                        operational=expected.operational,
                        inputs=tuple(chunk.text for chunk in chunk_batch),
                        attempt_observer=provider_usage_observer(
                            self._provider_usage_sink,
                            usage_context,
                        ),
                    )
                    if len(result.vectors) != len(chunk_batch):
                        raise ValueError("embedding result is invalid")
                    points = tuple(
                        qdrant_point_for(expected, chunk, vector, approved_metadata)
                        for chunk, vector in zip(chunk_batch, result.vectors, strict=True)
                    )
                    await self._qdrant.upsert_points(snapshot.collection_spec.name, points)
                    await self._hooks.reached("indexing.after_upsert")
                    next_progress = global_offset + chunk_batch[-1].chunk_index + 1

                    async def commit(session: AsyncSession) -> None:
                        await self._assert_control_plane_current(session, snapshot)

                    try:
                        if next_progress == snapshot.point_total:
                            await context.advance_stage(
                                commit,
                                stage="validating",
                                resume_stage="validating",
                                progress_current=next_progress,
                                progress_total=snapshot.point_total,
                            )
                            return
                        await context.commit_stage_checkpoint(
                            commit,
                            progress_current=next_progress,
                            progress_total=snapshot.point_total,
                        )
                    except _RepairTargetChanged:
                        await self._reset_after_drift(context, snapshot)
                    progress = next_progress
                    await self._hooks.reached("indexing.after_checkpoint")
            finally:
                await close_async_iterators_without_masking_primary(batches)
            global_offset = target_end
        raise PermanentJobError(
            "GENERATION_REPAIR_STAGE_CONFLICT",
            "Generation repair stage conflict",
        )

    async def _validate_qdrant(self, snapshot: _RepairSnapshot) -> None:
        await self._qdrant.verify_collection(snapshot.collection_spec)
        if await self._qdrant.count_points(snapshot.collection_spec.name) != snapshot.point_total:
            raise PermanentJobError(
                "GENERATION_REPAIR_VALIDATION_FAILED",
                "Generation repair validation failed",
            )
        for target in snapshot.targets:
            if (
                await self._qdrant.count_version_points(
                    snapshot.collection_spec.name,
                    target.version_id,
                )
                != target.chunk_count
            ):
                raise PermanentJobError(
                    "GENERATION_REPAIR_VALIDATION_FAILED",
                    "Generation repair validation failed",
                )
            expected = self._stage_input(snapshot, target)
            approved_metadata = approved_filter_metadata(expected)
            batches = self._verified_batches(expected, next_chunk_index=0)
            try:
                async for chunk_batch in batches:
                    expected_ids = tuple(
                        point_id(
                            target.version_id,
                            chunk.chunk_index,
                            chunk.chunk_hash,
                        )
                        for chunk in chunk_batch
                    )
                    try:
                        actual = await self._qdrant.retrieve_version_points(
                            snapshot.collection_spec.name,
                            target.version_id,
                            expected_ids,
                        )
                    except QdrantConfigurationError:
                        raise PermanentJobError(
                            "GENERATION_REPAIR_VALIDATION_FAILED",
                            "Generation repair validation failed",
                        ) from None
                    for chunk, inspected in zip(chunk_batch, actual, strict=True):
                        if (
                            inspected.id
                            != point_id(
                                target.version_id,
                                chunk.chunk_index,
                                chunk.chunk_hash,
                            )
                            or inspected.vector_dimension
                            != snapshot.embedding.gateway_snapshot.dimension
                            or inspected.payload_digest_sha256
                            != canonical_sha256(
                                qdrant_point_payload(
                                    expected,
                                    chunk,
                                    approved_metadata,
                                )
                            )
                        ):
                            raise PermanentJobError(
                                "GENERATION_REPAIR_VALIDATION_FAILED",
                                "Generation repair validation failed",
                            )
            finally:
                await close_async_iterators_without_masking_primary(batches)

    async def _validating_stage(self, context: JobExecutionContext) -> None:
        snapshot = await self._expected_snapshot(context)
        await self._validate_qdrant(snapshot)
        await self._hooks.reached("validating.after_qdrant")

        async def commit(session: AsyncSession) -> None:
            await self._assert_control_plane_current(session, snapshot)

        try:
            await context.advance_stage(
                commit,
                stage="complete",
                resume_stage="complete",
                progress_current=snapshot.point_total,
                progress_total=snapshot.point_total,
            )
        except _RepairTargetChanged:
            await self._reset_after_drift(context, snapshot)

    async def _finalize_success(
        self,
        context: JobExecutionContext,
        expected: _RepairSnapshot,
    ) -> None:
        lease = context.lease

        async def commit(session: AsyncSession) -> None:
            current, job = await self._locked_snapshot(
                session,
                expected.knowledge_base_id,
                expected.generation_id,
                lock_job=lease,
            )
            if (
                job is None
                or current.mutation_revision != lease.target_revision
                or current.fingerprint != expected.fingerprint
                or current.point_total != lease.progress_total
            ):
                raise _RepairTargetChanged
            finished_at = cast(datetime, await session.scalar(select(func.clock_timestamp())))
            job.status = "succeeded"
            job.retryable = False
            job.next_retry_at = None
            job.error_code = None
            job.error_message_sanitized = None
            job.lease_owner = None
            job.lease_expires_at = None
            job.worker_heartbeat_at = None
            job.finished_at = finished_at
            await session.flush()

        try:
            await context.finalize_domain(commit)
        except _RepairTargetChanged:
            await self._reset_after_drift(context, expected)

    async def _complete_stage(self, context: JobExecutionContext) -> None:
        snapshot = await self._expected_snapshot(context)
        await self._validate_qdrant(snapshot)
        await self._hooks.reached("complete.after_qdrant")
        await self._finalize_success(context, snapshot)

    async def _dispatch_stage(self, context: JobExecutionContext) -> None:
        self._validate_lease(context)
        try:
            if context.lease.stage == "indexing":
                await self._indexing_stage(context)
            elif context.lease.stage == "validating":
                await self._validating_stage(context)
            elif context.lease.stage == "complete":
                await self._complete_stage(context)
            else:
                raise PermanentJobError(
                    "GENERATION_REPAIR_STAGE_CONFLICT",
                    "Generation repair stage conflict",
                )
        except BusinessError as error:
            error_type = RetryableJobError if error.retryable else PermanentJobError
            raise error_type(error.code, error.message) from None
        except ObjectStoreError as error:
            error_type = RetryableJobError if error.retryable else PermanentJobError
            raise error_type(error.code, str(error)) from None
        except EmbeddingGatewayError as error:
            error_type = RetryableJobError if error.retryable else PermanentJobError
            raise error_type(error.code, str(error)) from None
        except QdrantTransientError:
            raise RetryableJobError("QDRANT_UNAVAILABLE", "Qdrant unavailable") from None
        except QdrantConfigurationError:
            raise PermanentJobError(
                "GENERATION_REPAIR_COLLECTION_CONFLICT",
                "Generation repair collection conflict",
            ) from None
        except (TypeError, ValueError, UnicodeError):
            raise PermanentJobError(
                "GENERATION_REPAIR_MANIFEST_INVALID",
                "Generation repair manifest is invalid",
            ) from None

    async def handle(self, context: JobExecutionContext) -> JobHandlerOutcome:
        await self._dispatch_stage(context)
        if context.finalized:
            return JobHandlerOutcome.COMPLETE
        return JobHandlerOutcome.CONTINUE


__all__ = [
    "GenerationRepairHooks",
    "GenerationRepairPipeline",
    "GenerationRepairReservation",
    "GenerationRepairService",
]
