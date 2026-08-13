"""PostgreSQL-authoritative leasing and fenced Job state transitions."""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Protocol, cast
from uuid import UUID, uuid4

from sqlalchemy import and_, case, func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.elements import ColumnElement

from rag_service.api.errors import BusinessError
from rag_service.auth.policies import AgentPrincipal, Capability
from rag_service.db.models.auth import ApiKey, ApiKeyKnowledgeBaseScope
from rag_service.db.models.documents import Document, DocumentIndexState, DocumentVersion, Job
from rag_service.db.models.knowledge_bases import (
    KnowledgeBase,
    KnowledgeBaseIndexGeneration,
    KnowledgeBaseMutation,
)
from rag_service.ingestion.chunkers import RecursiveTextChunker
from rag_service.ingestion.parsers import parser_for_extension

_CLAIM_SCAN_LIMIT = 32
_ATTEMPTS_EXHAUSTED_CODE = "JOB_ATTEMPTS_EXHAUSTED"
_ATTEMPTS_EXHAUSTED_MESSAGE = "Job attempts exhausted"


class LostLeaseError(RuntimeError):
    """Raised when a stale Worker attempts a fenced Job write."""

    def __init__(self) -> None:
        super().__init__("Job lease was lost")


@dataclass(frozen=True, slots=True)
class JobLease:
    id: UUID
    operation: str
    target_type: str
    target_id: UUID
    target_revision: int | None
    index_generation_id: UUID | None
    stage: str | None
    resume_stage: str | None
    progress_current: int
    progress_total: int | None
    attempt_count: int
    max_attempts: int
    lease_owner: str
    lease_epoch: int
    lease_expires_at: datetime
    cancel_requested_at: datetime | None
    mutation_id: UUID | None = None
    recovered: bool = False


@dataclass(frozen=True, slots=True)
class ExhaustedJob:
    """A read-only snapshot that must be terminalized in a fresh transaction."""

    id: UUID
    knowledge_base_id: UUID | None
    operation: str
    target_type: str
    target_id: UUID
    target_revision: int | None
    index_generation_id: UUID | None
    stage: str | None
    status: str
    attempt_count: int
    max_attempts: int
    next_retry_at: datetime | None
    lease_owner: str | None
    lease_epoch: int
    lease_expires_at: datetime | None
    cancel_requested_at: datetime | None
    mutation_id: UUID | None = None


@dataclass(frozen=True, slots=True)
class RebuildTargetRefresh:
    target_revision: int
    progress_total: int
    fingerprint: str

    def __post_init__(self) -> None:
        if (
            type(self.target_revision) is not int
            or self.target_revision < 0
            or type(self.progress_total) is not int
            or self.progress_total < 0
            or type(self.fingerprint) is not str
            or re.fullmatch(r"[0-9a-f]{64}", self.fingerprint) is None
        ):
            raise ValueError("rebuild target refresh is invalid")


@dataclass(frozen=True, slots=True)
class JobStatusRecord:
    id: UUID
    knowledge_base_id: UUID | None
    operation: str
    status: str
    stage: str | None
    progress_current: int
    progress_total: int | None
    attempt_count: int
    max_attempts: int
    retryable: bool
    error_code: str | None
    error_message: str | None
    parent_job_id: UUID | None
    root_job_id: UUID | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None


@dataclass(frozen=True, slots=True)
class ManualRetrySnapshot:
    id: UUID
    knowledge_base_id: UUID | None
    actor_api_key_id: UUID | None
    operation: str
    target_type: str
    target_id: UUID
    target_revision: int | None
    index_generation_id: UUID | None
    mutation_id: UUID | None
    parent_job_id: UUID | None
    root_job_id: UUID | None
    stage: str | None
    status: str
    progress_current: int
    progress_total: int | None
    attempt_count: int
    max_attempts: int
    retryable: bool
    resume_stage: str | None
    document_id: UUID | None


@dataclass(frozen=True, slots=True)
class ManualRetryReservation:
    job: JobStatusRecord
    created: bool


class JobRepository(Protocol):
    async def claim_next(
        self,
        *,
        lease_owner: str,
        lease_duration: timedelta,
        job_id: UUID | None = None,
    ) -> JobLease | ExhaustedJob | None: ...

    async def finalize_exhausted(
        self,
        candidate: ExhaustedJob,
        action: Callable[[AsyncSession], Awaitable[None]] | None = None,
    ) -> None: ...

    async def heartbeat(self, lease: JobLease, lease_duration: timedelta) -> JobLease: ...

    async def cancellation_requested(self, lease: JobLease) -> bool: ...

    async def release_claim(self, lease: JobLease) -> None: ...

    async def checkpoint(
        self,
        lease: JobLease,
        *,
        stage: str | None,
        resume_stage: str | None,
        progress_current: int,
        progress_total: int | None,
    ) -> None: ...

    async def reset_rebuild_target(
        self,
        lease: JobLease,
        action: Callable[[AsyncSession], Awaitable[RebuildTargetRefresh]],
    ) -> tuple[RebuildTargetRefresh, JobLease]: ...

    async def prepare_activation[Result](
        self,
        lease: JobLease,
        action: Callable[[AsyncSession], Awaitable[Result]],
    ) -> Result: ...

    async def finalize_domain[Result](
        self,
        lease: JobLease,
        action: Callable[[AsyncSession], Awaitable[Result]],
        *,
        terminal_status_observer: Callable[[str], None] | None = None,
    ) -> Result: ...

    async def commit_stage_facts[Result](
        self,
        lease: JobLease,
        action: Callable[[AsyncSession], Awaitable[Result]],
    ) -> Result: ...

    async def commit_stage_checkpoint[Result](
        self,
        lease: JobLease,
        action: Callable[[AsyncSession], Awaitable[Result]],
        *,
        progress_current: int,
        progress_total: int | None,
    ) -> tuple[Result, JobLease]: ...

    async def advance_stage[Result](
        self,
        lease: JobLease,
        action: Callable[[AsyncSession], Awaitable[Result]],
        *,
        stage: str,
        resume_stage: str,
        progress_current: int,
        progress_total: int | None,
    ) -> tuple[Result, JobLease]: ...

    async def mark_succeeded(self, lease: JobLease) -> str: ...

    async def mark_cancelled(self, lease: JobLease) -> None: ...

    async def record_failure(
        self,
        lease: JobLease,
        *,
        retryable: bool,
        error_code: str,
        error_message: str,
        retry_delay: timedelta,
    ) -> str: ...


class SqlAlchemyJobRepository:
    """Job persistence operations; callers own transactions."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    @staticmethod
    def lease_from_job(job: Job, *, recovered: bool = False) -> JobLease:
        if job.status != "running" or job.lease_owner is None or job.lease_expires_at is None:
            raise ValueError("Job does not hold a lease")
        return JobLease(
            id=job.id,
            operation=job.operation,
            target_type=job.target_type,
            target_id=job.target_id,
            target_revision=job.target_revision,
            index_generation_id=job.index_generation_id,
            stage=job.stage,
            resume_stage=job.resume_stage,
            progress_current=job.progress_current,
            progress_total=job.progress_total,
            attempt_count=job.attempt_count,
            max_attempts=job.max_attempts,
            lease_owner=job.lease_owner,
            lease_epoch=job.lease_epoch,
            lease_expires_at=job.lease_expires_at,
            cancel_requested_at=job.cancel_requested_at,
            mutation_id=job.mutation_id,
            recovered=recovered,
        )

    @staticmethod
    def exhausted_from_job(job: Job) -> ExhaustedJob:
        if job.attempt_count != job.max_attempts:
            raise ValueError("Job attempts are not exhausted")
        return ExhaustedJob(
            id=job.id,
            knowledge_base_id=job.knowledge_base_id,
            operation=job.operation,
            target_type=job.target_type,
            target_id=job.target_id,
            target_revision=job.target_revision,
            index_generation_id=job.index_generation_id,
            stage=job.stage,
            status=job.status,
            attempt_count=job.attempt_count,
            max_attempts=job.max_attempts,
            next_retry_at=job.next_retry_at,
            lease_owner=job.lease_owner,
            lease_epoch=job.lease_epoch,
            lease_expires_at=job.lease_expires_at,
            cancel_requested_at=job.cancel_requested_at,
            mutation_id=job.mutation_id,
        )

    @staticmethod
    def _db_now() -> ColumnElement[datetime]:
        return cast(ColumnElement[datetime], func.clock_timestamp())

    @classmethod
    def _fence(cls, lease: JobLease) -> tuple[ColumnElement[bool], ...]:
        return (
            Job.id == lease.id,
            Job.status == "running",
            Job.lease_owner == lease.lease_owner,
            Job.lease_epoch == lease.lease_epoch,
            Job.mutation_id.is_not_distinct_from(lease.mutation_id),
            Job.lease_expires_at.is_not(None),
            Job.lease_expires_at > cls._db_now(),
        )

    @classmethod
    def _stage_fence(cls, lease: JobLease) -> tuple[ColumnElement[bool], ...]:
        return (*cls._fence(lease), Job.stage.is_not_distinct_from(lease.stage))

    @classmethod
    def exhausted_fence(cls, candidate: ExhaustedJob) -> tuple[ColumnElement[bool], ...]:
        db_now = cls._db_now()
        eligibility: ColumnElement[bool]
        if candidate.status == "queued":
            eligibility = Job.status == "queued"
        elif candidate.status == "retry_wait":
            eligibility = (
                (Job.status == "retry_wait")
                & Job.next_retry_at.is_not(None)
                & (Job.next_retry_at <= db_now)
            )
        elif candidate.status == "running":
            eligibility = (
                (Job.status == "running")
                & Job.lease_expires_at.is_not(None)
                & (Job.lease_expires_at <= db_now)
            )
        else:
            raise ValueError("Exhausted Job state is invalid")
        return (
            Job.id == candidate.id,
            Job.knowledge_base_id.is_not_distinct_from(candidate.knowledge_base_id),
            Job.operation == candidate.operation,
            Job.target_type == candidate.target_type,
            Job.target_id == candidate.target_id,
            Job.target_revision.is_not_distinct_from(candidate.target_revision),
            Job.index_generation_id.is_not_distinct_from(candidate.index_generation_id),
            Job.mutation_id.is_not_distinct_from(candidate.mutation_id),
            Job.stage.is_not_distinct_from(candidate.stage),
            Job.attempt_count == candidate.attempt_count,
            Job.max_attempts == candidate.max_attempts,
            Job.attempt_count == Job.max_attempts,
            Job.next_retry_at.is_not_distinct_from(candidate.next_retry_at),
            Job.lease_owner.is_not_distinct_from(candidate.lease_owner),
            Job.lease_epoch == candidate.lease_epoch,
            Job.lease_expires_at.is_not_distinct_from(candidate.lease_expires_at),
            Job.cancel_requested_at.is_not_distinct_from(candidate.cancel_requested_at),
            eligibility,
        )

    @staticmethod
    def _validate_progress(progress_current: int, progress_total: int | None) -> None:
        if progress_current < 0 or (
            progress_total is not None and (progress_total < 0 or progress_current > progress_total)
        ):
            raise ValueError("job progress is invalid")

    @staticmethod
    def _validate_lease_input(lease_owner: str, lease_duration: timedelta) -> None:
        if type(lease_owner) is not str or not lease_owner or len(lease_owner) > 255:
            raise ValueError("lease owner is invalid")
        if lease_duration <= timedelta(0):
            raise ValueError("lease duration must be positive")

    @staticmethod
    def _status_from_job(job: Job) -> JobStatusRecord:
        return JobStatusRecord(
            id=job.id,
            knowledge_base_id=job.knowledge_base_id,
            operation=job.operation,
            status=job.status,
            stage=job.stage,
            progress_current=job.progress_current,
            progress_total=job.progress_total,
            attempt_count=job.attempt_count,
            max_attempts=job.max_attempts,
            retryable=job.retryable,
            error_code=job.error_code,
            error_message=job.error_message_sanitized,
            parent_job_id=job.parent_job_id,
            root_job_id=job.root_job_id,
            created_at=job.created_at,
            started_at=job.started_at,
            finished_at=job.finished_at,
        )

    async def get_status(self, job_id: UUID) -> JobStatusRecord | None:
        """Load only fields approved for the public Job response plus its scope owner."""

        row = (
            await self._session.execute(
                select(
                    Job.id,
                    Job.knowledge_base_id,
                    Job.operation,
                    Job.status,
                    Job.stage,
                    Job.progress_current,
                    Job.progress_total,
                    Job.attempt_count,
                    Job.max_attempts,
                    Job.retryable,
                    Job.error_code,
                    Job.error_message_sanitized,
                    Job.parent_job_id,
                    Job.root_job_id,
                    Job.created_at,
                    Job.started_at,
                    Job.finished_at,
                ).where(Job.id == job_id)
            )
        ).one_or_none()
        if row is None:
            return None
        return JobStatusRecord(
            id=row.id,
            knowledge_base_id=row.knowledge_base_id,
            operation=row.operation,
            status=row.status,
            stage=row.stage,
            progress_current=row.progress_current,
            progress_total=row.progress_total,
            attempt_count=row.attempt_count,
            max_attempts=row.max_attempts,
            retryable=row.retryable,
            error_code=row.error_code,
            error_message=row.error_message_sanitized,
            parent_job_id=row.parent_job_id,
            root_job_id=row.root_job_id,
            created_at=row.created_at,
            started_at=row.started_at,
            finished_at=row.finished_at,
        )

    async def get_manual_retry_snapshot(self, job_id: UUID) -> ManualRetrySnapshot | None:
        row = (
            await self._session.execute(
                select(Job, DocumentVersion.document_id)
                .outerjoin(
                    DocumentVersion,
                    (Job.operation == "ingest_document")
                    & (Job.target_type == "document_version")
                    & (DocumentVersion.id == Job.target_id),
                )
                .where(Job.id == job_id)
            )
        ).one_or_none()
        if row is None:
            return None
        job, document_id = row
        return ManualRetrySnapshot(
            id=job.id,
            knowledge_base_id=job.knowledge_base_id,
            actor_api_key_id=job.actor_api_key_id,
            operation=job.operation,
            target_type=job.target_type,
            target_id=job.target_id,
            target_revision=job.target_revision,
            index_generation_id=job.index_generation_id,
            mutation_id=job.mutation_id,
            parent_job_id=job.parent_job_id,
            root_job_id=job.root_job_id,
            stage=job.stage,
            status=job.status,
            progress_current=job.progress_current,
            progress_total=job.progress_total,
            attempt_count=job.attempt_count,
            max_attempts=job.max_attempts,
            retryable=job.retryable,
            resume_stage=job.resume_stage,
            document_id=document_id,
        )

    @staticmethod
    def manual_retry_fence(snapshot: ManualRetrySnapshot) -> tuple[ColumnElement[bool], ...]:
        """Fence every nullable target and checkpoint fact with NULL-safe equality."""

        return (
            Job.id == snapshot.id,
            Job.knowledge_base_id.is_not_distinct_from(snapshot.knowledge_base_id),
            Job.actor_api_key_id.is_not_distinct_from(snapshot.actor_api_key_id),
            Job.operation == snapshot.operation,
            Job.target_type == snapshot.target_type,
            Job.target_id == snapshot.target_id,
            Job.target_revision.is_not_distinct_from(snapshot.target_revision),
            Job.index_generation_id.is_not_distinct_from(snapshot.index_generation_id),
            Job.mutation_id.is_not_distinct_from(snapshot.mutation_id),
            Job.parent_job_id.is_not_distinct_from(snapshot.parent_job_id),
            Job.root_job_id.is_not_distinct_from(snapshot.root_job_id),
            Job.stage.is_not_distinct_from(snapshot.stage),
            Job.resume_stage.is_not_distinct_from(snapshot.resume_stage),
            Job.status == snapshot.status,
            Job.progress_current == snapshot.progress_current,
            Job.progress_total.is_not_distinct_from(snapshot.progress_total),
            Job.attempt_count == snapshot.attempt_count,
            Job.max_attempts == snapshot.max_attempts,
            Job.retryable == snapshot.retryable,
        )

    @staticmethod
    def _hidden_job() -> BusinessError:
        return BusinessError(404, "RESOURCE_NOT_FOUND", "Resource not found")

    @staticmethod
    def _not_retryable() -> BusinessError:
        return BusinessError(409, "JOB_NOT_RETRYABLE", "Job is not retryable")

    @staticmethod
    def _retry_mismatch() -> BusinessError:
        return BusinessError(
            409,
            "JOB_RETRY_STATE_MISMATCH",
            "Job retry state does not match",
        )

    @staticmethod
    def _retry_conflict() -> BusinessError:
        return BusinessError(
            409,
            "JOB_RETRY_CONFLICT",
            "Job retry conflicts with active work",
            retryable=True,
        )

    async def _lock_retry_actor(
        self,
        actor: AgentPrincipal,
        knowledge_base_id: UUID,
        operation: str,
    ) -> None:
        stored = (
            await self._session.execute(
                select(ApiKey)
                .where(
                    ApiKey.id == actor.key_id,
                    ApiKey.public_id == actor.public_id,
                    ApiKey.key_type == "agent",
                    ApiKey.status == "active",
                    ApiKey.revoked_at.is_(None),
                    or_(ApiKey.not_before.is_(None), ApiKey.not_before <= func.now()),
                    or_(ApiKey.expires_at.is_(None), ApiKey.expires_at > func.now()),
                )
                .with_for_update()
            )
        ).scalar_one_or_none()
        if stored is None:
            raise BusinessError(401, "INVALID_API_KEY", "Invalid API key")
        scope = await self._session.scalar(
            select(ApiKeyKnowledgeBaseScope).where(
                ApiKeyKnowledgeBaseScope.api_key_id == actor.key_id,
                ApiKeyKnowledgeBaseScope.knowledge_base_id == knowledge_base_id,
            )
        )
        required_capabilities = (
            (Capability.MANAGE.value, Capability.INGEST.value)
            if operation in {"ingest_document", "rebuild_generation"}
            else ()
        )
        if (
            scope is None
            or not required_capabilities
            or not any(capability in stored.capabilities for capability in required_capabilities)
        ):
            raise self._hidden_job()

    @staticmethod
    def _valid_parser_snapshot(document: Document, version: DocumentVersion) -> bool:
        if (
            document.checksum_sha256 != version.source_checksum_sha256
            or version.source_extension is None
            or type(version.parser_config) is not dict
        ):
            return False
        try:
            parser = parser_for_extension(version.source_extension)
        except ValueError:
            return False
        return (
            version.parser_name == parser.name
            and version.parser_version == parser.version
            and version.parser_config == dict(parser.config)
        )

    async def _active_manual_retry_conflict(
        self,
        source: Job,
        *,
        knowledge_base_id: UUID,
        generation_id: UUID,
    ) -> bool:
        same_ingestion_target = and_(
            Job.operation == "ingest_document",
            Job.target_type == source.target_type,
            Job.target_id == source.target_id,
            Job.target_revision.is_not_distinct_from(source.target_revision),
            Job.index_generation_id.is_not_distinct_from(source.index_generation_id),
        )
        same_generation_rebuild = and_(
            Job.operation == "rebuild_generation",
            Job.target_type == "index_generation",
            Job.target_id == generation_id,
            Job.index_generation_id == generation_id,
        )
        same_generation_ingestion = and_(
            Job.knowledge_base_id == knowledge_base_id,
            Job.operation == "ingest_document",
            Job.index_generation_id == generation_id,
        )
        if source.operation == "ingest_document":
            conflict = or_(same_ingestion_target, same_generation_rebuild)
        elif source.operation == "rebuild_generation":
            conflict = or_(same_generation_rebuild, same_generation_ingestion)
        else:
            return False
        active_job = await self._session.scalar(
            select(Job.id)
            .where(
                conflict,
                Job.status.in_(("queued", "running", "retry_wait")),
            )
            .order_by(Job.created_at, Job.id)
            .limit(1)
        )
        return active_job is not None

    @staticmethod
    def _valid_parsed_snapshot(version: DocumentVersion) -> bool:
        return (
            version.parsed_object_key is not None
            and version.parsed_object_checksum_sha256 is not None
        )

    @staticmethod
    def _valid_manifest_snapshot(
        version: DocumentVersion,
        state: DocumentIndexState,
        generation: KnowledgeBaseIndexGeneration,
    ) -> bool:
        if (
            version.chunk_manifest_object_key is None
            or version.chunk_manifest_checksum_sha256 is None
            or version.chunk_config_hash is None
            or version.chunker_name is None
            or version.chunker_version is None
            or type(version.chunker_config) is not dict
            or version.chunk_count is None
            or version.chunk_count < 1
            or generation.embedding_config_hash is None
        ):
            return False
        maximum = version.chunker_config.get("max_chunk_codepoints")
        overlap = version.chunker_config.get("target_overlap_codepoints")
        if (
            version.chunker_name != RecursiveTextChunker.name
            or version.chunker_version != RecursiveTextChunker.version
            or set(version.chunker_config) != {"max_chunk_codepoints", "target_overlap_codepoints"}
            or type(maximum) is not int
            or type(overlap) is not int
        ):
            return False
        try:
            chunker = RecursiveTextChunker(
                max_chunk_codepoints=maximum,
                target_overlap_codepoints=overlap,
            )
        except ValueError:
            return False
        return (
            version.chunk_config_hash == chunker.config_hash
            and state.expected_point_count == version.chunk_count
            and state.chunk_manifest_checksum_sha256 == version.chunk_manifest_checksum_sha256
            and 0 <= state.next_chunk_index <= version.chunk_count
            and (
                (
                    state.next_chunk_index == 0
                    and state.embedding_config_hash in {None, generation.embedding_config_hash}
                )
                or (
                    state.next_chunk_index > 0
                    and state.embedding_config_hash == generation.embedding_config_hash
                )
            )
        )

    def _reset_ingestion_retry(
        self,
        snapshot: ManualRetrySnapshot,
        *,
        document: Document,
        version: DocumentVersion,
        state: DocumentIndexState,
        generation: KnowledgeBaseIndexGeneration,
    ) -> None:
        stage_tuple = (snapshot.stage, snapshot.resume_stage)
        if not self._valid_parser_snapshot(document, version):
            raise self._retry_mismatch()

        if stage_tuple in {("parse", None), ("parse", "parse")}:
            compatible = (
                snapshot.progress_current == 0
                and snapshot.progress_total is None
                and version.parsed_object_key is None
                and version.parsed_object_checksum_sha256 is None
                and version.chunk_manifest_object_key is None
                and version.chunk_manifest_checksum_sha256 is None
                and version.chunk_config_hash is None
                and version.chunk_count is None
                and state.expected_point_count is None
                and state.actual_point_count is None
                and state.chunk_manifest_checksum_sha256 is None
                and state.embedding_config_hash is None
                and state.next_chunk_index == 0
                and state.validated_at is None
            )
            version_status, state_status = "uploaded", "queued"
        elif stage_tuple == ("chunk", "chunk"):
            compatible = (
                snapshot.progress_current == 0
                and snapshot.progress_total is None
                and self._valid_parsed_snapshot(version)
                and version.chunk_manifest_object_key is None
                and version.chunk_manifest_checksum_sha256 is None
                and version.chunk_config_hash is None
                and version.chunk_count is None
                and state.expected_point_count is None
                and state.actual_point_count is None
                and state.chunk_manifest_checksum_sha256 is None
                and state.embedding_config_hash is None
                and state.next_chunk_index == 0
                and state.validated_at is None
            )
            version_status, state_status = "chunking", "queued"
        elif stage_tuple in {
            ("embed_index", "embed_index"),
            ("validate", "validate"),
            ("activate", "activate"),
        }:
            compatible = self._valid_parsed_snapshot(version) and self._valid_manifest_snapshot(
                version, state, generation
            )
            if stage_tuple == ("embed_index", "embed_index"):
                compatible = (
                    compatible
                    and snapshot.progress_current == state.next_chunk_index
                    and snapshot.progress_total == version.chunk_count
                    and state.actual_point_count is None
                )
                version_status = "embedding" if state.next_chunk_index == 0 else "indexing"
                state_status = "embedding" if state.next_chunk_index == 0 else "indexing"
            elif stage_tuple == ("validate", "validate"):
                compatible = (
                    compatible
                    and version.chunk_count is not None
                    and state.next_chunk_index == version.chunk_count
                    and snapshot.progress_current == version.chunk_count
                    and snapshot.progress_total == version.chunk_count
                    and state.embedding_config_hash == generation.embedding_config_hash
                    and state.actual_point_count in {None, state.expected_point_count}
                    and state.validated_at is None
                )
                version_status, state_status = "indexing", "indexing"
            else:
                compatible = (
                    compatible
                    and version.chunk_count is not None
                    and state.next_chunk_index == version.chunk_count
                    and snapshot.progress_current == version.chunk_count
                    and snapshot.progress_total == version.chunk_count
                    and state.embedding_config_hash == generation.embedding_config_hash
                    and state.actual_point_count == state.expected_point_count
                    and state.validated_at is not None
                )
                version_status, state_status = "indexing", "validated"
        else:
            raise self._retry_mismatch()

        if not compatible:
            raise self._retry_mismatch()
        document.status = "processing"
        version.status = version_status
        state.status = state_status
        state.error_code = None
        state.safe_error_message = None

    async def _canonical_rebuild_point_total(
        self,
        knowledge_base: KnowledgeBase,
        generation: KnowledgeBaseIndexGeneration,
    ) -> int | None:
        rows = (
            await self._session.execute(
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
                    Document.knowledge_base_id == knowledge_base.id,
                    Document.status == "active",
                    Document.deleted_at.is_(None),
                )
                .order_by(Document.id)
            )
        ).all()
        point_total = 0
        for document, version, state in rows:
            if (
                version is None
                or state is None
                or document.current_version_id != version.id
                or version.document_id != document.id
                or version.status != "ready"
                or version.activated_at is None
                or state.status != "validated"
                or state.validated_at is None
                or version.chunk_count is None
                or version.chunk_count < 1
                or state.actual_point_count != version.chunk_count
                or state.next_chunk_index != version.chunk_count
                or type(document.metadata_) is not dict
                or not self._valid_parser_snapshot(document, version)
                or not self._valid_parsed_snapshot(version)
                or not self._valid_manifest_snapshot(version, state, generation)
            ):
                return None
            point_total += version.chunk_count
        return point_total

    @staticmethod
    def _valid_rebuild_anchor(
        knowledge_base: KnowledgeBase,
        generation: KnowledgeBaseIndexGeneration,
        mutation: KnowledgeBaseMutation | None,
        snapshot: ManualRetrySnapshot,
    ) -> bool:
        payload = None if mutation is None else mutation.payload
        return (
            snapshot.target_type == "index_generation"
            and snapshot.target_id == generation.id
            and snapshot.index_generation_id == generation.id
            and snapshot.mutation_id is not None
            and snapshot.mutation_id == (None if mutation is None else mutation.id)
            and snapshot.target_revision == knowledge_base.mutation_revision
            and snapshot.stage in {"indexing", "validating", "complete"}
            and snapshot.resume_stage == snapshot.stage
            and generation.caught_up_revision is not None
            and generation.embedding_profile_id is not None
            and generation.embedding_config_snapshot is not None
            and generation.embedding_config_hash is not None
            and mutation is not None
            and mutation.knowledge_base_id == knowledge_base.id
            and mutation.revision == generation.caught_up_revision
            and mutation.mutation_type == "index_config_changed"
            and mutation.target_type == "index_generation"
            and mutation.target_id == generation.id
            and type(payload) is dict
            and payload.get("generation_id") == str(generation.id)
            and payload.get("embedding_profile_id") == str(generation.embedding_profile_id)
            and payload.get("index_profile_hash") == generation.index_profile_hash
            and payload.get("embedding_config_hash") == generation.embedding_config_hash
            and payload.get("applied_filter_schema_revision")
            == generation.applied_filter_schema_revision
            and payload.get("validation_manifest_hash") == generation.validation_manifest_hash
        )

    async def reserve_manual_retry(
        self,
        snapshot: ManualRetrySnapshot,
        *,
        actor: AgentPrincipal,
        idempotency_key: str,
    ) -> ManualRetryReservation:
        """Atomically reserve a child Job after locking the domain in canonical order."""

        knowledge_base_id = snapshot.knowledge_base_id
        generation_id = snapshot.index_generation_id
        if knowledge_base_id is None or generation_id is None:
            raise self._hidden_job()
        await self._lock_retry_actor(actor, knowledge_base_id, snapshot.operation)

        knowledge_base = await self._session.scalar(
            select(KnowledgeBase).where(KnowledgeBase.id == knowledge_base_id).with_for_update()
        )
        generation = await self._session.scalar(
            select(KnowledgeBaseIndexGeneration)
            .where(KnowledgeBaseIndexGeneration.id == generation_id)
            .with_for_update()
        )

        document: Document | None = None
        version: DocumentVersion | None = None
        state: DocumentIndexState | None = None
        mutation: KnowledgeBaseMutation | None = None
        if snapshot.operation == "ingest_document" and snapshot.document_id is not None:
            document = await self._session.scalar(
                select(Document).where(Document.id == snapshot.document_id).with_for_update()
            )
            version = await self._session.scalar(
                select(DocumentVersion)
                .where(DocumentVersion.id == snapshot.target_id)
                .with_for_update()
            )
            state = await self._session.scalar(
                select(DocumentIndexState)
                .where(
                    DocumentIndexState.document_version_id == snapshot.target_id,
                    DocumentIndexState.index_generation_id == generation_id,
                )
                .with_for_update()
            )
        elif snapshot.operation == "rebuild_generation" and snapshot.mutation_id is not None:
            mutation = await self._session.scalar(
                select(KnowledgeBaseMutation).where(
                    KnowledgeBaseMutation.id == snapshot.mutation_id
                )
            )

        source = await self._session.scalar(
            select(Job).where(*self.manual_retry_fence(snapshot)).with_for_update()
        )
        if source is None:
            raise self._retry_conflict()

        replay = await self._session.scalar(
            select(Job)
            .where(
                Job.parent_job_id == source.id,
                Job.actor_api_key_id == actor.key_id,
                Job.idempotency_key == idempotency_key,
            )
            .order_by(Job.created_at, Job.id)
            .limit(1)
        )
        if replay is not None:
            return ManualRetryReservation(self._status_from_job(replay), created=False)

        if source.status != "failed" or source.retryable is not True:
            raise self._not_retryable()
        if (
            knowledge_base is None
            or generation is None
            or knowledge_base.status != "active"
            or knowledge_base.active_index_generation_id != generation.id
            or generation.knowledge_base_id != knowledge_base.id
            or generation.status != "active"
        ):
            raise self._retry_mismatch()

        if await self._active_manual_retry_conflict(
            source,
            knowledge_base_id=knowledge_base_id,
            generation_id=generation_id,
        ):
            raise self._retry_conflict()

        if source.operation == "ingest_document":
            if (
                document is None
                or version is None
                or state is None
                or source.target_type != "document_version"
                or document.knowledge_base_id != knowledge_base.id
                or document.current_version_id is not None
                or document.pending_version_id != version.id
                or document.status != "failed"
                or document.deleted_at is not None
                or version.document_id != document.id
                or version.base_version_id is not None
                or version.status != "failed"
                or state.status != "failed"
            ):
                raise self._retry_mismatch()
            self._reset_ingestion_retry(
                snapshot,
                document=document,
                version=version,
                state=state,
                generation=generation,
            )
        elif source.operation == "rebuild_generation":
            point_total = await self._canonical_rebuild_point_total(knowledge_base, generation)
            progress_is_compatible = (
                point_total is not None
                and snapshot.progress_total == point_total
                and (
                    0 <= snapshot.progress_current <= point_total
                    if snapshot.stage == "indexing"
                    else snapshot.progress_current == point_total
                )
            )
            if not (
                self._valid_rebuild_anchor(knowledge_base, generation, mutation, snapshot)
                and progress_is_compatible
            ):
                raise self._retry_mismatch()
        else:
            raise self._not_retryable()

        child = Job(
            id=uuid4(),
            knowledge_base_id=source.knowledge_base_id,
            actor_api_key_id=actor.key_id,
            target_type=source.target_type,
            target_id=source.target_id,
            target_revision=source.target_revision,
            index_generation_id=source.index_generation_id,
            mutation_id=source.mutation_id,
            parent_job_id=source.id,
            root_job_id=source.root_job_id or source.id,
            idempotency_key=idempotency_key,
            operation=source.operation,
            stage=source.stage,
            resume_stage=source.resume_stage,
            status="queued",
            progress_current=source.progress_current,
            progress_total=source.progress_total,
            attempt_count=0,
            max_attempts=source.max_attempts,
            retryable=True,
        )
        self._session.add(child)
        await self._session.flush()
        return ManualRetryReservation(self._status_from_job(child), created=True)

    async def claim_next(
        self,
        *,
        lease_owner: str,
        lease_duration: timedelta,
        job_id: UUID | None = None,
    ) -> JobLease | ExhaustedJob | None:
        self._validate_lease_input(lease_owner, lease_duration)
        db_now = self._db_now()
        eligibility = or_(
            Job.status == "queued",
            (Job.status == "retry_wait")
            & Job.next_retry_at.is_not(None)
            & (Job.next_retry_at <= db_now),
            (Job.status == "running")
            & Job.lease_expires_at.is_not(None)
            & (Job.lease_expires_at <= db_now),
        )
        statement = (
            select(Job)
            .where(eligibility)
            .order_by(Job.created_at, Job.id)
            .limit(_CLAIM_SCAN_LIMIT)
            .with_for_update(skip_locked=True)
        )
        if job_id is not None:
            statement = statement.where(Job.id == job_id)
        candidates = tuple((await self._session.scalars(statement)).all())
        if not candidates:
            return None

        for job in candidates:
            if job.attempt_count >= job.max_attempts:
                return self.exhausted_from_job(job)

            recovered = job.status == "running"
            current_time = cast(datetime, await self._session.scalar(select(self._db_now())))
            job.status = "running"
            job.attempt_count += 1
            job.lease_epoch += 1
            job.lease_owner = lease_owner
            job.lease_expires_at = current_time + lease_duration
            job.worker_heartbeat_at = current_time
            job.started_at = job.started_at or current_time
            job.finished_at = None
            job.next_retry_at = None
            job.error_code = None
            job.error_message_sanitized = None
            await self._session.flush()
            return self.lease_from_job(job, recovered=recovered)

        return None

    async def finalize_exhausted(
        self,
        candidate: ExhaustedJob,
        action: Callable[[AsyncSession], Awaitable[None]] | None = None,
    ) -> None:
        fence = self.exhausted_fence(candidate)
        if action is None:
            statement = (
                update(Job)
                .where(*fence)
                .values(
                    status="failed",
                    retryable=False,
                    next_retry_at=None,
                    error_code=_ATTEMPTS_EXHAUSTED_CODE,
                    error_message_sanitized=_ATTEMPTS_EXHAUSTED_MESSAGE,
                    lease_owner=None,
                    lease_expires_at=None,
                    worker_heartbeat_at=None,
                    finished_at=self._db_now(),
                )
                .returning(Job.id)
            )
            if await self._session.scalar(statement) is None:
                raise LostLeaseError
            return

        await action(self._session)
        terminal = await self._session.scalar(
            select(Job.id).where(
                Job.id == candidate.id,
                Job.lease_epoch == candidate.lease_epoch,
                Job.status == "failed",
                Job.error_code == _ATTEMPTS_EXHAUSTED_CODE,
                Job.lease_owner.is_(None),
                Job.lease_expires_at.is_(None),
            )
        )
        if terminal is None:
            raise LostLeaseError

    async def heartbeat(self, lease: JobLease, lease_duration: timedelta) -> JobLease:
        if lease_duration <= timedelta(0):
            raise ValueError("lease duration must be positive")
        db_now = self._db_now()
        statement = (
            update(Job)
            .where(*self._fence(lease))
            .values(
                worker_heartbeat_at=db_now,
                lease_expires_at=db_now + lease_duration,
            )
            .returning(Job)
        )
        row = cast(Job | None, await self._session.scalar(statement))
        if row is None:
            raise LostLeaseError
        return self.lease_from_job(row)

    async def cancellation_requested(self, lease: JobLease) -> bool:
        statement = select(Job.cancel_requested_at).where(*self._fence(lease))
        result = (await self._session.execute(statement)).one_or_none()
        if result is None:
            raise LostLeaseError
        return cast(datetime | None, result[0]) is not None

    async def release_claim(self, lease: JobLease) -> None:
        """Requeue admitted work that must not start after Worker shutdown."""

        statement = (
            update(Job)
            .where(*self._fence(lease))
            .values(
                status="queued",
                attempt_count=Job.attempt_count - 1,
                retryable=True,
                next_retry_at=None,
                error_code=None,
                error_message_sanitized=None,
                lease_owner=None,
                lease_expires_at=None,
                worker_heartbeat_at=None,
                finished_at=None,
            )
            .returning(Job.id)
        )
        if await self._session.scalar(statement) is None:
            raise LostLeaseError

    async def checkpoint(
        self,
        lease: JobLease,
        *,
        stage: str | None,
        resume_stage: str | None,
        progress_current: int,
        progress_total: int | None,
    ) -> None:
        self._validate_progress(progress_current, progress_total)
        statement = (
            update(Job)
            .where(*self._fence(lease))
            .values(
                stage=stage,
                resume_stage=resume_stage,
                progress_current=progress_current,
                progress_total=progress_total,
            )
            .returning(Job.id)
        )
        if await self._session.scalar(statement) is None:
            raise LostLeaseError

    async def reset_rebuild_target(
        self,
        lease: JobLease,
        action: Callable[[AsyncSession], Awaitable[RebuildTargetRefresh]],
    ) -> tuple[RebuildTargetRefresh, JobLease]:
        """Fence a revision refresh and restart the canonical rebuild scan."""

        if (
            lease.operation != "rebuild_generation"
            or lease.target_type != "index_generation"
            or lease.stage not in {"indexing", "validating", "complete"}
            or not callable(action)
        ):
            raise ValueError("rebuild target reset is invalid")
        refresh = await action(self._session)
        if type(refresh) is not RebuildTargetRefresh:
            raise ValueError("rebuild target reset is invalid")
        statement = (
            update(Job)
            .where(*self._stage_fence(lease))
            .values(
                target_revision=refresh.target_revision,
                stage="indexing",
                resume_stage="indexing",
                progress_current=0,
                progress_total=refresh.progress_total,
            )
            .returning(Job)
        )
        updated = cast(Job | None, await self._session.scalar(statement))
        if updated is None:
            raise LostLeaseError
        return refresh, self.lease_from_job(updated)

    async def advance_stage[Result](
        self,
        lease: JobLease,
        action: Callable[[AsyncSession], Awaitable[Result]],
        *,
        stage: str,
        resume_stage: str,
        progress_current: int,
        progress_total: int | None,
    ) -> tuple[Result, JobLease]:
        """Commit domain stage facts and the next Job checkpoint under one fence."""

        if (
            type(stage) is not str
            or not stage
            or len(stage) > 64
            or type(resume_stage) is not str
            or not resume_stage
            or len(resume_stage) > 64
        ):
            raise ValueError("job stage is invalid")
        self._validate_progress(progress_current, progress_total)
        if await self._session.scalar(select(Job.id).where(*self._stage_fence(lease))) is None:
            raise LostLeaseError

        result = await action(self._session)
        statement = (
            update(Job)
            .where(*self._stage_fence(lease))
            .values(
                stage=stage,
                resume_stage=resume_stage,
                progress_current=progress_current,
                progress_total=progress_total,
            )
            .returning(Job)
        )
        updated = cast(Job | None, await self._session.scalar(statement))
        if updated is None:
            raise LostLeaseError
        return result, self.lease_from_job(updated)

    async def commit_stage_facts[Result](
        self,
        lease: JobLease,
        action: Callable[[AsyncSession], Awaitable[Result]],
    ) -> Result:
        """Commit current-stage domain facts without advancing the Job stage."""

        if await self._session.scalar(select(Job.id).where(*self._stage_fence(lease))) is None:
            raise LostLeaseError
        result = await action(self._session)
        statement = (
            update(Job)
            .where(*self._stage_fence(lease))
            .values(lease_epoch=Job.lease_epoch)
            .returning(Job.id)
        )
        if await self._session.scalar(statement) is None:
            raise LostLeaseError
        return result

    async def commit_stage_checkpoint[Result](
        self,
        lease: JobLease,
        action: Callable[[AsyncSession], Awaitable[Result]],
        *,
        progress_current: int,
        progress_total: int | None,
    ) -> tuple[Result, JobLease]:
        """Atomically commit current-stage facts and its exclusive progress."""

        self._validate_progress(progress_current, progress_total)
        if await self._session.scalar(select(Job.id).where(*self._stage_fence(lease))) is None:
            raise LostLeaseError
        result = await action(self._session)
        statement = (
            update(Job)
            .where(*self._stage_fence(lease))
            .values(
                progress_current=progress_current,
                progress_total=progress_total,
            )
            .returning(Job)
        )
        updated = cast(Job | None, await self._session.scalar(statement))
        if updated is None:
            raise LostLeaseError
        return result, self.lease_from_job(updated)

    async def prepare_activation[Result](
        self,
        lease: JobLease,
        action: Callable[[AsyncSession], Awaitable[Result]],
    ) -> Result:
        """Fence domain activation before and after its caller-owned transaction work."""

        async def fence_activation() -> None:
            statement = (
                update(Job)
                .where(*self._fence(lease))
                .values(lease_epoch=Job.lease_epoch)
                .returning(Job.id)
            )
            if await self._session.scalar(statement) is None:
                raise LostLeaseError

        await fence_activation()
        result = await action(self._session)
        await fence_activation()
        return result

    async def finalize_domain[Result](
        self,
        lease: JobLease,
        action: Callable[[AsyncSession], Awaitable[Result]],
        *,
        terminal_status_observer: Callable[[str], None] | None = None,
    ) -> Result:
        """Run a domain-owned terminal transaction without locking Job first."""

        result = await action(self._session)
        terminal_status = await self._session.scalar(
            select(Job.status).where(
                Job.id == lease.id,
                Job.lease_epoch == lease.lease_epoch,
                Job.status.in_(("succeeded", "failed", "cancelled")),
                Job.lease_owner.is_(None),
                Job.lease_expires_at.is_(None),
            )
        )
        if terminal_status is None:
            raise LostLeaseError
        if terminal_status_observer is not None:
            terminal_status_observer(terminal_status)
        return result

    async def mark_succeeded(self, lease: JobLease) -> str:
        db_now = self._db_now()
        statement = (
            update(Job)
            .where(*self._fence(lease))
            .values(
                status=case(
                    (Job.cancel_requested_at.is_not(None), "cancelled"),
                    else_="succeeded",
                ),
                retryable=False,
                next_retry_at=None,
                error_code=None,
                error_message_sanitized=None,
                lease_owner=None,
                lease_expires_at=None,
                worker_heartbeat_at=None,
                finished_at=db_now,
            )
            .returning(Job.status)
        )
        status = cast(str | None, await self._session.scalar(statement))
        if status is None:
            raise LostLeaseError
        return status

    async def mark_cancelled(self, lease: JobLease) -> None:
        db_now = self._db_now()
        statement = (
            update(Job)
            .where(*self._fence(lease))
            .values(
                status="cancelled",
                retryable=False,
                next_retry_at=None,
                lease_owner=None,
                lease_expires_at=None,
                worker_heartbeat_at=None,
                finished_at=db_now,
            )
            .returning(Job.id)
        )
        if await self._session.scalar(statement) is None:
            raise LostLeaseError

    async def record_failure(
        self,
        lease: JobLease,
        *,
        retryable: bool,
        error_code: str,
        error_message: str,
        retry_delay: timedelta,
    ) -> str:
        if (
            type(error_code) is not str
            or not error_code
            or len(error_code) > 64
            or type(error_message) is not str
            or not error_message
            or len(error_message) > 500
            or retry_delay < timedelta(0)
        ):
            raise ValueError("job failure is invalid")
        db_now = self._db_now()
        cancellation_requested = Job.cancel_requested_at.is_not(None)
        status_value: object
        retryable_value: object
        next_retry_value: object
        error_code_value: object
        error_message_value: object
        finished_value: object
        failure_status: object
        failure_next_retry: object
        failure_finished: object
        if retryable:
            can_retry = Job.attempt_count < Job.max_attempts
            failure_status = case((can_retry, "retry_wait"), else_="failed")
            failure_next_retry = case(
                (can_retry, db_now + retry_delay),
                else_=None,
            )
            failure_finished = case((can_retry, None), else_=db_now)
        else:
            failure_status = "failed"
            failure_next_retry = None
            failure_finished = db_now

        status_value = case(
            (cancellation_requested, "cancelled"),
            else_=failure_status,
        )
        retryable_value = case(
            (cancellation_requested, False),
            else_=retryable,
        )
        if retryable:
            next_retry_value = case(
                (cancellation_requested, None),
                else_=failure_next_retry,
            )
            finished_value = case(
                (cancellation_requested, db_now),
                else_=failure_finished,
            )
        else:
            next_retry_value = None
            finished_value = db_now
        error_code_value = case(
            (cancellation_requested, None),
            else_=error_code,
        )
        error_message_value = case(
            (cancellation_requested, None),
            else_=error_message,
        )
        statement = (
            update(Job)
            .where(*self._fence(lease))
            .values(
                status=status_value,
                retryable=retryable_value,
                next_retry_at=next_retry_value,
                error_code=error_code_value,
                error_message_sanitized=error_message_value,
                lease_owner=None,
                lease_expires_at=None,
                worker_heartbeat_at=None,
                finished_at=finished_value,
            )
            .returning(Job.status)
        )
        result = cast(str | None, await self._session.scalar(statement))
        if result is None:
            raise LostLeaseError
        return result


__all__ = [
    "ExhaustedJob",
    "JobLease",
    "JobRepository",
    "JobStatusRecord",
    "LostLeaseError",
    "ManualRetryReservation",
    "ManualRetrySnapshot",
    "SqlAlchemyJobRepository",
]
