import asyncio
import base64
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from functools import partial
from typing import cast
from uuid import UUID, uuid4

import httpx
import pytest
from sqlalchemy import event, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from rag_service.api.errors import BusinessError
from rag_service.auth.dependencies import require_agent_principal
from rag_service.auth.policies import AgentPrincipal, Capability
from rag_service.config import Settings
from rag_service.db.models.auth import ApiKey, ApiKeyKnowledgeBaseScope
from rag_service.db.models.documents import Document, DocumentIndexState, DocumentVersion, Job
from rag_service.db.models.knowledge_bases import (
    KnowledgeBase,
    KnowledgeBaseIndexGeneration,
    KnowledgeBaseMutation,
)
from rag_service.db.models.providers import ModelProfile, ProviderConfig
from rag_service.db.session import Database
from rag_service.indexing.identities import canonical_sha256
from rag_service.infrastructure.minio_store import OrphanCandidate, OrphanPage
from rag_service.ingestion.pipeline import (
    IngestionPipeline,
    ObjectReconciliationCursor,
    PipelineObjectStore,
)
from rag_service.ingestion.repositories import SqlAlchemyIngestionPipelineRepository
from rag_service.jobs.repositories import (
    ExhaustedJob,
    JobLease,
    LostLeaseError,
    ManualRetryReservation,
    ManualRetrySnapshot,
    SqlAlchemyJobRepository,
)
from rag_service.jobs.runner import ExponentialBackoff, JobExecutionContext, JobRunner
from rag_service.main import create_app

pytestmark = pytest.mark.integration


def _job(
    *,
    status: str = "queued",
    attempt_count: int = 0,
    max_attempts: int = 3,
    target_id: UUID | None = None,
    lease_owner: str | None = None,
    lease_epoch: int = 0,
    lease_expires_at: datetime | None = None,
    next_retry_at: datetime | None = None,
) -> Job:
    heartbeat = None
    if status == "running":
        heartbeat = datetime.now(UTC) - timedelta(minutes=1)
    return Job(
        id=uuid4(),
        knowledge_base_id=None,
        target_type="document_version",
        target_id=target_id or uuid4(),
        target_revision=1,
        index_generation_id=None,
        mutation_id=None,
        parent_job_id=None,
        root_job_id=None,
        idempotency_key=None,
        operation="ingest_document",
        stage="parse",
        status=status,
        progress_current=0,
        progress_total=None,
        attempt_count=attempt_count,
        max_attempts=max_attempts,
        next_retry_at=next_retry_at,
        worker_heartbeat_at=heartbeat,
        cancel_requested_at=None,
        error_code=None,
        error_message_sanitized=None,
        started_at=None,
        finished_at=None,
        lease_owner=lease_owner,
        lease_epoch=lease_epoch,
        lease_expires_at=lease_expires_at,
        retryable=True,
        resume_stage=None,
    )


async def _insert(database: Database, job: Job) -> None:
    async with database.sessions() as session, session.begin():
        session.add(job)


async def _load(database: Database, job_id: UUID) -> Job:
    async with database.sessions() as session:
        row = await session.get(Job, job_id)
        assert row is not None
        return row


@dataclass(frozen=True, slots=True)
class _ManualRetrySeed:
    knowledge_base_id: UUID
    generation_id: UUID
    document_id: UUID
    version_id: UUID
    mutation_id: UUID
    job_id: UUID
    actor: AgentPrincipal
    source_object_key: str
    parsed_object_key: str
    manifest_object_key: str
    manifest_checksum: str
    embedding_config_hash: str


async def _seed_manual_retry_ingestion(
    database: Database,
    *,
    stage: str = "embed_index",
    resume_stage: str | None = "embed_index",
    retryable: bool = True,
    next_chunk_index: int | None = None,
    state_embedding_hash: str | None = None,
) -> _ManualRetrySeed:
    now = datetime.now(UTC)
    knowledge_base_id = uuid4()
    generation_id = uuid4()
    document_id = uuid4()
    version_id = uuid4()
    mutation_id = uuid4()
    job_id = uuid4()
    actor_id = uuid4()
    checksum = "a" * 64
    parsed_checksum = "b" * 64
    manifest_checksum = "c" * 64
    embedding_hash = "d" * 64
    chunker_config = {"max_chunk_codepoints": 512, "target_overlap_codepoints": 64}
    chunk_hash = canonical_sha256(
        {"config": chunker_config, "name": "recursive_text_v1", "version": "1"}
    )
    source_object_key = f"private/source/{uuid4()}"
    parsed_object_key = f"private/parsed/{uuid4()}"
    manifest_object_key = f"private/manifest/{uuid4()}"
    public_id = base64.urlsafe_b64encode(actor_id.bytes).decode("ascii").rstrip("=")
    if stage not in {"parse", "chunk", "embed_index", "validate", "activate"}:
        raise ValueError("unsupported manual retry test stage")
    has_parsed = stage != "parse"
    has_manifest = stage in {"embed_index", "validate", "activate"}
    if next_chunk_index is None:
        next_chunk_index = 3 if stage in {"validate", "activate"} else int(has_manifest)
    expected_point_count = 3 if has_manifest else None
    actual_point_count = 3 if stage == "activate" else None
    state_embedding_config_hash = embedding_hash if has_manifest and next_chunk_index > 0 else None
    if state_embedding_hash is not None:
        state_embedding_config_hash = state_embedding_hash
    job_progress_current = next_chunk_index if has_manifest else 0
    job_progress_total = 3 if has_manifest else None

    async with database.sessions() as session, session.begin():
        actor_row = ApiKey(
            id=actor_id,
            public_id=public_id,
            secret_digest=b"x" * 32,
            key_type="agent",
            name="manual-retry-agent",
            status="active",
            capabilities=[Capability.INGEST.value],
            raw_file_read=False,
            requests_per_minute=60,
            max_concurrency=4,
        )
        knowledge_base = KnowledgeBase(
            id=knowledge_base_id,
            name="Manual retry KB",
            status="active",
            mutation_revision=1,
        )
        session.add_all([actor_row, knowledge_base])
        await session.flush()
        generation = KnowledgeBaseIndexGeneration(
            id=generation_id,
            knowledge_base_id=knowledge_base_id,
            embedding_profile_id=None,
            sparse_profile_id=None,
            index_profile_hash=embedding_hash,
            qdrant_collection_name=f"rag_kb_{knowledge_base_id.hex}_g_{generation_id.hex}",
            status="active",
            rebuild_snapshot_at=now,
            caught_up_revision=0,
            validated_revision=0,
            validation_manifest_hash="e" * 64,
            expected_point_count=0,
            actual_point_count=0,
            validated_at=now,
            activated_at=now,
            distance="cosine",
            embedding_config_snapshot={},
            filter_schema_snapshot={"fields": []},
            applied_filter_schema_revision=0,
            embedding_config_hash=embedding_hash,
        )
        session.add_all(
            [
                generation,
                ApiKeyKnowledgeBaseScope(
                    api_key_id=actor_id,
                    knowledge_base_id=knowledge_base_id,
                ),
            ]
        )
        await session.flush()
        knowledge_base.active_index_generation_id = generation_id

        document = Document(
            id=document_id,
            knowledge_base_id=knowledge_base_id,
            display_name="retry.txt",
            mime_type="text/plain",
            checksum_sha256=checksum,
            current_version_id=None,
            pending_version_id=version_id,
            status="failed",
        )
        version = DocumentVersion(
            id=version_id,
            document_id=document_id,
            version_number=1,
            source_object_key=source_object_key,
            parsed_object_key=parsed_object_key if has_parsed else None,
            source_checksum_sha256=checksum,
            parsed_object_checksum_sha256=parsed_checksum if has_parsed else None,
            declared_mime_type="text/plain",
            detected_mime_type="text/plain",
            source_extension=".txt",
            base_version_id=None,
            parser_name="plain_text_v1",
            parser_version="1",
            parser_config={},
            chunker_name="recursive_text_v1" if has_manifest else None,
            chunker_version="1" if has_manifest else None,
            chunker_config=chunker_config if has_manifest else {},
            chunk_count=3 if has_manifest else None,
            status="failed",
            chunk_manifest_object_key=manifest_object_key if has_manifest else None,
            chunk_manifest_checksum_sha256=manifest_checksum if has_manifest else None,
            chunk_config_hash=chunk_hash if has_manifest else None,
        )
        mutation = KnowledgeBaseMutation(
            id=mutation_id,
            knowledge_base_id=knowledge_base_id,
            revision=1,
            mutation_type="document_version_created",
            target_type="document_version",
            target_id=version_id,
            payload={"document_id": str(document_id)},
        )
        session.add_all([document, version, mutation])
        await session.flush()
        state = DocumentIndexState(
            document_version_id=version_id,
            index_generation_id=generation_id,
            status="failed",
            expected_point_count=expected_point_count,
            actual_point_count=actual_point_count,
            error_code="UPSTREAM_UNAVAILABLE",
            validated_at=now if stage == "activate" else None,
            chunk_manifest_checksum_sha256=manifest_checksum if has_manifest else None,
            embedding_config_hash=state_embedding_config_hash,
            next_chunk_index=next_chunk_index,
            safe_error_message="Embedding service temporarily unavailable",
        )
        job = Job(
            id=job_id,
            knowledge_base_id=knowledge_base_id,
            actor_api_key_id=actor_id,
            target_type="document_version",
            target_id=version_id,
            target_revision=1,
            index_generation_id=generation_id,
            mutation_id=mutation_id,
            operation="ingest_document",
            stage=stage,
            resume_stage=resume_stage,
            status="failed",
            progress_current=job_progress_current,
            progress_total=job_progress_total,
            attempt_count=5,
            max_attempts=5,
            retryable=retryable,
            error_code="UPSTREAM_UNAVAILABLE",
            error_message_sanitized="Embedding service temporarily unavailable",
            finished_at=now,
        )
        session.add_all([state, job])

    return _ManualRetrySeed(
        knowledge_base_id=knowledge_base_id,
        generation_id=generation_id,
        document_id=document_id,
        version_id=version_id,
        mutation_id=mutation_id,
        job_id=job_id,
        actor=AgentPrincipal(
            key_id=actor_id,
            public_id=public_id,
            capabilities=frozenset({Capability.INGEST}),
            knowledge_base_ids=frozenset({knowledge_base_id}),
            query_profile_ids=frozenset(),
            default_query_profile_id=None,
            raw_file_read=False,
            requests_per_minute=60,
            max_concurrency=4,
        ),
        source_object_key=source_object_key,
        parsed_object_key=parsed_object_key,
        manifest_object_key=manifest_object_key,
        manifest_checksum=manifest_checksum,
        embedding_config_hash=embedding_hash,
    )


async def _manual_retry_snapshot(database: Database, job_id: UUID) -> ManualRetrySnapshot:
    async with database.sessions() as session:
        snapshot = await SqlAlchemyJobRepository(session).get_manual_retry_snapshot(job_id)
    assert snapshot is not None
    return snapshot


async def _seed_manual_retry_rebuild(
    database: Database,
) -> tuple[UUID, UUID, UUID, AgentPrincipal]:
    now = datetime.now(UTC)
    actor_id = uuid4()
    knowledge_base_id = uuid4()
    generation_id = uuid4()
    mutation_id = uuid4()
    job_id = uuid4()
    document_id = uuid4()
    version_id = uuid4()
    provider_id = uuid4()
    profile_id = uuid4()
    embedding_hash = "7" * 64
    validation_hash = "8" * 64
    source_checksum = "9" * 64
    parsed_checksum = "a" * 64
    manifest_checksum = "b" * 64
    chunker_config = {"max_chunk_codepoints": 512, "target_overlap_codepoints": 64}
    chunk_hash = canonical_sha256(
        {"config": chunker_config, "name": "recursive_text_v1", "version": "1"}
    )
    public_id = base64.urlsafe_b64encode(actor_id.bytes).decode("ascii").rstrip("=")
    provider = ProviderConfig(
        id=provider_id,
        name=f"rebuild-provider-{provider_id.hex}",
        provider_type="openai_compatible",
        base_url="https://api.example.test/v1",
        secret_ref="env:REBUILD_TEST_KEY",
        timeout_seconds=Decimal("30"),
        max_concurrency=4,
        requests_per_minute=60,
        enabled=True,
    )
    profile = ModelProfile(
        id=profile_id,
        name=f"rebuild-profile-{profile_id.hex}",
        capability="embedding",
        provider_config_id=provider_id,
        model_name="embedding-test",
        dimension=3,
        max_input_tokens=8192,
        batch_size=8,
        timeout_seconds=Decimal("30"),
        enabled=True,
    )
    async with database.sessions() as session, session.begin():
        actor_row = ApiKey(
            id=actor_id,
            public_id=public_id,
            secret_digest=b"r" * 32,
            key_type="agent",
            name="rebuild-retry-agent",
            status="active",
            capabilities=[Capability.MANAGE.value],
            raw_file_read=False,
            requests_per_minute=60,
            max_concurrency=4,
        )
        knowledge_base = KnowledgeBase(
            id=knowledge_base_id,
            name="Rebuild retry KB",
            status="active",
            mutation_revision=5,
        )
        session.add_all([actor_row, knowledge_base, provider])
        await session.flush()
        session.add(profile)
        await session.flush()
        generation = KnowledgeBaseIndexGeneration(
            id=generation_id,
            knowledge_base_id=knowledge_base_id,
            embedding_profile_id=profile_id,
            sparse_profile_id=None,
            index_profile_hash=embedding_hash,
            qdrant_collection_name=f"rag_kb_{knowledge_base_id.hex}_g_{generation_id.hex}",
            status="active",
            rebuild_snapshot_at=now,
            caught_up_revision=1,
            validated_revision=1,
            validation_manifest_hash=validation_hash,
            expected_point_count=10,
            actual_point_count=10,
            validated_at=now,
            activated_at=now,
            distance="cosine",
            embedding_config_snapshot={"model_name": "embedding-test"},
            filter_schema_snapshot={"fields": []},
            applied_filter_schema_revision=0,
            embedding_config_hash=embedding_hash,
        )
        anchor = KnowledgeBaseMutation(
            id=mutation_id,
            knowledge_base_id=knowledge_base_id,
            revision=1,
            mutation_type="index_config_changed",
            target_type="index_generation",
            target_id=generation_id,
            payload={
                "generation_id": str(generation_id),
                "embedding_profile_id": str(profile_id),
                "index_profile_hash": embedding_hash,
                "embedding_config_hash": embedding_hash,
                "applied_filter_schema_revision": 0,
                "validation_manifest_hash": validation_hash,
            },
        )
        session.add_all(
            [
                generation,
                anchor,
                ApiKeyKnowledgeBaseScope(
                    api_key_id=actor_id,
                    knowledge_base_id=knowledge_base_id,
                ),
            ]
        )
        await session.flush()
        knowledge_base.active_index_generation_id = generation_id
        document = Document(
            id=document_id,
            knowledge_base_id=knowledge_base_id,
            display_name="repair-source.txt",
            mime_type="text/plain",
            checksum_sha256=source_checksum,
            current_version_id=version_id,
            pending_version_id=None,
            status="active",
        )
        version = DocumentVersion(
            id=version_id,
            document_id=document_id,
            version_number=1,
            source_object_key=f"private/source/{uuid4()}",
            parsed_object_key=f"private/parsed/{uuid4()}",
            source_checksum_sha256=source_checksum,
            parsed_object_checksum_sha256=parsed_checksum,
            declared_mime_type="text/plain",
            detected_mime_type="text/plain",
            source_extension=".txt",
            parser_name="plain_text_v1",
            parser_version="1",
            parser_config={},
            chunker_name="recursive_text_v1",
            chunker_version="1",
            chunker_config=chunker_config,
            chunk_count=10,
            status="ready",
            activated_at=now,
            chunk_manifest_object_key=f"private/manifest/{uuid4()}",
            chunk_manifest_checksum_sha256=manifest_checksum,
            chunk_config_hash=chunk_hash,
        )
        session.add_all([document, version])
        await session.flush()
        session.add_all(
            [
                DocumentIndexState(
                    document_version_id=version_id,
                    index_generation_id=generation_id,
                    status="validated",
                    expected_point_count=10,
                    actual_point_count=10,
                    validated_at=now,
                    chunk_manifest_checksum_sha256=manifest_checksum,
                    embedding_config_hash=embedding_hash,
                    next_chunk_index=10,
                ),
                Job(
                    id=job_id,
                    knowledge_base_id=knowledge_base_id,
                    actor_api_key_id=actor_id,
                    target_type="index_generation",
                    target_id=generation_id,
                    target_revision=5,
                    index_generation_id=generation_id,
                    mutation_id=mutation_id,
                    operation="rebuild_generation",
                    stage="indexing",
                    resume_stage="indexing",
                    status="failed",
                    progress_current=4,
                    progress_total=10,
                    attempt_count=5,
                    max_attempts=5,
                    retryable=True,
                    error_code="QDRANT_UNAVAILABLE",
                    error_message_sanitized="Vector store temporarily unavailable",
                    finished_at=now,
                ),
            ]
        )

    return (
        job_id,
        mutation_id,
        generation_id,
        AgentPrincipal(
            key_id=actor_id,
            public_id=public_id,
            capabilities=frozenset({Capability.MANAGE}),
            knowledge_base_ids=frozenset({knowledge_base_id}),
            query_profile_ids=frozenset(),
            default_query_profile_id=None,
            raw_file_read=False,
            requests_per_minute=60,
            max_concurrency=4,
        ),
    )


@pytest.mark.asyncio
async def test_claim_uses_skip_locked_db_time_and_increments_attempt_and_epoch(
    migrated_database: Database,
) -> None:
    job = _job()
    unlocked_job = _job()
    await _insert(migrated_database, job)
    await _insert(migrated_database, unlocked_job)

    async with migrated_database.sessions() as locking_session, locking_session.begin():
        await locking_session.execute(select(Job).where(Job.id == job.id).with_for_update())
        async with migrated_database.sessions() as skipped_session, skipped_session.begin():
            skipped = await SqlAlchemyJobRepository(skipped_session).claim_next(
                lease_owner="worker-b",
                lease_duration=timedelta(seconds=30),
            )
            assert isinstance(skipped, JobLease)
            assert skipped.id == unlocked_job.id

    before = datetime.now(UTC)
    async with migrated_database.sessions() as session, session.begin():
        lease = await SqlAlchemyJobRepository(session).claim_next(
            lease_owner="worker-a",
            lease_duration=timedelta(seconds=30),
        )

    assert isinstance(lease, JobLease)
    assert lease.attempt_count == 1
    assert lease.lease_epoch == 1
    assert lease.lease_owner == "worker-a"
    assert lease.lease_expires_at > before + timedelta(seconds=20)


@pytest.mark.asyncio
async def test_due_retry_and_expired_running_jobs_are_recovered_but_future_retry_is_not(
    migrated_database: Database,
) -> None:
    now = datetime.now(UTC)
    future = _job(status="retry_wait", next_retry_at=now + timedelta(hours=1))
    due = _job(status="retry_wait", attempt_count=1, next_retry_at=now - timedelta(seconds=1))
    await _insert(migrated_database, future)
    await _insert(migrated_database, due)

    async with migrated_database.sessions() as session, session.begin():
        due_lease = await SqlAlchemyJobRepository(session).claim_next(
            lease_owner="worker-a",
            lease_duration=timedelta(seconds=30),
        )
    assert isinstance(due_lease, JobLease)
    assert due_lease.id == due.id
    assert due_lease.attempt_count == 2

    expired = _job(
        status="running",
        attempt_count=1,
        lease_owner="worker-old",
        lease_epoch=4,
        lease_expires_at=now - timedelta(seconds=1),
    )
    await _insert(migrated_database, expired)
    async with migrated_database.sessions() as session, session.begin():
        expired_lease = await SqlAlchemyJobRepository(session).claim_next(
            lease_owner="worker-b",
            lease_duration=timedelta(seconds=30),
        )
    assert isinstance(expired_lease, JobLease)
    assert expired_lease.id == expired.id
    assert expired_lease.attempt_count == 2
    assert expired_lease.lease_epoch == 5

    async with migrated_database.sessions() as session, session.begin():
        assert (
            await SqlAlchemyJobRepository(session).claim_next(
                lease_owner="worker-c",
                lease_duration=timedelta(seconds=30),
            )
            is None
        )


@pytest.mark.asyncio
async def test_exhausted_job_fails_without_exceeding_max_attempts(
    migrated_database: Database,
) -> None:
    job = _job(attempt_count=3, max_attempts=3)
    await _insert(migrated_database, job)

    async with migrated_database.sessions() as session, session.begin():
        candidate = await SqlAlchemyJobRepository(session).claim_next(
            lease_owner="worker-a",
            lease_duration=timedelta(seconds=30),
        )

    assert isinstance(candidate, ExhaustedJob)
    before = await _load(migrated_database, job.id)
    assert before.status == "queued"
    async with migrated_database.sessions() as session, session.begin():
        await SqlAlchemyJobRepository(session).finalize_exhausted(candidate)

    stored = await _load(migrated_database, job.id)
    assert stored.status == "failed"
    assert stored.attempt_count == stored.max_attempts == 3
    assert stored.lease_owner is None
    assert stored.lease_expires_at is None
    assert stored.finished_at is not None


@pytest.mark.asyncio
@pytest.mark.acceptance
async def test_worker_a_is_fenced_after_worker_b_reclaims_the_expired_lease(
    migrated_database: Database,
) -> None:
    job = _job(
        status="running",
        attempt_count=1,
        lease_owner="worker-a",
        lease_epoch=1,
        lease_expires_at=datetime.now(UTC) - timedelta(seconds=1),
    )
    await _insert(migrated_database, job)
    stale_lease = SqlAlchemyJobRepository.lease_from_job(job)

    async with migrated_database.sessions() as session, session.begin():
        current_lease = await SqlAlchemyJobRepository(session).claim_next(
            lease_owner="worker-b",
            lease_duration=timedelta(seconds=30),
        )
    assert isinstance(current_lease, JobLease)
    assert current_lease.lease_epoch == 2

    async def stale_write(name: str) -> None:
        async with migrated_database.sessions() as session, session.begin():
            repository = SqlAlchemyJobRepository(session)
            with pytest.raises(LostLeaseError):
                if name == "heartbeat":
                    await repository.heartbeat(stale_lease, timedelta(seconds=30))
                elif name == "checkpoint":
                    await repository.checkpoint(
                        stale_lease,
                        stage="chunk",
                        resume_stage="chunk",
                        progress_current=1,
                        progress_total=2,
                    )
                elif name == "failure":
                    await repository.record_failure(
                        stale_lease,
                        retryable=True,
                        error_code="TEMPORARY",
                        error_message="Temporary failure",
                        retry_delay=timedelta(seconds=5),
                    )
                elif name == "success":
                    await repository.mark_succeeded(stale_lease)
                elif name == "advance":

                    async def advance(_session: AsyncSession) -> None:
                        return None

                    await repository.advance_stage(
                        stale_lease,
                        advance,
                        stage="chunk",
                        resume_stage="chunk",
                        progress_current=0,
                        progress_total=None,
                    )
                elif name == "stage_facts":

                    async def commit_facts(_session: AsyncSession) -> None:
                        return None

                    await repository.commit_stage_facts(stale_lease, commit_facts)
                elif name == "stage_checkpoint":

                    async def checkpoint_facts(_session: AsyncSession) -> None:
                        return None

                    await repository.commit_stage_checkpoint(
                        stale_lease,
                        checkpoint_facts,
                        progress_current=1,
                        progress_total=2,
                    )
                else:

                    async def activate(_session: AsyncSession) -> None:
                        return None

                    await repository.prepare_activation(stale_lease, activate)

    for action in (
        "heartbeat",
        "checkpoint",
        "stage_facts",
        "stage_checkpoint",
        "advance",
        "failure",
        "success",
        "activation",
    ):
        await stale_write(action)

    stored = await _load(migrated_database, job.id)
    assert stored.status == "running"
    assert stored.lease_owner == "worker-b"
    assert stored.lease_epoch == 2


@pytest.mark.asyncio
async def test_advance_stage_rolls_back_domain_action_when_current_stage_changes(
    migrated_database: Database,
) -> None:
    job = _job()
    await _insert(migrated_database, job)
    async with migrated_database.sessions() as session, session.begin():
        lease = await SqlAlchemyJobRepository(session).claim_next(
            lease_owner="worker-a",
            lease_duration=timedelta(seconds=30),
        )
    assert isinstance(lease, JobLease)

    async def drift_stage(session: AsyncSession) -> None:
        stored = await session.get(Job, job.id)
        assert stored is not None
        stored.stage = "concurrent-stage"
        stored.error_message_sanitized = "must roll back"

    with pytest.raises(LostLeaseError):
        async with migrated_database.sessions() as session, session.begin():
            await SqlAlchemyJobRepository(session).advance_stage(
                lease,
                drift_stage,
                stage="chunk",
                resume_stage="chunk",
                progress_current=0,
                progress_total=None,
            )

    stored = await _load(migrated_database, job.id)
    assert stored.stage == "parse"
    assert stored.error_message_sanitized is None
    assert stored.status == "running"


@pytest.mark.asyncio
async def test_commit_stage_facts_rolls_back_domain_action_when_current_stage_changes(
    migrated_database: Database,
) -> None:
    job = _job()
    await _insert(migrated_database, job)
    async with migrated_database.sessions() as session, session.begin():
        lease = await SqlAlchemyJobRepository(session).claim_next(
            lease_owner="worker-a",
            lease_duration=timedelta(seconds=30),
        )
    assert isinstance(lease, JobLease)

    async def drift_stage(session: AsyncSession) -> None:
        stored = await session.get(Job, job.id)
        assert stored is not None
        stored.stage = "concurrent-stage"
        stored.error_message_sanitized = "must roll back"

    with pytest.raises(LostLeaseError):
        async with migrated_database.sessions() as session, session.begin():
            await SqlAlchemyJobRepository(session).commit_stage_facts(lease, drift_stage)

    stored = await _load(migrated_database, job.id)
    assert stored.stage == "parse"
    assert stored.error_message_sanitized is None
    assert stored.status == "running"


@pytest.mark.asyncio
async def test_cancellation_racing_success_is_resolved_atomically_to_cancelled(
    migrated_database: Database,
) -> None:
    job = _job()
    await _insert(migrated_database, job)
    async with migrated_database.sessions() as session, session.begin():
        repository = SqlAlchemyJobRepository(session)
        lease = await repository.claim_next(
            lease_owner="worker-a",
            lease_duration=timedelta(seconds=30),
        )
    assert isinstance(lease, JobLease)

    async with migrated_database.sessions() as session, session.begin():
        stored = await session.get(Job, job.id)
        assert stored is not None
        stored.cancel_requested_at = datetime.now(UTC)

    async with migrated_database.sessions() as session, session.begin():
        status = await SqlAlchemyJobRepository(session).mark_succeeded(lease)

    assert status == "cancelled"
    stored = await _load(migrated_database, job.id)
    assert stored.status == "cancelled"
    assert stored.lease_owner is None
    assert stored.lease_expires_at is None


@pytest.mark.asyncio
async def test_prepare_activation_runs_domain_callback_in_the_fenced_transaction(
    migrated_database: Database,
) -> None:
    job = _job()
    await _insert(migrated_database, job)
    async with migrated_database.sessions() as session, session.begin():
        repository = SqlAlchemyJobRepository(session)
        lease = await repository.claim_next(
            lease_owner="worker-a",
            lease_duration=timedelta(seconds=30),
        )
    assert isinstance(lease, JobLease)

    async with migrated_database.sessions() as session, session.begin():
        repository = SqlAlchemyJobRepository(session)

        async def activate(active_session: AsyncSession) -> str:
            stored = await active_session.get(Job, job.id)
            assert stored is not None
            stored.stage = "activate"
            return "activated"

        result = await repository.prepare_activation(lease, activate)

    assert result == "activated"
    assert (await _load(migrated_database, job.id)).stage == "activate"


@pytest.mark.asyncio
async def test_prepare_activation_rolls_back_callback_when_lease_expires_before_commit(
    migrated_database: Database,
) -> None:
    job = _job()
    await _insert(migrated_database, job)
    async with migrated_database.sessions() as session, session.begin():
        repository = SqlAlchemyJobRepository(session)
        lease = await repository.claim_next(
            lease_owner="worker-a",
            lease_duration=timedelta(seconds=1),
        )
    assert isinstance(lease, JobLease)
    activation_called = False

    with pytest.raises(LostLeaseError):
        async with migrated_database.sessions() as session, session.begin():
            repository = SqlAlchemyJobRepository(session)

            async def activate(active_session: AsyncSession) -> None:
                nonlocal activation_called
                activation_called = True
                stored = await active_session.get(Job, job.id)
                assert stored is not None
                stored.stage = "activate"
                await active_session.flush()
                await asyncio.sleep(1.2)

            await repository.prepare_activation(lease, activate)

    stored = await _load(migrated_database, job.id)
    assert activation_called is True
    assert stored.stage == "parse"
    assert stored.status == "running"
    assert stored.lease_owner == "worker-a"
    assert stored.lease_epoch == lease.lease_epoch


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "action",
    [
        "heartbeat",
        "checkpoint",
        "stage_facts",
        "advance",
        "failure",
        "success",
        "activation",
    ],
)
async def test_same_owner_and_epoch_are_fenced_after_lease_expiry(
    migrated_database: Database,
    *,
    action: str,
) -> None:
    job = _job()
    await _insert(migrated_database, job)
    async with migrated_database.sessions() as session, session.begin():
        repository = SqlAlchemyJobRepository(session)
        lease = await repository.claim_next(
            lease_owner="worker-a",
            lease_duration=timedelta(milliseconds=100),
        )
    assert isinstance(lease, JobLease)
    await asyncio.sleep(0.2)
    activation_called = False

    async with migrated_database.sessions() as session, session.begin():
        repository = SqlAlchemyJobRepository(session)
        with pytest.raises(LostLeaseError):
            if action == "heartbeat":
                await repository.heartbeat(lease, timedelta(seconds=1))
            elif action == "checkpoint":
                await repository.checkpoint(
                    lease,
                    stage="chunk",
                    resume_stage="chunk",
                    progress_current=1,
                    progress_total=2,
                )
            elif action == "advance":

                async def advance(_session: AsyncSession) -> None:
                    return None

                await repository.advance_stage(
                    lease,
                    advance,
                    stage="chunk",
                    resume_stage="chunk",
                    progress_current=0,
                    progress_total=None,
                )
            elif action == "stage_facts":

                async def commit_facts(_session: AsyncSession) -> None:
                    return None

                await repository.commit_stage_facts(lease, commit_facts)
            elif action == "failure":
                await repository.record_failure(
                    lease,
                    retryable=True,
                    error_code="TEMPORARY",
                    error_message="Temporary failure",
                    retry_delay=timedelta(seconds=1),
                )
            elif action == "success":
                await repository.mark_succeeded(lease)
            else:

                async def activate(_session: AsyncSession) -> None:
                    nonlocal activation_called
                    activation_called = True

                await repository.prepare_activation(lease, activate)

    stored = await _load(migrated_database, job.id)
    assert activation_called is False
    assert stored.status == "running"
    assert stored.lease_owner == "worker-a"
    assert stored.lease_epoch == lease.lease_epoch


@pytest.mark.asyncio
async def test_retryable_failure_uses_retry_wait_until_exhausted_and_terminal_states_clear_lease(
    migrated_database: Database,
) -> None:
    retry_job = _job(attempt_count=1, max_attempts=3)
    await _insert(migrated_database, retry_job)
    async with migrated_database.sessions() as session, session.begin():
        repository = SqlAlchemyJobRepository(session)
        lease = await repository.claim_next(
            lease_owner="worker-a",
            lease_duration=timedelta(seconds=30),
        )
        assert isinstance(lease, JobLease)
        status = await repository.record_failure(
            lease,
            retryable=True,
            error_code="TEMPORARY",
            error_message="Temporary failure",
            retry_delay=timedelta(seconds=5),
        )
    assert status == "retry_wait"
    stored = await _load(migrated_database, retry_job.id)
    assert stored.lease_owner is None
    assert stored.lease_expires_at is None
    assert stored.next_retry_at is not None

    exhausted_job = _job(attempt_count=2, max_attempts=3)
    await _insert(migrated_database, exhausted_job)
    async with migrated_database.sessions() as session, session.begin():
        repository = SqlAlchemyJobRepository(session)
        lease = await repository.claim_next(
            lease_owner="worker-b",
            lease_duration=timedelta(seconds=30),
        )
        assert isinstance(lease, JobLease)
        status = await repository.record_failure(
            lease,
            retryable=True,
            error_code="TEMPORARY",
            error_message="Temporary failure",
            retry_delay=timedelta(seconds=5),
        )
    assert status == "failed"
    stored = await _load(migrated_database, exhausted_job.id)
    assert stored.finished_at is not None
    assert stored.lease_owner is None
    assert stored.lease_expires_at is None
    assert stored.next_retry_at is None


@pytest.mark.asyncio
@pytest.mark.parametrize("retryable", [True, False])
async def test_cancellation_racing_failure_is_resolved_atomically_to_cancelled(
    migrated_database: Database,
    *,
    retryable: bool,
) -> None:
    job = _job(attempt_count=1, max_attempts=3)
    await _insert(migrated_database, job)
    async with migrated_database.sessions() as session, session.begin():
        repository = SqlAlchemyJobRepository(session)
        lease = await repository.claim_next(
            lease_owner="worker-a",
            lease_duration=timedelta(seconds=30),
        )
    assert isinstance(lease, JobLease)

    async with migrated_database.sessions() as session, session.begin():
        stored = await session.get(Job, job.id)
        assert stored is not None
        stored.cancel_requested_at = datetime.now(UTC)

    async with migrated_database.sessions() as session, session.begin():
        status = await SqlAlchemyJobRepository(session).record_failure(
            lease,
            retryable=retryable,
            error_code="PROVIDER_FAILED",
            error_message="Provider failed",
            retry_delay=timedelta(seconds=5),
        )

    assert status == "cancelled"
    stored = await _load(migrated_database, job.id)
    assert stored.status == "cancelled"
    assert stored.retryable is False
    assert stored.next_retry_at is None
    assert stored.error_code is None
    assert stored.error_message_sanitized is None
    assert stored.lease_owner is None
    assert stored.lease_expires_at is None
    assert stored.worker_heartbeat_at is None
    assert stored.finished_at is not None


@pytest.mark.asyncio
async def test_stop_rolls_back_a_postgres_claim_that_is_in_flight(
    migrated_database: Database,
) -> None:
    job = _job()
    await _insert(migrated_database, job)
    claim_started = asyncio.Event()
    release_claim = asyncio.Event()
    stop_event = asyncio.Event()
    handler_started = False

    class DelayedRepository(SqlAlchemyJobRepository):
        async def claim_next(
            self,
            *,
            lease_owner: str,
            lease_duration: timedelta,
            job_id: UUID | None = None,
        ) -> JobLease | ExhaustedJob | None:
            claim_started.set()
            await release_claim.wait()
            return await super().claim_next(
                lease_owner=lease_owner,
                lease_duration=lease_duration,
                job_id=job_id,
            )

    @asynccontextmanager
    async def repository_context() -> AsyncIterator[DelayedRepository]:
        async with migrated_database.sessions() as session, session.begin():
            yield DelayedRepository(session)

    runner = JobRunner(
        repository_context=repository_context,
        lease_owner="worker-a",
        lease_seconds=1,
        heartbeat_seconds=0.1,
        poll_interval_seconds=0.01,
        max_concurrency=1,
        shutdown_seconds=0.2,
        backoff=ExponentialBackoff(initial_seconds=1, maximum_seconds=2),
    )

    async def handler(_context: JobExecutionContext) -> None:
        nonlocal handler_started
        handler_started = True

    runner.register("ingest_document", handler)
    runner_task = asyncio.create_task(runner.run(stop_event))
    await asyncio.wait_for(claim_started.wait(), timeout=0.5)

    stop_event.set()
    release_claim.set()
    await asyncio.wait_for(runner_task, timeout=0.5)

    stored = await _load(migrated_database, job.id)
    assert handler_started is False
    assert stored.status == "queued"
    assert stored.attempt_count == 0
    assert stored.lease_owner is None
    assert stored.lease_expires_at is None


@pytest.mark.asyncio
async def test_stop_during_initial_cancellation_query_requeues_committed_claim(
    migrated_database: Database,
) -> None:
    job = _job()
    await _insert(migrated_database, job)
    cancellation_check_started = asyncio.Event()
    release_cancellation_check = asyncio.Event()
    stop_event = asyncio.Event()
    handler_started = False

    class DelayedCancellationRepository(SqlAlchemyJobRepository):
        async def cancellation_requested(self, lease: JobLease) -> bool:
            cancellation_check_started.set()
            await release_cancellation_check.wait()
            return await super().cancellation_requested(lease)

    @asynccontextmanager
    async def repository_context() -> AsyncIterator[DelayedCancellationRepository]:
        async with migrated_database.sessions() as session, session.begin():
            yield DelayedCancellationRepository(session)

    runner = JobRunner(
        repository_context=repository_context,
        lease_owner="worker-a",
        lease_seconds=1,
        heartbeat_seconds=0.1,
        poll_interval_seconds=0.01,
        max_concurrency=1,
        shutdown_seconds=0.2,
        backoff=ExponentialBackoff(initial_seconds=1, maximum_seconds=2),
    )

    async def handler(_context: JobExecutionContext) -> None:
        nonlocal handler_started
        handler_started = True

    runner.register("ingest_document", handler)
    runner_task = asyncio.create_task(runner.run(stop_event))
    await asyncio.wait_for(cancellation_check_started.wait(), timeout=0.5)

    stop_event.set()
    release_cancellation_check.set()
    await asyncio.wait_for(runner_task, timeout=0.5)

    stored = await _load(migrated_database, job.id)
    assert handler_started is False
    assert stored.status == "queued"
    assert stored.attempt_count == 0
    assert stored.lease_owner is None
    assert stored.lease_expires_at is None


@pytest.mark.asyncio
async def test_active_target_unique_index_rejects_parallel_jobs(db_session: AsyncSession) -> None:
    target_id = uuid4()
    db_session.add(_job(target_id=target_id))
    await db_session.flush()
    db_session.add(_job(status="retry_wait", target_id=target_id))

    with pytest.raises(IntegrityError):
        await db_session.flush()


@pytest.mark.asyncio
async def test_manual_retry_creates_child_preserves_checkpoint_and_replays_idempotently(
    migrated_database: Database,
) -> None:
    seed = await _seed_manual_retry_ingestion(migrated_database)
    snapshot = await _manual_retry_snapshot(migrated_database, seed.job_id)
    locked_tables: list[str] = []

    def capture_lock(
        _connection: object,
        _cursor: object,
        statement: str,
        _parameters: object,
        _context: object,
        _executemany: object,
    ) -> None:
        lowered = statement.lower()
        if " for update" not in lowered:
            return
        for table in (
            "api_keys",
            "knowledge_bases",
            "knowledge_base_index_generations",
            "documents",
            "document_versions",
            "document_index_states",
            "jobs",
        ):
            if f"from {table}" in lowered or f'from "{table}"' in lowered:
                locked_tables.append(table)
                break

    event.listen(
        migrated_database.engine.sync_engine,
        "before_cursor_execute",
        capture_lock,
    )
    try:
        async with migrated_database.sessions() as session, session.begin():
            created = await SqlAlchemyJobRepository(session).reserve_manual_retry(
                snapshot,
                actor=seed.actor,
                idempotency_key="manual-retry-1",
            )
    finally:
        event.remove(
            migrated_database.engine.sync_engine,
            "before_cursor_execute",
            capture_lock,
        )

    assert isinstance(created, ManualRetryReservation)
    assert created.created is True
    assert created.job.parent_job_id == seed.job_id
    assert created.job.root_job_id == seed.job_id
    assert created.job.status == "queued"
    assert created.job.stage == "embed_index"
    assert (created.job.progress_current, created.job.progress_total) == (1, 3)
    assert locked_tables == [
        "api_keys",
        "knowledge_bases",
        "knowledge_base_index_generations",
        "documents",
        "document_versions",
        "document_index_states",
        "jobs",
    ]

    async with migrated_database.sessions() as inspection:
        document = await inspection.get(Document, seed.document_id)
        version = await inspection.get(DocumentVersion, seed.version_id)
        state = await inspection.get(
            DocumentIndexState,
            (seed.version_id, seed.generation_id),
        )
        child = await inspection.get(Job, created.job.id)
        assert document is not None and version is not None and state is not None
        assert child is not None
        assert document.status == "processing"
        assert version.status == "indexing"
        assert state.status == "indexing"
        assert state.error_code is None
        assert state.safe_error_message is None
        assert state.next_chunk_index == 1
        assert state.embedding_config_hash == seed.embedding_config_hash
        assert version.source_object_key == seed.source_object_key
        assert version.parsed_object_key == seed.parsed_object_key
        assert version.chunk_manifest_object_key == seed.manifest_object_key
        assert version.chunk_manifest_checksum_sha256 == seed.manifest_checksum
        assert child.mutation_id == seed.mutation_id
        assert child.resume_stage == "embed_index"
        assert child.attempt_count == 0
        assert child.error_code is None
        assert child.lease_owner is None

    async with migrated_database.sessions() as session, session.begin():
        replay = await SqlAlchemyJobRepository(session).reserve_manual_retry(
            snapshot,
            actor=seed.actor,
            idempotency_key="manual-retry-1",
        )
    assert replay.created is False
    assert replay.job.id == created.job.id


@pytest.mark.asyncio
async def test_orphan_object_reconciliation_preserves_retryable_failed_version_artifacts(
    migrated_database: Database,
) -> None:
    now = datetime(2026, 7, 31, 12, 0, tzinfo=UTC)
    seed = await _seed_manual_retry_ingestion(
        migrated_database,
        stage="activate",
        resume_stage="activate",
        retryable=True,
    )
    base = (
        f"knowledge-bases/{seed.knowledge_base_id}/documents/{seed.document_id}"
        f"/versions/{seed.version_id}"
    )
    referenced_keys = (
        f"{base}/source/source.txt",
        f"{base}/parsed/text.txt",
        f"{base}/chunks/recursive_text_v1.jsonl",
    )
    orphan_key = (
        f"knowledge-bases/{seed.knowledge_base_id}/documents/{seed.document_id}"
        f"/versions/{uuid4()}/parsed/text.txt"
    )
    async with migrated_database.sessions() as session, session.begin():
        version = await session.get(DocumentVersion, seed.version_id)
        assert version is not None
        (
            version.source_object_key,
            version.parsed_object_key,
            version.chunk_manifest_object_key,
        ) = referenced_keys

    class _Store:
        def __init__(self) -> None:
            self.items = tuple(
                OrphanCandidate(key, 1, None, now - timedelta(hours=25))
                for key in (*referenced_keys, orphan_key)
            )
            self.deleted: list[str] = []

        async def list_older_than(
            self,
            *,
            prefix: str,
            older_than: datetime,
            limit: int,
            start_after: str | None = None,
        ) -> OrphanPage:
            matches = tuple(
                item
                for item in sorted(self.items, key=lambda candidate: candidate.object_key)
                if item.object_key.startswith(prefix)
                and item.last_modified < older_than
                and (start_after is None or item.object_key > start_after)
            )
            page = matches[:limit]
            next_start_after = page[-1].object_key if len(matches) > limit else None
            return OrphanPage(page, next_start_after)

        async def delete_best_effort(self, object_key: str) -> bool:
            self.deleted.append(object_key)
            return True

    @asynccontextmanager
    async def repository_context() -> AsyncIterator[SqlAlchemyIngestionPipelineRepository]:
        async with migrated_database.sessions() as session:
            yield SqlAlchemyIngestionPipelineRepository(session)

    store = _Store()
    pipeline = IngestionPipeline(
        repository_context=repository_context,
        object_store=cast(PipelineObjectStore, store),
        max_document_bytes=1024,
        reconciliation_clock=lambda: now,
    )
    try:
        temporary = await pipeline.reconcile_orphan_objects(
            grace_period=timedelta(hours=24),
            limit=10,
            cursor=None,
        )
        assert temporary.next_cursor == ObjectReconciliationCursor(1)
        canonical = await pipeline.reconcile_orphan_objects(
            grace_period=timedelta(hours=24),
            limit=10,
            cursor=temporary.next_cursor,
        )
    finally:
        await pipeline.aclose()

    assert canonical.deleted_count == 1
    assert canonical.failed_count == 0
    assert store.deleted == [orphan_key]


@pytest.mark.asyncio
async def test_cleanup_allows_terminal_nonretryable_unreferenced_canonical_artifacts(
    migrated_database: Database,
) -> None:
    seed = await _seed_manual_retry_ingestion(
        migrated_database,
        stage="parse",
        resume_stage="parse",
        retryable=False,
    )
    base = (
        f"knowledge-bases/{seed.knowledge_base_id}/documents/{seed.document_id}"
        f"/versions/{seed.version_id}"
    )
    candidates = (
        f"{base}/source/source.md",
        f"{base}/parsed/text.txt",
        f"{base}/chunks/recursive_text_v1.jsonl",
    )

    async with migrated_database.sessions() as session, session.begin():
        repository = SqlAlchemyIngestionPipelineRepository(session)
        allowed = tuple(
            [await repository.object_key_cleanup_is_allowed(candidate) for candidate in candidates]
        )

    assert allowed == (True, True, True)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("stage", "artifact_suffix"),
    (
        ("parse", "parsed/text.txt"),
        ("chunk", "chunks/recursive_text_v1.jsonl"),
    ),
)
async def test_cleanup_preserves_retryable_uncommitted_artifact_for_corresponding_stage(
    migrated_database: Database,
    stage: str,
    artifact_suffix: str,
) -> None:
    seed = await _seed_manual_retry_ingestion(
        migrated_database,
        stage=stage,
        resume_stage=stage,
        retryable=True,
    )
    object_key = (
        f"knowledge-bases/{seed.knowledge_base_id}/documents/{seed.document_id}"
        f"/versions/{seed.version_id}/{artifact_suffix}"
    )

    async with migrated_database.sessions() as session, session.begin():
        allowed = await SqlAlchemyIngestionPipelineRepository(
            session
        ).object_key_cleanup_is_allowed(object_key)

    assert allowed is False


@pytest.mark.asyncio
async def test_manual_retry_rejects_nonretryable_and_mismatch_without_partial_updates(
    migrated_database: Database,
) -> None:
    nonretryable = await _seed_manual_retry_ingestion(migrated_database, retryable=False)
    snapshot = await _manual_retry_snapshot(migrated_database, nonretryable.job_id)
    with pytest.raises(BusinessError) as raised:
        async with migrated_database.sessions() as session, session.begin():
            await SqlAlchemyJobRepository(session).reserve_manual_retry(
                snapshot,
                actor=nonretryable.actor,
                idempotency_key="manual-retry-not-allowed",
            )
    assert (raised.value.status_code, raised.value.code) == (409, "JOB_NOT_RETRYABLE")

    mismatch = await _seed_manual_retry_ingestion(
        migrated_database,
        state_embedding_hash="f" * 64,
    )
    mismatch_snapshot = await _manual_retry_snapshot(migrated_database, mismatch.job_id)
    with pytest.raises(BusinessError) as raised:
        async with migrated_database.sessions() as session, session.begin():
            await SqlAlchemyJobRepository(session).reserve_manual_retry(
                mismatch_snapshot,
                actor=mismatch.actor,
                idempotency_key="manual-retry-mismatch",
            )
    assert (raised.value.status_code, raised.value.code) == (
        409,
        "JOB_RETRY_STATE_MISMATCH",
    )

    async with migrated_database.sessions() as inspection:
        document = await inspection.get(Document, mismatch.document_id)
        version = await inspection.get(DocumentVersion, mismatch.version_id)
        state = await inspection.get(
            DocumentIndexState,
            (mismatch.version_id, mismatch.generation_id),
        )
        assert document is not None and version is not None and state is not None
        assert document.status == "failed"
        assert version.status == "failed"
        assert state.status == "failed"
        assert state.error_code == "UPSTREAM_UNAVAILABLE"
        assert (
            await inspection.scalar(
                select(func.count()).select_from(Job).where(Job.parent_job_id == mismatch.job_id)
            )
            == 0
        )


@pytest.mark.asyncio
async def test_manual_retry_rejects_noncanonical_parser_snapshot(
    migrated_database: Database,
) -> None:
    seed = await _seed_manual_retry_ingestion(migrated_database)
    async with migrated_database.sessions() as session, session.begin():
        version = await session.get(DocumentVersion, seed.version_id)
        assert version is not None
        version.parser_name = "legacy_plain_text"
    snapshot = await _manual_retry_snapshot(migrated_database, seed.job_id)

    with pytest.raises(BusinessError) as raised:
        async with migrated_database.sessions() as session, session.begin():
            await SqlAlchemyJobRepository(session).reserve_manual_retry(
                snapshot,
                actor=seed.actor,
                idempotency_key="manual-retry-parser-drift",
            )

    assert (raised.value.status_code, raised.value.code) == (
        409,
        "JOB_RETRY_STATE_MISMATCH",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("stage", "resume_stage", "progress_current", "progress_total"),
    (
        ("parse", None, 0, None),
        ("chunk", "chunk", 0, None),
        ("embed_index", "embed_index", 1, 3),
        ("validate", "validate", 3, 3),
        ("activate", "activate", 3, 3),
    ),
)
async def test_manual_retry_accepts_only_canonical_stage_progress_tuples(
    migrated_database: Database,
    stage: str,
    resume_stage: str | None,
    progress_current: int,
    progress_total: int | None,
) -> None:
    seed = await _seed_manual_retry_ingestion(
        migrated_database,
        stage=stage,
        resume_stage=resume_stage,
    )
    snapshot = await _manual_retry_snapshot(migrated_database, seed.job_id)

    async with migrated_database.sessions() as session, session.begin():
        reservation = await SqlAlchemyJobRepository(session).reserve_manual_retry(
            snapshot,
            actor=seed.actor,
            idempotency_key=f"manual-retry-compatible-{stage}",
        )

    assert reservation.created is True
    assert reservation.job.stage == stage
    assert reservation.job.progress_current == progress_current
    assert reservation.job.progress_total == progress_total


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("stage", "resume_stage"),
    (
        ("parse", None),
        ("chunk", "chunk"),
        ("embed_index", "embed_index"),
        ("validate", "validate"),
        ("activate", "activate"),
    ),
)
async def test_manual_retry_rejects_stage_progress_drift_without_partial_updates(
    migrated_database: Database,
    stage: str,
    resume_stage: str | None,
) -> None:
    seed = await _seed_manual_retry_ingestion(
        migrated_database,
        stage=stage,
        resume_stage=resume_stage,
    )
    async with migrated_database.sessions() as session, session.begin():
        job = await session.get(Job, seed.job_id)
        assert job is not None
        if stage in {"parse", "chunk"}:
            job.progress_total = 1
        else:
            job.progress_current -= 1
    snapshot = await _manual_retry_snapshot(migrated_database, seed.job_id)

    with pytest.raises(BusinessError) as raised:
        async with migrated_database.sessions() as session, session.begin():
            await SqlAlchemyJobRepository(session).reserve_manual_retry(
                snapshot,
                actor=seed.actor,
                idempotency_key=f"manual-retry-progress-drift-{stage}",
            )

    assert (raised.value.status_code, raised.value.code) == (
        409,
        "JOB_RETRY_STATE_MISMATCH",
    )
    async with migrated_database.sessions() as inspection:
        document = await inspection.get(Document, seed.document_id)
        version = await inspection.get(DocumentVersion, seed.version_id)
        state = await inspection.get(
            DocumentIndexState,
            (seed.version_id, seed.generation_id),
        )
        assert document is not None and version is not None and state is not None
        assert document.status == "failed"
        assert version.status == "failed"
        assert state.status == "failed"
        assert (
            await inspection.scalar(
                select(func.count()).select_from(Job).where(Job.parent_job_id == seed.job_id)
            )
            == 0
        )


@pytest.mark.asyncio
async def test_manual_ingestion_retry_conflicts_with_active_generation_rebuild(
    migrated_database: Database,
) -> None:
    seed = await _seed_manual_retry_ingestion(migrated_database)
    async with migrated_database.sessions() as session, session.begin():
        session.add(
            Job(
                id=uuid4(),
                knowledge_base_id=seed.knowledge_base_id,
                target_type="index_generation",
                target_id=seed.generation_id,
                target_revision=99,
                index_generation_id=seed.generation_id,
                operation="rebuild_generation",
                stage="indexing",
                resume_stage="indexing",
                status="queued",
                progress_current=0,
                progress_total=3,
                attempt_count=0,
                max_attempts=5,
                retryable=True,
            )
        )
    snapshot = await _manual_retry_snapshot(migrated_database, seed.job_id)

    with pytest.raises(BusinessError) as raised:
        async with migrated_database.sessions() as session, session.begin():
            await SqlAlchemyJobRepository(session).reserve_manual_retry(
                snapshot,
                actor=seed.actor,
                idempotency_key="manual-retry-active-rebuild",
            )

    assert (raised.value.status_code, raised.value.code) == (409, "JOB_RETRY_CONFLICT")


@pytest.mark.asyncio
async def test_manual_rebuild_retry_conflicts_across_revision_and_operation(
    migrated_database: Database,
) -> None:
    for blocker_operation in ("rebuild_generation", "ingest_document"):
        job_id, _mutation_id, generation_id, actor = await _seed_manual_retry_rebuild(
            migrated_database
        )
        snapshot = await _manual_retry_snapshot(migrated_database, job_id)
        async with migrated_database.sessions() as session, session.begin():
            source = await session.get(Job, job_id)
            assert source is not None and source.target_revision is not None
            session.add(
                Job(
                    id=uuid4(),
                    knowledge_base_id=source.knowledge_base_id,
                    target_type=(
                        "index_generation"
                        if blocker_operation == "rebuild_generation"
                        else "document_version"
                    ),
                    target_id=(
                        generation_id if blocker_operation == "rebuild_generation" else uuid4()
                    ),
                    target_revision=source.target_revision + 1,
                    index_generation_id=generation_id,
                    operation=blocker_operation,
                    stage=(
                        "indexing" if blocker_operation == "rebuild_generation" else "embed_index"
                    ),
                    resume_stage=(
                        "indexing" if blocker_operation == "rebuild_generation" else "embed_index"
                    ),
                    status="queued",
                    progress_current=0,
                    progress_total=10,
                    attempt_count=0,
                    max_attempts=5,
                    retryable=True,
                )
            )

        with pytest.raises(BusinessError) as raised:
            async with migrated_database.sessions() as session, session.begin():
                await SqlAlchemyJobRepository(session).reserve_manual_retry(
                    snapshot,
                    actor=actor,
                    idempotency_key=f"manual-retry-blocked-{blocker_operation}",
                )

        assert (raised.value.status_code, raised.value.code) == (409, "JOB_RETRY_CONFLICT")


@pytest.mark.asyncio
async def test_manual_retry_concurrent_reservations_allow_only_one_active_target(
    migrated_database: Database,
) -> None:
    seed = await _seed_manual_retry_ingestion(migrated_database)
    snapshots = await asyncio.gather(
        _manual_retry_snapshot(migrated_database, seed.job_id),
        _manual_retry_snapshot(migrated_database, seed.job_id),
    )

    async def reserve(snapshot: ManualRetrySnapshot, key: str) -> ManualRetryReservation | str:
        try:
            async with migrated_database.sessions() as session, session.begin():
                return await SqlAlchemyJobRepository(session).reserve_manual_retry(
                    snapshot,
                    actor=seed.actor,
                    idempotency_key=key,
                )
        except BusinessError as error:
            return error.code

    outcomes = await asyncio.gather(
        reserve(snapshots[0], "manual-concurrent-a"),
        reserve(snapshots[1], "manual-concurrent-b"),
    )

    assert sum(isinstance(outcome, ManualRetryReservation) for outcome in outcomes) == 1
    assert "JOB_RETRY_CONFLICT" in outcomes
    async with migrated_database.sessions() as inspection:
        assert (
            await inspection.scalar(
                select(func.count()).select_from(Job).where(Job.parent_job_id == seed.job_id)
            )
            == 1
        )


@pytest.mark.asyncio
async def test_manual_retry_rebuild_child_copies_configuration_anchor(
    migrated_database: Database,
) -> None:
    job_id, mutation_id, generation_id, actor = await _seed_manual_retry_rebuild(migrated_database)
    snapshot = await _manual_retry_snapshot(migrated_database, job_id)

    async with migrated_database.sessions() as session, session.begin():
        reservation = await SqlAlchemyJobRepository(session).reserve_manual_retry(
            snapshot,
            actor=actor,
            idempotency_key="manual-rebuild-retry",
        )

    assert reservation.created is True
    assert reservation.job.parent_job_id == job_id
    assert reservation.job.root_job_id == job_id
    assert reservation.job.stage == "indexing"
    assert (reservation.job.progress_current, reservation.job.progress_total) == (4, 10)
    async with migrated_database.sessions() as inspection:
        child = await inspection.get(Job, reservation.job.id)
        assert child is not None
        assert child.operation == "rebuild_generation"
        assert child.target_id == generation_id
        assert child.index_generation_id == generation_id
        assert child.mutation_id == mutation_id
        assert child.target_revision == 5
        assert child.resume_stage == "indexing"


@pytest.mark.asyncio
@pytest.mark.parametrize("drift", ("target_revision", "point_total", "stage_progress"))
async def test_manual_rebuild_retry_rejects_canonical_target_or_progress_drift(
    migrated_database: Database,
    drift: str,
) -> None:
    job_id, _mutation_id, _generation_id, actor = await _seed_manual_retry_rebuild(
        migrated_database
    )
    async with migrated_database.sessions() as session, session.begin():
        source = await session.get(Job, job_id)
        assert source is not None and source.knowledge_base_id is not None
        if drift == "target_revision":
            knowledge_base = await session.get(KnowledgeBase, source.knowledge_base_id)
            assert knowledge_base is not None
            knowledge_base.mutation_revision += 1
        elif drift == "point_total":
            version = await session.scalar(
                select(DocumentVersion)
                .join(Document, Document.current_version_id == DocumentVersion.id)
                .where(Document.knowledge_base_id == source.knowledge_base_id)
            )
            assert version is not None
            state = await session.get(
                DocumentIndexState,
                (version.id, source.index_generation_id),
            )
            assert state is not None
            version.chunk_count = 11
            state.expected_point_count = 11
            state.actual_point_count = 11
            state.next_chunk_index = 11
        else:
            source.stage = "validating"
            source.resume_stage = "validating"
            source.progress_current = 9
    snapshot = await _manual_retry_snapshot(migrated_database, job_id)

    with pytest.raises(BusinessError) as raised:
        async with migrated_database.sessions() as session, session.begin():
            await SqlAlchemyJobRepository(session).reserve_manual_retry(
                snapshot,
                actor=actor,
                idempotency_key=f"manual-rebuild-drift-{drift}",
            )

    assert (raised.value.status_code, raised.value.code) == (
        409,
        "JOB_RETRY_STATE_MISMATCH",
    )
    async with migrated_database.sessions() as inspection:
        assert (
            await inspection.scalar(
                select(func.count()).select_from(Job).where(Job.parent_job_id == job_id)
            )
            == 0
        )


@pytest.mark.asyncio
async def test_manual_rebuild_retry_api_accepts_ingest_and_hides_read_capabilities(
    migrated_database: Database,
) -> None:
    job_id, _mutation_id, _generation_id, actor = await _seed_manual_retry_rebuild(
        migrated_database
    )
    async with migrated_database.sessions() as session, session.begin():
        actor_row = await session.get(ApiKey, actor.key_id)
        assert actor_row is not None
        actor_row.capabilities = [Capability.INGEST.value]

    def principal(capability: Capability) -> AgentPrincipal:
        return AgentPrincipal(
            key_id=actor.key_id,
            public_id=actor.public_id,
            capabilities=frozenset({capability}),
            knowledge_base_ids=actor.knowledge_base_ids,
            query_profile_ids=frozenset(),
            default_query_profile_id=None,
            raw_file_read=False,
            requests_per_minute=60,
            max_concurrency=4,
        )

    app = create_app(settings=Settings(_env_file=None))
    app.state.database = migrated_database

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        for denied in (Capability.RETRIEVE, Capability.ANSWER):
            app.dependency_overrides[require_agent_principal] = partial(principal, denied)
            response = await client.post(
                f"/v1/jobs/{job_id}/retry",
                headers={
                    "X-Request-ID": f"job-rebuild-{denied.value}-denied",
                    "Idempotency-Key": f"rebuild-{denied.value}-denied",
                },
            )
            assert response.status_code == 404
            assert response.json()["error"]["code"] == "RESOURCE_NOT_FOUND"

        app.dependency_overrides[require_agent_principal] = partial(
            principal,
            Capability.INGEST,
        )
        accepted = await client.post(
            f"/v1/jobs/{job_id}/retry",
            headers={
                "X-Request-ID": "job-rebuild-ingest-accepted",
                "Idempotency-Key": "rebuild-ingest-accepted",
            },
        )

    assert accepted.status_code == 202
    assert accepted.json()["operation"] == "rebuild_generation"


@pytest.mark.asyncio
@pytest.mark.acceptance
async def test_manual_retry_notification_timeout_is_bounded_after_commit(
    migrated_database: Database,
) -> None:
    class BlockingNotifier:
        async def notify(self, _job_id: UUID) -> bool:
            await asyncio.Event().wait()
            return True

    seed = await _seed_manual_retry_ingestion(migrated_database)
    app = create_app(
        settings=Settings(
            _env_file=None,
            ingestion_notify_timeout_seconds=0.01,
        )
    )
    app.state.database = migrated_database
    app.state.job_notifier = BlockingNotifier()
    app.dependency_overrides[require_agent_principal] = lambda: seed.actor

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await asyncio.wait_for(
            client.post(
                f"/v1/jobs/{seed.job_id}/retry",
                headers={
                    "X-Request-ID": "job-retry-notify-timeout",
                    "Idempotency-Key": "retry-notify-timeout",
                },
            ),
            timeout=0.5,
        )

    assert response.status_code == 202


@pytest.mark.asyncio
async def test_job_api_hides_internal_fields_and_requires_valid_retry_idempotency(
    migrated_database: Database,
) -> None:
    seed = await _seed_manual_retry_ingestion(migrated_database)
    settings = Settings(_env_file=None, max_idempotency_key_length=16)
    app = create_app(settings=settings)
    app.state.database = migrated_database
    app.dependency_overrides[require_agent_principal] = lambda: seed.actor

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        status = await client.get(
            f"/v1/jobs/{seed.job_id}",
            headers={"X-Request-ID": "job-api-status"},
        )
        assert status.status_code == 200
        assert status.headers["cache-control"] == "no-store"
        document = status.json()
        assert set(document) == {
            "id",
            "operation",
            "status",
            "stage",
            "progress_current",
            "progress_total",
            "attempt_count",
            "max_attempts",
            "retryable",
            "error_code",
            "error_message",
            "parent_job_id",
            "root_job_id",
            "created_at",
            "started_at",
            "finished_at",
        }
        assert document["status"] == "failed"
        assert document["error_code"] == "UPSTREAM_UNAVAILABLE"
        for secret in (
            seed.source_object_key,
            seed.parsed_object_key,
            seed.manifest_object_key,
            seed.embedding_config_hash,
            "lease_owner",
            "resume_stage",
            "next_chunk_index",
            "traceback",
            "raw_upstream",
        ):
            assert secret not in status.text

        missing_key = await client.post(
            f"/v1/jobs/{seed.job_id}/retry",
            headers={"X-Request-ID": "job-api-missing-key"},
        )
        assert missing_key.status_code == 422
        assert missing_key.headers["cache-control"] == "no-store"

        invalid_key = await client.post(
            f"/v1/jobs/{seed.job_id}/retry",
            headers={
                "X-Request-ID": "job-api-invalid-key",
                "Idempotency-Key": "contains space",
            },
        )
        assert invalid_key.status_code == 422
        assert invalid_key.json()["error"]["code"] == "VALIDATION_ERROR"

        retriever = AgentPrincipal(
            key_id=seed.actor.key_id,
            public_id=seed.actor.public_id,
            capabilities=frozenset({Capability.RETRIEVE}),
            knowledge_base_ids=seed.actor.knowledge_base_ids,
            query_profile_ids=frozenset(),
            default_query_profile_id=None,
            raw_file_read=False,
            requests_per_minute=60,
            max_concurrency=4,
        )
        app.dependency_overrides[require_agent_principal] = lambda: retriever
        denied_retry = await client.post(
            f"/v1/jobs/{seed.job_id}/retry",
            headers={
                "X-Request-ID": "job-api-denied-retry",
                "Idempotency-Key": "retry-denied",
            },
        )
        assert denied_retry.status_code == 404
        assert denied_retry.json()["error"]["code"] == "RESOURCE_NOT_FOUND"

        app.dependency_overrides[require_agent_principal] = lambda: seed.actor
        retried = await client.post(
            f"/v1/jobs/{seed.job_id}/retry",
            headers={
                "X-Request-ID": "job-api-retry",
                "Idempotency-Key": "retry-http-1",
            },
            json={
                "stage": "parse",
                "index_generation_id": str(uuid4()),
                "provider_config_id": str(uuid4()),
            },
        )
        assert retried.status_code == 202
        assert retried.headers["cache-control"] == "no-store"
        retried_document = retried.json()
        assert retried_document["stage"] == "embed_index"
        assert retried_document["parent_job_id"] == str(seed.job_id)
        assert "index_generation_id" not in retried_document
        assert "provider_config_id" not in retried_document
