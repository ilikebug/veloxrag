import asyncio
import logging
import os
import subprocess
import sys
import textwrap
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from uuid import UUID, uuid4

import pytest
from sqlalchemy import select
from sqlalchemy.dialects import postgresql
from sqlalchemy.engine import Dialect
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

import rag_service.jobs.runner as jobs_runner
import rag_service.jobs.services as jobs_services
import rag_service.jobs.worker as jobs_worker
from rag_service.api.errors import BusinessError
from rag_service.auth.policies import AgentPrincipal, Capability
from rag_service.config import Settings
from rag_service.db.models.documents import Job
from rag_service.db.session import Database
from rag_service.jobs.notifier import (
    JOB_NOTIFICATION_CHANNEL,
    RedisJobNotifier,
    RedisJobSubscriber,
)
from rag_service.jobs.repositories import (
    ExhaustedJob,
    JobLease,
    JobStatusRecord,
    LostLeaseError,
    ManualRetryReservation,
    ManualRetrySnapshot,
    SqlAlchemyJobRepository,
)
from rag_service.jobs.runner import (
    ExponentialBackoff,
    JobExecutionContext,
    JobHandlerOutcome,
    JobRunner,
    PermanentJobError,
    RepositoryContextFactory,
    RetryableJobError,
    RetryableProviderJobError,
)
from rag_service.jobs.services import JobService, _safe_job
from rag_service.jobs.worker import (
    WorkerDependencyPreflight,
    WorkerHealthSnapshot,
    _install_signal_handlers,
    _periodic_health_check,
    _remove_signal_handlers,
    _start_notification_source,
    health_main,
    load_worker_health,
    write_worker_health,
)
from rag_service.main import create_app
from rag_service.observability.metrics import OperationalMetrics
from rag_service.providers.credentials import ProviderCredentialKeyring


def test_job_api_routes_expose_only_the_safe_contract() -> None:
    document = create_app(settings=Settings(_env_file=None)).openapi()

    status_operation = document["paths"]["/v1/jobs/{job_id}"]["get"]
    retry_operation = document["paths"]["/v1/jobs/{job_id}/retry"]["post"]
    safe_job = document["components"]["schemas"]["SafeJob"]

    assert status_operation["responses"]["200"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/SafeJob"
    }
    assert retry_operation["responses"]["202"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/SafeJob"
    }
    assert {
        parameter["name"]
        for parameter in retry_operation["parameters"]
        if parameter.get("required") is True
    } >= {"job_id", "Idempotency-Key"}
    assert set(safe_job["properties"]) == {
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


def _job_api_actor(
    capability: Capability,
    knowledge_base_id: UUID,
) -> AgentPrincipal:
    return AgentPrincipal(
        key_id=uuid4(),
        public_id="YWJjZGVmZ2hpamtsbW5vcA",
        capabilities=frozenset({capability}),
        knowledge_base_ids=frozenset({knowledge_base_id}),
        query_profile_ids=frozenset(),
        default_query_profile_id=None,
        raw_file_read=False,
        requests_per_minute=60,
        max_concurrency=4,
    )


@pytest.mark.parametrize(
    "capability",
    (Capability.MANAGE, Capability.INGEST, Capability.RETRIEVE),
)
def test_job_api_get_authorization_accepts_only_scoped_read_capabilities(
    capability: Capability,
) -> None:
    knowledge_base_id = uuid4()
    JobService._authorize(
        _job_api_actor(capability, knowledge_base_id),
        knowledge_base_id,
        frozenset({Capability.MANAGE, Capability.INGEST, Capability.RETRIEVE}),
    )

    for actor in (
        _job_api_actor(Capability.ANSWER, knowledge_base_id),
        _job_api_actor(capability, uuid4()),
    ):
        with pytest.raises(BusinessError) as raised:
            JobService._authorize(
                actor,
                knowledge_base_id,
                frozenset({Capability.MANAGE, Capability.INGEST, Capability.RETRIEVE}),
            )
        assert (raised.value.status_code, raised.value.code) == (404, "RESOURCE_NOT_FOUND")


@pytest.mark.parametrize("capability", (Capability.MANAGE, Capability.INGEST))
def test_manual_retry_authorization_requires_scoped_write_capability(
    capability: Capability,
) -> None:
    knowledge_base_id = uuid4()
    JobService._authorize(
        _job_api_actor(capability, knowledge_base_id),
        knowledge_base_id,
        frozenset({Capability.MANAGE, Capability.INGEST}),
    )

    for denied in (Capability.RETRIEVE, Capability.ANSWER):
        with pytest.raises(BusinessError) as raised:
            JobService._authorize(
                _job_api_actor(denied, knowledge_base_id),
                knowledge_base_id,
                frozenset({Capability.MANAGE, Capability.INGEST}),
            )
        assert (raised.value.status_code, raised.value.code) == (404, "RESOURCE_NOT_FOUND")


def test_manual_rebuild_retry_authorization_accepts_scoped_write_capability() -> None:
    knowledge_base_id = uuid4()
    for capability in (Capability.MANAGE, Capability.INGEST):
        JobService._authorize(
            _job_api_actor(capability, knowledge_base_id),
            knowledge_base_id,
            frozenset({Capability.MANAGE, Capability.INGEST}),
        )

    for denied in (Capability.RETRIEVE, Capability.ANSWER):
        with pytest.raises(BusinessError) as raised:
            JobService._authorize(
                _job_api_actor(denied, knowledge_base_id),
                knowledge_base_id,
                frozenset({Capability.MANAGE, Capability.INGEST}),
            )
        assert (raised.value.status_code, raised.value.code) == (404, "RESOURCE_NOT_FOUND")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("constraint_name", "expected"),
    (
        ("uq_jobs_active_target", (409, "JOB_RETRY_CONFLICT")),
        ("ck_jobs_progress", (500, "INTERNAL_ERROR")),
    ),
)
async def test_manual_retry_maps_only_expected_active_target_integrity_conflicts(
    monkeypatch: pytest.MonkeyPatch,
    constraint_name: str,
    expected: tuple[int, str],
) -> None:
    knowledge_base_id = uuid4()
    snapshot = ManualRetrySnapshot(
        id=uuid4(),
        knowledge_base_id=knowledge_base_id,
        actor_api_key_id=uuid4(),
        operation="ingest_document",
        target_type="document_version",
        target_id=uuid4(),
        target_revision=1,
        index_generation_id=uuid4(),
        mutation_id=uuid4(),
        parent_job_id=None,
        root_job_id=None,
        stage="parse",
        status="failed",
        progress_current=0,
        progress_total=None,
        attempt_count=5,
        max_attempts=5,
        retryable=True,
        resume_stage=None,
        document_id=uuid4(),
    )

    class SnapshotRepository:
        async def get_manual_retry_snapshot(self, _job_id: UUID) -> ManualRetrySnapshot:
            return snapshot

    @asynccontextmanager
    async def transaction() -> AsyncIterator[None]:
        yield

    class ReservationSession:
        def begin(self) -> AbstractAsyncContextManager[None]:
            return transaction()

    @asynccontextmanager
    async def reservation_sessions() -> AsyncIterator[AsyncSession]:
        yield cast(AsyncSession, ReservationSession())

    async def fail_reservation(
        _repository: SqlAlchemyJobRepository,
        _snapshot: ManualRetrySnapshot,
        *,
        actor: AgentPrincipal,
        idempotency_key: str,
    ) -> None:
        assert actor.key_id is not None
        assert idempotency_key == "integrity-test"
        original = Exception("private database detail")
        original.diag = SimpleNamespace(constraint_name=constraint_name)  # type: ignore[attr-defined]
        raise IntegrityError("private statement", {}, original)

    monkeypatch.setattr(
        SqlAlchemyJobRepository,
        "reserve_manual_retry",
        fail_reservation,
    )
    async with AsyncSession() as request_session:
        service = JobService(
            session=request_session,
            reservation_session_factory=reservation_sessions,
            max_idempotency_key_length=128,
        )
        service._repository = cast(SqlAlchemyJobRepository, SnapshotRepository())
        with pytest.raises(BusinessError) as raised:
            await service.retry_job(
                snapshot.id,
                actor=_job_api_actor(Capability.INGEST, knowledge_base_id),
                idempotency_key="integrity-test",
            )

    assert (raised.value.status_code, raised.value.code) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("created", (True, False))
async def test_manual_retry_observes_queued_only_after_new_transaction_commit(
    monkeypatch: pytest.MonkeyPatch,
    created: bool,
) -> None:
    knowledge_base_id = uuid4()
    generation_id = uuid4()
    version_id = uuid4()
    snapshot = ManualRetrySnapshot(
        id=uuid4(),
        knowledge_base_id=knowledge_base_id,
        actor_api_key_id=uuid4(),
        operation="ingest_document",
        target_type="document_version",
        target_id=version_id,
        target_revision=1,
        index_generation_id=generation_id,
        mutation_id=uuid4(),
        parent_job_id=None,
        root_job_id=None,
        stage="parse",
        status="failed",
        progress_current=0,
        progress_total=None,
        attempt_count=5,
        max_attempts=5,
        retryable=True,
        resume_stage=None,
        document_id=uuid4(),
    )
    now = datetime.now(UTC)
    reserved_job = JobStatusRecord(
        id=uuid4(),
        knowledge_base_id=knowledge_base_id,
        operation="ingest_document",
        status="queued",
        stage="parse",
        progress_current=0,
        progress_total=None,
        attempt_count=0,
        max_attempts=5,
        retryable=True,
        error_code=None,
        error_message=None,
        parent_job_id=snapshot.id,
        root_job_id=snapshot.id,
        created_at=now,
        started_at=None,
        finished_at=None,
    )

    class SnapshotRepository:
        async def get_manual_retry_snapshot(self, _job_id: UUID) -> ManualRetrySnapshot:
            return snapshot

    @asynccontextmanager
    async def transaction() -> AsyncIterator[None]:
        yield

    class ReservationSession:
        def begin(self) -> AbstractAsyncContextManager[None]:
            return transaction()

    @asynccontextmanager
    async def reservation_sessions() -> AsyncIterator[AsyncSession]:
        yield cast(AsyncSession, ReservationSession())

    async def reserve(
        _repository: SqlAlchemyJobRepository,
        _snapshot: ManualRetrySnapshot,
        *,
        actor: AgentPrincipal,
        idempotency_key: str,
    ) -> ManualRetryReservation:
        del actor, idempotency_key
        return ManualRetryReservation(reserved_job, created=created)

    monkeypatch.setattr(SqlAlchemyJobRepository, "reserve_manual_retry", reserve)
    metrics = OperationalMetrics()
    handler = _CollectingJobLogHandler()
    previous_handlers = list(jobs_services.logger.handlers)
    previous_propagate = jobs_services.logger.propagate
    previous_level = jobs_services.logger.level
    jobs_services.logger.handlers = [handler]
    jobs_services.logger.propagate = False
    jobs_services.logger.setLevel(logging.INFO)
    try:
        async with AsyncSession() as request_session:
            service = JobService(
                session=request_session,
                reservation_session_factory=reservation_sessions,
                max_idempotency_key_length=128,
                metrics=metrics,
            )
            service._repository = cast(SqlAlchemyJobRepository, SnapshotRepository())
            result = await service.retry_job(
                snapshot.id,
                actor=_job_api_actor(Capability.INGEST, knowledge_base_id),
                idempotency_key="manual-retry-observed",
                request_id="req-manual-retry",
            )
    finally:
        jobs_services.logger.handlers = previous_handlers
        jobs_services.logger.propagate = previous_propagate
        jobs_services.logger.setLevel(previous_level)

    assert result.id == reserved_job.id
    assert metrics.registry.get_sample_value(
        "rag_job_state_transitions_total", {"state": "queued"}
    ) == (1 if created else None)
    queued_records = [record for record in handler.records if record.msg == "job.state.changed"]
    assert len(queued_records) == (1 if created else 0)
    if queued_records:
        record = queued_records[0]
        assert record.__dict__["request_id"] == "req-manual-retry"
        assert record.__dict__["knowledge_base_id"] == str(knowledge_base_id)
        assert record.__dict__["version_id"] == str(version_id)
        assert record.__dict__["job_id"] == str(reserved_job.id)
        assert record.__dict__["generation_id"] == str(generation_id)
        assert record.__dict__["operation"] == "ingest_document"


def test_job_api_safe_serialization_is_an_explicit_allowlist() -> None:
    now = datetime.now(UTC)
    record = JobStatusRecord(
        id=uuid4(),
        knowledge_base_id=uuid4(),
        operation="ingest_document",
        status="failed",
        stage="embed_index",
        progress_current=2,
        progress_total=5,
        attempt_count=5,
        max_attempts=5,
        retryable=True,
        error_code="UPSTREAM_UNAVAILABLE",
        error_message="Embedding provider temporarily unavailable",
        parent_job_id=None,
        root_job_id=None,
        created_at=now,
        started_at=now,
        finished_at=now,
    )

    document = _safe_job(record).model_dump(mode="json")

    assert set(document) == set(
        create_app(settings=Settings(_env_file=None)).openapi()["components"]["schemas"]["SafeJob"][
            "properties"
        ]
    )
    serialized = str(document)
    for forbidden in (
        "object_key",
        "provider_config",
        "credential_id",
        "traceback",
        "raw_upstream",
        "lease_owner",
        "resume_stage",
        "next_chunk_index",
    ):
        assert forbidden not in serialized


def test_manual_retry_fence_uses_null_safe_checkpoint_and_target_comparisons() -> None:
    snapshot = ManualRetrySnapshot(
        id=uuid4(),
        knowledge_base_id=uuid4(),
        actor_api_key_id=None,
        operation="ingest_document",
        target_type="document_version",
        target_id=uuid4(),
        target_revision=None,
        index_generation_id=None,
        mutation_id=None,
        parent_job_id=None,
        root_job_id=None,
        stage="parse",
        status="failed",
        progress_current=0,
        progress_total=None,
        attempt_count=5,
        max_attempts=5,
        retryable=True,
        resume_stage=None,
        document_id=uuid4(),
    )

    statement = select(Job.id).where(*SqlAlchemyJobRepository.manual_retry_fence(snapshot))
    dialect = cast(Callable[[], Dialect], postgresql.dialect)()
    sql = str(statement.compile(dialect=dialect))

    assert sql.count("IS NOT DISTINCT FROM") >= 10
    assert "jobs.target_revision IS NOT DISTINCT FROM" in sql
    assert "jobs.index_generation_id IS NOT DISTINCT FROM" in sql
    assert "jobs.stage IS NOT DISTINCT FROM" in sql
    assert "jobs.resume_stage IS NOT DISTINCT FROM" in sql
    assert "jobs.progress_total IS NOT DISTINCT FROM" in sql


class _PublishingRedis:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.calls: list[tuple[str, str]] = []

    async def publish(self, channel: str, payload: str) -> int:
        self.calls.append((channel, payload))
        if self.error is not None:
            raise self.error
        return 1


class _CollectingJobLogHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def _lease(*, attempt_count: int = 1) -> JobLease:
    now = datetime.now(UTC)
    return JobLease(
        id=uuid4(),
        operation="ingest_document",
        target_type="document_version",
        target_id=uuid4(),
        target_revision=1,
        index_generation_id=None,
        stage="parse",
        resume_stage=None,
        progress_current=0,
        progress_total=None,
        attempt_count=attempt_count,
        max_attempts=5,
        lease_owner="worker-a",
        lease_epoch=1,
        lease_expires_at=now + timedelta(seconds=30),
        cancel_requested_at=None,
    )


def _exhausted_job() -> ExhaustedJob:
    lease = _lease(attempt_count=5)
    expired_at = datetime.now(UTC) - timedelta(seconds=1)
    return ExhaustedJob(
        id=lease.id,
        knowledge_base_id=uuid4(),
        operation=lease.operation,
        target_type=lease.target_type,
        target_id=lease.target_id,
        target_revision=lease.target_revision,
        index_generation_id=lease.index_generation_id,
        stage=lease.stage,
        status="running",
        attempt_count=5,
        max_attempts=5,
        next_retry_at=None,
        lease_owner=lease.lease_owner,
        lease_epoch=lease.lease_epoch,
        lease_expires_at=expired_at,
        cancel_requested_at=None,
    )


class _MemoryJobRepository:
    def __init__(self, lease: JobLease | ExhaustedJob | None) -> None:
        self.available = lease
        self.heartbeats = 0
        self.cancel_requested = False
        self.successes = 0
        self.cancellations = 0
        self.failures: list[tuple[bool, str, str, timedelta]] = []
        self.checkpoints: list[tuple[str | None, str | None, int, int | None]] = []
        self.activation_preparations = 0
        self.domain_finalizations = 0
        self.domain_finalized_status = "succeeded"
        self.exhausted_finalizations = 0
        self.stage_fact_commits = 0
        self.stage_advances = 0
        self.stage_checkpoints = 0
        self.lost = False
        self.heartbeat_error: Exception | None = None
        self.claim_calls = 0

    async def claim_next(
        self,
        *,
        lease_owner: str,
        lease_duration: timedelta,
        job_id: UUID | None = None,
    ) -> JobLease | ExhaustedJob | None:
        del lease_duration
        self.claim_calls += 1
        lease = self.available
        if lease is None or (job_id is not None and lease.id != job_id):
            return None
        self.available = None
        if isinstance(lease, ExhaustedJob):
            return lease
        return JobLease(
            id=lease.id,
            operation=lease.operation,
            target_type=lease.target_type,
            target_id=lease.target_id,
            target_revision=lease.target_revision,
            index_generation_id=lease.index_generation_id,
            stage=lease.stage,
            resume_stage=lease.resume_stage,
            progress_current=lease.progress_current,
            progress_total=lease.progress_total,
            attempt_count=lease.attempt_count,
            max_attempts=lease.max_attempts,
            lease_owner=lease_owner,
            lease_epoch=lease.lease_epoch,
            lease_expires_at=lease.lease_expires_at,
            cancel_requested_at=lease.cancel_requested_at,
            mutation_id=lease.mutation_id,
            recovered=lease.recovered,
        )

    async def finalize_exhausted(
        self,
        candidate: ExhaustedJob,
        action: Callable[[object], Awaitable[None]] | None = None,
    ) -> None:
        del candidate
        self._fence()
        self.exhausted_finalizations += 1
        if action is not None:
            await action(self)

    def _fence(self) -> None:
        if self.lost:
            raise LostLeaseError

    async def heartbeat(self, lease: JobLease, lease_duration: timedelta) -> JobLease:
        del lease_duration
        self._fence()
        if self.heartbeat_error is not None:
            raise self.heartbeat_error
        self.heartbeats += 1
        return lease

    async def cancellation_requested(self, lease: JobLease) -> bool:
        del lease
        self._fence()
        return self.cancel_requested

    async def release_claim(self, lease: JobLease) -> None:
        self._fence()
        self.available = lease

    async def checkpoint(
        self,
        lease: JobLease,
        *,
        stage: str | None,
        resume_stage: str | None,
        progress_current: int,
        progress_total: int | None,
    ) -> None:
        del lease
        self._fence()
        self.checkpoints.append((stage, resume_stage, progress_current, progress_total))

    async def prepare_activation[Result](
        self,
        lease: JobLease,
        action: Callable[[object], Awaitable[Result]],
    ) -> Result:
        del lease
        self._fence()
        self.activation_preparations += 1
        return await action(self)

    async def finalize_domain[Result](
        self,
        lease: JobLease,
        action: Callable[[object], Awaitable[Result]],
        *,
        terminal_status_observer: Callable[[str], None] | None = None,
    ) -> Result:
        del lease
        self.domain_finalizations += 1
        result = await action(self)
        if terminal_status_observer is not None:
            terminal_status_observer(getattr(self, "domain_finalized_status", "succeeded"))
        return result

    async def commit_stage_facts[Result](
        self,
        lease: JobLease,
        action: Callable[[object], Awaitable[Result]],
    ) -> Result:
        del lease
        self._fence()
        self.stage_fact_commits += 1
        return await action(self)

    async def commit_stage_checkpoint[Result](
        self,
        lease: JobLease,
        action: Callable[[object], Awaitable[Result]],
        *,
        progress_current: int,
        progress_total: int | None,
    ) -> tuple[Result, JobLease]:
        self._fence()
        result = await action(self)
        self.stage_checkpoints += 1
        return (
            result,
            JobLease(
                id=lease.id,
                operation=lease.operation,
                target_type=lease.target_type,
                target_id=lease.target_id,
                target_revision=lease.target_revision,
                index_generation_id=lease.index_generation_id,
                stage=lease.stage,
                resume_stage=lease.resume_stage,
                progress_current=progress_current,
                progress_total=progress_total,
                attempt_count=lease.attempt_count,
                max_attempts=lease.max_attempts,
                lease_owner=lease.lease_owner,
                lease_epoch=lease.lease_epoch,
                lease_expires_at=lease.lease_expires_at,
                cancel_requested_at=lease.cancel_requested_at,
            ),
        )

    async def advance_stage[Result](
        self,
        lease: JobLease,
        action: Callable[[object], Awaitable[Result]],
        *,
        stage: str,
        resume_stage: str,
        progress_current: int,
        progress_total: int | None,
    ) -> tuple[Result, JobLease]:
        self._fence()
        result = await action(self)
        self.stage_advances += 1
        return (
            result,
            JobLease(
                id=lease.id,
                operation=lease.operation,
                target_type=lease.target_type,
                target_id=lease.target_id,
                target_revision=lease.target_revision,
                index_generation_id=lease.index_generation_id,
                stage=stage,
                resume_stage=resume_stage,
                progress_current=progress_current,
                progress_total=progress_total,
                attempt_count=lease.attempt_count,
                max_attempts=lease.max_attempts,
                lease_owner=lease.lease_owner,
                lease_epoch=lease.lease_epoch,
                lease_expires_at=lease.lease_expires_at,
                cancel_requested_at=lease.cancel_requested_at,
            ),
        )

    async def mark_succeeded(self, lease: JobLease) -> str:
        del lease
        self._fence()
        if self.cancel_requested:
            self.cancellations += 1
            return "cancelled"
        self.successes += 1
        return "succeeded"

    async def mark_cancelled(self, lease: JobLease) -> None:
        del lease
        self._fence()
        self.cancellations += 1

    async def record_failure(
        self,
        lease: JobLease,
        *,
        retryable: bool,
        error_code: str,
        error_message: str,
        retry_delay: timedelta,
    ) -> str:
        del lease
        self._fence()
        self.failures.append((retryable, error_code, error_message, retry_delay))
        return "retry_wait" if retryable else "failed"


def _repository_context(
    repository: _MemoryJobRepository,
) -> RepositoryContextFactory:
    @asynccontextmanager
    async def context() -> AsyncIterator[_MemoryJobRepository]:
        yield repository

    return cast(RepositoryContextFactory, context)


def _runner(
    repository: _MemoryJobRepository,
    *,
    heartbeat_seconds: float = 0.01,
    shutdown_seconds: float = 0.1,
    metrics: OperationalMetrics | object | None = None,
) -> JobRunner:
    kwargs: dict[str, object] = {}
    if metrics is not None:
        kwargs["metrics"] = metrics
    return JobRunner(
        repository_context=_repository_context(repository),
        lease_owner="worker-a",
        lease_seconds=1,
        heartbeat_seconds=heartbeat_seconds,
        poll_interval_seconds=0.01,
        max_concurrency=2,
        shutdown_seconds=shutdown_seconds,
        backoff=ExponentialBackoff(
            initial_seconds=5,
            maximum_seconds=20,
            jitter_fraction=0,
        ),
        **kwargs,  # type: ignore[arg-type]
    )


@pytest.mark.asyncio
async def test_job_observability_uses_real_transitions_and_recovery_signal() -> None:
    generation_id = uuid4()
    lease = replace(
        _lease(),
        index_generation_id=generation_id,
        recovered=True,
    )
    repository = _MemoryJobRepository(lease)
    metrics = OperationalMetrics()
    handler = _CollectingJobLogHandler()
    previous_handlers = list(jobs_runner.logger.handlers)
    previous_propagate = jobs_runner.logger.propagate
    previous_level = jobs_runner.logger.level
    jobs_runner.logger.handlers = [handler]
    jobs_runner.logger.propagate = False
    jobs_runner.logger.setLevel(logging.INFO)
    runner = _runner(repository, metrics=metrics)

    async def succeed(_context: JobExecutionContext) -> None:
        return None

    runner.register("ingest_document", succeed)
    try:
        assert await runner.run_once() is True
    finally:
        jobs_runner.logger.handlers = previous_handlers
        jobs_runner.logger.propagate = previous_propagate
        jobs_runner.logger.setLevel(previous_level)

    assert (
        metrics.registry.get_sample_value("rag_job_state_transitions_total", {"state": "running"})
        == 1
    )
    assert (
        metrics.registry.get_sample_value("rag_job_state_transitions_total", {"state": "succeeded"})
        == 1
    )
    assert metrics.registry.get_sample_value("rag_job_lease_recoveries_total") == 1
    assert [record.msg for record in handler.records] == [
        "job.state.changed",
        "job.lease.recovered",
        "job.state.changed",
    ]
    for record in handler.records:
        assert type(record) is logging.LogRecord
        assert record.__dict__["job_id"] == str(lease.id)
        assert record.__dict__["generation_id"] == str(generation_id)
        assert record.__dict__["version_id"] == str(lease.target_id)
        rendered = repr(record.__dict__)
        assert "query" not in rendered
        assert "Authorization" not in rendered


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "expected_state"),
    [
        ("retry", "retry_wait"),
        ("failure", "failed"),
        ("cancel", "cancelled"),
    ],
)
async def test_job_observability_records_the_repository_terminal_state(
    mode: str,
    expected_state: str,
) -> None:
    repository = _MemoryJobRepository(_lease())
    repository.cancel_requested = mode == "cancel"
    metrics = OperationalMetrics()
    runner = _runner(repository, metrics=metrics)

    async def handler(_context: JobExecutionContext) -> None:
        if mode == "retry":
            raise RetryableJobError("PROVIDER_TIMEOUT", "raw-query-traceback-secret")
        if mode == "failure":
            raise PermanentJobError("JOB_HANDLER_FAILED", "raw-query-traceback-secret")

    runner.register("ingest_document", handler)

    assert await runner.run_once() is True
    assert (
        metrics.registry.get_sample_value("rag_job_state_transitions_total", {"state": "running"})
        == 1
    )
    assert (
        metrics.registry.get_sample_value(
            "rag_job_state_transitions_total", {"state": expected_state}
        )
        == 1
    )


@pytest.mark.asyncio
async def test_job_observability_records_release_and_exhaustion_only_after_success() -> None:
    metrics = OperationalMetrics()
    queued_repository = _MemoryJobRepository(None)
    queued_runner = _runner(queued_repository, metrics=metrics)
    await queued_runner._release_claim(_lease())

    exhausted_repository = _MemoryJobRepository(_exhausted_job())
    exhausted_runner = _runner(exhausted_repository, metrics=metrics)
    assert await exhausted_runner.run_once() is True

    assert (
        metrics.registry.get_sample_value("rag_job_state_transitions_total", {"state": "queued"})
        == 1
    )
    assert (
        metrics.registry.get_sample_value("rag_job_state_transitions_total", {"state": "failed"})
        == 1
    )


@pytest.mark.asyncio
async def test_job_observability_failures_do_not_change_claim_or_finalization() -> None:
    class FailingMetrics:
        def record_job_state(self, **kwargs: object) -> None:
            del kwargs
            raise BaseException("metrics-secret")

        def record_lease_recovery(self) -> None:
            raise BaseException("metrics-secret")

    class FailingHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            del record
            raise BaseException("logging-secret")

    repository = _MemoryJobRepository(_lease())
    runner = _runner(repository, metrics=FailingMetrics())
    previous_handlers = list(jobs_runner.logger.handlers)
    previous_propagate = jobs_runner.logger.propagate
    previous_level = jobs_runner.logger.level
    jobs_runner.logger.handlers = [FailingHandler()]
    jobs_runner.logger.propagate = False
    jobs_runner.logger.setLevel(logging.INFO)

    async def succeed(_context: JobExecutionContext) -> None:
        return None

    runner.register("ingest_document", succeed)
    try:
        assert await runner.run_once() is True
    finally:
        jobs_runner.logger.handlers = previous_handlers
        jobs_runner.logger.propagate = previous_propagate
        jobs_runner.logger.setLevel(previous_level)

    assert repository.successes == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_failure", (False, True))
async def test_provider_retry_is_counted_only_for_real_provider_retry_wait(
    provider_failure: bool,
) -> None:
    repository = _MemoryJobRepository(_lease())
    metrics = OperationalMetrics()
    runner = _runner(repository, metrics=metrics)

    async def retry(_context: JobExecutionContext) -> None:
        if provider_failure:
            raise RetryableProviderJobError(
                "PROVIDER_TIMEOUT",
                "provider-query-secret",
                provider_type="openrouter",
            )
        raise RetryableJobError("OBJECT_STORE_UNAVAILABLE", "object-key-secret")

    runner.register("ingest_document", retry)

    assert await runner.run_once() is True
    assert (
        metrics.registry.get_sample_value(
            "rag_job_state_transitions_total", {"state": "retry_wait"}
        )
        == 1
    )
    assert metrics.registry.get_sample_value(
        "rag_provider_retries_total", {"provider_type": "openrouter"}
    ) == (1 if provider_failure else None)


def test_exponential_backoff_is_capped_and_jitter_stays_in_bounds() -> None:
    minimum = ExponentialBackoff(
        initial_seconds=5,
        maximum_seconds=20,
        jitter_fraction=0.2,
        random_fraction=lambda: 0,
    )
    maximum = ExponentialBackoff(
        initial_seconds=5,
        maximum_seconds=20,
        jitter_fraction=0.2,
        random_fraction=lambda: 1,
    )

    assert minimum.delay(1) == timedelta(seconds=4)
    assert maximum.delay(1) == timedelta(seconds=6)
    assert minimum.delay(20) == timedelta(seconds=16)
    assert maximum.delay(20) == timedelta(seconds=20)


@pytest.mark.asyncio
async def test_redis_notification_contains_only_the_job_uuid_and_is_best_effort() -> None:
    job_id = uuid4()
    working_redis = _PublishingRedis()
    failed_redis = _PublishingRedis(ConnectionError("redis unavailable"))

    assert await RedisJobNotifier(working_redis).notify(job_id) is True
    assert working_redis.calls == [(JOB_NOTIFICATION_CHANNEL, str(job_id))]
    assert await RedisJobNotifier(failed_redis).notify(job_id) is False


@pytest.mark.asyncio
async def test_duplicate_notification_is_harmless_and_polling_recovers_lost_notification() -> None:
    first_repository = _MemoryJobRepository(_lease())
    first_runner = _runner(first_repository)
    first_runner.register("ingest_document", lambda _context: asyncio.sleep(0))
    assert first_repository.available is not None
    job_id = first_repository.available.id

    assert await first_runner.run_once(job_id) is True
    assert await first_runner.run_once(job_id) is False
    assert first_repository.successes == 1

    polling_repository = _MemoryJobRepository(_lease())
    polling_runner = _runner(polling_repository)
    polling_runner.register("ingest_document", lambda _context: asyncio.sleep(0))

    assert await polling_runner.poll_once() is True
    assert polling_repository.successes == 1


@pytest.mark.asyncio
async def test_notification_connection_failure_degrades_to_paced_database_polling() -> None:
    stop_event = asyncio.Event()

    class DelayedRepository(_MemoryJobRepository):
        async def claim_next(
            self,
            *,
            lease_owner: str,
            lease_duration: timedelta,
            job_id: UUID | None = None,
        ) -> JobLease | ExhaustedJob | None:
            if self.claim_calls == 0:
                self.claim_calls += 1
                return None
            return await super().claim_next(
                lease_owner=lease_owner,
                lease_duration=lease_duration,
                job_id=job_id,
            )

        async def mark_succeeded(self, lease: JobLease) -> str:
            status = await super().mark_succeeded(lease)
            stop_event.set()
            return status

    class UnavailableNotifications:
        def __init__(self) -> None:
            self.calls = 0

        async def get(self, timeout_seconds: float) -> UUID | None:
            del timeout_seconds
            self.calls += 1
            raise ConnectionError("redis unavailable")

    repository = DelayedRepository(_lease())
    notifications = UnavailableNotifications()
    runner = _runner(repository)
    runner.register("ingest_document", lambda _context: asyncio.sleep(0))

    await asyncio.wait_for(
        runner.run(stop_event, notifications=notifications),
        timeout=0.3,
    )

    assert repository.successes == 1
    assert repository.claim_calls == 2
    assert notifications.calls == 1


@pytest.mark.asyncio
async def test_stop_interrupts_a_notification_source_that_ignores_its_timeout() -> None:
    repository = _MemoryJobRepository(None)
    runner = _runner(repository)
    runner.register("ingest_document", lambda _context: asyncio.sleep(0))
    stop_event = asyncio.Event()

    class BlockingNotifications:
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.cancelled = asyncio.Event()

        async def get(self, timeout_seconds: float) -> UUID | None:
            del timeout_seconds
            self.started.set()
            try:
                await asyncio.Event().wait()
            finally:
                self.cancelled.set()
            return None

    notifications = BlockingNotifications()
    runner_task = asyncio.create_task(runner.run(stop_event, notifications=notifications))
    await asyncio.wait_for(notifications.started.wait(), timeout=0.2)

    stop_event.set()
    await asyncio.wait_for(runner_task, timeout=0.2)

    assert notifications.cancelled.is_set()
    assert repository.claim_calls == 1


@pytest.mark.asyncio
async def test_notification_source_that_ignores_timeout_cannot_block_database_polling() -> None:
    second_poll = asyncio.Event()

    class Repository(_MemoryJobRepository):
        async def claim_next(
            self,
            *,
            lease_owner: str,
            lease_duration: timedelta,
            job_id: UUID | None = None,
        ) -> JobLease | ExhaustedJob | None:
            lease = await super().claim_next(
                lease_owner=lease_owner,
                lease_duration=lease_duration,
                job_id=job_id,
            )
            if self.claim_calls >= 2:
                second_poll.set()
            return lease

    repository = Repository(None)
    runner = _runner(repository)
    runner.register("ingest_document", lambda _context: asyncio.sleep(0))
    stop_event = asyncio.Event()
    cancellations = 0

    class BlockingNotifications:
        async def get(self, timeout_seconds: float) -> UUID | None:
            nonlocal cancellations
            del timeout_seconds
            try:
                await asyncio.Event().wait()
            finally:
                cancellations += 1
            return None

    runner_task = asyncio.create_task(runner.run(stop_event, notifications=BlockingNotifications()))
    await asyncio.wait_for(second_poll.wait(), timeout=0.2)
    stop_event.set()
    await asyncio.wait_for(runner_task, timeout=0.2)

    assert cancellations >= 1


@pytest.mark.asyncio
async def test_cancellation_resistant_notification_read_is_single_flight() -> None:
    fourth_poll = asyncio.Event()

    class Repository(_MemoryJobRepository):
        async def claim_next(
            self,
            *,
            lease_owner: str,
            lease_duration: timedelta,
            job_id: UUID | None = None,
        ) -> JobLease | ExhaustedJob | None:
            lease = await super().claim_next(
                lease_owner=lease_owner,
                lease_duration=lease_duration,
                job_id=job_id,
            )
            if self.claim_calls >= 4:
                fourth_poll.set()
            return lease

    class ResistantNotifications:
        def __init__(self) -> None:
            self.calls = 0
            self.active = 0
            self.peak_active = 0
            self.cancelled = asyncio.Event()
            self.release = asyncio.Event()
            self.all_finished = asyncio.Event()

        async def get(self, timeout_seconds: float) -> UUID | None:
            del timeout_seconds
            self.calls += 1
            self.active += 1
            self.peak_active = max(self.peak_active, self.active)
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled.set()
                await self.release.wait()
            finally:
                self.active -= 1
                if self.active == 0:
                    self.all_finished.set()
            return None

    repository = Repository(None)
    runner = _runner(repository)
    runner.register("ingest_document", lambda _context: asyncio.sleep(0))
    stop_event = asyncio.Event()
    notifications = ResistantNotifications()
    runner_task = asyncio.create_task(runner.run(stop_event, notifications=notifications))

    await asyncio.wait_for(notifications.cancelled.wait(), timeout=0.2)
    await asyncio.wait_for(fourth_poll.wait(), timeout=0.2)
    stop_event.set()
    done, _pending = await asyncio.wait({runner_task}, timeout=0.12)
    returned_before_release = runner_task in done
    notifications.release.set()
    await asyncio.wait_for(notifications.all_finished.wait(), timeout=0.2)
    await asyncio.wait_for(runner_task, timeout=0.2)

    assert returned_before_release is True
    assert repository.claim_calls >= 4
    assert notifications.calls == 1
    assert notifications.peak_active == 1


@pytest.mark.asyncio
async def test_double_cancellation_does_not_wait_for_resistant_notification_cleanup() -> None:
    class ResistantNotifications:
        def __init__(self) -> None:
            self.calls = 0
            self.started = asyncio.Event()
            self.cancelled = asyncio.Event()
            self.release = asyncio.Event()
            self.finished = asyncio.Event()

        async def get(self, timeout_seconds: float) -> UUID | None:
            del timeout_seconds
            self.calls += 1
            self.started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled.set()
                await self.release.wait()
                raise RuntimeError("notification cleanup failure") from None
            finally:
                self.finished.set()
            return None

    repository = _MemoryJobRepository(None)
    runner = _runner(repository)
    runner.register("ingest_document", lambda _context: asyncio.sleep(0))
    stop_event = asyncio.Event()
    notifications = ResistantNotifications()
    runner_task = asyncio.create_task(runner.run(stop_event, notifications=notifications))
    await asyncio.wait_for(notifications.started.wait(), timeout=0.2)

    runner_task.cancel()
    runner_task.cancel()
    done, _pending = await asyncio.wait({runner_task}, timeout=0.12)
    returned_before_release = runner_task in done
    with pytest.raises(asyncio.CancelledError):
        await runner_task

    notifications.release.set()
    await asyncio.wait_for(notifications.finished.wait(), timeout=0.2)
    await asyncio.sleep(0)

    assert returned_before_release is True
    assert notifications.cancelled.is_set()
    assert notifications.calls == 1


@pytest.mark.asyncio
async def test_stop_rolls_back_an_in_flight_claim_before_handler_admission() -> None:
    claim_started = asyncio.Event()
    release_claim = asyncio.Event()

    class DelayedClaimRepository(_MemoryJobRepository):
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

    repository = DelayedClaimRepository(_lease())
    runner = _runner(repository)
    stop_event = asyncio.Event()
    handler_started = False

    async def handler(_context: JobExecutionContext) -> None:
        nonlocal handler_started
        handler_started = True

    runner.register("ingest_document", handler)
    runner_task = asyncio.create_task(runner.run(stop_event))
    await asyncio.wait_for(claim_started.wait(), timeout=0.2)

    stop_event.set()
    release_claim.set()
    await asyncio.wait_for(runner_task, timeout=0.2)

    assert handler_started is False
    assert repository.successes == 0


@pytest.mark.asyncio
async def test_stop_during_initial_cancellation_check_requeues_before_handler_start() -> None:
    cancellation_check_started = asyncio.Event()
    release_cancellation_check = asyncio.Event()

    class DelayedCancellationRepository(_MemoryJobRepository):
        async def cancellation_requested(self, lease: JobLease) -> bool:
            cancellation_check_started.set()
            await release_cancellation_check.wait()
            return await super().cancellation_requested(lease)

    repository = DelayedCancellationRepository(_lease())
    runner = _runner(repository)
    stop_event = asyncio.Event()
    handler_started = False

    async def handler(_context: JobExecutionContext) -> None:
        nonlocal handler_started
        handler_started = True

    runner.register("ingest_document", handler)
    runner_task = asyncio.create_task(runner.run(stop_event))
    await asyncio.wait_for(cancellation_check_started.wait(), timeout=0.2)

    stop_event.set()
    release_cancellation_check.set()
    await asyncio.wait_for(runner_task, timeout=0.2)

    assert handler_started is False
    assert repository.successes == 0
    assert repository.available is not None


@pytest.mark.asyncio
async def test_long_running_handler_is_heartbeated_and_can_checkpoint() -> None:
    repository = _MemoryJobRepository(_lease())
    runner = _runner(repository)

    async def handler(context: JobExecutionContext) -> None:
        await context.checkpoint(
            stage="embed_index",
            resume_stage="embed_index",
            progress_current=10,
            progress_total=20,
        )
        await asyncio.sleep(0.035)

    runner.register("ingest_document", handler)

    assert await runner.poll_once() is True
    assert repository.checkpoints == [("embed_index", "embed_index", 10, 20)]
    assert repository.heartbeats >= 2
    assert repository.successes == 1


@pytest.mark.asyncio
async def test_stage_checkpoint_commits_facts_and_progress_without_counting_as_advance() -> None:
    repository = _MemoryJobRepository(_lease())
    assert isinstance(repository.available, JobLease)
    context = JobExecutionContext(
        _repository_context(repository),
        repository.available,
        timedelta(seconds=30),
    )
    committed: list[str] = []

    async def action(_session: object) -> None:
        committed.append("facts")

    await context.commit_stage_checkpoint(
        action,
        progress_current=2,
        progress_total=5,
    )

    assert committed == ["facts"]
    assert context.lease.stage == "parse"
    assert context.lease.progress_current == 2
    assert context.lease.progress_total == 5
    assert context.stage_advance_count == 0
    assert repository.stage_checkpoints == 1


@pytest.mark.asyncio
async def test_continued_stages_share_one_claim_attempt_and_only_complete_marks_success() -> None:
    repository = _MemoryJobRepository(_lease(attempt_count=2))
    runner = _runner(repository)
    seen_stages: list[str | None] = []

    async def handler(context: JobExecutionContext) -> JobHandlerOutcome:
        seen_stages.append(context.lease.stage)
        if context.lease.stage == "parse":

            async def persist_parse(_session: object) -> None:
                return None

            await context.advance_stage(
                persist_parse,
                stage="chunk",
                resume_stage="chunk",
                progress_current=0,
                progress_total=None,
            )
            return JobHandlerOutcome.CONTINUE
        if context.lease.stage == "chunk":

            async def persist_chunk(_session: object) -> None:
                return None

            await context.advance_stage(
                persist_chunk,
                stage="embed_index",
                resume_stage="embed_index",
                progress_current=0,
                progress_total=3,
            )
            return JobHandlerOutcome.CONTINUE
        return JobHandlerOutcome.COMPLETE

    runner.register("ingest_document", handler)

    assert await runner.poll_once() is True
    assert repository.claim_calls == 1
    assert repository.stage_advances == 2
    assert seen_stages == ["parse", "chunk", "embed_index"]
    assert repository.successes == 1
    assert repository.available is None


@pytest.mark.asyncio
async def test_cancel_request_cancels_the_handler_and_marks_the_job_cancelled() -> None:
    repository = _MemoryJobRepository(_lease())
    runner = _runner(repository)
    handler_cancelled = asyncio.Event()

    async def handler(_context: JobExecutionContext) -> None:
        repository.cancel_requested = True
        try:
            await asyncio.Event().wait()
        finally:
            handler_cancelled.set()

    runner.register("ingest_document", handler)

    assert await runner.poll_once() is True
    assert handler_cancelled.is_set()
    assert repository.cancellations == 1
    assert repository.successes == 0


@pytest.mark.asyncio
async def test_cancel_request_present_at_claim_prevents_the_handler_from_starting() -> None:
    repository = _MemoryJobRepository(_lease())
    repository.cancel_requested = True
    runner = _runner(repository)
    handler_started = False

    async def handler(_context: JobExecutionContext) -> None:
        nonlocal handler_started
        handler_started = True

    runner.register("ingest_document", handler)

    assert await runner.poll_once() is True
    assert handler_started is False
    assert repository.cancellations == 1
    assert repository.successes == 0


@pytest.mark.asyncio
async def test_retryable_and_permanent_failures_are_classified_safely() -> None:
    retry_repository = _MemoryJobRepository(_lease(attempt_count=2))
    retry_runner = _runner(retry_repository)

    async def retryable(_context: JobExecutionContext) -> None:
        raise RetryableJobError("PROVIDER_TIMEOUT", "Provider request timed out")

    retry_runner.register("ingest_document", retryable)
    assert await retry_runner.poll_once() is True
    assert retry_repository.failures == [
        (
            True,
            "PROVIDER_TIMEOUT",
            "Provider request timed out",
            timedelta(seconds=10),
        )
    ]

    permanent_repository = _MemoryJobRepository(_lease())
    permanent_runner = _runner(permanent_repository)

    async def permanent(_context: JobExecutionContext) -> None:
        raise PermanentJobError("INVALID_SOURCE", "Source document is invalid")

    permanent_runner.register("ingest_document", permanent)
    assert await permanent_runner.poll_once() is True
    assert permanent_repository.failures == [
        (False, "INVALID_SOURCE", "Source document is invalid", timedelta(0))
    ]


@pytest.mark.asyncio
async def test_lost_lease_cancels_long_running_work_without_a_terminal_write() -> None:
    repository = _MemoryJobRepository(_lease())
    runner = _runner(repository)
    handler_cancelled = asyncio.Event()

    async def handler(_context: JobExecutionContext) -> None:
        repository.lost = True
        try:
            await asyncio.Event().wait()
        finally:
            handler_cancelled.set()

    runner.register("ingest_document", handler)

    assert await runner.poll_once() is False
    assert handler_cancelled.is_set()
    assert repository.cancellations == 0
    assert repository.successes == 0
    assert repository.failures == []


@pytest.mark.asyncio
async def test_heartbeat_dependency_failure_stops_work_without_a_terminal_write() -> None:
    repository = _MemoryJobRepository(_lease())
    repository.heartbeat_error = RuntimeError("database unavailable")
    runner = _runner(repository)
    handler_cancelled = asyncio.Event()

    async def handler(_context: JobExecutionContext) -> None:
        try:
            await asyncio.Event().wait()
        finally:
            handler_cancelled.set()

    runner.register("ingest_document", handler)

    assert await asyncio.wait_for(runner.poll_once(), timeout=0.2) is False
    assert handler_cancelled.is_set()
    assert repository.cancellations == 0
    assert repository.successes == 0
    assert repository.failures == []


@pytest.mark.asyncio
async def test_activation_callback_runs_inside_the_fenced_repository_context() -> None:
    repository = _MemoryJobRepository(_lease())
    assert isinstance(repository.available, JobLease)
    context = JobExecutionContext(
        _repository_context(repository),
        repository.available,
        timedelta(seconds=1),
    )
    callback_repository: object | None = None

    async def activate(active_repository: object) -> str:
        nonlocal callback_repository
        callback_repository = active_repository
        return "activated"

    assert await context.prepare_activation(activate) == "activated"
    assert callback_repository is repository
    assert repository.activation_preparations == 1


@pytest.mark.asyncio
async def test_domain_finalization_is_terminal_and_runner_does_not_mark_success_twice() -> None:
    repository = _MemoryJobRepository(_lease())
    runner = _runner(repository)
    committed: list[str] = []

    async def handler(context: JobExecutionContext) -> JobHandlerOutcome:
        async def finalize(_session: object) -> str:
            committed.append("atomic-domain-and-job")
            return "done"

        assert await context.finalize_domain(finalize) == "done"
        return JobHandlerOutcome.COMPLETE

    runner.register("ingest_document", handler)

    assert await runner.poll_once() is True
    assert committed == ["atomic-domain-and-job"]
    assert repository.domain_finalizations == 1
    assert repository.successes == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal_state", ("succeeded", "failed", "cancelled"))
async def test_domain_finalization_observes_the_repository_terminal_state(
    terminal_state: str,
) -> None:
    repository = _MemoryJobRepository(_lease())
    repository.domain_finalized_status = terminal_state
    metrics = OperationalMetrics()
    runner = _runner(repository, metrics=metrics)

    async def handler(context: JobExecutionContext) -> JobHandlerOutcome:
        async def finalize(_session: object) -> None:
            return None

        await context.finalize_domain(finalize)
        return JobHandlerOutcome.COMPLETE

    runner.register("ingest_document", handler)

    assert await runner.poll_once() is True
    assert (
        metrics.registry.get_sample_value(
            "rag_job_state_transitions_total", {"state": terminal_state}
        )
        == 1
    )
    assert repository.successes == 0


@pytest.mark.asyncio
async def test_worker_registration_preserves_exhaustion_finalizer_without_running_handler() -> None:
    candidate = _exhausted_job()
    repository = _MemoryJobRepository(candidate)
    runner = _runner(repository)
    handler_calls = 0
    finalizer_calls: list[ExhaustedJob] = []

    async def handler(_context: JobExecutionContext) -> JobHandlerOutcome:
        nonlocal handler_calls
        handler_calls += 1
        return JobHandlerOutcome.COMPLETE

    async def finalize(exhausted: ExhaustedJob, _session: AsyncSession) -> None:
        finalizer_calls.append(exhausted)

    jobs_worker._register_handlers(
        runner,
        {
            "ingest_document": jobs_worker.JobHandlerRegistration(
                handler,
                finalize,
            )
        },
    )

    assert await runner.run_once(candidate.id) is True
    assert handler_calls == 0
    assert finalizer_calls == [candidate]
    assert repository.exhausted_finalizations == 1
    assert repository.successes == 0


@pytest.mark.asyncio
async def test_exhausted_snapshot_without_operation_finalizer_uses_generic_job_cas() -> None:
    candidate = _exhausted_job()
    repository = _MemoryJobRepository(candidate)
    runner = _runner(repository)

    assert await runner.run_once(candidate.id) is True
    assert repository.exhausted_finalizations == 1
    assert repository.successes == 0


@pytest.mark.asyncio
async def test_heartbeat_lost_lease_during_successful_finalization_does_not_cancel() -> None:
    repository = _MemoryJobRepository(_lease())
    repository.heartbeat_error = LostLeaseError()
    runner = _runner(repository, heartbeat_seconds=0.001)
    finalization_started = asyncio.Event()
    release_finalization = asyncio.Event()

    async def handler(context: JobExecutionContext) -> JobHandlerOutcome:
        async def finalize(_session: object) -> None:
            finalization_started.set()
            await release_finalization.wait()

        finalization = asyncio.create_task(context.finalize_domain(finalize))
        await finalization_started.wait()
        await asyncio.sleep(0.01)
        release_finalization.set()
        await finalization
        return JobHandlerOutcome.COMPLETE

    runner.register("ingest_document", handler)

    assert await runner.poll_once() is True
    assert repository.domain_finalizations == 1
    assert repository.successes == 0
    assert repository.failures == []


def test_worker_health_file_is_local_bounded_and_fails_closed_when_stale(tmp_path: Path) -> None:
    path = tmp_path / "worker-health.json"
    now = datetime.now(UTC)
    snapshot = WorkerHealthSnapshot(
        pid=123,
        process_running=True,
        dependencies_ok=True,
        accepting_jobs=True,
        checked_at=now,
    )

    write_worker_health(path, snapshot)

    assert load_worker_health(path, now=now, max_age=timedelta(seconds=10)) == snapshot
    assert (
        load_worker_health(
            path,
            now=now + timedelta(seconds=11),
            max_age=timedelta(seconds=10),
        )
        is None
    )
    assert path.stat().st_size < 1024


@pytest.mark.asyncio
async def test_worker_marks_stale_health_unhealthy_before_loading_settings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "worker-health.json"
    write_worker_health(
        path,
        WorkerHealthSnapshot(
            pid=os.getpid(),
            process_running=True,
            dependencies_ok=True,
            accepting_jobs=True,
            checked_at=datetime.now(UTC),
        ),
    )

    class BrokenSettings:
        def __init__(self) -> None:
            raise RuntimeError("settings failed")

    monkeypatch.setattr(jobs_worker, "Settings", BrokenSettings)

    with pytest.raises(RuntimeError, match="^settings failed$"):
        await jobs_worker._run_worker(path, 1)

    snapshot = load_worker_health(
        path,
        now=datetime.now(UTC),
        max_age=timedelta(seconds=10),
    )
    assert snapshot is not None
    assert snapshot.pid == os.getpid()
    assert snapshot.process_running is True
    assert snapshot.dependencies_ok is False
    assert snapshot.accepting_jobs is False


@pytest.mark.asyncio
async def test_dependency_preflight_failure_prevents_claiming_new_work() -> None:
    repository = _MemoryJobRepository(_lease())

    async def unavailable() -> bool:
        return False

    runner = JobRunner(
        repository_context=_repository_context(repository),
        lease_owner="worker-a",
        lease_seconds=1,
        heartbeat_seconds=0.01,
        poll_interval_seconds=0.01,
        max_concurrency=2,
        backoff=ExponentialBackoff(initial_seconds=1, maximum_seconds=2),
        preflight=unavailable,
    )
    runner.register("ingest_document", lambda _context: asyncio.sleep(0))

    assert await runner.poll_once() is False
    assert repository.claim_calls == 0
    assert repository.available is not None


@pytest.mark.asyncio
async def test_run_loop_polls_without_notifications_and_never_exceeds_its_concurrency_bound() -> (
    None
):
    stop_event = asyncio.Event()

    class QueueRepository(_MemoryJobRepository):
        def __init__(self) -> None:
            super().__init__(None)
            self.queue = [_lease() for _ in range(4)]

        async def claim_next(
            self,
            *,
            lease_owner: str,
            lease_duration: timedelta,
            job_id: UUID | None = None,
        ) -> JobLease | ExhaustedJob | None:
            del lease_duration
            self.claim_calls += 1
            if job_id is not None:
                for index, lease in enumerate(self.queue):
                    if lease.id == job_id:
                        return self.queue.pop(index)
                return None
            return self.queue.pop(0) if self.queue else None

        async def mark_succeeded(self, lease: JobLease) -> str:
            status = await super().mark_succeeded(lease)
            if self.successes == 4:
                stop_event.set()
            return status

    repository = QueueRepository()
    runner = JobRunner(
        repository_context=_repository_context(repository),
        lease_owner="worker-a",
        lease_seconds=1,
        heartbeat_seconds=0.01,
        poll_interval_seconds=0.001,
        max_concurrency=2,
        backoff=ExponentialBackoff(initial_seconds=1, maximum_seconds=2),
    )
    active = 0
    peak = 0

    async def handler(_context: JobExecutionContext) -> None:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        try:
            await asyncio.sleep(0.02)
        finally:
            active -= 1

    runner.register("ingest_document", handler)

    await asyncio.wait_for(runner.run(stop_event), timeout=1)

    assert repository.successes == 4
    assert repository.claim_calls >= 4
    assert peak == 2


@pytest.mark.asyncio
async def test_shutdown_waits_for_active_work_before_forcing_cancellation() -> None:
    repository = _MemoryJobRepository(_lease())
    runner = _runner(repository, shutdown_seconds=0.1)
    stop_event = asyncio.Event()
    handler_cancelled = False

    async def handler(_context: JobExecutionContext) -> None:
        nonlocal handler_cancelled
        stop_event.set()
        try:
            await asyncio.sleep(0.02)
        except asyncio.CancelledError:
            handler_cancelled = True
            raise

    runner.register("ingest_document", handler)

    await asyncio.wait_for(runner.run(stop_event), timeout=0.5)

    assert handler_cancelled is False
    assert repository.successes == 1


@pytest.mark.asyncio
async def test_shutdown_timeout_wakes_a_fully_saturated_runner() -> None:
    class SaturatedRepository(_MemoryJobRepository):
        def __init__(self) -> None:
            super().__init__(None)
            self.queue = [_lease(), _lease()]

        async def claim_next(
            self,
            *,
            lease_owner: str,
            lease_duration: timedelta,
            job_id: UUID | None = None,
        ) -> JobLease | ExhaustedJob | None:
            del lease_owner, lease_duration, job_id
            self.claim_calls += 1
            return self.queue.pop(0) if self.queue else None

    repository = SaturatedRepository()
    runner = _runner(repository, shutdown_seconds=0.05)
    stop_event = asyncio.Event()
    both_started = asyncio.Event()
    cancelled = 0
    active = 0

    async def handler(_context: JobExecutionContext) -> None:
        nonlocal active, cancelled
        active += 1
        if active == 2:
            both_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled += 1

    runner.register("ingest_document", handler)
    runner_task = asyncio.create_task(runner.run(stop_event))
    await asyncio.wait_for(both_started.wait(), timeout=0.2)

    stop_event.set()
    await asyncio.wait_for(runner_task, timeout=0.2)

    assert cancelled == 2
    assert repository.successes == 0


@pytest.mark.asyncio
async def test_shutdown_hard_bound_detaches_a_cancellation_resistant_handler() -> None:
    repository = _MemoryJobRepository(_lease())
    runner = _runner(repository, shutdown_seconds=0.03)
    stop_event = asyncio.Event()
    handler_started = asyncio.Event()
    cancellation_seen = asyncio.Event()
    release_handler = asyncio.Event()
    handler_finished = asyncio.Event()

    async def handler(_context: JobExecutionContext) -> None:
        handler_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancellation_seen.set()
            await release_handler.wait()
        finally:
            handler_finished.set()

    runner.register("ingest_document", handler)
    runner_task = asyncio.create_task(runner.run(stop_event))
    await asyncio.wait_for(handler_started.wait(), timeout=0.2)
    stop_event.set()

    done, _pending = await asyncio.wait({runner_task}, timeout=0.12)
    returned_within_bound = runner_task in done
    release_handler.set()
    await asyncio.wait_for(handler_finished.wait(), timeout=0.2)
    await asyncio.wait_for(runner_task, timeout=0.2)

    assert returned_within_bound is True
    assert cancellation_seen.is_set()


def test_worker_cli_process_exits_with_a_cancellation_resistant_handler() -> None:
    script = textwrap.dedent(
        """
        import asyncio
        import time
        from pathlib import Path

        import rag_service.jobs.worker as worker

        async def resistant_handler() -> None:
            while True:
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    continue

        async def fake_run_worker(
            health_path: Path,
            max_concurrency: int,
            *,
            handlers: object | None = None,
        ) -> int:
            del health_path, max_concurrency, handlers
            asyncio.create_task(resistant_handler(), name="resistant-job-handler")
            await asyncio.sleep(0)
            return 0

        worker._run_worker = fake_run_worker
        started = time.monotonic()
        result = worker.main([])
        elapsed = time.monotonic() - started
        raise SystemExit(result if elapsed < 0.2 else 3)
        """
    )

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).parents[2],
        capture_output=True,
        text=True,
        timeout=2,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


@pytest.mark.asyncio
async def test_shutdown_signal_during_notification_wait_does_not_start_another_claim() -> None:
    repository = _MemoryJobRepository(None)
    runner = _runner(repository)
    runner.register("ingest_document", lambda _context: asyncio.sleep(0))
    stop_event = asyncio.Event()

    class StopNotification:
        async def get(self, timeout_seconds: float) -> UUID | None:
            del timeout_seconds
            stop_event.set()
            return uuid4()

    await asyncio.wait_for(
        runner.run(stop_event, notifications=StopNotification()),
        timeout=0.5,
    )

    assert repository.claim_calls == 1


@pytest.mark.asyncio
async def test_periodic_health_check_continues_independently_of_job_capacity() -> None:
    stop_event = asyncio.Event()
    calls = 0

    async def check() -> bool:
        nonlocal calls
        calls += 1
        if calls == 3:
            stop_event.set()
        return True

    await asyncio.wait_for(
        _periodic_health_check(stop_event, interval_seconds=0.01, check=check),
        timeout=0.2,
    )

    assert calls == 3


@pytest.mark.asyncio
async def test_periodic_reconciliation_isolates_pass_failures_and_bounds_cursor_state(
    caplog: pytest.LogCaptureFixture,
) -> None:
    stop_event = asyncio.Event()
    minio_cursors: list[object | None] = []
    qdrant_cursors: list[object | None] = []
    secret = "maintenance-secret-object-key"

    async def minio_pass(cursor: object | None) -> object | None:
        minio_cursors.append(cursor)
        if len(minio_cursors) == 1:
            return "minio-page-2"
        if len(minio_cursors) == 2:
            raise RuntimeError(secret)
        stop_event.set()
        return None

    async def qdrant_pass(cursor: object | None) -> object | None:
        qdrant_cursors.append(cursor)
        return "qdrant-page-2"

    passes = (
        jobs_worker._WorkerReconciliationPass("minio", minio_pass),
        jobs_worker._WorkerReconciliationPass("qdrant", qdrant_pass),
    )
    with caplog.at_level(logging.WARNING, logger="rag_service.jobs.worker"):
        await asyncio.wait_for(
            jobs_worker._periodic_reconciliation(
                stop_event,
                interval_seconds=0.001,
                passes=passes,
            ),
            timeout=0.2,
        )

    assert minio_cursors == [None, "minio-page-2", "minio-page-2"]
    assert qdrant_cursors
    assert secret not in caplog.text
    failure_records = [
        record
        for record in caplog.records
        if record.msg == "cleanup.action.completed"
        and getattr(record, "event", None) == "cleanup.action.completed"
        and getattr(record, "operation", None) == "orphan_cleanup"
        and getattr(record, "outcome", None) == "failed"
    ]
    assert len(failure_records) == 1
    assert getattr(failure_records[0], "phase", None) == "minio"
    assert getattr(failure_records[0], "count", None) == 1


@pytest.mark.asyncio
async def test_periodic_reconciliation_blocked_pass_does_not_starve_other_passes() -> None:
    stop_event = asyncio.Event()
    blocked_started = asyncio.Event()
    release_blocked = asyncio.Event()
    qdrant_ran = asyncio.Event()

    async def blocked_minio(_cursor: object | None) -> object | None:
        blocked_started.set()
        await release_blocked.wait()
        return None

    async def qdrant(_cursor: object | None) -> object | None:
        qdrant_ran.set()
        stop_event.set()
        return None

    task = asyncio.create_task(
        jobs_worker._periodic_reconciliation(
            stop_event,
            interval_seconds=60,
            passes=(
                jobs_worker._WorkerReconciliationPass("minio", blocked_minio),
                jobs_worker._WorkerReconciliationPass("qdrant", qdrant),
            ),
        )
    )
    await asyncio.wait_for(blocked_started.wait(), timeout=0.2)
    await asyncio.wait_for(qdrant_ran.wait(), timeout=0.2)
    release_blocked.set()
    await asyncio.wait_for(task, timeout=0.2)


@pytest.mark.asyncio
async def test_periodic_reconciliation_propagates_cancellation() -> None:
    started = asyncio.Event()
    cancellation_seen = asyncio.Event()

    async def blocked_pass(_cursor: object | None) -> object | None:
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancellation_seen.set()
        return None

    task = asyncio.create_task(
        jobs_worker._periodic_reconciliation(
            asyncio.Event(),
            interval_seconds=60,
            passes=(jobs_worker._WorkerReconciliationPass("minio", blocked_pass),),
        )
    )
    await asyncio.wait_for(started.wait(), timeout=0.2)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert cancellation_seen.is_set()


@pytest.mark.asyncio
async def test_worker_cleanup_awaits_reconciliation_quiescence_before_closing_resources() -> None:
    reconciliation_started = asyncio.Event()
    reconciliation_quiesced = asyncio.Event()
    close_order: list[str] = []

    async def reconciliation() -> None:
        reconciliation_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            await asyncio.sleep(0.01)
            reconciliation_quiesced.set()
            close_order.append("reconciliation")

    async def close_resource() -> None:
        assert reconciliation_quiesced.is_set()
        close_order.append("resource")

    reconciliation_task = asyncio.create_task(reconciliation())
    await asyncio.wait_for(reconciliation_started.wait(), timeout=0.2)
    await jobs_worker._bounded_worker_cleanup(
        health_task=None,
        quiescence_tasks=(reconciliation_task,),
        close_operations=(close_resource,),
        timeout_seconds=0.2,
    )

    assert close_order == ["reconciliation", "resource"]


@pytest.mark.asyncio
async def test_worker_cleanup_waits_beyond_short_grace_for_reconciliation_barrier() -> None:
    reconciliation_started = asyncio.Event()
    reconciliation_quiesced = asyncio.Event()

    async def reconciliation() -> None:
        reconciliation_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            await asyncio.sleep(0.08)
            reconciliation_quiesced.set()

    async def close_resource() -> None:
        assert reconciliation_quiesced.is_set()

    reconciliation_task = asyncio.create_task(reconciliation())
    await asyncio.wait_for(reconciliation_started.wait(), timeout=0.2)
    await jobs_worker._bounded_worker_cleanup(
        health_task=None,
        quiescence_tasks=(reconciliation_task,),
        close_operations=(close_resource,),
        timeout_seconds=0.2,
    )

    assert reconciliation_quiesced.is_set()


@pytest.mark.asyncio
async def test_worker_cleanup_closes_independent_resources_when_shared_task_misses_deadline() -> (
    None
):
    started = asyncio.Event()
    release = asyncio.Event()
    closed: list[str] = []

    async def resistant_shared_task() -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await release.wait()
            raise

    async def close_independent() -> None:
        closed.append("independent")

    async def close_dependent() -> None:
        closed.append("dependent")

    shared_task = asyncio.create_task(resistant_shared_task())
    await asyncio.wait_for(started.wait(), timeout=0.2)
    try:
        await jobs_worker._bounded_worker_cleanup(
            health_task=None,
            quiescence_tasks=(shared_task,),
            independent_close_operations=(close_independent,),
            close_operations=(close_dependent,),
            timeout_seconds=0.03,
        )
        assert closed == ["independent"]
    finally:
        shared_task.cancel()
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await shared_task


@pytest.mark.asyncio
async def test_worker_cleanup_parent_cancellation_cleans_tasks_during_barrier_wait() -> None:
    barrier_cancelled = asyncio.Event()
    release_barrier = asyncio.Event()
    close_started = asyncio.Event()
    close_cancelled = asyncio.Event()

    async def resistant_barrier() -> None:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            barrier_cancelled.set()
            await release_barrier.wait()
            raise

    async def close_independent() -> None:
        close_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            close_cancelled.set()

    barrier = asyncio.create_task(resistant_barrier())
    cleanup = asyncio.create_task(
        jobs_worker._bounded_worker_cleanup(
            health_task=None,
            quiescence_tasks=(barrier,),
            independent_close_operations=(close_independent,),
            close_operations=(),
            timeout_seconds=1.0,
        )
    )
    await asyncio.wait_for(barrier_cancelled.wait(), timeout=0.2)
    await asyncio.wait_for(close_started.wait(), timeout=0.2)
    cleanup.cancel()
    try:
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(cleanup, timeout=0.2)
        await asyncio.wait_for(close_cancelled.wait(), timeout=0.2)
    finally:
        release_barrier.set()

    with pytest.raises(asyncio.CancelledError):
        await barrier


@pytest.mark.asyncio
async def test_worker_cleanup_parent_cancellation_cleans_tasks_during_close_wait() -> None:
    independent_started = asyncio.Event()
    independent_cancelled = asyncio.Event()
    dependent_started = asyncio.Event()
    dependent_cancelled = asyncio.Event()

    async def hang(started: asyncio.Event, cancelled: asyncio.Event) -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    cleanup = asyncio.create_task(
        jobs_worker._bounded_worker_cleanup(
            health_task=None,
            independent_close_operations=(
                lambda: hang(independent_started, independent_cancelled),
            ),
            close_operations=(lambda: hang(dependent_started, dependent_cancelled),),
            timeout_seconds=1.0,
        )
    )
    await asyncio.wait_for(independent_started.wait(), timeout=0.2)
    await asyncio.wait_for(dependent_started.wait(), timeout=0.2)
    cleanup.cancel()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(cleanup, timeout=0.2)
    await asyncio.wait_for(independent_cancelled.wait(), timeout=0.2)
    await asyncio.wait_for(dependent_cancelled.wait(), timeout=0.2)


@pytest.mark.asyncio
async def test_default_handler_resource_cleanup_continues_after_non_cancel_base_exception() -> None:
    closed: list[str] = []

    class FatalPipeline:
        async def aclose(self) -> None:
            closed.append("pipeline")
            raise BaseException("fatal cleanup")

    class Closeable:
        def __init__(self, name: str) -> None:
            self.name = name

        async def aclose(self) -> None:
            closed.append(self.name)

    class Pool:
        def clear(self) -> None:
            closed.append("pool")

    with pytest.raises(ExceptionGroup):
        await jobs_worker._close_default_handler_resources(
            ingestion_pipeline=cast(jobs_worker._AsyncCloseable, FatalPipeline()),
            embedding_gateway=cast(jobs_worker._AsyncCloseable, Closeable("gateway")),
            provider_transport=None,
            qdrant=cast(jobs_worker._AsyncCloseable, Closeable("qdrant")),
            object_store=cast(jobs_worker._AsyncCloseable, Closeable("store")),
            minio_pool=cast(jobs_worker._Clearable, Pool()),
        )

    assert set(closed) == {"pipeline", "gateway", "qdrant", "store", "pool"}
    assert len(closed) == 5


@pytest.mark.asyncio
async def test_default_handler_cleanup_clears_pool_only_after_object_store_drain() -> None:
    store_started = asyncio.Event()
    finish_store = asyncio.Event()
    pool_cleared = asyncio.Event()
    events: list[str] = []
    pool_was_cleared_during_drain: bool | None = None
    loop = asyncio.get_running_loop()

    class DrainingObjectStore:
        async def aclose(self) -> None:
            nonlocal pool_was_cleared_during_drain
            events.append("store_started")
            store_started.set()
            await finish_store.wait()
            pool_was_cleared_during_drain = pool_cleared.is_set()
            events.append("store_finished")

    class Pool:
        def clear(self) -> None:
            events.append("pool_cleared")
            loop.call_soon_threadsafe(pool_cleared.set)

    cleanup = asyncio.create_task(
        jobs_worker._close_default_handler_resources(
            ingestion_pipeline=None,
            embedding_gateway=None,
            provider_transport=None,
            qdrant=None,
            object_store=cast(jobs_worker._AsyncCloseable, DrainingObjectStore()),
            minio_pool=cast(jobs_worker._Clearable, Pool()),
        )
    )
    await asyncio.wait_for(store_started.wait(), timeout=0.2)
    for _ in range(50):
        if pool_cleared.is_set():
            break
        await asyncio.sleep(0.001)
    finish_store.set()
    await asyncio.wait_for(cleanup, timeout=0.2)

    assert pool_was_cleared_during_drain is False
    assert events == ["store_started", "store_finished", "pool_cleared"]


@pytest.mark.asyncio
async def test_default_handler_cleanup_clears_pool_after_object_store_base_exception() -> None:
    events: list[str] = []

    class FatalObjectStore:
        async def aclose(self) -> None:
            events.append("store_failed")
            raise BaseException("fatal object store cleanup")

    class Pool:
        def clear(self) -> None:
            events.append("pool_cleared")

    with pytest.raises(ExceptionGroup):
        await jobs_worker._close_default_handler_resources(
            ingestion_pipeline=None,
            embedding_gateway=None,
            provider_transport=None,
            qdrant=None,
            object_store=cast(jobs_worker._AsyncCloseable, FatalObjectStore()),
            minio_pool=cast(jobs_worker._Clearable, Pool()),
        )

    assert events == ["store_failed", "pool_cleared"]


@pytest.mark.asyncio
async def test_default_handler_cleanup_clears_pool_after_store_cancellation_quiesces() -> None:
    store_started = asyncio.Event()
    store_retiring = asyncio.Event()
    finish_retirement = asyncio.Event()
    pool_cleared = asyncio.Event()
    events: list[str] = []
    loop = asyncio.get_running_loop()

    class CancellingObjectStore:
        async def aclose(self) -> None:
            store_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                events.append("store_retiring")
                store_retiring.set()
                await finish_retirement.wait()
                events.append("store_retired")
                raise

    class Pool:
        def clear(self) -> None:
            events.append("pool_cleared")
            loop.call_soon_threadsafe(pool_cleared.set)

    cleanup = asyncio.create_task(
        jobs_worker._close_default_handler_resources(
            ingestion_pipeline=None,
            embedding_gateway=None,
            provider_transport=None,
            qdrant=None,
            object_store=cast(jobs_worker._AsyncCloseable, CancellingObjectStore()),
            minio_pool=cast(jobs_worker._Clearable, Pool()),
        )
    )
    await asyncio.wait_for(store_started.wait(), timeout=0.2)
    cleanup.cancel()
    await asyncio.wait_for(store_retiring.wait(), timeout=0.2)
    assert pool_cleared.is_set() is False
    finish_retirement.set()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(cleanup, timeout=0.2)
    await asyncio.wait_for(pool_cleared.wait(), timeout=0.2)
    assert events == ["store_retiring", "store_retired", "pool_cleared"]


@pytest.mark.asyncio
async def test_default_handler_cleanup_deadline_does_not_clear_pool_before_store_quiesces() -> None:
    store_started = asyncio.Event()
    store_cancelled = asyncio.Event()
    release_store = asyncio.Event()
    pool_cleared = asyncio.Event()
    loop = asyncio.get_running_loop()

    class CancellationResistantObjectStore:
        async def aclose(self) -> None:
            store_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                store_cancelled.set()
                await release_store.wait()
                raise

    class Pool:
        def clear(self) -> None:
            loop.call_soon_threadsafe(pool_cleared.set)

    async def close_resources() -> None:
        await jobs_worker._close_default_handler_resources(
            ingestion_pipeline=None,
            embedding_gateway=None,
            provider_transport=None,
            qdrant=None,
            object_store=cast(
                jobs_worker._AsyncCloseable,
                CancellationResistantObjectStore(),
            ),
            minio_pool=cast(jobs_worker._Clearable, Pool()),
        )

    started_at = asyncio.get_running_loop().time()
    await jobs_worker._bounded_worker_cleanup(
        health_task=None,
        close_operations=(close_resources,),
        timeout_seconds=0.03,
    )
    elapsed = asyncio.get_running_loop().time() - started_at
    await asyncio.wait_for(store_started.wait(), timeout=0.2)
    await asyncio.wait_for(store_cancelled.wait(), timeout=0.2)
    assert elapsed < 0.15
    assert pool_cleared.is_set() is False

    release_store.set()
    await asyncio.wait_for(pool_cleared.wait(), timeout=0.2)


@pytest.mark.asyncio
async def test_default_handler_resource_cleanup_starts_independent_closes_when_pipeline_hangs() -> (
    None
):
    pipeline_started = asyncio.Event()
    completed = {
        name: asyncio.Event() for name in ("gateway", "transport", "qdrant", "store", "pool")
    }
    loop = asyncio.get_running_loop()

    class HangingPipeline:
        async def aclose(self) -> None:
            pipeline_started.set()
            await asyncio.Event().wait()

    class Closeable:
        def __init__(self, name: str) -> None:
            self.name = name

        async def aclose(self) -> None:
            completed[self.name].set()

    class Pool:
        def clear(self) -> None:
            loop.call_soon_threadsafe(completed["pool"].set)

    cleanup = asyncio.create_task(
        jobs_worker._close_default_handler_resources(
            ingestion_pipeline=cast(jobs_worker._AsyncCloseable, HangingPipeline()),
            embedding_gateway=cast(jobs_worker._AsyncCloseable, Closeable("gateway")),
            provider_transport=cast(jobs_worker._AsyncCloseable, Closeable("transport")),
            qdrant=cast(jobs_worker._AsyncCloseable, Closeable("qdrant")),
            object_store=cast(jobs_worker._AsyncCloseable, Closeable("store")),
            minio_pool=cast(jobs_worker._Clearable, Pool()),
        )
    )
    try:
        await asyncio.wait_for(pipeline_started.wait(), timeout=0.2)
        await asyncio.wait_for(
            asyncio.gather(*(event.wait() for event in completed.values())),
            timeout=0.2,
        )
    finally:
        cleanup.cancel()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(cleanup, timeout=0.2)


@pytest.mark.asyncio
async def test_default_handler_cleanup_attempts_transport_after_gateway_base_exception() -> None:
    attempted: list[str] = []

    class FatalGateway:
        async def aclose(self) -> None:
            attempted.append("gateway")
            raise BaseException("fatal gateway cleanup")

    class Transport:
        async def aclose(self) -> None:
            attempted.append("transport")

    with pytest.raises(ExceptionGroup):
        await jobs_worker._close_default_handler_resources(
            ingestion_pipeline=None,
            embedding_gateway=cast(jobs_worker._AsyncCloseable, FatalGateway()),
            provider_transport=cast(jobs_worker._AsyncCloseable, Transport()),
            qdrant=None,
            object_store=None,
            minio_pool=None,
        )

    assert set(attempted) == {"gateway", "transport"}


@pytest.mark.asyncio
async def test_default_handler_resource_cleanup_attempts_transport_while_gateway_hangs() -> None:
    gateway_started = asyncio.Event()
    transport_closed = asyncio.Event()

    class HangingGateway:
        async def aclose(self) -> None:
            gateway_started.set()
            await asyncio.Event().wait()

    class Transport:
        async def aclose(self) -> None:
            transport_closed.set()

    cleanup = asyncio.create_task(
        jobs_worker._close_default_handler_resources(
            ingestion_pipeline=None,
            embedding_gateway=cast(jobs_worker._AsyncCloseable, HangingGateway()),
            provider_transport=cast(jobs_worker._AsyncCloseable, Transport()),
            qdrant=None,
            object_store=None,
            minio_pool=None,
        )
    )
    try:
        await asyncio.wait_for(gateway_started.wait(), timeout=0.2)
        await asyncio.wait_for(transport_closed.wait(), timeout=0.2)
    finally:
        cleanup.cancel()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(cleanup, timeout=0.2)


@pytest.mark.asyncio
async def test_default_handler_resource_cleanup_propagates_external_cancellation() -> None:
    started = asyncio.Event()
    cancellation_seen = asyncio.Event()

    class HangingPipeline:
        async def aclose(self) -> None:
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancellation_seen.set()

    cleanup = asyncio.create_task(
        jobs_worker._close_default_handler_resources(
            ingestion_pipeline=cast(jobs_worker._AsyncCloseable, HangingPipeline()),
            embedding_gateway=None,
            provider_transport=None,
            qdrant=None,
            object_store=None,
            minio_pool=None,
        )
    )
    await asyncio.wait_for(started.wait(), timeout=0.2)
    cleanup.cancel()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(cleanup, timeout=0.2)
    assert cancellation_seen.is_set()


@pytest.mark.asyncio
async def test_default_handler_resource_cleanup_respects_worker_cleanup_deadline() -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    cancellation_finished = asyncio.Event()

    class CancellationResistantPipeline:
        async def aclose(self) -> None:
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                await release.wait()
                cancellation_finished.set()
                raise

    async def close_resources() -> None:
        await jobs_worker._close_default_handler_resources(
            ingestion_pipeline=cast(
                jobs_worker._AsyncCloseable,
                CancellationResistantPipeline(),
            ),
            embedding_gateway=None,
            provider_transport=None,
            qdrant=None,
            object_store=None,
            minio_pool=None,
        )

    started_at = asyncio.get_running_loop().time()
    try:
        await jobs_worker._bounded_worker_cleanup(
            health_task=None,
            close_operations=(close_resources,),
            timeout_seconds=0.03,
        )
        elapsed = asyncio.get_running_loop().time() - started_at
        assert started.is_set()
        assert elapsed < 0.15
    finally:
        release.set()

    await asyncio.wait_for(cancellation_finished.wait(), timeout=0.2)


@pytest.mark.asyncio
async def test_worker_preflight_does_not_gate_database_claiming_on_redis() -> None:
    class Rows:
        def all(self) -> list[object]:
            return []

    class Session:
        async def scalars(self, statement: object) -> Rows:
            del statement
            return Rows()

    class DatabaseStub:
        def __init__(self) -> None:
            self.ping_calls = 0

        async def ping(self) -> None:
            self.ping_calls += 1

        @asynccontextmanager
        async def sessions(self) -> AsyncIterator[Session]:
            yield Session()

    database = DatabaseStub()
    preflight = WorkerDependencyPreflight(
        database=cast(Database, database),
        keyring=cast(ProviderCredentialKeyring, object()),
        keyring_fingerprint="test-keyring",
    )

    assert await preflight() is True
    assert database.ping_calls == 1


@pytest.mark.asyncio
async def test_worker_notification_start_failure_degrades_to_database_polling() -> None:
    class UnavailableSubscriber:
        async def start(self) -> None:
            raise ConnectionError("redis unavailable")

    subscriber = cast(RedisJobSubscriber, UnavailableSubscriber())
    assert await _start_notification_source(subscriber) is None


class _WorkerTestDatabase:
    def __init__(self, *, close_error: Exception | None = None) -> None:
        self.close_error = close_error
        self.close_calls = 0

    async def close(self) -> None:
        self.close_calls += 1
        if self.close_error is not None:
            raise self.close_error


class _WorkerTestRedis:
    def __init__(self, *, close_error: Exception | None = None) -> None:
        self.close_error = close_error
        self.close_calls = 0

    async def aclose(self) -> None:
        self.close_calls += 1
        if self.close_error is not None:
            raise self.close_error


class _WorkerTestSubscriber:
    def __init__(self, *, close_error: Exception | None = None) -> None:
        self.close_error = close_error
        self.close_calls = 0

    async def start(self) -> None:
        return None

    async def close(self) -> None:
        self.close_calls += 1
        if self.close_error is not None:
            raise self.close_error


def _patch_worker_runtime(
    monkeypatch: pytest.MonkeyPatch,
    *,
    preflight: Callable[[], Awaitable[bool]],
    database: _WorkerTestDatabase | None = None,
    redis_client: _WorkerTestRedis | None = None,
    subscriber: _WorkerTestSubscriber | None = None,
    timeout_seconds: float = 0.03,
) -> tuple[
    _WorkerTestDatabase,
    _WorkerTestRedis,
    _WorkerTestSubscriber,
    dict[str, asyncio.Event],
]:
    database = database or _WorkerTestDatabase()
    redis_client = redis_client or _WorkerTestRedis()
    subscriber = subscriber or _WorkerTestSubscriber()
    settings = SimpleNamespace(
        redis_url=SimpleNamespace(get_secret_value=lambda: "redis://unused"),
        readiness_timeout_seconds=timeout_seconds,
        shutdown_timeout_seconds=timeout_seconds,
        worker_lease_seconds=30.0,
        worker_heartbeat_seconds=10.0,
        worker_poll_interval_seconds=0.01,
        worker_retry_initial_seconds=0.01,
        worker_retry_max_seconds=0.02,
        provider_credential_keyring=SimpleNamespace(
            get_secret_value=lambda: '{"test-v1":"redacted-test-key"}'
        ),
        provider_credential_active_key_version="test-v1",
    )
    captured_stop: dict[str, asyncio.Event] = {}

    def install_signals(
        _loop: asyncio.AbstractEventLoop,
        stop_event: asyncio.Event,
    ) -> tuple[object, ...]:
        captured_stop["event"] = stop_event
        return ()

    monkeypatch.setattr(jobs_worker, "Settings", lambda: settings)
    monkeypatch.setattr(
        jobs_worker,
        "Database",
        SimpleNamespace(from_settings=lambda _settings: database),
    )
    monkeypatch.setattr(
        jobs_worker,
        "Redis",
        SimpleNamespace(from_url=lambda *_args, **_kwargs: redis_client),
    )
    monkeypatch.setattr(jobs_worker, "RedisJobSubscriber", lambda _client: subscriber)
    monkeypatch.setattr(
        jobs_worker,
        "provider_credential_keyring_from_settings",
        lambda _settings: object(),
    )
    monkeypatch.setattr(
        jobs_worker,
        "WorkerDependencyPreflight",
        lambda **_kwargs: preflight,
    )
    monkeypatch.setattr(jobs_worker, "_install_signal_handlers", install_signals)
    monkeypatch.setattr(jobs_worker, "_remove_signal_handlers", lambda *_args: None)
    return database, redis_client, subscriber, captured_stop


@pytest.mark.asyncio
async def test_default_worker_registers_ingestion_and_generation_repair_handlers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def preflight() -> bool:
        return True

    async def ingest_handler(_context: JobExecutionContext) -> JobHandlerOutcome:
        return JobHandlerOutcome.COMPLETE

    async def ingest_finalizer(_candidate: object, _session: object) -> None:
        return None

    async def repair_handler(_context: JobExecutionContext) -> JobHandlerOutcome:
        return JobHandlerOutcome.COMPLETE

    closed: list[str] = []

    class Runtime:
        def __init__(self) -> None:
            self.registrations = {
                "ingest_document": jobs_worker.JobHandlerRegistration(
                    ingest_handler,
                    ingest_finalizer,
                ),
                "rebuild_generation": jobs_worker.JobHandlerRegistration(repair_handler),
            }
            self.close_calls = 0

        async def aclose(self) -> None:
            await asyncio.sleep(0)
            closed.append("handler_runtime")
            self.close_calls += 1

    class Database(_WorkerTestDatabase):
        async def close(self) -> None:
            closed.append("database")
            await super().close()

    runtime = Runtime()
    build_calls: list[tuple[object, object, object, int]] = []

    async def build_runtime(
        *,
        settings: object,
        database: object,
        keyring: object,
        max_concurrency: int,
    ) -> Runtime:
        build_calls.append((settings, database, keyring, max_concurrency))
        return runtime

    registered: list[tuple[str, object, object | None]] = []

    class Runner:
        def __init__(self, **_kwargs: object) -> None:
            self.has_handlers = False

        def register(
            self,
            operation: str,
            handler: object,
            *,
            exhaustion_finalizer: object | None = None,
        ) -> None:
            registered.append((operation, handler, exhaustion_finalizer))
            self.has_handlers = True

        async def run(self, _stop_event: asyncio.Event, **_kwargs: object) -> None:
            return None

    database, _redis_client, _subscriber, _captured_stop = _patch_worker_runtime(
        monkeypatch,
        preflight=preflight,
        database=Database(),
    )
    monkeypatch.setattr(jobs_worker, "_build_default_handler_runtime", build_runtime)
    monkeypatch.setattr(jobs_worker, "JobRunner", Runner)

    result = await jobs_worker._run_worker(tmp_path / "health.json", 3)

    assert result == 0
    assert len(build_calls) == 1
    assert build_calls[0][1] is database
    assert build_calls[0][3] == 3
    assert registered == [
        ("ingest_document", ingest_handler, ingest_finalizer),
        ("rebuild_generation", repair_handler, None),
    ]
    assert runtime.close_calls == 1
    assert closed == ["handler_runtime", "database"]


@pytest.mark.asyncio
async def test_worker_reconciliation_runs_beside_claim_loop_and_is_cancelled_on_shutdown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def preflight() -> bool:
        return True

    reconciliation_started = asyncio.Event()
    reconciliation_cancelled = asyncio.Event()
    runner_ran = False

    async def reconcile(_cursor: object | None) -> object | None:
        reconciliation_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            reconciliation_cancelled.set()
        return None

    async def handler(_context: JobExecutionContext) -> JobHandlerOutcome:
        return JobHandlerOutcome.COMPLETE

    class Runtime:
        def __init__(self) -> None:
            self.registrations = {
                "ingest_document": jobs_worker.JobHandlerRegistration(handler),
            }
            self.reconciliation_passes = (
                jobs_worker._WorkerReconciliationPass("minio", reconcile),
            )

        async def aclose(self) -> None:
            return None

    class Runner:
        def __init__(self, **_kwargs: object) -> None:
            self.has_handlers = False

        def register(self, _operation: str, _handler: object, **_kwargs: object) -> None:
            self.has_handlers = True

        async def run(self, stop_event: asyncio.Event, **_kwargs: object) -> None:
            nonlocal runner_ran
            await reconciliation_started.wait()
            runner_ran = True
            stop_event.set()

    runtime = Runtime()

    async def build_runtime(**_kwargs: object) -> Runtime:
        return runtime

    _patch_worker_runtime(monkeypatch, preflight=preflight)
    monkeypatch.setattr(jobs_worker, "_build_default_handler_runtime", build_runtime)
    monkeypatch.setattr(jobs_worker, "JobRunner", Runner)

    result = await asyncio.wait_for(
        jobs_worker._run_worker(tmp_path / "health.json", 1),
        timeout=0.3,
    )

    assert result == 0
    assert runner_ran is True
    assert reconciliation_cancelled.is_set()


@pytest.mark.asyncio
async def test_default_handler_runtime_composes_shared_dependencies_and_closes_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}
    closed: list[str] = []

    class Pool:
        def clear(self) -> None:
            closed.append("minio_pool")

    pool = Pool()

    class ObjectStore:
        async def aclose(self) -> None:
            closed.append("object_store")

    object_store = ObjectStore()

    class Qdrant:
        async def aclose(self) -> None:
            closed.append("qdrant")

    qdrant = Qdrant()

    class Transport:
        def __init__(self) -> None:
            self.closed = False

        async def aclose(self) -> None:
            if self.closed:
                return
            self.closed = True
            closed.append("provider_transport")

    transport = Transport()

    class Gateway:
        async def aclose(self) -> None:
            closed.append("embedding_gateway")
            await transport.aclose()

    gateway = Gateway()

    class Ingestion:
        async def handle(self, _context: JobExecutionContext) -> JobHandlerOutcome:
            return JobHandlerOutcome.COMPLETE

        async def finalize_exhausted(self, _candidate: object, _session: object) -> None:
            return None

        async def aclose(self) -> None:
            closed.append("ingestion_pipeline")

        async def reconcile_orphan_objects(self, **_kwargs: object) -> object:
            return SimpleNamespace(next_cursor="minio-next")

    ingestion = Ingestion()

    class Repair:
        async def handle(self, _context: JobExecutionContext) -> JobHandlerOutcome:
            return JobHandlerOutcome.COMPLETE

    repair = Repair()

    class Generation:
        async def reconcile_orphan_collections(self, **_kwargs: object) -> object:
            return SimpleNamespace(next_cursor="qdrant-next")

    generation = Generation()
    policy = object()

    def make_store(**kwargs: object) -> ObjectStore:
        calls["store"] = kwargs
        return object_store

    def make_gateway(**kwargs: object) -> Gateway:
        calls["gateway"] = kwargs
        return gateway

    def make_ingestion(**kwargs: object) -> Ingestion:
        calls["ingestion"] = kwargs
        return ingestion

    def make_repair(**kwargs: object) -> Repair:
        calls["repair"] = kwargs
        return repair

    def make_transport(**kwargs: object) -> Transport:
        calls["transport"] = kwargs
        return transport

    def make_generation(**kwargs: object) -> Generation:
        calls["generation"] = kwargs
        return generation

    dependencies = jobs_worker._DefaultHandlerDependencies(
        minio_client=lambda *args, **kwargs: (args, kwargs),
        pool_manager=lambda **_kwargs: pool,
        timeout=lambda **kwargs: ("timeout", kwargs),
        minio_object_store=make_store,
        validate_minio_url=lambda _url: ("minio:9000", False),
        qdrant_client_from_url=lambda *_args, **_kwargs: qdrant,
        secure_provider_transport=make_transport,
        provider_endpoint_policy_from_settings=lambda _settings: policy,
        credential_reader=lambda value: value,
        embedding_gateway=make_gateway,
        provider_usage_sink=lambda value: value,
        ingestion_pipeline=make_ingestion,
        ingestion_repository=lambda value: value,
        repair_pipeline=make_repair,
        generation_service=make_generation,
    )
    monkeypatch.setattr(jobs_worker, "_load_default_handler_dependencies", lambda: dependencies)

    sessions = object()
    database = SimpleNamespace(sessions=sessions)
    settings = SimpleNamespace(
        minio_url="http://minio:9000",
        minio_access_key="access",
        minio_secret_key=SimpleNamespace(get_secret_value=lambda: "secret"),
        minio_bucket="documents",
        upload_buffer_bytes=1024,
        minio_multipart_part_size_bytes=5 * 1024 * 1024,
        minio_operation_timeout_seconds=30.0,
        qdrant_url="http://qdrant:6333",
        qdrant_request_timeout_seconds=12.0,
        provider_ca_bundle=None,
        max_upload_bytes=50 * 1024 * 1024,
        chunk_max_codepoints=1200,
        chunk_overlap_codepoints=150,
        orphan_object_grace_seconds=24 * 60 * 60,
    )
    keyring = object()

    runtime = await jobs_worker._build_default_handler_runtime(
        settings=cast(Settings, settings),
        database=cast(Database, database),
        keyring=cast(ProviderCredentialKeyring, keyring),
        max_concurrency=4,
    )

    assert list(runtime.registrations) == [
        "ingest_document",
        "rebuild_generation",
        "purge_knowledge_base",
    ]
    assert runtime.registrations["ingest_document"].handler == ingestion.handle
    assert runtime.registrations["ingest_document"].exhaustion_finalizer == (
        ingestion.finalize_exhausted
    )
    assert runtime.registrations["rebuild_generation"].handler == repair.handle
    assert calls["gateway"] == {
        "keyring": keyring,
        "credential_reader": sessions,
        "transport": transport,
    }
    assert calls["transport"] == {"policy": policy, "ca_bundle": None}
    assert cast(dict[str, object], calls["ingestion"])["object_store"] is object_store
    assert cast(dict[str, object], calls["ingestion"])["embedding_gateway"] is gateway
    assert cast(dict[str, object], calls["ingestion"])["qdrant"] is qdrant
    assert cast(dict[str, object], calls["ingestion"])["cpu_concurrency"] == 4
    assert cast(dict[str, object], calls["repair"])["session_factory"] is sessions
    assert cast(dict[str, object], calls["repair"])["object_store"] is object_store
    assert cast(dict[str, object], calls["repair"])["embedding_gateway"] is gateway
    assert cast(dict[str, object], calls["repair"])["qdrant"] is qdrant
    assert calls["generation"] == {
        "session_factory": sessions,
        "qdrant": qdrant,
        "embedding_gateway": gateway,
    }
    assert [item.phase for item in runtime.reconciliation_passes] == ["minio", "qdrant"]
    assert await runtime.reconciliation_passes[0].run(None) == "minio-next"
    assert await runtime.reconciliation_passes[1].run(None) == "qdrant-next"

    await runtime.aclose()

    assert set(closed) == {
        "ingestion_pipeline",
        "embedding_gateway",
        "provider_transport",
        "qdrant",
        "object_store",
        "minio_pool",
    }
    assert len(closed) == 6


@pytest.mark.asyncio
async def test_default_handler_runtime_rolls_back_all_constructed_resources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closed: list[str] = []

    class Pool:
        def clear(self) -> None:
            closed.append("minio_pool")

    class ObjectStore:
        async def aclose(self) -> None:
            closed.append("object_store")

    class Qdrant:
        async def aclose(self) -> None:
            closed.append("qdrant")

    class Transport:
        def __init__(self) -> None:
            self.closed = False

        async def aclose(self) -> None:
            if self.closed:
                return
            self.closed = True
            closed.append("provider_transport")

    transport = Transport()

    class Gateway:
        async def aclose(self) -> None:
            closed.append("embedding_gateway")
            await transport.aclose()

    class Ingestion:
        async def handle(self, _context: JobExecutionContext) -> JobHandlerOutcome:
            return JobHandlerOutcome.COMPLETE

        async def finalize_exhausted(self, _candidate: object, _session: object) -> None:
            return None

        async def aclose(self) -> None:
            closed.append("ingestion_pipeline")
            raise RuntimeError("cleanup secret")

    pool = Pool()
    object_store = ObjectStore()
    qdrant = Qdrant()
    gateway = Gateway()
    ingestion = Ingestion()

    def fail_repair(**_kwargs: object) -> None:
        raise RuntimeError("repair construction failed")

    dependencies = jobs_worker._DefaultHandlerDependencies(
        minio_client=lambda *args, **kwargs: (args, kwargs),
        pool_manager=lambda **_kwargs: pool,
        timeout=lambda **kwargs: ("timeout", kwargs),
        minio_object_store=lambda **_kwargs: object_store,
        validate_minio_url=lambda _url: ("minio:9000", False),
        qdrant_client_from_url=lambda *_args, **_kwargs: qdrant,
        secure_provider_transport=lambda **_kwargs: transport,
        provider_endpoint_policy_from_settings=lambda _settings: object(),
        credential_reader=lambda value: value,
        embedding_gateway=lambda **_kwargs: gateway,
        provider_usage_sink=lambda value: value,
        ingestion_pipeline=lambda **_kwargs: ingestion,
        ingestion_repository=lambda value: value,
        repair_pipeline=fail_repair,
        generation_service=lambda **_kwargs: object(),
    )
    monkeypatch.setattr(jobs_worker, "_load_default_handler_dependencies", lambda: dependencies)
    settings = SimpleNamespace(
        minio_url="http://minio:9000",
        minio_access_key="access",
        minio_secret_key=SimpleNamespace(get_secret_value=lambda: "secret"),
        minio_bucket="documents",
        upload_buffer_bytes=1024,
        minio_multipart_part_size_bytes=5 * 1024 * 1024,
        minio_operation_timeout_seconds=30.0,
        qdrant_url="http://qdrant:6333",
        qdrant_request_timeout_seconds=12.0,
        provider_ca_bundle=None,
        max_upload_bytes=50 * 1024 * 1024,
        chunk_max_codepoints=1200,
        chunk_overlap_codepoints=150,
        orphan_object_grace_seconds=24 * 60 * 60,
    )

    with pytest.raises(RuntimeError, match="^repair construction failed$"):
        await jobs_worker._build_default_handler_runtime(
            settings=cast(Settings, settings),
            database=cast(Database, SimpleNamespace(sessions=object())),
            keyring=cast(ProviderCredentialKeyring, object()),
            max_concurrency=2,
        )

    assert set(closed) == {
        "ingestion_pipeline",
        "embedding_gateway",
        "provider_transport",
        "qdrant",
        "object_store",
        "minio_pool",
    }
    assert len(closed) == 6


@pytest.mark.asyncio
async def test_stop_interrupts_hanging_worker_startup_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    run_calls = 0

    async def preflight() -> bool:
        started.set()
        await release.wait()
        return True

    class Runner:
        has_handlers = True

        def __init__(self, **_kwargs: object) -> None:
            return None

        def register(self, _operation: str, _handler: object, **_kwargs: object) -> None:
            return None

        async def run(self, _stop_event: asyncio.Event, **_kwargs: object) -> None:
            nonlocal run_calls
            run_calls += 1

    monkeypatch.setattr(jobs_worker, "JobRunner", Runner)
    database, redis_client, subscriber, captured_stop = _patch_worker_runtime(
        monkeypatch,
        preflight=preflight,
    )
    task = asyncio.create_task(
        jobs_worker._run_worker(
            tmp_path / "health.json",
            1,
            handlers={
                "ingest_document": jobs_worker.JobHandlerRegistration(
                    lambda _context: asyncio.sleep(0)
                )
            },
        )
    )
    await asyncio.wait_for(started.wait(), timeout=0.2)
    captured_stop["event"].set()

    done, _pending = await asyncio.wait({task}, timeout=0.12)
    returned_before_release = task in done
    release.set()
    result = await asyncio.wait_for(task, timeout=0.2)

    assert returned_before_release is True
    assert result == 0
    assert database.close_calls == 1
    assert redis_client.close_calls == 1
    assert subscriber.close_calls == 1
    assert run_calls == 0


@pytest.mark.asyncio
async def test_worker_startup_preflight_has_a_hard_readiness_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    run_calls = 0

    async def preflight() -> bool:
        started.set()
        await release.wait()
        return False

    class Runner:
        has_handlers = True

        def __init__(self, **_kwargs: object) -> None:
            return None

        def register(self, _operation: str, _handler: object, **_kwargs: object) -> None:
            return None

        async def run(self, _stop_event: asyncio.Event, **_kwargs: object) -> None:
            nonlocal run_calls
            run_calls += 1

    monkeypatch.setattr(jobs_worker, "JobRunner", Runner)
    database, redis_client, subscriber, _captured_stop = _patch_worker_runtime(
        monkeypatch,
        preflight=preflight,
    )
    task = asyncio.create_task(
        jobs_worker._run_worker(
            tmp_path / "health.json",
            1,
            handlers={
                "ingest_document": jobs_worker.JobHandlerRegistration(
                    lambda _context: asyncio.sleep(0)
                )
            },
        )
    )
    await asyncio.wait_for(started.wait(), timeout=0.2)

    done, _pending = await asyncio.wait({task}, timeout=0.12)
    returned_before_release = task in done
    release.set()
    result = await asyncio.wait_for(task, timeout=0.2)

    assert returned_before_release is True
    assert result == 1
    assert database.close_calls == 1
    assert redis_client.close_calls == 1
    assert subscriber.close_calls == 1
    assert run_calls == 0


@pytest.mark.asyncio
async def test_periodic_preflight_cannot_block_worker_health_task_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    periodic_started = asyncio.Event()
    cancellation_seen = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    async def preflight() -> bool:
        nonlocal calls
        calls += 1
        if calls == 1:
            return True
        periodic_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancellation_seen.set()
            await release.wait()
            raise
        return False

    class Runner:
        has_handlers = True

        def __init__(self, **_kwargs: object) -> None:
            return None

        def register(self, _operation: str, _handler: object, **_kwargs: object) -> None:
            return None

        async def run(self, _stop_event: asyncio.Event, **_kwargs: object) -> None:
            await periodic_started.wait()

    monkeypatch.setattr(jobs_worker, "JobRunner", Runner)
    database, redis_client, subscriber, _captured_stop = _patch_worker_runtime(
        monkeypatch,
        preflight=preflight,
    )
    task = asyncio.create_task(
        jobs_worker._run_worker(
            tmp_path / "health.json",
            1,
            handlers={
                "ingest_document": jobs_worker.JobHandlerRegistration(
                    lambda _context: asyncio.sleep(0)
                )
            },
        )
    )
    await asyncio.wait_for(periodic_started.wait(), timeout=0.2)

    done, _pending = await asyncio.wait({task}, timeout=0.12)
    returned_before_release = task in done
    release.set()
    result = await asyncio.wait_for(task, timeout=0.2)

    assert returned_before_release is True
    assert result == 0
    assert cancellation_seen.is_set()
    assert database.close_calls == 0
    assert redis_client.close_calls == 1
    assert subscriber.close_calls == 1


@pytest.mark.asyncio
async def test_worker_preflight_gate_keeps_one_in_flight_check_after_timeout() -> None:
    calls = 0
    cancellation_seen = asyncio.Event()
    release = asyncio.Event()
    finished = asyncio.Event()

    async def preflight() -> bool:
        nonlocal calls
        calls += 1
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancellation_seen.set()
            await release.wait()
        finally:
            finished.set()
        return False

    gate = jobs_worker._WorkerPreflightGate(preflight, timeout_seconds=0.02)
    stop_event = asyncio.Event()

    assert await gate.run(stop_event) is False
    await asyncio.wait_for(cancellation_seen.wait(), timeout=0.1)
    assert await gate.run(stop_event) is False
    assert calls == 1

    release.set()
    await asyncio.wait_for(finished.wait(), timeout=0.1)


@pytest.mark.asyncio
async def test_worker_preflight_gate_shares_one_successful_in_flight_check() -> None:
    calls = 0
    started = asyncio.Event()
    release = asyncio.Event()

    async def preflight() -> bool:
        nonlocal calls
        calls += 1
        started.set()
        await release.wait()
        return True

    gate = jobs_worker._WorkerPreflightGate(preflight, timeout_seconds=0.2)
    stop_event = asyncio.Event()
    first = asyncio.create_task(gate.run(stop_event))
    await asyncio.wait_for(started.wait(), timeout=0.1)
    second = asyncio.create_task(gate.run(stop_event))
    await asyncio.sleep(0)
    release.set()

    assert list(await asyncio.gather(first, second)) == [True, True]
    assert calls == 1


@pytest.mark.asyncio
async def test_worker_preflight_gate_shared_timeout_fails_closed_for_every_waiter() -> None:
    started = asyncio.Event()

    async def preflight() -> bool:
        started.set()
        await asyncio.Event().wait()
        return True

    gate = jobs_worker._WorkerPreflightGate(preflight, timeout_seconds=0.03)
    stop_event = asyncio.Event()
    first = asyncio.create_task(gate.run(stop_event))
    await asyncio.wait_for(started.wait(), timeout=0.1)
    await asyncio.sleep(0.01)
    second = asyncio.create_task(gate.run(stop_event))

    assert list(await asyncio.gather(first, second)) == [False, False]


@pytest.mark.asyncio
async def test_worker_preflight_gate_rejects_late_success_after_shared_timeout() -> None:
    started = asyncio.Event()
    cancellation_seen = asyncio.Event()
    release_after_cancellation = asyncio.Event()

    async def preflight() -> bool:
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancellation_seen.set()
            await release_after_cancellation.wait()
            return True
        return False

    gate = jobs_worker._WorkerPreflightGate(preflight, timeout_seconds=0.1)
    stop_event = asyncio.Event()
    first = asyncio.create_task(gate.run(stop_event))
    await asyncio.wait_for(started.wait(), timeout=0.1)
    await asyncio.sleep(0.05)
    second = asyncio.create_task(gate.run(stop_event))

    assert await first is False
    await asyncio.wait_for(cancellation_seen.wait(), timeout=0.1)
    release_after_cancellation.set()

    assert await second is False


@pytest.mark.asyncio
async def test_worker_preflight_gate_rejects_late_success_after_shared_stop() -> None:
    started = asyncio.Event()
    cancellation_seen = asyncio.Event()
    release_after_cancellation = asyncio.Event()

    async def preflight() -> bool:
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancellation_seen.set()
            await release_after_cancellation.wait()
            return True
        return False

    gate = jobs_worker._WorkerPreflightGate(preflight, timeout_seconds=0.2)
    first_stop = asyncio.Event()
    second_stop = asyncio.Event()
    first = asyncio.create_task(gate.run(first_stop))
    await asyncio.wait_for(started.wait(), timeout=0.1)
    second = asyncio.create_task(gate.run(second_stop))
    await asyncio.sleep(0)

    first_stop.set()
    assert await first is False
    await asyncio.wait_for(cancellation_seen.wait(), timeout=0.1)
    release_after_cancellation.set()

    assert await second is False


@pytest.mark.asyncio
async def test_worker_preflight_gate_caller_cancellation_does_not_poison_check() -> None:
    calls = 0
    started = asyncio.Event()
    release = asyncio.Event()

    async def preflight() -> bool:
        nonlocal calls
        calls += 1
        started.set()
        await release.wait()
        return True

    gate = jobs_worker._WorkerPreflightGate(preflight, timeout_seconds=0.2)
    stop_event = asyncio.Event()
    cancelled_waiter = asyncio.create_task(gate.run(stop_event))
    await asyncio.wait_for(started.wait(), timeout=0.1)

    cancelled_waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled_waiter

    remaining_waiter = asyncio.create_task(gate.run(stop_event))
    await asyncio.sleep(0)
    release.set()

    assert await remaining_waiter is True
    assert calls == 1


@pytest.mark.asyncio
async def test_worker_preflight_gate_deadline_survives_waiter_cancellation() -> None:
    calls = 0
    started = asyncio.Event()
    cancellation_seen = asyncio.Event()
    release_after_cancellation = asyncio.Event()
    finished = asyncio.Event()

    async def preflight() -> bool:
        nonlocal calls
        calls += 1
        started.set()
        try:
            await release_after_cancellation.wait()
        except asyncio.CancelledError:
            cancellation_seen.set()
            await release_after_cancellation.wait()
        finally:
            finished.set()
        return True

    gate = jobs_worker._WorkerPreflightGate(preflight, timeout_seconds=0.04)
    stop_event = asyncio.Event()
    cancelled_waiter = asyncio.create_task(gate.run(stop_event))
    await asyncio.wait_for(started.wait(), timeout=0.1)

    cancelled_waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled_waiter
    await asyncio.sleep(0.06)

    late_waiter = asyncio.create_task(gate.run(stop_event))
    await asyncio.sleep(0)
    release_after_cancellation.set()

    assert await late_waiter is False
    assert calls == 1
    assert cancellation_seen.is_set()
    await asyncio.wait_for(finished.wait(), timeout=0.1)


@pytest.mark.asyncio
async def test_worker_preflight_gate_rejects_success_completed_after_deadline() -> None:
    async def preflight() -> bool:
        time.sleep(0.06)  # noqa: ASYNC251 - exercise a synchronously blocked loop
        return True

    gate = jobs_worker._WorkerPreflightGate(preflight, timeout_seconds=0.02)

    assert await gate.run(asyncio.Event()) is False


@pytest.mark.asyncio
async def test_final_health_report_failure_still_closes_all_worker_resources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def preflight() -> bool:
        return True

    health_started = asyncio.Event()
    health_cancelled = asyncio.Event()

    async def health_loop(
        _stop_event: asyncio.Event,
        *,
        interval_seconds: float,
        check: Callable[[], Awaitable[bool]],
    ) -> None:
        del interval_seconds, check
        health_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            health_cancelled.set()

    class Runner:
        has_handlers = True

        def __init__(self, **_kwargs: object) -> None:
            return None

        def register(self, _operation: str, _handler: object, **_kwargs: object) -> None:
            return None

        async def run(self, _stop_event: asyncio.Event, **_kwargs: object) -> None:
            await health_started.wait()

    database = _WorkerTestDatabase(close_error=RuntimeError("database secret"))
    redis_client = _WorkerTestRedis(close_error=RuntimeError("redis secret"))
    subscriber = _WorkerTestSubscriber(close_error=RuntimeError("subscriber secret"))
    _patch_worker_runtime(
        monkeypatch,
        preflight=preflight,
        database=database,
        redis_client=redis_client,
        subscriber=subscriber,
    )
    monkeypatch.setattr(jobs_worker, "JobRunner", Runner)
    monkeypatch.setattr(jobs_worker, "_periodic_health_check", health_loop)
    reports = 0

    def report(_path: Path, _snapshot: WorkerHealthSnapshot) -> None:
        nonlocal reports
        reports += 1
        if reports == 3:
            raise RuntimeError("health report failed")

    monkeypatch.setattr(jobs_worker, "write_worker_health", report)

    with pytest.raises(RuntimeError, match="^health report failed$"):
        await jobs_worker._run_worker(
            tmp_path / "health.json",
            1,
            handlers={
                "ingest_document": jobs_worker.JobHandlerRegistration(
                    lambda _context: asyncio.sleep(0)
                )
            },
        )

    assert database.close_calls == 1
    assert redis_client.close_calls == 1
    assert subscriber.close_calls == 1
    assert health_cancelled.is_set()


@pytest.mark.asyncio
async def test_teardown_failures_do_not_override_the_primary_worker_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def preflight() -> bool:
        return True

    class Runner:
        has_handlers = True

        def __init__(self, **_kwargs: object) -> None:
            return None

        def register(self, _operation: str, _handler: object, **_kwargs: object) -> None:
            return None

        async def run(self, _stop_event: asyncio.Event, **_kwargs: object) -> None:
            raise RuntimeError("primary worker failure")

    database = _WorkerTestDatabase(close_error=RuntimeError("database secret"))
    redis_client = _WorkerTestRedis(close_error=RuntimeError("redis secret"))
    subscriber = _WorkerTestSubscriber(close_error=RuntimeError("subscriber secret"))
    _patch_worker_runtime(
        monkeypatch,
        preflight=preflight,
        database=database,
        redis_client=redis_client,
        subscriber=subscriber,
    )
    monkeypatch.setattr(jobs_worker, "JobRunner", Runner)
    reports = 0

    def report(_path: Path, _snapshot: WorkerHealthSnapshot) -> None:
        nonlocal reports
        reports += 1
        if reports == 3:
            raise RuntimeError("health report failure")

    monkeypatch.setattr(jobs_worker, "write_worker_health", report)

    with pytest.raises(RuntimeError, match="^primary worker failure$"):
        await jobs_worker._run_worker(
            tmp_path / "health.json",
            1,
            handlers={
                "ingest_document": jobs_worker.JobHandlerRegistration(
                    lambda _context: asyncio.sleep(0)
                )
            },
        )

    assert database.close_calls == 1
    assert redis_client.close_calls == 1
    assert subscriber.close_calls == 1


@pytest.mark.asyncio
async def test_hanging_subscriber_close_cannot_block_worker_teardown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def preflight() -> bool:
        return False

    class HangingSubscriber(_WorkerTestSubscriber):
        def __init__(self) -> None:
            super().__init__()
            self.close_started = asyncio.Event()
            self.cancellation_seen = asyncio.Event()
            self.release = asyncio.Event()

        async def close(self) -> None:
            self.close_calls += 1
            self.close_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancellation_seen.set()
                await self.release.wait()

    subscriber = HangingSubscriber()
    database, redis_client, _subscriber, _captured_stop = _patch_worker_runtime(
        monkeypatch,
        preflight=preflight,
        subscriber=subscriber,
    )
    task = asyncio.create_task(
        jobs_worker._run_worker(
            tmp_path / "health.json",
            1,
            handlers={
                "ingest_document": jobs_worker.JobHandlerRegistration(
                    lambda _context: asyncio.sleep(0)
                )
            },
        )
    )
    await asyncio.wait_for(subscriber.close_started.wait(), timeout=0.2)

    done, _pending = await asyncio.wait({task}, timeout=0.12)
    returned_before_release = task in done
    subscriber.release.set()
    result = await asyncio.wait_for(task, timeout=0.2)

    assert returned_before_release is True
    assert result == 1
    assert subscriber.cancellation_seen.is_set()
    assert database.close_calls == 1
    assert redis_client.close_calls == 1


def test_runner_without_registered_handlers_is_not_ready_to_claim_jobs() -> None:
    repository = _MemoryJobRepository(_lease())
    runner = _runner(repository)

    assert runner.has_handlers is False
    runner.register("ingest_document", lambda _context: asyncio.sleep(0))
    assert runner.has_handlers is True


def test_health_main_reports_only_fresh_healthy_live_worker_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "worker-health.json"
    write_worker_health(
        path,
        WorkerHealthSnapshot(
            pid=321,
            process_running=True,
            dependencies_ok=True,
            accepting_jobs=True,
            checked_at=datetime.now(UTC),
        ),
    )
    monkeypatch.setattr("rag_service.jobs.worker._process_exists", lambda pid: pid == 321)

    assert health_main(["--health-file", str(path), "--max-age-seconds", "10"]) == 0

    write_worker_health(
        path,
        WorkerHealthSnapshot(
            pid=321,
            process_running=True,
            dependencies_ok=False,
            accepting_jobs=False,
            checked_at=datetime.now(UTC),
        ),
    )
    assert health_main(["--health-file", str(path), "--max-age-seconds", "10"]) == 1


def test_worker_signal_handlers_request_shutdown_and_are_removed() -> None:
    class SignalLoop:
        def __init__(self) -> None:
            self.callbacks: dict[object, object] = {}
            self.removed: list[object] = []

        def add_signal_handler(self, signum: object, callback: object) -> None:
            self.callbacks[signum] = callback

        def remove_signal_handler(self, signum: object) -> bool:
            self.removed.append(signum)
            return True

    loop = SignalLoop()
    stop_event = asyncio.Event()

    event_loop = cast(asyncio.AbstractEventLoop, loop)
    installed = _install_signal_handlers(event_loop, stop_event)
    callback = loop.callbacks[next(iter(loop.callbacks))]
    assert callable(callback)
    callback()

    assert stop_event.is_set()
    _remove_signal_handlers(event_loop, installed)
    assert loop.removed == list(installed)
