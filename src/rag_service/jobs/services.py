"""Authorization and transactional orchestration for public Job APIs."""

from __future__ import annotations

import asyncio
import logging
import math
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager, suppress
from typing import Protocol, cast
from uuid import UUID

from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from rag_service.api.errors import BusinessError
from rag_service.api.validation import validate_idempotency_key
from rag_service.auth.policies import AgentPrincipal, Capability
from rag_service.jobs.repositories import (
    JobStatusRecord,
    ManualRetrySnapshot,
    SqlAlchemyJobRepository,
)
from rag_service.jobs.schemas import JobStatus, SafeJob
from rag_service.observability.logging import SafeLogContext, emit_safe_log
from rag_service.observability.metrics import METRICS, OperationalMetrics

type SessionFactory = Callable[[], AbstractAsyncContextManager[AsyncSession]]

_ACTIVE_TARGET_UNIQUE_INDEX = "uq_jobs_active_target"
_MAX_EXCEPTION_NODES = 32
logger = logging.getLogger(__name__)


class JobNotifier(Protocol):
    async def notify(self, job_id: UUID) -> bool: ...


def _hidden_job() -> BusinessError:
    return BusinessError(404, "RESOURCE_NOT_FOUND", "Resource not found")


def _retry_conflict() -> BusinessError:
    return BusinessError(
        409,
        "JOB_RETRY_CONFLICT",
        "Job retry conflicts with active work",
        retryable=True,
    )


def _internal_error() -> BusinessError:
    return BusinessError(500, "INTERNAL_ERROR", "Internal server error")


def _constraint_name(error: IntegrityError) -> str | None:
    pending: list[BaseException] = [error]
    visited: set[int] = set()
    while pending and len(visited) < _MAX_EXCEPTION_NODES:
        current = pending.pop()
        identity = id(current)
        if identity in visited:
            continue
        visited.add(identity)
        name = getattr(current, "constraint_name", None)
        if isinstance(name, str):
            return name
        diagnostic = getattr(current, "diag", None)
        name = getattr(diagnostic, "constraint_name", None)
        if isinstance(name, str):
            return name
        original = getattr(current, "orig", None)
        if isinstance(original, BaseException):
            pending.append(original)
        for nested in (current.__cause__, current.__context__):
            if nested is not None:
                pending.append(nested)
    return None


def _safe_job(record: JobStatusRecord) -> SafeJob:
    return SafeJob(
        id=record.id,
        operation=record.operation,
        status=cast(JobStatus, record.status),
        stage=record.stage,
        progress_current=record.progress_current,
        progress_total=record.progress_total,
        attempt_count=record.attempt_count,
        max_attempts=record.max_attempts,
        retryable=record.retryable,
        error_code=record.error_code,
        error_message=record.error_message,
        parent_job_id=record.parent_job_id,
        root_job_id=record.root_job_id,
        created_at=record.created_at,
        started_at=record.started_at,
        finished_at=record.finished_at,
    )


class JobService:
    def __init__(
        self,
        *,
        session: AsyncSession,
        reservation_session_factory: SessionFactory,
        max_idempotency_key_length: int,
        notifier: JobNotifier | None = None,
        notification_timeout_seconds: float = 0.25,
        metrics: OperationalMetrics = METRICS,
    ) -> None:
        if (
            not callable(reservation_session_factory)
            or type(max_idempotency_key_length) is not int
            or max_idempotency_key_length < 1
            or isinstance(notification_timeout_seconds, bool)
            or not isinstance(notification_timeout_seconds, (int, float))
            or not math.isfinite(notification_timeout_seconds)
            or notification_timeout_seconds <= 0
        ):
            raise ValueError("Job service dependencies are invalid")
        self._repository = SqlAlchemyJobRepository(session)
        self._reservation_session_factory = reservation_session_factory
        self._max_idempotency_key_length = max_idempotency_key_length
        self._notifier = notifier
        self._notification_timeout_seconds = notification_timeout_seconds
        self._metrics = metrics

    @staticmethod
    def _authorize(
        actor: AgentPrincipal,
        knowledge_base_id: UUID | None,
        capabilities: frozenset[Capability],
    ) -> None:
        if (
            knowledge_base_id is None
            or knowledge_base_id not in actor.knowledge_base_ids
            or actor.capabilities.isdisjoint(capabilities)
        ):
            raise _hidden_job()

    async def get_job(self, job_id: UUID, *, actor: AgentPrincipal) -> SafeJob:
        record = await self._repository.get_status(job_id)
        if record is None:
            raise _hidden_job()
        self._authorize(
            actor,
            record.knowledge_base_id,
            frozenset({Capability.MANAGE, Capability.INGEST, Capability.RETRIEVE}),
        )
        return _safe_job(record)

    @staticmethod
    def _consume_notification(task: asyncio.Task[bool]) -> None:
        with suppress(BaseException):
            task.result()

    async def _notify_after_commit(self, job_id: UUID) -> None:
        notifier = self._notifier
        if notifier is None:
            return
        task = asyncio.create_task(
            notifier.notify(job_id),
            name="manual-job-retry-notification",
        )
        task.add_done_callback(self._consume_notification)
        try:
            completed, _pending = await asyncio.wait(
                (task,),
                timeout=self._notification_timeout_seconds,
            )
        except asyncio.CancelledError:
            task.cancel()
            raise
        if task not in completed:
            task.cancel()
            logger.warning(
                "Manual retry notification timed out after reservation commit",
                extra={"job_id": str(job_id)},
            )
            return
        try:
            task.result()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning(
                "Manual retry notification failed after reservation commit",
                extra={"job_id": str(job_id)},
            )

    def _observe_queued_retry(
        self,
        *,
        snapshot: ManualRetrySnapshot,
        job: JobStatusRecord,
        request_id: str | None,
    ) -> None:
        with suppress(BaseException):
            self._metrics.record_job_state(state="queued")
        try:
            emit_safe_log(
                logger,
                logging.INFO,
                "job.state.changed",
                context=SafeLogContext(
                    request_id=request_id,
                    knowledge_base_id=job.knowledge_base_id,
                    document_id=snapshot.document_id,
                    version_id=(
                        snapshot.target_id if snapshot.target_type == "document_version" else None
                    ),
                    job_id=job.id,
                    generation_id=snapshot.index_generation_id,
                ),
                operation=job.operation,
                state="queued",
            )
        except BaseException:
            return

    async def retry_job(
        self,
        job_id: UUID,
        *,
        actor: AgentPrincipal,
        idempotency_key: str,
        request_id: str | None = None,
    ) -> SafeJob:
        key = validate_idempotency_key(
            idempotency_key,
            self._max_idempotency_key_length,
        )
        snapshot = await self._repository.get_manual_retry_snapshot(job_id)
        if snapshot is None:
            raise _hidden_job()
        retry_capabilities = (
            frozenset({Capability.MANAGE, Capability.INGEST})
            if snapshot.operation in {"ingest_document", "rebuild_generation"}
            else frozenset()
        )
        self._authorize(actor, snapshot.knowledge_base_id, retry_capabilities)

        try:
            async with self._reservation_session_factory() as session, session.begin():
                reservation = await SqlAlchemyJobRepository(session).reserve_manual_retry(
                    snapshot,
                    actor=actor,
                    idempotency_key=key,
                )
        except IntegrityError as error:
            if _constraint_name(error) == _ACTIVE_TARGET_UNIQUE_INDEX:
                raise _retry_conflict() from None
            raise _internal_error() from None

        if reservation.created:
            self._observe_queued_retry(
                snapshot=snapshot,
                job=reservation.job,
                request_id=request_id,
            )
            await self._notify_after_commit(reservation.job.id)
        return _safe_job(reservation.job)


__all__ = ["JobNotifier", "JobService", "SessionFactory"]
