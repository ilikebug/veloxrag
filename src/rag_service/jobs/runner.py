"""Bounded Job execution with handler registration, heartbeat, and recovery polling."""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import AsyncIterator, Awaitable, Callable, Coroutine
from contextlib import AbstractAsyncContextManager, suppress
from dataclasses import dataclass
from datetime import timedelta
from enum import StrEnum
from typing import Any, Protocol
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from rag_service.jobs.repositories import (
    ExhaustedJob,
    JobLease,
    JobRepository,
    LostLeaseError,
    RebuildTargetRefresh,
)
from rag_service.observability.logging import SafeLogContext, emit_safe_log
from rag_service.observability.metrics import METRICS, OperationalMetrics

logger = logging.getLogger(__name__)


class JobHandlerOutcome(StrEnum):
    COMPLETE = "complete"
    CONTINUE = "continue"


type JobHandler = Callable[["JobExecutionContext"], Coroutine[Any, Any, JobHandlerOutcome | None]]
type JobExhaustionFinalizer = Callable[[ExhaustedJob, AsyncSession], Awaitable[None]]
type RepositoryContextFactory = Callable[[], AbstractAsyncContextManager[JobRepository]]
type DependencyPreflight = Callable[[], Awaitable[bool]]


def _consume_task_result(task: asyncio.Task[Any]) -> None:
    with suppress(asyncio.CancelledError, Exception):
        task.exception()


class _ClaimAborted(RuntimeError):
    pass


class JobNotificationSource(Protocol):
    async def get(self, timeout_seconds: float) -> UUID | None: ...


class _NotificationGate:
    """Keep at most one external notification read in flight."""

    def __init__(self, source: JobNotificationSource) -> None:
        self._source = source
        self._task: asyncio.Task[UUID | None] | None = None

    async def wait(
        self,
        stop_event: asyncio.Event,
        timeout_seconds: float,
    ) -> tuple[bool, bool, UUID | None]:
        """Return ``(stopped, available, job_id)`` within the caller's deadline."""

        if timeout_seconds < 0:
            raise ValueError("notification timeout must not be negative")
        if stop_event.is_set():
            return True, True, None
        existing = self._task
        if existing is not None:
            if not existing.done():
                return False, False, None
            self._task = None

        notification_task = asyncio.create_task(self._source.get(timeout_seconds))
        notification_task.add_done_callback(_consume_task_result)
        self._task = notification_task
        stop_task = asyncio.create_task(stop_event.wait())
        stop_task.add_done_callback(_consume_task_result)
        try:
            completed, _pending = await asyncio.wait(
                (notification_task, stop_task),
                timeout=timeout_seconds,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if stop_task in completed:
                notification_task.cancel()
                await asyncio.sleep(0)
                return True, True, None
            if notification_task in completed:
                return False, True, notification_task.result()

            notification_task.cancel()
            await asyncio.sleep(0)
            return False, notification_task.done(), None
        finally:
            if not stop_task.done():
                stop_task.cancel()

    def cancel(self) -> None:
        task = self._task
        if task is not None and not task.done():
            task.cancel()


class JobExecutionError(Exception):
    retryable: bool

    def __init__(self, code: str, message: str, *, retryable: bool) -> None:
        if (
            type(code) is not str
            or not code
            or len(code) > 64
            or type(message) is not str
            or not message
            or len(message) > 500
        ):
            raise ValueError("job execution error is invalid")
        self.code = code
        self.safe_message = message
        self.retryable = retryable
        super().__init__(message)


class RetryableJobError(JobExecutionError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(code, message, retryable=True)


class RetryableProviderJobError(RetryableJobError):
    """Retryable Provider failure carrying only a finite Provider type."""

    def __init__(self, code: str, message: str, *, provider_type: str) -> None:
        if provider_type not in {"openai_compatible", "openrouter", "vendor_specific"}:
            raise ValueError("provider type is invalid")
        self.provider_type = provider_type
        super().__init__(code, message)


class PermanentJobError(JobExecutionError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(code, message, retryable=False)


@dataclass(frozen=True, slots=True)
class ExponentialBackoff:
    initial_seconds: float
    maximum_seconds: float
    jitter_fraction: float = 0.2
    random_fraction: Callable[[], float] = random.random

    def __post_init__(self) -> None:
        if (
            self.initial_seconds <= 0
            or self.maximum_seconds <= 0
            or self.initial_seconds > self.maximum_seconds
            or not 0 <= self.jitter_fraction < 1
        ):
            raise ValueError("retry backoff is invalid")

    def delay(self, attempt_count: int) -> timedelta:
        if attempt_count < 1:
            raise ValueError("attempt count must be positive")
        base = min(
            self.maximum_seconds,
            self.initial_seconds * (2 ** (attempt_count - 1)),
        )
        lower = base * (1 - self.jitter_fraction)
        upper = min(self.maximum_seconds, base * (1 + self.jitter_fraction))
        fraction = self.random_fraction()
        if not 0 <= fraction <= 1:
            raise ValueError("random fraction must be between zero and one")
        return timedelta(seconds=lower + ((upper - lower) * fraction))


class JobExecutionContext:
    def __init__(
        self,
        repository_context: RepositoryContextFactory,
        lease: JobLease,
        lease_duration: timedelta,
        *,
        domain_finalization_enabled: bool = False,
    ) -> None:
        self._repository_context = repository_context
        self._lease = lease
        self._lease_duration = lease_duration
        self._cancel_requested = False
        self._lease_lost = False
        self._stage_advance_count = 0
        # Only runner-owned contexts may atomically terminalize domain state.
        # Direct stage contexts retain exception-only behavior for isolated calls.
        self._domain_finalization_enabled = domain_finalization_enabled
        self._finalization_in_progress = False
        self._finalized = False
        self._finalized_status: str | None = None

    @property
    def lease(self) -> JobLease:
        return self._lease

    @property
    def cancel_requested(self) -> bool:
        return self._cancel_requested

    @property
    def lease_lost(self) -> bool:
        return self._lease_lost

    @property
    def stage_advance_count(self) -> int:
        return self._stage_advance_count

    @property
    def finalization_in_progress(self) -> bool:
        return self._finalization_in_progress

    @property
    def finalized(self) -> bool:
        return self._finalized

    @property
    def finalized_status(self) -> str | None:
        return self._finalized_status

    @property
    def domain_finalization_enabled(self) -> bool:
        return self._domain_finalization_enabled

    async def heartbeat(self) -> None:
        async with self._repository_context() as repository:
            self._lease = await repository.heartbeat(self._lease, self._lease_duration)

    async def check_cancellation(self) -> bool:
        async with self._repository_context() as repository:
            requested = await repository.cancellation_requested(self._lease)
        self._cancel_requested = requested
        return requested

    async def checkpoint(
        self,
        *,
        stage: str | None,
        resume_stage: str | None,
        progress_current: int,
        progress_total: int | None,
    ) -> None:
        async with self._repository_context() as repository:
            await repository.checkpoint(
                self._lease,
                stage=stage,
                resume_stage=resume_stage,
                progress_current=progress_current,
                progress_total=progress_total,
            )

    async def reset_rebuild_target(
        self,
        action: Callable[[AsyncSession], Awaitable[RebuildTargetRefresh]],
    ) -> RebuildTargetRefresh:
        async with self._repository_context() as repository:
            result, updated_lease = await repository.reset_rebuild_target(
                self._lease,
                action,
            )
        self._lease = updated_lease
        return result

    async def prepare_activation[Result](
        self,
        action: Callable[[AsyncSession], Awaitable[Result]],
    ) -> Result:
        async with self._repository_context() as repository:
            return await repository.prepare_activation(self._lease, action)

    async def finalize_domain[Result](
        self,
        action: Callable[[AsyncSession], Awaitable[Result]],
    ) -> Result:
        if self._finalization_in_progress or self._finalized:
            raise RuntimeError("job finalization is invalid")
        self._finalization_in_progress = True
        finalized_status: str | None = None

        def observe_terminal_status(status: str) -> None:
            nonlocal finalized_status
            if status not in {"succeeded", "failed", "cancelled"}:
                raise LostLeaseError
            finalized_status = status

        try:
            async with self._repository_context() as repository:
                result = await repository.finalize_domain(
                    self._lease,
                    action,
                    terminal_status_observer=observe_terminal_status,
                )
        except BaseException:
            raise
        else:
            if finalized_status is None:
                raise LostLeaseError
            self._finalized = True
            self._finalized_status = finalized_status
            return result
        finally:
            self._finalization_in_progress = False

    async def commit_stage_facts[Result](
        self,
        action: Callable[[AsyncSession], Awaitable[Result]],
    ) -> Result:
        async with self._repository_context() as repository:
            return await repository.commit_stage_facts(self._lease, action)

    async def commit_stage_checkpoint[Result](
        self,
        action: Callable[[AsyncSession], Awaitable[Result]],
        *,
        progress_current: int,
        progress_total: int | None,
    ) -> Result:
        async with self._repository_context() as repository:
            result, updated_lease = await repository.commit_stage_checkpoint(
                self._lease,
                action,
                progress_current=progress_current,
                progress_total=progress_total,
            )
        self._lease = updated_lease
        return result

    async def advance_stage[Result](
        self,
        action: Callable[[AsyncSession], Awaitable[Result]],
        *,
        stage: str,
        resume_stage: str,
        progress_current: int,
        progress_total: int | None,
    ) -> Result:
        async with self._repository_context() as repository:
            result, updated_lease = await repository.advance_stage(
                self._lease,
                action,
                stage=stage,
                resume_stage=resume_stage,
                progress_current=progress_current,
                progress_total=progress_total,
            )
        self._lease = updated_lease
        self._stage_advance_count += 1
        return result

    def _mark_lease_lost(self) -> None:
        self._lease_lost = True


class JobRunner:
    def __init__(
        self,
        *,
        repository_context: RepositoryContextFactory,
        lease_owner: str,
        lease_seconds: float,
        heartbeat_seconds: float,
        poll_interval_seconds: float,
        max_concurrency: int,
        backoff: ExponentialBackoff,
        preflight: DependencyPreflight | None = None,
        shutdown_seconds: float = 30.0,
        metrics: OperationalMetrics = METRICS,
    ) -> None:
        if (
            type(lease_owner) is not str
            or not lease_owner
            or lease_seconds <= 0
            or heartbeat_seconds <= 0
            or heartbeat_seconds >= lease_seconds
            or poll_interval_seconds <= 0
            or max_concurrency < 1
            or shutdown_seconds <= 0
        ):
            raise ValueError("job runner configuration is invalid")
        self._repository_context = repository_context
        self._lease_owner = lease_owner
        self._lease_duration = timedelta(seconds=lease_seconds)
        self._heartbeat_seconds = heartbeat_seconds
        self._poll_interval_seconds = poll_interval_seconds
        self._max_concurrency = max_concurrency
        self._backoff = backoff
        self._preflight = preflight
        self._shutdown_seconds = shutdown_seconds
        self._metrics = metrics
        self._handlers: dict[str, JobHandler] = {}
        self._exhaustion_finalizers: dict[str, JobExhaustionFinalizer] = {}

    @staticmethod
    def _log_context(job: JobLease | ExhaustedJob) -> SafeLogContext:
        return SafeLogContext(
            knowledge_base_id=(job.knowledge_base_id if isinstance(job, ExhaustedJob) else None),
            version_id=job.target_id if job.target_type == "document_version" else None,
            job_id=job.id,
            generation_id=job.index_generation_id,
        )

    def _observe_job_state(self, job: JobLease | ExhaustedJob, state: str) -> None:
        with suppress(BaseException):
            self._metrics.record_job_state(state=state)
        try:
            fields: dict[str, object] = {
                "operation": "job",
                "state": state,
                "attempt": job.attempt_count,
            }
            if state == "failed":
                fields["error_code"] = "JOB_FAILED"
            emit_safe_log(
                logger,
                logging.INFO,
                "job.state.changed",
                context=self._log_context(job),
                **fields,
            )
        except BaseException:
            pass

    def _observe_lease_recovery(self, job: JobLease | ExhaustedJob) -> None:
        with suppress(BaseException):
            self._metrics.record_lease_recovery()
        with suppress(BaseException):
            emit_safe_log(
                logger,
                logging.INFO,
                "job.lease.recovered",
                context=self._log_context(job),
                operation="job",
                recovered=True,
                attempt=job.attempt_count,
            )

    def _observe_provider_retry(self, lease: JobLease, provider_type: str) -> None:
        with suppress(BaseException):
            self._metrics.record_provider_retry(provider_type=provider_type)
        with suppress(BaseException):
            emit_safe_log(
                logger,
                logging.INFO,
                "provider.retry.scheduled",
                context=self._log_context(lease),
                operation="provider_request",
                provider_type=provider_type,
                state="retry_wait",
                retryable=True,
            )

    def _observe_domain_finalization(self, context: JobExecutionContext) -> None:
        if context.finalized_status is not None:
            self._observe_job_state(context.lease, context.finalized_status)

    @property
    def has_handlers(self) -> bool:
        return bool(self._handlers)

    def register(
        self,
        operation: str,
        handler: JobHandler,
        *,
        exhaustion_finalizer: JobExhaustionFinalizer | None = None,
    ) -> None:
        if type(operation) is not str or not operation or operation in self._handlers:
            raise ValueError("job handler registration is invalid")
        self._handlers[operation] = handler
        if exhaustion_finalizer is not None:
            self._exhaustion_finalizers[operation] = exhaustion_finalizer

    async def _claim(
        self,
        job_id: UUID | None,
        stop_event: asyncio.Event | None,
    ) -> JobLease | ExhaustedJob | None:
        if stop_event is not None and stop_event.is_set():
            return None
        if self._preflight is not None and not await self._preflight():
            return None
        try:
            async with self._repository_context() as repository:
                lease = await repository.claim_next(
                    lease_owner=self._lease_owner,
                    lease_duration=self._lease_duration,
                    job_id=job_id,
                )
                if lease is not None and stop_event is not None and stop_event.is_set():
                    raise _ClaimAborted
        except _ClaimAborted:
            return None
        if isinstance(lease, JobLease):
            self._observe_job_state(lease, "running")
            if lease.recovered:
                self._observe_lease_recovery(lease)
        if isinstance(lease, JobLease) and stop_event is not None and stop_event.is_set():
            await self._release_claim(lease)
            return None
        return lease

    async def _finalize_exhausted(self, candidate: ExhaustedJob) -> bool:
        finalizer = self._exhaustion_finalizers.get(candidate.operation)
        try:
            async with self._repository_context() as repository:
                if finalizer is None:
                    await repository.finalize_exhausted(candidate)
                else:

                    async def action(session: AsyncSession) -> None:
                        await finalizer(candidate, session)

                    await repository.finalize_exhausted(candidate, action)
        except LostLeaseError:
            return False
        self._observe_job_state(candidate, "failed")
        if candidate.status == "running" and candidate.lease_expires_at is not None:
            self._observe_lease_recovery(candidate)
        return True

    async def _release_claim(self, lease: JobLease) -> None:
        async with self._repository_context() as repository:
            await repository.release_claim(lease)
        self._observe_job_state(lease, "queued")

    async def _maintain_lease(
        self,
        context: JobExecutionContext,
        handler_task: asyncio.Task[None],
    ) -> None:
        while not handler_task.done():
            await asyncio.sleep(self._heartbeat_seconds)
            if handler_task.done():
                return
            try:
                await context.heartbeat()
                if await context.check_cancellation():
                    handler_task.cancel()
                    return
            except Exception:
                if context.finalization_in_progress or context.finalized:
                    return
                context._mark_lease_lost()
                handler_task.cancel()
                return

    async def _mark_succeeded(self, lease: JobLease) -> str:
        async with self._repository_context() as repository:
            status = await repository.mark_succeeded(lease)
        self._observe_job_state(lease, status)
        return status

    async def _mark_cancelled(self, lease: JobLease) -> None:
        async with self._repository_context() as repository:
            await repository.mark_cancelled(lease)
        self._observe_job_state(lease, "cancelled")

    async def _record_failure(
        self,
        lease: JobLease,
        error: JobExecutionError,
    ) -> str:
        retry_delay = self._backoff.delay(lease.attempt_count) if error.retryable else timedelta(0)
        async with self._repository_context() as repository:
            status = await repository.record_failure(
                lease,
                retryable=error.retryable,
                error_code=error.code,
                error_message=error.safe_message,
                retry_delay=retry_delay,
            )
        self._observe_job_state(lease, status)
        if status == "retry_wait" and isinstance(error, RetryableProviderJobError):
            self._observe_provider_retry(lease, error.provider_type)
        return status

    @staticmethod
    async def _run_handler_loop(
        handler: JobHandler,
        context: JobExecutionContext,
    ) -> None:
        while True:
            previous_advance_count = context._stage_advance_count
            outcome = await handler(context)
            advanced = context._stage_advance_count - previous_advance_count
            if outcome is JobHandlerOutcome.CONTINUE:
                if advanced != 1:
                    raise PermanentJobError("JOB_HANDLER_FAILED", "Job handler failed")
                continue
            if outcome is not None and outcome is not JobHandlerOutcome.COMPLETE:
                raise PermanentJobError("JOB_HANDLER_FAILED", "Job handler failed")
            if advanced != 0:
                raise PermanentJobError("JOB_HANDLER_FAILED", "Job handler failed")
            return

    async def run_once(
        self,
        job_id: UUID | None = None,
        *,
        stop_event: asyncio.Event | None = None,
    ) -> bool:
        lease = await self._claim(job_id, stop_event)
        if lease is None:
            return False
        if isinstance(lease, ExhaustedJob):
            if stop_event is not None and stop_event.is_set():
                return False
            return await self._finalize_exhausted(lease)
        if stop_event is not None and stop_event.is_set():
            try:
                await self._release_claim(lease)
            except LostLeaseError:
                return False
            return False
        handler = self._handlers.get(lease.operation)
        context = JobExecutionContext(
            self._repository_context,
            lease,
            self._lease_duration,
            domain_finalization_enabled=True,
        )
        try:
            if await context.check_cancellation():
                await self._mark_cancelled(context.lease)
                return True
        except Exception:
            return False
        if stop_event is not None and stop_event.is_set():
            try:
                await self._release_claim(context.lease)
            except LostLeaseError:
                return False
            return False
        if handler is None:
            try:
                await self._record_failure(
                    lease,
                    PermanentJobError("JOB_HANDLER_UNAVAILABLE", "Job handler is unavailable"),
                )
            except LostLeaseError:
                return False
            return True

        handler_task: asyncio.Task[None] = asyncio.create_task(
            self._run_handler_loop(handler, context)
        )
        heartbeat_task = asyncio.create_task(self._maintain_lease(context, handler_task))
        try:
            await handler_task
        except asyncio.CancelledError:
            if context.finalized:
                self._observe_domain_finalization(context)
                return True
            if context.lease_lost:
                return False
            if context.cancel_requested:
                try:
                    await self._mark_cancelled(context.lease)
                except LostLeaseError:
                    return False
                return True
            raise
        except LostLeaseError:
            return False
        except JobExecutionError as error:
            if context.finalized:
                self._observe_domain_finalization(context)
                return True
            try:
                await self._record_failure(context.lease, error)
            except LostLeaseError:
                return False
            return True
        except Exception:
            if context.finalized:
                self._observe_domain_finalization(context)
                return True
            try:
                await self._record_failure(
                    context.lease,
                    PermanentJobError("JOB_HANDLER_FAILED", "Job handler failed"),
                )
            except LostLeaseError:
                return False
            return True
        else:
            if context.finalized:
                self._observe_domain_finalization(context)
                return True
            try:
                await self._mark_succeeded(context.lease)
            except LostLeaseError:
                return False
            return True
        finally:
            heartbeat_task.cancel()
            with suppress(asyncio.CancelledError):
                await heartbeat_task

    async def poll_once(self) -> bool:
        return await self.run_once()

    async def run(
        self,
        stop_event: asyncio.Event,
        *,
        notifications: JobNotificationSource | None = None,
    ) -> None:
        active: set[asyncio.Task[bool]] = set()
        loop = asyncio.get_running_loop()
        next_poll = loop.time()
        notification_gate = _NotificationGate(notifications) if notifications is not None else None
        notifications_enabled = notification_gate is not None
        try:
            while not stop_event.is_set():
                completed = {task for task in active if task.done()}
                active.difference_update(completed)
                if completed:
                    await asyncio.gather(*completed, return_exceptions=True)

                if len(active) >= self._max_concurrency:
                    stop_wait = asyncio.create_task(stop_event.wait())
                    try:
                        completed_or_stopped: set[asyncio.Task[Any]] = set(active)
                        completed_or_stopped.add(stop_wait)
                        completed, _pending = await asyncio.wait(
                            completed_or_stopped,
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                        if stop_wait in completed:
                            break
                    finally:
                        stop_wait.cancel()
                        with suppress(asyncio.CancelledError):
                            await stop_wait
                    continue

                now = loop.time()
                job_id: UUID | None = None
                if now >= next_poll:
                    next_poll = now + self._poll_interval_seconds
                elif notifications_enabled and notification_gate is not None:
                    try:
                        stopped, available, job_id = await notification_gate.wait(
                            stop_event,
                            max(0.0, next_poll - now),
                        )
                    except Exception:
                        notifications_enabled = False
                        continue
                    if stopped:
                        break
                    if not available:
                        notifications_enabled = False
                        continue
                    if job_id is None:
                        next_poll = loop.time() + self._poll_interval_seconds
                else:
                    try:
                        await asyncio.wait_for(
                            stop_event.wait(),
                            timeout=max(0.0, next_poll - now),
                        )
                    except TimeoutError:
                        next_poll = loop.time() + self._poll_interval_seconds
                    else:
                        break

                if stop_event.is_set():
                    break
                active.add(
                    asyncio.create_task(
                        self.run_once(job_id, stop_event=stop_event),
                    )
                )
                await asyncio.sleep(0)
        finally:
            if notification_gate is not None:
                notification_gate.cancel()
            shutdown_deadline = loop.time() + self._shutdown_seconds
            cancellation_grace = min(0.05, self._shutdown_seconds * 0.1)
            pending = active
            if active:
                completed, pending = await asyncio.wait(
                    active,
                    timeout=max(
                        0.0,
                        shutdown_deadline - loop.time() - cancellation_grace,
                    ),
                )
                if completed:
                    await asyncio.gather(*completed, return_exceptions=True)
            for task in pending:
                task.cancel()
            if pending:
                completed, detached = await asyncio.wait(
                    pending,
                    timeout=max(0.0, shutdown_deadline - loop.time()),
                )
                if completed:
                    await asyncio.gather(*completed, return_exceptions=True)
                for task in detached:
                    task.add_done_callback(_consume_task_result)


async def iter_notifications(
    source: JobNotificationSource,
    timeout_seconds: float,
) -> AsyncIterator[UUID]:
    """Small adapter useful for embedding the notification source in other loops."""

    while True:
        job_id = await source.get(timeout_seconds)
        if job_id is not None:
            yield job_id


__all__ = [
    "ExponentialBackoff",
    "JobExecutionContext",
    "JobExecutionError",
    "JobExhaustionFinalizer",
    "JobHandler",
    "JobHandlerOutcome",
    "JobNotificationSource",
    "JobRunner",
    "PermanentJobError",
    "RepositoryContextFactory",
    "RetryableJobError",
    "RetryableProviderJobError",
]
