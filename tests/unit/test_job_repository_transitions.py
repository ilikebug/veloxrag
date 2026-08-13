from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any, cast
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from rag_service.jobs.repositories import (
    ExhaustedJob,
    JobLease,
    LostLeaseError,
    SqlAlchemyJobRepository,
)


class _Rows:
    def __init__(self, values: Sequence[object]) -> None:
        self._values = tuple(values)

    def all(self) -> list[object]:
        return list(self._values)


class _OneOrNone:
    def __init__(self, value: tuple[object, ...] | None) -> None:
        self._value = value

    def one_or_none(self) -> tuple[object, ...] | None:
        return self._value


class _SequenceSession:
    def __init__(
        self,
        *,
        scalar_values: Sequence[object] = (),
        candidates: Sequence[object] = (),
        execute_values: Sequence[tuple[object, ...] | None] = (),
    ) -> None:
        self.scalar_values = list(scalar_values)
        self.candidates = tuple(candidates)
        self.execute_values = list(execute_values)
        self.statements: list[object] = []
        self.flushes = 0

    async def scalar(self, statement: object) -> object:
        self.statements.append(statement)
        return self.scalar_values.pop(0)

    async def scalars(self, statement: object) -> _Rows:
        self.statements.append(statement)
        return _Rows(self.candidates)

    async def execute(self, statement: object) -> _OneOrNone:
        self.statements.append(statement)
        return _OneOrNone(self.execute_values.pop(0))

    async def flush(self) -> None:
        self.flushes += 1


def _job(**changes: object) -> SimpleNamespace:
    now = datetime.now(UTC)
    values: dict[str, object] = {
        "id": uuid4(),
        "knowledge_base_id": uuid4(),
        "operation": "ingest_document",
        "target_type": "document_version",
        "target_id": uuid4(),
        "target_revision": 1,
        "index_generation_id": uuid4(),
        "mutation_id": None,
        "stage": "validate",
        "resume_stage": "validate",
        "progress_current": 2,
        "progress_total": 3,
        "attempt_count": 1,
        "max_attempts": 5,
        "lease_owner": "worker-a",
        "lease_epoch": 7,
        "lease_expires_at": now + timedelta(minutes=1),
        "cancel_requested_at": None,
        "status": "running",
        "next_retry_at": None,
        "worker_heartbeat_at": now,
        "error_code": None,
        "error_message_sanitized": None,
        "started_at": now,
        "finished_at": None,
        "retryable": True,
    }
    values.update(changes)
    return SimpleNamespace(**values)


def _lease(**changes: object) -> JobLease:
    job = _job(**changes)
    return SqlAlchemyJobRepository.lease_from_job(cast(Any, job))


def _repository(session: _SequenceSession) -> SqlAlchemyJobRepository:
    return SqlAlchemyJobRepository(cast(AsyncSession, session))


async def _action(value: object) -> Callable[[AsyncSession], Awaitable[object]]:
    async def action(session: AsyncSession) -> object:
        assert session is not None
        return value

    return action


def test_job_lease_and_input_validation_reject_invalid_state() -> None:
    with pytest.raises(ValueError, match="does not hold a lease"):
        SqlAlchemyJobRepository.lease_from_job(cast(Any, _job(status="queued")))
    with pytest.raises(ValueError, match="does not hold a lease"):
        SqlAlchemyJobRepository.lease_from_job(cast(Any, _job(lease_owner=None)))
    with pytest.raises(ValueError, match="does not hold a lease"):
        SqlAlchemyJobRepository.lease_from_job(cast(Any, _job(lease_expires_at=None)))

    for current, total in ((-1, None), (1, -1), (2, 1)):
        with pytest.raises(ValueError, match="progress is invalid"):
            SqlAlchemyJobRepository._validate_progress(current, total)
    for owner, duration in (("", timedelta(seconds=1)), ("x" * 256, timedelta(seconds=1))):
        with pytest.raises(ValueError, match="owner is invalid"):
            SqlAlchemyJobRepository._validate_lease_input(owner, duration)
    with pytest.raises(ValueError, match="duration must be positive"):
        SqlAlchemyJobRepository._validate_lease_input("worker", timedelta(0))


@pytest.mark.asyncio
async def test_claim_next_handles_empty_exhausted_and_claimable_candidates() -> None:
    empty_session = _SequenceSession()
    assert (
        await _repository(empty_session).claim_next(
            lease_owner="worker-b",
            lease_duration=timedelta(seconds=30),
            job_id=uuid4(),
        )
        is None
    )

    now = datetime.now(UTC)
    exhausted = _job(
        status="queued",
        lease_owner=None,
        lease_expires_at=None,
        attempt_count=5,
        max_attempts=5,
    )
    exhausted_session = _SequenceSession(candidates=(exhausted,))
    candidate = await _repository(exhausted_session).claim_next(
        lease_owner="worker-b",
        lease_duration=timedelta(seconds=30),
    )
    assert isinstance(candidate, ExhaustedJob)
    assert candidate.id == exhausted.id
    assert exhausted.status == "queued"
    assert exhausted.error_code is None
    assert exhausted_session.flushes == 0

    claimable = _job(
        status="queued",
        lease_owner=None,
        lease_expires_at=None,
        worker_heartbeat_at=None,
        started_at=None,
        next_retry_at=now,
        error_code="OLD",
        error_message_sanitized="old",
    )
    claim_session = _SequenceSession(scalar_values=(now,), candidates=(claimable,))
    lease = await _repository(claim_session).claim_next(
        lease_owner="worker-b",
        lease_duration=timedelta(seconds=30),
    )
    assert lease is not None
    assert isinstance(lease, JobLease)
    assert lease.lease_owner == "worker-b"
    assert lease.attempt_count == 2
    assert lease.lease_epoch == 8
    assert lease.recovered is False
    assert claimable.started_at == now
    assert claimable.next_retry_at is None
    assert claimable.error_code is None

    expired_running = _job(
        status="running",
        lease_expires_at=now - timedelta(seconds=1),
    )
    recovery_session = _SequenceSession(scalar_values=(now,), candidates=(expired_running,))
    recovered = await _repository(recovery_session).claim_next(
        lease_owner="worker-c",
        lease_duration=timedelta(seconds=30),
    )
    assert isinstance(recovered, JobLease)
    assert recovered.recovered is True


@pytest.mark.asyncio
async def test_heartbeat_cancellation_release_and_checkpoint_are_fenced() -> None:
    lease = _lease()
    updated = _job(
        id=lease.id,
        lease_epoch=lease.lease_epoch,
        lease_owner=lease.lease_owner,
        lease_expires_at=lease.lease_expires_at,
    )
    heartbeat_session = _SequenceSession(scalar_values=(updated,))
    heartbeat = await _repository(heartbeat_session).heartbeat(
        lease,
        timedelta(seconds=20),
    )
    assert heartbeat.id == lease.id
    with pytest.raises(ValueError, match="duration must be positive"):
        await _repository(_SequenceSession()).heartbeat(lease, timedelta(0))
    with pytest.raises(LostLeaseError):
        await _repository(_SequenceSession(scalar_values=(None,))).heartbeat(
            lease,
            timedelta(seconds=20),
        )

    assert (
        await _repository(_SequenceSession(execute_values=((None,),))).cancellation_requested(lease)
        is False
    )
    assert (
        await _repository(
            _SequenceSession(execute_values=((datetime.now(UTC),),))
        ).cancellation_requested(lease)
        is True
    )
    with pytest.raises(LostLeaseError):
        await _repository(_SequenceSession(execute_values=(None,))).cancellation_requested(lease)

    await _repository(_SequenceSession(scalar_values=(lease.id,))).release_claim(lease)
    with pytest.raises(LostLeaseError):
        await _repository(_SequenceSession(scalar_values=(None,))).release_claim(lease)

    await _repository(_SequenceSession(scalar_values=(lease.id,))).checkpoint(
        lease,
        stage="chunk",
        resume_stage="chunk",
        progress_current=1,
        progress_total=2,
    )
    with pytest.raises(LostLeaseError):
        await _repository(_SequenceSession(scalar_values=(None,))).checkpoint(
            lease,
            stage=None,
            resume_stage=None,
            progress_current=0,
            progress_total=None,
        )


@pytest.mark.asyncio
async def test_advance_and_stage_commits_require_both_fences() -> None:
    lease = _lease()
    updated = _job(
        id=lease.id,
        lease_epoch=lease.lease_epoch,
        lease_owner=lease.lease_owner,
        lease_expires_at=lease.lease_expires_at,
        stage="activate",
        resume_stage="activate",
        progress_current=3,
        progress_total=3,
    )
    action = await _action("committed")

    session = _SequenceSession(scalar_values=(lease.id, updated))
    result, advanced = await _repository(session).advance_stage(
        lease,
        action,
        stage="activate",
        resume_stage="activate",
        progress_current=3,
        progress_total=3,
    )
    assert result == "committed"
    assert advanced.stage == "activate"
    for stage, resume_stage in (("", "activate"), ("activate", ""), ("x" * 65, "activate")):
        with pytest.raises(ValueError, match="stage is invalid"):
            await _repository(_SequenceSession()).advance_stage(
                lease,
                action,
                stage=stage,
                resume_stage=resume_stage,
                progress_current=0,
                progress_total=None,
            )
    with pytest.raises(LostLeaseError):
        await _repository(_SequenceSession(scalar_values=(None,))).advance_stage(
            lease,
            action,
            stage="activate",
            resume_stage="activate",
            progress_current=0,
            progress_total=None,
        )
    with pytest.raises(LostLeaseError):
        await _repository(_SequenceSession(scalar_values=(lease.id, None))).advance_stage(
            lease,
            action,
            stage="activate",
            resume_stage="activate",
            progress_current=0,
            progress_total=None,
        )

    assert (
        await _repository(_SequenceSession(scalar_values=(lease.id, lease.id))).commit_stage_facts(
            lease,
            action,
        )
        == "committed"
    )
    with pytest.raises(LostLeaseError):
        await _repository(_SequenceSession(scalar_values=(None,))).commit_stage_facts(lease, action)
    with pytest.raises(LostLeaseError):
        await _repository(_SequenceSession(scalar_values=(lease.id, None))).commit_stage_facts(
            lease, action
        )

    checkpoint_session = _SequenceSession(scalar_values=(lease.id, updated))
    result, checkpointed = await _repository(checkpoint_session).commit_stage_checkpoint(
        lease,
        action,
        progress_current=3,
        progress_total=3,
    )
    assert result == "committed"
    assert checkpointed.progress_current == 3
    with pytest.raises(LostLeaseError):
        await _repository(_SequenceSession(scalar_values=(None,))).commit_stage_checkpoint(
            lease,
            action,
            progress_current=0,
            progress_total=None,
        )
    with pytest.raises(LostLeaseError):
        await _repository(_SequenceSession(scalar_values=(lease.id, None))).commit_stage_checkpoint(
            lease,
            action,
            progress_current=0,
            progress_total=None,
        )


@pytest.mark.asyncio
async def test_activation_terminal_and_status_writes_are_fenced() -> None:
    lease = _lease()
    action = await _action("done")

    assert (
        await _repository(_SequenceSession(scalar_values=(lease.id, lease.id))).prepare_activation(
            lease, action
        )
        == "done"
    )
    with pytest.raises(LostLeaseError):
        await _repository(_SequenceSession(scalar_values=(None,))).prepare_activation(lease, action)
    with pytest.raises(LostLeaseError):
        await _repository(_SequenceSession(scalar_values=(lease.id, None))).prepare_activation(
            lease, action
        )

    assert (
        await _repository(_SequenceSession(scalar_values=("failed",))).finalize_domain(
            lease,
            action,
        )
        == "done"
    )
    with pytest.raises(LostLeaseError):
        await _repository(_SequenceSession(scalar_values=(None,))).finalize_domain(lease, action)

    exhausted = SqlAlchemyJobRepository.exhausted_from_job(
        cast(Any, _job(attempt_count=5, max_attempts=5))
    )
    await _repository(_SequenceSession(scalar_values=(exhausted.id,))).finalize_exhausted(exhausted)
    with pytest.raises(LostLeaseError):
        await _repository(_SequenceSession(scalar_values=(None,))).finalize_exhausted(exhausted)
    exhausted_action = await _action(None)
    await _repository(_SequenceSession(scalar_values=(exhausted.id,))).finalize_exhausted(
        exhausted,
        cast(Callable[[AsyncSession], Awaitable[None]], exhausted_action),
    )
    with pytest.raises(LostLeaseError):
        await _repository(_SequenceSession(scalar_values=(None,))).finalize_exhausted(
            exhausted,
            cast(Callable[[AsyncSession], Awaitable[None]], exhausted_action),
        )

    assert (
        await _repository(_SequenceSession(scalar_values=("succeeded",))).mark_succeeded(lease)
        == "succeeded"
    )
    with pytest.raises(LostLeaseError):
        await _repository(_SequenceSession(scalar_values=(None,))).mark_succeeded(lease)
    await _repository(_SequenceSession(scalar_values=(lease.id,))).mark_cancelled(lease)
    with pytest.raises(LostLeaseError):
        await _repository(_SequenceSession(scalar_values=(None,))).mark_cancelled(lease)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("retryable", "result"),
    [(True, "retry_wait"), (True, "failed"), (False, "failed")],
)
async def test_record_failure_builds_retry_cancel_and_terminal_transitions(
    retryable: bool,
    result: str,
) -> None:
    lease = _lease()
    repository = _repository(_SequenceSession(scalar_values=(result,)))
    assert (
        await repository.record_failure(
            lease,
            retryable=retryable,
            error_code="PROVIDER_UNAVAILABLE",
            error_message="Provider unavailable",
            retry_delay=timedelta(seconds=3),
        )
        == result
    )


@pytest.mark.asyncio
async def test_record_failure_rejects_invalid_data_and_lost_lease() -> None:
    lease = _lease()
    invalid = (
        (True, "", "message", timedelta(0)),
        (True, "x" * 65, "message", timedelta(0)),
        (True, "CODE", "", timedelta(0)),
        (True, "CODE", "x" * 501, timedelta(0)),
        (False, "CODE", "message", timedelta(seconds=-1)),
    )
    for retryable, code, message, delay in invalid:
        with pytest.raises(ValueError, match="failure is invalid"):
            await _repository(_SequenceSession()).record_failure(
                lease,
                retryable=retryable,
                error_code=code,
                error_message=message,
                retry_delay=delay,
            )
    with pytest.raises(LostLeaseError):
        await _repository(_SequenceSession(scalar_values=(None,))).record_failure(
            lease,
            retryable=True,
            error_code="PROVIDER_UNAVAILABLE",
            error_message="Provider unavailable",
            retry_delay=timedelta(seconds=3),
        )
