"""Safe Job status and manual retry endpoints."""

from typing import Annotated, cast
from uuid import UUID

from fastapi import APIRouter, Depends, Header, Request
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from rag_service.api.errors import error_responses
from rag_service.api.middleware import get_request_id
from rag_service.auth.dependencies import require_agent_principal
from rag_service.auth.policies import AgentPrincipal
from rag_service.config import Settings, get_settings
from rag_service.db.dependencies import get_session
from rag_service.db.session import Database
from rag_service.jobs.schemas import SafeJob
from rag_service.jobs.services import JobNotifier, JobService

router = APIRouter(prefix="/v1/jobs", tags=["jobs"])


async def get_job_service(
    request: Request,
    session: Annotated[AsyncSession, Depends(get_session)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> JobService:
    database = cast(Database, request.app.state.database)
    notifier = cast(JobNotifier | None, getattr(request.app.state, "job_notifier", None))
    return JobService(
        session=session,
        reservation_session_factory=database.sessions,
        max_idempotency_key_length=settings.max_idempotency_key_length,
        notifier=notifier,
        notification_timeout_seconds=settings.ingestion_notify_timeout_seconds,
    )


def _safe_response(job: SafeJob, *, status_code: int = 200) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        headers={"Cache-Control": "no-store"},
        content=job.model_dump(mode="json"),
    )


@router.get(
    "/{job_id}",
    response_model=SafeJob,
    responses=error_responses(401, 404, 422, 500),
)
async def get_job(
    job_id: UUID,
    _request_id: Annotated[str, Depends(get_request_id)],
    actor: Annotated[AgentPrincipal, Depends(require_agent_principal)],
    service: Annotated[JobService, Depends(get_job_service)],
) -> JSONResponse:
    return _safe_response(await service.get_job(job_id, actor=actor))


@router.post(
    "/{job_id}/retry",
    response_model=SafeJob,
    status_code=202,
    responses=error_responses(401, 404, 409, 422, 500),
)
async def retry_job(
    job_id: UUID,
    request_id: Annotated[str, Depends(get_request_id)],
    actor: Annotated[AgentPrincipal, Depends(require_agent_principal)],
    service: Annotated[JobService, Depends(get_job_service)],
    idempotency_key: Annotated[str, Header(alias="Idempotency-Key")],
) -> JSONResponse:
    job = await service.retry_job(
        job_id,
        actor=actor,
        idempotency_key=idempotency_key,
        request_id=request_id,
    )
    return _safe_response(job, status_code=202)


__all__ = ["get_job_service", "router"]
