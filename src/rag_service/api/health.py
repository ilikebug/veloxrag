from typing import Literal, cast

from fastapi import APIRouter, Request, Response, status
from pydantic import BaseModel

from rag_service.readiness import ReadinessProvider, ReadinessScope, ReadinessSnapshot

router = APIRouter(tags=["operations"])


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"


class ComponentResponse(BaseModel):
    ok: bool
    latency_ms: float
    error: str | None


class CapabilityResponse(BaseModel):
    retrieval: bool
    ingest: bool
    answer: bool


CapabilityStatus = Literal["available", "unavailable", "not_checked"]


class CapabilityStatusResponse(BaseModel):
    retrieval: CapabilityStatus
    ingest: CapabilityStatus


class ReadinessResponse(BaseModel):
    ready: bool
    reason: str | None
    components: dict[str, ComponentResponse]
    capabilities: CapabilityResponse
    capability_status: CapabilityStatusResponse


def get_readiness_provider(request: Request) -> ReadinessProvider:
    return cast(ReadinessProvider, request.app.state.readiness_provider)


def _capability_status(capability: bool | None) -> CapabilityStatus:
    if capability is None:
        return "not_checked"
    return "available" if capability else "unavailable"


def response_body(
    snapshot: ReadinessSnapshot,
    *,
    ready: bool,
    reason: str | None = None,
) -> ReadinessResponse:
    return ReadinessResponse(
        ready=ready,
        reason=reason,
        components={
            name: ComponentResponse(
                ok=value.ok,
                latency_ms=value.latency_ms,
                error=value.error,
            )
            for name, value in snapshot.components.items()
        },
        capabilities=CapabilityResponse(
            retrieval=snapshot.core_ready,
            ingest=(
                snapshot.core_ready
                and snapshot.components["redis"].ok
                and snapshot.components["minio"].ok
            ),
            answer=snapshot.answer_ready,
        ),
        capability_status=CapabilityStatusResponse(
            retrieval=_capability_status(snapshot.retrieval_capability),
            ingest=_capability_status(snapshot.ingest_capability),
        ),
    )


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse()


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    responses={
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "model": ReadinessResponse,
            "description": "Core dependencies are unavailable.",
        }
    },
)
async def ready(request: Request, response: Response) -> ReadinessResponse:
    snapshot = await get_readiness_provider(request).snapshot(ReadinessScope.CORE)
    response.status_code = (
        status.HTTP_200_OK if snapshot.core_ready else status.HTTP_503_SERVICE_UNAVAILABLE
    )
    return response_body(snapshot, ready=snapshot.core_ready)


@router.get(
    "/ready/ingest",
    response_model=ReadinessResponse,
    responses={
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "model": ReadinessResponse,
            "description": "Ingest dependencies are unavailable.",
        }
    },
)
async def ready_ingest(request: Request, response: Response) -> ReadinessResponse:
    snapshot = await get_readiness_provider(request).snapshot(ReadinessScope.INGEST)
    response.status_code = (
        status.HTTP_200_OK if snapshot.ingest_ready else status.HTTP_503_SERVICE_UNAVAILABLE
    )
    reason = None if snapshot.ingest_ready else "ingest_dependencies_unavailable"
    return response_body(snapshot, ready=snapshot.ingest_ready, reason=reason)


@router.get(
    "/ready/retrieve",
    response_model=ReadinessResponse,
    responses={
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "model": ReadinessResponse,
            "description": "Retrieve dependencies are unavailable.",
        }
    },
)
async def ready_retrieve(request: Request, response: Response) -> ReadinessResponse:
    snapshot = await get_readiness_provider(request).snapshot(ReadinessScope.RETRIEVE)
    response.status_code = (
        status.HTTP_200_OK if snapshot.retrieve_ready else status.HTTP_503_SERVICE_UNAVAILABLE
    )
    reason = None if snapshot.retrieve_ready else "retrieve_dependencies_unavailable"
    return response_body(snapshot, ready=snapshot.retrieve_ready, reason=reason)


@router.get(
    "/ready/answer",
    response_model=ReadinessResponse,
    responses={
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "model": ReadinessResponse,
            "description": "Answer capability is unavailable.",
        }
    },
)
async def ready_answer(request: Request, response: Response) -> ReadinessResponse:
    snapshot = await get_readiness_provider(request).snapshot(ReadinessScope.CORE)
    response.status_code = (
        status.HTTP_200_OK if snapshot.answer_ready else status.HTTP_503_SERVICE_UNAVAILABLE
    )
    if snapshot.answer_ready:
        reason = None
    elif not snapshot.core_ready:
        reason = "core_dependencies_unavailable"
    else:
        reason = "query_profile_not_configured"
    return response_body(snapshot, ready=snapshot.answer_ready, reason=reason)
