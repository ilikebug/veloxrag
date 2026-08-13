from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Header, Query, Request
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from rag_service.api.errors import BusinessError, error_responses
from rag_service.api.middleware import get_request_id
from rag_service.auth.dependencies import get_database, require_admin_principal
from rag_service.auth.policies import AdminPrincipal
from rag_service.auth.schemas import AgentApiKeyCreate, AgentApiKeyUpdate, Page, SafeApiKey
from rag_service.auth.services import ApiKeyService
from rag_service.config import Settings, get_settings
from rag_service.db.dependencies import get_session
from rag_service.db.session import Database

router = APIRouter(prefix="/v1/admin", tags=["administrator"])
_NO_STORE = "no-store"


def _validation_error() -> BusinessError:
    return BusinessError(422, "VALIDATION_ERROR", "Invalid API key policy")


async def get_api_key_service(
    session: Annotated[AsyncSession, Depends(get_session)],
    database: Annotated[Database, Depends(get_database)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> ApiKeyService:
    return ApiKeyService(
        session=session,
        authentication_sessions=database.session,
        settings=settings,
    )


def _safe_response(
    document: SafeApiKey | Page[SafeApiKey],
    *,
    etag: str | None = None,
    location: str | None = None,
    status_code: int = 200,
) -> JSONResponse:
    headers = {"Cache-Control": _NO_STORE}
    if etag is not None:
        headers["ETag"] = etag
    if location is not None:
        headers["Location"] = location
    return JSONResponse(
        status_code=status_code,
        headers=headers,
        content=document.model_dump(mode="json"),
    )


@router.post(
    "/api-keys",
    status_code=201,
    responses=error_responses(401, 403, 409, 422, 500),
)
async def create_agent_key(
    command: AgentApiKeyCreate,
    request: Request,
    request_id: Annotated[str, Depends(get_request_id)],
    actor: Annotated[AdminPrincipal, Depends(require_admin_principal)],
    service: Annotated[ApiKeyService, Depends(get_api_key_service)],
) -> JSONResponse:
    if "idempotency-key" in request.headers:
        raise _validation_error()
    issued = await service.create_agent_key(command, actor=actor, request_id=request_id)
    safe = issued.api_key
    if safe.etag is None:
        raise BusinessError(500, "INTERNAL_ERROR", "Internal server error")
    return JSONResponse(
        status_code=201,
        headers={
            "Cache-Control": _NO_STORE,
            "ETag": safe.etag,
            "Location": f"/v1/admin/api-keys/{safe.id}",
        },
        content={
            "api_key": safe.model_dump(mode="json"),
            "token": issued.token.get_secret_value(),
        },
    )


@router.get("/api-keys", responses=error_responses(401, 422, 500))
async def list_agent_keys(
    _actor: Annotated[AdminPrincipal, Depends(require_admin_principal)],
    service: Annotated[ApiKeyService, Depends(get_api_key_service)],
    cursor: Annotated[str | None, Query()] = None,
    limit: Annotated[int | None, Query()] = None,
) -> JSONResponse:
    page = await service.list_agent_keys(cursor=cursor, limit=limit)
    return _safe_response(page)


@router.get(
    "/api-keys/{key_id}",
    responses=error_responses(401, 404, 422, 500),
)
async def get_agent_key(
    key_id: UUID,
    _actor: Annotated[AdminPrincipal, Depends(require_admin_principal)],
    service: Annotated[ApiKeyService, Depends(get_api_key_service)],
) -> JSONResponse:
    safe = await service.get_agent_key(key_id)
    if safe.etag is None:
        raise BusinessError(500, "INTERNAL_ERROR", "Internal server error")
    return _safe_response(safe, etag=safe.etag)


@router.patch(
    "/api-keys/{key_id}",
    responses=error_responses(401, 404, 409, 412, 422, 500),
)
async def update_agent_key(
    key_id: UUID,
    command: AgentApiKeyUpdate,
    request_id: Annotated[str, Depends(get_request_id)],
    actor: Annotated[AdminPrincipal, Depends(require_admin_principal)],
    service: Annotated[ApiKeyService, Depends(get_api_key_service)],
    if_match: Annotated[str | None, Header(alias="If-Match")] = None,
) -> JSONResponse:
    safe = await service.update_agent_key(
        key_id,
        command,
        actor=actor,
        request_id=request_id,
        expected_etag=if_match if if_match is not None else "",
    )
    if safe.etag is None:
        raise BusinessError(500, "INTERNAL_ERROR", "Internal server error")
    return _safe_response(safe, etag=safe.etag)


@router.post(
    "/api-keys/{key_id}/revoke",
    responses=error_responses(401, 404, 409, 412, 422, 500),
)
async def revoke_agent_key(
    key_id: UUID,
    request_id: Annotated[str, Depends(get_request_id)],
    actor: Annotated[AdminPrincipal, Depends(require_admin_principal)],
    service: Annotated[ApiKeyService, Depends(get_api_key_service)],
    if_match: Annotated[str | None, Header(alias="If-Match")] = None,
) -> JSONResponse:
    safe = await service.revoke_agent_key(
        key_id,
        actor=actor,
        request_id=request_id,
        expected_etag=if_match,
    )
    if safe.etag is None:
        raise BusinessError(500, "INTERNAL_ERROR", "Internal server error")
    return _safe_response(safe, etag=safe.etag)
