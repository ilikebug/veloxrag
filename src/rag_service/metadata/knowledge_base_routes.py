from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Header, Query, Response
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from rag_service.api.cursors import decode_cursor
from rag_service.api.errors import BusinessError, error_responses
from rag_service.api.middleware import get_request_id
from rag_service.auth.dependencies import require_agent_principal
from rag_service.auth.policies import AgentPrincipal
from rag_service.config import Settings, get_settings
from rag_service.db.dependencies import get_session
from rag_service.metadata.schemas import (
    FilterSchemaReplacement,
    KnowledgeBaseCreate,
    KnowledgeBasePage,
    KnowledgeBasePatch,
    SafeFilterSchema,
    SafeKnowledgeBase,
)
from rag_service.metadata.services import KnowledgeBaseService

_NO_STORE = "no-store"
_HARD_MAX_PAGE_SIZE = 100


def _validation_error() -> BusinessError:
    return BusinessError(422, "VALIDATION_ERROR", "Invalid knowledge base request")


router = APIRouter(
    prefix="/v1/knowledge-bases",
    tags=["knowledge bases"],
)


async def get_knowledge_base_service(
    session: Annotated[AsyncSession, Depends(get_session)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> KnowledgeBaseService:
    return KnowledgeBaseService(session=session, settings=settings)


def _safe_response(
    document: SafeKnowledgeBase | KnowledgeBasePage | SafeFilterSchema,
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


def _page_limit(limit: int | None, settings: Settings) -> int:
    maximum = min(settings.max_page_size, _HARD_MAX_PAGE_SIZE)
    selected = min(settings.default_page_size, maximum) if limit is None else limit
    if type(selected) is not int or not 1 <= selected <= maximum:
        raise _validation_error()
    return selected


@router.post(
    "",
    status_code=201,
    response_model=SafeKnowledgeBase,
    responses={
        200: {
            "model": SafeKnowledgeBase,
            "description": "An equivalent idempotent create was replayed.",
        },
        **error_responses(401, 403, 404, 409, 422, 500),
    },
)
async def create_knowledge_base(
    command: KnowledgeBaseCreate,
    request_id: Annotated[str, Depends(get_request_id)],
    actor: Annotated[AgentPrincipal, Depends(require_agent_principal)],
    service: Annotated[KnowledgeBaseService, Depends(get_knowledge_base_service)],
    idempotency_key: Annotated[str, Header(alias="Idempotency-Key")],
) -> JSONResponse:
    result = await service.create_knowledge_base(
        command,
        actor=actor,
        request_id=request_id,
        idempotency_key=idempotency_key,
    )
    knowledge_base = result.knowledge_base
    return _safe_response(
        knowledge_base,
        status_code=201 if result.created else 200,
        etag=knowledge_base.etag,
        location=f"/v1/knowledge-bases/{knowledge_base.id}",
    )


@router.get(
    "",
    response_model=KnowledgeBasePage,
    responses=error_responses(401, 403, 422, 500),
)
async def list_knowledge_bases(
    _request_id: Annotated[str, Depends(get_request_id)],
    actor: Annotated[AgentPrincipal, Depends(require_agent_principal)],
    service: Annotated[KnowledgeBaseService, Depends(get_knowledge_base_service)],
    settings: Annotated[Settings, Depends(get_settings)],
    cursor: Annotated[str | None, Query()] = None,
    limit: Annotated[int | None, Query()] = None,
) -> JSONResponse:
    if cursor is not None:
        decode_cursor(cursor)
    page = await service.list_knowledge_bases(
        actor=actor,
        cursor=cursor,
        limit=_page_limit(limit, settings),
    )
    return _safe_response(page)


@router.get(
    "/{knowledge_base_id}",
    response_model=SafeKnowledgeBase,
    responses=error_responses(401, 403, 404, 422, 500),
)
async def get_knowledge_base(
    knowledge_base_id: UUID,
    _request_id: Annotated[str, Depends(get_request_id)],
    actor: Annotated[AgentPrincipal, Depends(require_agent_principal)],
    service: Annotated[KnowledgeBaseService, Depends(get_knowledge_base_service)],
) -> JSONResponse:
    knowledge_base = await service.get_knowledge_base(knowledge_base_id, actor=actor)
    return _safe_response(knowledge_base, etag=knowledge_base.etag)


@router.patch(
    "/{knowledge_base_id}",
    response_model=SafeKnowledgeBase,
    responses=error_responses(401, 403, 404, 409, 412, 422, 500),
)
async def update_knowledge_base(
    knowledge_base_id: UUID,
    command: KnowledgeBasePatch,
    request_id: Annotated[str, Depends(get_request_id)],
    actor: Annotated[AgentPrincipal, Depends(require_agent_principal)],
    service: Annotated[KnowledgeBaseService, Depends(get_knowledge_base_service)],
    if_match: Annotated[str | None, Header(alias="If-Match")] = None,
) -> JSONResponse:
    knowledge_base = await service.update_knowledge_base(
        knowledge_base_id,
        command,
        actor=actor,
        request_id=request_id,
        expected_etag=if_match,
    )
    return _safe_response(knowledge_base, etag=knowledge_base.etag)


@router.put(
    "/{knowledge_base_id}/filter-schema",
    response_model=SafeFilterSchema,
    responses=error_responses(401, 403, 404, 409, 412, 422, 500),
)
async def replace_filter_schema(
    knowledge_base_id: UUID,
    command: FilterSchemaReplacement,
    request_id: Annotated[str, Depends(get_request_id)],
    actor: Annotated[AgentPrincipal, Depends(require_agent_principal)],
    service: Annotated[KnowledgeBaseService, Depends(get_knowledge_base_service)],
    if_match: Annotated[str | None, Header(alias="If-Match")] = None,
) -> JSONResponse:
    filter_schema = await service.replace_filter_schema(
        knowledge_base_id,
        command,
        actor=actor,
        request_id=request_id,
        expected_etag=if_match,
    )
    return _safe_response(filter_schema, etag=filter_schema.etag)


@router.delete(
    "/{knowledge_base_id}",
    status_code=204,
    response_class=Response,
    responses=error_responses(401, 403, 404, 412, 422, 500),
)
async def delete_knowledge_base(
    knowledge_base_id: UUID,
    request_id: Annotated[str, Depends(get_request_id)],
    actor: Annotated[AgentPrincipal, Depends(require_agent_principal)],
    service: Annotated[KnowledgeBaseService, Depends(get_knowledge_base_service)],
    if_match: Annotated[str | None, Header(alias="If-Match")] = None,
) -> Response:
    knowledge_base = await service.delete_knowledge_base(
        knowledge_base_id,
        actor=actor,
        request_id=request_id,
        expected_etag=if_match,
    )
    return Response(
        status_code=204,
        headers={"Cache-Control": _NO_STORE, "ETag": knowledge_base.etag},
    )
