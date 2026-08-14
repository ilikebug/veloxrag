from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from rag_service.api.cursors import decode_cursor
from rag_service.api.errors import error_responses
from rag_service.api.middleware import get_request_id
from rag_service.auth.dependencies import require_agent_principal
from rag_service.auth.policies import AgentPrincipal
from rag_service.config import Settings, get_settings
from rag_service.db.dependencies import get_session
from rag_service.metadata.knowledge_base_routes import _page_limit
from rag_service.metadata.schemas import (
    DocumentContent,
    DocumentPage,
    DocumentVersionPage,
    SafeDocument,
)
from rag_service.metadata.services import DocumentMetadataService

router = APIRouter(
    prefix="/v1",
    tags=["documents"],
)


async def get_document_metadata_service(
    session: Annotated[AsyncSession, Depends(get_session)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> DocumentMetadataService:
    return DocumentMetadataService(session=session, settings=settings)


def _safe_response(document: BaseModel) -> JSONResponse:
    return JSONResponse(
        headers={"Cache-Control": "no-store"},
        content=document.model_dump(mode="json"),
    )


def _validated_cursor(cursor: str | None) -> str | None:
    if cursor is not None:
        decode_cursor(cursor)
    return cursor


@router.get(
    "/knowledge-bases/{knowledge_base_id}/documents",
    response_model=DocumentPage,
    responses=error_responses(401, 403, 404, 422, 500),
)
async def list_documents(
    knowledge_base_id: UUID,
    _request_id: Annotated[str, Depends(get_request_id)],
    actor: Annotated[AgentPrincipal, Depends(require_agent_principal)],
    service: Annotated[DocumentMetadataService, Depends(get_document_metadata_service)],
    settings: Annotated[Settings, Depends(get_settings)],
    cursor: Annotated[str | None, Query()] = None,
    limit: Annotated[int | None, Query()] = None,
) -> JSONResponse:
    page = await service.list_documents(
        knowledge_base_id,
        actor=actor,
        cursor=_validated_cursor(cursor),
        limit=_page_limit(limit, settings),
    )
    return _safe_response(page)


@router.get(
    "/documents/{document_id}",
    response_model=SafeDocument,
    responses=error_responses(401, 403, 404, 422, 500),
)
async def get_document(
    document_id: UUID,
    _request_id: Annotated[str, Depends(get_request_id)],
    actor: Annotated[AgentPrincipal, Depends(require_agent_principal)],
    service: Annotated[DocumentMetadataService, Depends(get_document_metadata_service)],
) -> JSONResponse:
    document = await service.get_document(document_id, actor=actor)
    return _safe_response(document)


@router.get(
    "/documents/{document_id}/versions",
    response_model=DocumentVersionPage,
    responses=error_responses(401, 403, 404, 422, 500),
)
async def list_document_versions(
    document_id: UUID,
    _request_id: Annotated[str, Depends(get_request_id)],
    actor: Annotated[AgentPrincipal, Depends(require_agent_principal)],
    service: Annotated[DocumentMetadataService, Depends(get_document_metadata_service)],
    settings: Annotated[Settings, Depends(get_settings)],
    cursor: Annotated[str | None, Query()] = None,
    limit: Annotated[int | None, Query()] = None,
) -> JSONResponse:
    page = await service.list_versions(
        document_id,
        actor=actor,
        cursor=_validated_cursor(cursor),
        limit=_page_limit(limit, settings),
    )
    return _safe_response(page)


@router.get(
    "/documents/{document_id}/content",
    response_model=DocumentContent,
    responses=error_responses(401, 403, 404, 409, 422, 500),
)
async def read_document_content(
    request: Request,
    document_id: UUID,
    _request_id: Annotated[str, Depends(get_request_id)],
    actor: Annotated[AgentPrincipal, Depends(require_agent_principal)],
    session: Annotated[AsyncSession, Depends(get_session)],
    settings: Annotated[Settings, Depends(get_settings)],
    start: Annotated[int, Query(ge=0)] = 0,
    end: Annotated[int | None, Query(ge=0)] = None,
) -> JSONResponse:
    """Read a range of a document's normalized text.

    The offsets are the ones a search result already carries, so a consumer that
    got a truncated hit can widen it without a second addressing scheme. `end`
    defaults to the upload limit rather than to the document length, which is not
    known until the object is read.
    """

    service = DocumentMetadataService(
        session=session,
        settings=settings,
        object_store=request.app.state.upload_object_store,
    )
    content = await service.read_document_content(
        document_id,
        actor=actor,
        start_offset=start,
        end_offset=settings.max_upload_bytes if end is None else end,
    )
    return _safe_response(content)
