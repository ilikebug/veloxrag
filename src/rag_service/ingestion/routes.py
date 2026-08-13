"""Multipart document ingestion API."""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Header, Request
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from rag_service.api.errors import BusinessError, error_responses
from rag_service.api.middleware import get_request_id
from rag_service.auth.dependencies import require_agent_principal
from rag_service.auth.policies import AgentPrincipal
from rag_service.config import Settings, get_settings
from rag_service.db.dependencies import get_session
from rag_service.ingestion.repositories import SqlAlchemyUploadRepository
from rag_service.ingestion.schemas import UploadAccepted
from rag_service.ingestion.services import DocumentUploadService

router = APIRouter(prefix="/v1", tags=["documents"])

_MEBIBYTE = 1024 * 1024


def upload_limit_description(max_upload_bytes: int) -> str:
    if max_upload_bytes % _MEBIBYTE == 0:
        size = f"{max_upload_bytes // _MEBIBYTE} MiB ({max_upload_bytes} bytes)"
    else:
        size = f"{max_upload_bytes} bytes"
    return f"UTF-8 TXT or Markdown document, up to {size}."


def _upload_openapi_extra(max_upload_bytes: int) -> dict[str, object]:
    return {
        "requestBody": {
            "required": True,
            "content": {
                "multipart/form-data": {
                    "schema": {
                        "type": "object",
                        "required": ["file"],
                        "additionalProperties": False,
                        "properties": {
                            "file": {
                                "type": "string",
                                "format": "binary",
                                "description": upload_limit_description(max_upload_bytes),
                            },
                            "display_name": {
                                "type": "string",
                                "description": "Optional display name for the document.",
                            },
                            "metadata": {
                                "type": "string",
                                "description": "Optional JSON object encoded as a string.",
                            },
                            "tags": {
                                "type": "string",
                                "description": (
                                    "Optional JSON array of tag strings encoded as a string."
                                ),
                            },
                        },
                    }
                }
            },
        }
    }


_DEFAULT_MAX_UPLOAD_BYTES = Settings.model_fields["max_upload_bytes"].default
assert type(_DEFAULT_MAX_UPLOAD_BYTES) is int
_UPLOAD_OPENAPI_EXTRA = _upload_openapi_extra(_DEFAULT_MAX_UPLOAD_BYTES)


async def get_document_upload_service(
    request: Request,
    session: Annotated[AsyncSession, Depends(get_session)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> DocumentUploadService:
    try:
        object_store = request.app.state.upload_object_store
        notifier = request.app.state.job_notifier
        database = request.app.state.database
    except AttributeError:
        raise BusinessError(503, "INGESTION_UNAVAILABLE", "Ingestion is unavailable") from None
    return DocumentUploadService(
        repository=SqlAlchemyUploadRepository(session, object_store),
        object_store=object_store,
        notifier=notifier,
        max_upload_bytes=settings.max_upload_bytes,
        max_idempotency_key_length=settings.max_idempotency_key_length,
        notification_timeout_seconds=settings.ingestion_notify_timeout_seconds,
        session=session,
        reservation_session_factory=database.sessions,
        reservation_repository_factory=lambda reservation_session: SqlAlchemyUploadRepository(
            reservation_session,
            object_store,
        ),
    )


@router.post(
    "/knowledge-bases/{knowledge_base_id}/documents",
    response_model=UploadAccepted,
    status_code=202,
    responses=error_responses(401, 403, 404, 409, 413, 422, 500, 503),
    openapi_extra=_UPLOAD_OPENAPI_EXTRA,
)
async def upload_document(
    knowledge_base_id: UUID,
    request: Request,
    request_id: Annotated[str, Depends(get_request_id)],
    actor: Annotated[AgentPrincipal, Depends(require_agent_principal)],
    service: Annotated[DocumentUploadService, Depends(get_document_upload_service)],
    idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
) -> JSONResponse:
    result = await service.upload_multipart(
        knowledge_base_id=knowledge_base_id,
        actor=actor,
        body=request.stream(),
        content_type=request.headers.get("content-type"),
        idempotency_key=idempotency_key,
        request_id=request_id,
    )
    return JSONResponse(
        status_code=202,
        headers={"Cache-Control": "no-store"},
        content=result.model_dump(mode="json"),
    )


__all__ = ["get_document_upload_service", "router", "upload_limit_description"]
