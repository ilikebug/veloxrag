"""Administrator routes for initial index generations."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from datetime import UTC, datetime
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Header, Query
from fastapi.responses import JSONResponse

from rag_service.api.errors import error_responses
from rag_service.api.middleware import get_request_id
from rag_service.auth.dependencies import get_database, require_admin_principal
from rag_service.auth.policies import AdminPrincipal
from rag_service.config import Settings, get_settings
from rag_service.db.session import Database
from rag_service.indexing.generation_schemas import (
    IndexGenerationCreate,
    IndexGenerationPage,
    SafeIndexGeneration,
)
from rag_service.indexing.generation_services import (
    GenerationQueryService,
    GenerationSagaHooks,
    GenerationService,
)
from rag_service.indexing.qdrant import QdrantClient, qdrant_client_from_url
from rag_service.providers.embeddings import EmbeddingGateway
from rag_service.providers.gateway_provider import get_embedding_gateway

router = APIRouter(
    prefix="/v1/admin/knowledge-bases/{knowledge_base_id}/index-generations",
    tags=["administrator", "index generations"],
)
_NO_STORE = "no-store"


async def get_generation_qdrant(
    settings: Annotated[Settings, Depends(get_settings)],
) -> AsyncIterator[QdrantClient]:
    client = qdrant_client_from_url(
        settings.qdrant_url,
        timeout_seconds=settings.qdrant_request_timeout_seconds,
    )
    try:
        yield client
    finally:
        await client.aclose()


def get_generation_hooks() -> GenerationSagaHooks:
    return GenerationSagaHooks()


def get_generation_clock() -> Callable[[], datetime]:
    return lambda: datetime.now(UTC)


async def get_generation_service(
    database: Annotated[Database, Depends(get_database)],
    settings: Annotated[Settings, Depends(get_settings)],
    qdrant: Annotated[QdrantClient, Depends(get_generation_qdrant)],
    embedding_gateway: Annotated[
        EmbeddingGateway,
        Depends(get_embedding_gateway),
    ],
    hooks: Annotated[GenerationSagaHooks, Depends(get_generation_hooks)],
    clock: Annotated[Callable[[], datetime], Depends(get_generation_clock)],
) -> GenerationService:
    return GenerationService(
        session_factory=database.sessions,
        qdrant=qdrant,
        embedding_gateway=embedding_gateway,
        hooks=hooks,
        clock=clock,
        max_idempotency_key_length=min(settings.max_idempotency_key_length, 128),
    )


async def get_generation_query_service(
    database: Annotated[Database, Depends(get_database)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> GenerationQueryService:
    maximum = min(settings.max_page_size, 100)
    return GenerationQueryService(
        session_factory=database.sessions,
        default_page_size=min(settings.default_page_size, maximum),
        max_page_size=maximum,
    )


def _safe_response(
    document: SafeIndexGeneration | IndexGenerationPage,
    *,
    status_code: int = 200,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        headers={"Cache-Control": _NO_STORE},
        content=document.model_dump(mode="json"),
    )


@router.post(
    "",
    status_code=201,
    response_model=SafeIndexGeneration,
    responses={
        201: {
            "model": SafeIndexGeneration,
            "description": "The initial empty index generation is active.",
        },
        **error_responses(401, 404, 409, 422, 500, 503),
    },
)
async def create_initial_generation(
    knowledge_base_id: UUID,
    command: IndexGenerationCreate,
    _request_id: Annotated[str, Depends(get_request_id)],
    actor: Annotated[AdminPrincipal, Depends(require_admin_principal)],
    service: Annotated[GenerationService, Depends(get_generation_service)],
    idempotency_key: Annotated[str, Header(alias="Idempotency-Key")],
) -> JSONResponse:
    generation = await service.create_initial_generation(
        knowledge_base_id,
        command,
        actor=actor,
        idempotency_key=idempotency_key,
    )
    return _safe_response(generation, status_code=201)


@router.post(
    "/{generation_id}/abandon",
    response_model=SafeIndexGeneration,
    responses={
        200: {
            "model": SafeIndexGeneration,
            "description": "The building index generation is now failed.",
        },
        **error_responses(401, 404, 409, 422, 500, 503),
    },
)
async def abandon_index_generation(
    knowledge_base_id: UUID,
    generation_id: UUID,
    _request_id: Annotated[str, Depends(get_request_id)],
    actor: Annotated[AdminPrincipal, Depends(require_admin_principal)],
    service: Annotated[GenerationService, Depends(get_generation_service)],
) -> JSONResponse:
    generation = await service.abandon_generation(
        knowledge_base_id,
        generation_id,
        actor=actor,
    )
    return _safe_response(generation)


@router.get(
    "",
    response_model=IndexGenerationPage,
    responses={
        200: {
            "model": IndexGenerationPage,
            "description": "Safe index generation state for one knowledge base.",
        },
        **error_responses(401, 404, 422, 500, 503),
    },
)
async def list_index_generations(
    knowledge_base_id: UUID,
    _request_id: Annotated[str, Depends(get_request_id)],
    _actor: Annotated[AdminPrincipal, Depends(require_admin_principal)],
    service: Annotated[GenerationQueryService, Depends(get_generation_query_service)],
    cursor: Annotated[str | None, Query()] = None,
    limit: Annotated[int | None, Query()] = None,
) -> JSONResponse:
    return _safe_response(
        await service.list_generations(
            knowledge_base_id,
            cursor=cursor,
            limit=limit,
        )
    )


__all__ = [
    "get_generation_clock",
    "get_generation_hooks",
    "get_generation_qdrant",
    "get_generation_query_service",
    "get_generation_service",
    "router",
]
