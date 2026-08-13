"""Agent API for authorized single-knowledge-base Dense search."""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from rag_service.api.errors import error_responses
from rag_service.api.middleware import get_request_id
from rag_service.auth.dependencies import get_database, require_agent_principal
from rag_service.auth.policies import AgentPrincipal
from rag_service.db.dependencies import get_session
from rag_service.db.session import Database
from rag_service.indexing.generation_routes import get_generation_qdrant
from rag_service.indexing.qdrant import QdrantClient
from rag_service.observability.repositories import (
    SqlAlchemyProviderUsageSink,
    SqlAlchemyQueryLogSink,
)
from rag_service.providers.embeddings import EmbeddingGateway
from rag_service.providers.gateway_provider import get_embedding_gateway, get_rerank_gateway
from rag_service.providers.rerank import RerankGateway
from rag_service.retrieval.repositories import SqlAlchemyRetrievalRepository
from rag_service.retrieval.schemas import SearchRequest, SearchResponse
from rag_service.retrieval.services import SearchService

router = APIRouter(
    prefix="/v1/knowledge-bases/{knowledge_base_id}",
    tags=["retrieval"],
)


async def get_retrieval_service(
    session: Annotated[AsyncSession, Depends(get_session)],
    database: Annotated[Database, Depends(get_database)],
    qdrant: Annotated[QdrantClient, Depends(get_generation_qdrant)],
    embedding_gateway: Annotated[
        EmbeddingGateway,
        Depends(get_embedding_gateway),
    ],
    rerank_gateway: Annotated[
        RerankGateway | None,
        Depends(get_rerank_gateway),
    ],
) -> SearchService:
    return SearchService(
        repository=SqlAlchemyRetrievalRepository(session),
        embedding_gateway=embedding_gateway,
        rerank_gateway=rerank_gateway,
        search_index=qdrant,
        provider_usage_sink=SqlAlchemyProviderUsageSink(database.sessions),
        query_log_sink=SqlAlchemyQueryLogSink(database.sessions),
    )


@router.post(
    "/search",
    response_model=SearchResponse,
    responses={
        200: {
            "model": SearchResponse,
            "description": "Visible chunks ranked by raw Dense vector score.",
        },
        **error_responses(401, 403, 404, 409, 422, 500, 503),
    },
)
async def search_knowledge_base(
    knowledge_base_id: UUID,
    command: SearchRequest,
    request_id: Annotated[str, Depends(get_request_id)],
    actor: Annotated[AgentPrincipal, Depends(require_agent_principal)],
    service: Annotated[SearchService, Depends(get_retrieval_service)],
) -> JSONResponse:
    response = await service.search(
        knowledge_base_id=knowledge_base_id,
        actor=actor,
        request_id=request_id,
        command=command,
    )
    return JSONResponse(
        headers={"Cache-Control": "no-store"},
        content=response.model_dump(mode="json"),
    )


__all__ = ["get_retrieval_service", "router"]
