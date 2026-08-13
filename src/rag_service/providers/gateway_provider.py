"""Application-scoped provider gateway ownership and dependencies."""

from __future__ import annotations

import asyncio
from typing import Annotated

from fastapi import Depends, Request

from rag_service.auth.dependencies import get_database
from rag_service.config import Settings, get_settings
from rag_service.db.session import Database
from rag_service.indexing.generation_repositories import SessionProviderCredentialReader
from rag_service.providers.embeddings import EmbeddingGateway
from rag_service.providers.rerank import RerankGateway
from rag_service.providers.services import (
    provider_credential_keyring_from_settings,
    provider_endpoint_policy_from_settings,
)
from rag_service.providers.transport import SecureProviderTransport


def _embedding_gateway_from_dependencies(
    database: Database,
    settings: Settings,
) -> EmbeddingGateway:
    transport = SecureProviderTransport(
        policy=provider_endpoint_policy_from_settings(settings),
        ca_bundle=settings.provider_ca_bundle,
    )
    return EmbeddingGateway(
        keyring=provider_credential_keyring_from_settings(settings),
        credential_reader=SessionProviderCredentialReader(database.sessions),
        transport=transport,
    )


class EmbeddingGatewayProvider:
    """Lazily owns one process-wide embedding gateway and admission registry."""

    def __init__(self, injected: EmbeddingGateway | None = None) -> None:
        self._gateway = injected
        self._owns_gateway = False
        self._closed = False
        self._lock = asyncio.Lock()

    async def get(self, database: Database, settings: Settings) -> EmbeddingGateway:
        async with self._lock:
            if self._closed:
                raise RuntimeError("Embedding gateway provider is closed")
            if self._gateway is None:
                self._gateway = _embedding_gateway_from_dependencies(database, settings)
                self._owns_gateway = True
            return self._gateway

    async def aclose(self) -> None:
        gateway: EmbeddingGateway | None = None
        async with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._owns_gateway:
                gateway = self._gateway
            self._gateway = None
        if gateway is not None:
            await gateway.aclose()


class RerankGatewayProvider:
    """Lazily owns one process-wide rerank gateway and its transport.

    Its own transport rather than the embedding gateway's: the two are
    constructed independently, and sharing a connection pool would mean a
    reranker outage could exhaust the pool that query embedding depends on.
    """

    def __init__(self, injected: RerankGateway | None = None) -> None:
        self._gateway = injected
        self._transport: SecureProviderTransport | None = None
        self._closed = False
        self._lock = asyncio.Lock()

    async def get(self, database: Database, settings: Settings) -> RerankGateway:
        async with self._lock:
            if self._closed:
                raise RuntimeError("Rerank gateway provider is closed")
            if self._gateway is None:
                self._transport = SecureProviderTransport(
                    policy=provider_endpoint_policy_from_settings(settings),
                    ca_bundle=settings.provider_ca_bundle,
                )
                self._gateway = RerankGateway(
                    keyring=provider_credential_keyring_from_settings(settings),
                    credential_reader=SessionProviderCredentialReader(database.sessions),
                    transport=self._transport,
                )
            return self._gateway

    async def aclose(self) -> None:
        transport: SecureProviderTransport | None = None
        async with self._lock:
            if self._closed:
                return
            self._closed = True
            transport = self._transport
            self._transport = None
            self._gateway = None
        if transport is not None:
            await transport.aclose()


async def get_rerank_gateway(
    request: Request,
    database: Annotated[Database, Depends(get_database)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> RerankGateway | None:
    provider = getattr(request.app.state, "rerank_gateway_provider", None)
    if not isinstance(provider, RerankGatewayProvider):
        # Absent rather than fatal: a deployment that never reranks should not
        # be unable to search.
        return None
    return await provider.get(database, settings)


async def get_embedding_gateway(
    request: Request,
    database: Annotated[Database, Depends(get_database)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> EmbeddingGateway:
    provider = getattr(request.app.state, "embedding_gateway_provider", None)
    if not isinstance(provider, EmbeddingGatewayProvider):
        raise RuntimeError("Embedding gateway provider is unavailable")
    return await provider.get(database, settings)


__all__ = [
    "EmbeddingGatewayProvider",
    "RerankGatewayProvider",
    "get_embedding_gateway",
    "get_rerank_gateway",
]
