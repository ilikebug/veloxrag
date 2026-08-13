"""PostgreSQL facts required by single-knowledge-base retrieval."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Protocol, cast
from uuid import UUID

from sqlalchemy import and_, select, tuple_
from sqlalchemy.ext.asyncio import AsyncSession

from rag_service.db.models.documents import Document, DocumentIndexState, DocumentVersion
from rag_service.db.models.knowledge_bases import KnowledgeBase, KnowledgeBaseIndexGeneration
from rag_service.db.models.providers import ModelProfile, ProviderConfig, ProviderCredential


@dataclass(frozen=True, slots=True)
class ActiveSearchTarget:
    knowledge_base_id: UUID
    knowledge_base_status: str
    generation_id: UUID
    generation_status: str
    embedding_profile_id: UUID
    qdrant_collection_name: str
    embedding_config_snapshot: dict[str, object]
    embedding_config_hash: str
    index_profile_hash: str
    filter_schema_snapshot: dict[str, object]
    rerank_profile_id: UUID | None


@dataclass(frozen=True, slots=True)
class RerankRuntimeRecord:
    """Live rerank configuration.

    Read fresh rather than from the generation snapshot: the reranker does not
    shape the index, so changing it takes effect without a rebuild — the exact
    opposite of the embedding configuration, which is frozen so the vectors in a
    collection stay reproducible.
    """

    profile_id: UUID
    profile_capability: str
    profile_enabled: bool
    model_name: str
    timeout_seconds: Decimal
    provider_id: UUID
    provider_type: str | None
    base_url: str | None
    default_headers: dict[str, object] | None
    provider_enabled: bool | None
    credential_id: UUID | None
    credential_exists: bool


@dataclass(frozen=True, slots=True)
class EmbeddingRuntimeRecord:
    profile_id: UUID
    profile_capability: str
    profile_enabled: bool
    timeout_seconds: Decimal
    batch_size: int
    provider_id: UUID
    provider_enabled: bool | None
    max_concurrency: int | None
    requests_per_minute: int | None
    credential_id: UUID
    credential_exists: bool


@dataclass(frozen=True, slots=True)
class VisibleDocument:
    document_id: UUID
    version_id: UUID
    source_filename: str


class SearchRepository(Protocol):
    async def get_active_target(
        self,
        knowledge_base_id: UUID,
    ) -> ActiveSearchTarget | None: ...

    async def get_embedding_runtime(
        self,
        profile_id: UUID,
        provider_id: UUID,
        credential_id: UUID,
    ) -> EmbeddingRuntimeRecord | None: ...

    async def get_rerank_runtime(
        self,
        profile_id: UUID,
    ) -> RerankRuntimeRecord | None: ...

    async def load_visible_document_filters(
        self,
        *,
        knowledge_base_id: UUID,
        generation_id: UUID,
        document_ids: Sequence[UUID],
    ) -> Mapping[UUID, VisibleDocument]: ...

    async def load_visible_documents(
        self,
        *,
        knowledge_base_id: UUID,
        generation_id: UUID,
        identities: Sequence[tuple[UUID, UUID]],
    ) -> Mapping[tuple[UUID, UUID], VisibleDocument]: ...


def _mapping(row: object) -> Mapping[str, Any]:
    return cast(Mapping[str, Any], cast(Any, row)._mapping)


class SqlAlchemyRetrievalRepository:
    """Read the active snapshot and canonical document visibility in bounded queries."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_active_target(
        self,
        knowledge_base_id: UUID,
    ) -> ActiveSearchTarget | None:
        generation = KnowledgeBaseIndexGeneration
        statement = (
            select(
                KnowledgeBase.id.label("knowledge_base_id"),
                KnowledgeBase.status.label("knowledge_base_status"),
                generation.id.label("generation_id"),
                generation.status.label("generation_status"),
                generation.embedding_profile_id.label("embedding_profile_id"),
                generation.qdrant_collection_name.label("qdrant_collection_name"),
                generation.embedding_config_snapshot.label("embedding_config_snapshot"),
                generation.embedding_config_hash.label("embedding_config_hash"),
                generation.index_profile_hash.label("index_profile_hash"),
                generation.filter_schema_snapshot.label("filter_schema_snapshot"),
                KnowledgeBase.rerank_profile_id.label("rerank_profile_id"),
            )
            .select_from(KnowledgeBase)
            .join(
                generation,
                and_(
                    generation.id == KnowledgeBase.active_index_generation_id,
                    generation.knowledge_base_id == KnowledgeBase.id,
                ),
            )
            .where(KnowledgeBase.id == knowledge_base_id)
        )
        row = (await self._session.execute(statement)).one_or_none()
        if row is None:
            return None
        return ActiveSearchTarget(**dict(_mapping(row)))

    async def get_rerank_runtime(
        self,
        profile_id: UUID,
    ) -> RerankRuntimeRecord | None:
        statement = (
            select(
                ModelProfile.id.label("profile_id"),
                ModelProfile.capability.label("profile_capability"),
                ModelProfile.enabled.label("profile_enabled"),
                ModelProfile.model_name.label("model_name"),
                ModelProfile.timeout_seconds.label("timeout_seconds"),
                ModelProfile.provider_config_id.label("provider_id"),
                ProviderConfig.provider_type.label("provider_type"),
                ProviderConfig.base_url.label("base_url"),
                ProviderConfig.default_headers.label("default_headers"),
                ProviderConfig.enabled.label("provider_enabled"),
                ProviderConfig.credential_id.label("credential_id"),
                ProviderCredential.id.label("persisted_credential_id"),
            )
            .select_from(ModelProfile)
            .outerjoin(ProviderConfig, ProviderConfig.id == ModelProfile.provider_config_id)
            .outerjoin(ProviderCredential, ProviderCredential.id == ProviderConfig.credential_id)
            .where(ModelProfile.id == profile_id)
        )
        row = (await self._session.execute(statement)).one_or_none()
        if row is None:
            return None
        values = _mapping(row)
        return RerankRuntimeRecord(
            profile_id=values["profile_id"],
            profile_capability=values["profile_capability"],
            profile_enabled=values["profile_enabled"],
            model_name=values["model_name"],
            timeout_seconds=values["timeout_seconds"],
            provider_id=values["provider_id"],
            provider_type=values["provider_type"],
            base_url=values["base_url"],
            default_headers=values["default_headers"],
            provider_enabled=values["provider_enabled"],
            credential_id=values["credential_id"],
            credential_exists=values["persisted_credential_id"] is not None,
        )

    async def get_embedding_runtime(
        self,
        profile_id: UUID,
        provider_id: UUID,
        credential_id: UUID,
    ) -> EmbeddingRuntimeRecord | None:
        statement = (
            select(
                ModelProfile.id.label("profile_id"),
                ModelProfile.capability.label("profile_capability"),
                ModelProfile.enabled.label("profile_enabled"),
                ModelProfile.timeout_seconds.label("timeout_seconds"),
                ModelProfile.batch_size.label("batch_size"),
                ModelProfile.provider_config_id.label("provider_id"),
                ProviderConfig.enabled.label("provider_enabled"),
                ProviderConfig.max_concurrency.label("max_concurrency"),
                ProviderConfig.requests_per_minute.label("requests_per_minute"),
                ProviderCredential.id.label("persisted_credential_id"),
            )
            .select_from(ModelProfile)
            .outerjoin(
                ProviderConfig,
                and_(
                    ProviderConfig.id == provider_id,
                    ProviderConfig.id == ModelProfile.provider_config_id,
                ),
            )
            .outerjoin(
                ProviderCredential,
                and_(
                    ProviderCredential.id == credential_id,
                    ProviderConfig.credential_id == credential_id,
                ),
            )
            .where(ModelProfile.id == profile_id)
        )
        row = (await self._session.execute(statement)).one_or_none()
        if row is None:
            return None
        values = _mapping(row)
        return EmbeddingRuntimeRecord(
            profile_id=values["profile_id"],
            profile_capability=values["profile_capability"],
            profile_enabled=values["profile_enabled"],
            timeout_seconds=values["timeout_seconds"],
            batch_size=values["batch_size"],
            provider_id=values["provider_id"],
            provider_enabled=values["provider_enabled"],
            max_concurrency=values["max_concurrency"],
            requests_per_minute=values["requests_per_minute"],
            credential_id=credential_id,
            credential_exists=values["persisted_credential_id"] == credential_id,
        )

    async def load_visible_document_filters(
        self,
        *,
        knowledge_base_id: UUID,
        generation_id: UUID,
        document_ids: Sequence[UUID],
    ) -> Mapping[UUID, VisibleDocument]:
        canonical = tuple(dict.fromkeys(document_ids))
        if not canonical:
            return {}
        statement = self._visible_documents_statement(
            knowledge_base_id=knowledge_base_id,
            generation_id=generation_id,
        ).where(Document.id.in_(canonical))
        rows = (await self._session.execute(statement)).all()
        records = (VisibleDocument(**dict(_mapping(row))) for row in rows)
        return {record.document_id: record for record in records}

    async def load_visible_documents(
        self,
        *,
        knowledge_base_id: UUID,
        generation_id: UUID,
        identities: Sequence[tuple[UUID, UUID]],
    ) -> Mapping[tuple[UUID, UUID], VisibleDocument]:
        canonical = tuple(dict.fromkeys(identities))
        if not canonical:
            return {}
        statement = self._visible_documents_statement(
            knowledge_base_id=knowledge_base_id,
            generation_id=generation_id,
        ).where(tuple_(Document.id, DocumentVersion.id).in_(canonical))
        rows = (await self._session.execute(statement)).all()
        records = (VisibleDocument(**dict(_mapping(row))) for row in rows)
        return {(record.document_id, record.version_id): record for record in records}

    @staticmethod
    def _visible_documents_statement(
        *,
        knowledge_base_id: UUID,
        generation_id: UUID,
    ) -> Any:
        generation = KnowledgeBaseIndexGeneration
        return (
            select(
                Document.id.label("document_id"),
                DocumentVersion.id.label("version_id"),
                Document.display_name.label("source_filename"),
            )
            .select_from(Document)
            .join(
                DocumentVersion,
                and_(
                    DocumentVersion.id == Document.current_version_id,
                    DocumentVersion.document_id == Document.id,
                ),
            )
            .join(
                DocumentIndexState,
                and_(
                    DocumentIndexState.document_version_id == DocumentVersion.id,
                    DocumentIndexState.index_generation_id == generation_id,
                ),
            )
            .join(generation, generation.id == DocumentIndexState.index_generation_id)
            .where(
                Document.knowledge_base_id == knowledge_base_id,
                Document.status == "active",
                Document.deleted_at.is_(None),
                DocumentVersion.status == "ready",
                DocumentVersion.activated_at.is_not(None),
                DocumentVersion.chunk_count.is_not(None),
                DocumentVersion.chunk_count >= 1,
                DocumentIndexState.status == "validated",
                DocumentIndexState.validated_at.is_not(None),
                DocumentIndexState.embedding_config_hash == generation.embedding_config_hash,
                DocumentIndexState.expected_point_count == DocumentIndexState.actual_point_count,
                DocumentIndexState.expected_point_count == DocumentVersion.chunk_count,
                DocumentIndexState.actual_point_count == DocumentVersion.chunk_count,
                DocumentIndexState.next_chunk_index == DocumentVersion.chunk_count,
                DocumentIndexState.chunk_manifest_checksum_sha256
                == DocumentVersion.chunk_manifest_checksum_sha256,
            )
        )


__all__ = [
    "ActiveSearchTarget",
    "EmbeddingRuntimeRecord",
    "SearchRepository",
    "SqlAlchemyRetrievalRepository",
    "VisibleDocument",
]

SqlAlchemySearchRepository = SqlAlchemyRetrievalRepository
