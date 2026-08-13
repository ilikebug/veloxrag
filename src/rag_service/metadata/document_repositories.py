from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol, cast
from uuid import UUID

from sqlalchemy import and_, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from rag_service.api.cursors import CursorPosition
from rag_service.db.models.auth import ApiKeyKnowledgeBaseScope
from rag_service.db.models.documents import Document, DocumentVersion
from rag_service.db.models.knowledge_bases import KnowledgeBase


@dataclass(frozen=True, slots=True)
class DocumentRecord:
    id: UUID
    knowledge_base_id: UUID
    display_name: str
    mime_type: str | None
    checksum_sha256: str | None
    current_version_id: UUID | None
    pending_version_id: UUID | None
    status: str
    tags: list[str]
    metadata: dict[str, Any]
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class DocumentVersionRecord:
    id: UUID
    document_id: UUID
    version_number: int
    source_checksum_sha256: str
    parsed_object_checksum_sha256: str | None
    declared_mime_type: str | None
    detected_mime_type: str | None
    source_extension: str | None
    base_version_id: UUID | None
    parser_name: str | None
    parser_version: str | None
    parser_config: dict[str, Any]
    chunker_name: str | None
    chunker_version: str | None
    chunker_config: dict[str, Any]
    chunk_count: int | None
    status: str
    activated_at: datetime | None
    created_at: datetime


class DocumentMetadataRepository(Protocol):
    async def get_scoped_parent(
        self,
        actor_key_id: UUID,
        knowledge_base_id: UUID,
    ) -> UUID | None: ...

    async def list_documents(
        self,
        actor_key_id: UUID,
        knowledge_base_id: UUID,
        position: CursorPosition | None,
        limit: int,
    ) -> list[DocumentRecord]: ...

    async def get_document(
        self,
        actor_key_id: UUID,
        document_id: UUID,
    ) -> DocumentRecord | None: ...

    async def get_document_parent(
        self,
        actor_key_id: UUID,
        document_id: UUID,
    ) -> UUID | None: ...

    async def list_versions(
        self,
        actor_key_id: UUID,
        document_id: UUID,
        position: CursorPosition | None,
        limit: int,
    ) -> list[DocumentVersionRecord]: ...


_DOCUMENT_COLUMNS = (
    Document.id.label("id"),
    Document.knowledge_base_id.label("knowledge_base_id"),
    Document.display_name.label("display_name"),
    Document.mime_type.label("mime_type"),
    Document.checksum_sha256.label("checksum_sha256"),
    Document.current_version_id.label("current_version_id"),
    Document.pending_version_id.label("pending_version_id"),
    Document.status.label("status"),
    Document.tags.label("tags"),
    Document.metadata_.label("metadata"),
    Document.created_at.label("created_at"),
    Document.updated_at.label("updated_at"),
)

_VERSION_COLUMNS = (
    DocumentVersion.id.label("id"),
    DocumentVersion.document_id.label("document_id"),
    DocumentVersion.version_number.label("version_number"),
    DocumentVersion.source_checksum_sha256.label("source_checksum_sha256"),
    DocumentVersion.parsed_object_checksum_sha256.label("parsed_object_checksum_sha256"),
    DocumentVersion.declared_mime_type.label("declared_mime_type"),
    DocumentVersion.detected_mime_type.label("detected_mime_type"),
    DocumentVersion.source_extension.label("source_extension"),
    DocumentVersion.base_version_id.label("base_version_id"),
    DocumentVersion.parser_name.label("parser_name"),
    DocumentVersion.parser_version.label("parser_version"),
    DocumentVersion.parser_config.label("parser_config"),
    DocumentVersion.chunker_name.label("chunker_name"),
    DocumentVersion.chunker_version.label("chunker_version"),
    DocumentVersion.chunker_config.label("chunker_config"),
    DocumentVersion.chunk_count.label("chunk_count"),
    DocumentVersion.status.label("status"),
    DocumentVersion.activated_at.label("activated_at"),
    DocumentVersion.created_at.label("created_at"),
)


def _document_record(row: object) -> DocumentRecord:
    mapping = cast(Any, row)._mapping
    return DocumentRecord(**dict(mapping))


def _version_record(row: object) -> DocumentVersionRecord:
    mapping = cast(Any, row)._mapping
    return DocumentVersionRecord(**dict(mapping))


class SqlAlchemyDocumentMetadataRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    @staticmethod
    def _scoped_document_statement(actor_key_id: UUID) -> Any:
        return (
            select(*_DOCUMENT_COLUMNS)
            .select_from(Document)
            .join(KnowledgeBase, KnowledgeBase.id == Document.knowledge_base_id)
            .join(
                ApiKeyKnowledgeBaseScope,
                and_(
                    ApiKeyKnowledgeBaseScope.knowledge_base_id == KnowledgeBase.id,
                    ApiKeyKnowledgeBaseScope.api_key_id == actor_key_id,
                ),
            )
            .where(
                Document.deleted_at.is_(None),
                KnowledgeBase.status != "deleting",
            )
        )

    async def get_scoped_parent(
        self,
        actor_key_id: UUID,
        knowledge_base_id: UUID,
    ) -> UUID | None:
        statement = (
            select(KnowledgeBase.id)
            .select_from(KnowledgeBase)
            .join(
                ApiKeyKnowledgeBaseScope,
                and_(
                    ApiKeyKnowledgeBaseScope.knowledge_base_id == KnowledgeBase.id,
                    ApiKeyKnowledgeBaseScope.api_key_id == actor_key_id,
                ),
            )
            .where(
                KnowledgeBase.id == knowledge_base_id,
                KnowledgeBase.status != "deleting",
            )
        )
        return cast(UUID | None, await self._session.scalar(statement))

    async def list_documents(
        self,
        actor_key_id: UUID,
        knowledge_base_id: UUID,
        position: CursorPosition | None,
        limit: int,
    ) -> list[DocumentRecord]:
        statement = self._scoped_document_statement(actor_key_id).where(
            Document.knowledge_base_id == knowledge_base_id
        )
        if position is not None:
            statement = statement.where(
                or_(
                    Document.created_at > position.created_at,
                    and_(
                        Document.created_at == position.created_at,
                        Document.id > position.id,
                    ),
                )
            )
        statement = statement.order_by(Document.created_at, Document.id).limit(limit)
        return [_document_record(row) for row in (await self._session.execute(statement)).all()]

    async def get_document(
        self,
        actor_key_id: UUID,
        document_id: UUID,
    ) -> DocumentRecord | None:
        statement = self._scoped_document_statement(actor_key_id).where(Document.id == document_id)
        row = (await self._session.execute(statement)).one_or_none()
        return None if row is None else _document_record(row)

    async def get_document_parent(
        self,
        actor_key_id: UUID,
        document_id: UUID,
    ) -> UUID | None:
        statement = (
            select(Document.knowledge_base_id)
            .select_from(Document)
            .join(KnowledgeBase, KnowledgeBase.id == Document.knowledge_base_id)
            .join(
                ApiKeyKnowledgeBaseScope,
                and_(
                    ApiKeyKnowledgeBaseScope.knowledge_base_id == KnowledgeBase.id,
                    ApiKeyKnowledgeBaseScope.api_key_id == actor_key_id,
                ),
            )
            .where(
                Document.id == document_id,
                Document.deleted_at.is_(None),
                KnowledgeBase.status != "deleting",
            )
        )
        return cast(UUID | None, await self._session.scalar(statement))

    async def list_versions(
        self,
        actor_key_id: UUID,
        document_id: UUID,
        position: CursorPosition | None,
        limit: int,
    ) -> list[DocumentVersionRecord]:
        statement = (
            select(*_VERSION_COLUMNS)
            .select_from(DocumentVersion)
            .join(Document, Document.id == DocumentVersion.document_id)
            .join(KnowledgeBase, KnowledgeBase.id == Document.knowledge_base_id)
            .join(
                ApiKeyKnowledgeBaseScope,
                and_(
                    ApiKeyKnowledgeBaseScope.knowledge_base_id == KnowledgeBase.id,
                    ApiKeyKnowledgeBaseScope.api_key_id == actor_key_id,
                ),
            )
            .where(
                DocumentVersion.document_id == document_id,
                Document.id == document_id,
                Document.deleted_at.is_(None),
                KnowledgeBase.status != "deleting",
            )
        )
        if position is not None:
            statement = statement.where(
                or_(
                    DocumentVersion.created_at > position.created_at,
                    and_(
                        DocumentVersion.created_at == position.created_at,
                        DocumentVersion.id > position.id,
                    ),
                )
            )
        statement = statement.order_by(DocumentVersion.created_at, DocumentVersion.id).limit(limit)
        return [_version_record(row) for row in (await self._session.execute(statement)).all()]


def sqlalchemy_document_metadata_repository(
    session: AsyncSession,
) -> DocumentMetadataRepository:
    return SqlAlchemyDocumentMetadataRepository(session)
