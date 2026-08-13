"""Destroy the data behind a deleted knowledge base.

Deleting a knowledge base used to set its status to `deleting` and stop there:
nothing consumed that status, so the vectors, the uploaded files and every row
stayed. That is worse than a storage leak — a caller who asked for deletion had
no way to make it true.

The order below is forced by the schema. Every foreign key into a knowledge
base is RESTRICT, so children go first, and two pointers form cycles that have
to be broken before their targets can be removed: a knowledge base names its
active and pending generations, and a document names its current version.

Vectors and objects are destroyed directly rather than left to the orphan
reconcilers. Those reconcilers would eventually collect both, since a missing
generation row makes a collection an orphan and a missing version row makes an
object one — but "eventually" is the configured grace period, which defaults to
a day. Deletion the caller asked for should not wait on it.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol
from uuid import UUID

from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from rag_service.db.models.auth import ApiKeyKnowledgeBaseScope
from rag_service.db.models.documents import (
    Document,
    DocumentUploadIdempotency,
    DocumentVersion,
    Job,
)
from rag_service.db.models.knowledge_bases import (
    IndexGenerationCreationRequest,
    KnowledgeBase,
    KnowledgeBaseIndexGeneration,
    KnowledgeBaseMutation,
)
from rag_service.jobs.runner import JobExecutionContext, JobHandlerOutcome
from rag_service.observability import SafeLogContext, emit_safe_log

logger = logging.getLogger(__name__)

_OBJECT_PAGE_LIMIT = 1000
_EVERYTHING = datetime(9999, 12, 31, tzinfo=UTC)


class PurgeObjectCandidate(Protocol):
    object_key: str


class PurgeObjectPage(Protocol):
    items: tuple[PurgeObjectCandidate, ...]
    next_start_after: str | None


class PurgeObjectStore(Protocol):
    async def list_older_than(
        self,
        *,
        prefix: str,
        older_than: datetime,
        limit: int,
        start_after: str | None = None,
    ) -> PurgeObjectPage: ...

    async def delete_best_effort(self, object_key: str) -> bool: ...


class PurgeSearchIndex(Protocol):
    async def delete_collection(self, collection: str) -> None: ...


@dataclass(frozen=True, slots=True)
class PurgeResult:
    collections_deleted: int
    objects_deleted: int
    purged: bool


def knowledge_base_object_prefix(knowledge_base_id: UUID) -> str:
    return f"knowledge-bases/{knowledge_base_id}/"


class KnowledgeBasePurge:
    """Destroys one knowledge base's vectors, objects and rows."""

    def __init__(
        self,
        *,
        session_factory: object,
        object_store: PurgeObjectStore,
        search_index: PurgeSearchIndex,
    ) -> None:
        self._session_factory = session_factory
        self._object_store = object_store
        self._search_index = search_index

    async def purge(self, knowledge_base_id: UUID) -> PurgeResult:
        if type(knowledge_base_id) is not UUID:
            raise ValueError("knowledge base id is invalid")
        collections = await self._collections(knowledge_base_id)
        if collections is None:
            # Already gone, or never marked for deletion. Re-running a purge has
            # to be harmless: the job may be retried after a partial failure.
            return PurgeResult(collections_deleted=0, objects_deleted=0, purged=False)

        collections_deleted = 0
        for name in collections:
            await self._search_index.delete_collection(name)
            collections_deleted += 1

        objects_deleted = await self._delete_objects(knowledge_base_id)
        await self._delete_rows(knowledge_base_id)

        emit_safe_log(
            logger,
            logging.INFO,
            "knowledge_base.purged",
            context=SafeLogContext(knowledge_base_id=knowledge_base_id),
            operation="purge_knowledge_base",
            collections_deleted=collections_deleted,
            objects_deleted=objects_deleted,
        )
        return PurgeResult(
            collections_deleted=collections_deleted,
            objects_deleted=objects_deleted,
            purged=True,
        )

    async def _collections(self, knowledge_base_id: UUID) -> tuple[str, ...] | None:
        async with self._sessions() as session:
            status = await session.scalar(
                select(KnowledgeBase.status).where(KnowledgeBase.id == knowledge_base_id)
            )
            if status != "deleting":
                return None
            rows = await session.scalars(
                select(KnowledgeBaseIndexGeneration.qdrant_collection_name).where(
                    KnowledgeBaseIndexGeneration.knowledge_base_id == knowledge_base_id
                )
            )
            return tuple(name for name in rows if name)

    async def _delete_objects(self, knowledge_base_id: UUID) -> int:
        prefix = knowledge_base_object_prefix(knowledge_base_id)
        deleted = 0
        start_after: str | None = None
        while True:
            # The store lists by age; asking for everything older than the far
            # future is how this asks for all of them. A cutoff of "now" would
            # race an object written in the same second.
            page = await self._object_store.list_older_than(
                prefix=prefix,
                older_than=_EVERYTHING,
                limit=_OBJECT_PAGE_LIMIT,
                start_after=start_after,
            )
            for candidate in page.items:
                key = candidate.object_key
                # Guarded rather than trusted: a prefix sweep that drifted one
                # character would delete another knowledge base's artifacts.
                if not key.startswith(prefix):
                    continue
                if await self._object_store.delete_best_effort(key):
                    deleted += 1
            start_after = page.next_start_after
            if start_after is None or not page.items:
                return deleted

    async def _delete_rows(self, knowledge_base_id: UUID) -> None:
        async with self._sessions() as session, session.begin():
            document_ids = tuple(
                await session.scalars(
                    select(Document.id).where(Document.knowledge_base_id == knowledge_base_id)
                )
            )
            # Break both pointer cycles first: a knowledge base names its
            # generations and a document names its current version, so neither
            # target can be deleted while the pointer still stands.
            await session.execute(
                update(KnowledgeBase)
                .where(KnowledgeBase.id == knowledge_base_id)
                .values(active_index_generation_id=None, pending_index_generation_id=None)
            )
            if document_ids:
                await session.execute(
                    update(Document)
                    .where(Document.id.in_(document_ids))
                    .values(current_version_id=None, pending_version_id=None)
                )
            await self._delete_children(session, knowledge_base_id, document_ids)
            await session.execute(
                delete(KnowledgeBase).where(KnowledgeBase.id == knowledge_base_id)
            )

    @staticmethod
    async def _delete_children(
        session: AsyncSession,
        knowledge_base_id: UUID,
        document_ids: Sequence[UUID],
    ) -> None:
        # document_index_states cascades from both document_versions and
        # generations, so it is not listed here.
        await session.execute(
            delete(DocumentUploadIdempotency).where(
                DocumentUploadIdempotency.knowledge_base_id == knowledge_base_id
            )
        )
        await session.execute(
            delete(IndexGenerationCreationRequest).where(
                IndexGenerationCreationRequest.knowledge_base_id == knowledge_base_id
            )
        )
        # The purge job itself points at the knowledge base only through
        # target_id, which carries no foreign key, so clearing jobs here does not
        # delete the row that is running this.
        await session.execute(delete(Job).where(Job.knowledge_base_id == knowledge_base_id))
        await session.execute(
            delete(KnowledgeBaseMutation).where(
                KnowledgeBaseMutation.knowledge_base_id == knowledge_base_id
            )
        )
        await session.execute(
            delete(ApiKeyKnowledgeBaseScope).where(
                ApiKeyKnowledgeBaseScope.knowledge_base_id == knowledge_base_id
            )
        )
        if document_ids:
            await session.execute(
                delete(DocumentVersion).where(DocumentVersion.document_id.in_(document_ids))
            )
            await session.execute(delete(Document).where(Document.id.in_(document_ids)))
        await session.execute(
            delete(KnowledgeBaseIndexGeneration).where(
                KnowledgeBaseIndexGeneration.knowledge_base_id == knowledge_base_id
            )
        )

    def _sessions(self) -> AsyncSession:
        factory = self._session_factory
        if not callable(factory):
            raise ValueError("session factory is invalid")
        return factory()  # type: ignore[no-any-return]

    async def handle(self, context: JobExecutionContext) -> JobHandlerOutcome:
        """Job handler entry point.

        Reads the knowledge base from `target_id` rather than the job's
        `knowledge_base_id`, which is deliberately null: that column is a
        RESTRICT foreign key, so a job holding it could not outlive the row it
        exists to delete.
        """
        lease = context.lease
        if lease.operation != "purge_knowledge_base" or lease.target_type != "knowledge_base":
            raise ValueError("purge job target is invalid")
        await self.purge(lease.target_id)
        return JobHandlerOutcome.COMPLETE


__all__ = [
    "KnowledgeBasePurge",
    "PurgeObjectStore",
    "PurgeResult",
    "PurgeSearchIndex",
    "knowledge_base_object_prefix",
]
