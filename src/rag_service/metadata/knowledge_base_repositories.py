from dataclasses import dataclass
from typing import Protocol, cast
from uuid import UUID

from sqlalchemy import and_, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.elements import ColumnElement

from rag_service.api.cursors import CursorPosition
from rag_service.db.models.auth import (
    ApiKey,
    ApiKeyKnowledgeBaseScope,
    AuditEvent,
    IdempotencyRecord,
)
from rag_service.db.models.documents import Job
from rag_service.db.models.knowledge_bases import KnowledgeBase, KnowledgeBaseMutation
from rag_service.db.models.providers import ModelProfile


class KnowledgeBaseRepository(Protocol):
    async def list_scoped(
        self,
        actor_key_id: UUID,
        position: CursorPosition | None,
        limit: int,
    ) -> list[KnowledgeBase]: ...

    async def get_scoped(
        self,
        actor_key_id: UUID,
        knowledge_base_id: UUID,
        *,
        for_update: bool = False,
    ) -> KnowledgeBase | None: ...

    async def add(self, row: KnowledgeBase) -> None: ...

    async def save(self, row: KnowledgeBase) -> None: ...

    async def reload(self, row: KnowledgeBase) -> KnowledgeBase: ...


class MetadataActorRepository(Protocol):
    async def get_agent_for_update(self, key_id: UUID) -> ApiKey | None: ...

    async def add_scope(self, actor_key_id: UUID, knowledge_base_id: UUID) -> None: ...


class IdempotencyRepository(Protocol):
    async def get(
        self,
        actor_key_id: UUID,
        operation: str,
        idempotency_key: str,
    ) -> IdempotencyRecord | None: ...

    async def add(self, record: IdempotencyRecord) -> None: ...


class MetadataAuditRepository(Protocol):
    async def add(self, event: AuditEvent) -> None: ...


class ModelProfileLookup(Protocol):
    async def capability_of(self, profile_id: UUID) -> str | None: ...


class MetadataJobRepository(Protocol):
    async def add(self, job: Job) -> None: ...


class KnowledgeBaseMutationRepository(Protocol):
    async def add(self, mutation: KnowledgeBaseMutation) -> None: ...


@dataclass(frozen=True, slots=True)
class KnowledgeBaseRepositories:
    knowledge_bases: KnowledgeBaseRepository
    actors: MetadataActorRepository
    idempotency: IdempotencyRepository
    audits: MetadataAuditRepository
    mutations: KnowledgeBaseMutationRepository
    model_profiles: ModelProfileLookup
    jobs: MetadataJobRepository


def _scope_exists(actor_key_id: UUID) -> ColumnElement[bool]:
    return (
        select(1)
        .where(
            ApiKeyKnowledgeBaseScope.api_key_id == actor_key_id,
            ApiKeyKnowledgeBaseScope.knowledge_base_id == KnowledgeBase.id,
        )
        .exists()
    )


class SqlAlchemyKnowledgeBaseRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def list_scoped(
        self,
        actor_key_id: UUID,
        position: CursorPosition | None,
        limit: int,
    ) -> list[KnowledgeBase]:
        statement = select(KnowledgeBase).where(
            _scope_exists(actor_key_id),
            KnowledgeBase.status != "deleting",
        )
        if position is not None:
            statement = statement.where(
                or_(
                    KnowledgeBase.created_at > position.created_at,
                    and_(
                        KnowledgeBase.created_at == position.created_at,
                        KnowledgeBase.id > position.id,
                    ),
                )
            )
        statement = statement.order_by(KnowledgeBase.created_at, KnowledgeBase.id).limit(limit)
        return list((await self._session.scalars(statement)).all())

    async def get_scoped(
        self,
        actor_key_id: UUID,
        knowledge_base_id: UUID,
        *,
        for_update: bool = False,
    ) -> KnowledgeBase | None:
        statement = select(KnowledgeBase).where(
            KnowledgeBase.id == knowledge_base_id,
            _scope_exists(actor_key_id),
        )
        if for_update:
            statement = statement.with_for_update()
        return cast(KnowledgeBase | None, await self._session.scalar(statement))

    async def add(self, row: KnowledgeBase) -> None:
        self._session.add(row)
        await self._session.flush()
        await self._session.refresh(row, attribute_names=["created_at", "updated_at"])

    async def save(self, row: KnowledgeBase) -> None:
        await self._session.flush()
        await self._session.refresh(row, attribute_names=["updated_at"])

    async def reload(self, row: KnowledgeBase) -> KnowledgeBase:
        statement = (
            select(KnowledgeBase)
            .where(KnowledgeBase.id == row.id)
            .execution_options(populate_existing=True)
        )
        reloaded = cast(KnowledgeBase | None, await self._session.scalar(statement))
        if reloaded is None:
            raise LookupError("knowledge base disappeared")
        return reloaded


class SqlAlchemyMetadataActorRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_agent_for_update(self, key_id: UUID) -> ApiKey | None:
        return cast(
            ApiKey | None,
            await self._session.scalar(
                select(ApiKey)
                .where(ApiKey.id == key_id, ApiKey.key_type == "agent")
                .with_for_update()
            ),
        )

    async def add_scope(self, actor_key_id: UUID, knowledge_base_id: UUID) -> None:
        self._session.add(
            ApiKeyKnowledgeBaseScope(
                api_key_id=actor_key_id,
                knowledge_base_id=knowledge_base_id,
            )
        )
        await self._session.flush()


class SqlAlchemyIdempotencyRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get(
        self,
        actor_key_id: UUID,
        operation: str,
        idempotency_key: str,
    ) -> IdempotencyRecord | None:
        return cast(
            IdempotencyRecord | None,
            await self._session.scalar(
                select(IdempotencyRecord).where(
                    IdempotencyRecord.actor_key_id == actor_key_id,
                    IdempotencyRecord.operation == operation,
                    IdempotencyRecord.idempotency_key == idempotency_key,
                )
            ),
        )

    async def add(self, record: IdempotencyRecord) -> None:
        self._session.add(record)
        await self._session.flush()


class SqlAlchemyMetadataAuditRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def add(self, event: AuditEvent) -> None:
        self._session.add(event)
        await self._session.flush()


class SqlAlchemyKnowledgeBaseMutationRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def add(self, mutation: KnowledgeBaseMutation) -> None:
        self._session.add(mutation)
        await self._session.flush()


class SqlAlchemyMetadataJobRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def add(self, job: Job) -> None:
        self._session.add(job)
        await self._session.flush()


class SqlAlchemyModelProfileLookup:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def capability_of(self, profile_id: UUID) -> str | None:
        statement = select(ModelProfile.capability).where(ModelProfile.id == profile_id)
        return (await self._session.execute(statement)).scalar_one_or_none()


def sqlalchemy_knowledge_base_repositories(
    session: AsyncSession,
) -> KnowledgeBaseRepositories:
    return KnowledgeBaseRepositories(
        knowledge_bases=SqlAlchemyKnowledgeBaseRepository(session),
        actors=SqlAlchemyMetadataActorRepository(session),
        idempotency=SqlAlchemyIdempotencyRepository(session),
        audits=SqlAlchemyMetadataAuditRepository(session),
        mutations=SqlAlchemyKnowledgeBaseMutationRepository(session),
        model_profiles=SqlAlchemyModelProfileLookup(session),
        jobs=SqlAlchemyMetadataJobRepository(session),
    )
