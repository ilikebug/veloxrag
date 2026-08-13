from dataclasses import dataclass
from typing import Protocol, cast
from uuid import UUID

from sqlalchemy import and_, delete, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from rag_service.api.cursors import CursorPosition
from rag_service.auth.codec import KeyKind
from rag_service.db.models.auth import (
    ApiKey,
    ApiKeyKnowledgeBaseScope,
    ApiKeyQueryProfileScope,
    AuditEvent,
)
from rag_service.db.models.knowledge_bases import KnowledgeBase
from rag_service.db.models.providers import QueryProfile


class ApiKeyRepository(Protocol):
    async def get_by_public_id(
        self,
        kind: KeyKind,
        public_id: str,
        *,
        for_update: bool = False,
    ) -> ApiKey | None: ...

    async def get_by_id(
        self,
        key_id: UUID,
        kind: KeyKind,
        *,
        for_update: bool = False,
    ) -> ApiKey | None: ...

    async def add(self, key: ApiKey) -> None: ...

    async def replace_kb_scopes(self, key_id: UUID, ids: frozenset[UUID]) -> None: ...

    async def replace_query_profile_scopes(
        self,
        key_id: UUID,
        ids: frozenset[UUID],
        default_id: UUID | None,
    ) -> None: ...

    async def list_by_kind(
        self,
        kind: KeyKind,
        position: CursorPosition | None,
        limit: int,
    ) -> list[ApiKey]: ...


class ApiKeyScopeRepository(Protocol):
    async def get_knowledge_base_statuses(
        self,
        ids: frozenset[UUID],
        *,
        for_update: bool = False,
    ) -> dict[UUID, str]: ...

    async def get_query_profile_enabled(
        self,
        ids: frozenset[UUID],
        *,
        for_update: bool = False,
    ) -> dict[UUID, bool]: ...

    async def get_knowledge_base_scopes(self, key_id: UUID) -> frozenset[UUID]: ...

    async def get_query_profile_scopes(
        self,
        key_id: UUID,
    ) -> tuple[frozenset[UUID], UUID | None]: ...

    async def get_knowledge_base_scopes_batch(
        self,
        key_ids: frozenset[UUID],
    ) -> dict[UUID, frozenset[UUID]]: ...

    async def get_query_profile_scopes_batch(
        self,
        key_ids: frozenset[UUID],
    ) -> dict[UUID, tuple[frozenset[UUID], UUID | None]]: ...


class AuditEventRepository(Protocol):
    async def add(self, event: AuditEvent) -> None: ...


@dataclass(frozen=True, slots=True)
class AuthRepositories:
    api_keys: ApiKeyRepository
    scopes: ApiKeyScopeRepository
    audits: AuditEventRepository


class SqlAlchemyApiKeyRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_by_public_id(
        self,
        kind: KeyKind,
        public_id: str,
        *,
        for_update: bool = False,
    ) -> ApiKey | None:
        statement = select(ApiKey).where(
            ApiKey.key_type == kind.value,
            ApiKey.public_id == public_id,
        )
        if for_update:
            statement = statement.with_for_update()
        return cast(ApiKey | None, await self._session.scalar(statement))

    async def get_by_id(
        self,
        key_id: UUID,
        kind: KeyKind,
        *,
        for_update: bool = False,
    ) -> ApiKey | None:
        statement = select(ApiKey).where(
            ApiKey.id == key_id,
            ApiKey.key_type == kind.value,
        )
        if for_update:
            statement = statement.with_for_update()
        return cast(ApiKey | None, await self._session.scalar(statement))

    async def add(self, key: ApiKey) -> None:
        self._session.add(key)
        await self._session.flush()
        await self._session.refresh(key, attribute_names=["created_at", "updated_at"])

    async def replace_kb_scopes(self, key_id: UUID, ids: frozenset[UUID]) -> None:
        await self._session.execute(
            delete(ApiKeyKnowledgeBaseScope).where(ApiKeyKnowledgeBaseScope.api_key_id == key_id)
        )
        self._session.add_all(
            ApiKeyKnowledgeBaseScope(api_key_id=key_id, knowledge_base_id=identifier)
            for identifier in sorted(ids, key=str)
        )
        await self._session.flush()

    async def replace_query_profile_scopes(
        self,
        key_id: UUID,
        ids: frozenset[UUID],
        default_id: UUID | None,
    ) -> None:
        await self._session.execute(
            delete(ApiKeyQueryProfileScope).where(ApiKeyQueryProfileScope.api_key_id == key_id)
        )
        self._session.add_all(
            ApiKeyQueryProfileScope(
                api_key_id=key_id,
                query_profile_id=identifier,
                is_default=identifier == default_id,
            )
            for identifier in sorted(ids, key=str)
        )
        await self._session.flush()

    async def list_by_kind(
        self,
        kind: KeyKind,
        position: CursorPosition | None,
        limit: int,
    ) -> list[ApiKey]:
        statement = select(ApiKey).where(ApiKey.key_type == kind.value)
        if position is not None:
            statement = statement.where(
                or_(
                    ApiKey.created_at > position.created_at,
                    and_(
                        ApiKey.created_at == position.created_at,
                        ApiKey.id > position.id,
                    ),
                )
            )
        statement = statement.order_by(ApiKey.created_at, ApiKey.id).limit(limit)
        return list((await self._session.scalars(statement)).all())


class SqlAlchemyApiKeyScopeRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_knowledge_base_statuses(
        self,
        ids: frozenset[UUID],
        *,
        for_update: bool = False,
    ) -> dict[UUID, str]:
        if not ids:
            return {}
        statement = (
            select(KnowledgeBase.id, KnowledgeBase.status)
            .where(KnowledgeBase.id.in_(ids))
            .order_by(KnowledgeBase.id)
        )
        if for_update:
            statement = statement.with_for_update()
        rows = await self._session.execute(statement)
        return {identifier: status for identifier, status in rows}

    async def get_query_profile_enabled(
        self,
        ids: frozenset[UUID],
        *,
        for_update: bool = False,
    ) -> dict[UUID, bool]:
        if not ids:
            return {}
        statement = (
            select(QueryProfile.id, QueryProfile.enabled)
            .where(QueryProfile.id.in_(ids))
            .order_by(QueryProfile.id)
        )
        if for_update:
            statement = statement.with_for_update()
        rows = await self._session.execute(statement)
        return {identifier: enabled for identifier, enabled in rows}

    async def get_knowledge_base_scopes(self, key_id: UUID) -> frozenset[UUID]:
        identifiers = await self._session.scalars(
            select(ApiKeyKnowledgeBaseScope.knowledge_base_id).where(
                ApiKeyKnowledgeBaseScope.api_key_id == key_id
            )
        )
        return frozenset(identifiers.all())

    async def get_query_profile_scopes(
        self,
        key_id: UUID,
    ) -> tuple[frozenset[UUID], UUID | None]:
        rows = (
            await self._session.execute(
                select(
                    ApiKeyQueryProfileScope.query_profile_id,
                    ApiKeyQueryProfileScope.is_default,
                ).where(ApiKeyQueryProfileScope.api_key_id == key_id)
            )
        ).all()
        identifiers = frozenset(identifier for identifier, _is_default in rows)
        default_ids = [identifier for identifier, is_default in rows if is_default]
        default_id = default_ids[0] if len(default_ids) == 1 else None
        return identifiers, default_id

    async def get_knowledge_base_scopes_batch(
        self,
        key_ids: frozenset[UUID],
    ) -> dict[UUID, frozenset[UUID]]:
        if not key_ids:
            return {}
        rows = await self._session.execute(
            select(
                ApiKeyKnowledgeBaseScope.api_key_id,
                ApiKeyKnowledgeBaseScope.knowledge_base_id,
            )
            .where(ApiKeyKnowledgeBaseScope.api_key_id.in_(key_ids))
            .order_by(
                ApiKeyKnowledgeBaseScope.api_key_id,
                ApiKeyKnowledgeBaseScope.knowledge_base_id,
            )
        )
        grouped: dict[UUID, set[UUID]] = {key_id: set() for key_id in key_ids}
        for key_id, knowledge_base_id in rows:
            grouped[key_id].add(knowledge_base_id)
        return {key_id: frozenset(identifiers) for key_id, identifiers in grouped.items()}

    async def get_query_profile_scopes_batch(
        self,
        key_ids: frozenset[UUID],
    ) -> dict[UUID, tuple[frozenset[UUID], UUID | None]]:
        if not key_ids:
            return {}
        rows = await self._session.execute(
            select(
                ApiKeyQueryProfileScope.api_key_id,
                ApiKeyQueryProfileScope.query_profile_id,
                ApiKeyQueryProfileScope.is_default,
            )
            .where(ApiKeyQueryProfileScope.api_key_id.in_(key_ids))
            .order_by(
                ApiKeyQueryProfileScope.api_key_id,
                ApiKeyQueryProfileScope.query_profile_id,
            )
        )
        identifiers_by_key: dict[UUID, set[UUID]] = {key_id: set() for key_id in key_ids}
        default_by_key: dict[UUID, UUID] = {}
        for key_id, query_profile_id, is_default in rows:
            identifiers_by_key[key_id].add(query_profile_id)
            if is_default:
                default_by_key[key_id] = query_profile_id
        return {
            key_id: (frozenset(identifiers), default_by_key.get(key_id))
            for key_id, identifiers in identifiers_by_key.items()
        }


class SqlAlchemyAuditEventRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def add(self, event: AuditEvent) -> None:
        self._session.add(event)
        await self._session.flush()


def sqlalchemy_auth_repositories(session: AsyncSession) -> AuthRepositories:
    return AuthRepositories(
        api_keys=SqlAlchemyApiKeyRepository(session),
        scopes=SqlAlchemyApiKeyScopeRepository(session),
        audits=SqlAlchemyAuditEventRepository(session),
    )
