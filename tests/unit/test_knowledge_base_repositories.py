from datetime import UTC, datetime
from typing import Any, cast
from uuid import uuid4

import pytest
from sqlalchemy.dialects import postgresql
from sqlalchemy.sql import Select

from rag_service.api.cursors import CursorPosition
from rag_service.db.models.auth import (
    ApiKey,
    ApiKeyKnowledgeBaseScope,
    AuditEvent,
    IdempotencyRecord,
)
from rag_service.db.models.knowledge_bases import KnowledgeBase, KnowledgeBaseMutation
from rag_service.metadata.knowledge_base_repositories import (
    SqlAlchemyIdempotencyRepository,
    SqlAlchemyKnowledgeBaseMutationRepository,
    SqlAlchemyKnowledgeBaseRepository,
    SqlAlchemyMetadataActorRepository,
    SqlAlchemyMetadataAuditRepository,
    sqlalchemy_knowledge_base_repositories,
)


class _ScalarRows:
    def __init__(self, rows: list[object]) -> None:
        self._rows = rows

    def all(self) -> list[object]:
        return self._rows


class _RepositorySession:
    def __init__(self) -> None:
        self.scalar_results: list[object | None] = []
        self.row_results: list[list[object]] = []
        self.statements: list[Select[tuple[Any, ...]]] = []
        self.added: list[object] = []
        self.flush_count = 0
        self.refreshed: list[tuple[object, tuple[str, ...] | None]] = []

    async def scalar(self, statement: Select[tuple[Any, ...]]) -> object | None:
        self.statements.append(statement)
        return self.scalar_results.pop(0)

    async def scalars(self, statement: Select[tuple[Any, ...]]) -> _ScalarRows:
        self.statements.append(statement)
        return _ScalarRows(self.row_results.pop(0))

    def add(self, row: object) -> None:
        self.added.append(row)

    async def flush(self) -> None:
        self.flush_count += 1

    async def refresh(
        self,
        row: object,
        attribute_names: list[str] | None = None,
    ) -> None:
        attributes = None if attribute_names is None else tuple(attribute_names)
        self.refreshed.append((row, attributes))


def _sql(statement: Select[tuple[Any, ...]]) -> str:
    return str(statement.compile(dialect=postgresql.dialect()))  # type: ignore[no-untyped-call]


def _knowledge_base() -> KnowledgeBase:
    now = datetime(2026, 7, 25, 8, 0, tzinfo=UTC)
    return KnowledgeBase(
        id=uuid4(),
        name="Repository KB",
        description=None,
        status="active",
        metadata_={},
        filter_schema={"fields": []},
        resource_revision=1,
        mutation_revision=0,
        filter_schema_revision=0,
        active_index_generation_id=None,
        pending_index_generation_id=None,
        created_at=now,
        updated_at=now,
    )


@pytest.mark.asyncio
async def test_knowledge_base_repository_scopes_before_cursor_and_owns_no_transaction() -> None:
    session = _RepositorySession()
    repository = SqlAlchemyKnowledgeBaseRepository(cast(Any, session))
    actor_key_id = uuid4()
    row = _knowledge_base()
    session.row_results = [[row], [row]]

    assert await repository.list_scoped(actor_key_id, None, 3) == [row]
    position = CursorPosition(created_at=row.created_at, id=row.id)
    assert await repository.list_scoped(actor_key_id, position, 2) == [row]

    first_sql, cursor_sql = (_sql(statement) for statement in session.statements)
    for statement in (first_sql, cursor_sql):
        assert "EXISTS (SELECT 1" in statement
        assert "api_key_knowledge_base_scopes" in statement
        assert "knowledge_bases.status !=" in statement
        assert "ORDER BY knowledge_bases.created_at, knowledge_bases.id" in statement
    assert "knowledge_bases.created_at >" not in first_sql
    assert "knowledge_bases.created_at >" in cursor_sql
    assert "knowledge_bases.id >" in cursor_sql

    session.statements.clear()
    session.scalar_results = [row, row]
    assert await repository.get_scoped(actor_key_id, row.id) is row
    assert await repository.get_scoped(actor_key_id, row.id, for_update=True) is row
    unlocked_sql, locked_sql = (_sql(statement) for statement in session.statements)
    assert "api_key_knowledge_base_scopes" in unlocked_sql
    assert "FOR UPDATE" not in unlocked_sql
    assert "FOR UPDATE" in locked_sql

    await repository.add(row)
    await repository.save(row)
    assert session.added == [row]
    assert session.flush_count == 2
    assert session.refreshed == [
        (row, ("created_at", "updated_at")),
        (row, ("updated_at",)),
    ]

    session.scalar_results = [row, None]
    assert await repository.reload(row) is row
    with pytest.raises(LookupError, match="disappeared"):
        await repository.reload(row)


@pytest.mark.asyncio
async def test_supporting_repositories_flush_rows_without_committing() -> None:
    session = _RepositorySession()
    row = _knowledge_base()
    actor = ApiKey(
        id=uuid4(),
        public_id="abcdefghijklmnopqrstuv",
        secret_digest=b"x" * 32,
        key_type="agent",
        name="Repository actor",
        status="active",
        capabilities=["manage"],
        raw_file_read=False,
        requests_per_minute=60,
        max_concurrency=4,
        resource_revision=1,
    )
    idempotency = IdempotencyRecord(
        id=uuid4(),
        actor_key_id=actor.id,
        operation="knowledge_base.create",
        idempotency_key="repository-idempotency",
        request_fingerprint=b"x" * 32,
        result_resource_type="knowledge_base",
        result_resource_id=row.id,
        http_status=201,
    )
    audit = AuditEvent(
        id=uuid4(),
        request_id="req-repository-audit",
        actor_api_key_id=actor.id,
        actor_kind="agent_key",
        action="knowledge_base.created",
        target_type="knowledge_base",
        target_id=row.id,
        metadata_={},
    )
    mutation = KnowledgeBaseMutation(
        id=uuid4(),
        knowledge_base_id=row.id,
        revision=1,
        mutation_type="metadata_changed",
        target_type="knowledge_base",
        target_id=row.id,
        payload={},
    )

    actor_repository = SqlAlchemyMetadataActorRepository(cast(Any, session))
    session.scalar_results = [actor]
    assert await actor_repository.get_agent_for_update(actor.id) is actor
    assert "FOR UPDATE" in _sql(session.statements[-1])
    await actor_repository.add_scope(actor.id, row.id)

    idempotency_repository = SqlAlchemyIdempotencyRepository(cast(Any, session))
    session.scalar_results = [idempotency]
    assert (
        await idempotency_repository.get(
            actor.id,
            "knowledge_base.create",
            "repository-idempotency",
        )
        is idempotency
    )
    await idempotency_repository.add(idempotency)
    await SqlAlchemyMetadataAuditRepository(cast(Any, session)).add(audit)
    await SqlAlchemyKnowledgeBaseMutationRepository(cast(Any, session)).add(mutation)

    assert len(session.added) == 4
    scope = cast(ApiKeyKnowledgeBaseScope, session.added[0])
    assert scope.api_key_id == actor.id
    assert scope.knowledge_base_id == row.id
    assert session.added[1:] == [idempotency, audit, mutation]
    assert session.flush_count == 4

    repositories = sqlalchemy_knowledge_base_repositories(cast(Any, session))
    assert isinstance(repositories.knowledge_bases, SqlAlchemyKnowledgeBaseRepository)
    assert isinstance(repositories.actors, SqlAlchemyMetadataActorRepository)
    assert isinstance(repositories.idempotency, SqlAlchemyIdempotencyRepository)
    assert isinstance(repositories.audits, SqlAlchemyMetadataAuditRepository)
    assert isinstance(
        repositories.mutations,
        SqlAlchemyKnowledgeBaseMutationRepository,
    )
