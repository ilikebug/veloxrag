import asyncio
import json
from collections.abc import Callable
from dataclasses import replace
from typing import Any, cast
from uuid import UUID, uuid4

import httpx
import pytest
from pydantic import SecretStr
from sqlalchemy import event, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from rag_service.api.errors import BusinessError
from rag_service.auth.policies import AdminPrincipal, AgentPrincipal, Capability
from rag_service.auth.schemas import (
    AdminApiKeyCreate,
    AgentApiKeyCreate,
    AgentApiKeyUpdate,
    SafeApiKey,
)
from rag_service.auth.services import ApiKeyService
from rag_service.config import Settings, get_settings
from rag_service.db.models.auth import (
    ApiKeyKnowledgeBaseScope,
    AuditEvent,
    IdempotencyRecord,
)
from rag_service.db.models.documents import Job
from rag_service.db.models.knowledge_bases import KnowledgeBase, KnowledgeBaseMutation
from rag_service.db.session import Database
from rag_service.main import create_app
from rag_service.metadata.knowledge_base_repositories import (
    IdempotencyRepository,
    KnowledgeBaseRepositories,
    sqlalchemy_knowledge_base_repositories,
)
from rag_service.metadata.schemas import (
    FilterSchemaReplacement,
    KnowledgeBaseCreate,
    KnowledgeBasePatch,
)
from rag_service.metadata.services import KnowledgeBaseService

ADMIN_HMAC_SECRET = "kb-admin-test-hmac-secret-32-bytes"
AGENT_HMAC_SECRET = "kb-agent-test-hmac-secret-32-bytes"


def _settings(database_url: str) -> Settings:
    return Settings(
        _env_file=None,
        environment="test",
        database_url=SecretStr(database_url),
        admin_key_hmac_secret=SecretStr(ADMIN_HMAC_SECRET),
        agent_key_hmac_secret=SecretStr(AGENT_HMAC_SECRET),
        default_page_size=2,
        max_page_size=3,
        max_api_key_requests_per_minute=100,
        max_api_key_concurrency=10,
    )


def _api_key_service(
    database: Database,
    session: AsyncSession,
    settings: Settings,
) -> ApiKeyService:
    return ApiKeyService(
        session=session,
        authentication_sessions=database.sessions,
        settings=settings,
    )


def _knowledge_base_service(
    session: AsyncSession,
    settings: Settings,
    *,
    repository_factory: Callable[[AsyncSession], KnowledgeBaseRepositories] | None = None,
) -> KnowledgeBaseService:
    if repository_factory is None:
        return KnowledgeBaseService(session=session, settings=settings)
    return KnowledgeBaseService(
        session=session,
        settings=settings,
        repository_factory=repository_factory,
    )


async def _create_admin(
    database: Database,
    settings: Settings,
) -> tuple[AdminPrincipal, str]:
    async with database.sessions() as session:
        issued = await _api_key_service(database, session, settings).create_admin_key(
            command=AdminApiKeyCreate(name="kb-service-admin"),
            request_id="req-kb-admin-create",
        )
    return (
        AdminPrincipal(
            key_id=issued.api_key.id,
            public_id=issued.api_key.public_id,
        ),
        issued.token.get_secret_value(),
    )


async def _create_manage_agent(
    database: Database,
    settings: Settings,
    admin: AdminPrincipal,
    *,
    name: str,
) -> tuple[AgentPrincipal, SafeApiKey]:
    async with database.sessions() as session:
        issued = await _api_key_service(database, session, settings).create_agent_key(
            AgentApiKeyCreate(
                name=name,
                capabilities=frozenset({Capability.MANAGE}),
                knowledge_base_ids=frozenset(),
                query_profile_ids=frozenset(),
                default_query_profile_id=None,
                raw_file_read=False,
                requests_per_minute=60,
                max_concurrency=4,
            ),
            actor=admin,
            request_id=f"req-{name}-create",
        )
    safe = issued.api_key
    return (
        AgentPrincipal(
            key_id=safe.id,
            public_id=safe.public_id,
            capabilities=frozenset({Capability.MANAGE}),
            knowledge_base_ids=frozenset(),
            query_profile_ids=frozenset(),
            default_query_profile_id=None,
            raw_file_read=False,
            requests_per_minute=60,
            max_concurrency=4,
        ),
        safe,
    )


async def _agent_detail(
    database: Database,
    settings: Settings,
    key_id: UUID,
) -> SafeApiKey:
    async with database.sessions() as session:
        return await _api_key_service(database, session, settings).get_agent_key(key_id)


async def _agent_detail_via_admin_http(
    database: Database,
    settings: Settings,
    admin_token: str,
    key_id: UUID,
) -> SafeApiKey:
    app = create_app(settings=settings, database=database)
    app.state.database = database
    app.dependency_overrides[get_settings] = lambda: settings
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.get(
            f"/v1/admin/api-keys/{key_id}",
            headers={"Authorization": f"Bearer {admin_token}"},
        )

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    document = response.json()
    assert isinstance(document, dict)
    assert set(document) == set(SafeApiKey.model_fields)
    assert all("token" not in field.lower() and "digest" not in field.lower() for field in document)
    serialized = json.dumps(document, sort_keys=True)
    assert admin_token not in serialized
    safe = SafeApiKey.model_validate(document)
    assert safe.etag is not None
    assert response.headers["etag"] == safe.etag
    return safe


async def _count(database: Database, model: type[object]) -> int:
    async with database.sessions() as session:
        return cast(
            int,
            await session.scalar(select(func.count()).select_from(model)),
        )


async def _audit_events(database: Database, target_id: UUID) -> list[AuditEvent]:
    async with database.sessions() as session:
        return list(
            (
                await session.scalars(
                    select(AuditEvent)
                    .where(AuditEvent.target_id == target_id)
                    .order_by(AuditEvent.created_at, AuditEvent.id)
                )
            ).all()
        )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_create_equal_replay_conflict_and_removed_scope_are_atomic(
    migrated_database: Database,
    postgres_urls: tuple[str, str],
) -> None:
    async_url, _sync_url = postgres_urls
    settings = _settings(async_url)
    admin, admin_token = await _create_admin(migrated_database, settings)
    actor, initial_agent = await _create_manage_agent(
        migrated_database,
        settings,
        admin,
        name="creator-agent",
    )
    creator_before = await _agent_detail_via_admin_http(
        migrated_database,
        settings,
        admin_token,
        actor.key_id,
    )
    assert creator_before.etag == initial_agent.etag == (f'"agent-key:{actor.key_id}:r1"')
    command = KnowledgeBaseCreate(
        name="Product manuals",
        description="Support documentation",
        metadata={"owner": "support", "priority": 1},
    )

    async with migrated_database.sessions() as session:
        service = _knowledge_base_service(session, settings)
        first = await service.create_knowledge_base(
            command,
            actor=actor,
            request_id="req-kb-first",
            idempotency_key="kb-create-1",
        )
    assert first.created is True
    assert first.knowledge_base.resource_revision == 1
    assert first.knowledge_base.mutation_revision == 0
    assert first.knowledge_base.filter_schema_revision == 0
    assert first.knowledge_base.etag == (f'"kb:{first.knowledge_base.id}:r1"')

    creator_after_first = await _agent_detail_via_admin_http(
        migrated_database,
        settings,
        admin_token,
        actor.key_id,
    )
    assert initial_agent.etag is not None
    assert creator_after_first.resource_revision == initial_agent.resource_revision + 1
    assert creator_after_first.etag == f'"agent-key:{actor.key_id}:r2"'
    assert creator_after_first.knowledge_base_ids == (first.knowledge_base.id,)

    async with migrated_database.sessions() as session:
        replay = await _knowledge_base_service(session, settings).create_knowledge_base(
            KnowledgeBaseCreate(
                metadata={"priority": 1, "owner": "support"},
                description="Support documentation",
                name="Product manuals",
            ),
            actor=actor,
            request_id="req-kb-replay",
            idempotency_key="kb-create-1",
        )
    assert replay.created is False
    assert replay.knowledge_base == first.knowledge_base
    creator_after_replay = await _agent_detail_via_admin_http(
        migrated_database,
        settings,
        admin_token,
        actor.key_id,
    )
    assert (
        creator_after_replay.etag == creator_after_first.etag == (f'"agent-key:{actor.key_id}:r2"')
    )

    async with migrated_database.sessions() as session:
        with pytest.raises(BusinessError) as conflict:
            await _knowledge_base_service(session, settings).create_knowledge_base(
                KnowledgeBaseCreate(name="Different request"),
                actor=actor,
                request_id="req-kb-conflict",
                idempotency_key="kb-create-1",
            )
    assert (conflict.value.status_code, conflict.value.code) == (
        409,
        "IDEMPOTENCY_CONFLICT",
    )

    assert creator_after_first.etag is not None
    async with migrated_database.sessions() as session:
        api_key_service = _api_key_service(
            database=migrated_database,
            session=session,
            settings=settings,
        )
        await api_key_service.update_agent_key(
            actor.key_id,
            AgentApiKeyUpdate(knowledge_base_ids=frozenset()),
            actor=admin,
            request_id="req-remove-created-scope",
            expected_etag=creator_after_first.etag,
        )

    async with migrated_database.sessions() as session:
        with pytest.raises(BusinessError) as removed_scope:
            await _knowledge_base_service(session, settings).create_knowledge_base(
                command,
                actor=actor,
                request_id="req-kb-replay-without-scope",
                idempotency_key="kb-create-1",
            )
    assert (removed_scope.value.status_code, removed_scope.value.code) == (
        404,
        "RESOURCE_NOT_FOUND",
    )
    assert await _count(migrated_database, KnowledgeBase) == 1
    assert await _count(migrated_database, IdempotencyRecord) == 1
    events = await _audit_events(migrated_database, first.knowledge_base.id)
    assert [(event.action, event.metadata_) for event in events] == [
        ("knowledge_base.created", {}),
    ]


class _ConcurrentAbsenceGate:
    def __init__(self) -> None:
        self._count = 0
        self._lock = asyncio.Lock()
        self._both_absent = asyncio.Event()

    async def wait_for_both(self) -> None:
        async with self._lock:
            self._count += 1
            if self._count == 2:
                self._both_absent.set()
        await asyncio.wait_for(self._both_absent.wait(), timeout=5)


class _BarrierIdempotencyRepository:
    def __init__(
        self,
        delegate: IdempotencyRepository,
        gate: _ConcurrentAbsenceGate,
    ) -> None:
        self._delegate = delegate
        self._gate = gate
        self._initial_lookup_complete = False

    async def get(
        self,
        actor_key_id: UUID,
        operation: str,
        idempotency_key: str,
    ) -> IdempotencyRecord | None:
        record = await self._delegate.get(actor_key_id, operation, idempotency_key)
        if record is None and not self._initial_lookup_complete:
            self._initial_lookup_complete = True
            await self._gate.wait_for_both()
        return record

    async def add(self, record: IdempotencyRecord) -> None:
        await self._delegate.add(record)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_concurrent_equal_create_converges_through_unique_savepoint(
    migrated_database: Database,
    postgres_urls: tuple[str, str],
) -> None:
    async_url, _sync_url = postgres_urls
    settings = _settings(async_url)
    admin, admin_token = await _create_admin(migrated_database, settings)
    actor, initial_agent = await _create_manage_agent(
        migrated_database,
        settings,
        admin,
        name="concurrent-creator",
    )
    creator_before = await _agent_detail_via_admin_http(
        migrated_database,
        settings,
        admin_token,
        actor.key_id,
    )
    assert creator_before.etag == initial_agent.etag == (f'"agent-key:{actor.key_id}:r1"')
    command = KnowledgeBaseCreate(
        name="Concurrent manuals",
        metadata={"team": "docs"},
    )
    gate = _ConcurrentAbsenceGate()

    def repository_factory(session: AsyncSession) -> KnowledgeBaseRepositories:
        repositories = sqlalchemy_knowledge_base_repositories(session)
        return replace(
            repositories,
            idempotency=_BarrierIdempotencyRepository(repositories.idempotency, gate),
        )

    async def invoke(request_id: str) -> object:
        async with migrated_database.sessions() as session:
            return await _knowledge_base_service(
                session,
                settings,
                repository_factory=repository_factory,
            ).create_knowledge_base(
                command,
                actor=actor,
                request_id=request_id,
                idempotency_key="concurrent-create",
            )

    raw_results = await asyncio.gather(
        invoke("req-concurrent-one"),
        invoke("req-concurrent-two"),
    )
    results = cast(list[Any], raw_results)
    assert sorted(result.created for result in results) == [False, True]
    assert len({result.knowledge_base.id for result in results}) == 1
    knowledge_base_id = results[0].knowledge_base.id
    assert await _count(migrated_database, KnowledgeBase) == 1
    assert await _count(migrated_database, IdempotencyRecord) == 1
    assert await _count(migrated_database, ApiKeyKnowledgeBaseScope) == 1
    events = await _audit_events(migrated_database, knowledge_base_id)
    assert [event.action for event in events] == ["knowledge_base.created"]
    creator = await _agent_detail_via_admin_http(
        migrated_database,
        settings,
        admin_token,
        actor.key_id,
    )
    assert creator.etag == f'"agent-key:{actor.key_id}:r2"'
    assert creator.resource_revision == initial_agent.resource_revision + 1
    assert creator.knowledge_base_ids == (knowledge_base_id,)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_update_disable_delete_repeat_and_stale_preconditions_preserve_revisions(
    migrated_database: Database,
    postgres_urls: tuple[str, str],
) -> None:
    async_url, _sync_url = postgres_urls
    settings = _settings(async_url)
    admin, _admin_token = await _create_admin(migrated_database, settings)
    actor, _initial_agent = await _create_manage_agent(
        migrated_database,
        settings,
        admin,
        name="lifecycle-agent",
    )
    command = KnowledgeBaseCreate(
        name="Lifecycle KB",
        description="before",
        metadata={"phase": 1},
    )
    async with migrated_database.sessions() as session:
        service = _knowledge_base_service(session, settings)
        first = await service.create_knowledge_base(
            command,
            actor=actor,
            request_id="req-lifecycle-create",
            idempotency_key="lifecycle-create",
        )
    knowledge_base_id = first.knowledge_base.id
    create_etag = first.knowledge_base.etag

    async with migrated_database.sessions() as session:
        service = _knowledge_base_service(session, settings)
        with pytest.raises(BusinessError) as missing:
            await service.update_knowledge_base(
                knowledge_base_id,
                KnowledgeBasePatch(name="must-not-apply"),
                actor=actor,
                request_id="req-update-missing",
                expected_etag=None,
            )
    assert (missing.value.status_code, missing.value.code) == (412, "PRECONDITION_FAILED")

    async with migrated_database.sessions() as session:
        updated = await _knowledge_base_service(session, settings).update_knowledge_base(
            knowledge_base_id,
            KnowledgeBasePatch(
                name="Lifecycle KB disabled",
                description=None,
                metadata={"phase": 2},
                status="disabled",
            ),
            actor=actor,
            request_id="req-lifecycle-update",
            expected_etag=create_etag,
        )
    assert (
        updated.name,
        updated.description,
        updated.metadata,
        updated.status,
        updated.resource_revision,
        updated.mutation_revision,
        updated.filter_schema_revision,
    ) == (
        "Lifecycle KB disabled",
        None,
        {"phase": 2},
        "disabled",
        2,
        0,
        0,
    )

    async with migrated_database.sessions() as session:
        with pytest.raises(BusinessError) as stale_update:
            await _knowledge_base_service(session, settings).update_knowledge_base(
                knowledge_base_id,
                KnowledgeBasePatch(name="stale"),
                actor=actor,
                request_id="req-update-stale",
                expected_etag=create_etag,
            )
    assert (stale_update.value.status_code, stale_update.value.code) == (
        412,
        "PRECONDITION_FAILED",
    )

    async with migrated_database.sessions() as session:
        deleted = await _knowledge_base_service(session, settings).delete_knowledge_base(
            knowledge_base_id,
            actor=actor,
            request_id="req-delete-first",
            expected_etag=updated.etag,
        )
    assert (deleted.status, deleted.resource_revision) == ("deleting", 3)

    async with migrated_database.sessions() as session:
        repeated = await _knowledge_base_service(session, settings).delete_knowledge_base(
            knowledge_base_id,
            actor=actor,
            request_id="req-delete-repeat",
            expected_etag=deleted.etag,
        )
    assert repeated == deleted

    async with migrated_database.sessions() as session:
        with pytest.raises(BusinessError) as stale_delete:
            await _knowledge_base_service(session, settings).delete_knowledge_base(
                knowledge_base_id,
                actor=actor,
                request_id="req-delete-stale",
                expected_etag=updated.etag,
            )
    async with migrated_database.sessions() as session:
        with pytest.raises(BusinessError) as stale_patch:
            await _knowledge_base_service(session, settings).update_knowledge_base(
                knowledge_base_id,
                KnowledgeBasePatch(name="stale-before-state"),
                actor=actor,
                request_id="req-patch-deleting-stale",
                expected_etag=updated.etag,
            )
    async with migrated_database.sessions() as session:
        with pytest.raises(BusinessError) as current_patch:
            await _knowledge_base_service(session, settings).update_knowledge_base(
                knowledge_base_id,
                KnowledgeBasePatch(name="current-state-conflict"),
                actor=actor,
                request_id="req-patch-deleting-current",
                expected_etag=deleted.etag,
            )
    assert (stale_delete.value.status_code, stale_delete.value.code) == (
        412,
        "PRECONDITION_FAILED",
    )
    assert (stale_patch.value.status_code, stale_patch.value.code) == (
        412,
        "PRECONDITION_FAILED",
    )
    assert (current_patch.value.status_code, current_patch.value.code) == (
        409,
        "RESOURCE_STATE_CONFLICT",
    )

    async with migrated_database.sessions() as session:
        direct = await _knowledge_base_service(session, settings).get_knowledge_base(
            knowledge_base_id,
            actor=actor,
        )
    async with migrated_database.sessions() as session:
        page = await _knowledge_base_service(session, settings).list_knowledge_bases(
            actor=actor,
        )
    async with migrated_database.sessions() as session:
        replay = await _knowledge_base_service(session, settings).create_knowledge_base(
            command,
            actor=actor,
            request_id="req-lifecycle-replay",
            idempotency_key="lifecycle-create",
        )
    assert direct.status == "deleting"
    assert page.items == ()
    assert replay.created is False
    assert replay.knowledge_base == deleted
    assert await _count(migrated_database, KnowledgeBaseMutation) == 0
    events = await _audit_events(migrated_database, knowledge_base_id)
    assert [(event.action, event.metadata_) for event in events] == [
        ("knowledge_base.created", {}),
        (
            "knowledge_base.updated",
            {"changed_fields": ["description", "metadata", "name", "status"]},
        ),
        ("knowledge_base.deletion_requested", {}),
    ]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_scoped_reads_and_mutations_do_not_enumerate_other_knowledge_bases(
    migrated_database: Database,
    postgres_urls: tuple[str, str],
) -> None:
    async_url, _sync_url = postgres_urls
    settings = _settings(async_url)
    admin, _admin_token = await _create_admin(migrated_database, settings)
    owner, _owner_safe = await _create_manage_agent(
        migrated_database,
        settings,
        admin,
        name="scope-owner",
    )
    outsider, _outsider_safe = await _create_manage_agent(
        migrated_database,
        settings,
        admin,
        name="scope-outsider",
    )
    async with migrated_database.sessions() as session:
        created = await _knowledge_base_service(session, settings).create_knowledge_base(
            KnowledgeBaseCreate(name="Private KB"),
            actor=owner,
            request_id="req-private-create",
            idempotency_key="private-create",
        )
    target_id = created.knowledge_base.id

    failures: list[BusinessError] = []
    async with migrated_database.sessions() as session:
        assert (
            await _knowledge_base_service(session, settings).list_knowledge_bases(
                actor=outsider,
            )
        ).items == ()
    for identifier in (target_id, uuid4()):
        async with migrated_database.sessions() as session:
            with pytest.raises(BusinessError) as missing:
                await _knowledge_base_service(session, settings).get_knowledge_base(
                    identifier,
                    actor=outsider,
                )
            failures.append(missing.value)
    async with migrated_database.sessions() as session:
        with pytest.raises(BusinessError) as patch_denied:
            await _knowledge_base_service(session, settings).update_knowledge_base(
                target_id,
                KnowledgeBasePatch(name="enumeration-attempt"),
                actor=outsider,
                request_id="req-outsider-patch",
                expected_etag=created.knowledge_base.etag,
            )
    async with migrated_database.sessions() as session:
        with pytest.raises(BusinessError) as delete_denied:
            await _knowledge_base_service(session, settings).delete_knowledge_base(
                target_id,
                actor=outsider,
                request_id="req-outsider-delete",
                expected_etag=created.knowledge_base.etag,
            )
    assert [(error.status_code, error.code, error.message) for error in failures] == [
        (404, "RESOURCE_NOT_FOUND", "Resource not found"),
        (404, "RESOURCE_NOT_FOUND", "Resource not found"),
    ]
    assert (patch_denied.value.status_code, patch_denied.value.code) == (
        404,
        "RESOURCE_NOT_FOUND",
    )
    assert (delete_denied.value.status_code, delete_denied.value.code) == (
        404,
        "RESOURCE_NOT_FOUND",
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_scoped_pagination_is_stable_bounded_and_one_query_per_page(
    migrated_database: Database,
    postgres_urls: tuple[str, str],
) -> None:
    async_url, _sync_url = postgres_urls
    settings = _settings(async_url)
    admin, _admin_token = await _create_admin(migrated_database, settings)
    actor, _initial_agent = await _create_manage_agent(
        migrated_database,
        settings,
        admin,
        name="pagination-agent",
    )
    created = []
    for index in range(4):
        async with migrated_database.sessions() as session:
            result = await _knowledge_base_service(
                session,
                settings,
            ).create_knowledge_base(
                KnowledgeBaseCreate(name=f"Page KB {index}"),
                actor=actor,
                request_id=f"req-page-create-{index}",
                idempotency_key=f"page-create-{index}",
            )
        created.append(result.knowledge_base)

    excluded = created[1]
    async with migrated_database.sessions() as session:
        await _knowledge_base_service(session, settings).delete_knowledge_base(
            excluded.id,
            actor=actor,
            request_id="req-page-delete",
            expected_etag=excluded.etag,
        )

    select_statements: list[str] = []

    def capture_selects(
        _connection: object,
        _cursor: object,
        statement: str,
        _parameters: object,
        _context: object,
        _executemany: bool,
    ) -> None:
        if statement.lstrip().upper().startswith("SELECT"):
            select_statements.append(statement)

    event.listen(
        migrated_database.engine.sync_engine,
        "before_cursor_execute",
        capture_selects,
    )
    try:
        async with migrated_database.sessions() as session:
            first_page = await _knowledge_base_service(
                session,
                settings,
            ).list_knowledge_bases(actor=actor, limit=2)
        assert first_page.next_cursor is not None
        async with migrated_database.sessions() as session:
            second_page = await _knowledge_base_service(
                session,
                settings,
            ).list_knowledge_bases(
                actor=actor,
                cursor=first_page.next_cursor,
                limit=2,
            )
    finally:
        event.remove(
            migrated_database.engine.sync_engine,
            "before_cursor_execute",
            capture_selects,
        )

    expected = sorted(
        (item for item in created if item.id != excluded.id),
        key=lambda item: (item.created_at, item.id),
    )
    assert [*first_page.items, *second_page.items] == expected
    assert second_page.next_cursor is None
    assert len(select_statements) == 2
    assert all("FROM knowledge_bases" in statement for statement in select_statements)
    assert all("api_key_knowledge_base_scopes" in statement for statement in select_statements)

    for invalid_limit in (0, settings.max_page_size + 1, True):
        async with migrated_database.sessions() as session:
            with pytest.raises(BusinessError) as invalid_page:
                await _knowledge_base_service(
                    session,
                    settings,
                ).list_knowledge_bases(actor=actor, limit=invalid_limit)
        assert (invalid_page.value.status_code, invalid_page.value.code) == (
            422,
            "VALIDATION_ERROR",
        )
    async with migrated_database.sessions() as session:
        with pytest.raises(BusinessError) as invalid_cursor:
            await _knowledge_base_service(session, settings).list_knowledge_bases(
                actor=actor,
                cursor="not-a-canonical-cursor",
            )
    assert (invalid_cursor.value.status_code, invalid_cursor.value.code) == (
        422,
        "VALIDATION_ERROR",
    )


class _CancellingAuditRepository:
    def __init__(self, delegate: Any) -> None:
        self._delegate = delegate

    async def add(self, event: AuditEvent) -> None:
        await self._delegate.add(event)
        raise asyncio.CancelledError


@pytest.mark.integration
@pytest.mark.asyncio
async def test_cancellation_rolls_back_creation_scope_idempotency_revision_and_audit(
    migrated_database: Database,
    postgres_urls: tuple[str, str],
) -> None:
    async_url, _sync_url = postgres_urls
    settings = _settings(async_url)
    admin, _admin_token = await _create_admin(migrated_database, settings)
    actor, initial_agent = await _create_manage_agent(
        migrated_database,
        settings,
        admin,
        name="cancelled-creator",
    )

    def repository_factory(session: AsyncSession) -> KnowledgeBaseRepositories:
        repositories = sqlalchemy_knowledge_base_repositories(session)
        return replace(
            repositories,
            audits=_CancellingAuditRepository(repositories.audits),
        )

    with pytest.raises(asyncio.CancelledError):
        async with migrated_database.sessions() as session:
            await _knowledge_base_service(
                session,
                settings,
                repository_factory=repository_factory,
            ).create_knowledge_base(
                KnowledgeBaseCreate(name="Must roll back"),
                actor=actor,
                request_id="req-cancelled-create",
                idempotency_key="cancelled-create",
            )

    assert await _count(migrated_database, KnowledgeBase) == 0
    assert await _count(migrated_database, ApiKeyKnowledgeBaseScope) == 0
    assert await _count(migrated_database, IdempotencyRecord) == 0
    creator = await _agent_detail(migrated_database, settings, actor.key_id)
    assert creator.resource_revision == initial_agent.resource_revision
    async with migrated_database.sessions() as session:
        leaked_events = await session.scalar(
            select(func.count())
            .select_from(AuditEvent)
            .where(AuditEvent.action == "knowledge_base.created")
        )
        assert leaked_events == 0


def _filter_schema_command(
    *,
    source_path: str = "attributes.department",
) -> FilterSchemaReplacement:
    return FilterSchemaReplacement.model_validate(
        {
            "fields": [
                {
                    "name": "department",
                    "source_path": source_path,
                    "type": "keyword",
                    "operators": ["in", "eq"],
                },
                {
                    "name": "priority",
                    "source_path": "attributes.priority",
                    "type": "integer",
                    "operators": ["lte", "gte", "eq"],
                },
            ]
        }
    )


async def _knowledge_base_row(database: Database, identifier: UUID) -> KnowledgeBase:
    async with database.sessions() as session:
        row = await session.get(KnowledgeBase, identifier)
        assert row is not None
        return row


async def _mutations(
    database: Database,
    identifier: UUID,
) -> list[KnowledgeBaseMutation]:
    async with database.sessions() as session:
        return list(
            (
                await session.scalars(
                    select(KnowledgeBaseMutation)
                    .where(KnowledgeBaseMutation.knowledge_base_id == identifier)
                    .order_by(KnowledgeBaseMutation.revision)
                )
            ).all()
        )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_filter_schema_replacement_versions_storage_mutations_and_audit_without_jobs(
    migrated_database: Database,
    postgres_urls: tuple[str, str],
) -> None:
    async_url, _sync_url = postgres_urls
    settings = _settings(async_url)
    admin, _admin_token = await _create_admin(migrated_database, settings)
    actor, _initial_agent = await _create_manage_agent(
        migrated_database,
        settings,
        admin,
        name="filter-schema-manager",
    )
    async with migrated_database.sessions() as session:
        created = await _knowledge_base_service(session, settings).create_knowledge_base(
            KnowledgeBaseCreate(name="Filter schema KB"),
            actor=actor,
            request_id="req-filter-create",
            idempotency_key="filter-schema-create",
        )
    knowledge_base_id = created.knowledge_base.id

    async with migrated_database.sessions() as session:
        first = await _knowledge_base_service(session, settings).replace_filter_schema(
            knowledge_base_id,
            _filter_schema_command(),
            actor=actor,
            request_id="req-filter-first",
            expected_etag=created.knowledge_base.etag,
        )

    assert (
        first.resource_revision,
        first.mutation_revision,
        first.filter_schema_revision,
        first.etag,
    ) == (2, 1, 1, f'"kb:{knowledge_base_id}:r2"')
    first_document = first.model_dump(mode="json")
    assert set(first_document) == {
        "fields",
        "resource_revision",
        "mutation_revision",
        "filter_schema_revision",
        "etag",
    }
    assert [field["operators"] for field in first_document["fields"]] == [
        ["eq", "in"],
        ["eq", "gte", "lte"],
    ]
    serialized_public = json.dumps(first_document, sort_keys=True)
    assert "field_id" not in serialized_public
    assert "payload_path" not in serialized_public

    stored = await _knowledge_base_row(migrated_database, knowledge_base_id)
    assert (
        stored.resource_revision,
        stored.mutation_revision,
        stored.filter_schema_revision,
    ) == (2, 1, 1)
    stored_fields = cast(list[dict[str, object]], stored.filter_schema["fields"])
    retained_identifiers = {
        cast(str, field["name"]): (
            cast(str, field["field_id"]),
            cast(str, field["payload_path"]),
        )
        for field in stored_fields
    }
    assert all(field_id.startswith("fld_") for field_id, _path in retained_identifiers.values())
    assert all(path.startswith("metadata.f_") for _field_id, path in retained_identifiers.values())

    mutations = await _mutations(migrated_database, knowledge_base_id)
    assert len(mutations) == 1
    assert (
        mutations[0].revision,
        mutations[0].mutation_type,
        mutations[0].target_type,
        mutations[0].target_id,
    ) == (1, "filter_schema_changed", "knowledge_base", knowledge_base_id)
    assert mutations[0].payload == {
        "filter_schema_revision": 1,
        "fields": [
            {
                "field_id": retained_identifiers["department"][0],
                "type": "keyword",
                "operators": ["eq", "in"],
            },
            {
                "field_id": retained_identifiers["priority"][0],
                "type": "integer",
                "operators": ["eq", "gte", "lte"],
            },
        ],
    }
    mutation_text = json.dumps(mutations[0].payload, sort_keys=True)
    assert "source_path" not in mutation_text
    assert "payload_path" not in mutation_text
    assert "attributes." not in mutation_text
    assert await _count(migrated_database, Job) == 0

    events = await _audit_events(migrated_database, knowledge_base_id)
    assert [(event.action, event.metadata_) for event in events] == [
        ("knowledge_base.created", {}),
        (
            "knowledge_base.filter_schema_replaced",
            {
                "field_count": 2,
                "filter_schema_revision": 1,
                "mutation_revision": 1,
                "resource_revision": 2,
            },
        ),
    ]

    async with migrated_database.sessions() as session:
        identical = await _knowledge_base_service(session, settings).replace_filter_schema(
            knowledge_base_id,
            _filter_schema_command(),
            actor=actor,
            request_id="req-filter-identical",
            expected_etag=first.etag,
        )
    assert (
        identical.resource_revision,
        identical.mutation_revision,
        identical.filter_schema_revision,
    ) == (3, 2, 2)
    stored_identical = await _knowledge_base_row(migrated_database, knowledge_base_id)
    identical_fields = cast(
        list[dict[str, object]],
        stored_identical.filter_schema["fields"],
    )
    assert {
        cast(str, field["name"]): (
            cast(str, field["field_id"]),
            cast(str, field["payload_path"]),
        )
        for field in identical_fields
    } == retained_identifiers

    renamed = FilterSchemaReplacement.model_validate(
        {
            "fields": [
                {
                    "name": "departmentV2",
                    "source_path": "attributes.department",
                    "type": "keyword",
                    "operators": ["eq"],
                }
            ]
        }
    )
    async with migrated_database.sessions() as session:
        renamed_result = await _knowledge_base_service(session, settings).replace_filter_schema(
            knowledge_base_id,
            renamed,
            actor=actor,
            request_id="req-filter-renamed",
            expected_etag=identical.etag,
        )
    assert (
        renamed_result.resource_revision,
        renamed_result.mutation_revision,
        renamed_result.filter_schema_revision,
    ) == (4, 3, 3)
    stored_renamed = await _knowledge_base_row(migrated_database, knowledge_base_id)
    renamed_field = cast(list[dict[str, object]], stored_renamed.filter_schema["fields"])[0]
    assert renamed_field["field_id"] != retained_identifiers["department"][0]
    assert renamed_field["payload_path"] != retained_identifiers["department"][1]
    assert len(await _mutations(migrated_database, knowledge_base_id)) == 3
    assert await _count(migrated_database, Job) == 0


class _FailingKnowledgeBaseSave:
    def __init__(self, delegate: Any) -> None:
        self._delegate = delegate

    def __getattr__(self, name: str) -> object:
        return getattr(self._delegate, name)

    async def save(self, row: KnowledgeBase) -> None:
        await self._delegate.save(row)
        raise RuntimeError("filter-schema-save-failure")


class _FailingMutationAdd:
    def __init__(self, delegate: Any, failure: BaseException | None = None) -> None:
        self._delegate = delegate
        self._failure = (
            RuntimeError("filter-schema-mutation-failure") if failure is None else failure
        )

    async def add(self, mutation: KnowledgeBaseMutation) -> None:
        await self._delegate.add(mutation)
        raise self._failure


class _FailingFilterAuditAdd:
    def __init__(self, delegate: Any, failure: BaseException | None = None) -> None:
        self._delegate = delegate
        self._failure = RuntimeError("filter-schema-audit-failure") if failure is None else failure

    async def add(self, audit: AuditEvent) -> None:
        await self._delegate.add(audit)
        if audit.action == "knowledge_base.filter_schema_replaced":
            raise self._failure


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("failure_stage", ["save", "mutation", "audit"])
async def test_filter_schema_failures_roll_back_schema_revisions_mutation_audit_and_jobs(
    migrated_database: Database,
    postgres_urls: tuple[str, str],
    failure_stage: str,
) -> None:
    async_url, _sync_url = postgres_urls
    settings = _settings(async_url)
    admin, _admin_token = await _create_admin(migrated_database, settings)
    actor, _initial_agent = await _create_manage_agent(
        migrated_database,
        settings,
        admin,
        name=f"filter-rollback-{failure_stage}",
    )
    async with migrated_database.sessions() as session:
        created = await _knowledge_base_service(session, settings).create_knowledge_base(
            KnowledgeBaseCreate(name=f"Filter rollback {failure_stage}"),
            actor=actor,
            request_id=f"req-filter-rollback-create-{failure_stage}",
            idempotency_key=f"filter-rollback-create-{failure_stage}",
        )

    def repository_factory(session: AsyncSession) -> KnowledgeBaseRepositories:
        repositories = sqlalchemy_knowledge_base_repositories(session)
        if failure_stage == "save":
            return replace(
                repositories,
                knowledge_bases=cast(Any, _FailingKnowledgeBaseSave(repositories.knowledge_bases)),
            )
        if failure_stage == "mutation":
            return replace(
                repositories,
                mutations=cast(Any, _FailingMutationAdd(repositories.mutations)),
            )
        return replace(
            repositories,
            audits=cast(Any, _FailingFilterAuditAdd(repositories.audits)),
        )

    async with migrated_database.sessions() as session:
        with pytest.raises(BusinessError) as failure:
            await _knowledge_base_service(
                session,
                settings,
                repository_factory=repository_factory,
            ).replace_filter_schema(
                created.knowledge_base.id,
                _filter_schema_command(),
                actor=actor,
                request_id=f"req-filter-rollback-{failure_stage}",
                expected_etag=created.knowledge_base.etag,
            )
    assert (failure.value.status_code, failure.value.code) == (500, "INTERNAL_ERROR")

    persisted = await _knowledge_base_row(migrated_database, created.knowledge_base.id)
    assert persisted.filter_schema == {"fields": []}
    assert (
        persisted.resource_revision,
        persisted.mutation_revision,
        persisted.filter_schema_revision,
    ) == (1, 0, 0)
    assert await _count(migrated_database, KnowledgeBaseMutation) == 0
    assert await _count(migrated_database, Job) == 0
    events = await _audit_events(migrated_database, created.knowledge_base.id)
    assert [(event.action, event.metadata_) for event in events] == [("knowledge_base.created", {})]


def _filter_traceback_frame_surface(frame_locals: dict[str, object]) -> str:
    rendered = [repr(frame_locals)]
    for value in frame_locals.values():
        if isinstance(value, KnowledgeBase):
            rendered.append(repr(value.filter_schema))
        elif isinstance(value, KnowledgeBaseMutation):
            rendered.append(repr(value.payload))
        elif isinstance(value, FilterSchemaReplacement):
            rendered.append(repr(value.model_dump(mode="json")))
    return "\n".join(rendered)


def _assert_filter_metadata_traceback_redacted(
    error: BaseException,
    *forbidden_values: str,
) -> None:
    metadata_frames = 0
    traceback = error.__traceback__
    while traceback is not None:
        if "/rag_service/metadata/" in traceback.tb_frame.f_code.co_filename:
            metadata_frames += 1
            rendered = _filter_traceback_frame_surface(traceback.tb_frame.f_locals)
            for forbidden in forbidden_values:
                assert forbidden not in rendered
        traceback = traceback.tb_next
    assert metadata_frames >= 1


def _assert_filter_execution_traceback_redacted(
    error: BaseException,
    *forbidden_values: str,
) -> None:
    checked_frames = 0
    traceback = error.__traceback__
    while traceback is not None:
        frame = traceback.tb_frame
        if not frame.f_code.co_name.startswith("test_"):
            checked_frames += 1
            rendered = _filter_traceback_frame_surface(frame.f_locals)
            for forbidden in forbidden_values:
                assert forbidden not in rendered
        traceback = traceback.tb_next
    assert checked_frames >= 2


@pytest.mark.integration
@pytest.mark.asyncio
async def test_filter_schema_business_error_drops_stored_and_request_data_from_metadata_frames(
    migrated_database: Database,
    postgres_urls: tuple[str, str],
) -> None:
    async_url, _sync_url = postgres_urls
    settings = _settings(async_url)
    admin, _admin_token = await _create_admin(migrated_database, settings)
    actor, _initial_agent = await _create_manage_agent(
        migrated_database,
        settings,
        admin,
        name="filter-schema-business-redaction",
    )
    async with migrated_database.sessions() as session:
        created = await _knowledge_base_service(session, settings).create_knowledge_base(
            KnowledgeBaseCreate(name="Filter business redaction"),
            actor=actor,
            request_id="req-filter-business-create",
            idempotency_key="filter-business-create",
        )
    async with migrated_database.sessions() as session:
        initial = await _knowledge_base_service(session, settings).replace_filter_schema(
            created.knowledge_base.id,
            _filter_schema_command(),
            actor=actor,
            request_id="req-filter-business-initial",
            expected_etag=created.knowledge_base.etag,
        )
    stored = await _knowledge_base_row(migrated_database, created.knowledge_base.id)
    stored_field = cast(list[dict[str, object]], stored.filter_schema["fields"])[0]
    field_id = cast(str, stored_field["field_id"])
    payload_path = cast(str, stored_field["payload_path"])
    changed_source_path = "private.businessMarker"

    async with migrated_database.sessions() as session:
        with pytest.raises(BusinessError) as raised:
            await _knowledge_base_service(session, settings).replace_filter_schema(
                created.knowledge_base.id,
                _filter_schema_command(source_path=changed_source_path),
                actor=actor,
                request_id="req-filter-business-change",
                expected_etag=initial.etag,
            )
    assert (raised.value.status_code, raised.value.code) == (422, "VALIDATION_ERROR")
    _assert_filter_metadata_traceback_redacted(
        raised.value,
        changed_source_path,
        "attributes.department",
        field_id,
        payload_path,
    )
    persisted = await _knowledge_base_row(migrated_database, created.knowledge_base.id)
    assert persisted.filter_schema == stored.filter_schema
    assert (
        persisted.resource_revision,
        persisted.mutation_revision,
        persisted.filter_schema_revision,
    ) == (2, 1, 1)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_filter_schema_runtime_failure_sanitizes_exact_error_frames_and_rolls_back(
    migrated_database: Database,
    postgres_urls: tuple[str, str],
) -> None:
    async_url, _sync_url = postgres_urls
    settings = _settings(async_url)
    admin, _admin_token = await _create_admin(migrated_database, settings)
    actor, _initial_agent = await _create_manage_agent(
        migrated_database,
        settings,
        admin,
        name="filter-schema-runtime-redaction",
    )
    async with migrated_database.sessions() as session:
        created = await _knowledge_base_service(session, settings).create_knowledge_base(
            KnowledgeBaseCreate(name="Filter runtime redaction"),
            actor=actor,
            request_id="req-filter-runtime-create",
            idempotency_key="filter-runtime-create",
        )
    source_path = "private.runtimeMarker"
    async with migrated_database.sessions() as session:
        initial = await _knowledge_base_service(session, settings).replace_filter_schema(
            created.knowledge_base.id,
            _filter_schema_command(source_path=source_path),
            actor=actor,
            request_id="req-filter-runtime-initial",
            expected_etag=created.knowledge_base.etag,
        )
    stored = await _knowledge_base_row(migrated_database, created.knowledge_base.id)
    stored_field = cast(list[dict[str, object]], stored.filter_schema["fields"])[0]
    field_id = cast(str, stored_field["field_id"])
    payload_path = cast(str, stored_field["payload_path"])
    marker = "filter-runtime-retained-marker"
    retained_error = RuntimeError(marker)
    retained_error.__cause__ = ValueError(f"{marker}-cause")
    retained_error.__context__ = LookupError(f"{marker}-context")

    def repository_factory(session: AsyncSession) -> KnowledgeBaseRepositories:
        repositories = sqlalchemy_knowledge_base_repositories(session)
        return replace(
            repositories,
            audits=cast(
                Any,
                _FailingFilterAuditAdd(repositories.audits, retained_error),
            ),
        )

    async with migrated_database.sessions() as session:
        with pytest.raises(BusinessError) as raised:
            await _knowledge_base_service(
                session,
                settings,
                repository_factory=repository_factory,
            ).replace_filter_schema(
                created.knowledge_base.id,
                _filter_schema_command(source_path=source_path),
                actor=actor,
                request_id="req-filter-runtime-failure",
                expected_etag=initial.etag,
            )
    assert (
        raised.value.status_code,
        raised.value.code,
        raised.value.message,
    ) == (500, "INTERNAL_ERROR", "Internal server error")
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None
    assert retained_error.args == ("<redacted>",)
    assert retained_error.__traceback__ is None
    assert retained_error.__cause__ is None
    assert retained_error.__context__ is None
    _assert_filter_metadata_traceback_redacted(
        raised.value,
        marker,
        source_path,
        field_id,
        payload_path,
    )
    persisted = await _knowledge_base_row(migrated_database, created.knowledge_base.id)
    assert persisted.filter_schema == stored.filter_schema
    assert (
        persisted.resource_revision,
        persisted.mutation_revision,
        persisted.filter_schema_revision,
    ) == (2, 1, 1)
    assert len(await _mutations(migrated_database, created.knowledge_base.id)) == 1
    assert len(await _audit_events(migrated_database, created.knowledge_base.id)) == 2
    assert await _count(migrated_database, Job) == 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_filter_schema_cancellation_preserves_identity_redacts_frames_and_rolls_back(
    migrated_database: Database,
    postgres_urls: tuple[str, str],
) -> None:
    async_url, _sync_url = postgres_urls
    settings = _settings(async_url)
    admin, _admin_token = await _create_admin(migrated_database, settings)
    actor, _initial_agent = await _create_manage_agent(
        migrated_database,
        settings,
        admin,
        name="filter-schema-cancellation-redaction",
    )
    async with migrated_database.sessions() as session:
        created = await _knowledge_base_service(session, settings).create_knowledge_base(
            KnowledgeBaseCreate(name="Filter cancellation redaction"),
            actor=actor,
            request_id="req-filter-cancel-create",
            idempotency_key="filter-cancel-create",
        )
    source_path = "private.cancellationMarker"
    async with migrated_database.sessions() as session:
        initial = await _knowledge_base_service(session, settings).replace_filter_schema(
            created.knowledge_base.id,
            _filter_schema_command(source_path=source_path),
            actor=actor,
            request_id="req-filter-cancel-initial",
            expected_etag=created.knowledge_base.etag,
        )
    stored = await _knowledge_base_row(migrated_database, created.knowledge_base.id)
    stored_fields = cast(list[dict[str, object]], stored.filter_schema["fields"])
    internal_values = tuple(
        cast(str, field[key]) for field in stored_fields for key in ("field_id", "payload_path")
    )
    cancelled = asyncio.CancelledError("filter schema cancelled")

    def repository_factory(session: AsyncSession) -> KnowledgeBaseRepositories:
        repositories = sqlalchemy_knowledge_base_repositories(session)
        return replace(
            repositories,
            mutations=cast(
                Any,
                _FailingMutationAdd(repositories.mutations, cancelled),
            ),
        )

    async with migrated_database.sessions() as session:
        with pytest.raises(asyncio.CancelledError) as raised:
            await _knowledge_base_service(
                session,
                settings,
                repository_factory=repository_factory,
            ).replace_filter_schema(
                created.knowledge_base.id,
                _filter_schema_command(source_path=source_path),
                actor=actor,
                request_id="req-filter-cancel-failure",
                expected_etag=initial.etag,
            )
    assert raised.value is cancelled
    assert cancelled.args == ("filter schema cancelled",)
    _assert_filter_execution_traceback_redacted(
        cancelled,
        source_path,
        *internal_values,
    )
    persisted = await _knowledge_base_row(migrated_database, created.knowledge_base.id)
    assert persisted.filter_schema == stored.filter_schema
    assert (
        persisted.resource_revision,
        persisted.mutation_revision,
        persisted.filter_schema_revision,
    ) == (2, 1, 1)
    assert len(await _mutations(migrated_database, created.knowledge_base.id)) == 1
    assert len(await _audit_events(migrated_database, created.knowledge_base.id)) == 2
    assert await _count(migrated_database, Job) == 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_corrupt_stored_filter_identifier_pair_returns_sanitized_500_and_rolls_back(
    migrated_database: Database,
    postgres_urls: tuple[str, str],
) -> None:
    async_url, _sync_url = postgres_urls
    settings = _settings(async_url)
    admin, _admin_token = await _create_admin(migrated_database, settings)
    actor, _initial_agent = await _create_manage_agent(
        migrated_database,
        settings,
        admin,
        name="filter-schema-corrupt-pair",
    )
    async with migrated_database.sessions() as session:
        created = await _knowledge_base_service(session, settings).create_knowledge_base(
            KnowledgeBaseCreate(name="Filter corrupt pair"),
            actor=actor,
            request_id="req-filter-corrupt-create",
            idempotency_key="filter-corrupt-create",
        )
    source_path = "private.corruptMarker"
    field_id = "fld_ERERERERQRGBEREREREREQ"
    payload_path = "metadata.f_22222222222242228222222222222222"
    corrupted_schema = {
        "fields": [
            {
                **_filter_schema_command(source_path=source_path).fields[0].model_dump(mode="json"),
                "field_id": field_id,
                "payload_path": payload_path,
            }
        ]
    }
    async with migrated_database.sessions() as session, session.begin():
        row = await session.get(KnowledgeBase, created.knowledge_base.id)
        assert row is not None
        row.filter_schema = corrupted_schema

    async with migrated_database.sessions() as session:
        with pytest.raises(BusinessError) as raised:
            await _knowledge_base_service(session, settings).replace_filter_schema(
                created.knowledge_base.id,
                FilterSchemaReplacement(fields=()),
                actor=actor,
                request_id="req-filter-corrupt-replace",
                expected_etag=created.knowledge_base.etag,
            )
    assert (
        raised.value.status_code,
        raised.value.code,
        raised.value.message,
    ) == (500, "INTERNAL_ERROR", "Internal server error")
    _assert_filter_metadata_traceback_redacted(
        raised.value,
        source_path,
        field_id,
        payload_path,
    )
    persisted = await _knowledge_base_row(migrated_database, created.knowledge_base.id)
    assert persisted.filter_schema == corrupted_schema
    assert (
        persisted.resource_revision,
        persisted.mutation_revision,
        persisted.filter_schema_revision,
    ) == (1, 0, 0)
    assert await _count(migrated_database, KnowledgeBaseMutation) == 0
    assert await _count(migrated_database, Job) == 0
    assert [
        (event.action, event.metadata_)
        for event in await _audit_events(
            migrated_database,
            created.knowledge_base.id,
        )
    ] == [("knowledge_base.created", {})]
