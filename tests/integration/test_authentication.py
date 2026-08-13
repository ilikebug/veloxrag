import asyncio
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Literal
from uuid import UUID, uuid4

import pytest
from pydantic import SecretStr, ValidationError
from sqlalchemy import event, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from rag_service.api.errors import BusinessError
from rag_service.auth import services as auth_services
from rag_service.auth.codec import (
    GeneratedToken,
    KeyKind,
    digest_secret,
    generate_token,
    parse_token,
)
from rag_service.auth.policies import AdminPrincipal, AgentPrincipal, Capability
from rag_service.auth.repositories import (
    ApiKeyScopeRepository,
    AuthRepositories,
    sqlalchemy_auth_repositories,
)
from rag_service.auth.schemas import (
    AdminApiKeyCreate,
    AgentApiKeyCreate,
    AgentApiKeyUpdate,
)
from rag_service.auth.services import ApiKeyService
from rag_service.config import Settings
from rag_service.db.models.auth import (
    ApiKey,
    ApiKeyKnowledgeBaseScope,
    ApiKeyQueryProfileScope,
    AuditEvent,
)
from rag_service.db.models.knowledge_bases import KnowledgeBase
from rag_service.db.models.providers import QueryProfile
from rag_service.db.session import Database

NOW = datetime(2026, 7, 25, 8, 0, tzinfo=UTC)


class _BlockingReferenceScopeRepository:
    def __init__(
        self,
        delegate: ApiKeyScopeRepository,
        *,
        reference_kind: Literal["knowledge_base", "query_profile"],
        lock_acquired: asyncio.Event,
        release_validation: asyncio.Event,
    ) -> None:
        self._delegate = delegate
        self._reference_kind = reference_kind
        self._lock_acquired = lock_acquired
        self._release_validation = release_validation

    async def _block_after_lock(self, reference_kind: str, for_update: bool) -> None:
        if self._reference_kind == reference_kind and for_update:
            self._lock_acquired.set()
            await self._release_validation.wait()

    async def get_knowledge_base_statuses(
        self,
        ids: frozenset[UUID],
        *,
        for_update: bool = False,
    ) -> dict[UUID, str]:
        result = await self._delegate.get_knowledge_base_statuses(
            ids,
            for_update=for_update,
        )
        await self._block_after_lock("knowledge_base", for_update)
        return result

    async def get_query_profile_enabled(
        self,
        ids: frozenset[UUID],
        *,
        for_update: bool = False,
    ) -> dict[UUID, bool]:
        result = await self._delegate.get_query_profile_enabled(
            ids,
            for_update=for_update,
        )
        await self._block_after_lock("query_profile", for_update)
        return result

    async def get_knowledge_base_scopes(self, key_id: UUID) -> frozenset[UUID]:
        return await self._delegate.get_knowledge_base_scopes(key_id)

    async def get_query_profile_scopes(
        self,
        key_id: UUID,
    ) -> tuple[frozenset[UUID], UUID | None]:
        return await self._delegate.get_query_profile_scopes(key_id)

    async def get_knowledge_base_scopes_batch(
        self,
        key_ids: frozenset[UUID],
    ) -> dict[UUID, frozenset[UUID]]:
        return await self._delegate.get_knowledge_base_scopes_batch(key_ids)

    async def get_query_profile_scopes_batch(
        self,
        key_ids: frozenset[UUID],
    ) -> dict[UUID, tuple[frozenset[UUID], UUID | None]]:
        return await self._delegate.get_query_profile_scopes_batch(key_ids)


def _settings(
    *,
    admin_secret: str = "a" * 32,
    agent_secret: str = "b" * 32,
    max_request_id_length: int = 128,
) -> Settings:
    return Settings(
        environment="test",
        admin_key_hmac_secret=SecretStr(admin_secret),
        agent_key_hmac_secret=SecretStr(agent_secret),
        max_api_key_requests_per_minute=100,
        max_api_key_concurrency=10,
        max_request_id_length=max_request_id_length,
    )


def _service(
    database: Database, session: object, settings: Settings | None = None
) -> ApiKeyService:
    return ApiKeyService(
        session=session,  # type: ignore[arg-type]
        authentication_sessions=database.sessions,
        settings=settings or _settings(),
        clock=lambda: NOW,
    )


async def _insert_references(database: Database) -> tuple[UUID, UUID, UUID, UUID, UUID]:
    knowledge_base_id = uuid4()
    second_knowledge_base_id = uuid4()
    deleting_knowledge_base_id = uuid4()
    query_profile_id = uuid4()
    disabled_query_profile_id = uuid4()
    async with database.sessions() as session, session.begin():
        session.add_all(
            [
                KnowledgeBase(id=knowledge_base_id, name="primary"),
                KnowledgeBase(id=second_knowledge_base_id, name="secondary", status="disabled"),
                KnowledgeBase(
                    id=deleting_knowledge_base_id,
                    name="deleting",
                    status="deleting",
                ),
                QueryProfile(
                    id=query_profile_id,
                    name="query-profile",
                    dense_candidate_limit=20,
                    sparse_candidate_limit=20,
                    rrf_candidate_limit=20,
                    rerank_candidate_limit=10,
                    top_k_limit=5,
                    min_rerank_score=Decimal("0"),
                    min_rrf_score_when_degraded=Decimal("0"),
                    context_token_budget=4096,
                    enabled=True,
                    is_system_default=False,
                ),
                QueryProfile(
                    id=disabled_query_profile_id,
                    name="disabled-query-profile",
                    dense_candidate_limit=20,
                    sparse_candidate_limit=20,
                    rrf_candidate_limit=20,
                    rerank_candidate_limit=10,
                    top_k_limit=5,
                    min_rerank_score=Decimal("0"),
                    min_rrf_score_when_degraded=Decimal("0"),
                    context_token_budget=4096,
                    enabled=False,
                    is_system_default=False,
                ),
            ]
        )
    return (
        knowledge_base_id,
        second_knowledge_base_id,
        deleting_knowledge_base_id,
        query_profile_id,
        disabled_query_profile_id,
    )


def _agent_command(
    knowledge_base_id: UUID,
    query_profile_id: UUID,
    *,
    name: str = "retrieval-agent",
    not_before: datetime | None = None,
    expires_at: datetime | None = None,
) -> AgentApiKeyCreate:
    return AgentApiKeyCreate(
        name=name,
        capabilities=frozenset({Capability.RETRIEVE}),
        knowledge_base_ids=frozenset({knowledge_base_id}),
        query_profile_ids=frozenset({query_profile_id}),
        default_query_profile_id=query_profile_id,
        raw_file_read=True,
        requests_per_minute=60,
        max_concurrency=4,
        not_before=not_before,
        expires_at=expires_at,
    )


async def _create_admin(
    database: Database,
    session: object,
    *,
    request_id: str = "req-admin-create",
) -> tuple[str, AdminPrincipal]:
    service = _service(database, session)
    issued = await service.create_admin_key(
        AdminApiKeyCreate(name="local-admin"),
        request_id=request_id,
    )
    token = issued.token.get_secret_value()
    principal = await service.authenticate(token, KeyKind.ADMIN)
    assert isinstance(principal, AdminPrincipal)
    return token, principal


def _assert_invalid_api_key(error: BusinessError) -> None:
    assert (error.status_code, error.code, error.message) == (
        401,
        "INVALID_API_KEY",
        "Invalid API key",
    )


@pytest.mark.parametrize(
    "request_id",
    (
        "operator@example.com",
        "contains whitespace",
        "contains\nnewline",
        "contains\x00control",
        "x" * 33,
    ),
)
@pytest.mark.integration
@pytest.mark.asyncio
async def test_direct_service_rejects_unsafe_request_id_without_database_residue(
    migrated_database: Database,
    request_id: str,
) -> None:
    async with migrated_database.sessions() as session:
        service = _service(
            migrated_database,
            session,
            _settings(max_request_id_length=32),
        )
        with pytest.raises(BusinessError) as raised:
            await service.create_admin_key(
                AdminApiKeyCreate(name="local-admin"),
                request_id=request_id,
            )
        assert (
            raised.value.status_code,
            raised.value.code,
            raised.value.message,
            raised.value.args,
        ) == (
            422,
            "VALIDATION_ERROR",
            "Invalid API key policy",
            ("Invalid API key policy",),
        )
        assert request_id not in repr(raised.value)

    async with migrated_database.sessions() as inspection:
        assert await inspection.scalar(select(func.count()).select_from(ApiKey)) == 0
        assert await inspection.scalar(select(func.count()).select_from(AuditEvent)) == 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_committed_creation_authentication_key_class_and_hmac_rotation(
    migrated_database: Database,
) -> None:
    (
        knowledge_base_id,
        _second_kb_id,
        _deleting_kb_id,
        query_profile_id,
        _disabled_qp_id,
    ) = await _insert_references(migrated_database)
    async with migrated_database.sessions() as session:
        service = _service(migrated_database, session)
        admin_token, admin = await _create_admin(migrated_database, session)
        issued = await service.create_agent_key(
            _agent_command(knowledge_base_id, query_profile_id),
            actor=admin,
            request_id="req-agent-create",
        )
        agent_token = issued.token.get_secret_value()

        principal = await service.authenticate(agent_token, KeyKind.AGENT)
        assert principal == AgentPrincipal(
            key_id=issued.api_key.id,
            public_id=issued.api_key.public_id,
            capabilities=frozenset({Capability.RETRIEVE}),
            knowledge_base_ids=frozenset({knowledge_base_id}),
            query_profile_ids=frozenset({query_profile_id}),
            default_query_profile_id=query_profile_id,
            raw_file_read=True,
            requests_per_minute=60,
            max_concurrency=4,
        )

        wrong_secret = generate_token(KeyKind.AGENT, _settings().agent_key_hmac_secret)
        replacement_secret = wrong_secret.token.rsplit(".", 1)[1]
        forged = f"rag_agent_{issued.api_key.public_id}.{replacement_secret}"
        for token, kind in (
            (admin_token, KeyKind.AGENT),
            (agent_token, KeyKind.ADMIN),
            (forged, KeyKind.AGENT),
            ("malformed", KeyKind.AGENT),
        ):
            with pytest.raises(BusinessError) as raised:
                await service.authenticate(token, kind)
            _assert_invalid_api_key(raised.value)

        admin_rotated = _service(
            migrated_database,
            session,
            _settings(admin_secret="c" * 32),
        )
        with pytest.raises(BusinessError) as raised:
            await admin_rotated.authenticate(admin_token, KeyKind.ADMIN)
        _assert_invalid_api_key(raised.value)
        assert isinstance(
            await admin_rotated.authenticate(agent_token, KeyKind.AGENT),
            AgentPrincipal,
        )

        agent_rotated = _service(
            migrated_database,
            session,
            _settings(agent_secret="d" * 32),
        )
        assert isinstance(
            await agent_rotated.authenticate(admin_token, KeyKind.ADMIN),
            AdminPrincipal,
        )
        with pytest.raises(BusinessError) as raised:
            await agent_rotated.authenticate(agent_token, KeyKind.AGENT)
        _assert_invalid_api_key(raised.value)

        safe_dump = repr(issued.api_key.model_dump())
        assert agent_token not in safe_dump
        assert "secret" not in safe_dump.lower()
        assert "digest" not in safe_dump.lower()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_status_validity_windows_and_immediate_revocation_are_uniformly_invalid(
    migrated_database: Database,
) -> None:
    (
        knowledge_base_id,
        _second_kb_id,
        _deleting_kb_id,
        query_profile_id,
        _disabled_qp_id,
    ) = await _insert_references(migrated_database)
    async with migrated_database.sessions() as session:
        service = _service(migrated_database, session)
        _admin_token, admin = await _create_admin(migrated_database, session)
        active = await service.create_agent_key(
            _agent_command(knowledge_base_id, query_profile_id, name="active"),
            actor=admin,
            request_id="req-active",
        )
        future = await service.create_agent_key(
            _agent_command(
                knowledge_base_id,
                query_profile_id,
                name="future",
                not_before=NOW + timedelta(minutes=1),
            ),
            actor=admin,
            request_id="req-future",
        )
        expired = await service.create_agent_key(
            _agent_command(
                knowledge_base_id,
                query_profile_id,
                name="expired",
                expires_at=NOW,
            ),
            actor=admin,
            request_id="req-expired",
        )
        disabled = await service.create_agent_key(
            _agent_command(knowledge_base_id, query_profile_id, name="disabled"),
            actor=admin,
            request_id="req-disabled",
        )
        disabled_safe = await service.update_agent_key(
            disabled.api_key.id,
            AgentApiKeyUpdate(status="disabled"),
            actor=admin,
            request_id="req-disable",
            expected_etag=disabled.api_key.etag or "",
        )
        assert disabled_safe.status == "disabled"
        revoked_safe = await service.revoke_agent_key(
            active.api_key.id,
            actor=admin,
            request_id="req-revoke",
            expected_etag=active.api_key.etag,
        )
        assert revoked_safe.status == "revoked"

        for issued in (active, future, expired, disabled):
            with pytest.raises(BusinessError) as raised:
                await service.authenticate(issued.token.get_secret_value(), KeyKind.AGENT)
            _assert_invalid_api_key(raised.value)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_explicit_scopes_default_constraint_safe_lists_and_invalid_reference_rollback(
    migrated_database: Database,
) -> None:
    (
        knowledge_base_id,
        second_kb_id,
        _deleting_kb_id,
        query_profile_id,
        _disabled_qp_id,
    ) = await _insert_references(migrated_database)
    async with migrated_database.sessions() as session:
        service = _service(migrated_database, session)
        _admin_token, admin = await _create_admin(migrated_database, session)
        first = await service.create_agent_key(
            _agent_command(knowledge_base_id, query_profile_id, name="first"),
            actor=admin,
            request_id="req-first",
        )
        second = await service.create_agent_key(
            _agent_command(second_kb_id, query_profile_id, name="second"),
            actor=admin,
            request_id="req-second",
        )

        detail = await service.get_agent_key(first.api_key.id)
        assert detail.knowledge_base_ids == (knowledge_base_id,)
        assert detail.query_profile_ids == (query_profile_id,)
        assert detail.default_query_profile_id == query_profile_id
        assert "token" not in repr(detail.model_dump()).lower()

        page_one = await service.list_agent_keys(limit=1)
        assert len(page_one.items) == 1
        assert page_one.next_cursor is not None
        page_two = await service.list_agent_keys(cursor=page_one.next_cursor, limit=1)
        assert len(page_two.items) == 1
        assert {page_one.items[0].id, page_two.items[0].id} == {
            first.api_key.id,
            second.api_key.id,
        }

    async with migrated_database.sessions() as inspection:
        default_scope = await inspection.scalar(
            select(ApiKeyQueryProfileScope).where(
                ApiKeyQueryProfileScope.api_key_id == first.api_key.id,
                ApiKeyQueryProfileScope.is_default.is_(True),
            )
        )
        assert default_scope is not None
        assert default_scope.query_profile_id == query_profile_id
        before_keys = await inspection.scalar(select(func.count()).select_from(ApiKey))
        before_scopes = await inspection.scalar(
            select(func.count()).select_from(ApiKeyKnowledgeBaseScope)
        )
        before_audits = await inspection.scalar(select(func.count()).select_from(AuditEvent))

    async with migrated_database.sessions() as invalid_session:
        invalid_service = _service(migrated_database, invalid_session)
        with pytest.raises(BusinessError) as raised:
            await invalid_service.create_agent_key(
                _agent_command(uuid4(), query_profile_id, name="invalid"),
                actor=admin,
                request_id="req-invalid",
            )
        assert (raised.value.status_code, raised.value.code) == (422, "VALIDATION_ERROR")

    async with migrated_database.sessions() as inspection:
        assert await inspection.scalar(select(func.count()).select_from(ApiKey)) == before_keys
        assert (
            await inspection.scalar(select(func.count()).select_from(ApiKeyKnowledgeBaseScope))
            == before_scopes
        )
        assert (
            await inspection.scalar(select(func.count()).select_from(AuditEvent)) == before_audits
        )

    with pytest.raises(ValidationError):
        AgentApiKeyCreate(
            name="bad-default",
            capabilities=frozenset(),
            knowledge_base_ids=frozenset(),
            query_profile_ids=frozenset(),
            default_query_profile_id=query_profile_id,
            requests_per_minute=1,
            max_concurrency=1,
        )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_agent_list_loads_visible_page_scopes_in_three_selects(
    migrated_database: Database,
) -> None:
    (
        knowledge_base_id,
        disabled_knowledge_base_id,
        _deleting_knowledge_base_id,
        query_profile_id,
        _disabled_query_profile_id,
    ) = await _insert_references(migrated_database)
    issued_keys = []
    async with migrated_database.sessions() as session:
        service = _service(migrated_database, session)
        _admin_token, admin = await _create_admin(migrated_database, session)
        for index, scoped_knowledge_base_id in enumerate(
            (
                knowledge_base_id,
                disabled_knowledge_base_id,
                knowledge_base_id,
                disabled_knowledge_base_id,
            )
        ):
            issued_keys.append(
                await service.create_agent_key(
                    _agent_command(
                        scoped_knowledge_base_id,
                        query_profile_id,
                        name=f"listed-agent-{index}",
                    ),
                    actor=admin,
                    request_id=f"req-listed-agent-{index}",
                )
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
            page = await service.list_agent_keys(limit=3)
        finally:
            event.remove(
                migrated_database.engine.sync_engine,
                "before_cursor_execute",
                capture_selects,
            )

    assert len(select_statements) == 3
    assert any("FROM api_keys" in statement for statement in select_statements)
    assert any("FROM api_key_knowledge_base_scopes" in statement for statement in select_statements)
    assert any("FROM api_key_query_profile_scopes" in statement for statement in select_statements)
    expected_order = [
        issued.api_key.id
        for issued in sorted(
            issued_keys,
            key=lambda issued: (issued.api_key.created_at, issued.api_key.id),
        )[:3]
    ]
    assert [item.id for item in page.items] == expected_order
    assert page.next_cursor is not None
    serialized = repr(page.model_dump())
    assert all(issued.token.get_secret_value() not in serialized for issued in issued_keys)
    assert "secret" not in serialized.lower()
    assert "digest" not in serialized.lower()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_rejected_reference_states_leave_no_agent_key_scope_or_audit_residue(
    migrated_database: Database,
) -> None:
    (
        active_knowledge_base_id,
        _disabled_knowledge_base_id,
        deleting_knowledge_base_id,
        enabled_query_profile_id,
        disabled_query_profile_id,
    ) = await _insert_references(migrated_database)
    async with migrated_database.sessions() as admin_session:
        _admin_token, admin = await _create_admin(migrated_database, admin_session)

    async with migrated_database.sessions() as inspection:
        baseline_counts = (
            await inspection.scalar(select(func.count()).select_from(ApiKey)),
            await inspection.scalar(select(func.count()).select_from(ApiKeyKnowledgeBaseScope)),
            await inspection.scalar(select(func.count()).select_from(ApiKeyQueryProfileScope)),
            await inspection.scalar(select(func.count()).select_from(AuditEvent)),
        )

    rejected_commands = (
        (
            _agent_command(
                deleting_knowledge_base_id,
                enabled_query_profile_id,
                name="deleting-kb",
            ),
            deleting_knowledge_base_id,
            "req-deleting-kb",
        ),
        (
            _agent_command(
                active_knowledge_base_id,
                disabled_query_profile_id,
                name="disabled-query-profile",
            ),
            disabled_query_profile_id,
            "req-disabled-query-profile",
        ),
    )
    for command, rejected_reference_id, request_id in rejected_commands:
        async with migrated_database.sessions() as invalid_session:
            service = _service(migrated_database, invalid_session)
            with pytest.raises(BusinessError) as raised:
                await service.create_agent_key(
                    command,
                    actor=admin,
                    request_id=request_id,
                )
            assert (
                raised.value.status_code,
                raised.value.code,
                raised.value.message,
                raised.value.args,
            ) == (
                422,
                "VALIDATION_ERROR",
                "Invalid API key policy",
                ("Invalid API key policy",),
            )
            assert str(rejected_reference_id) not in repr(raised.value)

        async with migrated_database.sessions() as inspection:
            assert (
                await inspection.scalar(select(func.count()).select_from(ApiKey)),
                await inspection.scalar(select(func.count()).select_from(ApiKeyKnowledgeBaseScope)),
                await inspection.scalar(select(func.count()).select_from(ApiKeyQueryProfileScope)),
                await inspection.scalar(select(func.count()).select_from(AuditEvent)),
            ) == baseline_counts


@pytest.mark.parametrize("reference_kind", ("knowledge_base", "query_profile"))
@pytest.mark.integration
@pytest.mark.asyncio
async def test_reference_validation_lock_serializes_concurrent_state_transition(
    migrated_database: Database,
    reference_kind: Literal["knowledge_base", "query_profile"],
) -> None:
    (
        knowledge_base_id,
        _disabled_knowledge_base_id,
        _deleting_knowledge_base_id,
        query_profile_id,
        _disabled_query_profile_id,
    ) = await _insert_references(migrated_database)
    async with migrated_database.sessions() as admin_session:
        _admin_token, admin = await _create_admin(migrated_database, admin_session)

    lock_acquired = asyncio.Event()
    release_validation = asyncio.Event()
    transition_started = asyncio.Event()
    transition_committed = asyncio.Event()

    def blocking_repositories(session: AsyncSession) -> AuthRepositories:
        repositories = sqlalchemy_auth_repositories(session)
        return AuthRepositories(
            api_keys=repositories.api_keys,
            scopes=_BlockingReferenceScopeRepository(
                repositories.scopes,
                reference_kind=reference_kind,
                lock_acquired=lock_acquired,
                release_validation=release_validation,
            ),
            audits=repositories.audits,
        )

    async def transition_reference_state() -> None:
        async with (
            migrated_database.sessions() as transition_session,
            transition_session.begin(),
        ):
            transition_started.set()
            if reference_kind == "knowledge_base":
                await transition_session.execute(
                    update(KnowledgeBase)
                    .where(KnowledgeBase.id == knowledge_base_id)
                    .values(status="deleting")
                )
            else:
                await transition_session.execute(
                    update(QueryProfile)
                    .where(QueryProfile.id == query_profile_id)
                    .values(enabled=False)
                )
        transition_committed.set()

    async with migrated_database.sessions() as creation_session:
        service = ApiKeyService(
            session=creation_session,
            authentication_sessions=migrated_database.sessions,
            settings=_settings(),
            repository_factory=blocking_repositories,
            clock=lambda: NOW,
        )
        creation_task = asyncio.create_task(
            service.create_agent_key(
                _agent_command(knowledge_base_id, query_profile_id),
                actor=admin,
                request_id=f"req-lock-{reference_kind}",
            )
        )
        await asyncio.wait_for(lock_acquired.wait(), timeout=2)
        transition_task = asyncio.create_task(transition_reference_state())
        try:
            await asyncio.wait_for(transition_started.wait(), timeout=2)
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(transition_committed.wait(), timeout=0.2)
        finally:
            release_validation.set()
        issued, _transition_result = await asyncio.gather(creation_task, transition_task)

    async with migrated_database.sessions() as inspection:
        if reference_kind == "knowledge_base":
            assert (
                await inspection.scalar(
                    select(KnowledgeBase.status).where(KnowledgeBase.id == knowledge_base_id)
                )
                == "deleting"
            )
            assert (
                await inspection.scalar(
                    select(func.count())
                    .select_from(ApiKeyKnowledgeBaseScope)
                    .where(ApiKeyKnowledgeBaseScope.api_key_id == issued.api_key.id)
                )
                == 1
            )
        else:
            assert (
                await inspection.scalar(
                    select(QueryProfile.enabled).where(QueryProfile.id == query_profile_id)
                )
                is False
            )
            assert (
                await inspection.scalar(
                    select(func.count())
                    .select_from(ApiKeyQueryProfileScope)
                    .where(ApiKeyQueryProfileScope.api_key_id == issued.api_key.id)
                )
                == 1
            )
        assert (
            await inspection.scalar(
                select(func.count())
                .select_from(AuditEvent)
                .where(
                    AuditEvent.request_id == f"req-lock-{reference_kind}",
                    AuditEvent.target_id == issued.api_key.id,
                )
            )
            == 1
        )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_audit_update_revoke_etag_and_terminal_state_are_atomic_and_sanitized(
    migrated_database: Database,
) -> None:
    (
        old_kb_id,
        new_kb_id,
        _deleting_kb_id,
        query_profile_id,
        _disabled_qp_id,
    ) = await _insert_references(migrated_database)
    async with migrated_database.sessions() as session:
        service = _service(migrated_database, session)
        _admin_token, admin = await _create_admin(migrated_database, session)
        issued = await service.create_agent_key(
            _agent_command(old_kb_id, query_profile_id),
            actor=admin,
            request_id="req-create",
        )
        updated = await service.update_agent_key(
            issued.api_key.id,
            AgentApiKeyUpdate(
                name="renamed-sensitive-body-value",
                knowledge_base_ids=frozenset({new_kb_id}),
            ),
            actor=admin,
            request_id="req-update",
            expected_etag=issued.api_key.etag or "",
        )
        assert updated.resource_revision == 2

        with pytest.raises(BusinessError) as stale:
            await service.update_agent_key(
                issued.api_key.id,
                AgentApiKeyUpdate(name="stale-body-value"),
                actor=admin,
                request_id="req-stale",
                expected_etag=issued.api_key.etag or "",
            )
        assert (stale.value.status_code, stale.value.code) == (412, "PRECONDITION_FAILED")

        revoked = await service.revoke_agent_key(
            issued.api_key.id,
            actor=admin,
            request_id="req-revoke",
            expected_etag=updated.etag,
        )
        repeated = await service.revoke_agent_key(
            issued.api_key.id,
            actor=admin,
            request_id="req-repeat",
            expected_etag=revoked.etag,
        )
        assert repeated == revoked

        with pytest.raises(BusinessError) as terminal:
            await service.update_agent_key(
                issued.api_key.id,
                AgentApiKeyUpdate(status="active"),
                actor=admin,
                request_id="req-terminal",
                expected_etag=revoked.etag or "",
            )
        assert (terminal.value.status_code, terminal.value.code) == (
            409,
            "RESOURCE_STATE_CONFLICT",
        )
        with pytest.raises(BusinessError) as missing_etag:
            await service.update_agent_key(
                issued.api_key.id,
                AgentApiKeyUpdate(status="active"),
                actor=admin,
                request_id="req-missing",
                expected_etag="",
            )
        assert (missing_etag.value.status_code, missing_etag.value.code) == (
            412,
            "PRECONDITION_FAILED",
        )

    async with migrated_database.sessions() as inspection:
        row = await inspection.get(ApiKey, issued.api_key.id)
        assert row is not None
        assert (row.status, row.resource_revision, row.name) == (
            "revoked",
            3,
            "renamed-sensitive-body-value",
        )
        events = list(
            (
                await inspection.scalars(
                    select(AuditEvent)
                    .where(AuditEvent.target_id == issued.api_key.id)
                    .order_by(AuditEvent.created_at, AuditEvent.id)
                )
            ).all()
        )
        assert [event.action for event in events] == [
            "api_key.created",
            "api_key.policy_updated",
            "api_key.revoked",
        ]
        update_metadata = events[1].metadata_
        assert update_metadata == {
            "changed_fields": ["knowledge_base_ids", "name"],
            "knowledge_base_ids_added": [str(new_kb_id)],
            "knowledge_base_ids_removed": [str(old_kb_id)],
            "query_profile_ids_added": [],
            "query_profile_ids_removed": [],
        }
        serialized_events = repr([event.metadata_ for event in events])
        assert "renamed-sensitive-body-value" not in serialized_events
        assert "stale-body-value" not in serialized_events
        assert "token" not in serialized_events.lower()
        assert "secret" not in serialized_events.lower()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_revoked_admin_principal_is_revalidated_inside_agent_mutation_transaction(
    migrated_database: Database,
) -> None:
    (
        knowledge_base_id,
        _second_kb_id,
        _deleting_kb_id,
        query_profile_id,
        _disabled_qp_id,
    ) = await _insert_references(migrated_database)
    async with migrated_database.sessions() as admin_session:
        _admin_token, admin = await _create_admin(migrated_database, admin_session)
        service = _service(migrated_database, admin_session)
        revoked = await service.revoke_admin_key(
            admin.key_id,
            request_id="req-admin-revoke",
        )
        assert revoked.status == "revoked"

    async with migrated_database.sessions() as mutation_session:
        service = _service(migrated_database, mutation_session)
        with pytest.raises(BusinessError) as raised:
            await service.create_agent_key(
                _agent_command(knowledge_base_id, query_profile_id),
                actor=admin,
                request_id="req-revoked-admin-create",
            )
        _assert_invalid_api_key(raised.value)

    async with migrated_database.sessions() as inspection:
        assert (
            await inspection.scalar(
                select(func.count()).select_from(ApiKey).where(ApiKey.key_type == "agent")
            )
            == 0
        )
        assert (
            await inspection.scalar(
                select(func.count())
                .select_from(AuditEvent)
                .where(AuditEvent.action == "api_key.created", AuditEvent.target_id != admin.key_id)
            )
            == 0
        )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_database_write_failure_rolls_back_key_scopes_and_audit(
    migrated_database: Database,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        knowledge_base_id,
        _second_kb_id,
        _deleting_kb_id,
        query_profile_id,
        _disabled_qp_id,
    ) = await _insert_references(migrated_database)
    async with migrated_database.sessions() as session:
        service = _service(migrated_database, session)
        _admin_token, admin = await _create_admin(migrated_database, session)
        first = await service.create_agent_key(
            _agent_command(knowledge_base_id, query_profile_id, name="first"),
            actor=admin,
            request_id="req-first",
        )
        raw_token = first.token.get_secret_value()
        parsed = parse_token(raw_token, KeyKind.AGENT)
        duplicate = GeneratedToken(
            token=raw_token,
            public_id=parsed.public_id,
            digest=digest_secret(parsed.secret, _settings().agent_key_hmac_secret),
        )
        monkeypatch.setattr(auth_services, "generate_token", lambda _kind, _secret: duplicate)

        with pytest.raises(BusinessError) as raised:
            await service.create_agent_key(
                _agent_command(knowledge_base_id, query_profile_id, name="second"),
                actor=admin,
                request_id="req-duplicate",
            )
        assert (
            raised.value.status_code,
            raised.value.code,
            raised.value.message,
            raised.value.args,
        ) == (500, "INTERNAL_ERROR", "Internal server error", ("Internal server error",))

    async with migrated_database.sessions() as inspection:
        assert (
            await inspection.scalar(
                select(func.count()).select_from(ApiKey).where(ApiKey.key_type == "agent")
            )
            == 1
        )
        assert (
            await inspection.scalar(select(func.count()).select_from(ApiKeyKnowledgeBaseScope)) == 1
        )
        assert (
            await inspection.scalar(select(func.count()).select_from(ApiKeyQueryProfileScope)) == 1
        )
        assert (
            await inspection.scalar(
                select(func.count())
                .select_from(AuditEvent)
                .where(AuditEvent.target_id == first.api_key.id)
            )
            == 1
        )
        assert (
            await inspection.scalar(
                select(func.count())
                .select_from(AuditEvent)
                .where(AuditEvent.request_id == "req-duplicate")
            )
            == 0
        )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_admin_safe_list_and_idempotent_revoke_never_expose_token_or_digest(
    migrated_database: Database,
) -> None:
    async with migrated_database.sessions() as session:
        service = _service(migrated_database, session)
        first = await service.create_admin_key(
            AdminApiKeyCreate(name="first-admin"),
            request_id="req-first-admin",
        )
        second = await service.create_admin_key(
            AdminApiKeyCreate(name="second-admin"),
            request_id="req-second-admin",
        )
        page = await service.list_admin_keys(limit=10)
        assert {item.id for item in page.items} == {first.api_key.id, second.api_key.id}
        serialized = repr(page.model_dump())
        assert first.token.get_secret_value() not in serialized
        assert second.token.get_secret_value() not in serialized
        assert "secret" not in serialized.lower()
        assert "digest" not in serialized.lower()

    async with migrated_database.sessions() as revoke_session:
        service = _service(migrated_database, revoke_session)
        first_revoke = await service.revoke_admin_key(
            first.api_key.id,
            request_id="req-first-revoke",
        )
        second_revoke = await service.revoke_admin_key(
            first.api_key.id,
            request_id="req-first-repeat",
        )
        assert first_revoke == second_revoke
        with pytest.raises(BusinessError) as raised:
            await service.authenticate(first.token.get_secret_value(), KeyKind.ADMIN)
        _assert_invalid_api_key(raised.value)

    async with migrated_database.sessions() as inspection:
        assert (
            await inspection.scalar(
                select(func.count())
                .select_from(AuditEvent)
                .where(
                    AuditEvent.target_id == first.api_key.id,
                    AuditEvent.action == "api_key.revoked",
                )
            )
            == 1
        )
