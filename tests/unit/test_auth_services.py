import asyncio
from contextlib import AbstractAsyncContextManager
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from types import TracebackType
from typing import Any, Never, cast
from uuid import UUID, uuid4

import pytest
from pydantic import SecretStr
from sqlalchemy.ext.asyncio import AsyncSession

from rag_service.api.errors import BusinessError
from rag_service.api.etags import agent_key_etag
from rag_service.auth import services as auth_services
from rag_service.auth.codec import GeneratedToken, KeyKind, ParsedToken, generate_token
from rag_service.auth.policies import AdminPrincipal, AgentPrincipal, Capability
from rag_service.auth.repositories import (
    AuthRepositories,
    SqlAlchemyApiKeyScopeRepository,
)
from rag_service.auth.schemas import (
    AdminApiKeyCreate,
    AgentApiKeyCreate,
    AgentApiKeyUpdate,
)
from rag_service.auth.services import ApiKeyService
from rag_service.config import Settings
from rag_service.db.models.auth import ApiKey, AuditEvent

NOW = datetime(2026, 7, 25, 8, 0, tzinfo=UTC)


class FakeTransaction:
    def __init__(self, session: "FakeSession") -> None:
        self._session = session
        self._snapshot: object | None = None

    async def __aenter__(self) -> None:
        self._session.begin_count += 1
        self._snapshot = self._session.snapshot()

    async def __aexit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if exception_type is not None:
            self._session.restore(self._snapshot)


class FakeSession:
    def __init__(self) -> None:
        self.begin_count = 0
        self.keys: dict[UUID, ApiKey] = {}
        self.kb_scopes: dict[UUID, frozenset[UUID]] = {}
        self.qp_scopes: dict[UUID, tuple[frozenset[UUID], UUID | None]] = {}
        self.kb_statuses: dict[UUID, str] = {}
        self.qp_enabled: dict[UUID, bool] = {}
        self.audit_events: list[AuditEvent] = []
        self.operation_log: list[str] = []
        self.fail_audit = False
        self.locked_admin_reads = 0
        self.audit_exception: BaseException | None = None
        self.lookup_exception: BaseException | None = None
        self.id_exception: Exception | None = None
        self.list_exception: Exception | None = None

    def begin(self) -> FakeTransaction:
        return FakeTransaction(self)

    def snapshot(self) -> object:
        return deepcopy(
            (
                self.keys,
                self.kb_scopes,
                self.qp_scopes,
                self.audit_events,
                self.operation_log,
            )
        )

    def restore(self, snapshot: object | None) -> None:
        assert snapshot is not None
        (
            self.keys,
            self.kb_scopes,
            self.qp_scopes,
            self.audit_events,
            self.operation_log,
        ) = cast(
            tuple[
                dict[UUID, ApiKey],
                dict[UUID, frozenset[UUID]],
                dict[UUID, tuple[frozenset[UUID], UUID | None]],
                list[AuditEvent],
                list[str],
            ],
            snapshot,
        )


class _FailOnSqlSession:
    async def execute(self, *_args: object, **_kwargs: object) -> Never:
        raise AssertionError("empty identifiers must not execute SQL")


class FakeSessionContext:
    def __init__(self, session: FakeSession) -> None:
        self._session = session

    async def __aenter__(self) -> FakeSession:
        return self._session

    async def __aexit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        return None


class FakeApiKeyRepository:
    def __init__(self, session: FakeSession) -> None:
        self._session = session

    async def get_by_public_id(
        self,
        kind: KeyKind,
        public_id: str,
        *,
        for_update: bool = False,
    ) -> ApiKey | None:
        self._session.operation_log.append(f"get_public:{kind}:{for_update}")
        if self._session.lookup_exception is not None:
            raise self._session.lookup_exception
        return next(
            (
                key
                for key in self._session.keys.values()
                if key.key_type == kind.value and key.public_id == public_id
            ),
            None,
        )

    async def get_by_id(
        self,
        key_id: UUID,
        kind: KeyKind,
        *,
        for_update: bool = False,
    ) -> ApiKey | None:
        self._session.operation_log.append(f"get_id:{kind}:{for_update}")
        if self._session.id_exception is not None:
            raise self._session.id_exception
        if kind is KeyKind.ADMIN and for_update:
            self._session.locked_admin_reads += 1
        key = self._session.keys.get(key_id)
        return key if key is not None and key.key_type == kind.value else None

    async def add(self, key: ApiKey) -> None:
        self._session.operation_log.append("add_key")
        if key.id is None:
            key.id = uuid4()
        key.created_at = NOW
        key.updated_at = NOW
        self._session.keys[key.id] = key

    async def replace_kb_scopes(self, key_id: UUID, ids: frozenset[UUID]) -> None:
        self._session.operation_log.append("replace_kb_scopes")
        self._session.kb_scopes[key_id] = ids

    async def replace_query_profile_scopes(
        self,
        key_id: UUID,
        ids: frozenset[UUID],
        default_id: UUID | None,
    ) -> None:
        self._session.operation_log.append("replace_qp_scopes")
        self._session.qp_scopes[key_id] = (ids, default_id)

    async def list_by_kind(
        self,
        kind: KeyKind,
        position: object | None,
        limit: int,
    ) -> list[ApiKey]:
        del position
        self._session.operation_log.append("list_keys")
        if self._session.list_exception is not None:
            raise self._session.list_exception
        return [key for key in self._session.keys.values() if key.key_type == kind.value][:limit]


class FakeScopeRepository:
    def __init__(self, session: FakeSession) -> None:
        self._session = session

    async def get_knowledge_base_statuses(
        self,
        ids: frozenset[UUID],
        *,
        for_update: bool = False,
    ) -> dict[UUID, str]:
        self._session.operation_log.append(f"validate_kb:{for_update}")
        return {
            identifier: self._session.kb_statuses[identifier]
            for identifier in ids
            if identifier in self._session.kb_statuses
        }

    async def get_query_profile_enabled(
        self,
        ids: frozenset[UUID],
        *,
        for_update: bool = False,
    ) -> dict[UUID, bool]:
        self._session.operation_log.append(f"validate_qp:{for_update}")
        return {
            identifier: self._session.qp_enabled[identifier]
            for identifier in ids
            if identifier in self._session.qp_enabled
        }

    async def get_knowledge_base_scopes(self, key_id: UUID) -> frozenset[UUID]:
        self._session.operation_log.append("get_kb_scope")
        return self._session.kb_scopes.get(key_id, frozenset())

    async def get_query_profile_scopes(
        self,
        key_id: UUID,
    ) -> tuple[frozenset[UUID], UUID | None]:
        self._session.operation_log.append("get_qp_scope")
        return self._session.qp_scopes.get(key_id, (frozenset(), None))

    async def get_knowledge_base_scopes_batch(
        self,
        key_ids: frozenset[UUID],
    ) -> dict[UUID, frozenset[UUID]]:
        self._session.operation_log.append(f"batch_kb:{len(key_ids)}")
        return {key_id: self._session.kb_scopes.get(key_id, frozenset()) for key_id in key_ids}

    async def get_query_profile_scopes_batch(
        self,
        key_ids: frozenset[UUID],
    ) -> dict[UUID, tuple[frozenset[UUID], UUID | None]]:
        self._session.operation_log.append(f"batch_qp:{len(key_ids)}")
        return {
            key_id: self._session.qp_scopes.get(key_id, (frozenset(), None)) for key_id in key_ids
        }


class FakeAuditRepository:
    def __init__(self, session: FakeSession) -> None:
        self._session = session

    async def add(self, event: AuditEvent) -> None:
        self._session.operation_log.append("add_audit")
        self._session.audit_events.append(event)
        if self._session.fail_audit:
            raise RuntimeError("unsafe request body and token material")
        if self._session.audit_exception is not None:
            raise self._session.audit_exception


def _settings(*, max_request_id_length: int = 128) -> Settings:
    return Settings(
        environment="test",
        admin_key_hmac_secret=SecretStr("a" * 32),
        agent_key_hmac_secret=SecretStr("b" * 32),
        max_api_key_requests_per_minute=100,
        max_api_key_concurrency=10,
        max_request_id_length=max_request_id_length,
    )


def _service(
    session: FakeSession,
    *,
    settings: Settings | None = None,
) -> ApiKeyService:
    def repositories(_session: object) -> AuthRepositories:
        assert _session is session
        return AuthRepositories(
            api_keys=FakeApiKeyRepository(session),
            scopes=FakeScopeRepository(session),
            audits=FakeAuditRepository(session),
        )

    def auth_sessions() -> AbstractAsyncContextManager[Any]:
        return FakeSessionContext(session)

    return ApiKeyService(
        session=cast(Any, session),
        authentication_sessions=auth_sessions,
        settings=settings or _settings(),
        repository_factory=cast(Any, repositories),
        clock=lambda: NOW,
    )


def _admin(session: FakeSession) -> AdminPrincipal:
    generated = generate_token(KeyKind.ADMIN, _settings().admin_key_hmac_secret)
    key = ApiKey(
        id=uuid4(),
        public_id=generated.public_id,
        secret_digest=generated.digest,
        key_type="admin",
        name="administrator",
        status="active",
        capabilities=[],
        raw_file_read=False,
        requests_per_minute=None,
        max_concurrency=None,
        resource_revision=1,
        created_at=NOW,
        updated_at=NOW,
    )
    session.keys[key.id] = key
    return AdminPrincipal(key_id=key.id, public_id=key.public_id)


def _agent_command(kb_id: UUID, qp_id: UUID) -> AgentApiKeyCreate:
    return AgentApiKeyCreate(
        name="retrieval-agent",
        capabilities=frozenset({Capability.RETRIEVE}),
        knowledge_base_ids=frozenset({kb_id}),
        query_profile_ids=frozenset({qp_id}),
        default_query_profile_id=qp_id,
        raw_file_read=False,
        requests_per_minute=60,
        max_concurrency=4,
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
@pytest.mark.asyncio
async def test_direct_service_rejects_unsafe_request_id_before_transaction_or_audit(
    request_id: str,
) -> None:
    session = FakeSession()

    with pytest.raises(BusinessError) as raised:
        await _service(
            session,
            settings=_settings(max_request_id_length=32),
        ).create_admin_key(
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
    assert session.begin_count == 0
    assert session.keys == {}
    assert session.audit_events == []


@pytest.mark.asyncio
async def test_create_agent_validates_references_before_writes_and_audits_once() -> None:
    session = FakeSession()
    admin = _admin(session)
    kb_id, qp_id = uuid4(), uuid4()
    session.kb_statuses[kb_id] = "active"
    session.qp_enabled[qp_id] = True

    issued = await _service(session).create_agent_key(
        _agent_command(kb_id, qp_id),
        actor=admin,
        request_id="req-create",
    )

    assert session.begin_count == 1
    assert session.operation_log.index("validate_kb:True") < session.operation_log.index("add_key")
    assert session.operation_log.index("validate_qp:True") < session.operation_log.index("add_key")
    assert issued.token.get_secret_value().startswith("rag_agent_")
    assert issued.token.get_secret_value() not in repr(issued)
    assert len(session.audit_events) == 1
    event = session.audit_events[0]
    assert (event.action, event.actor_kind, event.actor_api_key_id) == (
        "api_key.created",
        "admin_key",
        admin.key_id,
    )
    assert event.metadata_ == {}
    assert "token" not in repr(event.metadata_).lower()


@pytest.mark.asyncio
async def test_create_agent_invalid_reference_rolls_back_without_write_or_audit() -> None:
    session = FakeSession()
    admin = _admin(session)
    existing_ids = frozenset(session.keys)

    with pytest.raises(BusinessError) as raised:
        await _service(session).create_agent_key(
            _agent_command(uuid4(), uuid4()),
            actor=admin,
            request_id="req-invalid",
        )

    assert (raised.value.status_code, raised.value.code) == (422, "VALIDATION_ERROR")
    assert frozenset(session.keys) == existing_ids
    assert session.audit_events == []
    assert session.begin_count == 1


@pytest.mark.asyncio
async def test_agent_patch_replaces_scopes_increments_once_and_audits_only_safe_deltas() -> None:
    session = FakeSession()
    admin = _admin(session)
    old_kb, new_kb, old_qp, new_qp = uuid4(), uuid4(), uuid4(), uuid4()
    session.kb_statuses.update({old_kb: "active", new_kb: "disabled"})
    session.qp_enabled.update({old_qp: True, new_qp: True})
    issued = await _service(session).create_agent_key(
        _agent_command(old_kb, old_qp),
        actor=admin,
        request_id="req-create",
    )
    key_id = issued.api_key.id
    issued_etag = issued.api_key.etag
    assert issued_etag is not None
    session.audit_events.clear()
    session.operation_log.clear()

    updated = await _service(session).update_agent_key(
        key_id,
        AgentApiKeyUpdate(
            name="renamed",
            knowledge_base_ids=frozenset({new_kb}),
            query_profile_ids=frozenset({new_qp}),
            default_query_profile_id=new_qp,
        ),
        actor=admin,
        request_id="req-update",
        expected_etag=issued_etag,
    )

    assert updated.resource_revision == 2
    assert session.keys[key_id].resource_revision == 2
    assert "validate_kb:True" in session.operation_log
    assert "validate_qp:True" in session.operation_log
    assert len(session.audit_events) == 1
    event = session.audit_events[0]
    assert event.action == "api_key.policy_updated"
    assert event.metadata_ == {
        "changed_fields": [
            "default_query_profile_id",
            "knowledge_base_ids",
            "name",
            "query_profile_ids",
        ],
        "knowledge_base_ids_added": [str(new_kb)],
        "knowledge_base_ids_removed": [str(old_kb)],
        "query_profile_ids_added": [str(new_qp)],
        "query_profile_ids_removed": [str(old_qp)],
    }
    serialized = repr(event.metadata_)
    assert "renamed" not in serialized
    assert "token" not in serialized.lower()
    assert "secret" not in serialized.lower()


@pytest.mark.asyncio
async def test_stale_etag_precedes_revoked_state_and_neither_failure_audits() -> None:
    session = FakeSession()
    admin = _admin(session)
    kb_id, qp_id = uuid4(), uuid4()
    session.kb_statuses[kb_id] = "active"
    session.qp_enabled[qp_id] = True
    issued = await _service(session).create_agent_key(
        _agent_command(kb_id, qp_id),
        actor=admin,
        request_id="req-create",
    )
    key_id = issued.api_key.id
    issued_etag = issued.api_key.etag
    assert issued_etag is not None
    await _service(session).revoke_agent_key(
        key_id,
        actor=admin,
        request_id="req-revoke",
        expected_etag=issued_etag,
    )
    session.audit_events.clear()
    current_etag = agent_key_etag(key_id, 2)

    with pytest.raises(BusinessError) as stale:
        await _service(session).update_agent_key(
            key_id,
            AgentApiKeyUpdate(name="blocked"),
            actor=admin,
            request_id="req-stale",
            expected_etag=issued_etag,
        )
    assert (stale.value.status_code, stale.value.code) == (412, "PRECONDITION_FAILED")

    with pytest.raises(BusinessError) as current:
        await _service(session).update_agent_key(
            key_id,
            AgentApiKeyUpdate(name="blocked"),
            actor=admin,
            request_id="req-current",
            expected_etag=current_etag,
        )
    assert (current.value.status_code, current.value.code) == (
        409,
        "RESOURCE_STATE_CONFLICT",
    )
    assert session.keys[key_id].name == "retrieval-agent"
    assert session.keys[key_id].resource_revision == 2
    assert session.audit_events == []


@pytest.mark.asyncio
async def test_revoke_is_idempotent_and_preserves_first_revocation_metadata() -> None:
    session = FakeSession()
    admin = _admin(session)
    kb_id, qp_id = uuid4(), uuid4()
    session.kb_statuses[kb_id] = "active"
    session.qp_enabled[qp_id] = True
    issued = await _service(session).create_agent_key(
        _agent_command(kb_id, qp_id),
        actor=admin,
        request_id="req-create",
    )
    session.audit_events.clear()

    first = await _service(session).revoke_agent_key(
        issued.api_key.id,
        actor=admin,
        request_id="req-first",
        expected_etag=issued.api_key.etag,
    )
    first_timestamp = session.keys[issued.api_key.id].revoked_at
    second = await _service(session).revoke_agent_key(
        issued.api_key.id,
        actor=admin,
        request_id="req-repeat",
        expected_etag=first.etag,
    )

    assert first.resource_revision == second.resource_revision == 2
    assert session.keys[issued.api_key.id].revoked_at == first_timestamp
    assert session.keys[issued.api_key.id].revoked_by_api_key_id == admin.key_id
    assert [event.action for event in session.audit_events] == ["api_key.revoked"]


@pytest.mark.asyncio
async def test_authentication_materializes_current_agent_scope_and_normalizes_failures() -> None:
    session = FakeSession()
    kb_id, qp_id = uuid4(), uuid4()
    generated = generate_token(KeyKind.AGENT, _settings().agent_key_hmac_secret)
    key = ApiKey(
        id=uuid4(),
        public_id=generated.public_id,
        secret_digest=generated.digest,
        key_type="agent",
        name="agent",
        status="active",
        capabilities=["retrieve"],
        raw_file_read=True,
        requests_per_minute=10,
        max_concurrency=2,
        not_before=NOW - timedelta(minutes=1),
        expires_at=NOW + timedelta(minutes=1),
        resource_revision=1,
        created_at=NOW,
        updated_at=NOW,
    )
    session.keys[key.id] = key
    session.kb_scopes[key.id] = frozenset({kb_id})
    session.qp_scopes[key.id] = (frozenset({qp_id}), qp_id)

    principal = await _service(session).authenticate(generated.token, KeyKind.AGENT)

    assert principal == AgentPrincipal(
        key_id=key.id,
        public_id=key.public_id,
        capabilities=frozenset({Capability.RETRIEVE}),
        knowledge_base_ids=frozenset({kb_id}),
        query_profile_ids=frozenset({qp_id}),
        default_query_profile_id=qp_id,
        raw_file_read=True,
        requests_per_minute=10,
        max_concurrency=2,
    )

    invalid_tokens = [
        "malformed",
        generate_token(KeyKind.ADMIN, _settings().admin_key_hmac_secret).token,
        generated.token[:-1] + ("A" if generated.token[-1] != "A" else "B"),
    ]
    for token in invalid_tokens:
        with pytest.raises(BusinessError) as raised:
            await _service(session).authenticate(token, KeyKind.AGENT)
        assert (
            raised.value.status_code,
            raised.value.code,
            raised.value.message,
        ) == (401, "INVALID_API_KEY", "Invalid API key")


@pytest.mark.asyncio
async def test_agent_list_batch_loads_visible_page_scopes_with_constant_repository_calls() -> None:
    session = FakeSession()
    expected_ids: list[UUID] = []
    raw_tokens: list[str] = []
    for index in range(4):
        knowledge_base_id = uuid4()
        query_profile_id = uuid4()
        generated = generate_token(KeyKind.AGENT, _settings().agent_key_hmac_secret)
        raw_tokens.append(generated.token)
        row = ApiKey(
            id=uuid4(),
            public_id=generated.public_id,
            secret_digest=generated.digest,
            key_type="agent",
            name=f"agent-{index}",
            status="active",
            capabilities=["retrieve"],
            raw_file_read=False,
            requests_per_minute=10,
            max_concurrency=2,
            resource_revision=1,
            created_at=NOW + timedelta(microseconds=index),
            updated_at=NOW,
        )
        session.keys[row.id] = row
        session.kb_scopes[row.id] = frozenset({knowledge_base_id})
        session.qp_scopes[row.id] = (frozenset({query_profile_id}), query_profile_id)
        expected_ids.append(row.id)

    page = await _service(session).list_agent_keys(limit=3)

    assert [item.id for item in page.items] == expected_ids[:3]
    assert page.next_cursor is not None
    assert session.operation_log == ["list_keys", "batch_kb:3", "batch_qp:3"]
    serialized = repr(page.model_dump())
    assert all(raw_token not in serialized for raw_token in raw_tokens)
    assert "secret" not in serialized.lower()
    assert "digest" not in serialized.lower()


@pytest.mark.asyncio
async def test_scope_repository_skips_sql_for_empty_reference_and_batch_identifiers() -> None:
    repository = SqlAlchemyApiKeyScopeRepository(cast(AsyncSession, _FailOnSqlSession()))

    assert (
        await repository.get_knowledge_base_statuses(
            frozenset(),
            for_update=True,
        )
        == {}
    )
    assert (
        await repository.get_query_profile_enabled(
            frozenset(),
            for_update=True,
        )
        == {}
    )
    assert await repository.get_knowledge_base_scopes_batch(frozenset()) == {}
    assert await repository.get_query_profile_scopes_batch(frozenset()) == {}


@pytest.mark.asyncio
async def test_unknown_well_formed_key_still_uses_constant_time_digest_verification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = FakeSession()
    unknown = generate_token(KeyKind.AGENT, _settings().agent_key_hmac_secret)
    calls: list[bytes] = []

    def spy_verify(secret: str, expected_digest: bytes, hmac_secret: SecretStr) -> bool:
        del secret, hmac_secret
        calls.append(expected_digest)
        return False

    monkeypatch.setattr(auth_services, "verify_secret", spy_verify)

    with pytest.raises(BusinessError) as raised:
        await _service(session).authenticate(unknown.token, KeyKind.AGENT)

    assert raised.value.code == "INVALID_API_KEY"
    assert calls == [b"\x00" * 32]


@pytest.mark.asyncio
async def test_mutation_revalidates_and_locks_admin_actor_before_any_write() -> None:
    session = FakeSession()
    admin = _admin(session)
    session.keys[admin.key_id].status = "disabled"
    kb_id, qp_id = uuid4(), uuid4()
    session.kb_statuses[kb_id] = "active"
    session.qp_enabled[qp_id] = True

    with pytest.raises(BusinessError) as raised:
        await _service(session).create_agent_key(
            _agent_command(kb_id, qp_id),
            actor=admin,
            request_id="req-disabled-actor",
        )

    assert (raised.value.status_code, raised.value.code) == (401, "INVALID_API_KEY")
    assert session.begin_count == 1
    assert session.locked_admin_reads == 1
    assert session.operation_log == []
    assert len(session.keys) == 1
    assert session.audit_events == []


@pytest.mark.asyncio
async def test_failed_audit_write_rolls_back_key_scopes_and_surfaces_sanitized_error() -> None:
    session = FakeSession()
    admin = _admin(session)
    kb_id, qp_id = uuid4(), uuid4()
    session.kb_statuses[kb_id] = "active"
    session.qp_enabled[qp_id] = True
    session.fail_audit = True

    with pytest.raises(BusinessError) as raised:
        await _service(session).create_agent_key(
            _agent_command(kb_id, qp_id),
            actor=admin,
            request_id="req-failed-audit",
        )

    assert (
        raised.value.status_code,
        raised.value.code,
        raised.value.message,
        raised.value.args,
    ) == (500, "INTERNAL_ERROR", "Internal server error", ("Internal server error",))
    assert session.begin_count == 1
    assert len(session.keys) == 1
    assert session.kb_scopes == {}
    assert session.qp_scopes == {}
    assert session.audit_events == []


@pytest.mark.asyncio
async def test_each_successful_patch_owns_one_transaction_revision_and_audit_event() -> None:
    session = FakeSession()
    admin = _admin(session)
    kb_id, qp_id = uuid4(), uuid4()
    session.kb_statuses[kb_id] = "active"
    session.qp_enabled[qp_id] = True
    issued = await _service(session).create_agent_key(
        _agent_command(kb_id, qp_id),
        actor=admin,
        request_id="req-create",
    )
    session.audit_events.clear()
    baseline_transactions = session.begin_count
    issued_etag = issued.api_key.etag
    assert issued_etag is not None

    first = await _service(session).update_agent_key(
        issued.api_key.id,
        AgentApiKeyUpdate(raw_file_read=True),
        actor=admin,
        request_id="req-patch-one",
        expected_etag=issued_etag,
    )
    first_etag = first.etag
    assert first_etag is not None
    second = await _service(session).update_agent_key(
        issued.api_key.id,
        AgentApiKeyUpdate(capabilities=frozenset()),
        actor=admin,
        request_id="req-patch-two",
        expected_etag=first_etag,
    )

    assert session.begin_count - baseline_transactions == 2
    assert (first.resource_revision, second.resource_revision) == (2, 3)
    assert [event.action for event in session.audit_events] == [
        "api_key.policy_updated",
        "api_key.policy_updated",
    ]


@pytest.mark.asyncio
async def test_local_admin_create_and_repeated_revoke_preserve_first_metadata() -> None:
    session = FakeSession()
    service = _service(session)
    issued = await service.create_admin_key(
        AdminApiKeyCreate(name="local-admin"),
        request_id="req-admin-create",
    )
    session.audit_events.clear()

    first = await service.revoke_admin_key(
        issued.api_key.id,
        request_id="req-admin-revoke",
    )
    first_timestamp = session.keys[issued.api_key.id].revoked_at
    second = await service.revoke_admin_key(
        issued.api_key.id,
        request_id="req-admin-repeat",
    )

    assert (first.resource_revision, second.resource_revision) == (2, 2)
    assert session.keys[issued.api_key.id].revoked_at == first_timestamp
    assert session.keys[issued.api_key.id].revoked_by_api_key_id is None
    assert [(event.action, event.actor_kind) for event in session.audit_events] == [
        ("api_key.revoked", "local_cli")
    ]


@pytest.mark.asyncio
async def test_command_schemas_and_service_reject_empty_null_unknown_and_over_limit_policy() -> (
    None
):
    with pytest.raises(ValueError):
        AgentApiKeyUpdate()
    with pytest.raises(ValueError):
        AgentApiKeyUpdate(name=None)

    bypassed = AgentApiKeyCreate.model_construct(
        name="agent",
        capabilities=cast(frozenset[Capability], frozenset({"unknown"})),
        knowledge_base_ids=frozenset(),
        query_profile_ids=frozenset(),
        default_query_profile_id=None,
        raw_file_read=False,
        requests_per_minute=101,
        max_concurrency=11,
        not_before=None,
        expires_at=None,
    )
    session = FakeSession()
    with pytest.raises(BusinessError) as raised:
        await _service(session).create_agent_key(
            bypassed,
            actor=None,
            request_id="req-bypassed",
        )
    assert (raised.value.status_code, raised.value.code) == (422, "VALIDATION_ERROR")
    assert session.begin_count == 0


@pytest.mark.asyncio
async def test_creation_cancellation_rolls_back_and_redacts_generated_plaintext_from_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = FakeSession()
    admin = _admin(session)
    kb_id, qp_id = uuid4(), uuid4()
    session.kb_statuses[kb_id] = "active"
    session.qp_enabled[qp_id] = True
    generated = generate_token(KeyKind.AGENT, _settings().agent_key_hmac_secret)
    monkeypatch.setattr(auth_services, "generate_token", lambda _kind, _secret: generated)
    session.audit_exception = asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError) as raised:
        await _service(session).create_agent_key(
            _agent_command(kb_id, qp_id),
            actor=admin,
            request_id="req-cancelled",
        )

    retained_generated_tokens: list[str] = []
    traceback = raised.value.__traceback__
    while traceback is not None:
        if traceback.tb_frame.f_code.co_filename.endswith("/rag_service/auth/services.py"):
            retained_generated = traceback.tb_frame.f_locals.get("generated")
            if isinstance(retained_generated, GeneratedToken):
                retained_generated_tokens.append(retained_generated.token)
        traceback = traceback.tb_next
    assert generated.token not in retained_generated_tokens
    assert len(session.keys) == 1
    assert session.audit_events == []


@pytest.mark.asyncio
async def test_authentication_cancellation_redacts_raw_token_and_secret_from_frames() -> None:
    session = FakeSession()
    generated = generate_token(KeyKind.AGENT, _settings().agent_key_hmac_secret)
    secret = generated.token.rsplit(".", 1)[1]
    session.lookup_exception = asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError) as raised:
        await _service(session).authenticate(generated.token, KeyKind.AGENT)

    retained_raw_tokens: list[str] = []
    retained_secrets: list[str] = []
    traceback = raised.value.__traceback__
    while traceback is not None:
        if not traceback.tb_frame.f_code.co_filename.endswith("/rag_service/auth/services.py"):
            traceback = traceback.tb_next
            continue
        frame_locals = traceback.tb_frame.f_locals
        raw_token = frame_locals.get("raw_token")
        if isinstance(raw_token, str):
            retained_raw_tokens.append(raw_token)
        retained_secret = frame_locals.get("secret")
        if isinstance(retained_secret, str):
            retained_secrets.append(retained_secret)
        parsed = frame_locals.get("parsed")
        if isinstance(parsed, ParsedToken):
            retained_secrets.append(parsed.secret)
        traceback = traceback.tb_next
    assert generated.token not in retained_raw_tokens
    assert secret not in retained_secrets


@pytest.mark.asyncio
async def test_read_failures_surface_only_sanitized_internal_errors() -> None:
    session = FakeSession()
    session.id_exception = RuntimeError("unsafe SQL and request body")
    service = _service(session)

    with pytest.raises(BusinessError) as detail:
        await service.get_agent_key(uuid4())
    assert (
        detail.value.status_code,
        detail.value.code,
        detail.value.message,
        detail.value.args,
    ) == (500, "INTERNAL_ERROR", "Internal server error", ("Internal server error",))

    session.id_exception = None
    session.list_exception = RuntimeError("unsafe SQL and request body")
    with pytest.raises(BusinessError) as listing:
        await service.list_agent_keys()
    assert (
        listing.value.status_code,
        listing.value.code,
        listing.value.message,
        listing.value.args,
    ) == (500, "INTERNAL_ERROR", "Internal server error", ("Internal server error",))
