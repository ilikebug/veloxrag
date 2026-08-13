import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import cast
from uuid import UUID

import pytest
from fastapi import Request
from pydantic import SecretStr
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.types import Scope

from rag_service.admin import routes as admin_routes
from rag_service.api.errors import BusinessError
from rag_service.auth.policies import AdminPrincipal, Capability
from rag_service.auth.schemas import (
    AgentApiKeyCreate,
    AgentApiKeyUpdate,
    IssuedApiKey,
    Page,
    SafeApiKey,
)
from rag_service.auth.services import ApiKeyService
from rag_service.config import Settings
from rag_service.db.session import Database

NOW = datetime(2026, 7, 26, 10, 0, tzinfo=UTC)
KEY_ID = UUID("30000000-0000-0000-0000-000000000003")
ACTOR = AdminPrincipal(
    key_id=UUID("10000000-0000-0000-0000-000000000001"),
    public_id="YWRtaW4tcHVibGljLWlk",
)
COMMAND = AgentApiKeyCreate(
    name="retrieval-agent",
    capabilities=frozenset({Capability.RETRIEVE}),
    requests_per_minute=60,
    max_concurrency=4,
)
UPDATE = AgentApiKeyUpdate(name="renamed-agent")


def _settings() -> Settings:
    return Settings(
        _env_file=None,
        environment="test",
        admin_key_hmac_secret=SecretStr("a" * 32),
        agent_key_hmac_secret=SecretStr("b" * 32),
    )


def _safe(*, etag: str | None = '"agent-key-1"') -> SafeApiKey:
    return SafeApiKey(
        id=KEY_ID,
        public_id="YWdlbnQtcHVibGljLWlk",
        name="retrieval-agent",
        status="active",
        key_type="agent",
        capabilities=(Capability.RETRIEVE,),
        knowledge_base_ids=(),
        query_profile_ids=(),
        default_query_profile_id=None,
        raw_file_read=False,
        requests_per_minute=60,
        max_concurrency=4,
        not_before=None,
        expires_at=None,
        resource_revision=1,
        etag=etag,
        created_at=NOW,
        updated_at=NOW,
        revoked_at=None,
    )


def _request(*, idempotency_key: str | None = None) -> Request:
    headers: list[tuple[bytes, bytes]] = []
    if idempotency_key is not None:
        headers.append((b"idempotency-key", idempotency_key.encode()))
    scope = cast(
        Scope,
        {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/v1/admin/api-keys",
            "raw_path": b"/v1/admin/api-keys",
            "query_string": b"",
            "headers": headers,
            "client": ("127.0.0.1", 1234),
            "server": ("test", 80),
            "state": {},
        },
    )
    return Request(scope)


def _document(response_body: bytes | memoryview) -> dict[str, object]:
    parsed = json.loads(bytes(response_body))
    if not isinstance(parsed, dict):
        raise AssertionError("response body must be a JSON object")
    return cast(dict[str, object], parsed)


class _ServiceDouble:
    def __init__(self, safe: SafeApiKey | None = None) -> None:
        self.safe = safe or _safe()
        self.issued = IssuedApiKey(
            api_key=self.safe,
            token=SecretStr("rag_agt_ONE-TIME-SECRET"),
        )
        self.page = Page[SafeApiKey](items=(self.safe,), next_cursor="next-page")
        self.calls: list[tuple[object, ...]] = []

    async def create_agent_key(
        self,
        command: AgentApiKeyCreate,
        *,
        actor: AdminPrincipal,
        request_id: str,
    ) -> IssuedApiKey:
        self.calls.append(("create", command, actor, request_id))
        return self.issued

    async def list_agent_keys(
        self,
        *,
        cursor: str | None,
        limit: int | None,
    ) -> Page[SafeApiKey]:
        self.calls.append(("list", cursor, limit))
        return self.page

    async def get_agent_key(self, key_id: UUID) -> SafeApiKey:
        self.calls.append(("get", key_id))
        return self.safe

    async def update_agent_key(
        self,
        key_id: UUID,
        command: AgentApiKeyUpdate,
        *,
        actor: AdminPrincipal,
        request_id: str,
        expected_etag: str,
    ) -> SafeApiKey:
        self.calls.append(("update", key_id, command, actor, request_id, expected_etag))
        return self.safe

    async def revoke_agent_key(
        self,
        key_id: UUID,
        *,
        actor: AdminPrincipal,
        request_id: str,
        expected_etag: str | None,
    ) -> SafeApiKey:
        self.calls.append(("revoke", key_id, actor, request_id, expected_etag))
        return self.safe


def _service(double: _ServiceDouble) -> ApiKeyService:
    return cast(ApiKeyService, double)


@pytest.mark.asyncio
async def test_get_api_key_service_wires_the_request_and_authentication_sessions() -> None:
    session = cast(AsyncSession, object())

    @asynccontextmanager
    async def authentication_session() -> AsyncIterator[AsyncSession]:
        yield session

    class DatabaseDouble:
        session = staticmethod(authentication_session)

    settings = _settings()
    service = await admin_routes.get_api_key_service(
        session,
        cast(Database, DatabaseDouble()),
        settings,
    )

    assert service._session is session
    assert service._settings is settings
    async with service._authentication_sessions() as borrowed:
        assert borrowed is session


def test_safe_response_sets_no_store_and_only_optional_resource_headers() -> None:
    safe = _safe()

    response = admin_routes._safe_response(
        safe,
        etag=safe.etag,
        location=f"/v1/admin/api-keys/{KEY_ID}",
        status_code=202,
    )
    page_response = admin_routes._safe_response(Page[SafeApiKey](items=(safe,)))

    assert response.status_code == 202
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["etag"] == safe.etag
    assert response.headers["location"] == f"/v1/admin/api-keys/{KEY_ID}"
    assert "token" not in _document(response.body)
    assert page_response.status_code == 200
    assert page_response.headers["cache-control"] == "no-store"
    assert "etag" not in page_response.headers
    assert "location" not in page_response.headers


@pytest.mark.asyncio
async def test_create_agent_key_returns_the_one_time_token_with_safe_headers() -> None:
    service = _ServiceDouble()

    response = await admin_routes.create_agent_key(
        COMMAND,
        _request(),
        "req-create",
        ACTOR,
        _service(service),
    )

    document = _document(response.body)
    assert response.status_code == 201
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["etag"] == service.safe.etag
    assert response.headers["location"] == f"/v1/admin/api-keys/{KEY_ID}"
    assert document["token"] == "rag_agt_ONE-TIME-SECRET"
    assert set(document) == {"api_key", "token"}
    assert service.calls == [("create", COMMAND, ACTOR, "req-create")]


@pytest.mark.asyncio
async def test_create_agent_key_rejects_idempotency_keys_before_service_dispatch() -> None:
    service = _ServiceDouble()

    with pytest.raises(BusinessError) as raised:
        await admin_routes.create_agent_key(
            COMMAND,
            _request(idempotency_key="must-not-be-accepted"),
            "req-create",
            ACTOR,
            _service(service),
        )

    assert (raised.value.status_code, raised.value.code, raised.value.message) == (
        422,
        "VALIDATION_ERROR",
        "Invalid API key policy",
    )
    assert service.calls == []


@pytest.mark.asyncio
async def test_create_agent_key_rejects_a_safe_document_without_an_etag() -> None:
    service = _ServiceDouble(_safe(etag=None))

    with pytest.raises(BusinessError) as raised:
        await admin_routes.create_agent_key(
            COMMAND,
            _request(),
            "req-create",
            ACTOR,
            _service(service),
        )

    assert (raised.value.status_code, raised.value.code) == (500, "INTERNAL_ERROR")


@pytest.mark.asyncio
async def test_list_and_get_agent_keys_delegate_and_never_expose_a_token() -> None:
    service = _ServiceDouble()

    page_response = await admin_routes.list_agent_keys(
        ACTOR,
        _service(service),
        cursor="cursor-value",
        limit=17,
    )
    get_response = await admin_routes.get_agent_key(KEY_ID, ACTOR, _service(service))

    assert service.calls == [("list", "cursor-value", 17), ("get", KEY_ID)]
    assert _document(page_response.body)["next_cursor"] == "next-page"
    assert "token" not in bytes(page_response.body).decode()
    assert get_response.headers["etag"] == service.safe.etag
    assert get_response.headers["cache-control"] == "no-store"
    assert "token" not in bytes(get_response.body).decode()


@pytest.mark.asyncio
@pytest.mark.parametrize(("if_match", "expected_etag"), [(None, ""), ('"old"', '"old"')])
async def test_update_agent_key_uses_the_route_precondition_contract(
    if_match: str | None,
    expected_etag: str,
) -> None:
    service = _ServiceDouble()

    response = await admin_routes.update_agent_key(
        KEY_ID,
        UPDATE,
        "req-update",
        ACTOR,
        _service(service),
        if_match,
    )

    assert service.calls == [("update", KEY_ID, UPDATE, ACTOR, "req-update", expected_etag)]
    assert response.headers["etag"] == service.safe.etag
    assert response.headers["cache-control"] == "no-store"


@pytest.mark.asyncio
@pytest.mark.parametrize("if_match", [None, '"old"'])
async def test_revoke_agent_key_preserves_an_optional_precondition(if_match: str | None) -> None:
    service = _ServiceDouble()

    response = await admin_routes.revoke_agent_key(
        KEY_ID,
        "req-revoke",
        ACTOR,
        _service(service),
        if_match,
    )

    assert service.calls == [("revoke", KEY_ID, ACTOR, "req-revoke", if_match)]
    assert response.headers["etag"] == service.safe.etag
    assert response.headers["cache-control"] == "no-store"


@pytest.mark.asyncio
async def test_read_and_mutation_routes_reject_safe_documents_without_etags() -> None:
    service = _ServiceDouble(_safe(etag=None))

    for operation in (
        admin_routes.get_agent_key(KEY_ID, ACTOR, _service(service)),
        admin_routes.update_agent_key(
            KEY_ID,
            UPDATE,
            "req-update",
            ACTOR,
            _service(service),
            None,
        ),
        admin_routes.revoke_agent_key(
            KEY_ID,
            "req-revoke",
            ACTOR,
            _service(service),
            None,
        ),
    ):
        with pytest.raises(BusinessError) as raised:
            await operation
        assert (raised.value.status_code, raised.value.code) == (500, "INTERNAL_ERROR")
