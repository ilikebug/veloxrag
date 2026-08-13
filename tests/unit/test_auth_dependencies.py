from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from types import TracebackType
from typing import cast
from uuid import UUID

import pytest
from fastapi import FastAPI, Request
from fastapi.security import HTTPAuthorizationCredentials
from pydantic import SecretStr
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.types import Scope

from rag_service.api.errors import BusinessError
from rag_service.auth import dependencies as auth_dependencies
from rag_service.auth.codec import KeyKind
from rag_service.auth.policies import AdminPrincipal, AgentPrincipal, Capability
from rag_service.config import Settings
from rag_service.db.session import Database

ADMIN = AdminPrincipal(
    key_id=UUID("10000000-0000-0000-0000-000000000001"),
    public_id="YWRtaW4tcHVibGljLWlk",
)
AGENT = AgentPrincipal(
    key_id=UUID("20000000-0000-0000-0000-000000000002"),
    public_id="YWdlbnQtcHVibGljLWlk",
    capabilities=frozenset({Capability.RETRIEVE}),
    knowledge_base_ids=frozenset(),
    query_profile_ids=frozenset(),
    default_query_profile_id=None,
    raw_file_read=False,
    requests_per_minute=60,
    max_concurrency=4,
)


def _settings() -> Settings:
    return Settings(
        _env_file=None,
        environment="test",
        admin_key_hmac_secret=SecretStr("a" * 32),
        agent_key_hmac_secret=SecretStr("b" * 32),
    )


class _SessionContext(AbstractAsyncContextManager[AsyncSession]):
    def __init__(self, session: AsyncSession, events: list[str]) -> None:
        self._session = session
        self._events = events

    async def __aenter__(self) -> AsyncSession:
        self._events.append("enter")
        return self._session

    async def __aexit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exception_type, exception, traceback
        self._events.append("exit")


class _DatabaseDouble:
    def __init__(self) -> None:
        self.session_object = cast(AsyncSession, object())
        self.events: list[str] = []

    def session(self) -> AbstractAsyncContextManager[AsyncSession]:
        return _SessionContext(self.session_object, self.events)


def _database() -> tuple[Database, _DatabaseDouble]:
    double = _DatabaseDouble()
    return cast(Database, double), double


def _install_service_double(
    monkeypatch: pytest.MonkeyPatch,
    outcome: AdminPrincipal | AgentPrincipal | BaseException,
    calls: list[tuple[str, KeyKind]],
) -> None:
    class ServiceDouble:
        def __init__(
            self,
            *,
            session: AsyncSession,
            authentication_sessions: Callable[[], AbstractAsyncContextManager[AsyncSession]],
            settings: Settings,
        ) -> None:
            self._session = session
            self._authentication_sessions = authentication_sessions
            self._settings = settings

        async def authenticate(
            self,
            raw_token: str,
            expected_kind: KeyKind,
        ) -> AdminPrincipal | AgentPrincipal:
            calls.append((raw_token, expected_kind))
            async with self._authentication_sessions() as borrowed:
                assert borrowed is self._session
            assert self._settings is SETTINGS
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome

    monkeypatch.setattr(auth_dependencies, "ApiKeyService", ServiceDouble)


SETTINGS = _settings()


def _assert_invalid_api_key(error: BusinessError) -> None:
    assert (error.status_code, error.code, error.message, error.retryable) == (
        401,
        "INVALID_API_KEY",
        "Invalid API key",
        False,
    )
    assert error.headers == {"WWW-Authenticate": "Bearer"}


def test_get_database_reads_the_application_owned_database() -> None:
    application = FastAPI()
    database, _double = _database()
    application.state.database = database
    request = Request(cast(Scope, {"type": "http", "app": application}))

    assert auth_dependencies.get_database(request) is database


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("principal", "expected_kind"),
    [(ADMIN, KeyKind.ADMIN), (AGENT, KeyKind.AGENT)],
)
async def test_authenticate_returns_only_the_exact_expected_principal(
    principal: AdminPrincipal | AgentPrincipal,
    expected_kind: KeyKind,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database, double = _database()
    calls: list[tuple[str, KeyKind]] = []
    _install_service_double(monkeypatch, principal, calls)

    authenticated = await auth_dependencies._authenticate(
        "sensitive-token",
        database,
        SETTINGS,
        expected_kind,
    )

    assert authenticated is principal
    assert calls == [("sensitive-token", expected_kind)]
    assert double.events == ["enter", "exit"]


@pytest.mark.asyncio
async def test_authenticate_normalizes_invalid_key_errors_and_disconnects_the_cause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database, double = _database()
    original = BusinessError(
        401,
        "INVALID_API_KEY",
        "unsafe provider detail",
        headers={"X-Unsafe": "detail"},
    )
    _install_service_double(monkeypatch, original, [])

    with pytest.raises(BusinessError) as raised:
        await auth_dependencies._authenticate(
            "sensitive-token",
            database,
            SETTINGS,
            KeyKind.ADMIN,
        )

    _assert_invalid_api_key(raised.value)
    assert raised.value is not original
    assert raised.value.__cause__ is None
    traceback = raised.value.__traceback__
    while traceback is not None:
        frame = traceback.tb_frame
        if frame.f_code.co_filename.endswith("rag_service/auth/dependencies.py"):
            assert all(value is not original for value in frame.f_locals.values())
            retained_locals = repr(frame.f_locals)
            assert "unsafe provider detail" not in retained_locals
            assert "X-Unsafe" not in retained_locals
        traceback = traceback.tb_next
    assert double.events == ["enter", "exit"]


@pytest.mark.asyncio
async def test_authenticate_preserves_non_authentication_business_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database, double = _database()
    original = BusinessError(503, "DEPENDENCY_UNAVAILABLE", "Dependency unavailable")
    _install_service_double(monkeypatch, original, [])

    with pytest.raises(BusinessError) as raised:
        await auth_dependencies._authenticate(
            "sensitive-token",
            database,
            SETTINGS,
            KeyKind.ADMIN,
        )

    assert raised.value is original
    assert double.events == ["enter", "exit"]


@pytest.mark.asyncio
async def test_authenticate_rejects_a_principal_of_the_wrong_runtime_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database, _double = _database()
    _install_service_double(monkeypatch, AGENT, [])

    with pytest.raises(BusinessError) as raised:
        await auth_dependencies._authenticate(
            "sensitive-token",
            database,
            SETTINGS,
            KeyKind.ADMIN,
        )

    _assert_invalid_api_key(raised.value)


@pytest.mark.asyncio
async def test_authenticate_redacts_the_token_before_a_base_exception_escapes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = "AUTH-DEPENDENCY-SECRET-MARKER"
    database, double = _database()
    _install_service_double(monkeypatch, KeyboardInterrupt("cancelled"), [])

    with pytest.raises(KeyboardInterrupt) as raised:
        await auth_dependencies._authenticate(marker, database, SETTINGS, KeyKind.ADMIN)

    traceback = raised.value.__traceback__
    while traceback is not None:
        frame = traceback.tb_frame
        if frame.f_code.co_filename.endswith("rag_service/auth/dependencies.py"):
            assert marker not in repr(frame.f_locals)
        traceback = traceback.tb_next
    assert double.events == ["enter", "exit"]


@pytest.mark.asyncio
async def test_require_principals_reject_missing_credentials_uniformly() -> None:
    database, _double = _database()

    with pytest.raises(BusinessError) as admin_error:
        await auth_dependencies.require_admin_principal(None, database, SETTINGS)
    with pytest.raises(BusinessError) as agent_error:
        await auth_dependencies.require_agent_principal(None, database, SETTINGS)

    _assert_invalid_api_key(admin_error.value)
    _assert_invalid_api_key(agent_error.value)


@pytest.mark.asyncio
async def test_require_principals_delegate_with_the_expected_kind(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database, _double = _database()
    observed: list[tuple[str, KeyKind]] = []

    async def authenticate(
        raw_token: str,
        _database: Database,
        _settings: Settings,
        expected_kind: KeyKind,
    ) -> AdminPrincipal | AgentPrincipal:
        observed.append((raw_token, expected_kind))
        return ADMIN if expected_kind is KeyKind.ADMIN else AGENT

    monkeypatch.setattr(auth_dependencies, "_authenticate", authenticate)
    credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials="opaque-token")

    admin = await auth_dependencies.require_admin_principal(credentials, database, SETTINGS)
    agent = await auth_dependencies.require_agent_principal(credentials, database, SETTINGS)

    assert admin is ADMIN
    assert agent is AGENT
    assert observed == [
        ("opaque-token", KeyKind.ADMIN),
        ("opaque-token", KeyKind.AGENT),
    ]


@pytest.mark.asyncio
async def test_require_principals_reject_a_wrong_runtime_type_from_the_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database, _double = _database()

    async def authenticate(
        _raw_token: str,
        _database: Database,
        _settings: Settings,
        expected_kind: KeyKind,
    ) -> AdminPrincipal | AgentPrincipal:
        return AGENT if expected_kind is KeyKind.ADMIN else ADMIN

    monkeypatch.setattr(auth_dependencies, "_authenticate", authenticate)
    credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials="opaque-token")

    with pytest.raises(BusinessError) as admin_error:
        await auth_dependencies.require_admin_principal(credentials, database, SETTINGS)
    with pytest.raises(BusinessError) as agent_error:
        await auth_dependencies.require_agent_principal(credentials, database, SETTINGS)

    _assert_invalid_api_key(admin_error.value)
    _assert_invalid_api_key(agent_error.value)
