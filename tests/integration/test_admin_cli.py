import asyncio
import json
import os
import subprocess
import sys
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, cast

import httpx
import pytest
from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials
from pydantic import SecretStr
from sqlalchemy.ext.asyncio import AsyncSession

from rag_service.api.errors import BusinessError
from rag_service.auth.dependencies import require_admin_principal, require_agent_principal
from rag_service.auth.policies import AdminPrincipal, AgentPrincipal, Capability
from rag_service.auth.schemas import AdminApiKeyCreate, AgentApiKeyCreate
from rag_service.auth.services import ApiKeyService
from rag_service.config import Settings, get_settings
from rag_service.db.dependencies import get_session
from rag_service.db.session import Database

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
ADMIN_HMAC_SECRET = "admin-test-hmac-secret-32-bytes!!"
AGENT_HMAC_SECRET = "agent-test-hmac-secret-32-bytes!!"


def _settings(database_url: str) -> Settings:
    return Settings(
        _env_file=None,
        environment="test",
        database_url=SecretStr(database_url),
        admin_key_hmac_secret=SecretStr(ADMIN_HMAC_SECRET),
        agent_key_hmac_secret=SecretStr(AGENT_HMAC_SECRET),
        max_api_key_requests_per_minute=100,
        max_api_key_concurrency=10,
    )


def _cli_environment(database_url: str) -> dict[str, str]:
    return os.environ | {
        "RAG_ENVIRONMENT": "test",
        "RAG_DATABASE_URL": database_url,
        "RAG_ADMIN_KEY_HMAC_SECRET": ADMIN_HMAC_SECRET,
        "RAG_AGENT_KEY_HMAC_SECRET": AGENT_HMAC_SECRET,
        "RAG_MAX_API_KEY_REQUESTS_PER_MINUTE": "100",
        "RAG_MAX_API_KEY_CONCURRENCY": "10",
        "PYTHONUNBUFFERED": "1",
    }


def _run_cli(
    database_url: str,
    *arguments: str,
    python_code: str | None = None,
) -> subprocess.CompletedProcess[str]:
    if python_code is None:
        command = [str(Path(sys.executable).with_name("velox-admin")), *arguments]
    else:
        command = [sys.executable, "-c", python_code]
    return subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        env=_cli_environment(database_url),
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )


def _json_stdout(result: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    assert result.stdout.count("\n") == 1
    parsed = json.loads(result.stdout)
    assert isinstance(parsed, dict)
    return cast(dict[str, Any], parsed)


def _assert_safe_document(document: object, *tokens: str) -> None:
    serialized = json.dumps(document, sort_keys=True)
    lowered = serialized.lower()
    assert "secret_digest" not in lowered
    assert "authorization" not in lowered
    for token in tokens:
        assert token not in serialized


def _write_policy(path: Path, document: Mapping[str, object]) -> None:
    path.write_text(json.dumps(document), encoding="utf-8")


class _TrackingDatabase:
    def __init__(self, delegate: Database) -> None:
        self._delegate = delegate
        self.opened: list[tuple[str, int]] = []
        self.closed: list[tuple[str, int]] = []

    @asynccontextmanager
    async def _open(self, purpose: str) -> AsyncIterator[AsyncSession]:
        async with self._delegate.session() as session:
            marker = (purpose, id(session))
            self.opened.append(marker)
            try:
                yield session
            finally:
                self.closed.append(marker)

    def session(self) -> Any:
        return self._open("auth")

    def sessions(self) -> Any:
        return self._open("business")


def _error_response(_request: Request, error: Exception) -> JSONResponse:
    assert isinstance(error, BusinessError)
    return JSONResponse(
        status_code=error.status_code,
        headers=dict(error.headers or {}),
        content={
            "error": {
                "code": error.code,
                "message": error.message,
                "retryable": error.retryable,
            }
        },
    )


def _dependency_test_app(database: _TrackingDatabase, settings: Settings) -> FastAPI:
    app = FastAPI()
    app.state.database = database
    app.dependency_overrides[get_settings] = lambda: settings
    app.add_exception_handler(BusinessError, _error_response)

    @app.get("/admin")
    async def admin_route(
        principal: Annotated[AdminPrincipal, Depends(require_admin_principal)],
        business_session: Annotated[AsyncSession, Depends(get_session)],
    ) -> dict[str, object]:
        return {
            "key_id": str(principal.key_id),
            "business_session_id": id(business_session),
            "business_in_transaction": business_session.in_transaction(),
        }

    @app.get("/agent")
    async def agent_route(
        principal: Annotated[AgentPrincipal, Depends(require_agent_principal)],
    ) -> dict[str, object]:
        return {"key_id": str(principal.key_id)}

    return app


async def _issued_tokens(database: Database, settings: Settings) -> tuple[str, str]:
    async with database.session() as session:
        service = ApiKeyService(
            session=session,
            authentication_sessions=database.session,
            settings=settings,
        )
        admin = await service.create_admin_key(
            AdminApiKeyCreate(name="dependency-admin"),
            request_id="req-dependency-admin",
        )
        agent = await service.create_agent_key(
            AgentApiKeyCreate(
                name="dependency-agent",
                capabilities=frozenset({Capability.RETRIEVE}),
                requests_per_minute=60,
                max_concurrency=4,
            ),
            actor=None,
            request_id="req-dependency-agent",
        )
    return admin.token.get_secret_value(), agent.token.get_secret_value()


async def _create_admin_recovery_batch(
    database_url: str,
    count: int,
) -> tuple[list[dict[str, Any]], str]:
    settings = _settings(database_url)
    database = Database.from_settings(settings)
    issued_keys: list[dict[str, Any]] = []
    last_token = ""
    try:
        async with database.session() as session:
            service = ApiKeyService(
                session=session,
                authentication_sessions=database.session,
                settings=settings,
            )
            for index in range(count):
                issued = await service.create_admin_key(
                    AdminApiKeyCreate(name=f"recovery-admin-{index:02d}"),
                    request_id=f"req-recovery-admin-{index:02d}",
                )
                issued_keys.append(issued.api_key.model_dump(mode="json"))
                last_token = issued.token.get_secret_value()
    finally:
        await database.close()
    return issued_keys, last_token


@pytest.mark.integration
@pytest.mark.asyncio
async def test_bearer_dependencies_are_uniform_and_use_an_isolated_auth_session(
    migrated_database: Database,
    postgres_urls: tuple[str, str],
) -> None:
    async_url, _sync_url = postgres_urls
    settings = _settings(async_url)
    admin_token, agent_token = await _issued_tokens(migrated_database, settings)
    tracked = _TrackingDatabase(migrated_database)
    app = _dependency_test_app(tracked, settings)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        invalid_requests: tuple[tuple[str, dict[str, str]], ...] = (
            ("/admin", {}),
            ("/admin", {"Authorization": "Basic opaque"}),
            ("/admin", {"Authorization": "Bearer malformed"}),
            ("/admin", {"Authorization": f"Bearer {agent_token}"}),
            ("/agent", {"Authorization": f"Bearer {admin_token}"}),
        )
        for path, headers in invalid_requests:
            before = len(tracked.opened)
            response = await client.get(path, headers=headers)
            assert response.status_code == 401
            assert response.headers["www-authenticate"] == "Bearer"
            assert response.json()["error"] == {
                "code": "INVALID_API_KEY",
                "message": "Invalid API key",
                "retryable": False,
            }
            assert admin_token not in response.text
            assert agent_token not in response.text
            assert tracked.closed[before:] == tracked.opened[before:]

        tracked.opened.clear()
        tracked.closed.clear()
        response = await client.get(
            "/admin",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert response.status_code == 200
        assert response.json()["business_in_transaction"] is False
        assert [purpose for purpose, _identifier in tracked.opened] == ["auth", "business"]
        assert tracked.closed == tracked.opened
        assert len({identifier for _purpose, identifier in tracked.opened}) == 2
        assert response.json()["business_session_id"] == tracked.opened[1][1]

        tracked.opened.clear()
        tracked.closed.clear()
        response = await client.get(
            "/agent",
            headers={"Authorization": f"Bearer {agent_token}"},
        )
        assert response.status_code == 200
        assert tracked.opened == tracked.closed
        assert tracked.opened[0][0] == "auth"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_bearer_dependency_does_not_retain_credentials_in_error_frames(
    migrated_database: Database,
    postgres_urls: tuple[str, str],
) -> None:
    async_url, _sync_url = postgres_urls
    settings = _settings(async_url)
    sensitive_token = "rag_adm_sensitive-invalid-token-material"
    credentials = HTTPAuthorizationCredentials(
        scheme="Bearer",
        credentials=sensitive_token,
    )

    with pytest.raises(BusinessError) as raised:
        await require_admin_principal(credentials, migrated_database, settings)

    error: BaseException | None = raised.value
    visited: set[int] = set()
    while error is not None and id(error) not in visited:
        visited.add(id(error))
        traceback_cursor = error.__traceback__
        while traceback_cursor is not None:
            frame = traceback_cursor.tb_frame
            if "/src/rag_service/" in frame.f_code.co_filename:
                local_values = tuple(frame.f_locals.values())
                assert all(value is not credentials for value in local_values)
                assert sensitive_token not in repr(frame.f_locals)
            traceback_cursor = traceback_cursor.tb_next
        error = error.__cause__ or error.__context__


@pytest.mark.integration
def test_cli_full_lifecycle_separates_key_classes_and_never_reprints_tokens(
    postgres_urls: tuple[str, str],
    migrated_autocommit_sync_connection: object,
    tmp_path: Path,
) -> None:
    del migrated_autocommit_sync_connection
    async_url, _sync_url = postgres_urls

    admin_create = _run_cli(async_url, "admin-key", "create", "--name", "local-admin")
    admin_created = _json_stdout(admin_create)
    admin_token = cast(str, admin_created["token"])
    admin_safe = cast(dict[str, Any], admin_created["api_key"])
    assert admin_create.stdout.count(admin_token) == 1
    assert admin_safe["key_type"] == "admin"
    assert set(admin_created) == {"api_key", "token"}

    agent_policy = tmp_path / "agent-create.json"
    _write_policy(
        agent_policy,
        {
            "name": "retrieval-agent",
            "capabilities": ["retrieve"],
            "knowledge_base_ids": [],
            "query_profile_ids": [],
            "default_query_profile_id": None,
            "raw_file_read": False,
            "requests_per_minute": 60,
            "max_concurrency": 4,
            "not_before": None,
            "expires_at": None,
        },
    )
    agent_create = _run_cli(
        async_url,
        "agent-key",
        "create",
        "--from-file",
        str(agent_policy),
    )
    agent_created = _json_stdout(agent_create)
    agent_token = cast(str, agent_created["token"])
    agent_safe = cast(dict[str, Any], agent_created["api_key"])
    assert agent_create.stdout.count(agent_token) == 1
    assert agent_safe["key_type"] == "agent"
    assert set(agent_created) == {"api_key", "token"}
    assert admin_token not in agent_create.stdout

    admin_list = _json_stdout(_run_cli(async_url, "admin-key", "list"))
    agent_list = _json_stdout(_run_cli(async_url, "agent-key", "list", "--limit", "1"))
    assert [item["id"] for item in admin_list["items"]] == [admin_safe["id"]]
    assert [item["id"] for item in agent_list["items"]] == [agent_safe["id"]]
    _assert_safe_document(admin_list, admin_token, agent_token)
    _assert_safe_document(agent_list, admin_token, agent_token)
    assert "token" not in admin_list
    assert "token" not in agent_list

    update_policy = tmp_path / "agent-update.json"
    _write_policy(
        update_policy,
        {
            "name": "retrieval-agent-renamed",
            "capabilities": ["retrieve", "answer"],
            "requests_per_minute": 30,
        },
    )
    agent_updated = _json_stdout(
        _run_cli(
            async_url,
            "agent-key",
            "update",
            cast(str, agent_safe["id"]),
            "--from-file",
            str(update_policy),
        )
    )
    assert agent_updated["name"] == "retrieval-agent-renamed"
    assert agent_updated["resource_revision"] == 2
    _assert_safe_document(agent_updated, admin_token, agent_token)
    assert "token" not in agent_updated

    agent_revoked = _json_stdout(
        _run_cli(async_url, "agent-key", "revoke", cast(str, agent_safe["id"]))
    )
    admin_revoked = _json_stdout(
        _run_cli(async_url, "admin-key", "revoke", cast(str, admin_safe["id"]))
    )
    assert agent_revoked["status"] == "revoked"
    assert admin_revoked["status"] == "revoked"
    for safe_document in (agent_revoked, admin_revoked):
        _assert_safe_document(safe_document, admin_token, agent_token)
        assert "token" not in safe_document


@pytest.mark.integration
def test_admin_recovery_can_page_beyond_the_default_list_window(
    postgres_urls: tuple[str, str],
    migrated_autocommit_sync_connection: object,
) -> None:
    del migrated_autocommit_sync_connection
    async_url, _sync_url = postgres_urls
    issued_keys, lost_token = asyncio.run(_create_admin_recovery_batch(async_url, 21))
    lost_key = issued_keys[-1]
    cursor: str | None = None
    found: dict[str, Any] | None = None

    for _page_number in range(4):
        arguments = ["admin-key", "list", "--limit", "7"]
        if cursor is not None:
            arguments.extend(("--cursor", cursor))
        page = _json_stdout(_run_cli(async_url, *arguments))
        _assert_safe_document(page, lost_token)
        found = next(
            (item for item in page["items"] if item["id"] == lost_key["id"]),
            None,
        )
        if found is not None:
            break
        cursor = cast(str | None, page["next_cursor"])
        assert cursor is not None

    assert found is not None
    assert found["name"] == "recovery-admin-20"
    revoked = _json_stdout(_run_cli(async_url, "admin-key", "revoke", found["id"]))
    assert revoked["status"] == "revoked"
    _assert_safe_document(revoked, lost_token)
    replacement = _json_stdout(
        _run_cli(async_url, "admin-key", "create", "--name", "recovery-replacement")
    )
    assert replacement["api_key"]["id"] != found["id"]
    assert replacement["token"] != lost_token


@pytest.mark.integration
def test_cli_sanitizes_errors_and_recovers_after_committed_token_output_failure(
    postgres_urls: tuple[str, str],
    migrated_autocommit_sync_connection: object,
    tmp_path: Path,
) -> None:
    del migrated_autocommit_sync_connection
    async_url, _sync_url = postgres_urls
    output_failure_code = """
import sys
from rag_service.admin.cli import main

class BrokenStdout:
    def write(self, value):
        if sys.argv[1] == "write-error":
            raise OSError("sensitive stdout failure")
        if sys.argv[1] == "short-write":
            return len(value) - 1
        return len(value)

    def flush(self):
        if sys.argv[1] == "flush-error":
            raise OSError("sensitive flush failure")

real_stdout = sys.stdout
sys.stdout = BrokenStdout()
exit_code = main(["admin-key", "create", "--name", sys.argv[2]])
sys.stdout = real_stdout
raise SystemExit(exit_code)
"""
    failure_cases = (
        ("write-error", "undelivered-write-admin"),
        ("short-write", "undelivered-short-admin"),
        ("flush-error", "undelivered-flush-admin"),
    )
    for failure_mode, name in failure_cases:
        command = [sys.executable, "-c", output_failure_code, failure_mode, name]
        undelivered = subprocess.run(
            command,
            cwd=REPOSITORY_ROOT,
            env=_cli_environment(async_url),
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
        assert undelivered.returncode != 0
        assert undelivered.stdout == ""
        output_error = json.loads(undelivered.stderr)
        assert output_error == {
            "error": {
                "code": "OUTPUT_DELIVERY_INDETERMINATE",
                "message": "Output delivery is uncertain; use list to verify before retrying",
                "retryable": False,
            }
        }
        assert "rag_adm_" not in undelivered.stderr
        assert "sensitive stdout failure" not in undelivered.stderr
        assert "sensitive flush failure" not in undelivered.stderr

    listed = _json_stdout(_run_cli(async_url, "admin-key", "list"))
    lost_keys = [
        next(item for item in listed["items"] if item["name"] == name)
        for _failure_mode, name in failure_cases
    ]
    for lost in lost_keys:
        assert "token" not in lost
        revoked = _json_stdout(_run_cli(async_url, "admin-key", "revoke", lost["id"]))
        assert revoked["status"] == "revoked"
        assert "token" not in revoked
    replacement = _json_stdout(
        _run_cli(async_url, "admin-key", "create", "--name", "replacement-admin")
    )
    assert replacement["api_key"]["id"] not in {lost["id"] for lost in lost_keys}
    assert replacement["token"].startswith("rag_adm_")

    sensitive_marker = "DO-NOT-ECHO-THIS-POLICY"
    invalid_policy = tmp_path / sensitive_marker
    invalid_policy.write_text(f'{{"name":"{sensitive_marker}"', encoding="utf-8")
    invalid = _run_cli(
        async_url,
        "agent-key",
        "create",
        "--from-file",
        str(invalid_policy),
    )
    assert invalid.returncode != 0
    assert invalid.stdout == ""
    assert invalid.stderr.count("\n") == 1
    error_document = json.loads(invalid.stderr)
    assert error_document == {
        "error": {
            "code": "VALIDATION_ERROR",
            "message": "Invalid command input",
            "retryable": False,
        }
    }
    assert sensitive_marker not in invalid.stderr
    assert str(invalid_policy) not in invalid.stderr

    malformed_id = _run_cli(async_url, "admin-key", "revoke", sensitive_marker)
    assert malformed_id.returncode != 0
    assert malformed_id.stdout == ""
    assert json.loads(malformed_id.stderr)["error"]["code"] == "VALIDATION_ERROR"
    assert sensitive_marker not in malformed_id.stderr

    missing_id = _run_cli(
        async_url,
        "admin-key",
        "revoke",
        "00000000-0000-0000-0000-000000000001",
    )
    assert missing_id.returncode != 0
    assert missing_id.stdout == ""
    assert json.loads(missing_id.stderr) == {
        "error": {
            "code": "RESOURCE_NOT_FOUND",
            "message": "Resource not found",
            "retryable": False,
        }
    }
    assert "Traceback" not in missing_id.stderr
