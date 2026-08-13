import asyncio
import logging
import re
from collections.abc import AsyncIterator, Iterator
from types import TracebackType
from typing import Annotated, cast
from uuid import UUID

import httpx
import pytest
from fastapi import BackgroundTasks, Body, Depends, FastAPI, Query, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, SecretStr
from starlette.types import Message, Scope

from rag_service.admin.routes import get_api_key_service
from rag_service.api.errors import BusinessError, validation_error_response
from rag_service.api.middleware import get_request_id
from rag_service.auth.dependencies import require_admin_principal, require_agent_principal
from rag_service.auth.policies import AdminPrincipal, AgentPrincipal, Capability
from rag_service.auth.services import ApiKeyService
from rag_service.config import Settings, get_settings
from rag_service.main import create_app
from rag_service.metadata.document_routes import get_document_metadata_service
from rag_service.metadata.knowledge_base_routes import get_knowledge_base_service
from rag_service.metadata.services import DocumentMetadataService, KnowledgeBaseService
from rag_service.readiness import ComponentStatus, ReadinessScope, ReadinessSnapshot

pytestmark = pytest.mark.integration

REQUEST_ID_PATTERN = re.compile(r"req_[A-Za-z0-9_-]+")
AUTHORIZATION_SENTINEL = "authorization-error-contract-secret"
HMAC_SENTINEL = "hmac-error-contract-secret-32-bytes"
DATABASE_SENTINEL = "database-error-contract-secret"
EXCEPTION_SENTINEL = "exception-error-contract-secret"


def _settings(*, max_request_id_length: int = 32) -> Settings:
    return Settings(
        _env_file=None,
        environment="test",
        admin_key_hmac_secret=SecretStr(HMAC_SENTINEL),
        agent_key_hmac_secret=SecretStr("agent-error-contract-secret-32-bytes"),
        max_request_id_length=max_request_id_length,
    )


class _ReadinessProvider:
    async def snapshot(self, _scope: ReadinessScope = ReadinessScope.ALL) -> ReadinessSnapshot:
        return ReadinessSnapshot(
            components={
                name: ComponentStatus(ok=True, latency_ms=1.0)
                for name in (
                    "postgres",
                    "qdrant",
                    "redis",
                    "minio",
                    "ingest_credentials",
                    "retrieve_credentials",
                )
            },
            answer_configured=True,
        )


def _app(settings: Settings | None = None) -> FastAPI:
    app = create_app(settings=settings or _settings())
    app.state.readiness_provider = _ReadinessProvider()
    return app


def _assert_error(
    response: httpx.Response,
    *,
    status_code: int,
    code: str,
    message: str,
    request_id: str,
    retryable: bool = False,
) -> None:
    assert response.status_code == status_code
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-request-id"] == request_id
    assert response.json() == {
        "error": {
            "code": code,
            "message": message,
            "retryable": retryable,
            "request_id": request_id,
        }
    }


def _agent() -> AgentPrincipal:
    return AgentPrincipal(
        key_id=UUID(int=2),
        public_id="BBBBBBBBBBBBBBBB",
        capabilities=frozenset({Capability.MANAGE}),
        knowledge_base_ids=frozenset(),
        query_profile_ids=frozenset(),
        default_query_profile_id=None,
        raw_file_read=False,
        requests_per_minute=60,
        max_concurrency=4,
    )


def _captured_exception(error: Exception, marker: str) -> Exception:
    retained_inner_marker = marker
    try:
        assert retained_inner_marker
        raise error
    except Exception as captured:
        return captured


def _exception_nodes(error: BaseException) -> list[BaseException]:
    pending = [error]
    visited: set[int] = set()
    nodes: list[BaseException] = []
    while pending:
        current = pending.pop()
        if id(current) in visited:
            continue
        visited.add(id(current))
        nodes.append(current)
        if isinstance(current, BaseExceptionGroup):
            pending.extend(current.exceptions)
        for nested in (current.__cause__, current.__context__):
            if nested is not None:
                pending.append(nested)
    return nodes


@pytest.mark.asyncio
async def test_health_ready_and_success_responses_have_a_middleware_request_id() -> None:
    settings = _settings(max_request_id_length=32)
    app = _app(settings)

    @app.get("/_contract/success")
    async def success(request: Request) -> dict[str, str]:
        return {"request_id": cast(str, request.state.request_id)}

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        health = await client.get("/health")
        ready = await client.get("/ready", headers={"X-Request-ID": "request-ready-1"})
        success_response = await client.get("/_contract/success")

    for response in (health, success_response):
        request_id = response.headers["x-request-id"]
        assert REQUEST_ID_PATTERN.fullmatch(request_id)
        assert len(request_id) <= settings.max_request_id_length
    assert ready.headers["x-request-id"] == "request-ready-1"
    assert success_response.json() == {"request_id": success_response.headers["x-request-id"]}


@pytest.mark.asyncio
async def test_settings_none_uses_the_runtime_request_id_limit_everywhere(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RAG_MAX_REQUEST_ID_LENGTH", "32")
    get_settings.cache_clear()
    try:
        app = create_app()

        @app.get("/_contract/runtime-settings")
        async def runtime_settings(
            request_id: Annotated[str, Depends(get_request_id)],
        ) -> dict[str, str]:
            return {"request_id": request_id}

        supplied = "x" * 33
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.get(
                "/_contract/runtime-settings",
                headers={"X-Request-ID": supplied},
            )
    finally:
        get_settings.cache_clear()

    assert response.status_code == 200
    generated = response.headers["x-request-id"]
    assert generated != supplied
    assert REQUEST_ID_PATTERN.fullmatch(generated)
    assert len(generated) <= 32
    assert response.json() == {"request_id": generated}


def test_settings_none_shares_a_monkeypatched_runtime_settings_instance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(max_request_id_length=32)
    monkeypatch.setattr("rag_service.main.get_settings", lambda: settings)

    app = create_app()

    assert app.dependency_overrides[get_settings]() is settings


@pytest.mark.asyncio
async def test_dynamic_sync_settings_override_drives_middleware_and_route_once() -> None:
    settings = _settings(max_request_id_length=32)
    calls = 0
    seen_settings: list[Settings] = []

    def settings_override() -> Settings:
        nonlocal calls
        calls += 1
        return settings

    app = create_app()
    app.dependency_overrides[get_settings] = settings_override
    assert app.dependency_overrides[get_settings] is settings_override
    assert app.dependency_overrides.get(get_settings) is settings_override

    @app.get("/_contract/dynamic-sync-settings")
    async def dynamic_sync_settings(
        route_settings: Annotated[Settings, Depends(get_settings)],
        request_id: Annotated[str, Depends(get_request_id)],
    ) -> dict[str, str]:
        seen_settings.append(route_settings)
        return {"request_id": request_id}

    supplied = "x" * 33
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.get(
            "/_contract/dynamic-sync-settings",
            headers={"X-Request-ID": supplied},
        )

    assert response.status_code == 200
    generated = response.headers["x-request-id"]
    assert generated != supplied
    assert len(generated) <= settings.max_request_id_length
    assert response.json() == {"request_id": generated}
    assert seen_settings == [settings]
    assert seen_settings[0] is settings
    assert calls == 1


@pytest.mark.asyncio
async def test_dynamic_async_settings_override_drives_middleware_and_route_once() -> None:
    settings = _settings(max_request_id_length=32)
    calls = 0
    seen_settings: list[Settings] = []

    async def settings_override() -> Settings:
        nonlocal calls
        calls += 1
        return settings

    app = create_app()
    app.dependency_overrides[get_settings] = settings_override

    @app.get("/_contract/dynamic-async-settings")
    async def dynamic_async_settings(
        route_settings: Annotated[Settings, Depends(get_settings)],
        request_id: Annotated[str, Depends(get_request_id)],
    ) -> dict[str, str]:
        seen_settings.append(route_settings)
        return {"request_id": request_id}

    supplied = "x" * 33
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.get(
            "/_contract/dynamic-async-settings",
            headers={"X-Request-ID": supplied},
        )

    assert response.status_code == 200
    generated = response.headers["x-request-id"]
    assert generated != supplied
    assert len(generated) <= settings.max_request_id_length
    assert response.json() == {"request_id": generated}
    assert seen_settings == [settings]
    assert seen_settings[0] is settings
    assert calls == 1


@pytest.mark.asyncio
async def test_dynamic_async_settings_override_can_receive_the_request() -> None:
    settings = _settings(max_request_id_length=32)
    calls = 0
    seen_paths: list[str] = []
    seen_settings: list[Settings] = []

    async def settings_override(request: Request) -> Settings:
        nonlocal calls
        calls += 1
        seen_paths.append(request.url.path)
        return settings

    app = create_app()
    app.dependency_overrides[get_settings] = settings_override

    @app.get("/_contract/request-settings-override")
    async def request_settings_override(
        route_settings: Annotated[Settings, Depends(get_settings)],
        request_id: Annotated[str, Depends(get_request_id)],
    ) -> dict[str, str]:
        seen_settings.append(route_settings)
        return {"request_id": request_id}

    supplied = "x" * 33
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.get(
            "/_contract/request-settings-override",
            headers={"X-Request-ID": supplied},
        )

    assert response.status_code == 200
    assert response.headers["x-request-id"] != supplied
    assert response.json() == {"request_id": response.headers["x-request-id"]}
    assert seen_paths == ["/_contract/request-settings-override"]
    assert seen_settings == [settings]
    assert seen_settings[0] is settings
    assert calls == 1


@pytest.mark.asyncio
async def test_settings_override_subdependencies_use_dynamic_fastapi_overrides() -> None:
    settings = _settings(max_request_id_length=32)
    child_calls = 0
    provider_calls = 0
    seen_settings: list[Settings] = []

    async def unavailable_child() -> str:
        raise AssertionError("the child dependency override must be honored")

    async def child_override(request: Request) -> str:
        nonlocal child_calls
        child_calls += 1
        return request.url.path

    def settings_override(
        child_value: Annotated[str, Depends(unavailable_child)],
    ) -> Settings:
        nonlocal provider_calls
        provider_calls += 1
        assert child_value == "/_contract/nested-settings-override"
        return settings

    app = create_app()
    app.dependency_overrides[get_settings] = settings_override
    app.dependency_overrides[unavailable_child] = child_override

    @app.get("/_contract/nested-settings-override")
    async def nested_settings_override(
        route_settings: Annotated[Settings, Depends(get_settings)],
        request_id: Annotated[str, Depends(get_request_id)],
    ) -> dict[str, str]:
        seen_settings.append(route_settings)
        return {"request_id": request_id}

    supplied = "x" * 33
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.get(
            "/_contract/nested-settings-override",
            headers={"X-Request-ID": supplied},
        )

    assert response.status_code == 200
    assert response.headers["x-request-id"] != supplied
    assert response.json() == {"request_id": response.headers["x-request-id"]}
    assert seen_settings == [settings]
    assert seen_settings[0] is settings
    assert child_calls == 1
    assert provider_calls == 1


@pytest.mark.asyncio
async def test_sync_yield_settings_override_cleans_up_after_the_response() -> None:
    settings = _settings(max_request_id_length=32)
    events: list[str] = []
    seen_settings: list[Settings] = []

    def settings_override(request: Request) -> Iterator[Settings]:
        events.append(f"enter:{request.url.path}")
        try:
            yield settings
        finally:
            events.append("cleanup")

    app = create_app()
    app.dependency_overrides[get_settings] = settings_override

    @app.get("/_contract/sync-yield-settings")
    async def sync_yield_settings(
        route_settings: Annotated[Settings, Depends(get_settings)],
        request_id: Annotated[str, Depends(get_request_id)],
    ) -> dict[str, str]:
        events.append("route")
        seen_settings.append(route_settings)
        return {"request_id": request_id}

    supplied = "x" * 33
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.get(
            "/_contract/sync-yield-settings",
            headers={"X-Request-ID": supplied},
        )

    assert response.status_code == 200
    assert response.headers["x-request-id"] != supplied
    assert response.json() == {"request_id": response.headers["x-request-id"]}
    assert events == ["enter:/_contract/sync-yield-settings", "route", "cleanup"]
    assert seen_settings == [settings]
    assert seen_settings[0] is settings


@pytest.mark.asyncio
async def test_async_yield_settings_override_cleans_up_after_the_response() -> None:
    settings = _settings(max_request_id_length=32)
    events: list[str] = []
    seen_settings: list[Settings] = []

    async def settings_override(request: Request) -> AsyncIterator[Settings]:
        events.append(f"enter:{request.url.path}")
        try:
            yield settings
        finally:
            events.append("cleanup")

    app = create_app()
    app.dependency_overrides[get_settings] = settings_override

    @app.get("/_contract/async-yield-settings")
    async def async_yield_settings(
        route_settings: Annotated[Settings, Depends(get_settings)],
        request_id: Annotated[str, Depends(get_request_id)],
    ) -> dict[str, str]:
        events.append("route")
        seen_settings.append(route_settings)
        return {"request_id": request_id}

    supplied = "x" * 33
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.get(
            "/_contract/async-yield-settings",
            headers={"X-Request-ID": supplied},
        )

    assert response.status_code == 200
    assert response.headers["x-request-id"] != supplied
    assert response.json() == {"request_id": response.headers["x-request-id"]}
    assert events == ["enter:/_contract/async-yield-settings", "route", "cleanup"]
    assert seen_settings == [settings]
    assert seen_settings[0] is settings


@pytest.mark.asyncio
async def test_settings_yield_cannot_suppress_an_endpoint_failure_without_a_response() -> None:
    settings = _settings(max_request_id_length=32)
    failure = _captured_exception(RuntimeError(EXCEPTION_SENTINEL), EXCEPTION_SENTINEL)
    events: list[str] = []

    async def settings_override() -> AsyncIterator[Settings]:
        events.append("provider-enter")
        try:
            yield settings
        except Exception:
            events.append("provider-suppressed")
            return
        finally:
            events.append("provider-cleanup")

    app = create_app()
    app.dependency_overrides[get_settings] = settings_override

    @app.get("/_contract/suppressed-endpoint-failure")
    async def suppressed_endpoint_failure() -> None:
        events.append("route")
        raise failure

    request_id = "req-suppressed-endpoint-failure"
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=True),
        base_url="http://test",
    ) as client:
        response = await client.get(
            "/_contract/suppressed-endpoint-failure",
            headers={"X-Request-ID": request_id},
        )

    _assert_error(
        response,
        status_code=500,
        code="INTERNAL_ERROR",
        message="Internal server error",
        request_id=request_id,
    )
    assert events == [
        "provider-enter",
        "route",
        "provider-suppressed",
        "provider-cleanup",
    ]
    assert failure.args == ("<redacted>",)
    assert failure.__traceback__ is None


@pytest.mark.asyncio
async def test_settings_yield_cannot_suppress_an_incomplete_started_response() -> None:
    settings = _settings(max_request_id_length=32)
    failure = RuntimeError(EXCEPTION_SENTINEL)

    async def settings_override() -> AsyncIterator[Settings]:
        try:
            yield settings
        except Exception:
            return

    async def failing_stream() -> AsyncIterator[bytes]:
        raise failure
        yield b"unreachable"

    app = create_app()
    app.dependency_overrides[get_settings] = settings_override

    @app.get("/_contract/suppressed-stream-failure")
    async def suppressed_stream_failure() -> StreamingResponse:
        return StreamingResponse(failing_stream())

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=True),
        base_url="http://test",
    ) as client:
        with pytest.raises(RuntimeError) as raised:
            await client.get("/_contract/suppressed-stream-failure")

    assert raised.value.args == ("Request processing failed",)
    assert failure.args == ("<redacted>",)
    assert failure.__traceback__ is None


@pytest.mark.asyncio
async def test_request_scoped_settings_child_cannot_suppress_a_nested_cleanup_failure() -> None:
    settings = _settings(max_request_id_length=32)
    failure = _captured_exception(RuntimeError(EXCEPTION_SENTINEL), EXCEPTION_SENTINEL)
    events: list[str] = []

    async def suppressing_child() -> AsyncIterator[str]:
        events.append("child-enter")
        try:
            yield "child-value"
        except Exception:
            events.append("child-suppressed")
            return
        finally:
            events.append("child-cleanup")

    async def settings_override(
        child_value: Annotated[str, Depends(suppressing_child, scope="request")],
    ) -> AsyncIterator[Settings]:
        assert child_value == "child-value"
        events.append("provider-enter")
        try:
            yield settings
        finally:
            events.append("provider-cleanup")
            raise failure

    app = create_app()
    app.dependency_overrides[get_settings] = settings_override

    @app.get("/_contract/request-nested-cleanup-suppression")
    async def request_nested_cleanup_suppression() -> dict[str, bool]:
        events.append("route")
        return {"ok": True}

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=True),
        base_url="http://test",
    ) as client:
        with pytest.raises(RuntimeError) as raised:
            await client.get("/_contract/request-nested-cleanup-suppression")

    assert raised.value.args == ("Request processing failed",)
    assert events == [
        "child-enter",
        "provider-enter",
        "route",
        "provider-cleanup",
        "child-suppressed",
        "child-cleanup",
    ]
    assert failure.args == ("<redacted>",)
    assert failure.__traceback__ is None


@pytest.mark.asyncio
async def test_function_scoped_settings_child_cleans_up_before_response_start() -> None:
    settings = _settings(max_request_id_length=32)
    events: list[str] = []

    def function_child() -> Iterator[str]:
        events.append("child-enter")
        try:
            yield "child-value"
        finally:
            events.append("child-cleanup")

    def settings_override(
        child_value: Annotated[str, Depends(function_child, scope="function")],
    ) -> Settings:
        assert child_value == "child-value"
        events.append("provider-enter")
        return settings

    app = create_app()
    app.dependency_overrides[get_settings] = settings_override

    @app.get("/_contract/function-settings-child")
    async def function_settings_child() -> dict[str, bool]:
        events.append("route")
        return {"ok": True}

    scope: Scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": "/_contract/function-settings-child",
        "raw_path": b"/_contract/function-settings-child",
        "root_path": "",
        "query_string": b"",
        "headers": [],
        "client": ("127.0.0.1", 1234),
        "server": ("test", 80),
        "state": {},
    }

    async def receive() -> Message:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: Message) -> None:
        if message["type"] == "http.response.start":
            events.append("response-start")

    await app(scope, receive, send)

    assert events == [
        "child-enter",
        "provider-enter",
        "route",
        "child-cleanup",
        "response-start",
    ]


@pytest.mark.asyncio
async def test_function_scoped_settings_child_cannot_suppress_a_nested_cleanup_failure() -> None:
    settings = _settings(max_request_id_length=32)
    failure = _captured_exception(RuntimeError(EXCEPTION_SENTINEL), EXCEPTION_SENTINEL)
    events: list[str] = []

    async def suppressing_child() -> AsyncIterator[str]:
        events.append("suppressor-enter")
        try:
            yield "suppressor-value"
        except Exception:
            events.append("suppressor-suppressed")
            return
        finally:
            events.append("suppressor-cleanup")

    async def failing_child(
        suppressor_value: Annotated[
            str,
            Depends(suppressing_child, scope="function"),
        ],
    ) -> AsyncIterator[str]:
        assert suppressor_value == "suppressor-value"
        events.append("failing-enter")
        try:
            yield "failing-value"
        finally:
            events.append("failing-cleanup")
            raise failure

    def settings_override(
        failing_value: Annotated[str, Depends(failing_child, scope="function")],
    ) -> Settings:
        assert failing_value == "failing-value"
        events.append("provider-enter")
        return settings

    app = create_app()
    app.dependency_overrides[get_settings] = settings_override

    @app.get("/_contract/function-nested-cleanup-suppression")
    async def function_nested_cleanup_suppression() -> dict[str, bool]:
        events.append("route")
        return {"ok": True}

    request_id = "req-function-nested-cleanup"
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=True),
        base_url="http://test",
    ) as client:
        response = await client.get(
            "/_contract/function-nested-cleanup-suppression",
            headers={"X-Request-ID": request_id},
        )

    _assert_error(
        response,
        status_code=500,
        code="INTERNAL_ERROR",
        message="Internal server error",
        request_id=request_id,
    )
    assert events == [
        "suppressor-enter",
        "failing-enter",
        "provider-enter",
        "route",
        "failing-cleanup",
        "suppressor-suppressed",
        "suppressor-cleanup",
    ]
    assert failure.args == ("<redacted>",)
    assert failure.__traceback__ is None


@pytest.mark.asyncio
async def test_function_scoped_settings_cleanup_business_error_uses_shared_handler() -> None:
    settings = _settings(max_request_id_length=32)
    cleanup_error = BusinessError(
        status_code=409,
        code="FUNCTION_SETTINGS_CONFLICT",
        message="Function settings conflict",
        retryable=True,
        headers={"WWW-Authenticate": 'Bearer realm="function-settings"'},
    )

    async def failing_child() -> AsyncIterator[str]:
        try:
            yield "child-value"
        finally:
            raise cleanup_error

    def settings_override(
        child_value: Annotated[str, Depends(failing_child, scope="function")],
    ) -> Settings:
        assert child_value == "child-value"
        return settings

    app = create_app()
    app.dependency_overrides[get_settings] = settings_override

    @app.get("/_contract/function-cleanup-business-error")
    async def function_cleanup_business_error() -> dict[str, bool]:
        return {"ok": True}

    request_id = "req-function-cleanup-business"
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=True),
        base_url="http://test",
    ) as client:
        response = await client.get(
            "/_contract/function-cleanup-business-error",
            headers={"X-Request-ID": request_id},
        )

    _assert_error(
        response,
        status_code=409,
        code="FUNCTION_SETTINGS_CONFLICT",
        message="Function settings conflict",
        request_id=request_id,
        retryable=True,
    )
    assert response.headers["www-authenticate"] == 'Bearer realm="function-settings"'
    assert cleanup_error.args == ("<redacted>",)
    assert cleanup_error.__traceback__ is None


@pytest.mark.asyncio
async def test_function_scoped_settings_cleanup_validation_error_uses_shared_handler() -> None:
    settings = _settings(max_request_id_length=32)
    sentinel = "function-cleanup-validation-secret"
    cleanup_error = RequestValidationError(
        [
            {
                "type": "int_parsing",
                "loc": ("query", "limit"),
                "msg": "Input should be a valid integer",
                "input": sentinel,
            }
        ],
        body={"limit": sentinel},
    )

    async def failing_child() -> AsyncIterator[str]:
        try:
            yield "child-value"
        finally:
            raise cleanup_error

    def settings_override(
        child_value: Annotated[str, Depends(failing_child, scope="function")],
    ) -> Settings:
        assert child_value == "child-value"
        return settings

    app = create_app()
    app.dependency_overrides[get_settings] = settings_override

    @app.get("/_contract/function-cleanup-validation-error")
    async def function_cleanup_validation_error() -> dict[str, bool]:
        return {"ok": True}

    request_id = "req-function-cleanup-validation"
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=True),
        base_url="http://test",
    ) as client:
        response = await client.get(
            "/_contract/function-cleanup-validation-error",
            headers={"X-Request-ID": request_id},
        )

    _assert_error(
        response,
        status_code=422,
        code="VALIDATION_ERROR",
        message="Invalid request",
        request_id=request_id,
    )
    assert cleanup_error.body is None
    assert cleanup_error.errors() == []
    assert cleanup_error.args == ("<redacted>",)
    assert cleanup_error.__traceback__ is None


@pytest.mark.asyncio
async def test_function_scoped_settings_cleanup_base_exception_preserves_identity() -> None:
    settings = _settings(max_request_id_length=32)
    cleanup_error = SystemExit("safe function cleanup exit")

    async def failing_child() -> AsyncIterator[str]:
        try:
            yield "child-value"
        finally:
            raise cleanup_error

    def settings_override(
        child_value: Annotated[str, Depends(failing_child, scope="function")],
    ) -> Settings:
        assert child_value == "child-value"
        return settings

    app = create_app()
    app.dependency_overrides[get_settings] = settings_override

    @app.get("/_contract/function-cleanup-base-exception")
    async def function_cleanup_base_exception() -> dict[str, bool]:
        return {"ok": True}

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=True),
        base_url="http://test",
    ) as client:
        with pytest.raises(SystemExit) as raised:
            await client.get("/_contract/function-cleanup-base-exception")

    assert raised.value is cleanup_error
    assert cleanup_error.args == ("safe function cleanup exit",)


@pytest.mark.asyncio
async def test_settings_provider_business_error_uses_the_shared_error_envelope() -> None:
    error = BusinessError(
        status_code=429,
        code="SETTINGS_RATE_LIMITED",
        message="Settings provider unavailable",
        retryable=True,
        headers={
            "WWW-Authenticate": 'Bearer realm="settings"',
            "X-Unsafe-Header": EXCEPTION_SENTINEL,
        },
    )

    def settings_override() -> Settings:
        provider_secret = EXCEPTION_SENTINEL
        assert provider_secret
        raise error

    app = create_app()
    app.dependency_overrides[get_settings] = settings_override
    request_id = "req-settings-business-error"
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=True),
        base_url="http://test",
    ) as client:
        response = await client.get(
            "/health",
            headers={"X-Request-ID": request_id},
        )

    _assert_error(
        response,
        status_code=429,
        code="SETTINGS_RATE_LIMITED",
        message="Settings provider unavailable",
        request_id=request_id,
        retryable=True,
    )
    assert response.headers["www-authenticate"] == 'Bearer realm="settings"'
    assert "x-unsafe-header" not in response.headers
    assert error.args == ("<redacted>",)
    assert error.__traceback__ is None


@pytest.mark.asyncio
async def test_settings_error_response_builder_base_exception_uses_safe_internal_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = BusinessError(
        status_code=409,
        code="SETTINGS_CONFLICT",
        message="Settings conflict",
    )
    builder_failure = SystemExit(EXCEPTION_SENTINEL)

    def settings_override() -> Settings:
        raise error

    def failing_builder(_error: BusinessError, _request_id: str) -> None:
        raise builder_failure

    monkeypatch.setattr(
        "rag_service.api.middleware.business_error_response",
        failing_builder,
    )
    app = create_app()
    app.dependency_overrides[get_settings] = settings_override
    request_id = "req-settings-builder-failure"
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=True),
        base_url="http://test",
    ) as client:
        response = await client.get(
            "/health",
            headers={"X-Request-ID": request_id},
        )

    _assert_error(
        response,
        status_code=500,
        code="INTERNAL_ERROR",
        message="Internal server error",
        request_id=request_id,
    )
    assert error.args == ("<redacted>",)
    assert error.__traceback__ is None
    assert builder_failure.args == ("<redacted>",)
    assert builder_failure.__traceback__ is None


@pytest.mark.asyncio
async def test_validation_error_is_scrubbed_before_a_failing_response_builder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = "validation-builder-input-secret"
    error = RequestValidationError(
        [
            {
                "type": "int_parsing",
                "loc": ("query", "limit"),
                "msg": "Input should be a valid integer",
                "input": sentinel,
            }
        ],
        body={"limit": sentinel},
    )
    builder_failure = SystemExit(EXCEPTION_SENTINEL)

    def settings_override() -> Settings:
        raise error

    def failing_builder(builder_error: BaseException, _request_id: str) -> None:
        assert isinstance(builder_error, RequestValidationError)
        assert builder_error.body is None
        assert builder_error.errors() == []
        assert builder_error.args == ("<redacted>",)
        assert builder_error.__traceback__ is None
        raise builder_failure

    monkeypatch.setattr(
        "rag_service.api.middleware.validation_error_response",
        failing_builder,
    )
    app = create_app()
    app.dependency_overrides[get_settings] = settings_override
    request_id = "req-validation-builder-failure"
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=True),
        base_url="http://test",
    ) as client:
        response = await client.get(
            "/health",
            headers={"X-Request-ID": request_id},
        )

    _assert_error(
        response,
        status_code=500,
        code="INTERNAL_ERROR",
        message="Internal server error",
        request_id=request_id,
    )
    assert error.body is None
    assert error.errors() == []
    assert error.args == ("<redacted>",)
    assert error.__traceback__ is None
    assert builder_failure.args == ("<redacted>",)
    assert builder_failure.__traceback__ is None


@pytest.mark.asyncio
async def test_settings_child_validation_error_uses_the_shared_422_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(max_request_id_length=32)
    created_errors: list[RequestValidationError] = []

    class CapturingRequestValidationError(RequestValidationError):
        def __init__(self, errors: list[dict[str, object]]) -> None:
            super().__init__(errors)
            created_errors.append(self)

    monkeypatch.setattr(
        "rag_service.main.RequestValidationError",
        CapturingRequestValidationError,
    )

    def query_child(limit: Annotated[int, Query(gt=0)]) -> int:
        return limit

    def settings_override(
        _limit: Annotated[int, Depends(query_child)],
    ) -> Settings:
        return settings

    app = create_app()
    app.dependency_overrides[get_settings] = settings_override
    request_id = "req-settings-validation-error"
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=True),
        base_url="http://test",
    ) as client:
        response = await client.get(
            "/health",
            headers={"X-Request-ID": request_id},
        )

    _assert_error(
        response,
        status_code=422,
        code="VALIDATION_ERROR",
        message="Invalid request",
        request_id=request_id,
    )
    assert len(created_errors) == 1
    assert created_errors[0].body is None
    assert created_errors[0].errors() == []
    assert created_errors[0].args == ("<redacted>",)
    assert created_errors[0].__traceback__ is None


@pytest.mark.asyncio
async def test_settings_dependency_background_tasks_are_rejected_before_provider_execution() -> (
    None
):
    settings = _settings(max_request_id_length=32)
    provider_calls = 0

    def settings_override(background_tasks: BackgroundTasks) -> Settings:
        nonlocal provider_calls
        provider_calls += 1
        background_tasks.add_task(lambda: None)
        return settings

    app = create_app()
    app.dependency_overrides[get_settings] = settings_override
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=True),
        base_url="http://test",
    ) as client:
        response = await client.get("/health")

    assert response.status_code == 500
    assert provider_calls == 0


@pytest.mark.asyncio
async def test_settings_dependency_response_is_rejected_before_provider_execution() -> None:
    settings = _settings(max_request_id_length=32)
    provider_calls = 0

    def settings_override(response: Response) -> Settings:
        nonlocal provider_calls
        provider_calls += 1
        response.headers["X-Provider-Ran"] = "true"
        return settings

    app = create_app()
    app.dependency_overrides[get_settings] = settings_override
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=True),
        base_url="http://test",
    ) as client:
        response = await client.get("/health")

    assert response.status_code == 500
    assert "x-provider-ran" not in response.headers
    assert provider_calls == 0


@pytest.mark.asyncio
async def test_settings_yield_cleanup_failure_uses_the_safe_response_started_boundary() -> None:
    settings = _settings(max_request_id_length=32)
    events: list[str] = []

    async def settings_override(request: Request) -> AsyncIterator[Settings]:
        events.append(f"enter:{request.url.path}")
        try:
            yield settings
        finally:
            cleanup_secret = EXCEPTION_SENTINEL
            events.append("cleanup")
            assert cleanup_secret
            raise RuntimeError(EXCEPTION_SENTINEL)

    app = create_app()
    app.dependency_overrides[get_settings] = settings_override

    @app.get("/_contract/failing-settings-cleanup")
    async def failing_settings_cleanup(
        route_settings: Annotated[Settings, Depends(get_settings)],
    ) -> dict[str, bool]:
        events.append("route")
        return {"same_settings": route_settings is settings}

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=True),
        base_url="http://test",
    ) as client:
        with pytest.raises(RuntimeError) as raised:
            await client.get(
                "/_contract/failing-settings-cleanup",
                headers={"X-Request-ID": "request-cleanup-failure"},
            )

    assert raised.value.args == ("Request processing failed",)
    assert events == [
        "enter:/_contract/failing-settings-cleanup",
        "route",
        "cleanup",
    ]
    for node in _exception_nodes(raised.value):
        assert EXCEPTION_SENTINEL not in repr(node.args)
        traceback = node.__traceback__
        while traceback is not None:
            module_name = traceback.tb_frame.f_globals.get("__name__", "")
            if isinstance(module_name, str) and module_name.startswith("rag_service"):
                frame_locals = traceback.tb_frame.f_locals
                assert EXCEPTION_SENTINEL not in repr(frame_locals)
                assert not {
                    "scope",
                    "request",
                    "route_settings",
                    "provided_settings",
                    "request_settings",
                    "request_settings_stack",
                    "function_stack",
                    "function_settings_stack",
                    "function_stack_controller",
                    "pending_error",
                    "error_response",
                    "active_exception",
                    "dependant",
                    "solved",
                    "state",
                    "selected_provider",
                }.intersection(frame_locals)
            traceback = traceback.tb_next


@pytest.mark.asyncio
async def test_settings_cleanup_business_error_after_response_uses_safe_wrapper() -> None:
    settings = _settings(max_request_id_length=32)
    cleanup_error = BusinessError(
        status_code=409,
        code="SETTINGS_CLEANUP_CONFLICT",
        message="Settings cleanup conflict",
    )

    async def settings_override() -> AsyncIterator[Settings]:
        try:
            yield settings
        finally:
            raise cleanup_error

    app = create_app()
    app.dependency_overrides[get_settings] = settings_override

    @app.get("/_contract/settings-cleanup-business-error")
    async def settings_cleanup_business_error() -> dict[str, bool]:
        return {"ok": True}

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=True),
        base_url="http://test",
    ) as client:
        with pytest.raises(RuntimeError) as raised:
            await client.get("/_contract/settings-cleanup-business-error")

    assert raised.value.args == ("Request processing failed",)
    assert cleanup_error.args == ("<redacted>",)
    assert cleanup_error.__traceback__ is None


@pytest.mark.asyncio
async def test_settings_yield_cleanup_cancellation_clears_request_scope_state() -> None:
    settings = _settings(max_request_id_length=32)
    cancellation = asyncio.CancelledError("safe settings cleanup cancellation")
    events: list[str] = []
    seen_scopes: list[Scope] = []

    async def settings_override(request: Request) -> AsyncIterator[Settings]:
        seen_scopes.append(request.scope)
        events.append("enter")
        try:
            yield settings
        finally:
            events.append("cleanup")
            raise cancellation

    app = create_app()
    app.dependency_overrides[get_settings] = settings_override

    @app.get("/_contract/cancelled-settings-cleanup")
    async def cancelled_settings_cleanup() -> dict[str, bool]:
        events.append("route")
        return {"ok": True}

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=True),
        base_url="http://test",
    ) as client:
        with pytest.raises(asyncio.CancelledError) as raised:
            await client.get("/_contract/cancelled-settings-cleanup")

    assert raised.value is cancellation
    assert raised.value.args == ("safe settings cleanup cancellation",)
    assert events == ["enter", "route", "cleanup"]
    assert len(seen_scopes) == 1
    state = seen_scopes[0]["state"]
    assert isinstance(state, dict)
    assert "_rag_runtime_settings" not in state


@pytest.mark.asyncio
async def test_settings_override_body_dependency_is_rejected_without_reading_request_body() -> None:
    settings = _settings(max_request_id_length=32)
    provider_calls = 0
    route_calls = 0

    def settings_override(
        payload: Annotated[dict[str, str] | None, Body()] = None,
    ) -> Settings:
        nonlocal provider_calls
        provider_calls += 1
        assert payload is not None
        return settings

    app = create_app()
    app.dependency_overrides[get_settings] = settings_override

    @app.post("/_contract/body-settings-override")
    async def body_settings_override(payload: dict[str, str]) -> dict[str, str]:
        nonlocal route_calls
        route_calls += 1
        return payload

    body = b'{"value":"request-body-secret"}'
    scope: Scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/_contract/body-settings-override",
        "raw_path": b"/_contract/body-settings-override",
        "root_path": "",
        "query_string": b"",
        "headers": [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode("ascii")),
        ],
        "client": ("127.0.0.1", 1234),
        "server": ("test", 80),
        "state": {},
    }
    receive_calls = 0
    sent: list[Message] = []

    async def receive() -> Message:
        nonlocal receive_calls
        receive_calls += 1
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(message: Message) -> None:
        sent.append(message)

    await app(scope, receive, send)

    response_start = next(message for message in sent if message["type"] == "http.response.start")
    assert response_start["status"] == 500
    assert provider_calls == 0
    assert route_calls == 0
    assert receive_calls == 0


@pytest.mark.asyncio
async def test_invalid_missing_and_duplicate_request_ids_are_replaced_safely() -> None:
    settings = _settings(max_request_id_length=32)
    app = _app(settings)

    @app.get("/_contract/request-id")
    async def request_id(request: Request) -> dict[str, str]:
        return {"request_id": cast(str, request.state.request_id)}

    headers_to_replace: tuple[list[tuple[str, str]] | None, ...] = (
        None,
        [("X-Request-ID", "")],
        [("X-Request-ID", "unsafe request id")],
        [("X-Request-ID", "unsafe\trequest")],
        [("X-Request-ID", "x" * 33)],
        [("X-Request-ID", "first"), ("X-Request-ID", "second")],
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        responses = [
            await client.get("/_contract/request-id", headers=headers)
            for headers in headers_to_replace
        ]

    for response in responses:
        generated = response.headers["x-request-id"]
        assert REQUEST_ID_PATTERN.fullmatch(generated)
        assert len(generated) <= settings.max_request_id_length
        assert response.json() == {"request_id": generated}


class _AdminFailureService:
    async def list_agent_keys(self, *, cursor: str | None, limit: int | None) -> None:
        del cursor, limit
        raise BusinessError(
            403,
            "INSUFFICIENT_CAPABILITY",
            "Insufficient capability",
            headers={
                "WWW-Authenticate": "Bearer",
                "Cache-Control": "public",
                "X-Request-ID": "unsafe-downstream-request-id",
                "X-Internal-Secret": EXCEPTION_SENTINEL,
            },
        )


@pytest.mark.asyncio
async def test_admin_local_boundary_cannot_omit_the_global_error_request_id() -> None:
    app = _app()
    app.dependency_overrides[require_admin_principal] = lambda: AdminPrincipal(
        key_id=UUID(int=1),
        public_id="AAAAAAAAAAAAAAAA",
    )
    app.dependency_overrides[get_api_key_service] = lambda: cast(
        ApiKeyService,
        _AdminFailureService(),
    )
    request_id = "req-admin-global-error"

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.get(
            "/v1/admin/api-keys",
            headers={"X-Request-ID": request_id},
        )

    _assert_error(
        response,
        status_code=403,
        code="INSUFFICIENT_CAPABILITY",
        message="Insufficient capability",
        request_id=request_id,
    )
    assert response.headers["www-authenticate"] == "Bearer"
    assert "x-internal-secret" not in response.headers
    assert EXCEPTION_SENTINEL not in response.text


@pytest.mark.asyncio
async def test_business_errors_share_one_exact_envelope_and_header_policy() -> None:
    app = _app()
    cases = (
        (401, "INVALID_API_KEY", "Invalid API key"),
        (403, "INSUFFICIENT_CAPABILITY", "Insufficient capability"),
        (404, "RESOURCE_NOT_FOUND", "Resource not found"),
        (409, "IDEMPOTENCY_CONFLICT", "Idempotency key conflict"),
        (409, "RESOURCE_STATE_CONFLICT", "Resource state conflict"),
        (412, "PRECONDITION_FAILED", "Precondition failed"),
    )

    @app.get("/_contract/business/{case_index}")
    async def business_error(case_index: int) -> None:
        status_code, code, message = cases[case_index]
        headers = {
            "WWW-Authenticate": "Bearer",
            "Cache-Control": "public",
            "X-Request-ID": "unsafe-overridden-request-id",
            "X-Internal-Secret": EXCEPTION_SENTINEL,
            "X-Unsafe-Newline": "unsafe\r\nvalue",
        }
        raise BusinessError(status_code, code, message, headers=headers)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        responses = [
            await client.get(
                f"/_contract/business/{index}",
                headers={"X-Request-ID": f"req-business-{index}"},
            )
            for index in range(len(cases))
        ]

    for index, (response, (status_code, code, message)) in enumerate(
        zip(responses, cases, strict=True)
    ):
        _assert_error(
            response,
            status_code=status_code,
            code=code,
            message=message,
            request_id=f"req-business-{index}",
        )
        assert response.headers["www-authenticate"] == "Bearer"
        assert "x-internal-secret" not in response.headers
        assert "x-unsafe-newline" not in response.headers


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "challenge",
    [
        "Bearer unicode-\u2603",
        "Bearer nul-\x00",
        "Bearer unit-separator-\x1f",
        "Bearer delete-\x7f",
        "Bearer c1-control-\x85",
        "B" * 129,
    ],
)
async def test_invalid_www_authenticate_values_are_dropped_without_changing_error(
    challenge: str,
) -> None:
    app = _app()

    @app.get("/_contract/unsafe-authenticate")
    async def unsafe_authenticate() -> None:
        raise BusinessError(
            401,
            "INVALID_API_KEY",
            "Invalid API key",
            headers={"WWW-Authenticate": challenge},
        )

    request_id = "req-invalid-authenticate"
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.get(
            "/_contract/unsafe-authenticate",
            headers={"X-Request-ID": request_id},
        )

    _assert_error(
        response,
        status_code=401,
        code="INVALID_API_KEY",
        message="Invalid API key",
        request_id=request_id,
    )
    assert "www-authenticate" not in response.headers


class _ContractBody(BaseModel):
    count: int


@pytest.mark.asyncio
async def test_request_validation_is_generic_for_custom_admin_kb_and_document_routes() -> None:
    app = _app()

    @app.post("/_contract/validation")
    async def validation(body: _ContractBody) -> _ContractBody:
        return body

    app.dependency_overrides[require_agent_principal] = _agent
    app.dependency_overrides[get_knowledge_base_service] = lambda: cast(
        KnowledgeBaseService,
        object(),
    )
    app.dependency_overrides[get_document_metadata_service] = lambda: cast(
        DocumentMetadataService,
        object(),
    )
    requests = (
        ("POST", "/_contract/validation", {"count": "invalid-body-contract-sentinel"}),
        ("GET", "/v1/knowledge-bases/not-a-uuid", None),
        ("GET", "/v1/documents/not-a-uuid", None),
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        responses = [
            await client.request(
                method,
                path,
                json=body,
                headers={"X-Request-ID": f"req-validation-{index}"},
            )
            for index, (method, path, body) in enumerate(requests)
        ]

    for index, response in enumerate(responses):
        _assert_error(
            response,
            status_code=422,
            code="VALIDATION_ERROR",
            message="Invalid request",
            request_id=f"req-validation-{index}",
        )
        assert "invalid-body-contract-sentinel" not in response.text
        assert "RequestValidationError" not in response.text


def test_request_validation_error_does_not_retain_body_or_error_inputs() -> None:
    sentinel = "retained-validation-input-secret"
    error = RequestValidationError(
        [
            {
                "type": "int_parsing",
                "loc": ("body", "count"),
                "msg": "Input should be a valid integer",
                "input": sentinel,
            }
        ],
        body={"count": sentinel},
    )
    error = cast(RequestValidationError, _captured_exception(error, sentinel))

    response = validation_error_response(error, "req-retained-validation")

    assert response.status_code == 422
    assert error.body is None
    assert error.errors() == []
    assert sentinel not in repr(error.__dict__)
    assert sentinel not in str(error)
    assert error.args == ("<redacted>",)
    assert error.__traceback__ is None


def test_hostile_request_validation_error_is_scrubbed_without_leaking_scrub_failures() -> None:
    sentinel = "hostile-validation-input-secret"
    scrub_failure = SystemExit("hostile-validation-scrub-secret")

    class HostileRequestValidationError(RequestValidationError):
        def __init__(self) -> None:
            object.__setattr__(self, "_locked", False)
            super().__init__(
                [
                    {
                        "type": "int_parsing",
                        "loc": ("body", "count"),
                        "msg": "Input should be a valid integer",
                        "input": sentinel,
                    }
                ],
                body={"count": sentinel},
            )
            object.__setattr__(self, "_locked", True)

        def __setattr__(self, name: str, value: object) -> None:
            if object.__getattribute__(self, "__dict__").get("_locked") and name in {
                "body",
                "_errors",
                "args",
                "__traceback__",
                "__cause__",
                "__context__",
            }:
                raise scrub_failure
            super().__setattr__(name, value)

    error: RequestValidationError = HostileRequestValidationError()
    error = cast(RequestValidationError, _captured_exception(error, sentinel))

    response = validation_error_response(error, "req-hostile-validation")

    assert response.status_code == 422
    assert object.__getattribute__(error, "body") is None
    assert RequestValidationError.errors(error) == []
    assert BaseException.__getattribute__(error, "args") == ("<redacted>",)
    assert BaseException.__getattribute__(error, "__traceback__") is None
    assert scrub_failure.__traceback__ is None


@pytest.mark.asyncio
async def test_nested_request_validation_error_is_scrubbed_in_an_unhandled_graph() -> None:
    sentinel = "nested-validation-input-secret"
    validation_error = RequestValidationError(
        [
            {
                "type": "int_parsing",
                "loc": ("body", "count"),
                "msg": "Input should be a valid integer",
                "input": sentinel,
            }
        ],
        body={"count": sentinel},
    )
    validation_error = cast(
        RequestValidationError,
        _captured_exception(validation_error, sentinel),
    )
    failure = ExceptionGroup("nested validation failure", [validation_error])
    app = _app()

    @app.get("/_contract/nested-validation-failure")
    async def nested_validation_failure() -> None:
        raise failure

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=True),
        base_url="http://test",
    ) as client:
        response = await client.get("/_contract/nested-validation-failure")

    assert response.status_code == 500
    assert validation_error.body is None
    assert validation_error.errors() == []
    assert validation_error.args == ("<redacted>",)
    assert validation_error.__traceback__ is None


@pytest.mark.asyncio
async def test_unexpected_exception_returns_safe_500_and_safe_log_with_raise_enabled(
    caplog: pytest.LogCaptureFixture,
) -> None:
    settings = _settings()
    app = _app(settings)

    child = _captured_exception(RuntimeError(EXCEPTION_SENTINEL), EXCEPTION_SENTINEL)
    cause = _captured_exception(ValueError(DATABASE_SENTINEL), DATABASE_SENTINEL)
    nested = ExceptionGroup("safe nested group", [child])
    failure = ExceptionGroup("safe root group", [nested])
    child.__cause__ = cause
    child.__context__ = cause
    retained_nodes = (failure, nested, child, cause)

    async def failing_dependency(request: Request) -> None:
        authorization = request.headers.get("authorization")
        hmac_secret = settings.admin_key_hmac_secret.get_secret_value()
        database_error = DATABASE_SENTINEL
        assert authorization and hmac_secret and database_error
        raise failure

    @app.get("/_contract/unexpected", dependencies=[Depends(failing_dependency)])
    async def unexpected() -> None:
        raise AssertionError("dependency must fail first")

    request_id = "req-unexpected-global-error"
    caplog.set_level(logging.ERROR, logger="rag_service.api.middleware")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=True),
        base_url="http://test",
    ) as client:
        response = await client.get(
            "/_contract/unexpected",
            headers={
                "Authorization": f"Bearer {AUTHORIZATION_SENTINEL}",
                "X-Request-ID": request_id,
            },
        )

    _assert_error(
        response,
        status_code=500,
        code="INTERNAL_ERROR",
        message="Internal server error",
        request_id=request_id,
    )
    records = [record for record in caplog.records if record.name == "rag_service.api.middleware"]
    assert len(records) == 1
    record = records[0]
    assert record.getMessage() == "Unhandled application exception"
    assert record.args == ()
    assert record.exc_info is None
    assert record.stack_info is None
    assert record.__dict__["request_id"] == request_id
    assert record.__dict__["exception_type"] == "ExceptionGroup"
    retained = " ".join(
        (response.text, record.getMessage(), repr(record.args), repr(record.__dict__))
    )
    for sentinel in (
        AUTHORIZATION_SENTINEL,
        HMAC_SENTINEL,
        DATABASE_SENTINEL,
        EXCEPTION_SENTINEL,
    ):
        assert sentinel not in retained
    for node in retained_nodes:
        assert node.args == ("<redacted>",)
        assert node.__traceback__ is None
        assert node.__cause__ is None
        assert node.__context__ is None
        traceback: TracebackType | None = node.__traceback__
        while traceback is not None:
            assert EXCEPTION_SENTINEL not in repr(traceback.tb_frame.f_locals)
            traceback = traceback.tb_next


@pytest.mark.asyncio
async def test_malicious_dynamic_exception_type_is_not_logged() -> None:
    app = _app()
    malicious_type = type(EXCEPTION_SENTINEL, (RuntimeError,), {})

    @app.get("/_contract/malicious-exception-type")
    async def malicious_exception_type() -> None:
        raise malicious_type("safe args")

    request_id = "req-malicious-exception-type"
    logger = logging.getLogger("rag_service.api.middleware")
    records: list[logging.LogRecord] = []

    class _Handler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = _Handler()
    logger.addHandler(handler)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, raise_app_exceptions=True),
            base_url="http://test",
        ) as client:
            response = await client.get(
                "/_contract/malicious-exception-type",
                headers={"X-Request-ID": request_id},
            )
    finally:
        logger.removeHandler(handler)

    _assert_error(
        response,
        status_code=500,
        code="INTERNAL_ERROR",
        message="Internal server error",
        request_id=request_id,
    )
    assert len(records) == 1
    assert records[0].__dict__["exception_type"] == "Exception"
    assert EXCEPTION_SENTINEL not in repr(records[0].__dict__)


def test_openapi_uses_shared_error_envelopes_and_role_scoped_bearer_security() -> None:
    document = _app().openapi()
    for path in ("/health", "/ready", "/ready/ingest", "/ready/retrieve", "/ready/answer"):
        assert document["paths"][path]["get"].get("security") is None

    for path, path_item in document["paths"].items():
        if not path.startswith("/v1/"):
            continue
        for operation in path_item.values():
            assert operation["security"] == [{"HTTPBearer": []}]
            for status_code, response in operation["responses"].items():
                if int(status_code) < 400:
                    continue
                schema = response["content"]["application/json"]["schema"]
                assert schema == {"$ref": "#/components/schemas/ErrorEnvelope"}

    assert "HTTPValidationError" not in document["components"]["schemas"]
