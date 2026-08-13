import asyncio
import threading
import traceback
from collections.abc import Awaitable, Callable
from contextlib import AbstractAsyncContextManager
from pathlib import Path
from typing import Any

import pytest
from urllib3 import PoolManager

from fixtures import provider_stub

AsyncStep = tuple[str, Callable[[], Awaitable[None]]]
SyncStep = tuple[str, Callable[[], None]]
AsyncLifecycle = Callable[
    [tuple[AsyncStep, ...], Callable[[], None]],
    AbstractAsyncContextManager[None],
]
SyncTeardown = Callable[[tuple[SyncStep, ...], Callable[[], None]], None]
FreshCleanup = Callable[..., None]
SyncLifecycle = Callable[..., Any]
ServiceContainerLifecycle = Callable[..., Any]
MinioConnectionLifecycle = Callable[[Any], Any]
MinioHttpPoolFactory = Callable[[], PoolManager]


def _body_secret_error() -> LookupError:
    return LookupError("synthetic-body-password and SQL value")


def _leaf_exceptions(error: BaseException) -> list[BaseException]:
    if isinstance(error, BaseExceptionGroup):
        return [leaf for child in error.exceptions for leaf in _leaf_exceptions(child)]
    return [error]


def _exception_debug_rendering(error: BaseException) -> str:
    rendered = ["".join(traceback.format_exception(error))]
    pending = [error]
    visited: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in visited:
            continue
        visited.add(id(current))
        rendered.extend((repr(current), repr(current.__cause__), repr(current.__context__)))
        current_traceback = current.__traceback__
        while current_traceback is not None:
            rendered.extend(repr(value) for value in current_traceback.tb_frame.f_locals.values())
            current_traceback = current_traceback.tb_next
        if isinstance(current, BaseExceptionGroup):
            pending.extend(current.exceptions)
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        if current.__context__ is not None:
            pending.append(current.__context__)
    return "\n".join(rendered)


def _assert_no_sync_secret_is_reachable(error: BaseException) -> None:
    rendered = _exception_debug_rendering(error)
    for marker in ("synthetic", "password", "SQL value", "dsn://"):
        assert marker not in rendered


@pytest.mark.integration
def test_sync_lifecycle_sanitizes_setup_and_fresh_cleanup_failures(
    fixture_sync_connection_lifecycle: SyncLifecycle,
) -> None:
    calls = {"setup": 0, "cleanup": 0}

    def setup_connect(_url: str, *, autocommit: bool) -> Any:
        assert autocommit is False
        calls["setup"] += 1
        raise ConnectionError("synthetic-setup-dsn://user:setup-password@host/database")

    def cleanup_connect(_url: str, *, autocommit: bool) -> Any:
        assert autocommit is True
        calls["cleanup"] += 1
        raise OSError("synthetic-cleanup-dsn://user:cleanup-password@host/database")

    lifecycle = fixture_sync_connection_lifecycle(
        "synthetic-setup-dsn://user:setup-password@host/database",
        autocommit=False,
        rollback_on_exit=True,
        connect=setup_connect,
        cleanup_connect=cleanup_connect,
    )

    with pytest.raises(ExceptionGroup) as captured:
        next(lifecycle)

    assert calls == {"setup": 1, "cleanup": 1}
    assert [str(error) for error in captured.value.exceptions] == [
        "connection.setup primary failed (ConnectionError)",
        "increment_2.cleanup_connection teardown failed (OSError)",
    ]
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
    _assert_no_sync_secret_is_reachable(captured.value)


@pytest.mark.integration
def test_sync_lifecycle_sanitizes_body_and_all_teardown_failures(
    fixture_sync_lifecycle: SyncLifecycle,
) -> None:
    events: list[str] = []

    def teardown_steps(_resource: object) -> tuple[SyncStep, ...]:
        def fail(resource: str) -> None:
            events.append(resource)
            raise RuntimeError(f"synthetic-{resource}-password")

        return (
            ("connection.rollback", lambda: fail("rollback")),
            ("connection.close", lambda: fail("close")),
        )

    def cleanup() -> None:
        events.append("fresh_cleanup")
        raise ValueError("synthetic-cleanup SQL value and password")

    lifecycle = fixture_sync_lifecycle(
        setup_resource="connection.setup",
        setup=object,
        teardown_steps=teardown_steps,
        cleanup=cleanup,
    )
    assert next(lifecycle) is not None

    with pytest.raises(ExceptionGroup) as captured:
        lifecycle.throw(_body_secret_error())

    assert events == ["rollback", "close", "fresh_cleanup"]
    assert [str(error) for error in captured.value.exceptions] == [
        "fixture.body primary failed (LookupError)",
        "connection.rollback teardown failed (RuntimeError)",
        "connection.close teardown failed (RuntimeError)",
        "increment_2.fresh_cleanup teardown failed (ValueError)",
    ]
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
    _assert_no_sync_secret_is_reachable(captured.value)


@pytest.mark.integration
def test_sync_lifecycle_reports_only_teardown_failures(
    fixture_sync_lifecycle: SyncLifecycle,
) -> None:
    events: list[str] = []

    def teardown_steps(_resource: object) -> tuple[SyncStep, ...]:
        def close() -> None:
            events.append("close")
            raise RuntimeError("synthetic-close-password")

        return (("connection.close", close),)

    lifecycle = fixture_sync_lifecycle(
        setup_resource="connection.setup",
        setup=object,
        teardown_steps=teardown_steps,
        cleanup=lambda: events.append("fresh_cleanup"),
    )
    assert next(lifecycle) is not None

    with pytest.raises(ExceptionGroup) as captured:
        next(lifecycle)

    assert events == ["close", "fresh_cleanup"]
    assert [str(error) for error in captured.value.exceptions] == [
        "connection.close teardown failed (RuntimeError)"
    ]
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
    _assert_no_sync_secret_is_reachable(captured.value)


@pytest.mark.integration
def test_sync_lifecycle_success_runs_each_step_once(
    fixture_sync_lifecycle: SyncLifecycle,
) -> None:
    events: list[str] = []
    resource = object()

    def setup() -> object:
        events.append("setup")
        return resource

    def teardown_steps(_resource: object) -> tuple[SyncStep, ...]:
        return (
            ("connection.rollback", lambda: events.append("rollback")),
            ("connection.close", lambda: events.append("close")),
        )

    lifecycle = fixture_sync_lifecycle(
        setup_resource="connection.setup",
        setup=setup,
        teardown_steps=teardown_steps,
        cleanup=lambda: events.append("fresh_cleanup"),
    )
    assert next(lifecycle) is resource

    with pytest.raises(StopIteration):
        next(lifecycle)

    assert events == ["setup", "rollback", "close", "fresh_cleanup"]


@pytest.mark.asyncio
@pytest.mark.integration
async def test_async_lifecycle_attempts_every_close_and_preserves_primary_error(
    fixture_async_lifecycle: AsyncLifecycle,
) -> None:
    events: list[str] = []

    async def failing_step(resource: str) -> None:
        events.append(resource)
        raise RuntimeError("postgresql://user:password@host/sensitive")

    def failing_cleanup() -> None:
        events.append("fresh_cleanup")
        raise ValueError("secret cleanup SQL value")

    steps: tuple[AsyncStep, ...] = (
        ("db_session.rollback", lambda: failing_step("rollback")),
        ("db_session.context_close", lambda: failing_step("session_close")),
        ("database.close", lambda: failing_step("database_close")),
    )

    with pytest.raises(BaseExceptionGroup) as captured:
        async with fixture_async_lifecycle(steps, failing_cleanup):
            events.append("body")
            raise LookupError("primary test failure remains visible")

    assert events == [
        "body",
        "rollback",
        "session_close",
        "database_close",
        "fresh_cleanup",
    ]
    leaves = _leaf_exceptions(captured.value)
    assert any(
        isinstance(error, LookupError) and str(error) == "primary test failure remains visible"
        for error in leaves
    )
    teardown_messages = [str(error) for error in leaves if not isinstance(error, LookupError)]
    assert teardown_messages == [
        "db_session.rollback teardown failed (RuntimeError)",
        "db_session.context_close teardown failed (RuntimeError)",
        "database.close teardown failed (RuntimeError)",
        "increment_2.fresh_cleanup teardown failed (ValueError)",
    ]
    rendered = "".join(traceback.format_exception(captured.value))
    assert "postgresql://" not in rendered
    assert "password" not in rendered
    assert "secret cleanup SQL value" not in rendered


@pytest.mark.integration
def test_sync_lifecycle_attempts_rollback_close_and_cleanup_once(
    fixture_sync_teardown: SyncTeardown,
) -> None:
    events: list[str] = []

    def failing_step(resource: str) -> None:
        events.append(resource)
        raise RuntimeError("password=sensitive")

    def cleanup() -> None:
        events.append("fresh_cleanup")

    with pytest.raises(ExceptionGroup) as captured:
        fixture_sync_teardown(
            (
                ("connection.rollback", lambda: failing_step("rollback")),
                ("connection.close", lambda: failing_step("close")),
            ),
            cleanup,
        )

    assert events == ["rollback", "close", "fresh_cleanup"]
    assert [str(error) for error in captured.value.exceptions] == [
        "connection.rollback teardown failed (RuntimeError)",
        "connection.close teardown failed (RuntimeError)",
    ]
    rendered = "".join(traceback.format_exception(captured.value))
    assert "password=sensitive" not in rendered


@pytest.mark.integration
def test_fresh_cleanup_closes_connection_after_execute_failure_without_leaking_secrets(
    fixture_fresh_cleanup: FreshCleanup,
) -> None:
    events: list[str] = []

    class FailingCleanupConnection:
        def execute(self, _statement: str) -> None:
            events.append("execute")
            raise RuntimeError("TRUNCATE value with password=sensitive")

        def close(self) -> None:
            events.append("close")
            raise ValueError("postgresql://user:password@host/database")

    def connect(_url: str, *, autocommit: bool) -> Any:
        events.append(f"connect:{autocommit}")
        return FailingCleanupConnection()

    with pytest.raises(ExceptionGroup) as captured:
        fixture_fresh_cleanup(
            "postgresql://user:password@host/database",
            connect=connect,
        )

    assert events == ["connect:True", "execute", "close"]
    assert [str(error) for error in captured.value.exceptions] == [
        "increment_2.truncate teardown failed (RuntimeError)",
        "increment_2.cleanup_connection.close teardown failed (ValueError)",
    ]
    rendered = "".join(traceback.format_exception(captured.value))
    assert "TRUNCATE value" not in rendered
    assert "postgresql://" not in rendered
    assert "password" not in rendered


@pytest.mark.integration
def test_service_container_lifecycle_stops_partial_setup_and_sanitizes_failure(
    fixture_service_container_lifecycle: ServiceContainerLifecycle,
) -> None:
    events: list[str] = []

    class Container:
        def start(self) -> None:
            events.append("start")

        def stop(self) -> None:
            events.append("stop")

    def connection(_container: object) -> object:
        events.append("connection")
        raise RuntimeError("redis://user:synthetic-password@host/0")

    lifecycle = fixture_service_container_lifecycle(
        resource="redis",
        container=Container(),
        connection=connection,
        readiness=lambda _connection: True,
    )
    with pytest.raises(ExceptionGroup) as captured:
        next(lifecycle)

    assert events == ["start", "connection", "stop"]
    assert [str(error) for error in captured.value.exceptions] == [
        "redis.setup primary failed (RuntimeError)"
    ]
    _assert_no_sync_secret_is_reachable(captured.value)


@pytest.mark.integration
def test_service_container_lifecycle_preserves_setup_cancellation_and_still_stops(
    fixture_service_container_lifecycle: ServiceContainerLifecycle,
) -> None:
    events: list[str] = []

    class Container:
        def start(self) -> None:
            events.append("start")
            raise asyncio.CancelledError

        def stop(self) -> None:
            events.append("stop")

    lifecycle = fixture_service_container_lifecycle(
        resource="redis",
        container=Container(),
        connection=lambda _container: object(),
        readiness=lambda _connection: True,
    )

    with pytest.raises(asyncio.CancelledError):
        next(lifecycle)

    assert events == ["start", "stop"]


@pytest.mark.integration
def test_minio_fixture_traceback_does_not_retain_credentials(
    fixture_minio_connection_lifecycle: MinioConnectionLifecycle,
) -> None:
    class Container:
        def start(self) -> None:
            raise RuntimeError("synthetic-start-failure")

        def stop(self) -> None:
            return None

    lifecycle = fixture_minio_connection_lifecycle(Container())
    with pytest.raises(ExceptionGroup) as captured:
        next(lifecycle)

    rendered = _exception_debug_rendering(captured.value)
    assert "rag-fixture-password-not-for-production" not in rendered
    assert "rag-fixture" not in rendered


@pytest.mark.integration
def test_minio_http_pool_bounds_connect_and_read_waits(
    minio_http_pool_factory: MinioHttpPoolFactory,
) -> None:
    pool = minio_http_pool_factory()
    try:
        timeout = pool.connection_pool_kw["timeout"]
        assert timeout.connect_timeout == 1.0
        assert timeout.read_timeout == 1.0
    finally:
        pool.clear()


@pytest.mark.integration
def test_service_container_lifecycle_attempts_stop_after_body_and_sanitizes_teardown(
    fixture_service_container_lifecycle: ServiceContainerLifecycle,
) -> None:
    events: list[str] = []

    class Container:
        def start(self) -> None:
            events.append("start")

        def stop(self) -> None:
            events.append("stop")
            raise RuntimeError("qdrant-api-key=synthetic-password")

    connection = object()
    lifecycle = fixture_service_container_lifecycle(
        resource="qdrant",
        container=Container(),
        connection=lambda _container: connection,
        readiness=lambda resolved: resolved is connection,
    )
    assert next(lifecycle) is connection

    with pytest.raises(ExceptionGroup) as captured:
        next(lifecycle)

    assert events == ["start", "stop"]
    assert [str(error) for error in captured.value.exceptions] == [
        "qdrant.stop teardown failed (RuntimeError)"
    ]
    _assert_no_sync_secret_is_reachable(captured.value)


@pytest.mark.integration
def test_service_container_lifecycle_preserves_safe_body_failure_when_stop_also_fails(
    fixture_service_container_lifecycle: ServiceContainerLifecycle,
) -> None:
    class Container:
        def start(self) -> None:
            return None

        def stop(self) -> None:
            raise RuntimeError("synthetic-stop-password")

    lifecycle = fixture_service_container_lifecycle(
        resource="minio",
        container=Container(),
        connection=lambda _container: object(),
        readiness=lambda _connection: True,
    )
    next(lifecycle)

    with pytest.raises(BaseExceptionGroup) as captured:
        lifecycle.throw(_body_secret_error())

    assert [str(error) for error in captured.value.exceptions] == [
        "fixture.body primary failed (LookupError)",
        "minio.stop teardown failed (RuntimeError)",
    ]
    _assert_no_sync_secret_is_reachable(captured.value)


@pytest.mark.integration
def test_service_container_lifecycle_preserves_cancellation_when_stop_also_fails(
    fixture_service_container_lifecycle: ServiceContainerLifecycle,
) -> None:
    class Container:
        def start(self) -> None:
            return None

        def stop(self) -> None:
            raise RuntimeError("synthetic-stop-password")

    lifecycle = fixture_service_container_lifecycle(
        resource="redis",
        container=Container(),
        connection=lambda _container: object(),
        readiness=lambda _connection: True,
    )
    next(lifecycle)

    with pytest.raises(BaseExceptionGroup) as captured:
        lifecycle.throw(asyncio.CancelledError())

    assert isinstance(captured.value.exceptions[0], asyncio.CancelledError)
    assert str(captured.value.exceptions[1]) == "redis.stop teardown failed (RuntimeError)"


@pytest.mark.integration
def test_provider_fixture_closes_partial_setup_when_thread_start_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_start(_thread: object) -> None:
        raise RuntimeError("synthetic-provider-key-password")

    monkeypatch.setattr(threading.Thread, "start", fail_start)

    with (
        pytest.raises(provider_stub.ProviderStubFixtureError) as captured,
        provider_stub.running_provider_https_stub(tmp_path),
    ):
        raise AssertionError("provider fixture unexpectedly yielded")

    assert str(captured.value) == "provider HTTPS stub setup failed (RuntimeError)"
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
    _assert_no_sync_secret_is_reachable(captured.value)
    rendered = _exception_debug_rendering(captured.value)
    assert "provider-key.pem" not in rendered
    assert "provider-ca.pem" not in rendered
    assert "https://127.0.0.1" not in rendered


@pytest.mark.integration
def test_provider_fixture_restricts_and_removes_runtime_tls_material(tmp_path: Path) -> None:
    paths: tuple[Path, ...]
    with provider_stub.running_provider_https_stub(tmp_path) as provider:
        directory = Path(provider.ca_bundle).parent
        paths = (
            directory / "provider-ca.pem",
            directory / "provider-cert.pem",
            directory / "provider-key.pem",
        )
        assert paths[2].stat().st_mode & 0o077 == 0
        assert all(path.exists() for path in paths)

    assert all(not path.exists() for path in paths)


@pytest.mark.integration
def test_provider_fixture_preserves_setup_cancellation_and_removes_tls_material(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def cancel_start(_thread: object) -> None:
        raise asyncio.CancelledError

    monkeypatch.setattr(threading.Thread, "start", cancel_start)

    with (
        pytest.raises(asyncio.CancelledError),
        provider_stub.running_provider_https_stub(tmp_path),
    ):
        raise AssertionError("provider fixture unexpectedly yielded")

    assert not tuple(tmp_path.glob("provider-*.pem"))
