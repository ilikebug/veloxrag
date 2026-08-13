import asyncio
from typing import Any, cast

import pytest
from fastapi.testclient import TestClient

from rag_service.config import Settings
from rag_service.db.session import Database
from rag_service.main import create_app
from rag_service.readiness import ComponentStatus, ReadinessScope, ReadinessSnapshot


def component(ok: bool) -> ComponentStatus:
    return ComponentStatus(ok=ok, latency_ms=1.0, error=None if ok else "Unavailable")


class StaticReadinessProvider:
    def __init__(self, snapshot: ReadinessSnapshot) -> None:
        self._snapshot = snapshot
        self.close_calls = 0
        self.scopes: list[ReadinessScope] = []

    async def snapshot(self, scope: ReadinessScope = ReadinessScope.ALL) -> ReadinessSnapshot:
        self.scopes.append(scope)
        return self._snapshot

    async def close(self) -> None:
        self.close_calls += 1


class ScopedReadinessProvider:
    def __init__(self, *, core: bool = True, ingest_credentials: bool = True) -> None:
        self.core = core
        self.ingest_credentials = ingest_credentials

    async def snapshot(self, scope: ReadinessScope = ReadinessScope.ALL) -> ReadinessSnapshot:
        not_checked = ComponentStatus(ok=False, latency_ms=0.0, error="NotChecked")
        components = {
            "postgres": component(self.core),
            "qdrant": component(self.core),
            "redis": not_checked,
            "minio": not_checked,
            "ingest_credentials": not_checked,
            "retrieve_credentials": not_checked,
        }
        if scope in {ReadinessScope.ALL, ReadinessScope.INGEST}:
            components["redis"] = component(True)
            components["minio"] = component(True)
            components["ingest_credentials"] = component(self.ingest_credentials)
        if scope in {ReadinessScope.ALL, ReadinessScope.RETRIEVE}:
            components["retrieve_credentials"] = component(True)
        return ReadinessSnapshot(components=components, answer_configured=False)


class FalseySettings(Settings):
    def __bool__(self) -> bool:
        return False


class TrackingLiveProvider:
    def __init__(self, close_error: BaseException | None = None) -> None:
        self.close_error = close_error
        self.settings: Settings | None = None
        self.database: object | None = None
        self.construct_calls = 0
        self.close_calls = 0

    async def create(
        self,
        settings: Settings,
        *,
        database: object,
    ) -> "TrackingLiveProvider":
        self.settings = settings
        self.database = database
        self.construct_calls += 1
        return self

    async def close(self) -> None:
        self.close_calls += 1
        if self.close_error is not None:
            raise self.close_error


class BlockingLiveProvider(TrackingLiveProvider):
    def __init__(self, cleanup_error: BaseException | None = None) -> None:
        super().__init__()
        self.cleanup_error = cleanup_error
        self.cleanup_started = asyncio.Event()
        self.release_cleanup = asyncio.Event()
        self.cleanup_completed = asyncio.Event()
        self.cleanup_task: asyncio.Task[None] | None = None

    async def close(self) -> None:
        self.close_calls += 1
        if self.cleanup_task is None:
            self.cleanup_task = asyncio.create_task(self._cleanup())
        await asyncio.shield(self.cleanup_task)

    async def _cleanup(self) -> None:
        self.cleanup_started.set()
        await self.release_cleanup.wait()
        self.cleanup_completed.set()
        if self.cleanup_error is not None:
            raise self.cleanup_error


def snapshot(
    *,
    core: bool,
    ingest: bool,
    answer: bool = False,
    ingest_credentials: bool = True,
    retrieve_credentials: bool = True,
) -> ReadinessSnapshot:
    return ReadinessSnapshot(
        components={
            "postgres": component(core),
            "qdrant": component(core),
            "redis": component(ingest),
            "minio": component(ingest),
            "ingest_credentials": component(ingest_credentials),
            "retrieve_credentials": component(retrieve_credentials),
        },
        answer_configured=answer,
    )


def test_health_is_process_liveness() -> None:
    app = create_app(readiness_provider=StaticReadinessProvider(snapshot(core=False, ingest=False)))

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.parametrize(
    ("path", "expected_scope"),
    [
        ("/ready", ReadinessScope.CORE),
        ("/ready/ingest", ReadinessScope.INGEST),
        ("/ready/retrieve", ReadinessScope.RETRIEVE),
        ("/ready/answer", ReadinessScope.CORE),
    ],
)
def test_readiness_routes_request_only_their_capability_scope(
    path: str,
    expected_scope: ReadinessScope,
) -> None:
    provider = StaticReadinessProvider(snapshot(core=True, ingest=True, answer=True))
    app = create_app(readiness_provider=provider)

    with TestClient(app) as client:
        response = client.get(path)

    assert response.status_code == 200
    assert provider.scopes == [expected_scope]


@pytest.mark.parametrize(
    (
        "path",
        "expected_legacy_retrieval",
        "expected_legacy_ingest",
        "expected_retrieval_status",
        "expected_ingest_status",
    ),
    [
        ("/ready", True, False, "not_checked", "not_checked"),
        ("/ready/retrieve", True, False, "available", "not_checked"),
        ("/ready/ingest", True, True, "available", "available"),
    ],
)
def test_scoped_readiness_preserves_legacy_booleans_and_reports_strict_status(
    path: str,
    expected_legacy_retrieval: bool,
    expected_legacy_ingest: bool,
    expected_retrieval_status: str,
    expected_ingest_status: str,
) -> None:
    app = create_app(readiness_provider=ScopedReadinessProvider())

    with TestClient(app) as client:
        response = client.get(path)

    assert response.status_code == 200
    assert response.json()["capabilities"]["retrieval"] is expected_legacy_retrieval
    assert response.json()["capabilities"]["ingest"] is expected_legacy_ingest
    assert response.json()["capability_status"] == {
        "retrieval": expected_retrieval_status,
        "ingest": expected_ingest_status,
    }


def test_core_failure_makes_all_core_dependent_capabilities_definitively_false() -> None:
    app = create_app(readiness_provider=ScopedReadinessProvider(core=False))

    with TestClient(app) as client:
        response = client.get("/ready")

    assert response.status_code == 503
    assert response.json()["capabilities"]["retrieval"] is False
    assert response.json()["capabilities"]["ingest"] is False
    assert response.json()["capability_status"] == {
        "retrieval": "unavailable",
        "ingest": "unavailable",
    }


def test_failed_ingest_credentials_do_not_imply_active_retrieval_failure() -> None:
    app = create_app(readiness_provider=ScopedReadinessProvider(ingest_credentials=False))

    with TestClient(app) as client:
        response = client.get("/ready/ingest")

    assert response.status_code == 503
    assert response.json()["capabilities"]["retrieval"] is True
    assert response.json()["capabilities"]["ingest"] is True
    assert response.json()["capability_status"] == {
        "retrieval": "not_checked",
        "ingest": "unavailable",
    }


class BorrowedDatabase:
    async def close(self) -> None:
        raise AssertionError("borrowed database must not be closed")


def test_fully_injected_resources_do_not_resolve_live_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_get_settings() -> Settings:
        raise AssertionError("borrowed readiness provider must not resolve live settings")

    monkeypatch.setattr("rag_service.main.get_settings", unexpected_get_settings)
    app = create_app(
        database=cast(Database, BorrowedDatabase()),
        readiness_provider=StaticReadinessProvider(snapshot(core=True, ingest=True)),
    )

    with TestClient(app) as client:
        response = client.get("/ready")

    assert response.status_code == 200


def test_core_ready_can_succeed_while_ingest_is_degraded() -> None:
    app = create_app(readiness_provider=StaticReadinessProvider(snapshot(core=True, ingest=False)))

    with TestClient(app) as client:
        response = client.get("/ready")

    assert response.status_code == 200
    assert response.json()["capabilities"] == {
        "retrieval": True,
        "ingest": False,
        "answer": False,
    }


def test_ingest_ready_requires_redis_and_minio() -> None:
    app = create_app(readiness_provider=StaticReadinessProvider(snapshot(core=True, ingest=False)))

    with TestClient(app) as client:
        response = client.get("/ready/ingest")

    assert response.status_code == 503


def test_role_ready_endpoints_use_their_exact_credential_sets() -> None:
    app = create_app(
        readiness_provider=StaticReadinessProvider(
            snapshot(
                core=True,
                ingest=True,
                ingest_credentials=False,
                retrieve_credentials=True,
            )
        )
    )

    with TestClient(app) as client:
        core_response = client.get("/ready")
        ingest_response = client.get("/ready/ingest")
        retrieve_response = client.get("/ready/retrieve")

    assert core_response.status_code == 200
    assert core_response.json()["capabilities"]["retrieval"] is True
    assert core_response.json()["capabilities"]["ingest"] is True
    assert core_response.json()["capability_status"]["ingest"] == "unavailable"
    assert ingest_response.status_code == 503
    assert ingest_response.json()["reason"] == "ingest_dependencies_unavailable"
    assert retrieve_response.status_code == 200
    assert retrieve_response.json()["ready"] is True


def test_retrieve_ready_fails_when_an_active_credential_is_unavailable() -> None:
    app = create_app(
        readiness_provider=StaticReadinessProvider(
            snapshot(
                core=True,
                ingest=True,
                ingest_credentials=True,
                retrieve_credentials=False,
            )
        )
    )

    with TestClient(app) as client:
        response = client.get("/ready/retrieve")

    assert response.status_code == 503
    assert response.json()["ready"] is False
    assert response.json()["reason"] == "retrieve_dependencies_unavailable"


def test_answer_ready_reports_unconfigured_profile() -> None:
    app = create_app(readiness_provider=StaticReadinessProvider(snapshot(core=True, ingest=True)))

    with TestClient(app) as client:
        response = client.get("/ready/answer")

    assert response.status_code == 503
    assert response.json()["reason"] == "query_profile_not_configured"


@pytest.mark.parametrize(
    ("path", "readiness_snapshot", "expected_status", "expected_ready", "expected_reason"),
    [
        ("/ready", snapshot(core=False, ingest=False), 503, False, None),
        ("/ready/ingest", snapshot(core=True, ingest=True), 200, True, None),
        ("/ready/retrieve", snapshot(core=True, ingest=False), 200, True, None),
        ("/ready/answer", snapshot(core=True, ingest=True, answer=True), 200, True, None),
        (
            "/ready/answer",
            snapshot(core=False, ingest=False),
            503,
            False,
            "core_dependencies_unavailable",
        ),
    ],
)
def test_readiness_endpoint_significant_branches(
    path: str,
    readiness_snapshot: ReadinessSnapshot,
    expected_status: int,
    expected_ready: bool,
    expected_reason: str | None,
) -> None:
    app = create_app(readiness_provider=StaticReadinessProvider(readiness_snapshot))

    with TestClient(app) as client:
        response = client.get(path)

    assert response.status_code == expected_status
    assert response.json()["ready"] is expected_ready
    assert response.json()["reason"] == expected_reason


@pytest.mark.parametrize(
    "path",
    ["/ready", "/ready/ingest", "/ready/retrieve", "/ready/answer"],
)
def test_readiness_openapi_declares_service_unavailable(path: str) -> None:
    app = create_app(readiness_provider=StaticReadinessProvider(snapshot(core=True, ingest=True)))

    operation = app.openapi()["paths"][path]["get"]
    responses: dict[str, Any] = operation["responses"]

    assert {"200", "503"}.issubset(responses)
    assert responses["503"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/ReadinessResponse"
    }


def test_readiness_openapi_preserves_boolean_capabilities_and_adds_statuses() -> None:
    app = create_app(readiness_provider=StaticReadinessProvider(snapshot(core=True, ingest=True)))

    schemas = app.openapi()["components"]["schemas"]
    capability_schema = schemas["CapabilityResponse"]
    status_schema = schemas["CapabilityStatusResponse"]

    assert capability_schema["required"] == ["retrieval", "ingest", "answer"]
    assert capability_schema["properties"] == {
        "retrieval": {"type": "boolean", "title": "Retrieval"},
        "ingest": {"type": "boolean", "title": "Ingest"},
        "answer": {"type": "boolean", "title": "Answer"},
    }
    assert status_schema["required"] == ["retrieval", "ingest"]
    for property_schema in status_schema["properties"].values():
        assert property_schema["enum"] == ["available", "unavailable", "not_checked"]


def test_injected_provider_is_not_closed() -> None:
    provider = StaticReadinessProvider(snapshot(core=True, ingest=True))
    app = create_app(readiness_provider=provider)

    with TestClient(app):
        pass

    assert provider.close_calls == 0


def test_default_live_provider_uses_explicit_settings_and_closes_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = TrackingLiveProvider()
    settings = FalseySettings(_env_file=None)

    def unexpected_get_settings() -> Settings:
        raise AssertionError("explicit settings must not be replaced")

    monkeypatch.setattr("rag_service.main.LiveReadinessProvider", provider)
    monkeypatch.setattr("rag_service.main.get_settings", unexpected_get_settings)
    app = create_app(settings=settings)

    with TestClient(app):
        pass

    assert provider.settings is settings
    assert provider.construct_calls == 1
    assert provider.close_calls == 1


def test_default_settings_are_resolved_lazily_during_lifespan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = TrackingLiveProvider()
    settings = Settings(_env_file=None)
    settings_calls = 0

    def resolve_settings() -> Settings:
        nonlocal settings_calls
        settings_calls += 1
        return settings

    monkeypatch.setattr("rag_service.main.LiveReadinessProvider", provider)
    monkeypatch.setattr("rag_service.main.get_settings", resolve_settings)
    app = create_app()

    assert settings_calls == 0
    assert provider.construct_calls == 0

    with TestClient(app):
        assert settings_calls == 1
        assert provider.settings is settings

    assert provider.close_calls == 1


@pytest.mark.asyncio
async def test_owned_close_failure_propagates_on_normal_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = TrackingLiveProvider(RuntimeError("safe cleanup failure"))
    monkeypatch.setattr("rag_service.main.LiveReadinessProvider", provider)
    app = create_app(settings=Settings(_env_file=None))

    with pytest.raises(ExceptionGroup) as captured:
        async with app.router.lifespan_context(app):
            pass

    assert provider.close_calls == 1
    assert [str(error) for error in captured.value.exceptions] == [
        "readiness cleanup failed with RuntimeError"
    ]
    assert "safe cleanup failure" not in repr(captured.value)


@pytest.mark.asyncio
async def test_lifespan_cancellation_waits_for_owned_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = BlockingLiveProvider()
    monkeypatch.setattr("rag_service.main.LiveReadinessProvider", provider)
    app = create_app(settings=Settings(_env_file=None))
    lifespan = app.router.lifespan_context(app)
    await lifespan.__aenter__()
    exit_task = asyncio.create_task(lifespan.__aexit__(None, None, None))

    try:
        async with asyncio.timeout(0.2):
            await provider.cleanup_started.wait()
        exit_task.cancel()
        await asyncio.sleep(0)

        assert exit_task.done() is False

        provider.release_cleanup.set()
        with pytest.raises(asyncio.CancelledError):
            await exit_task

        assert provider.cleanup_completed.is_set()
        assert provider.close_calls == 1
    finally:
        provider.release_cleanup.set()
        await asyncio.gather(exit_task, return_exceptions=True)
        if provider.cleanup_task is not None:
            await asyncio.gather(provider.cleanup_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_repeated_lifespan_cancellation_still_drains_owned_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = BlockingLiveProvider()
    monkeypatch.setattr("rag_service.main.LiveReadinessProvider", provider)
    app = create_app(settings=Settings(_env_file=None))
    lifespan = app.router.lifespan_context(app)
    await lifespan.__aenter__()
    exit_task = asyncio.create_task(lifespan.__aexit__(None, None, None))

    try:
        async with asyncio.timeout(0.2):
            await provider.cleanup_started.wait()
        exit_task.cancel("original cancellation")
        await asyncio.sleep(0)

        exit_task.cancel("repeated cancellation")
        await asyncio.sleep(0)

        assert exit_task.done() is False
        assert provider.cleanup_completed.is_set() is False
        assert provider.close_calls == 1

        provider.release_cleanup.set()
        with pytest.raises(asyncio.CancelledError) as captured:
            await exit_task

        assert captured.value.args == ("original cancellation",)
        assert provider.cleanup_completed.is_set()
        assert provider.close_calls == 1
    finally:
        provider.release_cleanup.set()
        await asyncio.gather(exit_task, return_exceptions=True)
        if provider.cleanup_task is not None:
            await asyncio.gather(provider.cleanup_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_lifespan_cancellation_groups_owned_cleanup_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = BlockingLiveProvider(RuntimeError("safe cleanup failure"))
    monkeypatch.setattr("rag_service.main.LiveReadinessProvider", provider)
    app = create_app(settings=Settings(_env_file=None))
    lifespan = app.router.lifespan_context(app)
    await lifespan.__aenter__()
    exit_task = asyncio.create_task(lifespan.__aexit__(None, None, None))

    try:
        async with asyncio.timeout(0.2):
            await provider.cleanup_started.wait()
        exit_task.cancel()
        await asyncio.sleep(0)
        provider.release_cleanup.set()

        with pytest.raises(BaseExceptionGroup) as captured:
            await exit_task

        cancellation, cleanup_group = captured.value.exceptions
        assert isinstance(cancellation, asyncio.CancelledError)
        assert isinstance(cleanup_group, ExceptionGroup)
        assert [str(error) for error in cleanup_group.exceptions] == [
            "readiness cleanup failed with RuntimeError"
        ]
        assert "safe cleanup failure" not in repr(captured.value)
        assert captured.value.__cause__ is None
    finally:
        provider.release_cleanup.set()
        await asyncio.gather(exit_task, return_exceptions=True)
        if provider.cleanup_task is not None:
            await asyncio.gather(provider.cleanup_task, return_exceptions=True)
