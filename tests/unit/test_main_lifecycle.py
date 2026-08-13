import asyncio
from collections.abc import AsyncGenerator
from types import SimpleNamespace
from typing import Any, cast

import httpx
import pytest
from fastapi import Request
from fastapi.dependencies.utils import get_dependant

import rag_service.infrastructure.probes as probes
import rag_service.providers.gateway_provider as gateway_provider
from rag_service.config import Settings
from rag_service.db.dependencies import get_session
from rag_service.db.session import Database
from rag_service.indexing.generation_routes import get_generation_service
from rag_service.main import create_app
from rag_service.providers.gateway_provider import get_embedding_gateway
from rag_service.readiness import (
    ComponentStatus,
    ReadinessProvider,
    ReadinessScope,
    ReadinessSnapshot,
)
from rag_service.retrieval.routes import get_retrieval_service


def healthy_snapshot() -> ReadinessSnapshot:
    status = ComponentStatus(ok=True, latency_ms=1.0)
    return ReadinessSnapshot(
        components={
            "postgres": status,
            "qdrant": status,
            "redis": status,
            "minio": status,
            "ingest_credentials": status,
            "retrieve_credentials": status,
        },
        answer_configured=False,
    )


class FakeSession:
    def __init__(self) -> None:
        self.commit_calls = 0
        self.rollback_calls = 0
        self.close_calls = 0

    async def commit(self) -> None:
        self.commit_calls += 1

    async def rollback(self) -> None:
        self.rollback_calls += 1

    async def close(self) -> None:
        self.close_calls += 1


class FakeSessionContext:
    def __init__(self, session: FakeSession) -> None:
        self.session = session
        self.enter_calls = 0
        self.exit_calls = 0

    async def __aenter__(self) -> FakeSession:
        self.enter_calls += 1
        return self.session

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _traceback: object | None,
    ) -> None:
        self.exit_calls += 1
        if exc_type is not None and exc_type is not GeneratorExit:
            await self.session.rollback()
        await self.session.close()


class FakeSessionFactory:
    def __init__(self, context: FakeSessionContext) -> None:
        self.context = context
        self.calls = 0

    def __call__(self) -> FakeSessionContext:
        self.calls += 1
        return self.context


class TrackingDatabase:
    def __init__(
        self,
        *,
        order: list[str] | None = None,
        close_error: BaseException | None = None,
    ) -> None:
        self.session_value = FakeSession()
        self.session_context = FakeSessionContext(self.session_value)
        self.sessions = FakeSessionFactory(self.session_context)
        self.close_calls = 0
        self.order = order
        self.close_error = close_error

    async def ping(self) -> None:
        return None

    async def close(self) -> None:
        self.close_calls += 1
        if self.order is not None:
            self.order.append("database")
        if self.close_error is not None:
            raise self.close_error


class StaticReadinessProvider:
    def __init__(
        self,
        *,
        order: list[str] | None = None,
        close_error: BaseException | None = None,
    ) -> None:
        self.close_calls = 0
        self.order = order
        self.close_error = close_error

    async def snapshot(self, _scope: ReadinessScope = ReadinessScope.ALL) -> ReadinessSnapshot:
        return healthy_snapshot()

    async def close(self) -> None:
        self.close_calls += 1
        if self.order is not None:
            self.order.append("readiness")
        if self.close_error is not None:
            raise self.close_error


class TrackingGenerationGateway:
    def __init__(self) -> None:
        self.close_calls = 0

    async def aclose(self) -> None:
        self.close_calls += 1


class BlockingGenerationGateway(TrackingGenerationGateway):
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.completed = asyncio.Event()
        self.cancelled = False

    async def aclose(self) -> None:
        self.close_calls += 1
        self.started.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        self.completed.set()


class TrackingDatabaseFactory:
    def __init__(self, database: TrackingDatabase) -> None:
        self.database = database
        self.settings: Settings | None = None
        self.calls = 0

    def from_settings(self, settings: Settings) -> TrackingDatabase:
        self.calls += 1
        self.settings = settings
        return self.database


class TrackingReadinessFactory:
    def __init__(self, provider: StaticReadinessProvider) -> None:
        self.provider = provider
        self.settings: Settings | None = None
        self.database: object | None = None
        self.calls = 0

    async def create(
        self,
        settings: Settings,
        *,
        database: object,
    ) -> StaticReadinessProvider:
        self.calls += 1
        self.settings = settings
        self.database = database
        return self.provider


def as_database(database: TrackingDatabase) -> Database:
    return cast(Database, database)


def as_readiness(provider: StaticReadinessProvider) -> ReadinessProvider:
    return cast(ReadinessProvider, provider)


def request_for(database: TrackingDatabase) -> Request:
    scope: dict[str, Any] = {
        "type": "http",
        "method": "GET",
        "path": "/",
        "headers": [],
        "app": SimpleNamespace(state=SimpleNamespace(database=database)),
    }
    return Request(scope)


@pytest.mark.asyncio
async def test_database_session_returns_sessionmaker_context_without_transaction() -> None:
    session = FakeSession()
    context = FakeSessionContext(session)
    sessions = FakeSessionFactory(context)
    database = object.__new__(Database)
    database.sessions = cast(Any, sessions)

    result = database.session()

    assert cast(Any, result) is context
    assert sessions.calls == 1
    assert context.enter_calls == 0


@pytest.mark.asyncio
async def test_request_session_success_closes_without_commit_or_rollback() -> None:
    database = TrackingDatabase()
    dependency = cast(AsyncGenerator[Any, None], get_session(request_for(database)))

    yielded = await anext(dependency)
    await dependency.aclose()

    assert yielded is cast(Any, database.session_value)
    assert database.sessions.calls == 1
    assert database.session_context.exit_calls == 1
    assert database.session_value.commit_calls == 0
    assert database.session_value.rollback_calls == 0
    assert database.session_value.close_calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [RuntimeError("body failure"), asyncio.CancelledError()])
async def test_request_session_error_rolls_back_closes_and_propagates(
    error: BaseException,
) -> None:
    database = TrackingDatabase()
    dependency = cast(AsyncGenerator[Any, None], get_session(request_for(database)))
    await anext(dependency)

    with pytest.raises(type(error)) as captured:
        await dependency.athrow(error)

    assert captured.value is error
    assert database.session_value.commit_calls == 0
    assert database.session_value.rollback_calls == 1
    assert database.session_value.close_calls == 1


@pytest.mark.asyncio
async def test_app_resources_are_lazy_and_live_readiness_shares_owned_database(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings(_env_file=None)
    database = TrackingDatabase()
    provider = StaticReadinessProvider()
    database_factory = TrackingDatabaseFactory(database)
    readiness_factory = TrackingReadinessFactory(provider)
    settings_calls = 0

    def resolve_settings() -> Settings:
        nonlocal settings_calls
        settings_calls += 1
        return settings

    monkeypatch.setattr("rag_service.main.Database", database_factory)
    monkeypatch.setattr("rag_service.main.LiveReadinessProvider", readiness_factory)
    monkeypatch.setattr("rag_service.main.get_settings", resolve_settings)
    app = create_app()

    assert settings_calls == 0
    assert database_factory.calls == 0
    assert readiness_factory.calls == 0
    assert not hasattr(app.state, "database")

    async with app.router.lifespan_context(app):
        assert settings_calls == 1
        assert app.state.database is database
        assert app.state.readiness_provider is provider
        assert readiness_factory.database is database

    assert provider.close_calls == 1
    assert database.close_calls == 1
    assert not hasattr(app.state, "database")
    assert not hasattr(app.state, "readiness_provider")


@pytest.mark.asyncio
async def test_generation_and_retrieval_dependencies_share_injected_embedding_gateway() -> None:
    settings = Settings(_env_file=None)
    database = TrackingDatabase()
    readiness = StaticReadinessProvider()
    gateway = TrackingGenerationGateway()
    app = create_app(
        settings=settings,
        database=as_database(database),
        readiness_provider=as_readiness(readiness),
        generation_embedding_gateway=cast(Any, gateway),
    )

    async with app.router.lifespan_context(app):
        provider = app.state.embedding_gateway_provider
        assert isinstance(provider, gateway_provider.EmbeddingGatewayProvider)
        generation_dependency = next(
            dependency.call
            for dependency in get_dependant(
                path="/generation", call=get_generation_service
            ).dependencies
            if dependency.name == "embedding_gateway"
        )
        retrieval_dependency = next(
            dependency.call
            for dependency in get_dependant(
                path="/retrieval", call=get_retrieval_service
            ).dependencies
            if dependency.name == "embedding_gateway"
        )
        assert generation_dependency is retrieval_dependency is get_embedding_gateway
        request = Request(
            {
                "type": "http",
                "method": "GET",
                "path": "/",
                "headers": [],
                "app": app,
            }
        )
        generation_gateway = await get_embedding_gateway(
            request,
            as_database(database),
            settings,
        )
        retrieval_gateway = await get_embedding_gateway(
            request,
            as_database(database),
            settings,
        )
        assert generation_gateway is retrieval_gateway is gateway
        assert gateway.close_calls == 0

    assert gateway.close_calls == 0
    assert not hasattr(app.state, "embedding_gateway_provider")


@pytest.mark.asyncio
async def test_owned_embedding_gateway_is_lazy_and_closed_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings(_env_file=None)
    database = TrackingDatabase()
    readiness = StaticReadinessProvider()
    gateway = TrackingGenerationGateway()
    factory_calls = 0

    def create_gateway(selected_database: Database, selected_settings: Settings) -> object:
        nonlocal factory_calls
        factory_calls += 1
        assert selected_database is cast(Database, database)
        assert selected_settings is settings
        return gateway

    monkeypatch.setattr(
        gateway_provider,
        "_embedding_gateway_from_dependencies",
        create_gateway,
    )
    app = create_app(
        settings=settings,
        database=as_database(database),
        readiness_provider=as_readiness(readiness),
    )

    assert factory_calls == 0
    async with app.router.lifespan_context(app):
        provider = app.state.embedding_gateway_provider
        first = await provider.get(as_database(database), settings)
        second = await provider.get(as_database(database), settings)
        assert first is second is gateway
        assert factory_calls == 1
        assert gateway.close_calls == 0

    assert gateway.close_calls == 1
    assert not hasattr(app.state, "embedding_gateway_provider")


@pytest.mark.asyncio
async def test_embedding_gateway_cleanup_is_cancellation_safe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings(_env_file=None)
    database = TrackingDatabase()
    readiness = StaticReadinessProvider()
    gateway = BlockingGenerationGateway()

    monkeypatch.setattr(
        gateway_provider,
        "_embedding_gateway_from_dependencies",
        lambda _database, _settings: gateway,
    )
    app = create_app(
        settings=settings,
        database=as_database(database),
        readiness_provider=as_readiness(readiness),
    )
    lifespan = app.router.lifespan_context(app)
    await lifespan.__aenter__()
    await app.state.embedding_gateway_provider.get(as_database(database), settings)
    exit_task = asyncio.create_task(lifespan.__aexit__(None, None, None))

    try:
        await gateway.started.wait()
        exit_task.cancel("gateway shutdown cancelled")
        await asyncio.sleep(0)
        assert exit_task.done() is False
        gateway.release.set()
        with pytest.raises(asyncio.CancelledError, match="gateway shutdown cancelled"):
            await exit_task
        assert gateway.close_calls == 1
        assert gateway.cancelled is False
        assert gateway.completed.is_set()
    finally:
        gateway.release.set()
        await asyncio.gather(exit_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_owned_resources_close_readiness_then_database_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    order: list[str] = []
    database = TrackingDatabase(order=order)
    provider = StaticReadinessProvider(order=order)
    monkeypatch.setattr("rag_service.main.Database", TrackingDatabaseFactory(database))
    monkeypatch.setattr(
        "rag_service.main.LiveReadinessProvider",
        TrackingReadinessFactory(provider),
    )
    app = create_app(settings=Settings(_env_file=None))

    async with app.router.lifespan_context(app):
        pass

    assert order == ["readiness", "database"]
    assert provider.close_calls == 1
    assert database.close_calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("inject_database", "inject_readiness", "expected_database_close", "expected_readiness_close"),
    [
        (True, True, 0, 0),
        (True, False, 0, 1),
        (False, True, 1, 0),
        (False, False, 1, 1),
    ],
)
async def test_injected_resource_ownership_combinations(
    monkeypatch: pytest.MonkeyPatch,
    inject_database: bool,
    inject_readiness: bool,
    expected_database_close: int,
    expected_readiness_close: int,
) -> None:
    live_database = TrackingDatabase()
    injected_database = TrackingDatabase()
    live_provider = StaticReadinessProvider()
    injected_provider = StaticReadinessProvider()
    database_factory = TrackingDatabaseFactory(live_database)
    readiness_factory = TrackingReadinessFactory(live_provider)
    monkeypatch.setattr("rag_service.main.Database", database_factory)
    monkeypatch.setattr("rag_service.main.LiveReadinessProvider", readiness_factory)
    app = create_app(
        settings=Settings(_env_file=None),
        database=as_database(injected_database) if inject_database else None,
        readiness_provider=as_readiness(injected_provider) if inject_readiness else None,
    )

    async with app.router.lifespan_context(app):
        expected_database = injected_database if inject_database else live_database
        expected_provider = injected_provider if inject_readiness else live_provider
        assert app.state.database is expected_database
        assert app.state.readiness_provider is expected_provider
        if not inject_readiness:
            assert readiness_factory.database is expected_database

    selected_database = injected_database if inject_database else live_database
    selected_provider = injected_provider if inject_readiness else live_provider
    assert selected_database.close_calls == expected_database_close
    assert selected_provider.close_calls == expected_readiness_close


@pytest.mark.asyncio
async def test_both_injected_resources_do_not_resolve_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = TrackingDatabase()
    provider = StaticReadinessProvider()

    def unexpected_settings() -> Settings:
        raise AssertionError("fully injected application must not resolve settings")

    monkeypatch.setattr("rag_service.main.get_settings", unexpected_settings)
    app = create_app(database=as_database(database), readiness_provider=as_readiness(provider))

    async with app.router.lifespan_context(app):
        assert app.state.database is database


@pytest.mark.asyncio
async def test_owned_minio_data_plane_uses_operation_timeout_not_readiness_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings(
        _env_file=None,
        readiness_timeout_seconds=0.25,
        minio_operation_timeout_seconds=7.5,
    )
    database = TrackingDatabase()
    readiness = StaticReadinessProvider()
    captured: dict[str, object] = {}

    class Pool:
        def clear(self) -> None:
            pool_clear_calls = captured.get("pool_clear_calls", 0)
            assert isinstance(pool_clear_calls, int)
            captured["pool_clear_calls"] = pool_clear_calls + 1

    class ObjectStore:
        def __init__(self, **kwargs: object) -> None:
            captured["store_kwargs"] = kwargs

        async def aclose(self) -> None:
            return None

    class RedisClient:
        async def aclose(self) -> None:
            return None

    class RedisFactory:
        @classmethod
        def from_url(cls, *_args: object, **_kwargs: object) -> RedisClient:
            return RedisClient()

    def pool_manager(**kwargs: object) -> Pool:
        captured["pool_kwargs"] = kwargs
        return Pool()

    def minio_client(*_args: object, **kwargs: object) -> object:
        captured["minio_kwargs"] = kwargs
        return object()

    monkeypatch.setattr("rag_service.main.PoolManager", pool_manager)
    monkeypatch.setattr("rag_service.main.Minio", minio_client)
    monkeypatch.setattr("rag_service.main.MinioObjectStore", ObjectStore)
    monkeypatch.setattr("rag_service.main.Redis", RedisFactory)
    app = create_app(
        settings=settings,
        database=as_database(database),
        readiness_provider=as_readiness(readiness),
    )

    async with app.router.lifespan_context(app):
        timeout = cast(Any, cast(dict[str, object], captured["pool_kwargs"])["timeout"])
        assert timeout.connect_timeout == 7.5
        assert timeout.read_timeout == 7.5
        assert settings.readiness_timeout_seconds == 0.25
        assert cast(dict[str, object], captured["store_kwargs"])["operation_timeout_seconds"] == 7.5

    assert captured["pool_clear_calls"] == 1


@pytest.mark.asyncio
async def test_owned_minio_pool_is_not_cleared_while_store_close_is_still_running(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings(_env_file=None, shutdown_timeout_seconds=0.01)
    database = TrackingDatabase()
    readiness = StaticReadinessProvider()

    class Pool:
        def __init__(self) -> None:
            self.clear_calls = 0

        def clear(self) -> None:
            self.clear_calls += 1

    class ObjectStore:
        def __init__(self, **_kwargs: object) -> None:
            self.close_started = asyncio.Event()
            self.close_release = asyncio.Event()
            self.close_finished = asyncio.Event()

        async def aclose(self) -> None:
            self.close_started.set()
            try:
                await self.close_release.wait()
            except asyncio.CancelledError:
                await self.close_release.wait()
            self.close_finished.set()

    class RedisClient:
        async def aclose(self) -> None:
            return None

    class RedisFactory:
        @classmethod
        def from_url(cls, *_args: object, **_kwargs: object) -> RedisClient:
            return RedisClient()

    pool = Pool()
    store: ObjectStore | None = None

    def object_store(**kwargs: object) -> ObjectStore:
        nonlocal store
        store = ObjectStore(**kwargs)
        return store

    monkeypatch.setattr("rag_service.main.PoolManager", lambda **_kwargs: pool)
    monkeypatch.setattr("rag_service.main.Minio", lambda *_args, **_kwargs: object())
    monkeypatch.setattr("rag_service.main.MinioObjectStore", object_store)
    monkeypatch.setattr("rag_service.main.Redis", RedisFactory)
    app = create_app(
        settings=settings,
        database=as_database(database),
        readiness_provider=as_readiness(readiness),
    )

    try:
        with pytest.raises(ExceptionGroup) as captured:
            async with app.router.lifespan_context(app):
                pass

        assert store is not None
        assert store.close_started.is_set()
        assert not store.close_finished.is_set()
        assert pool.clear_calls == 0
        assert [str(error) for error in captured.value.exceptions] == [
            "upload object store cleanup failed with TimeoutError"
        ]
    finally:
        assert store is not None
        store.close_release.set()
        async with asyncio.timeout(0.2):
            await store.close_finished.wait()


class BlockingDatabase(TrackingDatabase):
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.completed = asyncio.Event()
        self.cancelled = False

    async def close(self) -> None:
        self.close_calls += 1
        self.started.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        self.completed.set()


class CancellationResistantDatabase(TrackingDatabase):
    def __init__(self, *, close_error: BaseException | None = None) -> None:
        super().__init__(close_error=close_error)
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.finished = asyncio.Event()

    async def close(self) -> None:
        self.close_calls += 1
        self.started.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            await self.release.wait()
        self.finished.set()
        if self.close_error is not None:
            raise self.close_error


class CancellationResistantReadinessProvider(StaticReadinessProvider):
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.finished = asyncio.Event()

    async def close(self) -> None:
        self.close_calls += 1
        self.started.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            await self.release.wait()
        self.finished.set()


@pytest.mark.asyncio
async def test_shutdown_cancellation_drains_each_owned_closer_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = BlockingDatabase()
    provider = StaticReadinessProvider()
    monkeypatch.setattr("rag_service.main.Database", TrackingDatabaseFactory(database))
    monkeypatch.setattr(
        "rag_service.main.LiveReadinessProvider",
        TrackingReadinessFactory(provider),
    )
    app = create_app(settings=Settings(_env_file=None))
    lifespan = app.router.lifespan_context(app)
    await lifespan.__aenter__()
    exit_task = asyncio.create_task(lifespan.__aexit__(None, None, None))

    try:
        async with asyncio.timeout(0.2):
            await database.started.wait()
        exit_task.cancel("shutdown cancelled")
        await asyncio.sleep(0)
        assert exit_task.done() is False

        database.release.set()
        with pytest.raises(asyncio.CancelledError) as captured:
            await exit_task

        assert captured.value.args == ("shutdown cancelled",)
        assert database.cancelled is False
        assert database.completed.is_set()
        assert provider.close_calls == 1
        assert database.close_calls == 1
    finally:
        database.release.set()
        await asyncio.gather(exit_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_hanging_readiness_cleanup_is_bounded_and_database_still_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = TrackingDatabase()
    provider = CancellationResistantReadinessProvider()
    monkeypatch.setattr("rag_service.main.Database", TrackingDatabaseFactory(database))
    monkeypatch.setattr(
        "rag_service.main.LiveReadinessProvider",
        TrackingReadinessFactory(provider),
    )
    app = create_app(settings=Settings(_env_file=None, shutdown_timeout_seconds=0.01))

    try:
        async with asyncio.timeout(0.2):
            with pytest.raises(ExceptionGroup) as captured:
                async with app.router.lifespan_context(app):
                    pass

        assert [str(error) for error in captured.value.exceptions] == [
            "readiness cleanup failed with TimeoutError"
        ]
        assert provider.close_calls == 1
        assert database.close_calls == 1
        assert not hasattr(app.state, "readiness_provider")
        assert not hasattr(app.state, "database")
    finally:
        provider.release.set()
        async with asyncio.timeout(0.2):
            await provider.finished.wait()


@pytest.mark.asyncio
async def test_hanging_database_cleanup_is_bounded_and_state_is_cleared(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = CancellationResistantDatabase()
    provider = StaticReadinessProvider()
    monkeypatch.setattr("rag_service.main.Database", TrackingDatabaseFactory(database))
    monkeypatch.setattr(
        "rag_service.main.LiveReadinessProvider",
        TrackingReadinessFactory(provider),
    )
    app = create_app(settings=Settings(_env_file=None, shutdown_timeout_seconds=0.01))

    try:
        async with asyncio.timeout(0.2):
            with pytest.raises(ExceptionGroup) as captured:
                async with app.router.lifespan_context(app):
                    pass

        assert [str(error) for error in captured.value.exceptions] == [
            "database cleanup failed with TimeoutError"
        ]
        assert provider.close_calls == 1
        assert database.close_calls == 1
        assert not hasattr(app.state, "readiness_provider")
        assert not hasattr(app.state, "database")
    finally:
        database.release.set()
        async with asyncio.timeout(0.2):
            await database.finished.wait()


@pytest.mark.asyncio
async def test_each_hanging_closer_reports_its_specific_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = CancellationResistantDatabase()
    provider = CancellationResistantReadinessProvider()
    monkeypatch.setattr("rag_service.main.Database", TrackingDatabaseFactory(database))
    monkeypatch.setattr(
        "rag_service.main.LiveReadinessProvider",
        TrackingReadinessFactory(provider),
    )
    app = create_app(settings=Settings(_env_file=None, shutdown_timeout_seconds=0.01))

    try:
        async with asyncio.timeout(0.2):
            with pytest.raises(ExceptionGroup) as captured:
                async with app.router.lifespan_context(app):
                    pass

        assert [str(error) for error in captured.value.exceptions] == [
            "readiness cleanup failed with TimeoutError",
            "database cleanup failed with TimeoutError",
        ]
        assert provider.close_calls == 1
        assert database.close_calls == 1
        assert not hasattr(app.state, "readiness_provider")
        assert not hasattr(app.state, "database")
    finally:
        provider.release.set()
        database.release.set()
        async with asyncio.timeout(0.2):
            await asyncio.gather(
                provider.finished.wait(),
                database.finished.wait(),
            )


@pytest.mark.asyncio
async def test_repeated_cancellation_cannot_extend_hanging_cleanup_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = TrackingDatabase(close_error=RuntimeError("database-secret"))
    provider = CancellationResistantReadinessProvider()
    monkeypatch.setattr("rag_service.main.Database", TrackingDatabaseFactory(database))
    monkeypatch.setattr(
        "rag_service.main.LiveReadinessProvider",
        TrackingReadinessFactory(provider),
    )
    app = create_app(settings=Settings(_env_file=None, shutdown_timeout_seconds=0.01))
    lifespan = app.router.lifespan_context(app)
    await lifespan.__aenter__()
    exit_task = asyncio.create_task(lifespan.__aexit__(None, None, None))

    try:
        async with asyncio.timeout(0.2):
            await provider.started.wait()
        exit_task.cancel("original cancellation")
        await asyncio.sleep(0)
        exit_task.cancel("repeated cancellation")

        async with asyncio.timeout(0.2):
            with pytest.raises(BaseExceptionGroup) as captured:
                await exit_task

        cancellation, cleanup_group = captured.value.exceptions
        assert isinstance(cancellation, asyncio.CancelledError)
        assert cancellation.args == ("original cancellation",)
        assert isinstance(cleanup_group, ExceptionGroup)
        assert {str(error) for error in cleanup_group.exceptions} == {
            "readiness cleanup failed with TimeoutError",
            "database cleanup failed with RuntimeError",
        }
        assert "database-secret" not in exception_surface(captured.value)
        assert provider.close_calls == 1
        assert database.close_calls == 1
        assert not hasattr(app.state, "readiness_provider")
        assert not hasattr(app.state, "database")
    finally:
        provider.release.set()
        await asyncio.gather(exit_task, return_exceptions=True)
        async with asyncio.timeout(0.2):
            await provider.finished.wait()


@pytest.mark.asyncio
async def test_readiness_startup_failure_still_closes_owned_database(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = TrackingDatabase()

    async def fail_readiness(*_args: object, **_kwargs: object) -> StaticReadinessProvider:
        raise RuntimeError("startup failed")

    monkeypatch.setattr("rag_service.main.Database", TrackingDatabaseFactory(database))
    monkeypatch.setattr(
        "rag_service.main.LiveReadinessProvider",
        SimpleNamespace(create=fail_readiness),
    )
    app = create_app(settings=Settings(_env_file=None))

    with pytest.raises(RuntimeError, match="startup failed"):
        async with app.router.lifespan_context(app):
            pass

    assert database.close_calls == 1
    assert not hasattr(app.state, "database")


@pytest.mark.asyncio
async def test_partial_live_readiness_failure_rolls_back_then_closes_app_database(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = TrackingDatabase()

    class RedisClient:
        def __init__(self) -> None:
            self.close_calls = 0

        async def aclose(self) -> None:
            self.close_calls += 1

    class RedisFactory:
        @classmethod
        def from_url(cls, *_args: object, **_kwargs: object) -> RedisClient:
            return redis_client

    class Pool:
        def __init__(self) -> None:
            self.clear_calls = 0

        def clear(self) -> None:
            self.clear_calls += 1

    redis_client = RedisClient()
    pool = Pool()

    monkeypatch.setattr("rag_service.main.Database", TrackingDatabaseFactory(database))
    monkeypatch.setattr(probes, "Redis", RedisFactory)
    monkeypatch.setattr(probes, "PoolManager", lambda **_kwargs: pool)
    monkeypatch.setattr(probes, "Minio", lambda *_args, **_kwargs: object())

    def fail_http(**_kwargs: object) -> object:
        raise RuntimeError("http-client-construction-secret")

    monkeypatch.setattr(httpx, "AsyncClient", fail_http)
    app = create_app(settings=Settings(_env_file=None))

    with pytest.raises(ExceptionGroup) as captured:
        async with app.router.lifespan_context(app):
            pass

    assert [str(error) for error in captured.value.exceptions] == [
        "http construction failed with RuntimeError"
    ]
    assert "http-client-construction-secret" not in exception_surface(captured.value)
    assert redis_client.close_calls == 1
    assert pool.clear_calls == 1
    assert database.close_calls == 1
    assert not hasattr(app.state, "database")
    assert not hasattr(app.state, "readiness_provider")


@pytest.mark.asyncio
async def test_startup_cancellation_still_closes_owned_database(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = TrackingDatabase()
    cancellation = asyncio.CancelledError("startup cancelled")

    async def cancel_readiness(*_args: object, **_kwargs: object) -> StaticReadinessProvider:
        raise cancellation

    monkeypatch.setattr("rag_service.main.Database", TrackingDatabaseFactory(database))
    monkeypatch.setattr(
        "rag_service.main.LiveReadinessProvider",
        SimpleNamespace(create=cancel_readiness),
    )
    app = create_app(settings=Settings(_env_file=None))

    with pytest.raises(asyncio.CancelledError) as captured:
        async with app.router.lifespan_context(app):
            pass

    assert captured.value is cancellation
    assert database.close_calls == 1
    assert not hasattr(app.state, "database")


@pytest.mark.asyncio
async def test_startup_and_cleanup_failures_are_grouped_without_cleanup_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cleanup_marker = "postgresql://startup-cleanup-secret"
    database = TrackingDatabase(close_error=RuntimeError(cleanup_marker))
    startup_error = LookupError("safe startup failure")

    async def fail_readiness(*_args: object, **_kwargs: object) -> StaticReadinessProvider:
        raise startup_error

    monkeypatch.setattr("rag_service.main.Database", TrackingDatabaseFactory(database))
    monkeypatch.setattr(
        "rag_service.main.LiveReadinessProvider",
        SimpleNamespace(create=fail_readiness),
    )
    app = create_app(settings=Settings(_env_file=None))

    with pytest.raises(BaseExceptionGroup) as captured:
        async with app.router.lifespan_context(app):
            pass

    assert captured.value.exceptions[0] is startup_error
    assert cleanup_marker not in exception_surface(captured.value)
    assert database.close_calls == 1


@pytest.mark.asyncio
async def test_body_cancellation_still_closes_all_owned_resources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = TrackingDatabase()
    provider = StaticReadinessProvider()
    monkeypatch.setattr("rag_service.main.Database", TrackingDatabaseFactory(database))
    monkeypatch.setattr(
        "rag_service.main.LiveReadinessProvider",
        TrackingReadinessFactory(provider),
    )
    app = create_app(settings=Settings(_env_file=None))
    cancellation = asyncio.CancelledError("request body cancelled")

    with pytest.raises(asyncio.CancelledError) as captured:
        async with app.router.lifespan_context(app):
            raise cancellation

    assert captured.value is cancellation
    assert provider.close_calls == 1
    assert database.close_calls == 1


def exception_surface(error: BaseException) -> str:
    rendered = [str(error), repr(error)]
    if isinstance(error, BaseExceptionGroup):
        rendered.extend(exception_surface(nested) for nested in error.exceptions)
    if error.__cause__ is not None:
        rendered.append(exception_surface(error.__cause__))
    if error.__context__ is not None:
        rendered.append(exception_surface(error.__context__))
    traceback = error.__traceback__
    while traceback is not None:
        module_name = traceback.tb_frame.f_globals.get("__name__", "")
        if isinstance(module_name, str) and module_name.startswith("rag_service"):
            rendered.append(repr(traceback.tb_frame.f_locals))
        traceback = traceback.tb_next
    return "\n".join(rendered)


@pytest.mark.asyncio
async def test_cleanup_failures_are_aggregated_sanitized_and_do_not_skip_database(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    readiness_marker = "readiness-secret-marker"
    database_marker = "postgresql://secret-marker"
    database = TrackingDatabase(close_error=RuntimeError(database_marker))
    provider = StaticReadinessProvider(close_error=ValueError(readiness_marker))
    monkeypatch.setattr("rag_service.main.Database", TrackingDatabaseFactory(database))
    monkeypatch.setattr(
        "rag_service.main.LiveReadinessProvider",
        TrackingReadinessFactory(provider),
    )
    app = create_app(settings=Settings(_env_file=None))

    with pytest.raises(ExceptionGroup) as captured:
        async with app.router.lifespan_context(app):
            pass

    surface = exception_surface(captured.value)
    assert readiness_marker not in surface
    assert database_marker not in surface
    assert provider.close_calls == 1
    assert database.close_calls == 1
    assert {str(error) for error in captured.value.exceptions} == {
        "readiness cleanup failed with ValueError",
        "database cleanup failed with RuntimeError",
    }


@pytest.mark.asyncio
async def test_body_failure_and_sanitized_cleanup_failure_are_both_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_marker = "postgresql://cleanup-secret-marker"
    database = TrackingDatabase(close_error=RuntimeError(database_marker))
    provider = StaticReadinessProvider()
    monkeypatch.setattr("rag_service.main.Database", TrackingDatabaseFactory(database))
    monkeypatch.setattr(
        "rag_service.main.LiveReadinessProvider",
        TrackingReadinessFactory(provider),
    )
    app = create_app(settings=Settings(_env_file=None))
    body_error = LookupError("body failure")

    with pytest.raises(BaseExceptionGroup) as captured:
        async with app.router.lifespan_context(app):
            raise body_error

    assert captured.value.exceptions[0] is body_error
    assert database_marker not in exception_surface(captured.value)
