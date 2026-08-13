import os
import subprocess
import sys
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple, cast

import httpx
import psycopg
import pytest
import pytest_asyncio
from minio import Minio
from qdrant_client import QdrantClient
from redis import Redis
from sqlalchemy.ext.asyncio import AsyncSession
from testcontainers.core.container import DockerContainer
from testcontainers.postgres import PostgresContainer
from urllib3 import PoolManager, Timeout

from fixtures.provider_stub import RunningProviderStub, running_provider_https_stub
from rag_service.db.session import Database

BASELINE_REVISION = "20260723_0001"
HEAD_REVISION = "20260730_0005"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
ALEMBIC_CONFIG = REPOSITORY_ROOT / "alembic.ini"
INCREMENT_2_TABLES: frozenset[str] = frozenset(
    {
        "api_key_knowledge_base_scopes",
        "api_key_query_profile_scopes",
        "api_keys",
        "audit_events",
        "document_index_states",
        "document_versions",
        "documents",
        "idempotency_records",
        "index_generation_creation_requests",
        "index_generation_cleanup_claims",
        "jobs",
        "knowledge_base_index_generations",
        "knowledge_base_mutations",
        "knowledge_bases",
        "model_profile_fallbacks",
        "model_profiles",
        "provider_configs",
        "provider_credentials",
        "provider_usage",
        "query_logs",
        "query_profiles",
        "sparse_profiles",
        "document_upload_idempotency",
    }
)
INCREMENT_2_TRUNCATE_SQL = (
    "TRUNCATE TABLE "
    + ", ".join(f'"{table_name}"' for table_name in sorted(INCREMENT_2_TABLES))
    + " CASCADE"
)
AsyncTeardownStep = tuple[str, Callable[[], Awaitable[None]]]
SyncTeardownStep = tuple[str, Callable[[], None]]
AsyncFixtureLifecycle = Callable[
    [tuple[AsyncTeardownStep, ...], Callable[[], None]],
    AbstractAsyncContextManager[None],
]
SyncFixtureTeardown = Callable[[tuple[SyncTeardownStep, ...], Callable[[], None]], None]
DatabaseSessionLifecycle = Callable[[], AbstractAsyncContextManager[AsyncSession]]
ServiceContainerLifecycle = Callable[..., Iterator[Any]]
_MISSING_SYNC_RESOURCE = object()
_MINIO_FIXTURE_ACCESS_KEY = "rag-fixture"
_MINIO_FIXTURE_SECRET_KEY = "rag-fixture-password-not-for-production"
_MINIO_FIXTURE_BUCKET = "rag-integration"
_MINIO_HTTP_TIMEOUT_SECONDS = 1.0


class PostgresUrls(NamedTuple):
    async_url: str
    sync_url: str

    def __repr__(self) -> str:
        return "PostgresUrls(<redacted>)"


@dataclass(frozen=True, slots=True, repr=False)
class MinioConnection:
    endpoint: str
    url: str
    access_key: str
    secret_key: str
    bucket: str
    secure: bool = False

    def __repr__(self) -> str:
        return "MinioConnection(<redacted>)"


@dataclass(frozen=True, slots=True, repr=False)
class RedisConnection:
    url: str

    def __repr__(self) -> str:
        return "RedisConnection(<redacted>)"


@dataclass(frozen=True, slots=True, repr=False)
class QdrantConnection:
    url: str

    def __repr__(self) -> str:
        return "QdrantConnection(<redacted>)"


@dataclass(frozen=True, slots=True, repr=False)
class FullStackConnections:
    postgres: PostgresUrls
    minio: MinioConnection
    redis: RedisConnection
    qdrant: QdrantConnection
    provider: RunningProviderStub

    def __repr__(self) -> str:
        return "FullStackConnections(<redacted>)"


MinioConnectionLifecycle = Callable[[DockerContainer], Iterator[MinioConnection]]
MinioHttpPoolFactory = Callable[[], PoolManager]


class FixtureTeardownError(RuntimeError):
    def __init__(self, resource: str, error: BaseException) -> None:
        self.resource = resource
        self.exception_type = type(error).__name__
        super().__init__(f"{resource} teardown failed ({self.exception_type})")

    @classmethod
    def from_exception_type(
        cls,
        resource: str,
        exception_type: str,
    ) -> "FixtureTeardownError":
        instance = cls.__new__(cls)
        instance.resource = resource
        instance.exception_type = exception_type
        RuntimeError.__init__(
            instance,
            f"{resource} teardown failed ({exception_type})",
        )
        return instance


class FixturePrimaryError(RuntimeError):
    def __init__(self, resource: str, error: BaseException) -> None:
        self.resource = resource
        self.exception_type = type(error).__name__
        super().__init__(f"{resource} primary failed ({self.exception_type})")


def _copy_safe_teardown_failures(
    error: BaseException,
) -> list[FixtureTeardownError] | None:
    if isinstance(error, FixtureTeardownError):
        return [
            FixtureTeardownError.from_exception_type(
                error.resource,
                error.exception_type,
            )
        ]
    if isinstance(error, BaseExceptionGroup):
        failures: list[FixtureTeardownError] = []
        for nested_error in error.exceptions:
            nested_failures = _copy_safe_teardown_failures(nested_error)
            if nested_failures is None:
                return None
            failures.extend(nested_failures)
        return failures
    return None


def _sanitize_teardown_failure(
    resource: str,
    error: BaseException,
) -> list[FixtureTeardownError]:
    safe_failures = _copy_safe_teardown_failures(error)
    if safe_failures is not None:
        return safe_failures
    return [FixtureTeardownError(resource, error)]


def _raise_teardown_failures(message: str, failures: list[FixtureTeardownError]) -> None:
    if failures:
        raise ExceptionGroup(message, failures)


def _truncate_increment_2(
    sync_url: str,
    *,
    connect: Callable[..., Any] = psycopg.connect,
) -> None:
    failures: list[FixtureTeardownError] = []
    cleanup_connection: Any | None = None
    try:
        cleanup_connection = connect(sync_url, autocommit=True)
    except BaseException as error:
        failures.append(FixtureTeardownError("increment_2.cleanup_connection", error))
    else:
        try:
            cleanup_connection.execute(INCREMENT_2_TRUNCATE_SQL)
        except BaseException as error:
            failures.append(FixtureTeardownError("increment_2.truncate", error))
        try:
            cleanup_connection.close()
        except BaseException as error:
            failures.append(FixtureTeardownError("increment_2.cleanup_connection.close", error))

    _raise_teardown_failures("increment 2 cleanup failed", failures)


async def _collect_async_teardown_failures(
    steps: tuple[AsyncTeardownStep, ...],
    cleanup: Callable[[], None],
) -> list[FixtureTeardownError]:
    failures: list[FixtureTeardownError] = []
    for resource, action in steps:
        try:
            await action()
        except BaseException as error:
            failures.extend(_sanitize_teardown_failure(resource, error))
    try:
        cleanup()
    except BaseException as error:
        failures.extend(_sanitize_teardown_failure("increment_2.fresh_cleanup", error))
    return failures


@asynccontextmanager
async def _async_fixture_lifecycle(
    steps: tuple[AsyncTeardownStep, ...],
    cleanup: Callable[[], None],
) -> AsyncIterator[None]:
    primary_error: BaseException | None = None
    try:
        yield
    except BaseException as error:
        primary_error = error

    failures = await _collect_async_teardown_failures(steps, cleanup)
    if primary_error is not None:
        if failures:
            raise BaseExceptionGroup(
                "fixture body and teardown failed",
                [primary_error, *failures],
            )
        raise primary_error.with_traceback(primary_error.__traceback__)
    _raise_teardown_failures("fixture teardown failed", failures)


def _collect_sync_teardown_failures(
    steps: tuple[SyncTeardownStep, ...],
    cleanup: Callable[[], None],
) -> list[FixtureTeardownError]:
    failures: list[FixtureTeardownError] = []
    for resource, action in steps:
        try:
            action()
        except BaseException as error:
            failures.extend(_sanitize_teardown_failure(resource, error))
    try:
        cleanup()
    except BaseException as error:
        failures.extend(_sanitize_teardown_failure("increment_2.fresh_cleanup", error))
    return failures


def _run_sync_fixture_teardown(
    steps: tuple[SyncTeardownStep, ...],
    cleanup: Callable[[], None],
) -> None:
    failures = _collect_sync_teardown_failures(steps, cleanup)
    _raise_teardown_failures("fixture teardown failed", failures)


def _raise_sync_lifecycle_failures(
    primary_error: FixturePrimaryError | None,
    teardown_failures: list[FixtureTeardownError],
) -> None:
    failures: list[Exception] = []
    if primary_error is not None:
        failures.append(primary_error)
    failures.extend(teardown_failures)
    if failures:
        raise ExceptionGroup("sync fixture lifecycle failed", failures)


def _sync_fixture_lifecycle[SyncResource](
    *,
    setup_resource: str,
    setup: Callable[[], SyncResource],
    teardown_steps: Callable[[SyncResource], tuple[SyncTeardownStep, ...]],
    cleanup: Callable[[], None],
) -> Iterator[SyncResource]:
    primary_error: FixturePrimaryError | None = None
    resource: SyncResource | object = _MISSING_SYNC_RESOURCE
    try:
        resource = setup()
    except BaseException as error:
        primary_error = FixturePrimaryError(setup_resource, error)

    steps: tuple[SyncTeardownStep, ...] = ()
    step_factory_failures: list[FixtureTeardownError] = []
    if resource is not _MISSING_SYNC_RESOURCE:
        typed_resource = cast(SyncResource, resource)
        try:
            yield typed_resource
        except BaseException as error:
            primary_error = FixturePrimaryError("fixture.body", error)
        try:
            steps = teardown_steps(typed_resource)
        except BaseException as error:
            step_factory_failures.extend(
                _sanitize_teardown_failure("connection.teardown_steps", error)
            )
        del typed_resource

    teardown_failures = [
        *step_factory_failures,
        *_collect_sync_teardown_failures(steps, cleanup),
    ]
    del setup, teardown_steps, cleanup, resource, steps, step_factory_failures
    _raise_sync_lifecycle_failures(primary_error, teardown_failures)


@asynccontextmanager
async def _database_lifecycle(async_url: str, sync_url: str) -> AsyncIterator[Database]:
    database: Database | None = None

    async def close_database() -> None:
        if database is not None:
            await database.close()

    async with _async_fixture_lifecycle(
        (("database.close", close_database),),
        lambda: _truncate_increment_2(sync_url),
    ):
        database = Database(async_url)
        yield database


@asynccontextmanager
async def _database_session_lifecycle(
    async_url: str,
    sync_url: str,
) -> AsyncIterator[AsyncSession]:
    database: Database | None = None
    session_context: AbstractAsyncContextManager[AsyncSession] | None = None
    session: AsyncSession | None = None
    session_entered = False

    async def rollback_session() -> None:
        if session is not None:
            await session.rollback()

    async def close_session_context() -> None:
        if session_context is not None and session_entered:
            await session_context.__aexit__(None, None, None)

    async def close_database() -> None:
        if database is not None:
            await database.close()

    async with _async_fixture_lifecycle(
        (
            ("db_session.rollback", rollback_session),
            ("db_session.context_close", close_session_context),
            ("database.close", close_database),
        ),
        lambda: _truncate_increment_2(sync_url),
    ):
        database = Database(async_url)
        session_context = database.sessions()
        session = await session_context.__aenter__()
        session_entered = True
        yield session


def _sync_connection_lifecycle(
    sync_url: str,
    *,
    autocommit: bool,
    rollback_on_exit: bool,
    connect: Callable[..., psycopg.Connection[Any]] = psycopg.connect,
    cleanup_connect: Callable[..., Any] = psycopg.connect,
) -> Iterator[psycopg.Connection[Any]]:
    def connection_teardown_steps(
        connection: psycopg.Connection[Any],
    ) -> tuple[SyncTeardownStep, ...]:
        close_step: SyncTeardownStep = ("connection.close", connection.close)
        if not rollback_on_exit:
            return (close_step,)
        return (
            ("connection.rollback", connection.rollback),
            close_step,
        )

    def setup_connection(
        url: str = sync_url,
        connect_fn: Callable[..., psycopg.Connection[Any]] = connect,
    ) -> psycopg.Connection[Any]:
        return connect_fn(url, autocommit=autocommit)

    def cleanup_database(
        url: str = sync_url,
        connect_fn: Callable[..., Any] = cleanup_connect,
    ) -> None:
        _truncate_increment_2(url, connect=connect_fn)

    lifecycle = _sync_fixture_lifecycle(
        setup_resource="connection.setup",
        setup=setup_connection,
        teardown_steps=connection_teardown_steps,
        cleanup=cleanup_database,
    )
    del (
        sync_url,
        connect,
        cleanup_connect,
        connection_teardown_steps,
        setup_connection,
        cleanup_database,
    )
    yield from lifecycle


def _postgres_urls(postgres: PostgresContainer) -> PostgresUrls:
    host = postgres.get_container_host_ip()
    port = postgres.get_exposed_port(5432)
    async_url = f"postgresql+psycopg://rag:rag-test-password@{host}:{port}/rag"
    sync_url = f"postgresql://rag:rag-test-password@{host}:{port}/rag"
    return PostgresUrls(async_url, sync_url)


def _run_alembic(async_url: str, *arguments: str) -> None:
    env = os.environ | {
        "RAG_DATABASE_URL": async_url,
        "RAG_ENVIRONMENT": "test",
        "TESTCONTAINERS_RYUK_DISABLED": "true",
    }
    subprocess.run(
        [sys.executable, "-m", "alembic", "-c", str(ALEMBIC_CONFIG), *arguments],
        check=True,
        capture_output=True,
        cwd=REPOSITORY_ROOT,
        env=env,
        text=True,
        timeout=120,
    )


def _wait_for_readiness(
    resource: str,
    probe: Callable[[], bool],
    *,
    timeout_seconds: float = 30,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_failure = "not_ready"
    while time.monotonic() < deadline:
        try:
            if probe():
                return
            last_failure = "not_ready"
        except Exception as error:
            last_failure = type(error).__name__
        time.sleep(0.05)
    raise RuntimeError(f"{resource} readiness failed ({last_failure})") from None


def _stop_service_container(
    resource: str,
    container: DockerContainer,
) -> list[FixtureTeardownError]:
    try:
        container.stop()
    except BaseException as error:
        return _sanitize_teardown_failure(f"{resource}.stop", error)
    return []


def _service_container_lifecycle[Connection](
    *,
    resource: str,
    container: DockerContainer,
    connection: Callable[[DockerContainer], Connection],
    readiness: Callable[[Connection], bool],
) -> Iterator[Connection]:
    resolved: Connection | object = _MISSING_SYNC_RESOURCE
    primary_error: BaseException | None = None
    try:
        container.start()
        resolved = connection(container)
        _wait_for_readiness(resource, lambda: readiness(resolved))
    except BaseException as error:
        primary_error = (
            error
            if not isinstance(error, Exception)
            else FixturePrimaryError(
                f"{resource}.setup",
                error,
            )
        )

    if primary_error is not None:
        teardown_failures = _stop_service_container(resource, container)
        if teardown_failures:
            raise BaseExceptionGroup(
                f"{resource} fixture setup and teardown failed",
                [primary_error, *teardown_failures],
            ) from None
        if isinstance(primary_error, FixturePrimaryError):
            raise ExceptionGroup("sync fixture lifecycle failed", [primary_error]) from None
        raise primary_error.with_traceback(primary_error.__traceback__) from None

    body_error: BaseException | None = None
    try:
        yield cast(Connection, resolved)
    except BaseException as error:
        body_error = (
            error
            if not isinstance(error, Exception)
            else FixturePrimaryError(
                "fixture.body",
                error,
            )
        )

    teardown_failures = _stop_service_container(resource, container)
    if body_error is not None:
        if teardown_failures:
            raise BaseExceptionGroup(
                f"{resource} fixture body and teardown failed",
                [body_error, *teardown_failures],
            ) from None
        raise body_error.with_traceback(body_error.__traceback__) from None
    _raise_teardown_failures(f"{resource} fixture teardown failed", teardown_failures)


def _container_host_port(container: DockerContainer, port: int) -> tuple[str, int]:
    return container.get_container_host_ip(), int(container.get_exposed_port(port))


def _new_minio_container() -> DockerContainer:
    return (
        DockerContainer("minio/minio:RELEASE.2025-09-07T16-13-09Z")
        .with_env("MINIO_ROOT_USER", _MINIO_FIXTURE_ACCESS_KEY)
        .with_env("MINIO_ROOT_PASSWORD", _MINIO_FIXTURE_SECRET_KEY)
        .with_command(["server", "/data", "--console-address", ":9001"])
        .with_exposed_ports(9000)
    )


def _minio_connection_from_container(started: DockerContainer) -> MinioConnection:
    host, port = _container_host_port(started, 9000)
    return MinioConnection(
        endpoint=f"{host}:{port}",
        url=f"http://{host}:{port}",
        access_key=_MINIO_FIXTURE_ACCESS_KEY,
        secret_key=_MINIO_FIXTURE_SECRET_KEY,
        bucket=_MINIO_FIXTURE_BUCKET,
    )


def _new_minio_http_pool() -> PoolManager:
    return PoolManager(
        timeout=Timeout(
            connect=_MINIO_HTTP_TIMEOUT_SECONDS,
            read=_MINIO_HTTP_TIMEOUT_SECONDS,
        )
    )


def _minio_readiness(details: MinioConnection) -> bool:
    response = httpx.get(
        f"{details.url}/minio/health/ready",
        timeout=1,
        trust_env=False,
    )
    if response.status_code != 200:
        return False
    pool = _new_minio_http_pool()
    try:
        client = Minio(
            details.endpoint,
            access_key=details.access_key,
            secret_key=details.secret_key,
            secure=details.secure,
            http_client=pool,
        )
        if not client.bucket_exists(details.bucket):
            client.make_bucket(details.bucket)
        return client.bucket_exists(details.bucket)
    finally:
        pool.clear()


def _minio_connection_lifecycle(container: DockerContainer) -> Iterator[MinioConnection]:
    yield from _service_container_lifecycle(
        resource="minio",
        container=container,
        connection=_minio_connection_from_container,
        readiness=_minio_readiness,
    )


@pytest.fixture(scope="session")
def minio_connection() -> Iterator[MinioConnection]:
    yield from _minio_connection_lifecycle(_new_minio_container())


@pytest.fixture(scope="session")
def minio_http_pool_factory() -> MinioHttpPoolFactory:
    return _new_minio_http_pool


@pytest.fixture(scope="session")
def redis_connection() -> Iterator[RedisConnection]:
    container = (
        DockerContainer("redis:8.8.0-alpine")
        .with_command(["redis-server", "--appendonly", "no"])
        .with_exposed_ports(6379)
    )

    def connection(started: DockerContainer) -> RedisConnection:
        host, port = _container_host_port(started, 6379)
        return RedisConnection(url=f"redis://{host}:{port}/0")

    def readiness(details: RedisConnection) -> bool:
        client = Redis.from_url(
            details.url,
            socket_connect_timeout=1,
            socket_timeout=1,
        )
        try:
            return bool(client.ping())
        finally:
            client.close()

    yield from _service_container_lifecycle(
        resource="redis",
        container=container,
        connection=connection,
        readiness=readiness,
    )


@pytest.fixture(scope="session")
def qdrant_connection() -> Iterator[QdrantConnection]:
    container = DockerContainer("qdrant/qdrant:v1.18.3").with_exposed_ports(6333)

    def connection(started: DockerContainer) -> QdrantConnection:
        host, port = _container_host_port(started, 6333)
        return QdrantConnection(url=f"http://{host}:{port}")

    def readiness(details: QdrantConnection) -> bool:
        response = httpx.get(f"{details.url}/readyz", timeout=1, trust_env=False)
        if response.status_code != 200:
            return False
        client = QdrantClient(url=details.url, timeout=1)
        try:
            client.get_collections()
        finally:
            client.close()
        return True

    yield from _service_container_lifecycle(
        resource="qdrant",
        container=container,
        connection=connection,
        readiness=readiness,
    )


@pytest.fixture
def provider_https_stub(tmp_path: Path) -> Iterator[RunningProviderStub]:
    with running_provider_https_stub(tmp_path) as provider:
        yield provider


@pytest.fixture
def full_stack_connections(
    postgres_urls: PostgresUrls,
    minio_connection: MinioConnection,
    redis_connection: RedisConnection,
    qdrant_connection: QdrantConnection,
    provider_https_stub: RunningProviderStub,
) -> Iterator[FullStackConnections]:
    yield FullStackConnections(
        postgres=postgres_urls,
        minio=minio_connection,
        redis=redis_connection,
        qdrant=qdrant_connection,
        provider=provider_https_stub,
    )


@pytest.fixture(scope="session")
def postgres_urls() -> Iterator[PostgresUrls]:
    os.environ.setdefault("TESTCONTAINERS_RYUK_DISABLED", "true")
    with PostgresContainer(
        image="postgres:18.4-alpine",
        username="rag",
        password="rag-test-password",
        dbname="rag",
    ) as postgres:
        yield _postgres_urls(postgres)


@pytest.fixture
def migration_postgres_urls() -> Iterator[PostgresUrls]:
    """A fresh database reserved for migration round-trip tests."""
    os.environ.setdefault("TESTCONTAINERS_RYUK_DISABLED", "true")
    with PostgresContainer(
        image="postgres:18.4-alpine",
        username="rag",
        password="rag-test-password",
        dbname="rag",
    ) as postgres:
        yield _postgres_urls(postgres)


@pytest.fixture(scope="session")
def upgrade_head(postgres_urls: PostgresUrls) -> Callable[[], None]:
    async_url, _sync_url = postgres_urls

    def upgrade() -> None:
        _run_alembic(async_url, "upgrade", "head")

    return upgrade


# Engines are function-owned so creation, session use, and disposal stay on one
# pytest-asyncio loop; only the loop-agnostic container URLs remain session-scoped.
@pytest_asyncio.fixture(loop_scope="function")
async def migrated_database(
    postgres_urls: PostgresUrls,
    upgrade_head: Callable[[], None],
) -> AsyncIterator[Database]:
    upgrade_head()
    async_url, sync_url = postgres_urls
    async with _database_lifecycle(async_url, sync_url) as database:
        yield database


@pytest.fixture
def db_session_lifecycle(
    postgres_urls: PostgresUrls,
    upgrade_head: Callable[[], None],
) -> DatabaseSessionLifecycle:
    # This context owns its Database and performs exactly one ordered cleanup;
    # db_session intentionally does not stack cleanup on migrated_database.
    upgrade_head()
    async_url, sync_url = postgres_urls

    def lifecycle() -> AbstractAsyncContextManager[AsyncSession]:
        return _database_session_lifecycle(async_url, sync_url)

    return lifecycle


@pytest_asyncio.fixture(loop_scope="function")
async def db_session(
    db_session_lifecycle: DatabaseSessionLifecycle,
) -> AsyncIterator[AsyncSession]:
    async with db_session_lifecycle() as session:
        yield session


@pytest.fixture(scope="session")
def increment_2_tables() -> frozenset[str]:
    return INCREMENT_2_TABLES


@pytest.fixture(scope="session")
def increment_2_truncate_sql() -> str:
    return INCREMENT_2_TRUNCATE_SQL


@pytest.fixture
def fixture_async_lifecycle() -> AsyncFixtureLifecycle:
    return _async_fixture_lifecycle


@pytest.fixture
def fixture_sync_teardown() -> SyncFixtureTeardown:
    return _run_sync_fixture_teardown


@pytest.fixture
def fixture_sync_lifecycle() -> Callable[..., Iterator[Any]]:
    return _sync_fixture_lifecycle


@pytest.fixture
def fixture_sync_connection_lifecycle() -> Callable[..., Iterator[Any]]:
    return _sync_connection_lifecycle


@pytest.fixture
def fixture_service_container_lifecycle() -> ServiceContainerLifecycle:
    return _service_container_lifecycle


@pytest.fixture
def fixture_minio_connection_lifecycle() -> MinioConnectionLifecycle:
    return _minio_connection_lifecycle


@pytest.fixture
def fixture_fresh_cleanup() -> Callable[..., None]:
    return _truncate_increment_2


@pytest.fixture
def run_alembic() -> Callable[[str, str], None]:
    def run(async_url: str, target: str) -> None:
        _run_alembic(async_url, target.split()[0], target.split()[1])

    return run


@pytest.fixture
def migrated_sync_connection(
    postgres_urls: PostgresUrls,
    upgrade_head: Callable[[], None],
) -> Iterator[psycopg.Connection[Any]]:
    upgrade_head()
    _async_url, sync_url = postgres_urls
    lifecycle = _sync_connection_lifecycle(
        sync_url,
        autocommit=False,
        rollback_on_exit=True,
    )
    del _async_url, sync_url, postgres_urls, upgrade_head
    yield from lifecycle


@pytest.fixture
def migrated_autocommit_sync_connection(
    postgres_urls: PostgresUrls,
    upgrade_head: Callable[[], None],
) -> Iterator[psycopg.Connection[Any]]:
    upgrade_head()
    _async_url, sync_url = postgres_urls
    lifecycle = _sync_connection_lifecycle(
        sync_url,
        autocommit=True,
        rollback_on_exit=False,
    )
    del _async_url, sync_url, postgres_urls, upgrade_head
    yield from lifecycle
