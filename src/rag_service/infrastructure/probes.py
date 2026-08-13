import asyncio
from collections.abc import Awaitable, Callable, Collection
from time import perf_counter
from typing import Any, Protocol, Self, cast
from urllib.parse import urlparse
from uuid import UUID

import httpx
from minio import Minio
from redis.asyncio import Redis
from sqlalchemy import select
from urllib3 import PoolManager, Timeout

from rag_service.config import Settings
from rag_service.db.models.knowledge_bases import KnowledgeBaseIndexGeneration
from rag_service.db.models.providers import ProviderCredential
from rag_service.db.session import Database
from rag_service.providers.credentials import EncryptedProviderCredential
from rag_service.providers.services import provider_credential_keyring_from_settings
from rag_service.readiness import (
    INGEST_GENERATION_STATUSES,
    NOT_CHECKED_ERROR,
    RETRIEVE_GENERATION_STATUSES,
    ComponentStatus,
    ReadinessScope,
    ReadinessSnapshot,
    ReferencedCredential,
    ReferencedCredentialValidator,
    provider_credential_keyring_fingerprint,
)


class DatabaseProbe(Protocol):
    async def ping(self) -> None: ...

    async def close(self) -> None: ...


class CredentialDatabaseProbe(DatabaseProbe, Protocol):
    def sessions(self) -> Any: ...


class RedisProbe(Protocol):
    async def ping(self) -> bool: ...

    async def aclose(self) -> None: ...


class MinioProbe(Protocol):
    def bucket_exists(self, bucket_name: str) -> bool: ...


class ReadinessCleanupError(RuntimeError):
    """Sanitized failure raised while closing an owned readiness resource."""


class ReadinessConstructionError(RuntimeError):
    """Sanitized failure raised while constructing an owned readiness resource."""


class DatabaseReferencedCredentialLoader:
    """Load only credentials referenced by generations in the requested states."""

    def __init__(self, database: CredentialDatabaseProbe) -> None:
        self._database = database

    async def __call__(self, statuses: Collection[str]) -> tuple[ReferencedCredential, ...]:
        selected_statuses = sorted(frozenset(statuses))
        if not selected_statuses:
            raise ValueError("generation statuses are required")
        async with self._database.sessions() as session:
            snapshots = (
                await session.scalars(
                    select(KnowledgeBaseIndexGeneration.embedding_config_snapshot).where(
                        KnowledgeBaseIndexGeneration.status.in_(selected_statuses)
                    )
                )
            ).all()
            credential_ids: set[UUID] = set()
            for snapshot in snapshots:
                if type(snapshot) is not dict:
                    raise ValueError("generation credential snapshot is invalid")
                raw_credential_id = snapshot.get("credential_id")
                if type(raw_credential_id) is not str:
                    raise ValueError("generation credential snapshot is invalid")
                credential_ids.add(UUID(raw_credential_id))

            if not credential_ids:
                return ()

            ordered_ids = sorted(credential_ids, key=str)
            rows = (
                await session.scalars(
                    select(ProviderCredential).where(ProviderCredential.id.in_(ordered_ids))
                )
            ).all()

        by_id = {row.id: row for row in rows}
        if set(by_id) != credential_ids:
            raise ValueError("referenced provider credential is missing")
        return tuple(
            ReferencedCredential(
                credential_id=credential_id,
                resource_revision=by_id[credential_id].resource_revision,
                encrypted=EncryptedProviderCredential(
                    ciphertext=bytes(by_id[credential_id].ciphertext),
                    nonce=bytes(by_id[credential_id].nonce),
                    key_version=by_id[credential_id].key_version,
                    algorithm=by_id[credential_id].algorithm,
                ),
            )
            for credential_id in ordered_ids
        )


def _consume_task_exception(task: asyncio.Task[object]) -> None:
    if not task.cancelled():
        task.exception()


def _sanitize_cleanup_failure(resource: str, error: BaseException) -> ReadinessCleanupError:
    return ReadinessCleanupError(f"{resource} cleanup failed with {type(error).__name__}")


def _sanitize_construction_failure(
    resource: str,
    error: BaseException,
) -> ReadinessConstructionError:
    return ReadinessConstructionError(f"{resource} construction failed with {type(error).__name__}")


async def _invoke_cleanup(close: Callable[[], Awaitable[object]]) -> None:
    await close()


def _task_cleanup_failure(
    resource: str,
    task: asyncio.Task[None],
) -> ReadinessCleanupError | None:
    try:
        task.result()
    except BaseException as error:
        return _sanitize_cleanup_failure(resource, error)
    return None


async def _attempt_cleanup_before(
    resource: str,
    close: Callable[[], Awaitable[object]],
    deadline: float,
) -> ReadinessCleanupError | None:
    close_task: asyncio.Task[None] = asyncio.create_task(_invoke_cleanup(close))
    close_task.add_done_callback(_consume_task_exception)
    remaining = max(0.0, deadline - asyncio.get_running_loop().time())
    try:
        done, _pending = await asyncio.wait({close_task}, timeout=remaining)
    except BaseException:
        close_task.cancel()
        raise
    if close_task not in done:
        close_task.cancel()
        return ReadinessCleanupError(f"{resource} cleanup failed with TimeoutError")
    return _task_cleanup_failure(resource, close_task)


async def _rollback_registered_resources(
    closers: tuple[tuple[str, Callable[[], Awaitable[object]]], ...],
    timeout_seconds: float,
) -> list[ReadinessCleanupError]:
    failures: list[ReadinessCleanupError] = []
    loop = asyncio.get_running_loop()
    for resource, close in reversed(closers):
        failure = await _attempt_cleanup_before(
            resource,
            close,
            loop.time() + timeout_seconds,
        )
        if failure is not None:
            failures.append(failure)
    return failures


async def _drain_registered_rollback(
    closers: tuple[tuple[str, Callable[[], Awaitable[object]]], ...],
    timeout_seconds: float,
) -> tuple[list[ReadinessCleanupError], asyncio.CancelledError | None]:
    rollback_task = asyncio.create_task(_rollback_registered_resources(closers, timeout_seconds))
    rollback_task.add_done_callback(_consume_task_exception)
    cancellation: asyncio.CancelledError | None = None
    while not rollback_task.done():
        try:
            await asyncio.shield(rollback_task)
        except asyncio.CancelledError as error:
            if cancellation is None:
                cancellation = error
        except BaseException:
            break

    if rollback_task.cancelled():
        return [ReadinessCleanupError("readiness cleanup failed with CancelledError")], cancellation
    try:
        return rollback_task.result(), cancellation
    except BaseException as error:
        return [_sanitize_cleanup_failure("readiness", error)], cancellation


def _construction_failure_result(
    construction_failure: ReadinessConstructionError,
    cleanup_failures: list[ReadinessCleanupError],
    cancellation: asyncio.CancelledError | None,
) -> BaseException:
    failure_group = ExceptionGroup(
        "Failed to construct readiness resources",
        [construction_failure, *cleanup_failures],
    )
    if cancellation is None:
        return failure_group
    return BaseExceptionGroup(
        "Readiness construction was cancelled during rollback",
        [cancellation, failure_group],
    )


async def _collect_cleanup_failures(
    closers: list[tuple[str, Awaitable[object]]],
) -> list[ReadinessCleanupError]:
    results = await asyncio.gather(
        *(closer for _resource, closer in closers),
        return_exceptions=True,
    )
    return [
        _sanitize_cleanup_failure(resource, result)
        for (resource, _closer), result in zip(closers, results, strict=True)
        if isinstance(result, BaseException)
    ]


def _validate_minio_url(url: str) -> tuple[str, bool]:
    try:
        parsed = urlparse(url)
    except ValueError as error:
        raise ValueError("Invalid MinIO URL: malformed URL") from error

    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Invalid MinIO URL: scheme must be http or https")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("Invalid MinIO URL: userinfo is not allowed")
    if parsed.path not in {"", "/"}:
        raise ValueError("Invalid MinIO URL: path must be empty or /")
    if parsed.params:
        raise ValueError("Invalid MinIO URL: params are not allowed")
    if parsed.query:
        raise ValueError("Invalid MinIO URL: query is not allowed")
    if parsed.fragment:
        raise ValueError("Invalid MinIO URL: fragment is not allowed")

    try:
        hostname = parsed.hostname
        _ = parsed.port
    except ValueError as error:
        raise ValueError("Invalid MinIO URL: malformed host or port") from error
    if hostname is None:
        raise ValueError("Invalid MinIO URL: hostname is required")
    if parsed.netloc.endswith(":"):
        raise ValueError("Invalid MinIO URL: malformed port")

    return parsed.netloc, parsed.scheme == "https"


class LiveReadinessProvider:
    """Probe dependencies while owning only clients it constructs."""

    def __init__(
        self,
        settings: Settings,
        *,
        database: DatabaseProbe,
        redis_client: RedisProbe,
        minio_client: MinioProbe,
        http_client: httpx.AsyncClient,
        credential_validator: ReferencedCredentialValidator | None = None,
        _minio_http: PoolManager | None = None,
        _owns_database: bool = False,
        _owns_redis: bool = False,
        _owns_minio: bool = False,
        _owns_http: bool = False,
    ) -> None:
        self._settings = settings
        self._close_lock = asyncio.Lock()
        self._close_task: asyncio.Task[None] | None = None
        self._database = database
        self._redis = redis_client
        self._minio = minio_client
        self._http = http_client
        self._credential_validator = credential_validator or ReferencedCredentialValidator(
            load_referenced_credentials=DatabaseReferencedCredentialLoader(
                cast(CredentialDatabaseProbe, database)
            )
        )
        self._minio_http = _minio_http
        self._owns_database = _owns_database
        self._owns_redis = _owns_redis
        self._owns_minio = _owns_minio
        self._owns_http = _owns_http

    @classmethod
    async def create(
        cls,
        settings: Settings,
        *,
        database: DatabaseProbe | None = None,
        redis_client: RedisProbe | None = None,
        minio_client: MinioProbe | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> Self:
        result = await cls._construct(
            settings,
            database=database,
            redis_client=redis_client,
            minio_client=minio_client,
            http_client=http_client,
        )
        if isinstance(result, BaseException):
            raise result from None
        return result

    @classmethod
    async def _construct(
        cls,
        settings: Settings,
        *,
        database: DatabaseProbe | None,
        redis_client: RedisProbe | None,
        minio_client: MinioProbe | None,
        http_client: httpx.AsyncClient | None,
    ) -> Self | BaseException:
        try:
            minio_endpoint, minio_secure = _validate_minio_url(settings.minio_url)
        except ValueError as error:
            return ValueError(str(error))

        rollback: list[tuple[str, Callable[[], Awaitable[object]]]] = []
        stage = "postgres"
        try:
            owns_database = database is None
            selected_database = Database.from_settings(settings) if database is None else database
            if owns_database:
                rollback.append(("postgres", selected_database.close))

            stage = "redis"
            owns_redis = redis_client is None
            selected_redis = (
                cast(
                    RedisProbe,
                    Redis.from_url(
                        settings.redis_url.get_secret_value(),
                        socket_connect_timeout=settings.readiness_timeout_seconds,
                        socket_timeout=settings.readiness_timeout_seconds,
                    ),
                )
                if redis_client is None
                else redis_client
            )
            if owns_redis:
                rollback.append(("redis", selected_redis.aclose))

            owns_minio = minio_client is None
            minio_http: PoolManager | None = None
            if minio_client is None:
                stage = "minio_pool"
                minio_http = PoolManager(
                    timeout=Timeout(
                        connect=settings.readiness_timeout_seconds,
                        read=settings.readiness_timeout_seconds,
                    ),
                    retries=False,
                )
                rollback.append(("minio", lambda: asyncio.to_thread(minio_http.clear)))

                stage = "minio"
                selected_minio = cast(
                    MinioProbe,
                    Minio(
                        minio_endpoint,
                        access_key=settings.minio_access_key,
                        secret_key=settings.minio_secret_key.get_secret_value(),
                        secure=minio_secure,
                        http_client=minio_http,
                    ),
                )
            else:
                selected_minio = minio_client

            stage = "http"
            owns_http = http_client is None
            selected_http = (
                httpx.AsyncClient(timeout=settings.readiness_timeout_seconds)
                if http_client is None
                else http_client
            )
            if owns_http:
                rollback.append(("http", selected_http.aclose))
        except BaseException as error:
            construction_failure = _sanitize_construction_failure(stage, error)
            cleanup_failures, cancellation = await _drain_registered_rollback(
                tuple(rollback),
                settings.shutdown_timeout_seconds,
            )
            return _construction_failure_result(
                construction_failure,
                cleanup_failures,
                cancellation,
            )

        return cls(
            settings,
            database=selected_database,
            redis_client=selected_redis,
            minio_client=selected_minio,
            http_client=selected_http,
            _minio_http=minio_http,
            _owns_database=owns_database,
            _owns_redis=owns_redis,
            _owns_minio=owns_minio,
            _owns_http=owns_http,
        )

    async def snapshot(
        self,
        scope: ReadinessScope = ReadinessScope.ALL,
    ) -> ReadinessSnapshot:
        if type(scope) is not ReadinessScope:
            raise ValueError("readiness scope is invalid")
        checks: dict[str, Callable[[], Awaitable[object]]] = {
            "postgres": self._database.ping,
            "qdrant": self._check_qdrant,
            "redis": self._check_redis,
            "minio": self._check_minio,
            "ingest_credentials": lambda: self._check_referenced_credentials(
                INGEST_GENERATION_STATUSES
            ),
            "retrieve_credentials": lambda: self._check_referenced_credentials(
                RETRIEVE_GENERATION_STATUSES
            ),
        }
        selected_names = {
            ReadinessScope.ALL: tuple(checks),
            ReadinessScope.CORE: ("postgres", "qdrant"),
            ReadinessScope.INGEST: (
                "postgres",
                "qdrant",
                "redis",
                "minio",
                "ingest_credentials",
            ),
            ReadinessScope.RETRIEVE: ("postgres", "qdrant", "retrieve_credentials"),
        }[scope]
        measured = await asyncio.gather(*(self._measure(checks[name]) for name in selected_names))
        components = {
            name: ComponentStatus(ok=False, latency_ms=0.0, error=NOT_CHECKED_ERROR)
            for name in checks
        }
        components.update(dict(zip(selected_names, measured, strict=True)))
        return ReadinessSnapshot(
            components=components,
            answer_configured=False,
        )

    async def close(self) -> None:
        async with self._close_lock:
            if self._close_task is None:
                self._close_task = asyncio.create_task(self._close_owned_resources())
                self._close_task.add_done_callback(_consume_task_exception)
            close_task = self._close_task
        await asyncio.shield(close_task)

    async def _close_owned_resources(self) -> None:
        closers: list[tuple[str, Awaitable[object]]] = []
        if self._owns_redis:
            closers.append(("redis", self._redis.aclose()))
        if self._owns_http:
            closers.append(("http", self._http.aclose()))
        if self._owns_minio and self._minio_http is not None:
            closers.append(("minio", asyncio.to_thread(self._minio_http.clear)))

        failures = await _collect_cleanup_failures(closers)
        if self._owns_database:
            failures.extend(await _collect_cleanup_failures([("postgres", self._database.close())]))

        if failures:
            raise ExceptionGroup("Failed to close readiness resources", failures) from None

    async def _check_qdrant(self) -> None:
        response = await self._http.get(f"{self._settings.qdrant_url.rstrip('/')}/healthz")
        response.raise_for_status()

    async def _check_redis(self) -> None:
        if not await self._redis.ping():
            raise RuntimeError("redis ping returned false")

    async def _check_minio(self) -> None:
        # Async cancellation cannot stop a running worker thread; transport timeouts still bound it.
        exists = await asyncio.to_thread(
            self._minio.bucket_exists,
            self._settings.minio_bucket,
        )
        if not exists:
            raise RuntimeError("configured MinIO bucket is unavailable")

    async def _check_referenced_credentials(self, statuses: Collection[str]) -> None:
        keyring = provider_credential_keyring_from_settings(self._settings)
        keyring_fingerprint = provider_credential_keyring_fingerprint(
            self._settings.provider_credential_keyring.get_secret_value(),
            self._settings.provider_credential_active_key_version,
        )
        await self._credential_validator.validate(
            statuses=statuses,
            keyring=keyring,
            keyring_fingerprint=keyring_fingerprint,
        )

    async def _measure(self, call: Callable[[], Awaitable[object]]) -> ComponentStatus:
        started = perf_counter()
        try:
            async with asyncio.timeout(self._settings.readiness_timeout_seconds):
                await call()
        except Exception as error:
            return ComponentStatus(
                ok=False,
                latency_ms=round((perf_counter() - started) * 1000, 3),
                error=type(error).__name__,
            )
        return ComponentStatus(
            ok=True,
            latency_ms=round((perf_counter() - started) * 1000, 3),
        )
