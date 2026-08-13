"""Signal-aware Worker process and local healthcheck entry points."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
import tempfile
from collections.abc import AsyncIterator, Awaitable, Callable, Coroutine, Mapping, Sequence
from contextlib import asynccontextmanager, suppress
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol, cast
from uuid import uuid4

from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from rag_service.config import Settings
from rag_service.db.session import Database
from rag_service.infrastructure.probes import DatabaseReferencedCredentialLoader
from rag_service.jobs.notifier import RedisJobSubscriber, RedisSubscriberClient
from rag_service.jobs.repositories import JobRepository, SqlAlchemyJobRepository
from rag_service.jobs.runner import (
    ExponentialBackoff,
    JobExhaustionFinalizer,
    JobHandler,
    JobNotificationSource,
    JobRunner,
    RepositoryContextFactory,
)
from rag_service.metadata.purge import KnowledgeBasePurge
from rag_service.observability.logging import emit_safe_log
from rag_service.providers.credentials import ProviderCredentialKeyring
from rag_service.providers.services import provider_credential_keyring_from_settings
from rag_service.readiness import (
    INGEST_GENERATION_STATUSES,
    ReferencedCredentialValidator,
    provider_credential_keyring_fingerprint,
)

_DEFAULT_HEALTH_PATH = "/tmp/velox-worker-health.json"
_DEFAULT_MAX_CONCURRENCY = 4
_HEALTH_MAX_BYTES = 1024
_PROCESS_CANCELLATION_GRACE_SECONDS = 0.05
_DEFAULT_RECONCILIATION_INTERVAL_SECONDS = 60.0
_RECONCILIATION_PAGE_LIMIT = 100

logger = logging.getLogger(__name__)


def _consume_process_task(task: asyncio.Task[Any]) -> None:
    with suppress(asyncio.CancelledError, Exception):
        task.exception()


class WorkerRedis(Protocol):
    async def aclose(self) -> None: ...


class _AsyncCloseable(Protocol):
    async def aclose(self) -> None: ...


class _Clearable(Protocol):
    def clear(self) -> None: ...


@dataclass(frozen=True, slots=True)
class _DefaultHandlerDependencies:
    minio_client: Callable[..., Any]
    pool_manager: Callable[..., Any]
    timeout: Callable[..., Any]
    minio_object_store: Callable[..., Any]
    validate_minio_url: Callable[[str], tuple[str, bool]]
    qdrant_client_from_url: Callable[..., Any]
    secure_provider_transport: Callable[..., Any]
    provider_endpoint_policy_from_settings: Callable[[Settings], Any]
    credential_reader: Callable[..., Any]
    embedding_gateway: Callable[..., Any]
    provider_usage_sink: Callable[..., Any]
    ingestion_pipeline: Callable[..., Any]
    ingestion_repository: Callable[..., Any]
    repair_pipeline: Callable[..., Any]
    generation_service: Callable[..., Any]


def _load_default_handler_dependencies() -> _DefaultHandlerDependencies:
    """Import the heavy indexing stack only in a real default Worker runtime."""

    from minio import Minio
    from urllib3 import PoolManager, Timeout

    from rag_service.indexing.generation_repositories import SessionProviderCredentialReader
    from rag_service.indexing.generation_services import GenerationService
    from rag_service.indexing.qdrant import qdrant_client_from_url
    from rag_service.indexing.repair import GenerationRepairPipeline
    from rag_service.infrastructure.minio_store import MinioObjectStore
    from rag_service.infrastructure.probes import _validate_minio_url
    from rag_service.ingestion.pipeline import IngestionPipeline
    from rag_service.ingestion.repositories import SqlAlchemyIngestionPipelineRepository
    from rag_service.observability.repositories import SqlAlchemyProviderUsageSink
    from rag_service.providers.embeddings import EmbeddingGateway
    from rag_service.providers.services import provider_endpoint_policy_from_settings
    from rag_service.providers.transport import SecureProviderTransport

    return _DefaultHandlerDependencies(
        minio_client=Minio,
        pool_manager=PoolManager,
        timeout=Timeout,
        minio_object_store=MinioObjectStore,
        validate_minio_url=_validate_minio_url,
        qdrant_client_from_url=qdrant_client_from_url,
        secure_provider_transport=SecureProviderTransport,
        provider_endpoint_policy_from_settings=provider_endpoint_policy_from_settings,
        credential_reader=SessionProviderCredentialReader,
        embedding_gateway=EmbeddingGateway,
        provider_usage_sink=SqlAlchemyProviderUsageSink,
        ingestion_pipeline=IngestionPipeline,
        ingestion_repository=SqlAlchemyIngestionPipelineRepository,
        repair_pipeline=GenerationRepairPipeline,
        generation_service=GenerationService,
    )


@dataclass(frozen=True, slots=True)
class JobHandlerRegistration:
    handler: JobHandler
    exhaustion_finalizer: JobExhaustionFinalizer | None = None

    def __post_init__(self) -> None:
        if not callable(self.handler) or (
            self.exhaustion_finalizer is not None and not callable(self.exhaustion_finalizer)
        ):
            raise ValueError("job handler registration is invalid")


@dataclass(frozen=True, slots=True)
class _WorkerHandlerRuntime:
    registrations: Mapping[str, JobHandlerRegistration]
    close: Callable[[], Awaitable[None]]
    reconciliation_passes: tuple[_WorkerReconciliationPass, ...] = ()

    async def aclose(self) -> None:
        await self.close()


@dataclass(frozen=True, slots=True)
class _WorkerReconciliationPass:
    phase: str
    run: Callable[[object | None], Awaitable[object | None]]

    def __post_init__(self) -> None:
        if self.phase not in {"minio", "qdrant"} or not callable(self.run):
            raise ValueError("worker reconciliation pass is invalid")


@dataclass(frozen=True, slots=True)
class WorkerHealthSnapshot:
    pid: int
    process_running: bool
    dependencies_ok: bool
    accepting_jobs: bool
    checked_at: datetime

    @property
    def healthy(self) -> bool:
        return self.process_running and self.dependencies_ok and self.accepting_jobs


def write_worker_health(path: Path, snapshot: WorkerHealthSnapshot) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    document = asdict(snapshot)
    document["checked_at"] = snapshot.checked_at.astimezone(UTC).isoformat()
    payload = json.dumps(document, separators=(",", ":"), sort_keys=True).encode("utf-8")
    if len(payload) > _HEALTH_MAX_BYTES:
        raise ValueError("worker health payload is too large")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            descriptor = -1
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        with suppress(FileNotFoundError):
            os.unlink(temporary_name)


def load_worker_health(
    path: Path,
    *,
    now: datetime,
    max_age: timedelta,
) -> WorkerHealthSnapshot | None:
    if max_age <= timedelta(0):
        raise ValueError("worker health max age must be positive")
    try:
        if path.stat().st_size > _HEALTH_MAX_BYTES:
            return None
        raw = json.loads(path.read_text(encoding="utf-8"))
        if type(raw) is not dict or set(raw) != {
            "pid",
            "process_running",
            "dependencies_ok",
            "accepting_jobs",
            "checked_at",
        }:
            return None
        checked_at = datetime.fromisoformat(raw["checked_at"])
        snapshot = WorkerHealthSnapshot(
            pid=raw["pid"],
            process_running=raw["process_running"],
            dependencies_ok=raw["dependencies_ok"],
            accepting_jobs=raw["accepting_jobs"],
            checked_at=checked_at,
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError, KeyError):
        return None
    if (
        type(snapshot.pid) is not int
        or snapshot.pid < 1
        or type(snapshot.process_running) is not bool
        or type(snapshot.dependencies_ok) is not bool
        or type(snapshot.accepting_jobs) is not bool
        or snapshot.checked_at.tzinfo is None
    ):
        return None
    age = now.astimezone(UTC) - snapshot.checked_at.astimezone(UTC)
    if age < timedelta(0) or age > max_age:
        return None
    return snapshot


def _process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (OSError, ValueError):
        return False
    return True


def _install_signal_handlers(
    loop: asyncio.AbstractEventLoop,
    stop_event: asyncio.Event,
) -> tuple[signal.Signals, ...]:
    installed: list[signal.Signals] = []
    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(signum, stop_event.set)
        except (NotImplementedError, RuntimeError):
            continue
        installed.append(signum)
    return tuple(installed)


def _remove_signal_handlers(
    loop: asyncio.AbstractEventLoop,
    installed: tuple[signal.Signals, ...],
) -> None:
    for signum in installed:
        loop.remove_signal_handler(signum)


async def _periodic_health_check(
    stop_event: asyncio.Event,
    *,
    interval_seconds: float,
    check: Callable[[], Awaitable[bool]],
) -> None:
    if interval_seconds <= 0:
        raise ValueError("health check interval must be positive")
    while not stop_event.is_set():
        await check()
        with suppress(TimeoutError):
            await asyncio.wait_for(stop_event.wait(), timeout=interval_seconds)


async def _periodic_reconciliation(
    stop_event: asyncio.Event,
    *,
    interval_seconds: float,
    passes: Sequence[_WorkerReconciliationPass],
) -> None:
    if (
        interval_seconds <= 0
        or not passes
        or any(type(item) is not _WorkerReconciliationPass for item in passes)
    ):
        raise ValueError("worker reconciliation configuration is invalid")

    async def run_independently(reconciliation_pass: _WorkerReconciliationPass) -> None:
        cursor: object | None = None
        while not stop_event.is_set():
            try:
                cursor = await reconciliation_pass.run(cursor)
            except asyncio.CancelledError:
                raise
            except BaseException:
                with suppress(BaseException):
                    emit_safe_log(
                        logger,
                        logging.WARNING,
                        "cleanup.action.completed",
                        operation="orphan_cleanup",
                        phase=reconciliation_pass.phase,
                        outcome="failed",
                        count=1,
                    )
            if stop_event.is_set():
                return
            with suppress(TimeoutError):
                await asyncio.wait_for(stop_event.wait(), timeout=interval_seconds)

    tasks = tuple(
        asyncio.create_task(
            run_independently(reconciliation_pass),
            name=f"worker-reconciliation-{reconciliation_pass.phase}",
        )
        for reconciliation_pass in passes
    )
    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


@dataclass(slots=True)
class _SharedPreflightCheck:
    task: asyncio.Task[bool]
    deadline: float
    invalidated_event: asyncio.Event
    completed_at: float | None = None
    invalidated: bool = False
    deadline_handle: asyncio.TimerHandle | None = None


class _WorkerPreflightGate:
    """Keep at most one authoritative dependency check in flight."""

    def __init__(
        self,
        check: Callable[[], Awaitable[bool]],
        *,
        timeout_seconds: float,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("preflight timeout must be positive")
        self._check = check
        self._timeout_seconds = timeout_seconds
        self._in_flight: _SharedPreflightCheck | None = None

    @staticmethod
    def _invalidate(shared_check: _SharedPreflightCheck) -> None:
        if shared_check.invalidated:
            return
        shared_check.invalidated = True
        shared_check.invalidated_event.set()
        if shared_check.deadline_handle is not None:
            shared_check.deadline_handle.cancel()
            shared_check.deadline_handle = None
        if not shared_check.task.done():
            shared_check.task.cancel()

    @classmethod
    def _expire(cls, shared_check: _SharedPreflightCheck) -> None:
        shared_check.deadline_handle = None
        if not shared_check.task.done():
            cls._invalidate(shared_check)

    @classmethod
    def _check_finished(
        cls,
        shared_check: _SharedPreflightCheck,
        task: asyncio.Task[bool],
    ) -> None:
        if shared_check.completed_at is None or shared_check.completed_at >= shared_check.deadline:
            cls._invalidate(shared_check)
        elif shared_check.deadline_handle is not None:
            shared_check.deadline_handle.cancel()
            shared_check.deadline_handle = None
        _consume_process_task(task)

    async def run(self, stop_event: asyncio.Event) -> bool:
        if stop_event.is_set():
            shared_check = self._in_flight
            if shared_check is not None:
                self._invalidate(shared_check)
            return False
        shared_check = self._in_flight
        if shared_check is not None and shared_check.task.done():
            _consume_process_task(shared_check.task)
            self._in_flight = None
            shared_check = None

        if shared_check is None:
            loop = asyncio.get_running_loop()
            created_check: _SharedPreflightCheck | None = None

            async def execute_check() -> bool:
                try:
                    return await self._check()
                finally:
                    if created_check is not None:
                        created_check.completed_at = loop.time()

            check_task = asyncio.create_task(execute_check())
            shared_check = _SharedPreflightCheck(
                task=check_task,
                deadline=loop.time() + self._timeout_seconds,
                invalidated_event=asyncio.Event(),
            )
            created_check = shared_check
            shared_check.deadline_handle = loop.call_at(
                shared_check.deadline,
                self._expire,
                shared_check,
            )
            check_task.add_done_callback(lambda task: self._check_finished(shared_check, task))
            self._in_flight = shared_check
        elif asyncio.get_running_loop().time() >= shared_check.deadline:
            self._invalidate(shared_check)
        if shared_check.invalidated:
            return False

        check_task = shared_check.task
        stop_task = asyncio.create_task(stop_event.wait())
        invalidated_task = asyncio.create_task(shared_check.invalidated_event.wait())
        try:
            completed, _pending = await asyncio.wait(
                (check_task, stop_task, invalidated_task),
                return_when=asyncio.FIRST_COMPLETED,
            )
            if stop_task in completed:
                self._invalidate(shared_check)
                return False
            if invalidated_task in completed or shared_check.invalidated:
                return False
            if check_task not in completed or check_task.cancelled():
                return False
            if (
                shared_check.completed_at is None
                or shared_check.completed_at >= shared_check.deadline
            ):
                self._invalidate(shared_check)
                return False
            try:
                return bool(check_task.result())
            except Exception:
                return False
        finally:
            wait_tasks = (stop_task, invalidated_task)
            for wait_task in wait_tasks:
                if not wait_task.done():
                    wait_task.cancel()
            await asyncio.sleep(0)
            for wait_task in wait_tasks:
                if wait_task.done():
                    _consume_process_task(wait_task)
                else:
                    wait_task.add_done_callback(_consume_process_task)

    def cancel(self) -> asyncio.Task[bool] | None:
        shared_check = self._in_flight
        if shared_check is None:
            return None
        self._invalidate(shared_check)
        return shared_check.task


async def _bounded_worker_cleanup(
    *,
    health_task: asyncio.Task[None] | None,
    background_tasks: Sequence[asyncio.Task[Any]] = (),
    quiescence_tasks: Sequence[asyncio.Task[Any]] = (),
    independent_close_operations: Sequence[Callable[[], Awaitable[None]]] = (),
    close_operations: Sequence[Callable[[], Awaitable[None]]],
    timeout_seconds: float,
) -> None:
    """Attempt every cleanup under one absolute shutdown deadline."""

    if timeout_seconds <= 0:
        raise ValueError("worker cleanup timeout must be positive")
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_seconds
    resource_barriers: set[asyncio.Task[Any]] = {*background_tasks, *quiescence_tasks}
    auxiliary_tasks: set[asyncio.Task[Any]] = set()
    if health_task is not None:
        auxiliary_tasks.add(health_task)
    managed_tasks = {*resource_barriers, *auxiliary_tasks}

    async def close_safely(operation: Callable[[], Awaitable[None]]) -> None:
        try:
            await operation()
        except asyncio.CancelledError:
            raise
        except BaseException:
            return

    try:
        for task in managed_tasks:
            task.cancel()

        independent_tasks = {
            asyncio.create_task(close_safely(operation))
            for operation in independent_close_operations
        }
        managed_tasks.update(independent_tasks)
        resource_quiesced = True
        if resource_barriers:
            completed_barriers, pending_barriers = await asyncio.wait(
                resource_barriers,
                timeout=max(0.0, deadline - loop.time()),
            )
            for task in completed_barriers:
                _consume_process_task(task)
            resource_quiesced = not pending_barriers
        dependent_tasks = (
            {asyncio.create_task(close_safely(operation)) for operation in close_operations}
            if resource_quiesced
            else set()
        )
        managed_tasks.update(dependent_tasks)
        remaining = {task for task in managed_tasks if not task.done()}
        if remaining:
            completed, _pending = await asyncio.wait(
                remaining,
                timeout=max(0.0, deadline - loop.time()),
            )
            for task in completed:
                _consume_process_task(task)
    finally:
        for task in managed_tasks:
            if not task.done():
                task.cancel()
        for task in managed_tasks:
            if task.done():
                _consume_process_task(task)
            else:
                task.add_done_callback(_consume_process_task)
        await asyncio.sleep(0)


class WorkerDependencyPreflight:
    """Validate authoritative database and referenced Credential authenticity."""

    def __init__(
        self,
        *,
        database: Database,
        keyring: ProviderCredentialKeyring,
        credential_validator: ReferencedCredentialValidator | None = None,
        keyring_fingerprint: str,
    ) -> None:
        if type(keyring_fingerprint) is not str or not keyring_fingerprint:
            raise ValueError("keyring fingerprint is invalid")
        self._database = database
        self._keyring = keyring
        self._credential_validator = credential_validator or ReferencedCredentialValidator(
            load_referenced_credentials=DatabaseReferencedCredentialLoader(database)
        )
        self._keyring_fingerprint = keyring_fingerprint

    async def __call__(self) -> bool:
        try:
            await self._database.ping()
            await self._credential_validator.validate(
                statuses=INGEST_GENERATION_STATUSES,
                keyring=self._keyring,
                keyring_fingerprint=self._keyring_fingerprint,
            )
        except Exception:
            return False
        return True


async def _start_notification_source(
    subscriber: RedisJobSubscriber,
) -> JobNotificationSource | None:
    try:
        await subscriber.start()
    except Exception:
        return None
    return subscriber


def _repository_context(database: Database) -> RepositoryContextFactory:
    @asynccontextmanager
    async def context() -> AsyncIterator[JobRepository]:
        async with database.sessions() as session, session.begin():
            yield SqlAlchemyJobRepository(session)

    return cast(RepositoryContextFactory, context)


def _ingestion_repository_context(
    database: Database,
    repository_factory: Callable[[AsyncSession], Any],
) -> Callable[[], Any]:
    @asynccontextmanager
    async def context() -> AsyncIterator[Any]:
        async with database.sessions() as session, session.begin():
            yield repository_factory(session)

    return context


async def _close_default_handler_resources(
    *,
    ingestion_pipeline: _AsyncCloseable | None,
    embedding_gateway: _AsyncCloseable | None,
    provider_transport: _AsyncCloseable | None,
    qdrant: _AsyncCloseable | None,
    object_store: _AsyncCloseable | None,
    minio_pool: _Clearable | None,
) -> None:
    async def attempt(
        name: str,
        operation: Callable[[], Awaitable[None]],
    ) -> Exception | None:
        try:
            await operation()
        except asyncio.CancelledError:
            raise
        except BaseException as error:
            return RuntimeError(f"{name} cleanup failed with {type(error).__name__}")
        return None

    operations: list[tuple[str, Callable[[], Awaitable[None]]]] = []
    if ingestion_pipeline is not None:
        operations.append(("ingestion pipeline", ingestion_pipeline.aclose))
    if embedding_gateway is not None:
        operations.append(("embedding gateway", embedding_gateway.aclose))
    if provider_transport is not None:
        operations.append(("provider transport", provider_transport.aclose))
    if qdrant is not None:
        operations.append(("qdrant", qdrant.aclose))
    if object_store is not None or minio_pool is not None:

        async def close_object_store_then_minio_pool() -> None:
            store_failure: BaseException | None = None
            pool_failure: BaseException | None = None
            try:
                if object_store is not None:
                    await object_store.aclose()
            except BaseException as error:
                store_failure = error
            finally:
                try:
                    if minio_pool is not None:
                        await asyncio.to_thread(minio_pool.clear)
                except BaseException as error:
                    pool_failure = error

            if isinstance(store_failure, asyncio.CancelledError) or isinstance(
                pool_failure,
                asyncio.CancelledError,
            ):
                raise asyncio.CancelledError from None
            if store_failure is not None:
                raise store_failure
            if pool_failure is not None:
                raise pool_failure

        operations.append(("object store and minio pool", close_object_store_then_minio_pool))

    close_tasks = tuple(
        asyncio.create_task(attempt(name, operation)) for name, operation in operations
    )
    close_group = asyncio.gather(*close_tasks, return_exceptions=True)
    try:
        results = await asyncio.shield(close_group)
    except asyncio.CancelledError:
        for task in close_tasks:
            if not task.done():
                task.cancel()
        for task in close_tasks:
            if task.done():
                _consume_process_task(task)
            else:
                task.add_done_callback(_consume_process_task)
        await asyncio.sleep(0)
        raise

    if any(isinstance(result, asyncio.CancelledError) for result in results):
        raise asyncio.CancelledError from None
    failures = [result for result in results if isinstance(result, Exception)]
    if failures:
        raise ExceptionGroup("Worker handler resource cleanup failed", failures)


async def _build_default_handler_runtime(
    *,
    settings: Settings,
    database: Database,
    keyring: ProviderCredentialKeyring,
    max_concurrency: int,
) -> _WorkerHandlerRuntime:
    if type(max_concurrency) is not int or max_concurrency < 1:
        raise ValueError("worker handler concurrency is invalid")

    dependencies = _load_default_handler_dependencies()
    minio_pool: Any = None
    object_store: Any = None
    qdrant: Any = None
    provider_transport: Any = None
    embedding_gateway: Any = None
    ingestion_pipeline: Any = None
    try:
        endpoint, secure = dependencies.validate_minio_url(settings.minio_url)
        minio_pool = dependencies.pool_manager(
            timeout=dependencies.timeout(
                connect=settings.minio_operation_timeout_seconds,
                read=settings.minio_operation_timeout_seconds,
            ),
            retries=False,
        )
        object_store = dependencies.minio_object_store(
            client=cast(
                Any,
                dependencies.minio_client(
                    endpoint,
                    access_key=settings.minio_access_key,
                    secret_key=settings.minio_secret_key.get_secret_value(),
                    secure=secure,
                    http_client=minio_pool,
                ),
            ),
            bucket=settings.minio_bucket,
            buffer_bytes=settings.upload_buffer_bytes,
            part_size_bytes=settings.minio_multipart_part_size_bytes,
            operation_timeout_seconds=settings.minio_operation_timeout_seconds,
        )
        qdrant = dependencies.qdrant_client_from_url(
            settings.qdrant_url,
            timeout_seconds=settings.qdrant_request_timeout_seconds,
        )
        provider_transport = dependencies.secure_provider_transport(
            policy=dependencies.provider_endpoint_policy_from_settings(settings),
            ca_bundle=settings.provider_ca_bundle,
        )
        embedding_gateway = dependencies.embedding_gateway(
            keyring=keyring,
            credential_reader=dependencies.credential_reader(database.sessions),
            transport=provider_transport,
        )
        usage_sink = dependencies.provider_usage_sink(database.sessions)
        ingestion_pipeline = dependencies.ingestion_pipeline(
            repository_context=_ingestion_repository_context(
                database,
                dependencies.ingestion_repository,
            ),
            object_store=object_store,
            max_document_bytes=settings.max_upload_bytes,
            cpu_concurrency=max_concurrency,
            chunk_max_codepoints=settings.chunk_max_codepoints,
            chunk_overlap_codepoints=settings.chunk_overlap_codepoints,
            embedding_gateway=embedding_gateway,
            qdrant=qdrant,
            provider_usage_sink=usage_sink,
        )
        knowledge_base_purge = KnowledgeBasePurge(
            session_factory=database.sessions,
            object_store=object_store,
            search_index=qdrant,
        )
        repair_pipeline = dependencies.repair_pipeline(
            session_factory=database.sessions,
            object_store=object_store,
            embedding_gateway=embedding_gateway,
            qdrant=qdrant,
            provider_usage_sink=usage_sink,
        )
        generation_service = dependencies.generation_service(
            session_factory=database.sessions,
            qdrant=qdrant,
            embedding_gateway=embedding_gateway,
        )

        async def reconcile_objects(cursor: object | None) -> object | None:
            result = await ingestion_pipeline.reconcile_orphan_objects(
                grace_period=timedelta(seconds=settings.orphan_object_grace_seconds),
                limit=_RECONCILIATION_PAGE_LIMIT,
                cursor=cursor,
            )
            return cast(object | None, result.next_cursor)

        async def reconcile_collections(cursor: object | None) -> object | None:
            result = await generation_service.reconcile_orphan_collections(
                grace_period=timedelta(seconds=settings.orphan_object_grace_seconds),
                limit=_RECONCILIATION_PAGE_LIMIT,
                cursor=cursor,
            )
            return cast(object | None, result.next_cursor)

        async def close() -> None:
            await _close_default_handler_resources(
                ingestion_pipeline=ingestion_pipeline,
                embedding_gateway=embedding_gateway,
                provider_transport=provider_transport,
                qdrant=qdrant,
                object_store=object_store,
                minio_pool=minio_pool,
            )

        return _WorkerHandlerRuntime(
            registrations={
                "ingest_document": JobHandlerRegistration(
                    ingestion_pipeline.handle,
                    ingestion_pipeline.finalize_exhausted,
                ),
                "rebuild_generation": JobHandlerRegistration(repair_pipeline.handle),
                "purge_knowledge_base": JobHandlerRegistration(knowledge_base_purge.handle),
            },
            close=close,
            reconciliation_passes=(
                _WorkerReconciliationPass("minio", reconcile_objects),
                _WorkerReconciliationPass("qdrant", reconcile_collections),
            ),
        )
    except BaseException:
        with suppress(BaseException):
            await _close_default_handler_resources(
                ingestion_pipeline=ingestion_pipeline,
                embedding_gateway=embedding_gateway,
                provider_transport=provider_transport,
                qdrant=qdrant,
                object_store=object_store,
                minio_pool=minio_pool,
            )
        raise


def _register_handlers(
    runner: JobRunner,
    registrations: Mapping[str, JobHandlerRegistration],
) -> None:
    for operation, registration in registrations.items():
        runner.register(
            operation,
            registration.handler,
            exhaustion_finalizer=registration.exhaustion_finalizer,
        )


def _worker_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="velox-worker")
    parser.add_argument(
        "--health-file",
        default=os.environ.get("RAG_WORKER_HEALTH_FILE", _DEFAULT_HEALTH_PATH),
    )
    parser.add_argument("--max-concurrency", type=int, default=_DEFAULT_MAX_CONCURRENCY)
    return parser


def _health_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="velox-worker-health")
    parser.add_argument(
        "--health-file",
        default=os.environ.get("RAG_WORKER_HEALTH_FILE", _DEFAULT_HEALTH_PATH),
    )
    parser.add_argument("--max-age-seconds", type=float, default=5.0)
    return parser


async def _run_worker(
    health_path: Path,
    max_concurrency: int,
    *,
    handlers: Mapping[str, JobHandlerRegistration] | None = None,
) -> int:
    write_worker_health(
        health_path,
        WorkerHealthSnapshot(
            pid=os.getpid(),
            process_running=True,
            dependencies_ok=False,
            accepting_jobs=False,
            checked_at=datetime.now(UTC),
        ),
    )
    if max_concurrency < 1:
        return 2
    settings = Settings()
    database = Database.from_settings(settings)
    redis_client = Redis.from_url(
        settings.redis_url.get_secret_value(),
        socket_connect_timeout=settings.readiness_timeout_seconds,
        socket_timeout=settings.readiness_timeout_seconds,
    )
    subscriber = RedisJobSubscriber(cast(RedisSubscriberClient, redis_client))
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    installed_signals = _install_signal_handlers(loop, stop_event)

    accepting_jobs = False
    dependencies_ok = False
    handlers_ready = False
    health_task: asyncio.Task[None] | None = None
    reconciliation_task: asyncio.Task[None] | None = None
    preflight_gate: _WorkerPreflightGate | None = None
    handler_runtime: _WorkerHandlerRuntime | None = None

    def report() -> None:
        write_worker_health(
            health_path,
            WorkerHealthSnapshot(
                pid=os.getpid(),
                process_running=True,
                dependencies_ok=dependencies_ok,
                accepting_jobs=accepting_jobs,
                checked_at=datetime.now(UTC),
            ),
        )

    try:
        keyring = provider_credential_keyring_from_settings(settings)
        preflight = WorkerDependencyPreflight(
            database=database,
            keyring=keyring,
            keyring_fingerprint=provider_credential_keyring_fingerprint(
                settings.provider_credential_keyring.get_secret_value(),
                settings.provider_credential_active_key_version,
            ),
        )
        preflight_gate = _WorkerPreflightGate(
            preflight,
            timeout_seconds=settings.readiness_timeout_seconds,
        )

        async def checked_preflight() -> bool:
            nonlocal accepting_jobs, dependencies_ok
            dependencies_ok = await preflight_gate.run(stop_event)
            accepting_jobs = dependencies_ok and handlers_ready and not stop_event.is_set()
            report()
            return accepting_jobs

        runner = JobRunner(
            repository_context=_repository_context(database),
            lease_owner=f"{os.getpid()}:{uuid4()}",
            lease_seconds=settings.worker_lease_seconds,
            heartbeat_seconds=settings.worker_heartbeat_seconds,
            poll_interval_seconds=settings.worker_poll_interval_seconds,
            max_concurrency=max_concurrency,
            backoff=ExponentialBackoff(
                initial_seconds=settings.worker_retry_initial_seconds,
                maximum_seconds=settings.worker_retry_max_seconds,
            ),
            preflight=checked_preflight,
            shutdown_seconds=settings.shutdown_timeout_seconds,
        )
        selected_handlers = handlers
        if selected_handlers is None:
            handler_runtime = await _build_default_handler_runtime(
                settings=settings,
                database=database,
                keyring=keyring,
                max_concurrency=max_concurrency,
            )
            selected_handlers = handler_runtime.registrations
        _register_handlers(runner, selected_handlers)
        handlers_ready = runner.has_handlers
        if not await checked_preflight():
            return 0 if stop_event.is_set() else 1
        health_task = asyncio.create_task(
            _periodic_health_check(
                stop_event,
                interval_seconds=min(settings.worker_poll_interval_seconds, 1.0),
                check=checked_preflight,
            )
        )
        reconciliation_passes = (
            () if handler_runtime is None else getattr(handler_runtime, "reconciliation_passes", ())
        )
        if reconciliation_passes:
            reconciliation_task = asyncio.create_task(
                _periodic_reconciliation(
                    stop_event,
                    interval_seconds=_DEFAULT_RECONCILIATION_INTERVAL_SECONDS,
                    passes=reconciliation_passes,
                )
            )
        notifications = await _start_notification_source(subscriber)
        await runner.run(stop_event, notifications=notifications)
        return 0
    finally:
        primary_error_active = sys.exc_info()[0] is not None
        accepting_jobs = False
        report_error: Exception | None = None
        try:
            report()
        except Exception as error:
            report_error = error
        with suppress(Exception):
            _remove_signal_handlers(loop, installed_signals)
        preflight_task = preflight_gate.cancel() if preflight_gate is not None else None

        async def close_handlers_then_database() -> None:
            try:
                if handler_runtime is not None:
                    await handler_runtime.aclose()
            finally:
                await database.close()

        background_tasks = tuple(task for task in (preflight_task,) if task is not None)
        quiescence_tasks = tuple(task for task in (reconciliation_task,) if task is not None)
        await _bounded_worker_cleanup(
            health_task=health_task,
            background_tasks=background_tasks,
            quiescence_tasks=quiescence_tasks,
            independent_close_operations=(
                subscriber.close,
                redis_client.aclose,
            ),
            close_operations=(close_handlers_then_database,),
            timeout_seconds=settings.shutdown_timeout_seconds,
        )
        if report_error is not None and not primary_error_active:
            raise report_error


def _run_worker_event_loop(worker: Coroutine[Any, Any, int]) -> int:
    """Run the Worker without asyncio.run's unbounded pending-task drain."""

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(worker)
    finally:
        pending = {task for task in asyncio.all_tasks(loop) if not task.done()}
        for task in pending:
            task.cancel()
        if pending:

            async def drain_once() -> tuple[set[asyncio.Task[Any]], set[asyncio.Task[Any]]]:
                return await asyncio.wait(
                    pending,
                    timeout=_PROCESS_CANCELLATION_GRACE_SECONDS,
                )

            completed, abandoned = loop.run_until_complete(drain_once())
            for task in completed:
                _consume_process_task(task)
            for task in abandoned:
                task.add_done_callback(_consume_process_task)
        loop.close()
        asyncio.set_event_loop(None)


def main(arguments: Sequence[str] | None = None) -> int:
    parsed = _worker_parser().parse_args(arguments)
    try:
        return _run_worker_event_loop(_run_worker(Path(parsed.health_file), parsed.max_concurrency))
    except KeyboardInterrupt:
        return 130
    except Exception:
        return 1


def health_main(arguments: Sequence[str] | None = None) -> int:
    parsed = _health_parser().parse_args(arguments)
    if parsed.max_age_seconds <= 0:
        return 2
    snapshot = load_worker_health(
        Path(parsed.health_file),
        now=datetime.now(UTC),
        max_age=timedelta(seconds=parsed.max_age_seconds),
    )
    if snapshot is None or not snapshot.healthy or not _process_exists(snapshot.pid):
        return 1
    return 0


__all__ = [
    "WorkerDependencyPreflight",
    "WorkerHealthSnapshot",
    "health_main",
    "load_worker_health",
    "main",
    "write_worker_health",
]
