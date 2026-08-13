import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any, cast

from fastapi import Depends, FastAPI, Request
from fastapi.dependencies.models import Dependant
from fastapi.dependencies.utils import get_dependant, solve_dependencies
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from minio import Minio
from redis.asyncio import Redis
from starlette.types import Scope
from urllib3 import PoolManager, Timeout

from rag_service import __version__
from rag_service.admin.routes import router as admin_router
from rag_service.api.constants import MAX_HEADER_VALUE_LENGTH
from rag_service.api.errors import (
    BusinessError,
    business_error_response,
    validation_error_response,
)
from rag_service.api.health import router as health_router
from rag_service.api.middleware import (
    _REQUEST_SETTINGS_STATE_KEY,
    RequestErrorMiddleware,
    get_request_id,
)
from rag_service.config import Settings, get_settings
from rag_service.db.session import Database
from rag_service.indexing.generation_routes import (
    router as generation_router,
)
from rag_service.infrastructure.minio_store import MinioObjectStore
from rag_service.infrastructure.probes import LiveReadinessProvider, _validate_minio_url
from rag_service.ingestion.routes import router as ingestion_router
from rag_service.ingestion.routes import upload_limit_description
from rag_service.jobs.notifier import RedisJobNotifier
from rag_service.jobs.routes import router as job_router
from rag_service.metadata.document_routes import router as document_router
from rag_service.metadata.knowledge_base_routes import router as knowledge_base_router
from rag_service.providers.embeddings import EmbeddingGateway
from rag_service.providers.gateway_provider import (
    EmbeddingGatewayProvider,
    RerankGatewayProvider,
)
from rag_service.providers.routes import router as provider_credential_router
from rag_service.readiness import ReadinessProvider
from rag_service.retrieval.routes import router as retrieval_router

_SETTINGS_DEPENDENCY = get_settings
_SETTINGS_VALUE_PARAMETER = "settings_value"
_MISSING_SCOPE_VALUE = object()


class _SettingsDependencyOverridesProvider:
    def __init__(
        self,
        app: FastAPI,
        default_settings_provider: Callable[[], Settings],
        request_settings_dependency: Callable[..., Awaitable[Settings]],
    ) -> None:
        self._app = app
        self._default_settings_provider = default_settings_provider
        self._request_settings_dependency = request_settings_dependency

    @property
    def dependency_overrides(self) -> "_SettingsDependencyOverridesProvider":
        return self

    def __bool__(self) -> bool:
        return bool(self._app.dependency_overrides)

    def get(self, key: Callable[..., Any], default: Any = None) -> Any:
        if key is _SETTINGS_DEPENDENCY and key in self._app.dependency_overrides:
            return self._request_settings_dependency
        return self._app.dependency_overrides.get(key, default)

    def settings_provider(self) -> Callable[..., object]:
        provider = self._app.dependency_overrides.get(
            _SETTINGS_DEPENDENCY,
            self._default_settings_provider,
        )
        if not callable(provider):
            raise TypeError("Settings dependency override must be callable")
        return cast(Callable[..., object], provider)


def _settings_dependency_dependant(path: str, provider: Callable[..., object]) -> Dependant:
    def dependency_target(settings_value: object = Depends(provider)) -> None:
        del settings_value

    return get_dependant(path=path, call=dependency_target)


def _effective_dependency(
    dependant: Dependant,
    dependency_overrides_provider: _SettingsDependencyOverridesProvider,
) -> Dependant:
    call = dependant.call
    if call is None or not dependency_overrides_provider:
        return dependant
    override = dependency_overrides_provider.dependency_overrides.get(call, call)
    if override is call:
        return dependant
    path = dependant.path or ""
    return get_dependant(
        path=path,
        call=override,
        name=dependant.name,
        parent_oauth_scopes=dependant.oauth_scopes,
        scope=dependant.scope,
    )


def _dependency_unsupported_feature(
    dependant: Dependant,
    dependency_overrides_provider: _SettingsDependencyOverridesProvider,
) -> str | None:
    pending = list(dependant.dependencies)
    while pending:
        current = _effective_dependency(pending.pop(), dependency_overrides_provider)
        if current.body_params:
            return "body parameters"
        if current.response_param_name or current.background_tasks_param_name:
            return "Response or BackgroundTasks parameters"
        pending.extend(current.dependencies)
    return None


def _restore_scope_value(scope: Scope, key: str, previous_value: object) -> None:
    if previous_value is _MISSING_SCOPE_VALUE:
        scope.pop(key, None)
    else:
        scope[key] = previous_value


async def _solve_settings_dependency(
    *,
    scope: Scope,
    provider: Callable[..., object],
    dependency_overrides_provider: _SettingsDependencyOverridesProvider,
    request_settings_stack: AsyncExitStack,
    function_settings_stack: AsyncExitStack,
) -> Settings:
    path_value = scope.get("path")
    path = path_value if isinstance(path_value, str) else ""
    dependant = _settings_dependency_dependant(path, provider)
    unsupported_feature = _dependency_unsupported_feature(
        dependant,
        dependency_overrides_provider,
    )
    if unsupported_feature is not None:
        raise TypeError(f"Settings dependency override cannot declare {unsupported_feature}")

    previous_inner_stack = scope.get("fastapi_inner_astack", _MISSING_SCOPE_VALUE)
    previous_function_stack = scope.get("fastapi_function_astack", _MISSING_SCOPE_VALUE)
    request = Request(scope)
    try:
        scope["fastapi_inner_astack"] = request_settings_stack
        scope["fastapi_function_astack"] = function_settings_stack
        # This middleware-level solve has its own FastAPI dependency cache. A child
        # also declared independently by the route can therefore run again there;
        # Settings providers and their non-lifecycle dependencies must be idempotent.
        solved = await solve_dependencies(
            request=request,
            dependant=dependant,
            dependency_overrides_provider=dependency_overrides_provider,
            async_exit_stack=request_settings_stack,
            embed_body_fields=False,
        )
    finally:
        _restore_scope_value(scope, "fastapi_inner_astack", previous_inner_stack)
        _restore_scope_value(scope, "fastapi_function_astack", previous_function_stack)

    if solved.errors:
        raise RequestValidationError(solved.errors)
    provided_settings = solved.values.get(_SETTINGS_VALUE_PARAMETER)
    if not isinstance(provided_settings, Settings):
        raise TypeError("Settings dependency override must return Settings")
    return provided_settings


class ApplicationCleanupError(RuntimeError):
    """Sanitized failure raised while closing an application-owned resource."""


async def _business_error_response(request: Request, error: Exception) -> JSONResponse:
    assert isinstance(error, BusinessError)
    return business_error_response(error, await get_request_id(request))


async def _request_validation_error_response(
    request: Request,
    error: Exception,
) -> JSONResponse:
    assert isinstance(error, RequestValidationError)
    return validation_error_response(error, await get_request_id(request))


def _sanitize_cleanup_failure(resource: str, error: BaseException) -> ApplicationCleanupError:
    return ApplicationCleanupError(f"{resource} cleanup failed with {type(error).__name__}")


def _consume_task_exception(task: asyncio.Task[object]) -> None:
    if not task.cancelled():
        task.exception()


def _task_cleanup_failure(
    resource: str,
    task: asyncio.Task[None],
) -> ApplicationCleanupError | None:
    try:
        task.result()
    except BaseException as error:
        return _sanitize_cleanup_failure(resource, error)
    return None


async def _invoke_closer(close: Callable[[], Awaitable[None]]) -> None:
    await close()


async def _attempt_close(
    resource: str,
    close: Callable[[], Awaitable[None]],
    timeout_seconds: float,
) -> ApplicationCleanupError | None:
    close_task: asyncio.Task[None] = asyncio.create_task(_invoke_closer(close))
    close_task.add_done_callback(_consume_task_exception)
    try:
        done, _pending = await asyncio.wait(
            {close_task},
            timeout=timeout_seconds,
        )
    except BaseException:
        close_task.cancel()
        raise
    if close_task not in done:
        close_task.cancel()
        return ApplicationCleanupError(f"{resource} cleanup failed with TimeoutError")
    return _task_cleanup_failure(resource, close_task)


async def _close_owned_resources(
    closers: tuple[tuple[str, Callable[[], Awaitable[None]]], ...],
    timeout_seconds: float,
) -> None:
    failures: list[ApplicationCleanupError] = []
    for resource, close in closers:
        failure = await _attempt_close(resource, close, timeout_seconds)
        if failure is not None:
            failures.append(failure)

    if failures:
        raise ExceptionGroup("Failed to close application resources", failures) from None


async def _drain_owned_cleanup(
    closers: tuple[tuple[str, Callable[[], Awaitable[None]]], ...],
    timeout_seconds: float,
) -> tuple[BaseException | None, asyncio.CancelledError | None]:
    if not closers:
        return None, None

    cleanup_task = asyncio.create_task(_close_owned_resources(closers, timeout_seconds))
    cleanup_task.add_done_callback(_consume_task_exception)
    cancellation: asyncio.CancelledError | None = None
    while not cleanup_task.done():
        try:
            await asyncio.shield(cleanup_task)
        except asyncio.CancelledError as error:
            if cancellation is None:
                cancellation = error
        except BaseException:
            break

    if cleanup_task.cancelled():
        return ApplicationCleanupError(
            "application cleanup failed with CancelledError"
        ), cancellation
    return cleanup_task.exception(), cancellation


def _raise_lifespan_failures(
    primary_error: BaseException | None,
    cleanup_failure: BaseException | None,
    cleanup_cancellation: asyncio.CancelledError | None,
) -> None:
    failures = [
        error
        for error in (primary_error, cleanup_cancellation, cleanup_failure)
        if error is not None
    ]
    if not failures:
        return
    if len(failures) == 1:
        failure = failures[0]
        raise failure.with_traceback(failure.__traceback__) from None
    raise BaseExceptionGroup("Application lifespan failed", failures) from None


def create_app(
    *,
    settings: Settings | None = None,
    readiness_provider: ReadinessProvider | None = None,
    database: Database | None = None,
    generation_embedding_gateway: EmbeddingGateway | None = None,
    upload_object_store: MinioObjectStore | None = None,
    job_notifier: RedisJobNotifier | None = None,
) -> FastAPI:
    resolved_settings = settings
    dependency_overrides_provider: _SettingsDependencyOverridesProvider

    def resolve_settings() -> Settings:
        nonlocal resolved_settings
        if resolved_settings is None:
            resolved_settings = get_settings()
        return resolved_settings

    async def resolve_request_settings(
        scope: Scope,
        request_settings_stack: AsyncExitStack,
        function_settings_stack: AsyncExitStack,
        provider: Callable[..., object] | None = None,
    ) -> Settings:
        state = scope.setdefault("state", {})
        if isinstance(state, dict):
            request_settings = state.get(_REQUEST_SETTINGS_STATE_KEY)
            if isinstance(request_settings, Settings):
                return request_settings

        selected_provider = provider or dependency_overrides_provider.settings_provider()
        provided_settings = await _solve_settings_dependency(
            scope=scope,
            provider=selected_provider,
            dependency_overrides_provider=dependency_overrides_provider,
            request_settings_stack=request_settings_stack,
            function_settings_stack=function_settings_stack,
        )
        if isinstance(state, dict):
            state[_REQUEST_SETTINGS_STATE_KEY] = provided_settings
        return provided_settings

    async def request_settings_dependency(request: Request) -> Settings:
        request_settings_stack = request.scope.get("fastapi_inner_astack")
        function_settings_stack = request.scope.get("fastapi_function_astack")
        if not isinstance(request_settings_stack, AsyncExitStack):
            raise RuntimeError("FastAPI request dependency stack is unavailable")
        if not isinstance(function_settings_stack, AsyncExitStack):
            raise RuntimeError("FastAPI function dependency stack is unavailable")
        return await resolve_request_settings(
            request.scope,
            request_settings_stack,
            function_settings_stack,
        )

    async def request_id_max_length(
        scope: Scope,
        request_settings_stack: AsyncExitStack,
        function_settings_stack: AsyncExitStack,
    ) -> int:
        path = scope.get("path")
        selected_provider = dependency_overrides_provider.settings_provider()
        if (
            resolved_settings is None
            and database is not None
            and readiness_provider is not None
            and selected_provider is resolve_settings
            and isinstance(path, str)
            and not path.startswith("/v1")
        ):
            return MAX_HEADER_VALUE_LENGTH
        return (
            await resolve_request_settings(
                scope,
                request_settings_stack,
                function_settings_stack,
                selected_provider,
            )
        ).max_request_id_length

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        selected_database = database
        selected_readiness = readiness_provider
        selected_object_store = upload_object_store
        selected_job_notifier = job_notifier
        owned_redis: Redis | None = None
        owned_minio_pool: PoolManager | None = None
        owns_object_store = False
        owned_object_store_closed = False
        embedding_gateway_provider = EmbeddingGatewayProvider(generation_embedding_gateway)
        rerank_gateway_provider = RerankGatewayProvider()
        owns_database = False
        owns_readiness = False
        primary_error: BaseException | None = None
        try:
            if selected_database is None:
                selected_database = Database.from_settings(resolve_settings())
                owns_database = True
            app.state.database = selected_database
            app.state.embedding_gateway_provider = embedding_gateway_provider
            app.state.rerank_gateway_provider = rerank_gateway_provider

            configure_owned_upload_resources = (
                resolved_settings is not None
                or database is None
                or readiness_provider is None
                or selected_object_store is not None
                or selected_job_notifier is not None
            )
            if selected_object_store is None and configure_owned_upload_resources:
                settings_value = resolve_settings()
                endpoint, secure = _validate_minio_url(settings_value.minio_url)
                owned_minio_pool = PoolManager(
                    timeout=Timeout(
                        connect=settings_value.minio_operation_timeout_seconds,
                        read=settings_value.minio_operation_timeout_seconds,
                    ),
                    retries=False,
                )
                selected_object_store = MinioObjectStore(
                    client=cast(
                        Any,
                        Minio(
                            endpoint,
                            access_key=settings_value.minio_access_key,
                            secret_key=settings_value.minio_secret_key.get_secret_value(),
                            secure=secure,
                            http_client=owned_minio_pool,
                        ),
                    ),
                    bucket=settings_value.minio_bucket,
                    buffer_bytes=settings_value.upload_buffer_bytes,
                    part_size_bytes=settings_value.minio_multipart_part_size_bytes,
                    operation_timeout_seconds=settings_value.minio_operation_timeout_seconds,
                )
                owns_object_store = True
            if selected_job_notifier is None and configure_owned_upload_resources:
                owned_redis = Redis.from_url(resolve_settings().redis_url.get_secret_value())
                selected_job_notifier = RedisJobNotifier(cast(Any, owned_redis))
            if selected_object_store is not None:
                app.state.upload_object_store = selected_object_store
            if selected_job_notifier is not None:
                app.state.job_notifier = selected_job_notifier

            if selected_readiness is None:
                selected_readiness = await LiveReadinessProvider.create(
                    resolve_settings(),
                    database=selected_database,
                )
                owns_readiness = True
            app.state.readiness_provider = selected_readiness
            yield
        except BaseException as error:
            primary_error = error

        closers: list[tuple[str, Callable[[], Awaitable[None]]]] = []
        closers.append(("embedding gateway", embedding_gateway_provider.aclose))
        closers.append(("rerank gateway", rerank_gateway_provider.aclose))
        if owns_object_store and selected_object_store is not None:

            async def close_upload_object_store() -> None:
                nonlocal owned_object_store_closed
                assert selected_object_store is not None
                await selected_object_store.aclose()
                owned_object_store_closed = True

            closers.append(("upload object store", close_upload_object_store))
        if owned_redis is not None:
            closers.append(("upload redis", owned_redis.aclose))
        if owned_minio_pool is not None:

            async def close_minio_pool() -> None:
                assert owned_minio_pool is not None
                if owns_object_store and not owned_object_store_closed:
                    return
                await asyncio.to_thread(owned_minio_pool.clear)

            closers.append(("upload minio pool", close_minio_pool))
        if owns_readiness:
            live_readiness = cast(LiveReadinessProvider, selected_readiness)
            closers.append(("readiness", live_readiness.close))
        if owns_database:
            assert selected_database is not None
            closers.append(("database", selected_database.close))

        cleanup_timeout = (
            resolved_settings.shutdown_timeout_seconds if resolved_settings is not None else 2.0
        )
        try:
            cleanup_failure, cleanup_cancellation = await _drain_owned_cleanup(
                tuple(closers),
                cleanup_timeout,
            )
        finally:
            if hasattr(app.state, "job_notifier"):
                del app.state.job_notifier
            if hasattr(app.state, "upload_object_store"):
                del app.state.upload_object_store
            if hasattr(app.state, "readiness_provider"):
                del app.state.readiness_provider
            if hasattr(app.state, "embedding_gateway_provider"):
                del app.state.embedding_gateway_provider
            if hasattr(app.state, "rerank_gateway_provider"):
                del app.state.rerank_gateway_provider
            if hasattr(app.state, "database"):
                del app.state.database
        _raise_lifespan_failures(primary_error, cleanup_failure, cleanup_cancellation)

    app = FastAPI(
        title="RAG Service",
        version=__version__,
        lifespan=lifespan,
    )
    dependency_overrides_provider = _SettingsDependencyOverridesProvider(
        app,
        resolve_settings,
        request_settings_dependency,
    )
    app.dependency_overrides[_SETTINGS_DEPENDENCY] = resolve_settings
    app.router.dependency_overrides_provider = dependency_overrides_provider
    app.add_middleware(
        RequestErrorMiddleware,
        max_request_id_length_provider=request_id_max_length,
    )
    app.add_exception_handler(BusinessError, _business_error_response)
    app.add_exception_handler(RequestValidationError, _request_validation_error_response)
    app.include_router(health_router)
    app.include_router(admin_router)
    app.include_router(provider_credential_router)
    app.include_router(generation_router)
    app.include_router(knowledge_base_router)
    app.include_router(document_router)
    app.include_router(ingestion_router)
    app.include_router(job_router)
    app.include_router(retrieval_router)

    generate_openapi = app.openapi

    def generate_configured_openapi() -> dict[str, Any]:
        schema = generate_openapi()
        file_schema = schema["paths"]["/v1/knowledge-bases/{knowledge_base_id}/documents"]["post"][
            "requestBody"
        ]["content"]["multipart/form-data"]["schema"]["properties"]["file"]
        file_schema["description"] = upload_limit_description(resolve_settings().max_upload_bytes)
        return schema

    app.openapi = generate_configured_openapi  # type: ignore[method-assign]
    return app


app = create_app()
