import logging
import re
import secrets
import sys
from collections.abc import Awaitable, Callable
from contextlib import (
    AbstractAsyncContextManager,
    AbstractContextManager,
    AsyncExitStack,
    ExitStack,
)
from inspect import isawaitable
from types import TracebackType
from typing import Any, Never, Self, TypeGuard, cast

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from rag_service.api.constants import (
    GENERATED_REQUEST_ID_LENGTH,
    MAX_HEADER_VALUE_LENGTH,
    REQUEST_ID_PREFIX,
    REQUEST_ID_RANDOM_BYTES,
)
from rag_service.api.errors import (
    BusinessError,
    business_error_response,
    internal_error_response,
    safe_exception_type,
    sanitize_exception_graph,
    validation_error_response,
)

_REQUEST_ID_PATTERN = re.compile(r"[A-Za-z0-9._:-]+")
_REQUEST_ID_HEADER = b"x-request-id"
_REQUEST_SETTINGS_STATE_KEY = "_rag_runtime_settings"
logger = logging.getLogger(__name__)
_SAFE_PROPAGATION_MESSAGE = "Request processing failed"
_SyncExitCallback = Callable[
    [type[BaseException] | None, BaseException | None, TracebackType | None],
    bool | None,
]
_AsyncExitCallback = Callable[
    [type[BaseException] | None, BaseException | None, TracebackType | None],
    Awaitable[bool | None],
]


def is_valid_request_id(value: object, max_length: int) -> TypeGuard[str]:
    """Return whether a request ID satisfies the shared safe-value contract."""
    return (
        isinstance(value, str)
        and 0 < len(value) <= max_length
        and _REQUEST_ID_PATTERN.fullmatch(value) is not None
    )


def request_id_for(value: str | None, max_length: int = MAX_HEADER_VALUE_LENGTH) -> str:
    """Return a safe request ID, replacing invalid client-supplied values."""
    if not GENERATED_REQUEST_ID_LENGTH <= max_length <= MAX_HEADER_VALUE_LENGTH:
        raise ValueError("request ID length is outside supported bounds")
    if is_valid_request_id(value, max_length):
        return value
    return f"{REQUEST_ID_PREFIX}{secrets.token_urlsafe(REQUEST_ID_RANDOM_BYTES)}"


def request_id_from_header(value: str | None, max_length: int = MAX_HEADER_VALUE_LENGTH) -> str:
    """Alias kept for call sites that work directly with the request header."""
    return request_id_for(value, max_length)


def _request_id_from_scope(scope: Scope, max_length: int) -> str:
    values = [
        value for name, value in scope.get("headers", ()) if name.lower() == _REQUEST_ID_HEADER
    ]
    supplied: str | None = None
    if len(values) == 1:
        try:
            supplied = values[0].decode("ascii")
        except (UnicodeDecodeError, ValueError):
            supplied = None
    return request_id_for(supplied, max_length)


async def get_request_id(request: Request) -> str:
    value = getattr(request.state, "request_id", None)
    if is_valid_request_id(value, MAX_HEADER_VALUE_LENGTH):
        return value
    generated = request_id_for(None)
    request.state.request_id = generated
    return generated


async def _raise_safe_wrapper() -> Never:
    raise RuntimeError(_SAFE_PROPAGATION_MESSAGE) from None


async def _raise_sanitized_group(group: BaseExceptionGroup) -> Never:
    raise group from None


class _RequestIdSend:
    def __init__(
        self,
        send: Send,
        request_id: str,
        function_stack: "_FunctionStackController | None",
    ) -> None:
        self._send = send
        self._request_id = request_id
        self._function_stack = function_stack
        self._trailers_expected = False
        self.response_started = False
        self.response_complete = False

    async def __call__(self, message: Message) -> None:
        if self._function_stack is not None:
            self._function_stack.raise_if_poisoned()
        if message["type"] == "http.response.start":
            if self._function_stack is not None:
                await self._function_stack.close()
            self.response_started = True
            self._trailers_expected = bool(message.get("trailers", False))
            headers = [
                (name, value)
                for name, value in message.get("headers", ())
                if name.lower() != _REQUEST_ID_HEADER
            ]
            headers.append((_REQUEST_ID_HEADER, self._request_id.encode("ascii")))
            message = {**message, "headers": headers}
        elif message["type"] in {"http.response.body", "http.response.pathsend"}:
            body_complete = message["type"] == "http.response.pathsend" or not message.get(
                "more_body",
                False,
            )
            if body_complete and not self._trailers_expected:
                self.response_complete = True
        elif message["type"] == "http.response.trailers" and not message.get(
            "more_trailers",
            False,
        ):
            self.response_complete = True
        await self._send(message)


class _TrackedAsyncExitStack(AsyncExitStack[bool | None]):
    def __init__(self) -> None:
        super().__init__()
        self._teardown_errors: list[BaseException] = []

    @staticmethod
    def _reject_untracked_exit_stack(value: object) -> None:
        if isinstance(value, (ExitStack, AsyncExitStack)) and not isinstance(
            value,
            _TrackedAsyncExitStack,
        ):
            raise RuntimeError("Untracked nested exit stacks are not allowed")

    @classmethod
    def _reject_untracked_registration(cls, value: object) -> None:
        cls._reject_untracked_exit_stack(value)
        cls._reject_untracked_exit_stack(getattr(value, "__self__", None))

    def enter_context[ContextValue](
        self,
        context_manager: AbstractContextManager[ContextValue, bool | None],
    ) -> ContextValue:
        self._reject_untracked_exit_stack(context_manager)
        return super().enter_context(context_manager)

    async def enter_async_context[ContextValue](
        self,
        context_manager: AbstractAsyncContextManager[ContextValue, bool | None],
    ) -> ContextValue:
        self._reject_untracked_exit_stack(context_manager)
        return await super().enter_async_context(context_manager)

    def push[ExitCallback](self, exit: ExitCallback) -> ExitCallback:
        self._reject_untracked_registration(exit)
        return cast(ExitCallback, super().push(cast(Any, exit)))

    def callback[**CallbackParams, CallbackResult](
        self,
        callback: Callable[CallbackParams, CallbackResult],
        /,
        *args: CallbackParams.args,
        **kwargs: CallbackParams.kwargs,
    ) -> Callable[CallbackParams, CallbackResult]:
        self._reject_untracked_registration(callback)
        return super().callback(callback, *args, **kwargs)

    def push_async_exit[ExitCallback](self, exit: ExitCallback) -> ExitCallback:
        self._reject_untracked_registration(exit)
        return cast(ExitCallback, super().push_async_exit(cast(Any, exit)))

    def push_async_callback[**CallbackParams, CallbackResult](
        self,
        callback: Callable[CallbackParams, Awaitable[CallbackResult]],
        /,
        *args: CallbackParams.args,
        **kwargs: CallbackParams.kwargs,
    ) -> Callable[CallbackParams, Awaitable[CallbackResult]]:
        self._reject_untracked_registration(callback)
        return super().push_async_callback(callback, *args, **kwargs)

    def _push_exit_callback(
        self,
        callback: _SyncExitCallback | _AsyncExitCallback,
        is_sync: bool = True,
    ) -> None:
        self._reject_untracked_registration(callback)
        teardown_errors = self._teardown_errors
        if is_sync:
            sync_callback = cast(_SyncExitCallback, callback)

            def tracked_sync_callback(
                exc_type: type[BaseException] | None,
                exc_value: BaseException | None,
                traceback: TracebackType | None,
            ) -> bool | None:
                try:
                    return sync_callback(exc_type, exc_value, traceback)
                except BaseException:
                    teardown_error = sys.exception()
                    assert teardown_error is not None
                    teardown_errors.append(teardown_error)
                    del teardown_error
                    raise

            super()._push_exit_callback(tracked_sync_callback, True)  # type: ignore[misc]
            return

        async_callback = cast(_AsyncExitCallback, callback)

        async def tracked_async_callback(
            exc_type: type[BaseException] | None,
            exc_value: BaseException | None,
            traceback: TracebackType | None,
        ) -> bool | None:
            try:
                return await async_callback(exc_type, exc_value, traceback)
            except BaseException:
                teardown_error = sys.exception()
                assert teardown_error is not None
                teardown_errors.append(teardown_error)
                del teardown_error
                raise

        super()._push_exit_callback(tracked_async_callback, False)  # type: ignore[misc]

    def pop_all(self) -> Self:
        transferred_stack = super().pop_all()
        transferred_stack._teardown_errors = self._teardown_errors
        self._teardown_errors = []
        return transferred_stack

    def _sanitize_recorded_teardown_errors(
        self,
        propagated_error: BaseException | None = None,
    ) -> bool:
        recorded_error = False
        while self._teardown_errors:
            teardown_error = self._teardown_errors.pop()
            recorded_error = True
            if teardown_error is not propagated_error:
                sanitize_exception_graph(teardown_error)
            del teardown_error
        return recorded_error

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        try:
            suppressed = await super().__aexit__(exc_type, exc_value, traceback)
        except BaseException:
            propagated_error = sys.exception()
            assert propagated_error is not None
            self._sanitize_recorded_teardown_errors(propagated_error)
            del propagated_error
            raise
        if self._sanitize_recorded_teardown_errors():
            raise RuntimeError("Dependency teardown failed") from None
        return suppressed


class _FunctionStackCleanupSignal(BaseException):
    pass


class _FunctionStackController:
    def __init__(self, stack: AsyncExitStack) -> None:
        self._stack = stack
        self._cleanup_error: BaseException | None = None
        self._cleanup_signal: _FunctionStackCleanupSignal | None = None
        self.poisoned = False
        self.closed = False

    def _record_cleanup_failure(self, error: BaseException) -> None:
        self.poisoned = True
        if self._cleanup_error is None:
            self._cleanup_error = error
            if isinstance(error, Exception):
                self._cleanup_signal = _FunctionStackCleanupSignal()

    def _raise_cleanup_failure(self) -> Never:
        cleanup_error = self._cleanup_error
        if isinstance(cleanup_error, Exception):
            cleanup_signal = self._cleanup_signal
            if cleanup_signal is None:
                cleanup_signal = _FunctionStackCleanupSignal()
                self._cleanup_signal = cleanup_signal
            raise cleanup_signal.with_traceback(cleanup_signal.__traceback__) from None
        if cleanup_error is not None:
            raise cleanup_error.with_traceback(cleanup_error.__traceback__) from None
        raise RuntimeError("Function dependency cleanup failed") from None

    def raise_if_poisoned(self) -> None:
        if self.poisoned:
            self._raise_cleanup_failure()

    async def close(self) -> None:
        self.raise_if_poisoned()
        if self.closed:
            return
        self.closed = True
        try:
            await self._stack.aclose()
        except BaseException:
            cleanup_error = sys.exception()
            assert cleanup_error is not None
            self._record_cleanup_failure(cleanup_error)
            del cleanup_error
            self._raise_cleanup_failure()

    async def close_for_error(self, error: BaseException) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            await self._stack.__aexit__(type(error), error, error.__traceback__)
        except BaseException:
            cleanup_error = sys.exception()
            assert cleanup_error is not None
            self._record_cleanup_failure(cleanup_error)
            del cleanup_error
            raise

    def resolve_error(
        self,
        error: BaseException,
        *,
        consume: bool = True,
    ) -> BaseException:
        cleanup_error = self._cleanup_error
        if cleanup_error is not None:
            if error is not cleanup_error:
                # First detach and clean the discarded exception graph without
                # mutating cleanup arguments whose identity must be preserved.
                sanitize_exception_graph(error, redact_args=False)
                sanitize_exception_graph(error)
            if consume:
                self._cleanup_error = None
                self._cleanup_signal = None
            return cleanup_error
        if isinstance(error, _FunctionStackCleanupSignal):
            if consume:
                self._cleanup_signal = None
            sanitize_exception_graph(error, redact_args=False)
            return RuntimeError("Function dependency cleanup failed")
        return error


def _log_unhandled_exception(
    error: BaseException,
    request_id: str,
) -> BaseExceptionGroup | None:
    exception_type = safe_exception_type(error)
    sanitized_error = sanitize_exception_graph(error)
    sanitized_group = sanitized_error if isinstance(sanitized_error, BaseExceptionGroup) else None
    try:
        logger.error(
            "Unhandled application exception",
            extra={
                "request_id": request_id,
                "exception_type": exception_type,
            },
            exc_info=None,
            stack_info=False,
        )
    except BaseException:
        logging_error = sys.exception()
        if logging_error is not None:
            sanitize_exception_graph(logging_error)
        del logging_error
    del sanitized_error
    return sanitized_group


def _prepare_error_response(
    error: Exception,
    request_id: str,
    response_started: bool,
) -> tuple[JSONResponse | None, BaseExceptionGroup | None]:
    if not response_started:
        try:
            if isinstance(error, BusinessError):
                return business_error_response(error, request_id), None
            if isinstance(error, RequestValidationError):
                sanitize_exception_graph(error)
                return validation_error_response(error, request_id), None
        except Exception as response_error:
            sanitize_exception_graph(error)
            return None, _log_unhandled_exception(response_error, request_id)
        except BaseException:
            base_response_error = sys.exception()
            assert base_response_error is not None
            sanitize_exception_graph(error)
            _log_unhandled_exception(base_response_error, request_id)
            del base_response_error
            return None, None
    return None, _log_unhandled_exception(error, request_id)


async def _run_http_request(
    app: ASGIApp,
    max_request_id_length: int,
    max_request_id_length_provider: (
        Callable[[Scope, AsyncExitStack, AsyncExitStack], int | Awaitable[int]] | None
    ),
    scope: Scope,
    receive: Receive,
    send: Send,
) -> None:
    request_id = _request_id_from_scope(scope, max_request_id_length)
    request_state = scope.setdefault("state", {})
    if isinstance(request_state, dict):
        request_state["request_id"] = request_id
    wrapped_send: _RequestIdSend | None = None
    error_response: JSONResponse | None = None
    sanitized_group: BaseExceptionGroup | None = None
    response_started = False
    provided_max_length: int | Awaitable[int] | None = None
    request_settings_stack = _TrackedAsyncExitStack()
    function_settings_stack = _TrackedAsyncExitStack()
    function_stack_controller = _FunctionStackController(function_settings_stack)
    downstream_completed = False
    pending_error: BaseException | None = None
    try:
        try:
            async with request_settings_stack:
                await function_settings_stack.__aenter__()
                try:
                    if max_request_id_length_provider is not None:
                        provided_max_length = max_request_id_length_provider(
                            scope,
                            request_settings_stack,
                            function_settings_stack,
                        )
                        max_request_id_length = (
                            await provided_max_length
                            if isawaitable(provided_max_length)
                            else provided_max_length
                        )
                        provided_max_length = None
                    request_id = _request_id_from_scope(scope, max_request_id_length)
                    if isinstance(request_state, dict):
                        request_state["request_id"] = request_id
                    wrapped_send = _RequestIdSend(
                        send,
                        request_id,
                        function_stack_controller,
                    )
                    await app(scope, receive, wrapped_send)
                    function_stack_controller.raise_if_poisoned()
                    downstream_completed = True
                except BaseException:
                    caught_exception = sys.exception()
                    assert caught_exception is not None
                    active_exception = function_stack_controller.resolve_error(
                        caught_exception,
                        consume=False,
                    )
                    pending_error = active_exception
                    try:
                        await function_stack_controller.close_for_error(active_exception)
                    except BaseException:
                        cleanup_exception = sys.exception()
                        assert cleanup_exception is not None
                        pending_error = cleanup_exception
                        if cleanup_exception is not active_exception:
                            sanitize_exception_graph(active_exception, redact_args=False)
                            sanitize_exception_graph(active_exception)
                        del cleanup_exception
                        del caught_exception
                        del active_exception
                        raise
                    if active_exception is caught_exception:
                        del caught_exception
                        del active_exception
                        raise
                    del caught_exception
                    raise active_exception.with_traceback(active_exception.__traceback__) from None
                else:
                    try:
                        await function_stack_controller.close()
                    except BaseException:
                        close_exception = sys.exception()
                        assert close_exception is not None
                        pending_error = function_stack_controller.resolve_error(
                            close_exception,
                            consume=False,
                        )
                        del close_exception
                        raise pending_error.with_traceback(pending_error.__traceback__) from None
            function_stack_controller.raise_if_poisoned()
            if pending_error is not None:
                raise pending_error.with_traceback(pending_error.__traceback__) from None
            if not downstream_completed:
                raise RuntimeError("ASGI application did not complete request processing")
            if wrapped_send is None or not wrapped_send.response_started:
                raise RuntimeError("ASGI application returned without starting a response")
            if not wrapped_send.response_complete:
                raise RuntimeError("ASGI application returned before completing the response")
            if isinstance(request_state, dict):
                request_state.pop(_REQUEST_SETTINGS_STATE_KEY, None)
            return
        except BaseException:
            request_exit_error = sys.exception()
            assert request_exit_error is not None
            resolved_error = function_stack_controller.resolve_error(request_exit_error)
            if resolved_error is request_exit_error:
                del request_exit_error
                del resolved_error
                raise
            del request_exit_error
            raise resolved_error.with_traceback(resolved_error.__traceback__) from None
    except Exception as error:
        if isinstance(request_state, dict):
            request_state.pop(_REQUEST_SETTINGS_STATE_KEY, None)
        response_started = wrapped_send is not None and wrapped_send.response_started
        error_response, sanitized_group = _prepare_error_response(
            error,
            request_id,
            response_started,
        )
    except BaseException:
        if isinstance(request_state, dict):
            request_state.pop(_REQUEST_SETTINGS_STATE_KEY, None)
        base_exception = sys.exception()
        assert base_exception is not None
        sanitize_exception_graph(base_exception, redact_args=False)
        del (
            base_exception,
            app,
            scope,
            receive,
            send,
            request_state,
            wrapped_send,
            error_response,
            provided_max_length,
            request_settings_stack,
            function_settings_stack,
            function_stack_controller,
            pending_error,
            max_request_id_length_provider,
        )
        raise

    if response_started:
        del app, scope, receive, send, request_state, wrapped_send, error_response
        del provided_max_length, request_settings_stack, function_settings_stack
        del function_stack_controller, pending_error
        del max_request_id_length_provider
        if sanitized_group is not None:
            await _raise_sanitized_group(sanitized_group)
        await _raise_safe_wrapper()

    wrapped_send = _RequestIdSend(send, request_id, None)
    try:
        response = error_response or internal_error_response(request_id)
        await response(scope, receive, wrapped_send)
        return
    except Exception as send_error:
        sanitized_send_error = sanitize_exception_graph(send_error)
        if isinstance(sanitized_send_error, BaseExceptionGroup):
            sanitized_group = sanitized_send_error
        else:
            sanitized_group = None
        del sanitized_send_error
    except BaseException:
        response_base_exception = sys.exception()
        assert response_base_exception is not None
        sanitize_exception_graph(response_base_exception, redact_args=False)
        del (
            response_base_exception,
            app,
            scope,
            receive,
            send,
            request_state,
            wrapped_send,
            error_response,
            response,
            provided_max_length,
            request_settings_stack,
            function_settings_stack,
            function_stack_controller,
            pending_error,
            max_request_id_length_provider,
        )
        raise

    del app, scope, receive, send, request_state, wrapped_send, error_response, response
    del provided_max_length, request_settings_stack, function_settings_stack
    del function_stack_controller, pending_error
    del max_request_id_length_provider
    if sanitized_group is not None:
        await _raise_sanitized_group(sanitized_group)
    await _raise_safe_wrapper()


class RequestErrorMiddleware:
    def __init__(
        self,
        app: ASGIApp,
        *,
        max_request_id_length: int = MAX_HEADER_VALUE_LENGTH,
        max_request_id_length_provider: (
            Callable[[Scope, AsyncExitStack, AsyncExitStack], int | Awaitable[int]] | None
        ) = None,
    ) -> None:
        self._app = app
        self._max_request_id_length = max_request_id_length
        self._max_request_id_length_provider = max_request_id_length_provider

    def __call__(self, scope: Scope, receive: Receive, send: Send) -> Awaitable[None]:
        if scope["type"] != "http":
            return self._app(scope, receive, send)
        return _run_http_request(
            self._app,
            self._max_request_id_length,
            self._max_request_id_length_provider,
            scope,
            receive,
            send,
        )
