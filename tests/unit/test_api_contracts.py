import asyncio
import base64
import logging
import re
from collections.abc import Iterator
from contextlib import AsyncExitStack, ExitStack, contextmanager
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta
from types import TracebackType
from typing import Any, cast
from uuid import UUID

import httpx
import pytest
from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from starlette.types import Message, Receive, Scope, Send

from rag_service.api import errors as errors_module
from rag_service.api import middleware as middleware_module
from rag_service.api import validation as validation_module
from rag_service.api.cursors import CursorPosition, decode_cursor, encode_cursor
from rag_service.api.errors import BusinessError, ErrorEnvelope, sanitize_exception_graph
from rag_service.api.etags import agent_key_etag, knowledge_base_etag, require_matching_etag
from rag_service.api.middleware import RequestErrorMiddleware, request_id_for
from rag_service.api.validation import validate_bounded_json, validate_idempotency_key
from rag_service.config import Settings
from rag_service.main import create_app

FIXED_ID = UUID("12345678-1234-5678-1234-567812345678")
FIXED_TIMESTAMP = datetime(2025, 1, 2, 3, 4, 5, 678900, tzinfo=UTC)
TRACEBACK_SENTINEL = "traceback-local-contract-secret"


class _StickyRuntimeError(RuntimeError):
    def __setattr__(self, name: str, value: object) -> None:
        if name == "args" and hasattr(self, "args"):
            raise TypeError("args are immutable")
        BaseException.__setattr__(self, name, value)


class _HostileExceptionGroup(ExceptionGroup):
    @property
    def exceptions(self) -> tuple[Exception, ...]:
        raise SystemExit(TRACEBACK_SENTINEL)


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


def _assert_safe_escaping_tracebacks(error: BaseException) -> None:
    for node in _exception_nodes(error):
        assert TRACEBACK_SENTINEL not in repr(node.args)
        traceback = node.__traceback__
        while traceback is not None:
            module_name = traceback.tb_frame.f_globals.get("__name__", "")
            if isinstance(module_name, str) and module_name.startswith("rag_service"):
                frame_locals = traceback.tb_frame.f_locals
                assert TRACEBACK_SENTINEL not in repr(frame_locals)
                assert not {
                    "scope",
                    "receive",
                    "send",
                    "authorization",
                    "path",
                    "query_string",
                    "error",
                    "sanitized_error",
                    "sanitized_send_error",
                    "propagation_error",
                    "logging_error",
                    "provided_max_length",
                    "request_settings_stack",
                    "function_settings_stack",
                    "function_stack_controller",
                    "pending_error",
                    "error_response",
                    "active_exception",
                    "state",
                }.intersection(frame_locals)
            traceback = traceback.tb_next


def _http_scope() -> Scope:
    return {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": f"/{TRACEBACK_SENTINEL}",
        "raw_path": f"/{TRACEBACK_SENTINEL}".encode(),
        "query_string": f"token={TRACEBACK_SENTINEL}".encode(),
        "headers": [
            (b"authorization", f"Bearer {TRACEBACK_SENTINEL}".encode()),
        ],
        "client": ("127.0.0.1", 1234),
        "server": ("test", 80),
        "state": {},
    }


def _captured_runtime_error(marker: str) -> RuntimeError:
    retained_marker = marker
    try:
        assert retained_marker
        raise RuntimeError(marker)
    except RuntimeError as captured:
        return captured


async def _receive_request() -> Message:
    return {"type": "http.request", "body": b"", "more_body": False}


def test_request_id_accepts_safe_bounded_value_or_generates_prefixed_value() -> None:
    assert request_id_for("request-1:api.test") == "request-1:api.test"
    assert request_id_for("x" * 129).startswith("req_")
    assert request_id_for("not safe space").startswith("req_")


def test_generated_request_id_honors_configured_length() -> None:
    generated_length = len(request_id_for(None))

    assert len(request_id_for(None, max_length=generated_length)) <= generated_length
    with pytest.raises(ValueError, match="request ID length"):
        request_id_for(None, max_length=generated_length - 1)


def test_business_error_keeps_public_fields_frozen_but_allows_exception_state() -> None:
    error = BusinessError(422, "VALIDATION_ERROR", "Invalid request")

    error.__traceback__ = None
    error.__cause__ = None
    error.__context__ = None

    with pytest.raises(FrozenInstanceError):
        error.message = "must remain immutable"


def test_business_error_snapshots_headers_and_is_not_value_hashed() -> None:
    source_headers = {"WWW-Authenticate": "Bearer"}
    error = BusinessError(
        401,
        "INVALID_API_KEY",
        "Invalid API key",
        headers=source_headers,
    )

    source_headers["WWW-Authenticate"] = "Mutated"

    assert error.headers == {"WWW-Authenticate": "Bearer"}
    assert error == BusinessError(
        401,
        "INVALID_API_KEY",
        "Invalid API key",
        headers={"WWW-Authenticate": "Bearer"},
    )
    assert error.headers is not None
    with pytest.raises(TypeError):
        error.headers["WWW-Authenticate"] = "Mutated"  # type: ignore[index]
    with pytest.raises(TypeError):
        hash(BusinessError(500, "INTERNAL_ERROR", "Internal server error"))


def test_nested_exception_groups_are_replaced_with_a_disconnected_safe_graph() -> None:
    child = RuntimeError(f"child-{TRACEBACK_SENTINEL}")
    nested = ExceptionGroup(f"nested-{TRACEBACK_SENTINEL}", [child])
    original = ExceptionGroup(f"root-{TRACEBACK_SENTINEL}", [nested])

    sanitized = sanitize_exception_graph(original)

    assert sanitized is not original
    assert isinstance(sanitized, ExceptionGroup)
    assert sanitized.exceptions[0] is not nested
    for node in _exception_nodes(sanitized):
        assert TRACEBACK_SENTINEL not in str(node)
        assert TRACEBACK_SENTINEL not in getattr(node, "message", "")
        assert TRACEBACK_SENTINEL not in repr(node.args)


def test_exception_group_conversion_reuses_shared_safe_nodes() -> None:
    shared = ExceptionGroup(
        f"shared-{TRACEBACK_SENTINEL}",
        [RuntimeError(TRACEBACK_SENTINEL)],
    )
    original = ExceptionGroup(
        f"root-{TRACEBACK_SENTINEL}",
        [shared, shared],
    )

    sanitized = sanitize_exception_graph(original)

    assert isinstance(sanitized, ExceptionGroup)
    assert sanitized.exceptions[0] is sanitized.exceptions[1]


def test_exception_group_conversion_is_bounded_by_depth_and_clears_original() -> None:
    original: Exception = RuntimeError(TRACEBACK_SENTINEL)
    for depth in range(1_500):
        original = ExceptionGroup(f"depth-{depth}-{TRACEBACK_SENTINEL}", [original])
    try:
        raise original
    except Exception as captured:
        original = captured

    sanitized = sanitize_exception_graph(original)

    assert type(sanitized) is RuntimeError
    assert sanitized.args == ("<redacted>",)
    assert original.__traceback__ is None
    assert original.__cause__ is None
    assert original.__context__ is None


def test_exception_group_conversion_is_bounded_by_nodes_and_clears_original() -> None:
    original = ExceptionGroup(
        f"large-{TRACEBACK_SENTINEL}",
        [RuntimeError(f"child-{index}-{TRACEBACK_SENTINEL}") for index in range(300)],
    )
    try:
        raise original
    except ExceptionGroup:
        pass

    sanitized = sanitize_exception_graph(original)

    assert type(sanitized) is RuntimeError
    assert sanitized.args == ("<redacted>",)
    for node in _exception_nodes(original):
        assert node.__traceback__ is None
        assert node.__cause__ is None
        assert node.__context__ is None


def test_hostile_exception_group_uses_builtin_children_and_scrubs_hidden_validation() -> None:
    validation_error = RequestValidationError(
        [
            {
                "type": "int_parsing",
                "loc": ("body", "count"),
                "msg": "Input should be a valid integer",
                "input": TRACEBACK_SENTINEL,
            }
        ],
        body={"count": TRACEBACK_SENTINEL},
    )
    try:
        raise validation_error
    except RequestValidationError as captured:
        validation_error = captured
    original = _HostileExceptionGroup(
        f"hostile-{TRACEBACK_SENTINEL}",
        [validation_error],
    )
    try:
        raise original
    except ExceptionGroup:
        pass

    sanitized = sanitize_exception_graph(original)

    assert type(sanitized) is ExceptionGroup
    assert sanitized.args == ("<redacted>", sanitized.exceptions)
    assert original.__traceback__ is None
    assert original.__cause__ is None
    assert original.__context__ is None
    assert validation_error.body is None
    assert validation_error.errors() == []
    assert validation_error.args == ("<redacted>",)
    assert validation_error.__traceback__ is None


@pytest.mark.parametrize("nested", [False, True])
def test_hostile_exception_attribute_read_failures_are_discarded(
    nested: bool,
) -> None:
    hook_failures = {
        name: SystemExit(f"{name}-{TRACEBACK_SENTINEL}")
        for name in ("__cause__", "__context__", "__traceback__")
    }

    class HostileValidationError(RequestValidationError):
        @property
        def __cause__(self) -> BaseException | None:  # type: ignore[override]
            raise hook_failures["__cause__"]

        @property
        def __context__(self) -> BaseException | None:  # type: ignore[override]
            raise hook_failures["__context__"]

        @property
        def __traceback__(self) -> object:  # type: ignore[override]
            raise hook_failures["__traceback__"]

    validation_error = HostileValidationError(
        [
            {
                "type": "int_parsing",
                "loc": ("body", "count"),
                "msg": "Input should be a valid integer",
                "input": TRACEBACK_SENTINEL,
            }
        ],
        body={"count": TRACEBACK_SENTINEL},
    )
    try:
        raise validation_error
    except RequestValidationError as captured:
        validation_error = cast(HostileValidationError, captured)
    root: BaseException = (
        ExceptionGroup("nested hostile validation", [validation_error])
        if nested
        else validation_error
    )

    sanitize_exception_graph(root)

    assert object.__getattribute__(validation_error, "body") is None
    assert RequestValidationError.errors(validation_error) == []
    args_descriptor = cast(Any, vars(BaseException)["args"])
    traceback_descriptor = cast(Any, vars(BaseException)["__traceback__"])
    assert args_descriptor.__get__(validation_error, BaseException) == ("<redacted>",)
    assert traceback_descriptor.__get__(validation_error, BaseException) is None
    for hook_failure in hook_failures.values():
        assert hook_failure.args == ("<redacted>",)
        assert hook_failure.__traceback__ is None


@pytest.mark.parametrize("nested", [False, True])
def test_hostile_getattribute_is_bypassed_for_exception_graph_slots(nested: bool) -> None:
    hook_failures = {
        name: SystemExit(f"{name}-{TRACEBACK_SENTINEL}")
        for name in ("__cause__", "__context__", "__traceback__")
    }
    intercepted_reads: list[str] = []

    class HostileValidationError(RequestValidationError):
        def __getattribute__(self, name: str) -> object:
            if name in hook_failures:
                intercepted_reads.append(name)
                raise hook_failures[name]
            return super().__getattribute__(name)

    validation_error = HostileValidationError([], body={"count": TRACEBACK_SENTINEL})
    try:
        raise validation_error
    except RequestValidationError as captured:
        validation_error = cast(HostileValidationError, captured)
    root: BaseException = (
        ExceptionGroup("nested hostile getattribute", [validation_error])
        if nested
        else validation_error
    )

    sanitize_exception_graph(root)

    traceback_descriptor = cast(Any, vars(BaseException)["__traceback__"])
    assert intercepted_reads == []
    assert object.__getattribute__(validation_error, "body") is None
    assert RequestValidationError.errors(validation_error) == []
    assert traceback_descriptor.__get__(validation_error, BaseException) is None
    for hook_failure in hook_failures.values():
        assert hook_failure.__traceback__ is None


def test_safe_group_construction_failure_falls_back_after_clearing_original(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = ExceptionGroup(
        f"root-{TRACEBACK_SENTINEL}",
        [RuntimeError(TRACEBACK_SENTINEL)],
    )
    try:
        raise original
    except ExceptionGroup as captured:
        original = captured

    def fail_construction(*_args: object) -> BaseException:
        raise SystemExit(TRACEBACK_SENTINEL)

    monkeypatch.setattr(
        errors_module,
        "_build_safe_exception_group",
        fail_construction,
        raising=False,
    )

    sanitized = sanitize_exception_graph(original)

    assert type(sanitized) is RuntimeError
    assert sanitized.args == ("<redacted>",)
    for node in _exception_nodes(original):
        assert node.__traceback__ is None
        assert node.__cause__ is None
        assert node.__context__ is None


@pytest.mark.asyncio
async def test_response_started_exception_escapes_only_as_a_safe_wrapper() -> None:
    original = _StickyRuntimeError(TRACEBACK_SENTINEL)

    async def downstream(
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        authorization = dict(scope["headers"])[b"authorization"]
        path = scope["path"]
        query_string = scope["query_string"]
        assert authorization and path and query_string
        await send({"type": "http.response.start", "status": 200, "headers": []})
        raise original

    sent: list[Message] = []

    async def send(message: Message) -> None:
        sent.append(message)

    middleware = RequestErrorMiddleware(downstream)
    with pytest.raises(RuntimeError) as raised:
        await middleware(_http_scope(), _receive_request, send)

    assert sent[0]["type"] == "http.response.start"
    assert raised.value is not original
    assert raised.value.args == ("Request processing failed",)
    assert original.args == (TRACEBACK_SENTINEL,)
    assert original.__traceback__ is None
    _assert_safe_escaping_tracebacks(raised.value)


@pytest.mark.asyncio
async def test_response_started_exception_group_uses_the_safe_replacement_graph() -> None:
    original = ExceptionGroup(
        f"root-{TRACEBACK_SENTINEL}",
        [
            ExceptionGroup(
                f"nested-{TRACEBACK_SENTINEL}",
                [RuntimeError(TRACEBACK_SENTINEL)],
            )
        ],
    )

    async def downstream(
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        authorization = dict(scope["headers"])[b"authorization"]
        assert authorization
        await send({"type": "http.response.start", "status": 200, "headers": []})
        raise original

    async def send(_message: Message) -> None:
        return None

    middleware = RequestErrorMiddleware(downstream)
    with pytest.raises(ExceptionGroup) as raised:
        await middleware(_http_scope(), _receive_request, send)

    assert raised.value is not original
    for node in _exception_nodes(raised.value):
        assert TRACEBACK_SENTINEL not in str(node)
        assert TRACEBACK_SENTINEL not in getattr(node, "message", "")
        assert TRACEBACK_SENTINEL not in repr(node.args)
    _assert_safe_escaping_tracebacks(raised.value)


@pytest.mark.asyncio
async def test_overdeep_response_started_group_escapes_only_as_a_safe_wrapper() -> None:
    original: Exception = RuntimeError(TRACEBACK_SENTINEL)
    for depth in range(1_500):
        original = ExceptionGroup(f"depth-{depth}-{TRACEBACK_SENTINEL}", [original])

    async def downstream(scope: Scope, receive: Receive, send: Send) -> None:
        authorization = dict(scope["headers"])[b"authorization"]
        assert authorization
        await send({"type": "http.response.start", "status": 200, "headers": []})
        raise original

    async def send(_message: Message) -> None:
        return None

    middleware = RequestErrorMiddleware(downstream)
    with pytest.raises(RuntimeError) as raised:
        await middleware(_http_scope(), _receive_request, send)

    assert raised.value.args == ("Request processing failed",)
    assert original.__traceback__ is None
    _assert_safe_escaping_tracebacks(raised.value)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "logging_failure",
    [
        SystemExit(TRACEBACK_SENTINEL),
        asyncio.CancelledError(TRACEBACK_SENTINEL),
        BaseExceptionGroup(
            f"logging-{TRACEBACK_SENTINEL}",
            [SystemExit(TRACEBACK_SENTINEL)],
        ),
    ],
)
async def test_logging_base_exceptions_are_isolated_from_the_fallback_response(
    logging_failure: BaseException,
) -> None:
    class _FailingHandler(logging.Handler):
        def emit(self, _record: logging.LogRecord) -> None:
            raise logging_failure

    async def downstream(scope: Scope, receive: Receive, send: Send) -> None:
        authorization = dict(scope["headers"])[b"authorization"]
        assert authorization
        raise RuntimeError(TRACEBACK_SENTINEL)

    sent: list[Message] = []

    async def send(message: Message) -> None:
        sent.append(message)

    handler = _FailingHandler()
    logger = logging.getLogger("rag_service.api.middleware")
    logger.addHandler(handler)
    try:
        await RequestErrorMiddleware(downstream)(_http_scope(), _receive_request, send)
    finally:
        logger.removeHandler(handler)

    assert sent[0]["type"] == "http.response.start"
    assert sent[0]["status"] == 500
    for node in _exception_nodes(logging_failure):
        assert node.__traceback__ is None
        assert node.__cause__ is None
        assert node.__context__ is None


@pytest.mark.asyncio
async def test_logging_base_exception_does_not_change_response_started_semantics() -> None:
    logging_failure = SystemExit(TRACEBACK_SENTINEL)

    class _FailingHandler(logging.Handler):
        def emit(self, _record: logging.LogRecord) -> None:
            raise logging_failure

    async def downstream(scope: Scope, receive: Receive, send: Send) -> None:
        authorization = dict(scope["headers"])[b"authorization"]
        assert authorization
        await send({"type": "http.response.start", "status": 200, "headers": []})
        raise RuntimeError(TRACEBACK_SENTINEL)

    async def send(_message: Message) -> None:
        return None

    handler = _FailingHandler()
    logger = logging.getLogger("rag_service.api.middleware")
    logger.addHandler(handler)
    try:
        with pytest.raises(RuntimeError) as raised:
            await RequestErrorMiddleware(downstream)(
                _http_scope(),
                _receive_request,
                send,
            )
    finally:
        logger.removeHandler(handler)

    assert raised.value.args == ("Request processing failed",)
    assert logging_failure.__traceback__ is None
    _assert_safe_escaping_tracebacks(raised.value)


@pytest.mark.asyncio
async def test_fallback_response_send_failure_escapes_only_as_a_safe_wrapper() -> None:
    async def downstream(scope: Scope, receive: Receive, send: Send) -> None:
        authorization = dict(scope["headers"])[b"authorization"]
        assert authorization
        raise RuntimeError(TRACEBACK_SENTINEL)

    async def failing_send(message: Message) -> None:
        transport_secret = TRACEBACK_SENTINEL
        assert message and transport_secret
        raise _StickyRuntimeError(TRACEBACK_SENTINEL)

    middleware = RequestErrorMiddleware(downstream)
    with pytest.raises(RuntimeError) as raised:
        await middleware(_http_scope(), _receive_request, failing_send)

    assert raised.value.args == ("Request processing failed",)
    _assert_safe_escaping_tracebacks(raised.value)


@pytest.mark.asyncio
async def test_request_id_limit_provider_failure_is_handled_inside_the_safe_boundary() -> None:
    async def downstream(scope: Scope, receive: Receive, send: Send) -> None:
        raise AssertionError(f"downstream must not run: {scope!r} {receive!r} {send!r}")

    def failing_provider(
        scope: Scope,
        _request_settings_stack: object,
        _function_settings_stack: object,
    ) -> int:
        provider_scope = scope
        assert provider_scope["path"]
        raise _StickyRuntimeError(TRACEBACK_SENTINEL)

    sent: list[Message] = []

    async def send(message: Message) -> None:
        sent.append(message)

    middleware = RequestErrorMiddleware(
        downstream,
        max_request_id_length_provider=failing_provider,
    )

    await middleware(_http_scope(), _receive_request, send)

    assert sent[0]["type"] == "http.response.start"
    assert sent[0]["status"] == 500


@pytest.mark.asyncio
async def test_handled_provider_error_response_send_failure_uses_safe_wrapper() -> None:
    business_error = BusinessError(
        status_code=429,
        code="SETTINGS_BUSY",
        message="Settings provider busy",
    )
    send_failure = _StickyRuntimeError(TRACEBACK_SENTINEL)

    async def downstream(scope: Scope, receive: Receive, send: Send) -> None:
        raise AssertionError(f"downstream must not run: {scope!r} {receive!r} {send!r}")

    def failing_provider(
        _scope: Scope,
        _request_settings_stack: object,
        _function_settings_stack: object,
    ) -> int:
        raise business_error

    async def failing_send(_message: Message) -> None:
        raise send_failure

    middleware = RequestErrorMiddleware(
        downstream,
        max_request_id_length_provider=failing_provider,
    )
    with pytest.raises(RuntimeError) as raised:
        await middleware(_http_scope(), _receive_request, failing_send)

    assert raised.value.args == ("Request processing failed",)
    assert business_error.args == ("<redacted>",)
    assert business_error.__traceback__ is None
    assert send_failure.__traceback__ is None
    _assert_safe_escaping_tracebacks(raised.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("stack_scope", ["request", "function"])
@pytest.mark.parametrize(
    "registration",
    [
        "enter_context",
        "push",
        "callback",
        "push_async_exit",
        "push_async_callback",
    ],
)
async def test_all_public_stack_registration_paths_detect_suppressed_teardown_errors(
    stack_scope: str,
    registration: str,
) -> None:
    failure = _captured_runtime_error(TRACEBACK_SENTINEL)

    def provider(
        _scope: Scope,
        request_stack: AsyncExitStack,
        function_stack: AsyncExitStack,
    ) -> int:
        stack = request_stack if stack_scope == "request" else function_stack

        def suppressing_exit(
            _exc_type: type[BaseException] | None,
            _exc_value: BaseException | None,
            _traceback: object,
        ) -> bool:
            return True

        stack.push(suppressing_exit)
        if registration == "enter_context":

            @contextmanager
            def failing_context() -> Iterator[object]:
                yield object()
                raise failure

            stack.enter_context(failing_context())
        elif registration == "push":

            def failing_exit(
                _exc_type: type[BaseException] | None,
                _exc_value: BaseException | None,
                _traceback: object,
            ) -> None:
                raise failure

            stack.push(failing_exit)
        elif registration == "callback":

            def failing_callback() -> None:
                raise failure

            stack.callback(failing_callback)
        elif registration == "push_async_exit":

            async def failing_async_exit(
                _exc_type: type[BaseException] | None,
                _exc_value: BaseException | None,
                _traceback: object,
            ) -> None:
                raise failure

            stack.push_async_exit(failing_async_exit)
        else:

            async def failing_async_callback() -> None:
                raise failure

            stack.push_async_callback(failing_async_callback)
        return 32

    async def downstream(_scope: Scope, _receive: Receive, send: Send) -> None:
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    sent: list[Message] = []

    async def send(message: Message) -> None:
        sent.append(message)

    middleware = RequestErrorMiddleware(
        downstream,
        max_request_id_length_provider=provider,
    )
    if stack_scope == "request":
        with pytest.raises(RuntimeError) as raised:
            await middleware(_http_scope(), _receive_request, send)
        assert raised.value.args == ("Request processing failed",)
        assert sent[0]["status"] == 200
    else:
        await middleware(_http_scope(), _receive_request, send)
        assert sent[0]["status"] == 500

    assert failure.args == ("<redacted>",)
    assert failure.__traceback__ is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [
        RuntimeError(TRACEBACK_SENTINEL),
        SystemExit("safe function cleanup exit"),
        asyncio.CancelledError("safe function cleanup cancellation"),
    ],
    ids=["runtime-error", "system-exit", "cancelled-error"],
)
async def test_function_cleanup_failure_poisons_all_downstream_send_retries(
    failure: BaseException,
) -> None:
    caught_failures: list[BaseException] = []

    def provider(
        _scope: Scope,
        _request_stack: AsyncExitStack,
        function_stack: AsyncExitStack,
    ) -> int:
        async def fail_cleanup() -> None:
            raise failure

        function_stack.push_async_callback(fail_cleanup)
        return 32

    async def downstream(_scope: Scope, _receive: Receive, send: Send) -> None:
        attempts: tuple[Message, ...] = (
            {"type": "http.response.start", "status": 200, "headers": []},
            {"type": "http.response.start", "status": 201, "headers": []},
            {"type": "http.response.body", "body": b"must-not-send"},
        )
        for message in attempts:
            try:
                await send(message)
            except BaseException as caught:
                caught_failures.append(caught)

    sent: list[Message] = []

    async def send(message: Message) -> None:
        sent.append(message)

    middleware = RequestErrorMiddleware(
        downstream,
        max_request_id_length_provider=provider,
    )
    if isinstance(failure, Exception):
        await middleware(_http_scope(), _receive_request, send)
        assert [
            message.get("status") for message in sent if message["type"] == "http.response.start"
        ] == [500]
        assert all(message.get("body") != b"must-not-send" for message in sent)
        assert failure.args == ("<redacted>",)
        assert failure.__traceback__ is None
    else:
        with pytest.raises(type(failure)) as raised:
            await middleware(_http_scope(), _receive_request, send)
        assert raised.value is failure
        assert sent == []

    assert len(caught_failures) == 3
    assert all(caught is caught_failures[0] for caught in caught_failures)
    if isinstance(failure, Exception):
        assert caught_failures[0].__traceback__ is None
    else:
        _assert_safe_escaping_tracebacks(failure)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [
        RuntimeError(TRACEBACK_SENTINEL),
        SystemExit("safe close-for-error exit"),
        asyncio.CancelledError("safe close-for-error cancellation"),
    ],
    ids=["runtime-error", "system-exit", "cancelled-error"],
)
async def test_close_for_error_records_every_cleanup_base_exception(
    failure: BaseException,
) -> None:
    stack = AsyncExitStack()
    await stack.__aenter__()

    async def fail_cleanup() -> None:
        raise failure

    stack.push_async_callback(fail_cleanup)
    controller = middleware_module._FunctionStackController(stack)

    with pytest.raises(type(failure)) as raised:
        await controller.close_for_error(RuntimeError("route failed"))

    assert raised.value is failure
    assert controller.poisoned
    with pytest.raises(BaseException) as replayed:
        controller.raise_if_poisoned()
    if isinstance(failure, Exception):
        assert replayed.value is not failure
        assert controller.resolve_error(replayed.value) is failure
    else:
        assert replayed.value is failure


@pytest.mark.asyncio
@pytest.mark.parametrize("request_cleanup", ["replace", "suppress-then-raise"])
@pytest.mark.parametrize(
    "failure_kind",
    ["runtime-error", "system-exit", "cancelled-error"],
)
async def test_request_cleanup_cannot_mask_recorded_function_cleanup_poison(
    failure_kind: str,
    request_cleanup: str,
) -> None:
    if failure_kind == "runtime-error":
        function_failure: BaseException = RuntimeError(TRACEBACK_SENTINEL)
    elif failure_kind == "system-exit":
        function_failure = SystemExit("safe masked function cleanup exit")
    else:
        function_failure = asyncio.CancelledError("safe masked function cleanup cancellation")
    original_function_failure_args = function_failure.args
    request_replacement = BusinessError(
        status_code=409,
        code="REQUEST_CLEANUP_REPLACEMENT",
        message=f"request-replacement-{TRACEBACK_SENTINEL}",
    )
    route_failure = RuntimeError(f"route-{TRACEBACK_SENTINEL}")
    route_failure.__cause__ = function_failure
    route_traceback: TracebackType | None = None

    def provider(
        _scope: Scope,
        request_stack: AsyncExitStack,
        function_stack: AsyncExitStack,
    ) -> int:
        async def replace_request_error(
            _exc_type: type[BaseException] | None,
            _exc_value: BaseException | None,
            _traceback: object,
        ) -> None:
            raise request_replacement

        request_stack.push_async_exit(replace_request_error)
        if request_cleanup == "suppress-then-raise":

            async def suppress_function_error(
                _exc_type: type[BaseException] | None,
                _exc_value: BaseException | None,
                _traceback: object,
            ) -> bool:
                return True

            request_stack.push_async_exit(suppress_function_error)

        async def fail_function_cleanup() -> None:
            raise function_failure

        function_stack.push_async_callback(fail_function_cleanup)
        return 32

    async def downstream(_scope: Scope, _receive: Receive, _send: Send) -> None:
        nonlocal route_traceback
        route_local_secret = f"route-local-{TRACEBACK_SENTINEL}"
        try:
            raise route_failure
        except RuntimeError:
            route_traceback = route_failure.__traceback__
            assert route_local_secret
            raise

    sent: list[Message] = []

    async def send(message: Message) -> None:
        sent.append(message)

    middleware = RequestErrorMiddleware(
        downstream,
        max_request_id_length_provider=provider,
    )
    if isinstance(function_failure, Exception):
        await middleware(_http_scope(), _receive_request, send)
        assert sent[0]["type"] == "http.response.start"
        assert sent[0]["status"] == 500
        assert function_failure.args == ("<redacted>",)
        assert function_failure.__traceback__ is None
    else:
        with pytest.raises(type(function_failure)) as raised:
            await middleware(_http_scope(), _receive_request, send)
        assert raised.value is function_failure
        assert function_failure.args == original_function_failure_args
        assert sent == []
        _assert_safe_escaping_tracebacks(function_failure)

    assert request_replacement.args == ("<redacted>",)
    assert request_replacement.__traceback__ is None
    assert route_failure.args == ("<redacted>",)
    assert route_failure.__traceback__ is None
    assert route_traceback is not None
    while route_traceback is not None:
        assert TRACEBACK_SENTINEL not in repr(route_traceback.tb_frame.f_locals)
        route_traceback = route_traceback.tb_next


@pytest.mark.asyncio
@pytest.mark.parametrize("stack_scope", ["request", "function"])
@pytest.mark.parametrize("composition", ["sync", "async"])
async def test_untracked_nested_exit_stack_composition_is_rejected(
    stack_scope: str,
    composition: str,
) -> None:
    cleanup_events: list[str] = []
    downstream_calls = 0

    async def provider(
        _scope: Scope,
        request_stack: AsyncExitStack,
        function_stack: AsyncExitStack,
    ) -> int:
        stack = request_stack if stack_scope == "request" else function_stack
        if composition == "sync":
            nested_stack = ExitStack()

            def suppressing_exit(
                _exc_type: type[BaseException] | None,
                _exc_value: BaseException | None,
                _traceback: object,
            ) -> bool:
                cleanup_events.append("sync-suppressed")
                return True

            def failing_callback() -> None:
                cleanup_events.append("sync-failed")
                raise RuntimeError(TRACEBACK_SENTINEL)

            nested_stack.push(suppressing_exit)
            nested_stack.callback(failing_callback)
            stack.enter_context(nested_stack)
        else:
            nested_async_stack = AsyncExitStack()

            async def suppressing_async_exit(
                _exc_type: type[BaseException] | None,
                _exc_value: BaseException | None,
                _traceback: object,
            ) -> bool:
                cleanup_events.append("async-suppressed")
                return True

            async def failing_async_callback() -> None:
                cleanup_events.append("async-failed")
                raise RuntimeError(TRACEBACK_SENTINEL)

            nested_async_stack.push_async_exit(suppressing_async_exit)
            nested_async_stack.push_async_callback(failing_async_callback)
            await stack.enter_async_context(nested_async_stack)
        return 32

    async def downstream(_scope: Scope, _receive: Receive, send: Send) -> None:
        nonlocal downstream_calls
        downstream_calls += 1
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    sent: list[Message] = []

    async def send(message: Message) -> None:
        sent.append(message)

    await RequestErrorMiddleware(
        downstream,
        max_request_id_length_provider=provider,
    )(_http_scope(), _receive_request, send)

    assert downstream_calls == 0
    assert cleanup_events == []
    assert sent[0]["type"] == "http.response.start"
    assert sent[0]["status"] == 500


@pytest.mark.asyncio
@pytest.mark.parametrize("stack_scope", ["request", "function"])
@pytest.mark.parametrize("composition", ["sync", "async"])
async def test_untracked_nested_stack_bound_cleanup_callback_is_rejected(
    stack_scope: str,
    composition: str,
) -> None:
    cleanup_events: list[str] = []
    downstream_calls = 0

    def provider(
        _scope: Scope,
        request_stack: AsyncExitStack,
        function_stack: AsyncExitStack,
    ) -> int:
        stack = request_stack if stack_scope == "request" else function_stack
        if composition == "sync":
            nested_stack = ExitStack()

            def suppressing_exit(
                _exc_type: type[BaseException] | None,
                _exc_value: BaseException | None,
                _traceback: object,
            ) -> bool:
                cleanup_events.append("sync-suppressed")
                return True

            def failing_callback() -> None:
                cleanup_events.append("sync-failed")
                raise RuntimeError(TRACEBACK_SENTINEL)

            nested_stack.push(suppressing_exit)
            nested_stack.callback(failing_callback)
            stack.callback(nested_stack.close)
        else:
            nested_async_stack = AsyncExitStack()

            async def suppressing_async_exit(
                _exc_type: type[BaseException] | None,
                _exc_value: BaseException | None,
                _traceback: object,
            ) -> bool:
                cleanup_events.append("async-suppressed")
                return True

            async def failing_async_callback() -> None:
                cleanup_events.append("async-failed")
                raise RuntimeError(TRACEBACK_SENTINEL)

            nested_async_stack.push_async_exit(suppressing_async_exit)
            nested_async_stack.push_async_callback(failing_async_callback)
            stack.push_async_callback(nested_async_stack.aclose)
        return 32

    async def downstream(_scope: Scope, _receive: Receive, send: Send) -> None:
        nonlocal downstream_calls
        downstream_calls += 1
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    sent: list[Message] = []

    async def send(message: Message) -> None:
        sent.append(message)

    await RequestErrorMiddleware(
        downstream,
        max_request_id_length_provider=provider,
    )(_http_scope(), _receive_request, send)

    assert downstream_calls == 0
    assert cleanup_events == []
    assert sent[0]["type"] == "http.response.start"
    assert sent[0]["status"] == 500


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [
        asyncio.CancelledError("safe cancellation"),
        SystemExit("safe system exit"),
        BaseExceptionGroup("safe base group", [SystemExit("safe child")]),
    ],
)
async def test_base_exceptions_preserve_identity_and_args_without_traceback_locals(
    failure: BaseException,
) -> None:
    original_args = failure.args
    scope = _http_scope()
    request_state = scope["state"]
    assert isinstance(request_state, dict)
    request_state["_rag_runtime_settings"] = Settings(_env_file=None)

    async def downstream(scope: Scope, receive: Receive, send: Send) -> None:
        authorization = dict(scope["headers"])[b"authorization"]
        path = scope["path"]
        query_string = scope["query_string"]
        assert authorization and path and query_string
        raise failure

    async def send(_message: Message) -> None:
        raise AssertionError("no response should be sent")

    middleware = RequestErrorMiddleware(downstream)
    with pytest.raises(type(failure)) as raised:
        await middleware(scope, _receive_request, send)

    assert raised.value is failure
    assert raised.value.args == original_args
    assert "_rag_runtime_settings" not in request_state
    _assert_safe_escaping_tracebacks(raised.value)


@pytest.mark.asyncio
async def test_application_assigns_request_id_before_routes_and_overrides_response_spoofing() -> (
    None
):
    settings = Settings(_env_file=None, max_request_id_length=32)
    app = create_app(settings=settings)

    @app.get("/_contract/request-id")
    async def request_id_contract(request: Request) -> JSONResponse:
        return JSONResponse(
            {"request_id": request.state.request_id},
            headers={"X-Request-ID": "unsafe-downstream-value"},
        )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        echoed = await client.get(
            "/_contract/request-id",
            headers={"X-Request-ID": "request-1:api.test"},
        )
        generated = await client.get("/_contract/request-id")
        duplicate = await client.get(
            "/_contract/request-id",
            headers=[("X-Request-ID", "first"), ("X-Request-ID", "second")],
        )

    assert echoed.headers["x-request-id"] == "request-1:api.test"
    assert echoed.json() == {"request_id": "request-1:api.test"}
    for response in (generated, duplicate):
        request_id = response.headers["x-request-id"]
        assert re.fullmatch(r"req_[A-Za-z0-9_-]+", request_id)
        assert len(request_id) <= settings.max_request_id_length
        assert response.json() == {"request_id": request_id}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [SystemExit("safe system exit"), asyncio.CancelledError("safe cancellation")],
)
async def test_application_error_boundary_does_not_swallow_base_exceptions(
    failure: BaseException,
) -> None:
    app = create_app(settings=Settings(_env_file=None))

    @app.get("/_contract/base-exception")
    async def base_exception_contract() -> None:
        raise failure

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=True),
        base_url="http://test",
    ) as client:
        with pytest.raises(type(failure)) as raised:
            await client.get("/_contract/base-exception")

    assert raised.value is failure


def test_cursor_round_trip_has_canonical_wire_format() -> None:
    position = CursorPosition(created_at=FIXED_TIMESTAMP, id=FIXED_ID)

    cursor = encode_cursor(position)

    expected_cursor = (
        "eyJ2IjoxLCJjcmVhdGVkX2F0IjoiMjAyNS0wMS0wMlQwMzowNDowNS42Nzg5MDBaIiwiaWQiOiIx"
        "MjM0NTY3OC0xMjM0LTU2NzgtMTIzNC01Njc4MTIzNDU2NzgifQ"
    )
    assert cursor == expected_cursor
    assert decode_cursor(cursor) == position


@pytest.mark.parametrize(
    "value",
    [
        "!not-base64!",
        base64.urlsafe_b64encode(
            b'{"v":2,"created_at":"2025-01-02T03:04:05.678900Z","id":"12345678-1234-5678-1234-567812345678"}'
        )
        .decode()
        .rstrip("="),
        base64.urlsafe_b64encode(
            b'{"v":1,"created_at":"2025-01-02T03:04:05+00:00","id":"12345678-1234-5678-1234-567812345678"}'
        )
        .decode()
        .rstrip("="),
        base64.urlsafe_b64encode(
            b'{"v":1,"created_at":"2025-01-02T03:04:05.678900Z","id":"not-a-uuid"}'
        )
        .decode()
        .rstrip("="),
        "x" * 513,
    ],
)
def test_cursor_rejects_malformed_or_unsafe_values_without_echoing_input(value: str) -> None:
    with pytest.raises(BusinessError) as exc_info:
        decode_cursor(value)

    assert exc_info.value.status_code == 422
    assert exc_info.value.code == "VALIDATION_ERROR"
    assert exc_info.value.retryable is False
    assert value not in str(exc_info.value)


def test_invalid_cursor_failures_have_independent_bounded_exceptions() -> None:
    def capture(value: str) -> BusinessError:
        try:
            decode_cursor(value)
        except BusinessError as error:
            return error
        raise AssertionError("invalid cursor was accepted")

    def traceback_depth(error: BusinessError) -> int:
        depth = 0
        traceback = error.__traceback__
        while traceback is not None:
            depth += 1
            traceback = traceback.tb_next
        return depth

    first = capture("!first-invalid-cursor!")
    first_traceback = first.__traceback__
    second = capture("!second-invalid-cursor!")

    assert first is not second
    assert first.__traceback__ is first_traceback
    assert first.__context__ is None
    assert second.__context__ is None
    assert traceback_depth(first) <= 2
    assert traceback_depth(second) <= 2

    for error, value in (
        (first, "!first-invalid-cursor!"),
        (second, "!second-invalid-cursor!"),
    ):
        traceback = error.__traceback__
        while traceback is not None:
            module_name = traceback.tb_frame.f_globals.get("__name__", "")
            if isinstance(module_name, str) and module_name.startswith("rag_service"):
                assert value not in repr(traceback.tb_frame.f_locals)
            traceback = traceback.tb_next


def test_etags_have_exact_wire_forms_and_require_current_strong_value() -> None:
    kb = knowledge_base_etag(FIXED_ID, 3)
    agent = agent_key_etag(FIXED_ID, 4)

    assert kb == '"kb:12345678-1234-5678-1234-567812345678:r3"'
    assert agent == '"agent-key:12345678-1234-5678-1234-567812345678:r4"'
    require_matching_etag(kb, kb)

    stale = '"kb:12345678-1234-5678-1234-567812345678:r2"'
    for invalid in (None, "W/" + kb, "*", kb + ', "other"', "3", stale):
        with pytest.raises(BusinessError) as exc_info:
            require_matching_etag(invalid, kb)
        assert exc_info.value == BusinessError(412, "PRECONDITION_FAILED", "Precondition failed")


@pytest.mark.parametrize("revision", [0, -1, True, 1.5])
def test_etags_require_positive_integer_revisions(revision: int | float) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        knowledge_base_etag(FIXED_ID, revision)  # type: ignore[arg-type]


def test_idempotency_key_validation_is_bounded_and_safe() -> None:
    assert validate_idempotency_key("idempotency-key-1") == "idempotency-key-1"
    for value in ("", "contains space", "x" * 129):
        with pytest.raises(BusinessError) as exc_info:
            validate_idempotency_key(value)
        assert exc_info.value.status_code == 422
        assert exc_info.value.code == "VALIDATION_ERROR"
        assert exc_info.value.retryable is False
        if value:
            assert value not in str(exc_info.value)


def test_idempotency_key_honors_configured_and_hard_limits() -> None:
    assert validate_idempotency_key("x" * 8, max_length=8) == "x" * 8

    for value, max_length in (("x" * 9, 8), ("x" * 129, 1_000)):
        with pytest.raises(BusinessError) as exc_info:
            validate_idempotency_key(value, max_length=max_length)
        assert exc_info.value.status_code == 422
        assert exc_info.value.code == "VALIDATION_ERROR"
        assert value not in str(exc_info.value)


def test_idempotency_failures_are_fresh_and_do_not_retain_request_values() -> None:
    def capture(value: str) -> BusinessError:
        try:
            validate_idempotency_key(value)
        except BusinessError as error:
            return error
        raise AssertionError("invalid idempotency key was accepted")

    values = ("first idempotency sentinel", "second idempotency sentinel")
    errors = tuple(capture(value) for value in values)

    assert errors[0] is not errors[1]
    for error, value in zip(errors, values, strict=True):
        assert error.__context__ is None
        assert error.__cause__ is None
        service_frames = 0
        traceback = error.__traceback__
        while traceback is not None:
            module_name = traceback.tb_frame.f_globals.get("__name__", "")
            if isinstance(module_name, str) and module_name.startswith("rag_service"):
                service_frames += 1
                assert value not in repr(traceback.tb_frame.f_locals)
            traceback = traceback.tb_next
        assert service_frames <= 1


def test_bounded_json_rejects_key_count_depth_and_canonical_size_without_echoing_payload() -> None:
    oversized = "sensitive-payload-" * 3_000
    values = [
        {str(index): index for index in range(65)},
        {"a": {"b": {"c": {"d": {"e": 1}}}}},
        {"payload": oversized},
    ]

    assert validate_bounded_json({"ok": [True, None, 1.5]}) == {"ok": [True, None, 1.5]}
    for value in values:
        with pytest.raises(BusinessError) as exc_info:
            validate_bounded_json(value)
        assert exc_info.value.status_code == 422
        assert exc_info.value.code == "VALIDATION_ERROR"
        assert exc_info.value.retryable is False
        assert "sensitive-payload" not in str(exc_info.value)


def test_bounded_json_rejects_non_json_compatible_values() -> None:
    with pytest.raises(BusinessError, match="Invalid JSON payload") as exc_info:
        validate_bounded_json({"at": FIXED_TIMESTAMP + timedelta(seconds=1)})

    assert exc_info.value.status_code == 422
    assert exc_info.value.code == "VALIDATION_ERROR"
    assert exc_info.value.retryable is False


@pytest.mark.parametrize(
    "value",
    [
        {"outer": {str(index): index for index in range(64)}},
        {"value": float("nan")},
        {"value": float("inf")},
        {"value": float("-inf")},
        [0] * 16_384,
    ],
)
def test_bounded_json_rejects_cumulative_keys_non_finite_numbers_and_large_arrays(
    value: object,
) -> None:
    with pytest.raises(BusinessError) as exc_info:
        validate_bounded_json(value)

    assert exc_info.value.status_code == 422
    assert exc_info.value.code == "VALIDATION_ERROR"
    assert exc_info.value.retryable is False


def test_bounded_json_rejects_adversarial_array_without_traversing_it() -> None:
    class TraversalBomb(list[int]):
        def __iter__(self):  # type: ignore[no-untyped-def]
            raise AssertionError("oversized array must be rejected before traversal")

    value = TraversalBomb([0] * 100_000)

    with pytest.raises(BusinessError) as exc_info:
        validate_bounded_json(value)

    assert exc_info.value.status_code == 422
    assert exc_info.value.code == "VALIDATION_ERROR"


def test_bounded_json_rejects_oversized_dict_before_key_traversal() -> None:
    class DictionaryTraversalBomb(dict[str, int]):
        def __len__(self) -> int:
            return 1_000_000

        def __iter__(self):  # type: ignore[no-untyped-def]
            raise AssertionError("oversized dictionary keys must not be traversed")

    value = DictionaryTraversalBomb({"sentinel": 1})

    with pytest.raises(BusinessError) as exc_info:
        validate_bounded_json(value)

    assert exc_info.value.status_code == 422
    assert exc_info.value.code == "VALIDATION_ERROR"


@pytest.mark.parametrize(
    "value",
    [
        pytest.param("long-value-sentinel" * 2_000, id="single-value"),
        pytest.param({"long-key-sentinel" * 2_000: None}, id="single-key"),
        pytest.param(
            ["first-string-sentinel" * 900, "second-string-sentinel" * 900],
            id="cumulative-values",
        ),
    ],
)
def test_bounded_json_rejects_raw_string_budget_before_encoder(
    value: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class EncoderBomb:
        def iterencode(self, value: object):  # type: ignore[no-untyped-def]
            raise AssertionError("oversized strings must be rejected before encoding")

    monkeypatch.setattr(validation_module, "_JSON_ENCODER", EncoderBomb())

    with pytest.raises(BusinessError) as exc_info:
        validate_bounded_json(value)

    assert exc_info.value.status_code == 422
    assert exc_info.value.code == "VALIDATION_ERROR"


def test_bounded_json_raw_string_budget_keeps_canonical_boundary_valid() -> None:
    value = "x" * (32 * 1024 - 2)

    assert validate_bounded_json(value) is value


def test_bounded_json_failures_are_fresh_and_do_not_retain_request_values() -> None:
    def capture(value: object) -> BusinessError:
        try:
            validate_bounded_json(value)
        except BusinessError as error:
            return error
        raise AssertionError("invalid JSON value was accepted")

    sentinels = (
        "first-json-sentinel" * 2_000,
        "second-json-sentinel" * 2_000,
    )
    errors = tuple(capture({"payload": sentinel}) for sentinel in sentinels)

    assert errors[0] is not errors[1]
    for error, sentinel in zip(errors, sentinels, strict=True):
        assert error.__context__ is None
        assert error.__cause__ is None
        service_frames = 0
        traceback = error.__traceback__
        while traceback is not None:
            module_name = traceback.tb_frame.f_globals.get("__name__", "")
            if isinstance(module_name, str) and module_name.startswith("rag_service"):
                service_frames += 1
                frame_locals = repr(traceback.tb_frame.f_locals)
                assert sentinel not in frame_locals
                assert "payload" not in frame_locals
            traceback = traceback.tb_next
        assert service_frames <= 1


def test_bounded_json_validates_in_place_and_accepts_maximal_compact_array() -> None:
    value = [0] * 16_383

    assert validate_bounded_json(value) is value


def test_business_error_serializes_to_the_safe_contract() -> None:
    error = BusinessError(429, "RATE_LIMITED", "Try again later", retryable=True)
    envelope = ErrorEnvelope(
        error={
            "code": error.code,
            "message": error.message,
            "retryable": error.retryable,
            "request_id": "req_fixed",
        }
    )

    assert envelope.model_dump() == {
        "error": {
            "code": "RATE_LIMITED",
            "message": "Try again later",
            "retryable": True,
            "request_id": "req_fixed",
        }
    }
    assert str(error) == "Try again later"


def test_error_contract_requires_all_fields() -> None:
    with pytest.raises(ValidationError):
        ErrorEnvelope(error={"code": "BAD"})
