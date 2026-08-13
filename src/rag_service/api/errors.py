import builtins
import re
import sys
import traceback
import unicodedata
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import FrozenInstanceError, dataclass
from types import MappingProxyType, TracebackType
from typing import Any, Protocol, cast

from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

from rag_service.api.constants import MAX_HEADER_VALUE_LENGTH

_NO_STORE = "no-store"
_ALLOWED_ERROR_HEADERS = frozenset({"www-authenticate"})
_SAFE_EXCEPTION_TYPE_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,63}")
_SAFE_EXCEPTION_MAX_NODES = 256
_SAFE_EXCEPTION_MAX_DEPTH = 64
_SAFE_EXCEPTION_MAX_CHILD_REFERENCES = 1_024


class _ExceptionAttributeDescriptor(Protocol):
    def __get__(
        self,
        instance: BaseException,
        owner: type[BaseException],
    ) -> object: ...

    def __set__(self, instance: BaseException, value: object) -> None: ...


@dataclass(slots=True)
class BusinessError(Exception):
    status_code: int
    code: str
    message: str
    retryable: bool = False
    headers: Mapping[str, str] | None = None

    _PUBLIC_FIELDS = frozenset({"status_code", "code", "message", "retryable", "headers"})

    def __setattr__(self, name: str, value: object) -> None:
        if name in self._PUBLIC_FIELDS:
            try:
                object.__getattribute__(self, name)
            except AttributeError:
                object.__setattr__(self, name, value)
                return
            raise FrozenInstanceError(f"cannot assign to field {name!r}")
        BaseException.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        if name in self._PUBLIC_FIELDS:
            raise FrozenInstanceError(f"cannot delete field {name!r}")
        BaseException.__delattr__(self, name)

    def __post_init__(self) -> None:
        if self.headers is not None:
            object.__setattr__(self, "headers", MappingProxyType(dict(self.headers)))
        object.__setattr__(self, "args", (self.message,))


class ErrorBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    message: str
    retryable: bool
    request_id: str


class ErrorEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    error: ErrorBody


def error_responses(*status_codes: int) -> dict[int | str, dict[str, Any]]:
    return {
        status_code: {
            "model": ErrorEnvelope,
            "description": "The request was rejected using the stable error envelope.",
        }
        for status_code in status_codes
    }


def _build_safe_exception_group(
    error: BaseExceptionGroup,
    children: tuple[BaseException, ...],
) -> BaseException:
    if isinstance(error, ExceptionGroup):
        exception_children = tuple(
            child if isinstance(child, Exception) else RuntimeError("<redacted>")
            for child in children
        )
        return ExceptionGroup("<redacted>", exception_children)
    return BaseExceptionGroup("<redacted>", children)


def _exception_group_children(error: BaseExceptionGroup) -> tuple[BaseException, ...]:
    children = BaseExceptionGroup.exceptions.__get__(error, BaseExceptionGroup)
    if (
        not isinstance(children, tuple)
        or not children
        or len(children) > _SAFE_EXCEPTION_MAX_CHILD_REFERENCES
    ):
        raise RuntimeError("unsafe exception graph")
    if not all(isinstance(child, BaseException) for child in children):
        raise RuntimeError("unsafe exception graph")
    return children


def _safe_exception_graph(error: BaseException) -> BaseException:
    pending: list[tuple[BaseException, int, bool]] = [(error, 0, False)]
    visiting: set[int] = set()
    discovered: set[int] = set()
    replacements: dict[int, BaseException] = {}
    group_children: dict[int, tuple[BaseException, ...]] = {}
    child_references = 0

    while pending:
        current, depth, expanded = pending.pop()
        identity = id(current)
        if identity in replacements:
            continue
        if expanded:
            visiting.discard(identity)
            if not isinstance(current, BaseExceptionGroup):
                replacements[identity] = (
                    RuntimeError("<redacted>")
                    if isinstance(current, Exception)
                    else BaseException("<redacted>")
                )
                continue
            children = group_children[identity]
            safe_children = tuple(replacements[id(child)] for child in children)
            replacements[identity] = _build_safe_exception_group(current, safe_children)
            continue

        if depth > _SAFE_EXCEPTION_MAX_DEPTH or identity in visiting:
            raise RuntimeError("unsafe exception graph")
        if identity not in discovered:
            discovered.add(identity)
            if len(discovered) > _SAFE_EXCEPTION_MAX_NODES:
                raise RuntimeError("unsafe exception graph")
        visiting.add(identity)
        pending.append((current, depth, True))
        if not isinstance(current, BaseExceptionGroup):
            continue
        children = _exception_group_children(current)
        group_children[identity] = children
        child_references += len(children)
        if child_references > _SAFE_EXCEPTION_MAX_CHILD_REFERENCES:
            raise RuntimeError("unsafe exception graph")
        for child in reversed(children):
            pending.append((child, depth + 1, False))

    return replacements[id(error)]


def _discard_scrub_failure(error: BaseException) -> None:
    traceback_descriptor = _exception_attribute_descriptor("__traceback__")
    try:
        retained_traceback = cast(
            TracebackType | None,
            traceback_descriptor.__get__(error, BaseException),
        )
    except BaseException:
        retained_traceback = None
    if retained_traceback is not None:
        with suppress(BaseException):
            traceback.clear_frames(retained_traceback)
    for name, value in (
        ("args", ("<redacted>",)),
        ("__traceback__", None),
        ("__cause__", None),
        ("__context__", None),
    ):
        with suppress(BaseException):
            descriptor = _exception_attribute_descriptor(name)
            descriptor.__set__(error, value)


def _exception_attribute_descriptor(name: str) -> _ExceptionAttributeDescriptor:
    if name not in {"args", "__traceback__", "__cause__", "__context__"}:
        raise ValueError("unsupported exception attribute")
    return cast(_ExceptionAttributeDescriptor, vars(BaseException)[name])


def _read_exception_attribute(error: BaseException, name: str) -> object | None:
    try:
        return cast(object | None, BaseException.__getattribute__(error, name))
    except BaseException:
        scrub_failure = sys.exception()
        if scrub_failure is not None:
            _discard_scrub_failure(scrub_failure)
        del scrub_failure
    descriptor = _exception_attribute_descriptor(name)
    try:
        return descriptor.__get__(error, BaseException)
    except BaseException:
        scrub_failure = sys.exception()
        if scrub_failure is not None:
            _discard_scrub_failure(scrub_failure)
        del scrub_failure
        return None


def _force_exception_attribute(error: BaseException, name: str, value: object) -> None:
    descriptor = _exception_attribute_descriptor(name)
    try:
        descriptor.__set__(error, value)
        return
    except BaseException:
        scrub_failure = sys.exception()
        if scrub_failure is not None:
            _discard_scrub_failure(scrub_failure)
        del scrub_failure
    for setter in (BaseException.__setattr__, object.__setattr__):
        try:
            setter(error, name, value)
            return
        except BaseException:
            scrub_failure = sys.exception()
            if scrub_failure is not None:
                _discard_scrub_failure(scrub_failure)
            del scrub_failure


def _scrub_validation_attribute(
    error: RequestValidationError,
    name: str,
    value: object,
) -> None:
    try:
        state = object.__getattribute__(error, "__dict__")
        if isinstance(state, dict):
            dict.__setitem__(state, name, value)
            return
    except BaseException:
        scrub_failure = sys.exception()
        if scrub_failure is not None:
            _discard_scrub_failure(scrub_failure)
        del scrub_failure
    _force_exception_attribute(error, name, value)


def _scrub_request_validation_error(error: RequestValidationError) -> None:
    _scrub_validation_attribute(error, "body", None)
    _scrub_validation_attribute(error, "_errors", [])
    retained_traceback = cast(
        TracebackType | None,
        _read_exception_attribute(error, "__traceback__"),
    )
    if retained_traceback is not None:
        with suppress(BaseException):
            traceback.clear_frames(retained_traceback)
    _force_exception_attribute(error, "args", ("<redacted>",))
    _force_exception_attribute(error, "__traceback__", None)


def _clear_exception_graph(error: BaseException, *, redact_args: bool) -> None:
    pending = [error]
    visited: set[int] = set()
    while pending:
        current = pending.pop()
        identity = id(current)
        if identity in visited:
            continue
        visited.add(identity)
        if isinstance(current, RequestValidationError):
            _scrub_request_validation_error(current)
            is_validation_error = True
        else:
            is_validation_error = False
        try:
            if isinstance(current, BaseExceptionGroup):
                children = _exception_group_children(current)
                pending.extend(children)
        except BaseException:
            scrub_failure = sys.exception()
            if scrub_failure is not None:
                _discard_scrub_failure(scrub_failure)
            del scrub_failure
        for attribute in ("__cause__", "__context__"):
            nested = _read_exception_attribute(current, attribute)
            if isinstance(nested, BaseException):
                pending.append(nested)
        retained_traceback = cast(
            TracebackType | None,
            _read_exception_attribute(current, "__traceback__"),
        )
        if retained_traceback is not None:
            with suppress(BaseException):
                traceback.clear_frames(retained_traceback)
        if redact_args and not is_validation_error:
            try:
                current.args = ("<redacted>",)
            except BaseException:
                scrub_failure = sys.exception()
                if scrub_failure is not None:
                    _discard_scrub_failure(scrub_failure)
                del scrub_failure
        for attribute in ("__traceback__", "__cause__", "__context__"):
            _force_exception_attribute(current, attribute, None)


def sanitize_exception_graph(
    error: BaseException,
    *,
    redact_args: bool = True,
) -> BaseException:
    try:
        if not redact_args or not isinstance(error, BaseExceptionGroup):
            return error
        try:
            return _safe_exception_graph(error)
        except BaseException:
            graph_failure = sys.exception()
            if graph_failure is not None:
                _discard_scrub_failure(graph_failure)
            del graph_failure
            return RuntimeError("<redacted>")
    finally:
        _clear_exception_graph(error, redact_args=redact_args)


def safe_exception_type(error: BaseException) -> str:
    try:
        error_type = type(error)
        name = error_type.__name__
        if (
            isinstance(name, str)
            and _SAFE_EXCEPTION_TYPE_PATTERN.fullmatch(name) is not None
            and getattr(builtins, name, None) is error_type
        ):
            return name
    except BaseException:
        pass
    return "Exception"


def _allowed_error_headers(headers: Mapping[str, str] | None) -> dict[str, str]:
    allowed: dict[str, str] = {}
    try:
        items = () if headers is None else headers.items()
        for name, value in items:
            if (
                isinstance(name, str)
                and isinstance(value, str)
                and name.lower() in _ALLOWED_ERROR_HEADERS
                and 0 < len(value) <= MAX_HEADER_VALUE_LENGTH
                and _is_safe_latin1_header_value(value)
            ):
                allowed["WWW-Authenticate"] = value
    except BaseException:
        return {}
    return allowed


def _is_safe_latin1_header_value(value: str) -> bool:
    try:
        value.encode("latin-1")
    except UnicodeEncodeError:
        return False
    return all(unicodedata.category(character) != "Cc" for character in value)


def _error_response(
    *,
    status_code: int,
    code: str,
    message: str,
    retryable: bool,
    request_id: str,
    headers: Mapping[str, str] | None = None,
) -> JSONResponse:
    response_headers = _allowed_error_headers(headers)
    response_headers["Cache-Control"] = _NO_STORE
    response_headers["X-Request-ID"] = request_id
    envelope = ErrorEnvelope(
        error={
            "code": code,
            "message": message,
            "retryable": retryable,
            "request_id": request_id,
        }
    )
    return JSONResponse(
        status_code=status_code,
        headers=response_headers,
        content=envelope.model_dump(mode="json"),
    )


def business_error_response(error: BusinessError, request_id: str) -> JSONResponse:
    status_code = error.status_code
    code = error.code
    message = error.message
    retryable = error.retryable
    headers = error.headers
    sanitized_error = sanitize_exception_graph(error)
    assert sanitized_error is error
    return _error_response(
        status_code=status_code,
        code=code,
        message=message,
        retryable=retryable,
        request_id=request_id,
        headers=headers,
    )


def validation_error_response(error: BaseException, request_id: str) -> JSONResponse:
    sanitized_error = sanitize_exception_graph(error)
    assert sanitized_error is error
    return _error_response(
        status_code=422,
        code="VALIDATION_ERROR",
        message="Invalid request",
        retryable=False,
        request_id=request_id,
    )


def internal_error_response(request_id: str) -> JSONResponse:
    return _error_response(
        status_code=500,
        code="INTERNAL_ERROR",
        message="Internal server error",
        retryable=False,
        request_id=request_id,
    )
