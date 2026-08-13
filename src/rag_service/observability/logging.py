"""Content-free structured logging primitives.

The sanitizer is deliberately fail-closed. Application strings are unsafe by
default; only exact code-owned event/status values and bounded operational IDs
survive.
"""

from __future__ import annotations

import logging
import math
import re
import time
from dataclasses import dataclass
from typing import cast
from uuid import UUID

_MAX_REQUEST_ID_LENGTH = 128
_MAX_EXCEPTION_TYPES = 32
_MAX_SANITIZE_DEPTH = 8
_MAX_COUNTER_VALUE = 2**63 - 1
_REQUEST_ID_PATTERN = re.compile(r"[A-Za-z0-9._:-]+")
_MISSING = object()

_EVENT_NAMES = frozenset(
    {
        "upload.completed",
        "upload.failed",
        "job.state.changed",
        "job.lease.recovered",
        "ingestion.stage.completed",
        "ingestion.stage.failed",
        "provider.request.completed",
        "provider.retry.scheduled",
        "qdrant.upsert.completed",
        "qdrant.search.completed",
        "cleanup.action.completed",
    }
)
_ERROR_CODES = frozenset(
    {
        "OTHER",
        "UPLOAD_REJECTED",
        "UPLOAD_FAILED",
        "JOB_FAILED",
        "JOB_LEASE_LOST",
        "STAGE_FAILED",
        "PARSE_FAILED",
        "CHUNK_FAILED",
        "PROVIDER_ERROR",
        "PROVIDER_RATE_LIMITED",
        "PROVIDER_TIMEOUT",
        "QDRANT_UPSERT_FAILED",
        "QDRANT_SEARCH_FAILED",
        "VALIDATION_FAILED",
        "ACTIVATION_FAILED",
        "LEASE_LOST",
    }
)
_BUILTIN_EXCEPTION_TYPES: tuple[tuple[type[BaseException], str], ...] = (
    (BaseException, "BaseException"),
    (BaseExceptionGroup, "BaseExceptionGroup"),
    (Exception, "Exception"),
    (ExceptionGroup, "ExceptionGroup"),
    (ArithmeticError, "ArithmeticError"),
    (AssertionError, "AssertionError"),
    (AttributeError, "AttributeError"),
    (BlockingIOError, "BlockingIOError"),
    (BufferError, "BufferError"),
    (BytesWarning, "BytesWarning"),
    (ChildProcessError, "ChildProcessError"),
    (DeprecationWarning, "DeprecationWarning"),
    (EOFError, "EOFError"),
    (EncodingWarning, "EncodingWarning"),
    (FloatingPointError, "FloatingPointError"),
    (FutureWarning, "FutureWarning"),
    (GeneratorExit, "GeneratorExit"),
    (ImportError, "ImportError"),
    (ImportWarning, "ImportWarning"),
    (IndentationError, "IndentationError"),
    (IndexError, "IndexError"),
    (KeyError, "KeyError"),
    (KeyboardInterrupt, "KeyboardInterrupt"),
    (LookupError, "LookupError"),
    (MemoryError, "MemoryError"),
    (ModuleNotFoundError, "ModuleNotFoundError"),
    (NameError, "NameError"),
    (NotImplementedError, "NotImplementedError"),
    (OSError, "OSError"),
    (OverflowError, "OverflowError"),
    (BrokenPipeError, "BrokenPipeError"),
    (ConnectionError, "ConnectionError"),
    (ConnectionAbortedError, "ConnectionAbortedError"),
    (ConnectionRefusedError, "ConnectionRefusedError"),
    (ConnectionResetError, "ConnectionResetError"),
    (FileExistsError, "FileExistsError"),
    (FileNotFoundError, "FileNotFoundError"),
    (InterruptedError, "InterruptedError"),
    (IsADirectoryError, "IsADirectoryError"),
    (NotADirectoryError, "NotADirectoryError"),
    (PermissionError, "PermissionError"),
    (ProcessLookupError, "ProcessLookupError"),
    (TimeoutError, "TimeoutError"),
    (PendingDeprecationWarning, "PendingDeprecationWarning"),
    (RecursionError, "RecursionError"),
    (ReferenceError, "ReferenceError"),
    (ResourceWarning, "ResourceWarning"),
    (RuntimeError, "RuntimeError"),
    (RuntimeWarning, "RuntimeWarning"),
    (StopAsyncIteration, "StopAsyncIteration"),
    (StopIteration, "StopIteration"),
    (SyntaxError, "SyntaxError"),
    (SyntaxWarning, "SyntaxWarning"),
    (SystemError, "SystemError"),
    (SystemExit, "SystemExit"),
    (TabError, "TabError"),
    (TypeError, "TypeError"),
    (UnboundLocalError, "UnboundLocalError"),
    (UnicodeDecodeError, "UnicodeDecodeError"),
    (UnicodeEncodeError, "UnicodeEncodeError"),
    (UnicodeError, "UnicodeError"),
    (UnicodeTranslateError, "UnicodeTranslateError"),
    (UnicodeWarning, "UnicodeWarning"),
    (UserWarning, "UserWarning"),
    (ValueError, "ValueError"),
    (Warning, "Warning"),
    (ZeroDivisionError, "ZeroDivisionError"),
)

_CONTEXT_ID_KEYS = frozenset(
    {
        "request_id",
        "knowledge_base_id",
        "document_id",
        "version_id",
        "job_id",
        "generation_id",
    }
)
_UUID_CONTEXT_KEYS = _CONTEXT_ID_KEYS - {"request_id"}
_ENUM_VALUES: dict[str, frozenset[str]] = {
    "capability": frozenset({"embedding", "retrieval", "ingestion"}),
    "operation": frozenset(
        {
            "upload",
            "job",
            "parse",
            "chunk",
            "embed_index",
            "validate",
            "activate",
            "provider_request",
            "qdrant_upsert",
            "qdrant_search",
            "orphan_cleanup",
            "ingest_document",
            "rebuild_generation",
        }
    ),
    "outcome": frozenset({"succeeded", "rejected", "failed", "cancelled"}),
    "phase": frozenset({"temporary", "canonical", "minio", "qdrant"}),
    "provider_type": frozenset({"openai_compatible", "openrouter", "vendor_specific"}),
    "stage": frozenset({"parse", "chunk", "embed_index", "validate", "activate"}),
    "state": frozenset({"queued", "running", "retry_wait", "succeeded", "failed", "cancelled"}),
    "status": frozenset(
        {
            "queued",
            "running",
            "retry_wait",
            "succeeded",
            "rejected",
            "failed",
            "rate_limited",
            "timeout",
            "cancelled",
        }
    ),
}
_BOOLEAN_KEYS = frozenset({"degraded", "retryable", "recovered"})
_INTEGER_KEYS = frozenset(
    {
        "attempt",
        "batch_count",
        "byte_count",
        "candidate_count",
        "character_count",
        "chunk_count",
        "cost_micros",
        "count",
        "input_tokens",
        "latency_ms",
        "output_tokens",
        "point_count",
        "result_count",
        "retry_count",
        "visibility_drop_count",
    }
)
_FLOAT_KEYS = frozenset({"duration_seconds", "latency_seconds"})
_NESTED_KEYS = frozenset({"context", "details", "metrics"})
_SAFE_EXTRA_KEYS = tuple(
    sorted(
        _CONTEXT_ID_KEYS
        | _ENUM_VALUES.keys()
        | _BOOLEAN_KEYS
        | _INTEGER_KEYS
        | _FLOAT_KEYS
        | _NESTED_KEYS
        | {"event", "error_code", "exception_type", "exception_types"}
    )
)


def _valid_request_id(value: object) -> bool:
    return (
        type(value) is str
        and 0 < len(value) <= _MAX_REQUEST_ID_LENGTH
        and _REQUEST_ID_PATTERN.fullmatch(value) is not None
    )


def _normalized_uuid(value: object) -> str | None:
    if type(value) is UUID:
        return str(value)
    if type(value) is not str or len(value) != 36:
        return None
    try:
        parsed = UUID(value)
    except (ValueError, AttributeError):
        return None
    canonical = str(parsed)
    return canonical if value == canonical else None


@dataclass(frozen=True, slots=True, repr=False)
class SafeLogContext:
    """Bounded application-owned identifiers that may be placed in a log record."""

    request_id: str | None = None
    knowledge_base_id: UUID | str | None = None
    document_id: UUID | str | None = None
    version_id: UUID | str | None = None
    job_id: UUID | str | None = None
    generation_id: UUID | str | None = None

    def __post_init__(self) -> None:
        if self.request_id is not None and not _valid_request_id(self.request_id):
            raise ValueError("log context is invalid")
        for key in _UUID_CONTEXT_KEYS:
            if (value := getattr(self, key)) is not None and _normalized_uuid(value) is None:
                raise ValueError("log context is invalid")

    def __repr__(self) -> str:
        return "SafeLogContext(<bounded-identifiers>)"

    def as_extra(self) -> dict[str, str]:
        extra: dict[str, str] = {}
        if self.request_id is not None:
            extra["request_id"] = self.request_id
        for key in (
            "knowledge_base_id",
            "document_id",
            "version_id",
            "job_id",
            "generation_id",
        ):
            value = getattr(self, key)
            if value is not None:
                normalized = _normalized_uuid(value)
                if normalized is None:
                    raise ValueError("log context is invalid")
                extra[key] = normalized
        return extra


def _safe_exception_type(error: BaseException) -> str:
    error_type = type(error)
    for allowed_type, name in _BUILTIN_EXCEPTION_TYPES:
        if error_type is allowed_type:
            return name
    return "Exception"


def _exception_children(error: BaseException) -> tuple[BaseException, ...]:
    if not isinstance(error, BaseExceptionGroup):
        return ()
    try:
        children = BaseExceptionGroup.exceptions.__get__(error, BaseExceptionGroup)
    except BaseException:
        return ()
    return children if type(children) is tuple else ()


def _exception_types(exc_info: object) -> tuple[str, ...]:
    if type(exc_info) is not tuple or len(exc_info) != 3:
        return ()
    root = exc_info[1]
    if not isinstance(root, BaseException):
        return ()
    collected: list[str] = []
    pending = [root]
    visited: set[int] = set()
    while pending and len(collected) < _MAX_EXCEPTION_TYPES:
        current = pending.pop()
        identity = id(current)
        if identity in visited:
            continue
        visited.add(identity)
        collected.append(_safe_exception_type(current))
        children = _exception_children(current)
        remaining = _MAX_EXCEPTION_TYPES - len(collected)
        child_count = min(len(children), remaining)
        for index in range(child_count - 1, -1, -1):
            child = children[index]
            if isinstance(child, BaseException):
                pending.append(child)
    return tuple(collected)


def _sanitize_exception_name(value: object) -> str:
    if type(value) is str:
        for _, name in _BUILTIN_EXCEPTION_TYPES:
            if value == name:
                return name
    return "Exception"


def _sanitize_exception_names(value: object) -> tuple[str, ...] | None:
    if type(value) not in {list, tuple}:
        return None
    members = cast(list[object] | tuple[object, ...], value)
    count = min(len(members), _MAX_EXCEPTION_TYPES)
    return tuple(_sanitize_exception_name(members[index]) for index in range(count))


def _sanitize_mapping(value: object, *, depth: int) -> dict[str, object] | None:
    if type(value) is not dict or depth > _MAX_SANITIZE_DEPTH:
        return None
    source = cast(dict[object, object], value)
    sanitized: dict[str, object] = {}
    for key in _SAFE_EXTRA_KEYS:
        member = source.get(key, _MISSING)
        if member is _MISSING:
            continue
        safe_value = _sanitize_member(key, member, depth=depth)
        if safe_value is not None:
            sanitized[key] = safe_value
    return sanitized


def _sanitize_member(key: str, value: object, *, depth: int) -> object | None:
    if depth > _MAX_SANITIZE_DEPTH:
        return None
    if key == "request_id":
        return value if _valid_request_id(value) else None
    if key in _UUID_CONTEXT_KEYS:
        return _normalized_uuid(value)
    if key == "event":
        return value if type(value) is str and value in _EVENT_NAMES else "redacted_log_event"
    if key == "error_code":
        return value if type(value) is str and value in _ERROR_CODES else "OTHER"
    if key == "exception_type":
        return _sanitize_exception_name(value)
    if key == "exception_types":
        return _sanitize_exception_names(value)
    if allowed := _ENUM_VALUES.get(key):
        return value if type(value) is str and value in allowed else None
    if key in _BOOLEAN_KEYS:
        return value if type(value) is bool else None
    if key in _INTEGER_KEYS:
        return value if type(value) is int and 0 <= value <= _MAX_COUNTER_VALUE else None
    if key in _FLOAT_KEYS:
        if type(value) not in {int, float}:
            return None
        numeric = float(cast(int | float, value))
        return numeric if math.isfinite(numeric) and numeric >= 0 else None
    if key in _NESTED_KEYS:
        return _sanitize_mapping(value, depth=depth + 1)
    return None


def _safe_record_values(message: object, level: object) -> dict[str, object]:
    safe_message = (
        message if type(message) is str and message in _EVENT_NAMES else "redacted_log_event"
    )
    safe_level = level if type(level) is int and level in {10, 20, 30, 40, 50} else logging.INFO
    created = time.time()
    return {
        "name": "rag_service",
        "msg": safe_message,
        "args": (),
        "levelname": logging.getLevelName(safe_level),
        "levelno": safe_level,
        "pathname": "redacted",
        "filename": "redacted",
        "module": "redacted",
        "exc_info": None,
        "exc_text": None,
        "stack_info": None,
        "lineno": 0,
        "funcName": "redacted",
        "created": created,
        "msecs": (created - int(created)) * 1000,
        "relativeCreated": 0.0,
        "thread": 0,
        "threadName": "worker",
        "processName": "service",
        "process": 0,
        "taskName": "task",
        "message": safe_message,
    }


def _new_safe_record(values: dict[str, object]) -> logging.LogRecord:
    replacement = logging.LogRecord(
        name="rag_service",
        level=logging.INFO,
        pathname="redacted",
        lineno=0,
        msg="redacted_log_event",
        args=(),
        exc_info=None,
        func="redacted",
        sinfo=None,
    )
    object.__setattr__(replacement, "__dict__", values)
    return replacement


def sanitize_log_record(record: logging.LogRecord) -> logging.LogRecord:
    """Return a new exact ``LogRecord`` without rendering input-owned values."""

    safe_base = _safe_record_values(None, logging.INFO)
    if type(record) is not logging.LogRecord:
        return _new_safe_record(safe_base)
    data = object.__getattribute__(record, "__dict__")
    if type(data) is not dict:
        return _new_safe_record(safe_base)
    source = cast(dict[str, object], data)
    raw_message: object = None
    raw_level: object = logging.INFO
    raw_exc_info: object = None
    raw_extra: dict[str, object] = {}
    try:
        raw_message = source.get("msg")
        raw_level = source.get("levelno")
        raw_exc_info = source.get("exc_info")
        for key in _SAFE_EXTRA_KEYS:
            value = source.get(key, _MISSING)
            if value is not _MISSING:
                raw_extra[key] = value
    except BaseException:
        return _new_safe_record(safe_base)

    safe_values = _safe_record_values(raw_message, raw_level)
    try:
        for key, value in raw_extra.items():
            safe_value = _sanitize_member(key, value, depth=0)
            if safe_value is not None:
                safe_values[key] = safe_value
        derived_exception_types = _exception_types(raw_exc_info)
        if derived_exception_types:
            safe_values["exception_types"] = derived_exception_types
    except BaseException:
        return _new_safe_record(safe_base)
    return _new_safe_record(safe_values)


def emit_safe_log(
    logger: logging.Logger,
    level: int,
    event: str,
    *,
    context: SafeLogContext | None = None,
    **fields: object,
) -> None:
    """Fail open after delivering only a fresh sanitized record to handlers."""

    try:
        if not logger.isEnabledFor(level):
            return
        record = logging.LogRecord(
            name="rag_service",
            level=level,
            pathname="redacted",
            lineno=0,
            msg=event,
            args=(),
            exc_info=None,
            func="redacted",
            sinfo=None,
        )
        data = object.__getattribute__(record, "__dict__")
        data["event"] = event
        if context is not None:
            data.update(context.as_extra())
        data.update(fields)
        logger.handle(sanitize_log_record(record))
    except BaseException:
        return


class SanitizingLogFilter(logging.Filter):
    """Logging filter that replaces records via :func:`sanitize_log_record`."""

    def filter(self, record: logging.LogRecord) -> bool | logging.LogRecord:
        return sanitize_log_record(record)


__all__ = [
    "SafeLogContext",
    "SanitizingLogFilter",
    "emit_safe_log",
    "sanitize_log_record",
]
