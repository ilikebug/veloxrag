"""Async, bounded object-store boundary over the synchronous MinIO SDK."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import math
import re
import tempfile
import threading
from collections import deque
from collections.abc import AsyncIterable, AsyncIterator, Callable, Iterable, Mapping
from concurrent.futures import Future as ConcurrentFuture
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import BinaryIO, Final, Protocol, cast
from uuid import uuid4
from xml.etree.ElementTree import Element, SubElement, tostring

import minio as minio
from minio.commonconfig import CopySource
from minio.helpers import md5sum_hash

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_NOT_FOUND_CODES: Final = frozenset(
    {"NoSuchKey", "NoSuchObject", "NotFound", "XMinioInvalidObjectName"}
)
_PRECONDITION_CODES: Final = frozenset({"PreconditionFailed", "ConditionalRequestConflict"})
_SUPPORTED_MINIO_VERSION: Final = "7.2.20"
_POSITIONAL_OR_KEYWORD: Final = inspect.Parameter.POSITIONAL_OR_KEYWORD
_EMPTY: Final = inspect.Parameter.empty
_PRIVATE_API_SIGNATURES: Final = {
    "_create_multipart_upload": (
        ("bucket_name", _POSITIONAL_OR_KEYWORD, _EMPTY),
        ("object_name", _POSITIONAL_OR_KEYWORD, _EMPTY),
        ("headers", _POSITIONAL_OR_KEYWORD, _EMPTY),
    ),
    "_upload_part_copy": (
        ("bucket_name", _POSITIONAL_OR_KEYWORD, _EMPTY),
        ("object_name", _POSITIONAL_OR_KEYWORD, _EMPTY),
        ("upload_id", _POSITIONAL_OR_KEYWORD, _EMPTY),
        ("part_number", _POSITIONAL_OR_KEYWORD, _EMPTY),
        ("headers", _POSITIONAL_OR_KEYWORD, _EMPTY),
    ),
    "_abort_multipart_upload": (
        ("bucket_name", _POSITIONAL_OR_KEYWORD, _EMPTY),
        ("object_name", _POSITIONAL_OR_KEYWORD, _EMPTY),
        ("upload_id", _POSITIONAL_OR_KEYWORD, _EMPTY),
    ),
    "_execute": (
        ("method", _POSITIONAL_OR_KEYWORD, _EMPTY),
        ("bucket_name", _POSITIONAL_OR_KEYWORD, None),
        ("object_name", _POSITIONAL_OR_KEYWORD, None),
        ("body", _POSITIONAL_OR_KEYWORD, None),
        ("headers", _POSITIONAL_OR_KEYWORD, None),
        ("query_params", _POSITIONAL_OR_KEYWORD, None),
        ("preload_content", _POSITIONAL_OR_KEYWORD, True),
        ("no_body_trace", _POSITIONAL_OR_KEYWORD, False),
    ),
}


class ObjectStat(Protocol):
    object_name: str
    size: int
    metadata: Mapping[str, str]
    last_modified: datetime
    etag: str
    content_type: str


class ObjectWriteResult(Protocol):
    @property
    def etag(self) -> str: ...


class ObjectReadResponse(Protocol):
    def stream(self, amount: int) -> Iterable[bytes]: ...

    def close(self) -> None: ...

    def release_conn(self) -> None: ...


type _MinioHeaderValue = str | list[str] | tuple[str]
type _MinioHeaders = dict[str, _MinioHeaderValue]


class MinioClient(Protocol):
    def put_object(
        self,
        bucket_name: str,
        object_name: str,
        data: BinaryIO,
        length: int,
        content_type: str = "application/octet-stream",
        metadata: dict[str, str] | None = None,
        *,
        part_size: int = 0,
        num_parallel_uploads: int = 3,
    ) -> ObjectWriteResult: ...

    def stat_object(self, bucket_name: str, object_name: str) -> ObjectStat: ...

    def get_object(self, bucket_name: str, object_name: str) -> ObjectReadResponse: ...

    def remove_object(self, bucket_name: str, object_name: str) -> None: ...

    def list_objects(
        self,
        bucket_name: str,
        *,
        prefix: str | None = None,
        recursive: bool = False,
        start_after: str | None = None,
        include_user_meta: bool = False,
    ) -> Iterable[ObjectStat]: ...

    def _create_multipart_upload(
        self,
        bucket_name: str,
        object_name: str,
        headers: _MinioHeaders,
    ) -> str: ...

    def _upload_part_copy(
        self,
        bucket_name: str,
        object_name: str,
        upload_id: str,
        part_number: int,
        headers: _MinioHeaders,
    ) -> tuple[str, datetime | None]: ...

    def _abort_multipart_upload(
        self,
        bucket_name: str,
        object_name: str,
        upload_id: str,
    ) -> object: ...

    def _execute(
        self,
        method: str,
        bucket_name: str | None = None,
        object_name: str | None = None,
        body: bytes | None = None,
        headers: _MinioHeaders | None = None,
        query_params: _MinioHeaders | None = None,
        preload_content: bool = True,
        no_body_trace: bool = False,
    ) -> object: ...


_ERROR_MESSAGES: Final = {
    "FILE_TOO_LARGE": "Document exceeds the upload limit",
    "UPLOAD_STREAM_FAILED": "Upload stream failed",
    "OBJECT_STORE_UNAVAILABLE": "Object store unavailable",
    "OBJECT_STORE_API_INCOMPATIBLE": "Object store client is incompatible",
    "OBJECT_VERIFICATION_FAILED": "Object verification failed",
    "ARTIFACT_CHECKSUM_CONFLICT": "Artifact checksum conflict",
}


class ObjectStoreError(Exception):
    """Sanitized object-store failure that never includes bucket or object names."""

    __slots__ = ("code", "retryable")

    def __init__(self, code: str, *, retryable: bool = True) -> None:
        if type(code) is not str or code not in _ERROR_MESSAGES:
            raise ValueError("object store error code is invalid")
        self.code = code
        self.retryable = retryable
        super().__init__(_ERROR_MESSAGES[code])

    def __repr__(self) -> str:
        return f"{type(self).__name__}(code={self.code!r}, retryable={self.retryable!r})"


class UploadLimitExceeded(ObjectStoreError):
    def __init__(self) -> None:
        super().__init__("FILE_TOO_LARGE", retryable=False)


class ArtifactChecksumConflict(ObjectStoreError):
    def __init__(self) -> None:
        super().__init__("ARTIFACT_CHECKSUM_CONFLICT", retryable=False)


class _UploadStreamFailure(Exception):
    pass


class _UploadCancelled(Exception):
    pass


class _ConsumerStopped(Exception):
    pass


class _DestinationAlreadyExists(Exception):
    code = "PreconditionFailed"
    status = 412


class _SourceVersionChanged(Exception):
    pass


class _TemporaryDownloadAbandoned(Exception):
    pass


class _TemporaryDownloadSlot:
    """Transfer one tempfile across threads or close it if the caller left."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._temporary: BinaryIO | None = None
        self._abandoned = False

    def publish(self, temporary: BinaryIO) -> BinaryIO:
        with self._lock:
            if self._abandoned:
                temporary.close()
                raise _TemporaryDownloadAbandoned
            self._temporary = temporary
            return temporary

    def abandon(self) -> None:
        temporary: BinaryIO | None = None
        with self._lock:
            self._abandoned = True
            temporary = self._temporary
            self._temporary = None
        if temporary is not None:
            temporary.close()


@dataclass(frozen=True, slots=True, repr=False)
class StoredObject:
    """Internal storage receipt; its opaque key must not cross public API/log boundaries."""

    object_key: str
    size: int
    checksum_sha256: str
    max_pipe_buffered_bytes: int
    producer_slice_bytes: int
    adapter_memory_bound_bytes: int


@dataclass(frozen=True, slots=True, repr=False)
class PublishedObject:
    """Internal immutable-publish receipt with a deliberately redacted representation."""

    object_key: str
    size: int
    checksum_sha256: str
    reused: bool


@dataclass(frozen=True, slots=True, repr=False)
class OrphanCandidate:
    """Internal sweeper candidate; object identity is deliberately absent from repr."""

    object_key: str
    size: int
    checksum_sha256: str | None
    last_modified: datetime


@dataclass(frozen=True, slots=True, repr=False)
class OrphanPage:
    items: tuple[OrphanCandidate, ...]
    next_start_after: str | None


class _BoundedBytePipe:
    """A byte-capacity-bounded synchronous reader fed by one async producer."""

    def __init__(self, capacity: int, loop: asyncio.AbstractEventLoop) -> None:
        self._capacity = capacity
        self._loop = loop
        self._condition = threading.Condition()
        self._chunks: deque[bytes] = deque()
        self._buffered_bytes = 0
        self._max_buffered_bytes = 0
        self._eof = False
        self._failure: Exception | None = None
        self._consumer_stopped = False
        self._space_available = asyncio.Event()
        self._space_available.set()

    @property
    def max_buffered_bytes(self) -> int:
        with self._condition:
            return self._max_buffered_bytes

    def _wake_writer(self) -> None:
        with suppress(RuntimeError):
            self._loop.call_soon_threadsafe(self._space_available.set)

    async def write(self, data: bytes) -> None:
        position = 0
        while position < len(data):
            with self._condition:
                if self._failure is not None:
                    raise self._failure
                if self._consumer_stopped:
                    raise _ConsumerStopped
                available = self._capacity - self._buffered_bytes
                if available > 0:
                    end = min(len(data), position + available)
                    piece = data[position:end]
                    self._chunks.append(piece)
                    self._buffered_bytes += len(piece)
                    self._max_buffered_bytes = max(
                        self._max_buffered_bytes,
                        self._buffered_bytes,
                    )
                    position = end
                    if self._buffered_bytes >= self._capacity:
                        self._space_available.clear()
                    self._condition.notify_all()
                    continue
                self._space_available.clear()
            await self._space_available.wait()

    def read(self, size: int = -1) -> bytes:
        if type(size) is not int:
            raise TypeError("read size must be an integer")
        with self._condition:
            while not self._chunks and not self._eof and self._failure is None:
                if self._consumer_stopped:
                    return b""
                self._condition.wait()
            if self._chunks:
                chunk = self._chunks.popleft()
                if 0 <= size < len(chunk):
                    result = chunk[:size]
                    remainder = chunk[size:]
                    self._chunks.appendleft(remainder)
                else:
                    result = chunk
                self._buffered_bytes -= len(result)
                self._condition.notify_all()
                self._wake_writer()
                return result
            if self._failure is not None:
                raise self._failure
            return b""

    def close(self) -> None:
        with self._condition:
            self._eof = True
            self._condition.notify_all()

    def fail(self, error: Exception, *, discard: bool = False) -> None:
        with self._condition:
            if discard:
                self._chunks.clear()
                self._buffered_bytes = 0
            if self._failure is None:
                self._failure = error
            self._condition.notify_all()
        self._wake_writer()

    def stop_consumer(self) -> None:
        with self._condition:
            self._consumer_stopped = True
            self._chunks.clear()
            self._buffered_bytes = 0
            self._condition.notify_all()
        self._wake_writer()


class _MinioIoExecutor:
    """Dedicated bounded executor whose capacity follows real SDK completion."""

    __slots__ = (
        "_active",
        "_closed",
        "_condition",
        "_executor",
        "_max_workers",
        "_shutdown",
        "_thread_name_prefix",
    )

    def __init__(self, *, max_workers: int) -> None:
        self._max_workers = max_workers
        self._thread_name_prefix = f"rag-minio-io-{id(self):x}"
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix=self._thread_name_prefix,
        )
        self._condition = asyncio.Condition()
        self._active = 0
        self._closed = False
        self._shutdown = False

    @property
    def active_count(self) -> int:
        return self._active

    @property
    def thread_name_prefix(self) -> str:
        return self._thread_name_prefix

    @staticmethod
    def _consume_release_task(task: asyncio.Task[None]) -> None:
        if not task.cancelled():
            task.exception()

    async def _release_capacity(self) -> None:
        async with self._condition:
            if self._active <= 0:
                raise AssertionError("MinIO executor capacity state is invalid")
            self._active -= 1
            self._condition.notify_all()

    def _completed_on_loop(self) -> None:
        task = asyncio.create_task(
            self._release_capacity(),
            name="minio-executor-capacity-release",
        )
        task.add_done_callback(self._consume_release_task)

    async def submit(
        self,
        callback: Callable[..., object],
        *args: object,
        deadline: float,
    ) -> asyncio.Future[object]:
        loop = asyncio.get_running_loop()
        async with self._condition:
            while self._active >= self._max_workers:
                if self._closed:
                    raise RuntimeError("MinIO executor is closed")
                remaining = deadline - loop.time()
                if remaining <= 0:
                    raise TimeoutError
                async with asyncio.timeout(remaining):
                    await self._condition.wait()
            if self._closed:
                raise RuntimeError("MinIO executor is closed")
            self._active += 1

        future: ConcurrentFuture[object]
        try:
            future = self._executor.submit(callback, *args)
        except BaseException:
            await self._release_capacity()
            raise

        def release_when_done(_future: ConcurrentFuture[object]) -> None:
            del _future
            with suppress(RuntimeError):
                loop.call_soon_threadsafe(self._completed_on_loop)

        future.add_done_callback(release_when_done)
        return asyncio.wrap_future(future)

    async def aclose(self, *, deadline: float) -> bool:
        loop = asyncio.get_running_loop()
        async with self._condition:
            self._closed = True
            self._condition.notify_all()
            while self._active > 0:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    break
                try:
                    async with asyncio.timeout(remaining):
                        await self._condition.wait()
                except TimeoutError:
                    break
            drained = self._active == 0
            first_shutdown = not self._shutdown
            self._shutdown = True
        if first_shutdown:
            self._executor.shutdown(wait=drained, cancel_futures=True)
        elif drained:
            self._executor.shutdown(wait=True, cancel_futures=True)
        return drained


def _valid_object_key(object_key: str) -> bool:
    if (
        type(object_key) is not str
        or not object_key
        or object_key.startswith("/")
        or object_key.endswith("/")
        or "\\" in object_key
        or "\x00" in object_key
        or any(ord(character) < 32 or ord(character) == 127 for character in object_key)
        or len(object_key.encode("utf-8")) > 1024
    ):
        return False
    return all(segment and segment not in {".", ".."} for segment in object_key.split("/"))


def _valid_checksum(checksum: str) -> bool:
    return type(checksum) is str and _SHA256_PATTERN.fullmatch(checksum) is not None


def _valid_content_type(content_type: object) -> bool:
    return (
        type(content_type) is str
        and bool(content_type)
        and "\r" not in content_type
        and "\n" not in content_type
    )


def _error_code(error: BaseException) -> str | None:
    code = getattr(error, "code", None)
    return code if type(code) is str else None


def _error_status(error: BaseException) -> int | None:
    status = getattr(error, "status", None)
    return status if type(status) is int else None


def _is_not_found(error: BaseException) -> bool:
    return _error_code(error) in _NOT_FOUND_CODES or _error_status(error) == 404


def _is_precondition_failure(error: BaseException) -> bool:
    return _error_code(error) in _PRECONDITION_CODES or _error_status(error) in {409, 412}


def _metadata_checksum(metadata: object) -> str | None:
    if not isinstance(metadata, Mapping):
        return None
    normalized = {
        str(key).lower(): value
        for key, value in metadata.items()
        if isinstance(key, str) and isinstance(value, str)
    }
    checksum = normalized.get("x-amz-meta-sha256", normalized.get("sha256"))
    return checksum if checksum is not None and _valid_checksum(checksum) else None


def _object_etag(value: object) -> str | None:
    etag = getattr(value, "etag", None)
    if type(etag) is not str:
        return None
    normalized = etag.strip().strip('"')
    if (
        not normalized
        or len(normalized) > 256
        or any(ord(character) < 32 or ord(character) == 127 for character in normalized)
    ):
        return None
    return normalized


def _object_content_type(value: object) -> str | None:
    content_type = getattr(value, "content_type", None)
    return cast(str, content_type) if _valid_content_type(content_type) else None


def _signature_matches(
    callback: object,
    expected: tuple[tuple[str, inspect._ParameterKind, object], ...],
) -> bool:
    try:
        parameters = tuple(
            inspect.signature(cast(Callable[..., object], callback)).parameters.values()
        )
    except (TypeError, ValueError):
        return False
    if len(parameters) != len(expected):
        return False
    return all(
        parameter.name == name and parameter.kind is kind and parameter.default == default
        for parameter, (name, kind, default) in zip(parameters, expected, strict=True)
    )


def _complete_multipart_body(parts: list[tuple[int, str]]) -> bytes:
    root = Element("CompleteMultipartUpload")
    for part_number, etag in parts:
        part = SubElement(root, "Part")
        SubElement(part, "PartNumber").text = str(part_number)
        SubElement(part, "ETag").text = f'"{etag}"'
    return cast(bytes, tostring(root, encoding="utf-8"))


def _consume_task_result(task: asyncio.Task[object]) -> None:
    with suppress(BaseException):
        task.result()


class MinioObjectStore:
    """Keep synchronous SDK I/O off-loop and publish immutable verified objects."""

    def __init__(
        self,
        *,
        client: MinioClient,
        bucket: str,
        buffer_bytes: int,
        part_size_bytes: int,
        producer_slice_bytes: int | None = None,
        operation_timeout_seconds: float = 300.0,
        cancel_grace_seconds: float = 0.05,
        io_max_workers: int = 8,
    ) -> None:
        resolved_slice_bytes = (
            min(buffer_bytes, 64 * 1024) if producer_slice_bytes is None else producer_slice_bytes
        )
        if (
            not callable(getattr(client, "put_object", None))
            or not callable(getattr(client, "stat_object", None))
            or not callable(getattr(client, "get_object", None))
            or not callable(getattr(client, "remove_object", None))
            or not callable(getattr(client, "list_objects", None))
            or type(bucket) is not str
            or not bucket
            or len(bucket) > 255
            or type(buffer_bytes) is not int
            or buffer_bytes <= 0
            or type(part_size_bytes) is not int
            or part_size_bytes < 5 * 1024 * 1024
            or type(resolved_slice_bytes) is not int
            or resolved_slice_bytes <= 0
            or type(operation_timeout_seconds) not in {int, float}
            or isinstance(operation_timeout_seconds, bool)
            or not math.isfinite(float(operation_timeout_seconds))
            or not 0 < float(operation_timeout_seconds) <= 600
            or type(cancel_grace_seconds) not in {int, float}
            or isinstance(cancel_grace_seconds, bool)
            or not math.isfinite(float(cancel_grace_seconds))
            or not 0 < float(cancel_grace_seconds) <= 5
            or type(io_max_workers) is not int
            or not 1 <= io_max_workers <= 64
        ):
            raise ValueError("object store configuration is invalid")
        self._client = client
        self._bucket = bucket
        self._buffer_bytes = buffer_bytes
        self._part_size_bytes = part_size_bytes
        self._producer_slice_bytes = resolved_slice_bytes
        self._operation_timeout_seconds = float(operation_timeout_seconds)
        self._cancel_grace_seconds = float(cancel_grace_seconds)
        self._io_executor = _MinioIoExecutor(max_workers=io_max_workers)
        self._lifecycle_lock = asyncio.Lock()
        self._active_operations: set[asyncio.Task[object]] = set()
        self._close_cancelled_tasks: set[asyncio.Task[object]] = set()
        self._blocking_operations: set[asyncio.Task[object]] = set()
        self._background_tasks: set[asyncio.Task[object]] = set()
        self._eventual_cleanup_tokens: set[tuple[int, str, str | None]] = set()
        self._closed = False
        self._close_failed = False

    @property
    def pending_operation_count(self) -> int:
        """Return live tracked SDK task wrappers retained for shutdown/diagnostics."""

        return sum(not task.done() for task in self._blocking_operations)

    @property
    def adapter_memory_bound_bytes(self) -> int:
        """Conservative adapter-owned peak, excluding the upstream-owned input chunk."""

        sdk_aggregation_peak = 2 * (self._part_size_bytes + 1)
        return self._buffer_bytes + sdk_aggregation_peak + self._producer_slice_bytes

    @property
    def producer_slice_bytes(self) -> int:
        return self._producer_slice_bytes

    def _track_blocking(self, task: asyncio.Task[object]) -> asyncio.Task[object]:
        self._blocking_operations.add(task)

        def done(completed: asyncio.Task[object]) -> None:
            self._blocking_operations.discard(completed)
            _consume_task_result(completed)

        task.add_done_callback(done)
        return task

    def _track_background(self, task: asyncio.Task[object]) -> None:
        self._background_tasks.add(task)

        def done(completed: asyncio.Task[object]) -> None:
            self._background_tasks.discard(completed)
            _consume_task_result(completed)

        task.add_done_callback(done)

    async def _start_blocking(
        self,
        callback: Callable[..., object],
        *args: object,
        deadline: float,
    ) -> asyncio.Task[object]:
        try:
            future = await self._io_executor.submit(
                callback,
                *args,
                deadline=deadline,
            )
        except (RuntimeError, TimeoutError):
            raise ObjectStoreError("OBJECT_STORE_UNAVAILABLE") from None

        async def wait_for_result() -> object:
            return await future

        return self._track_blocking(
            asyncio.create_task(
                wait_for_result(),
                name="minio-sdk-operation",
            )
        )

    async def _admit_operation(self) -> asyncio.Task[object]:
        operation = cast(asyncio.Task[object], asyncio.current_task())
        async with self._lifecycle_lock:
            if self._closed:
                raise ObjectStoreError("OBJECT_STORE_UNAVAILABLE", retryable=False) from None
            self._active_operations.add(operation)
        return operation

    def _release_operation(self, operation: asyncio.Task[object]) -> None:
        self._active_operations.discard(operation)

    def _operation_cancelled_by_close(self) -> bool:
        current = asyncio.current_task()
        return (
            current is not None
            and current in self._close_cancelled_tasks
            and current.cancelling() > 0
        )

    def _cancel_for_close(self, task: asyncio.Task[object]) -> None:
        if task not in self._close_cancelled_tasks:
            self._close_cancelled_tasks.add(task)

            def done(completed: asyncio.Task[object]) -> None:
                self._close_cancelled_tasks.discard(completed)

            task.add_done_callback(done)
        task.cancel()

    def _live_lifecycle_tasks(self) -> set[asyncio.Task[object]]:
        return {
            task
            for task in (
                *self._active_operations,
                *self._blocking_operations,
                *self._background_tasks,
            )
            if not task.done()
        }

    async def _cancel_lifecycle_tasks(self) -> None:
        deadline = asyncio.get_running_loop().time() + self._cancel_grace_seconds
        while True:
            tasks = self._live_lifecycle_tasks()
            if not tasks:
                break
            for task in tasks:
                self._cancel_for_close(task)
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                break
            await asyncio.wait(tasks, timeout=remaining)
            await asyncio.sleep(0)

        for task in self._live_lifecycle_tasks():
            self._cancel_for_close(task)
        await asyncio.sleep(0)

    async def _retire_failed_close(self) -> None:
        self._closed = True
        self._close_failed = True
        await self._cancel_lifecycle_tasks()
        await self._io_executor.aclose(
            deadline=asyncio.get_running_loop().time() + self._cancel_grace_seconds
        )

    async def _wait_grace(self, task: asyncio.Task[object]) -> bool:
        if task.done():
            return True
        try:
            done, _pending = await asyncio.wait(
                {task},
                timeout=self._cancel_grace_seconds,
            )
        except asyncio.CancelledError:
            return task.done()
        return bool(done)

    async def _close_unstarted_stream(self, stream: AsyncIterable[bytes]) -> None:
        closer = getattr(stream, "aclose", None)
        if not callable(closer):
            return

        async def close() -> object:
            with suppress(BaseException):
                return await closer()
            return None

        task = cast(
            asyncio.Task[object],
            asyncio.create_task(close(), name="minio-upload-stream-close"),
        )
        if not await self._wait_grace(task):
            task.cancel()
            await self._wait_grace(task)
        if not task.done():
            self._track_background(task)

    def _schedule_eventual_cleanup(
        self,
        dependency: asyncio.Task[object],
        object_key: str,
        *,
        late_success_cleanup_key: str | None = None,
    ) -> None:
        if self._operation_cancelled_by_close():
            return
        token = (id(dependency), object_key, late_success_cleanup_key)
        if token in self._eventual_cleanup_tokens:
            return
        self._eventual_cleanup_tokens.add(token)

        async def finalize() -> None:
            try:
                completed_successfully = False
                try:
                    await asyncio.shield(dependency)
                except asyncio.CancelledError:
                    current = asyncio.current_task()
                    if current is not None and current.cancelling():
                        raise
                except BaseException:
                    pass
                else:
                    completed_successfully = True
                if completed_successfully and late_success_cleanup_key is not None:
                    await self._delete_best_effort(late_success_cleanup_key)
                await self._delete_best_effort(object_key)
            finally:
                self._eventual_cleanup_tokens.discard(token)

        self._track_background(
            asyncio.create_task(finalize(), name="minio-eventual-staging-cleanup")
        )

    async def _cleanup_staging(self, object_key: str) -> None:
        if self._operation_cancelled_by_close():
            return
        cleanup = cast(
            asyncio.Task[object],
            asyncio.create_task(
                self._delete_best_effort(object_key),
                name="minio-staging-cleanup",
            ),
        )
        self._track_background(cleanup)
        current = asyncio.current_task()
        if current is not None and current.cancelling():
            return
        await asyncio.shield(cleanup)

    async def _call_blocking(
        self,
        callback: Callable[..., object],
        *args: object,
        eventual_cleanup_key: str | None = None,
        late_success_cleanup_key: str | None = None,
    ) -> object:
        deadline = asyncio.get_running_loop().time() + self._operation_timeout_seconds
        task = await self._start_blocking(callback, *args, deadline=deadline)
        try:
            done, _pending = await asyncio.wait(
                {task},
                timeout=max(0.0, deadline - asyncio.get_running_loop().time()),
            )
        except asyncio.CancelledError:
            await self._wait_grace(task)
            if eventual_cleanup_key is not None:
                self._schedule_eventual_cleanup(
                    task,
                    eventual_cleanup_key,
                    late_success_cleanup_key=late_success_cleanup_key,
                )
            raise
        if not done:
            if eventual_cleanup_key is not None:
                self._schedule_eventual_cleanup(
                    task,
                    eventual_cleanup_key,
                    late_success_cleanup_key=late_success_cleanup_key,
                )
            raise ObjectStoreError("OBJECT_STORE_UNAVAILABLE") from None
        return task.result()

    async def aclose(self) -> None:
        """Close admission and drain public operations plus their tracked task wrappers."""

        deadline = asyncio.get_running_loop().time() + self._operation_timeout_seconds
        try:
            async with self._lifecycle_lock:
                self._closed = True
                close_failed = self._close_failed
            if close_failed:
                await self._retire_failed_close()
                raise ObjectStoreError("OBJECT_STORE_UNAVAILABLE") from None
            while True:
                tasks = self._live_lifecycle_tasks()
                if not tasks:
                    executor_drained = await self._io_executor.aclose(deadline=deadline)
                    if not executor_drained:
                        self._close_failed = True
                        raise ObjectStoreError("OBJECT_STORE_UNAVAILABLE") from None
                    if self._close_failed:
                        raise ObjectStoreError("OBJECT_STORE_UNAVAILABLE") from None
                    return
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    await self._retire_failed_close()
                    raise ObjectStoreError("OBJECT_STORE_UNAVAILABLE") from None
                await asyncio.wait(tasks, timeout=remaining)
        except asyncio.CancelledError:
            await self._retire_failed_close()
            raise

    async def upload_stream(
        self,
        object_key: str,
        stream: AsyncIterable[bytes],
        *,
        content_type: str,
        max_bytes: int,
    ) -> StoredObject:
        operation = await self._admit_operation()
        try:
            if (
                not _valid_object_key(object_key)
                or not hasattr(stream, "__aiter__")
                or not _valid_content_type(content_type)
                or type(max_bytes) is not int
                or max_bytes <= 0
            ):
                raise ValueError("upload request is invalid")
            staging_key = f"tmp/uploads/{uuid4().hex}"
            sdk_admitted = asyncio.Event()
            try:
                size, checksum, source_etag, max_buffered = await self._put_stream(
                    staging_key,
                    stream,
                    content_type=content_type,
                    max_bytes=max_bytes,
                    sdk_admitted=sdk_admitted,
                )
                source_stat = await self._verify_source_version(
                    staging_key,
                    expected_size=size,
                    expected_etag=source_etag,
                )
                source_content_type = _object_content_type(source_stat)
                if source_content_type != content_type:
                    raise ObjectStoreError("OBJECT_VERIFICATION_FAILED", retryable=False) from None
                published = await self._publish_verified_source(
                    staging_key,
                    object_key,
                    expected_size=size,
                    expected_checksum=checksum,
                    source_etag=source_etag,
                    content_type=source_content_type,
                )
                return StoredObject(
                    published.object_key,
                    published.size,
                    published.checksum_sha256,
                    max_buffered,
                    self._producer_slice_bytes,
                    self.adapter_memory_bound_bytes,
                )
            finally:
                if sdk_admitted.is_set():
                    await self._cleanup_staging(staging_key)
        finally:
            self._release_operation(operation)

    async def _put_stream(
        self,
        staging_key: str,
        stream: AsyncIterable[bytes],
        *,
        content_type: str,
        max_bytes: int,
        sdk_admitted: asyncio.Event,
    ) -> tuple[int, str, str, int]:
        deadline = asyncio.get_running_loop().time() + self._operation_timeout_seconds
        pipe = _BoundedBytePipe(self._buffer_bytes, asyncio.get_running_loop())
        try:
            consumer = await self._start_blocking(
                self._put_sync,
                staging_key,
                pipe,
                content_type,
                deadline=deadline,
            )
            sdk_admitted.set()
        except BaseException:
            pipe.fail(_UploadCancelled(), discard=True)
            await self._close_unstarted_stream(stream)
            raise
        producer = asyncio.create_task(
            self._produce_stream(pipe, stream, max_bytes=max_bytes),
            name="minio-upload-producer",
        )
        try:
            try:
                done, _pending = await asyncio.wait(
                    {consumer},
                    timeout=max(0.0, deadline - asyncio.get_running_loop().time()),
                )
            except asyncio.CancelledError as cancellation:
                pipe.fail(_UploadCancelled(), discard=True)
                producer.cancel()
                await self._wait_grace(cast(asyncio.Task[object], producer))
                if not await self._wait_grace(consumer):
                    self._schedule_eventual_cleanup(consumer, staging_key)
                raise cancellation
            if not done:
                pipe.fail(_UploadCancelled(), discard=True)
                producer.cancel()
                await self._wait_grace(cast(asyncio.Task[object], producer))
                self._schedule_eventual_cleanup(consumer, staging_key)
                raise ObjectStoreError("OBJECT_STORE_UNAVAILABLE") from None

            try:
                consumer_result = consumer.result()
                consumer_error: BaseException | None = None
            except BaseException as error:
                consumer_result = None
                consumer_error = error
            if consumer_error is not None:
                pipe.stop_consumer()
                if not producer.done():
                    producer.cancel()
                await self._wait_grace(cast(asyncio.Task[object], producer))
                try:
                    producer_result = producer.result()
                    producer_error: BaseException | None = None
                except BaseException as error:
                    producer_result = None
                    producer_error = error
                if isinstance(consumer_error, UploadLimitExceeded) or isinstance(
                    producer_error,
                    UploadLimitExceeded,
                ):
                    raise UploadLimitExceeded from None
                if isinstance(consumer_error, _UploadStreamFailure) or (
                    producer_error is not None
                    and not isinstance(producer_error, (asyncio.CancelledError, _ConsumerStopped))
                ):
                    raise ObjectStoreError("UPLOAD_STREAM_FAILED", retryable=False) from None
                raise ObjectStoreError("OBJECT_STORE_UNAVAILABLE") from None

            if not producer.done():
                try:
                    await asyncio.wait(
                        {producer},
                        timeout=max(0.0, deadline - asyncio.get_running_loop().time()),
                    )
                except asyncio.CancelledError as cancellation:
                    pipe.fail(_UploadCancelled(), discard=True)
                    producer.cancel()
                    await self._wait_grace(cast(asyncio.Task[object], producer))
                    raise cancellation
            if not producer.done():
                pipe.stop_consumer()
                producer.cancel()
                await self._wait_grace(cast(asyncio.Task[object], producer))
                raise ObjectStoreError("OBJECT_VERIFICATION_FAILED", retryable=False) from None
            try:
                producer_result = producer.result()
                producer_error = None
            except BaseException as error:
                producer_result = None
                producer_error = error
            if isinstance(producer_error, UploadLimitExceeded):
                raise UploadLimitExceeded from None
            if producer_error is not None:
                raise ObjectStoreError("OBJECT_STORE_UNAVAILABLE") from None
            size, checksum = cast(tuple[int, str], producer_result)
            source_etag = _object_etag(consumer_result)
            if source_etag is None:
                raise ObjectStoreError("OBJECT_VERIFICATION_FAILED", retryable=False) from None
            return size, checksum, source_etag, pipe.max_buffered_bytes
        finally:
            pipe.stop_consumer()
            if not producer.done():
                producer.cancel()
            await self._wait_grace(cast(asyncio.Task[object], producer))
            if not producer.done():
                self._track_background(cast(asyncio.Task[object], producer))
            if not consumer.done():
                self._schedule_eventual_cleanup(consumer, staging_key)

    async def _produce_stream(
        self,
        pipe: _BoundedBytePipe,
        stream: AsyncIterable[bytes],
        *,
        max_bytes: int,
    ) -> tuple[int, str]:
        total = 0
        digest = hashlib.sha256()
        try:
            async for chunk in stream:
                if type(chunk) is not bytes:
                    raise _UploadStreamFailure
                if not chunk:
                    continue
                next_total = total + len(chunk)
                if next_total > max_bytes:
                    raise UploadLimitExceeded
                digest.update(chunk)
                total = next_total
                view = memoryview(chunk)
                for start in range(0, len(view), self._producer_slice_bytes):
                    piece = bytes(view[start : start + self._producer_slice_bytes])
                    await pipe.write(piece)
            pipe.close()
            return total, digest.hexdigest()
        except UploadLimitExceeded as error:
            pipe.fail(error)
            raise
        except asyncio.CancelledError:
            pipe.fail(_UploadCancelled(), discard=True)
            raise
        except BaseException:
            pipe.fail(_UploadStreamFailure())
            raise
        finally:
            closer = getattr(stream, "aclose", None)
            if callable(closer):
                with suppress(BaseException):
                    await closer()

    def _put_sync(
        self,
        staging_key: str,
        pipe: _BoundedBytePipe,
        content_type: str,
    ) -> ObjectWriteResult:
        try:
            return self._client.put_object(
                self._bucket,
                staging_key,
                cast(BinaryIO, pipe),
                -1,
                content_type=content_type,
                part_size=self._part_size_bytes,
                num_parallel_uploads=1,
            )
        finally:
            pipe.stop_consumer()

    async def _head(self, object_key: str) -> ObjectStat | None:
        try:
            return cast(
                ObjectStat,
                await self._call_blocking(
                    self._client.stat_object,
                    self._bucket,
                    object_key,
                ),
            )
        except BaseException as error:
            if isinstance(error, asyncio.CancelledError):
                raise
            if _is_not_found(error):
                return None
            raise ObjectStoreError("OBJECT_STORE_UNAVAILABLE") from None

    async def _verify_source_version(
        self,
        object_key: str,
        *,
        expected_size: int,
        expected_etag: str,
    ) -> ObjectStat:
        stat = await self._head(object_key)
        if (
            stat is None
            or type(stat.size) is not int
            or stat.size != expected_size
            or _object_etag(stat) != expected_etag
        ):
            raise ObjectStoreError("OBJECT_VERIFICATION_FAILED", retryable=False) from None
        return stat

    async def _verified_stat(
        self,
        object_key: str,
        *,
        expected_size: int,
        expected_checksum: str,
    ) -> ObjectStat:
        stat = await self._head(object_key)
        if (
            stat is None
            or type(stat.size) is not int
            or stat.size != expected_size
            or _metadata_checksum(stat.metadata) != expected_checksum
            or _object_etag(stat) is None
        ):
            raise ObjectStoreError("OBJECT_VERIFICATION_FAILED", retryable=False) from None
        return stat

    async def verify_object(
        self,
        object_key: str,
        *,
        expected_size: int,
        expected_checksum: str,
    ) -> PublishedObject:
        operation = await self._admit_operation()
        try:
            if (
                not _valid_object_key(object_key)
                or type(expected_size) is not int
                or expected_size < 0
                or not _valid_checksum(expected_checksum)
            ):
                raise ValueError("object verification request is invalid")
            return await self._verify_object(
                object_key,
                expected_size=expected_size,
                expected_checksum=expected_checksum,
            )
        finally:
            self._release_operation(operation)

    async def read_bytes(
        self,
        object_key: str,
        *,
        expected_checksum: str,
        max_bytes: int,
    ) -> bytes:
        """Read one verified object within a caller-owned hard byte limit."""

        operation = await self._admit_operation()
        try:
            if (
                not _valid_object_key(object_key)
                or not _valid_checksum(expected_checksum)
                or type(max_bytes) is not int
                or max_bytes <= 0
            ):
                raise ValueError("object read request is invalid")
            stat = await self._head(object_key)
            if (
                stat is None
                or type(stat.size) is not int
                or stat.size < 0
                or stat.size > max_bytes
                or _metadata_checksum(stat.metadata) != expected_checksum
                or _object_etag(stat) is None
            ):
                raise ObjectStoreError("OBJECT_VERIFICATION_FAILED", retryable=False) from None
            try:
                return cast(
                    bytes,
                    await self._call_blocking(
                        self._read_bytes_sync,
                        object_key,
                        stat.size,
                        expected_checksum,
                    ),
                )
            except (asyncio.CancelledError, ObjectStoreError):
                raise
            except BaseException:
                raise ObjectStoreError("OBJECT_STORE_UNAVAILABLE") from None
        finally:
            self._release_operation(operation)

    async def read_stream(
        self,
        object_key: str,
        *,
        expected_checksum: str,
        max_bytes: int,
    ) -> AsyncIterator[bytes]:
        """Yield a disk-spooled object only after its complete checksum is verified."""

        operation = await self._admit_operation()
        temporary: BinaryIO | None = None
        try:
            if (
                not _valid_object_key(object_key)
                or not _valid_checksum(expected_checksum)
                or type(max_bytes) is not int
                or max_bytes <= 0
            ):
                raise ValueError("object read request is invalid")
            stat = await self._head(object_key)
            if (
                stat is None
                or type(stat.size) is not int
                or stat.size < 0
                or stat.size > max_bytes
                or _metadata_checksum(stat.metadata) != expected_checksum
                or _object_etag(stat) is None
            ):
                raise ObjectStoreError("OBJECT_VERIFICATION_FAILED", retryable=False) from None
            try:
                temporary = await self._download_verified_temporary(
                    object_key,
                    stat.size,
                    expected_checksum,
                )
                while True:
                    # The verified spool is local and each read is hard-bounded.
                    # Keeping this operation synchronous removes any close/read
                    # ownership race when the async consumer is cancelled.
                    chunk = temporary.read(self._producer_slice_bytes)
                    if not chunk:
                        break
                    yield chunk
            except (asyncio.CancelledError, ObjectStoreError):
                raise
            except BaseException:
                raise ObjectStoreError("OBJECT_STORE_UNAVAILABLE") from None
        finally:
            if temporary is not None:
                with suppress(BaseException):
                    temporary.close()
            self._release_operation(operation)

    def _download_verified_temporary_sync(
        self,
        object_key: str,
        expected_size: int,
        expected_checksum: str,
        slot: _TemporaryDownloadSlot,
    ) -> BinaryIO:
        # Ownership escapes to the async reader only after download verification.
        temporary = tempfile.TemporaryFile(mode="w+b")  # noqa: SIM115
        try:
            self._download_verified_sync(
                object_key,
                expected_size,
                expected_checksum,
                temporary,
            )
            return slot.publish(temporary)
        except BaseException:
            temporary.close()
            raise

    async def _download_verified_temporary(
        self,
        object_key: str,
        expected_size: int,
        expected_checksum: str,
    ) -> BinaryIO:
        deadline = asyncio.get_running_loop().time() + self._operation_timeout_seconds
        slot = _TemporaryDownloadSlot()
        task = await self._start_blocking(
            self._download_verified_temporary_sync,
            object_key,
            expected_size,
            expected_checksum,
            slot,
            deadline=deadline,
        )
        try:
            done, _pending = await asyncio.wait(
                {task},
                timeout=max(0.0, deadline - asyncio.get_running_loop().time()),
            )
        except asyncio.CancelledError:
            slot.abandon()
            raise
        if not done:
            slot.abandon()
            raise ObjectStoreError("OBJECT_STORE_UNAVAILABLE") from None
        return cast(BinaryIO, task.result())

    def _download_verified_sync(
        self,
        object_key: str,
        expected_size: int,
        expected_checksum: str,
        destination: BinaryIO,
    ) -> None:
        response: ObjectReadResponse | None = None
        try:
            response = self._client.get_object(self._bucket, object_key)
            total = 0
            digest = hashlib.sha256()
            for chunk in response.stream(self._producer_slice_bytes):
                if type(chunk) is not bytes or not chunk:
                    if chunk:
                        raise ObjectStoreError(
                            "OBJECT_VERIFICATION_FAILED", retryable=False
                        ) from None
                    continue
                total += len(chunk)
                if total > expected_size:
                    raise ObjectStoreError("OBJECT_VERIFICATION_FAILED", retryable=False) from None
                destination.write(chunk)
                digest.update(chunk)
            if total != expected_size or digest.hexdigest() != expected_checksum:
                raise ObjectStoreError("OBJECT_VERIFICATION_FAILED", retryable=False) from None
            destination.seek(0)
        finally:
            if response is not None:
                with suppress(BaseException):
                    response.close()
                with suppress(BaseException):
                    response.release_conn()

    def _read_bytes_sync(
        self,
        object_key: str,
        expected_size: int,
        expected_checksum: str,
    ) -> bytes:
        response: ObjectReadResponse | None = None
        try:
            response = self._client.get_object(self._bucket, object_key)
            content = bytearray()
            digest = hashlib.sha256()
            for chunk in response.stream(self._producer_slice_bytes):
                if type(chunk) is not bytes or not chunk:
                    if chunk:
                        raise ObjectStoreError(
                            "OBJECT_VERIFICATION_FAILED", retryable=False
                        ) from None
                    continue
                if len(content) + len(chunk) > expected_size:
                    raise ObjectStoreError("OBJECT_VERIFICATION_FAILED", retryable=False) from None
                content.extend(chunk)
                digest.update(chunk)
            if len(content) != expected_size or digest.hexdigest() != expected_checksum:
                raise ObjectStoreError("OBJECT_VERIFICATION_FAILED", retryable=False) from None
            return bytes(content)
        finally:
            if response is not None:
                with suppress(BaseException):
                    response.close()
                with suppress(BaseException):
                    response.release_conn()

    async def _verify_object(
        self,
        object_key: str,
        *,
        expected_size: int,
        expected_checksum: str,
    ) -> PublishedObject:
        await self._verified_stat(
            object_key,
            expected_size=expected_size,
            expected_checksum=expected_checksum,
        )
        return PublishedObject(object_key, expected_size, expected_checksum, False)

    async def publish_temp(
        self,
        temp_key: str,
        final_key: str,
        *,
        expected_size: int,
        expected_checksum: str,
    ) -> PublishedObject:
        operation = await self._admit_operation()
        try:
            if (
                not _valid_object_key(temp_key)
                or not _valid_object_key(final_key)
                or type(expected_size) is not int
                or expected_size <= 0
                or not _valid_checksum(expected_checksum)
            ):
                raise ValueError("artifact publish request is invalid")
            try:
                source_stat = await self._verified_stat(
                    temp_key,
                    expected_size=expected_size,
                    expected_checksum=expected_checksum,
                )
                source_content_type = _object_content_type(source_stat)
                if source_content_type is None:
                    raise ObjectStoreError("OBJECT_VERIFICATION_FAILED", retryable=False) from None
                return await self._publish_verified_source(
                    temp_key,
                    final_key,
                    expected_size=expected_size,
                    expected_checksum=expected_checksum,
                    source_etag=cast(str, _object_etag(source_stat)),
                    content_type=source_content_type,
                )
            finally:
                await self._cleanup_staging(temp_key)
        finally:
            self._release_operation(operation)

    async def _publish_verified_source(
        self,
        source_key: str,
        final_key: str,
        *,
        expected_size: int,
        expected_checksum: str,
        source_etag: str,
        content_type: str,
    ) -> PublishedObject:
        existing = await self._head(final_key)
        if existing is not None:
            return self._reuse_or_conflict(
                final_key,
                existing,
                expected_size=expected_size,
                expected_checksum=expected_checksum,
            )
        try:
            await self._copy_if_absent(
                source_key,
                final_key,
                expected_size=expected_size,
                checksum=expected_checksum,
                source_etag=source_etag,
                content_type=content_type,
            )
        except asyncio.CancelledError:
            raise
        except _SourceVersionChanged:
            raise ObjectStoreError("OBJECT_VERIFICATION_FAILED", retryable=False) from None
        except _DestinationAlreadyExists:
            raced = await self._head(final_key)
            if raced is None:
                raise ObjectStoreError("OBJECT_STORE_UNAVAILABLE") from None
            return self._reuse_or_conflict(
                final_key,
                raced,
                expected_size=expected_size,
                expected_checksum=expected_checksum,
            )
        except BaseException as error:
            if isinstance(error, ObjectStoreError):
                raise error from None
            raise ObjectStoreError("OBJECT_STORE_UNAVAILABLE") from None
        verified = await self._verify_object(
            final_key,
            expected_size=expected_size,
            expected_checksum=expected_checksum,
        )
        return PublishedObject(
            verified.object_key,
            verified.size,
            verified.checksum_sha256,
            False,
        )

    def _reuse_or_conflict(
        self,
        final_key: str,
        stat: ObjectStat,
        *,
        expected_size: int,
        expected_checksum: str,
    ) -> PublishedObject:
        if stat.size != expected_size or _metadata_checksum(stat.metadata) != expected_checksum:
            raise ArtifactChecksumConflict from None
        return PublishedObject(final_key, expected_size, expected_checksum, True)

    async def _copy_if_absent(
        self,
        source_key: str,
        final_key: str,
        *,
        expected_size: int,
        checksum: str,
        source_etag: str,
        content_type: str,
    ) -> None:
        self._validate_private_minio_api()
        await self._call_blocking(
            self._copy_if_absent_sync,
            source_key,
            final_key,
            expected_size,
            checksum,
            source_etag,
            content_type,
            eventual_cleanup_key=source_key,
            late_success_cleanup_key=final_key,
        )

    def _copy_if_absent_sync(
        self,
        source_key: str,
        final_key: str,
        expected_size: int,
        checksum: str,
        source_etag: str,
        content_type: str,
    ) -> None:
        create = cast(Callable[..., str], self._client._create_multipart_upload)
        upload_part_copy = cast(
            Callable[..., tuple[str, datetime | None]],
            self._client._upload_part_copy,
        )
        abort = cast(Callable[..., object], self._client._abort_multipart_upload)
        execute = cast(Callable[..., object], self._client._execute)
        upload_id: str | None = None
        completed = False
        try:
            upload_id = create(
                self._bucket,
                final_key,
                {
                    "Content-Type": content_type,
                    "x-amz-meta-sha256": checksum,
                },
            )
            parts: list[tuple[int, str]] = []
            for part_number, start in enumerate(
                range(0, expected_size, self._part_size_bytes),
                start=1,
            ):
                end = min(expected_size - 1, start + self._part_size_bytes - 1)
                headers = CopySource(self._bucket, source_key).gen_copy_headers()
                headers["x-amz-copy-source-if-match"] = source_etag
                headers["x-amz-copy-source-range"] = f"bytes={start}-{end}"
                try:
                    etag, _last_modified = upload_part_copy(
                        self._bucket,
                        final_key,
                        upload_id,
                        part_number,
                        headers,
                    )
                except BaseException as error:
                    if _is_precondition_failure(error):
                        raise _SourceVersionChanged from None
                    raise
                parts.append((part_number, etag))
            body = _complete_multipart_body(parts)
            headers = {
                "Content-Type": "application/xml",
                "Content-MD5": cast(str, md5sum_hash(body)),
                "If-None-Match": "*",
            }
            try:
                execute(
                    "POST",
                    self._bucket,
                    final_key,
                    body=body,
                    headers=headers,
                    query_params={"uploadId": upload_id},
                )
            except BaseException as error:
                if _is_precondition_failure(error):
                    raise _DestinationAlreadyExists from None
                raise
            completed = True
        finally:
            if upload_id is not None and not completed:
                with suppress(BaseException):
                    abort(self._bucket, final_key, upload_id)

    def _validate_private_minio_api(self) -> None:
        if minio.__version__ != _SUPPORTED_MINIO_VERSION:
            raise ObjectStoreError("OBJECT_STORE_API_INCOMPATIBLE", retryable=False) from None
        for name, expected in _PRIVATE_API_SIGNATURES.items():
            callback = getattr(self._client, name, None)
            if not callable(callback) or not _signature_matches(callback, expected):
                raise ObjectStoreError(
                    "OBJECT_STORE_API_INCOMPATIBLE",
                    retryable=False,
                ) from None

    async def delete_best_effort(self, object_key: str) -> bool:
        operation = await self._admit_operation()
        try:
            return await self._delete_best_effort(object_key)
        finally:
            self._release_operation(operation)

    async def _delete_best_effort(self, object_key: str) -> bool:
        if not _valid_object_key(object_key):
            return False
        try:
            await self._call_blocking(
                self._client.remove_object,
                self._bucket,
                object_key,
            )
        except asyncio.CancelledError:
            raise
        except BaseException:
            return False
        return True

    async def list_older_than(
        self,
        *,
        prefix: str,
        older_than: datetime,
        limit: int,
        start_after: str | None = None,
    ) -> OrphanPage:
        operation = await self._admit_operation()
        try:
            if (
                type(prefix) is not str
                or not prefix
                or not _valid_object_key(prefix.rstrip("/"))
                or older_than.tzinfo is None
                or older_than.utcoffset() != UTC.utcoffset(older_than)
                or type(limit) is not int
                or not 1 <= limit <= 1000
                or (start_after is not None and not _valid_object_key(start_after))
            ):
                raise ValueError("orphan listing request is invalid")
            try:
                return cast(
                    OrphanPage,
                    await self._call_blocking(
                        self._list_older_than_sync,
                        prefix,
                        older_than,
                        limit,
                        start_after,
                    ),
                )
            except (asyncio.CancelledError, ObjectStoreError):
                raise
            except BaseException:
                raise ObjectStoreError("OBJECT_STORE_UNAVAILABLE") from None
        finally:
            self._release_operation(operation)

    def _list_older_than_sync(
        self,
        prefix: str,
        older_than: datetime,
        limit: int,
        start_after: str | None,
    ) -> OrphanPage:
        items: list[OrphanCandidate] = []
        has_more = False
        scanned = 0
        next_start_after: str | None = None
        last_scanned: str | None = None
        scan_limit = min(1000, max(limit + 1, limit * 8))
        objects = self._client.list_objects(
            self._bucket,
            prefix=prefix,
            recursive=True,
            start_after=start_after,
            include_user_meta=True,
        )
        for stat in objects:
            scanned += 1
            if scanned > scan_limit:
                has_more = True
                next_start_after = last_scanned
                break
            last_scanned = stat.object_name
            if (
                type(stat.object_name) is not str
                or not _valid_object_key(stat.object_name)
                or type(stat.size) is not int
                or stat.size < 0
                or not isinstance(stat.last_modified, datetime)
                or stat.last_modified.tzinfo is None
            ):
                raise ObjectStoreError("OBJECT_VERIFICATION_FAILED", retryable=False)
            if stat.last_modified >= older_than:
                continue
            if len(items) == limit:
                has_more = True
                next_start_after = items[-1].object_key
                break
            items.append(
                OrphanCandidate(
                    stat.object_name,
                    stat.size,
                    _metadata_checksum(stat.metadata),
                    stat.last_modified,
                )
            )
        if has_more and next_start_after is None and items:
            next_start_after = items[-1].object_key
        return OrphanPage(tuple(items), next_start_after)


__all__ = [
    "ArtifactChecksumConflict",
    "MinioObjectStore",
    "ObjectStoreError",
    "OrphanCandidate",
    "OrphanPage",
    "PublishedObject",
    "StoredObject",
    "UploadLimitExceeded",
]
