from __future__ import annotations

import asyncio
import hashlib
import inspect
import tempfile
import threading
import time
from collections.abc import AsyncIterator, Callable, Coroutine, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from types import MappingProxyType
from typing import Any, BinaryIO, cast
from urllib.parse import unquote

import pytest
from minio import Minio
from minio.commonconfig import CopySource
from minio.helpers import read_part_data

import rag_service.infrastructure.minio_store as minio_store
from rag_service.infrastructure.minio_store import (
    ArtifactChecksumConflict,
    MinioObjectStore,
    ObjectStoreError,
    PublishedObject,
    UploadLimitExceeded,
)

type MinioHeaderValue = str | list[str] | tuple[str]
type MinioHeaders = dict[str, MinioHeaderValue]


@dataclass(slots=True)
class FakeStat:
    object_name: str
    size: int
    metadata: Mapping[str, str]
    last_modified: datetime
    etag: str
    content_type: str = "application/octet-stream"


@dataclass(slots=True)
class FakeObject:
    body: bytes
    metadata: dict[str, str]
    last_modified: datetime
    content_type: str = "application/octet-stream"


@dataclass(frozen=True, slots=True)
class FakePutResult:
    etag: str


class FakeReadResponse:
    def __init__(self, body: bytes) -> None:
        self._body = body
        self._offset = 0
        self.closed = False
        self.released = False

    def stream(self, amount: int) -> Iterator[bytes]:
        while self._offset < len(self._body):
            end = min(len(self._body), self._offset + amount)
            chunk = self._body[self._offset : end]
            self._offset = end
            yield chunk

    def close(self) -> None:
        self.closed = True

    def release_conn(self) -> None:
        self.released = True


class GatedReadResponse(FakeReadResponse):
    def __init__(self, body: bytes) -> None:
        super().__init__(body)
        self.started = threading.Event()
        self.release = threading.Event()
        self.finished = threading.Event()

    def stream(self, amount: int) -> Iterator[bytes]:
        self.started.set()
        self.release.wait(5)
        try:
            yield from super().stream(amount)
        finally:
            self.finished.set()


class TrackingTemporary:
    def __init__(self, wrapped: BinaryIO) -> None:
        self._wrapped = wrapped
        self.closed_event = threading.Event()

    def write(self, value: bytes) -> int:
        assert not self.closed_event.is_set()
        return self._wrapped.write(value)

    def read(self, size: int = -1) -> bytes:
        return self._wrapped.read(size)

    def seek(self, offset: int, whence: int = 0) -> int:
        return self._wrapped.seek(offset, whence)

    def close(self) -> None:
        self.closed_event.set()
        self._wrapped.close()


class FakeS3Error(Exception):
    def __init__(self, code: str, status: int = 400) -> None:
        self.code = code
        self.status = status
        super().__init__("sanitized fake S3 failure")


class BlockingFakeMinio:
    def __init__(self, *, read_size: int = 31, read_delay: float = 0.0) -> None:
        self.read_size = read_size
        self.read_delay = read_delay
        self.objects: dict[str, FakeObject] = {}
        self.put_calls: list[dict[str, object]] = []
        self.copy_calls: list[dict[str, object]] = []
        self.remove_calls: list[str] = []
        self.abort_count = 0
        self.reader_started = threading.Event()
        self.reader_finished = threading.Event()
        self.copy_race: FakeObject | None = None
        self.remove_error: BaseException | None = None
        self.put_error_after_reads: int | None = None
        self.tamper_staging_before_stat: bytes | None = None
        self.tamper_source_before_copy: bytes | None = None
        self.multipart_uploads: dict[str, dict[str, object]] = {}
        self.multipart_sequence = 0
        self.complete_started = threading.Event()
        self.complete_release: threading.Event | None = None
        self.read_responses: list[FakeReadResponse] = []

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
    ) -> FakePutResult:
        self.put_calls.append(
            {
                "bucket": bucket_name,
                "object": object_name,
                "length": length,
                "content_type": content_type,
                "metadata": dict(metadata or {}),
                "part_size": part_size,
                "parallel": num_parallel_uploads,
            }
        )
        body = bytearray()
        reads = 0
        self.reader_started.set()
        try:
            while True:
                chunk = data.read(self.read_size)
                reads += 1
                if self.put_error_after_reads == reads:
                    raise PermissionError("minio-secret-from-consumer")
                if not chunk:
                    break
                body.extend(chunk)
                if self.read_delay:
                    time.sleep(self.read_delay)
        except Exception:
            self.abort_count += 1
            raise
        finally:
            self.reader_finished.set()
        self.objects[object_name] = FakeObject(
            bytes(body),
            dict(metadata or {}),
            datetime.now(UTC),
            content_type,
        )
        return FakePutResult(hashlib.md5(bytes(body), usedforsecurity=False).hexdigest())

    def stat_object(self, bucket_name: str, object_name: str) -> FakeStat:
        del bucket_name
        if object_name.startswith("tmp/uploads/") and self.tamper_staging_before_stat is not None:
            stored = self.objects[object_name]
            self.objects[object_name] = FakeObject(
                self.tamper_staging_before_stat,
                dict(stored.metadata),
                stored.last_modified,
                stored.content_type,
            )
            self.tamper_staging_before_stat = None
        try:
            stored = self.objects[object_name]
        except KeyError:
            raise FakeS3Error("NoSuchKey", 404) from None
        return FakeStat(
            object_name,
            len(stored.body),
            MappingProxyType(
                {f"x-amz-meta-{key}": value for key, value in stored.metadata.items()}
            ),
            stored.last_modified,
            hashlib.md5(stored.body, usedforsecurity=False).hexdigest(),
            stored.content_type,
        )

    def get_object(self, bucket_name: str, object_name: str) -> FakeReadResponse:
        del bucket_name
        try:
            stored = self.objects[object_name]
        except KeyError:
            raise FakeS3Error("NoSuchKey", 404) from None
        response = FakeReadResponse(stored.body)
        self.read_responses.append(response)
        return response

    def _create_multipart_upload(
        self,
        bucket_name: str,
        object_name: str,
        headers: MinioHeaders,
    ) -> str:
        del bucket_name
        self.multipart_sequence += 1
        upload_id = f"upload-{self.multipart_sequence}"
        self.multipart_uploads[upload_id] = {
            "destination": object_name,
            "metadata": {
                key.removeprefix("x-amz-meta-"): value
                for key, value in headers.items()
                if key.startswith("x-amz-meta-")
            },
            "content_type": headers["Content-Type"],
            "parts": {},
        }
        return upload_id

    def _upload_part_copy(
        self,
        bucket_name: str,
        object_name: str,
        upload_id: str,
        part_number: int,
        headers: MinioHeaders,
    ) -> tuple[str, datetime | None]:
        del bucket_name
        upload = self.multipart_uploads[upload_id]
        assert upload["destination"] == object_name
        source_path = unquote(cast(str, headers["x-amz-copy-source"]))
        source_name = source_path.removeprefix("/private-rag-bucket/")
        if self.tamper_source_before_copy is not None:
            stored = self.objects[source_name]
            self.objects[source_name] = FakeObject(
                self.tamper_source_before_copy,
                dict(stored.metadata),
                stored.last_modified,
            )
            self.tamper_source_before_copy = None
        source = self.objects[source_name]
        source_etag = hashlib.md5(source.body, usedforsecurity=False).hexdigest()
        if headers.get("x-amz-copy-source-if-match") != source_etag:
            raise FakeS3Error("PreconditionFailed", 412)
        byte_range = cast(str | None, headers.get("x-amz-copy-source-range"))
        if byte_range:
            start_text, end_text = byte_range.removeprefix("bytes=").split("-", 1)
            body = source.body[int(start_text) : int(end_text) + 1]
        else:
            body = source.body
        parts = upload["parts"]
        assert isinstance(parts, dict)
        parts[part_number] = body
        self.copy_calls.append(
            {
                "destination": object_name,
                "source": source_name,
                "source_if_match": headers["x-amz-copy-source-if-match"],
                "range": byte_range,
            }
        )
        return hashlib.md5(body, usedforsecurity=False).hexdigest(), None

    def _execute(
        self,
        method: str,
        bucket_name: str | None = None,
        object_name: str | None = None,
        body: bytes | None = None,
        headers: MinioHeaders | None = None,
        query_params: MinioHeaders | None = None,
        preload_content: bool = True,
        no_body_trace: bool = False,
    ) -> object:
        del bucket_name, body, preload_content, no_body_trace
        assert method == "POST"
        assert object_name is not None
        assert headers is not None
        assert query_params is not None
        upload_id = cast(str, query_params["uploadId"])
        upload = self.multipart_uploads[upload_id]
        self.complete_started.set()
        if self.complete_release is not None:
            self.complete_release.wait(2)
        if self.copy_race is not None:
            self.objects[object_name] = self.copy_race
            self.copy_race = None
        if headers.get("If-None-Match") != "*" or object_name in self.objects:
            raise FakeS3Error("PreconditionFailed", 412)
        parts = upload["parts"]
        metadata = upload["metadata"]
        assert isinstance(parts, dict)
        assert isinstance(metadata, dict)
        self.objects[object_name] = FakeObject(
            b"".join(parts[index] for index in sorted(parts)),
            dict(metadata),
            datetime.now(UTC),
            str(upload["content_type"]),
        )
        del self.multipart_uploads[upload_id]
        self.copy_calls.append({"destination": object_name, "if_none_match": "*"})
        return object()

    def _abort_multipart_upload(
        self,
        bucket_name: str,
        object_name: str,
        upload_id: str,
    ) -> None:
        del bucket_name, object_name
        self.abort_count += 1
        self.multipart_uploads.pop(upload_id, None)

    def remove_object(
        self,
        bucket_name: str,
        object_name: str,
        version_id: str | None = None,
    ) -> None:
        del bucket_name, version_id
        self.remove_calls.append(object_name)
        if self.remove_error is not None:
            raise self.remove_error
        self.objects.pop(object_name, None)

    def list_objects(
        self,
        bucket_name: str,
        *,
        prefix: str | None = None,
        recursive: bool = False,
        start_after: str | None = None,
        include_user_meta: bool = False,
    ) -> Iterator[FakeStat]:
        del bucket_name, recursive, include_user_meta
        for name in sorted(self.objects):
            if prefix is not None and not name.startswith(prefix):
                continue
            if start_after is not None and name <= start_after:
                continue
            yield self.stat_object("ignored", name)


class PrivateExecuteFakeMinio(BlockingFakeMinio):
    def __init__(self) -> None:
        super().__init__()
        self.execute_headers: MinioHeaders | None = None
        self.execute_query: MinioHeaders | None = None

    def _execute(
        self,
        method: str,
        bucket_name: str | None = None,
        object_name: str | None = None,
        body: bytes | None = None,
        headers: MinioHeaders | None = None,
        query_params: MinioHeaders | None = None,
        preload_content: bool = True,
        no_body_trace: bool = False,
    ) -> object:
        assert headers is not None
        self.execute_headers = dict(headers)
        self.execute_query = dict(query_params or {})
        return super()._execute(
            method,
            bucket_name,
            object_name,
            body,
            headers,
            query_params,
            preload_content,
            no_body_trace,
        )


class PublicOnlyRaceFakeMinio(BlockingFakeMinio):
    def __init__(self, final_key: str) -> None:
        super().__init__()
        self._create_multipart_upload = None  # type: ignore[assignment]
        self._final_key = final_key
        self._blind_missing_stats = 2
        self._stat_lock = threading.Lock()

    def stat_object(self, bucket_name: str, object_name: str) -> FakeStat:
        if object_name == self._final_key:
            with self._stat_lock:
                if self._blind_missing_stats > 0:
                    self._blind_missing_stats -= 1
                    raise FakeS3Error("NoSuchKey", 404)
        return super().stat_object(bucket_name, object_name)


class WrongSignatureFakeMinio(BlockingFakeMinio):
    def _execute(
        self,
        method: str,
        bucket_name: str | None = None,
        object_name: str | None = None,
        body: bytes | None = None,
        headers: MinioHeaders | None = None,
        query_params: MinioHeaders | None = None,
        preload_content: bool = False,
        no_body_trace: bool = False,
    ) -> object:
        return super()._execute(
            method,
            bucket_name,
            object_name,
            body,
            headers,
            query_params,
            preload_content,
            no_body_trace,
        )


class BlockingConditionalCopyFakeMinio(BlockingFakeMinio):
    def __init__(self) -> None:
        super().__init__()
        self.copy_started = self.complete_started
        self.copy_release = threading.Event()
        self.complete_release = self.copy_release
        self.copy_finished = threading.Event()

    def _execute(
        self,
        method: str,
        bucket_name: str | None = None,
        object_name: str | None = None,
        body: bytes | None = None,
        headers: MinioHeaders | None = None,
        query_params: MinioHeaders | None = None,
        preload_content: bool = True,
        no_body_trace: bool = False,
    ) -> object:
        try:
            return super()._execute(
                method,
                bucket_name,
                object_name,
                body,
                headers,
                query_params,
                preload_content,
                no_body_trace,
            )
        finally:
            self.copy_finished.set()


class BlockingAfterReadFakeMinio(BlockingFakeMinio):
    def __init__(self) -> None:
        super().__init__()
        self.network_hang_started = threading.Event()
        self.network_release = threading.Event()
        self.raise_after_store = False

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
    ) -> FakePutResult:
        del bucket_name, length, part_size, num_parallel_uploads
        body = bytearray()
        self.reader_started.set()
        while True:
            chunk = data.read(64 * 1024)
            if not chunk:
                break
            body.extend(chunk)
        self.reader_finished.set()
        self.network_hang_started.set()
        self.network_release.wait(5)
        self.objects[object_name] = FakeObject(
            bytes(body),
            dict(metadata or {}),
            datetime.now(UTC),
            content_type,
        )
        if self.raise_after_store:
            raise TimeoutError("late network failure")
        return FakePutResult(hashlib.md5(bytes(body), usedforsecurity=False).hexdigest())


class BlockingPutAndRemoveFakeMinio(BlockingAfterReadFakeMinio):
    def __init__(self) -> None:
        super().__init__()
        self.delete_started = threading.Event()
        self.delete_release = threading.Event()
        self.delete_finished = threading.Event()

    def remove_object(
        self,
        bucket_name: str,
        object_name: str,
        version_id: str | None = None,
    ) -> None:
        del version_id
        try:
            self.delete_started.set()
            self.delete_release.wait(5)
            super().remove_object(bucket_name, object_name)
        finally:
            self.delete_finished.set()


class BlockingCopyAndRemoveFakeMinio(BlockingConditionalCopyFakeMinio):
    def __init__(self) -> None:
        super().__init__()
        self.delete_started = threading.Event()
        self.delete_release = threading.Event()

    def remove_object(
        self,
        bucket_name: str,
        object_name: str,
        version_id: str | None = None,
    ) -> None:
        del version_id
        self.delete_started.set()
        self.delete_release.wait(5)
        super().remove_object(bucket_name, object_name)


class SdkAggregatingFakeMinio(BlockingFakeMinio):
    def __init__(self) -> None:
        super().__init__()
        self.max_single_pipe_read = 0

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
    ) -> FakePutResult:
        del bucket_name, length, num_parallel_uploads

        class RecordingReader:
            def __init__(self, source: BinaryIO, owner: SdkAggregatingFakeMinio) -> None:
                self.source = source
                self.owner = owner

            def read(self, size: int = -1) -> bytes:
                result = self.source.read(size)
                self.owner.max_single_pipe_read = max(
                    self.owner.max_single_pipe_read,
                    len(result),
                )
                return result

        reader = RecordingReader(data, self)
        first = read_part_data(cast(BinaryIO, reader), part_size + 1)
        if len(first) <= part_size:
            body = first
        else:
            body = first[:-1] + read_part_data(cast(BinaryIO, reader), part_size + 1, first[-1:])
        self.objects[object_name] = FakeObject(
            body,
            dict(metadata or {}),
            datetime.now(UTC),
            content_type,
        )
        return FakePutResult(hashlib.md5(body, usedforsecurity=False).hexdigest())


class EarlyStopFakeMinio(BlockingFakeMinio):
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
    ) -> FakePutResult:
        del bucket_name, length, part_size, num_parallel_uploads
        body = data.read(1)
        self.objects[object_name] = FakeObject(
            body,
            dict(metadata or {}),
            datetime.now(UTC),
            content_type,
        )
        return FakePutResult(hashlib.md5(body, usedforsecurity=False).hexdigest())


class BlockingOperationFakeMinio(BlockingFakeMinio):
    def __init__(self, operation: str) -> None:
        super().__init__()
        self.operation = operation
        self.operation_started = threading.Event()
        self.operation_release = threading.Event()

    def _block(self, operation: str) -> None:
        if self.operation == operation:
            self.operation_started.set()
            self.operation_release.wait(5)

    def stat_object(self, bucket_name: str, object_name: str) -> FakeStat:
        self._block("stat")
        return super().stat_object(bucket_name, object_name)

    def remove_object(
        self,
        bucket_name: str,
        object_name: str,
        version_id: str | None = None,
    ) -> None:
        del version_id
        self._block("delete")
        super().remove_object(bucket_name, object_name)

    def list_objects(
        self,
        bucket_name: str,
        *,
        prefix: str | None = None,
        recursive: bool = False,
        start_after: str | None = None,
        include_user_meta: bool = False,
    ) -> Iterator[FakeStat]:
        self._block("list")
        yield from super().list_objects(
            bucket_name,
            prefix=prefix,
            recursive=recursive,
            start_after=start_after,
            include_user_meta=include_user_meta,
        )


class BlockingHeadAndRemoveFakeMinio(BlockingOperationFakeMinio):
    def __init__(self) -> None:
        super().__init__("stat")
        self.delete_started = threading.Event()
        self.delete_release = threading.Event()
        self.delete_finished = threading.Event()

    def remove_object(
        self,
        bucket_name: str,
        object_name: str,
        version_id: str | None = None,
    ) -> None:
        del version_id
        try:
            self.delete_started.set()
            self.delete_release.wait(5)
            BlockingFakeMinio.remove_object(self, bucket_name, object_name)
        finally:
            self.delete_finished.set()


class MultipartAbortProbeMinio(Minio):
    def __init__(self) -> None:
        super().__init__(
            "127.0.0.1:1",
            access_key="test-access",
            secret_key="test-secret",
            secure=False,
        )
        self.created_upload_id: str | None = None
        self.uploaded_parts: list[int] = []
        self.aborted_upload_ids: list[str] = []
        self.remove_calls: list[str] = []

    def _create_multipart_upload(
        self,
        bucket_name: str,
        object_name: str,
        headers: MinioHeaders,
    ) -> str:
        del bucket_name, object_name, headers
        self.created_upload_id = "real-sdk-upload-id"
        return self.created_upload_id

    def _upload_part(
        self,
        bucket_name: str,
        object_name: str,
        data: bytes,
        headers: MinioHeaders | None,
        upload_id: str,
        part_number: int,
    ) -> str:
        del bucket_name, object_name, data, headers
        assert upload_id == self.created_upload_id
        self.uploaded_parts.append(part_number)
        return f"etag-{part_number}"

    def _abort_multipart_upload(
        self,
        bucket_name: str,
        object_name: str,
        upload_id: str,
    ) -> None:
        del bucket_name, object_name
        self.aborted_upload_ids.append(upload_id)

    def remove_object(
        self,
        bucket_name: str,
        object_name: str,
        version_id: str | None = None,
    ) -> None:
        del bucket_name, version_id
        self.remove_calls.append(object_name)


async def chunks(*values: bytes, delay: float = 0.0) -> AsyncIterator[bytes]:
    for value in values:
        if delay:
            await asyncio.sleep(delay)
        yield value


class SlowClosingStream:
    def __init__(self, body: bytes, *, close_delay: float) -> None:
        self._body = body
        self._close_delay = close_delay
        self._yielded = False
        self.close_started = asyncio.Event()
        self.close_finished = asyncio.Event()

    def __aiter__(self) -> SlowClosingStream:
        return self

    async def __anext__(self) -> bytes:
        if self._yielded:
            raise StopAsyncIteration
        self._yielded = True
        return self._body

    async def aclose(self) -> None:
        self.close_started.set()
        await asyncio.sleep(self._close_delay)
        self.close_finished.set()


class CancellationResistantClosingStream:
    def __init__(self, body: bytes) -> None:
        self._body = body
        self._yielded = False
        self.close_started = asyncio.Event()
        self.close_release = asyncio.Event()
        self.close_finished = asyncio.Event()

    def __aiter__(self) -> CancellationResistantClosingStream:
        return self

    async def __anext__(self) -> bytes:
        if self._yielded:
            raise StopAsyncIteration
        self._yielded = True
        return self._body

    async def aclose(self) -> None:
        self.close_started.set()
        while not self.close_release.is_set():
            try:
                await self.close_release.wait()
            except asyncio.CancelledError:
                continue
        self.close_finished.set()


class ObservedClosingStream:
    def __init__(self, body: bytes) -> None:
        self._body = body
        self._yielded = False
        self.consumed_bytes = 0
        self.close_calls = 0

    def __aiter__(self) -> ObservedClosingStream:
        return self

    async def __anext__(self) -> bytes:
        if self._yielded:
            raise StopAsyncIteration
        self._yielded = True
        self.consumed_bytes += len(self._body)
        return self._body

    async def aclose(self) -> None:
        self.close_calls += 1


async def heartbeat(stop: asyncio.Event, ticks: list[int]) -> None:
    while not stop.is_set():
        ticks[0] += 1
        await asyncio.sleep(0)


def make_store(client: BlockingFakeMinio, *, buffer_bytes: int = 64) -> MinioObjectStore:
    return MinioObjectStore(
        client=client,
        bucket="private-rag-bucket",
        buffer_bytes=buffer_bytes,
        part_size_bytes=5 * 1024 * 1024,
    )


@pytest.mark.asyncio
async def test_bounded_read_verifies_metadata_size_and_content_checksum() -> None:
    client = BlockingFakeMinio()
    store = make_store(client)
    key = "knowledge-bases/k/documents/d/versions/v/source/source.txt"
    body = b"verified source\n"
    checksum = hashlib.sha256(body).hexdigest()
    client.objects[key] = FakeObject(body, {"sha256": checksum}, datetime.now(UTC))

    loaded = await store.read_bytes(
        key,
        expected_checksum=checksum,
        max_bytes=len(body),
    )

    assert loaded == body
    assert client.read_responses[-1].closed is True
    assert client.read_responses[-1].released is True


@pytest.mark.asyncio
async def test_verified_stream_checks_full_object_before_yielding_bounded_chunks() -> None:
    client = BlockingFakeMinio()
    store = make_store(client)
    key = "knowledge-bases/k/documents/d/versions/v/chunks/recursive_text_v1.jsonl"
    body = b'{"chunk_index":0}\n' * 10_000
    checksum = hashlib.sha256(body).hexdigest()
    client.objects[key] = FakeObject(body, {"sha256": checksum}, datetime.now(UTC))

    chunks = [
        chunk
        async for chunk in store.read_stream(
            key,
            expected_checksum=checksum,
            max_bytes=len(body),
        )
    ]

    assert b"".join(chunks) == body
    assert chunks
    assert max(map(len, chunks)) <= store.producer_slice_bytes
    assert client.read_responses[-1].closed is True
    assert client.read_responses[-1].released is True


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["cancel", "timeout", "close"])
async def test_verified_stream_download_owns_temporary_until_blocking_writer_settles(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    client = BlockingFakeMinio()
    key = "knowledge-bases/k/documents/d/versions/v/chunks/gated.jsonl"
    body = b'{"chunk_index":0}\n'
    checksum = hashlib.sha256(body).hexdigest()
    client.objects[key] = FakeObject(body, {"sha256": checksum}, datetime.now(UTC))
    response = GatedReadResponse(body)

    def get_object(_bucket: str, object_name: str) -> FakeReadResponse:
        assert object_name == key
        client.read_responses.append(response)
        return response

    real_temporary = cast(Callable[..., BinaryIO], tempfile.TemporaryFile)
    temporaries: list[TrackingTemporary] = []

    def temporary_file(*args: object, **kwargs: object) -> TrackingTemporary:
        wrapped = real_temporary(*args, **kwargs)
        tracked = TrackingTemporary(wrapped)
        temporaries.append(tracked)
        return tracked

    monkeypatch.setattr(client, "get_object", get_object)
    monkeypatch.setattr(
        "rag_service.infrastructure.minio_store.tempfile.TemporaryFile",
        temporary_file,
    )
    store = MinioObjectStore(
        client=client,
        bucket="private-rag-bucket",
        buffer_bytes=64,
        part_size_bytes=5 * 1024 * 1024,
        producer_slice_bytes=8,
        operation_timeout_seconds=0.05 if mode == "timeout" else 1,
        cancel_grace_seconds=0.01,
        io_max_workers=1,
    )
    stream = store.read_stream(
        key,
        expected_checksum=checksum,
        max_bytes=len(body),
    )

    async def read_first() -> bytes:
        return await anext(stream)

    read: asyncio.Task[bytes] = asyncio.create_task(read_first())
    assert await asyncio.to_thread(response.started.wait, 1)
    assert len(temporaries) == 1

    if mode == "cancel":
        read.cancel("stop verified stream")
        with pytest.raises(asyncio.CancelledError):
            await read
    elif mode == "timeout":
        with pytest.raises(ObjectStoreError) as exc_info:
            await read
        assert exc_info.value.code == "OBJECT_STORE_UNAVAILABLE"
    else:
        with pytest.raises(ObjectStoreError) as exc_info:
            await store.aclose()
        assert exc_info.value.code == "OBJECT_STORE_UNAVAILABLE"
        try:
            await read
        except asyncio.CancelledError:
            pass
        except ObjectStoreError as read_error:
            assert read_error.code == "OBJECT_STORE_UNAVAILABLE"
        else:
            pytest.fail("closing the store must stop the active read")

    assert temporaries[0].closed_event.is_set() is False
    assert response.closed is False
    response.release.set()
    assert await asyncio.to_thread(response.finished.wait, 1)
    if mode != "close":
        await store.aclose()
    assert await asyncio.to_thread(temporaries[0].closed_event.wait, 1)
    await asyncio.sleep(0)

    assert temporaries[0].closed_event.is_set()
    assert response.closed is True
    assert response.released is True
    assert store.pending_operation_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["oversize", "metadata", "content"])
async def test_bounded_read_rejects_oversize_or_checksum_mismatch(failure: str) -> None:
    client = BlockingFakeMinio()
    store = make_store(client)
    key = "knowledge-bases/k/documents/d/versions/v/source/source.txt"
    body = b"verified source\n"
    checksum = hashlib.sha256(body).hexdigest()
    metadata_checksum = checksum
    expected_checksum = checksum
    max_bytes = len(body)
    if failure == "oversize":
        max_bytes -= 1
    elif failure == "metadata":
        metadata_checksum = "f" * 64
    else:
        expected_checksum = hashlib.sha256(b"different").hexdigest()
        metadata_checksum = expected_checksum
    client.objects[key] = FakeObject(body, {"sha256": metadata_checksum}, datetime.now(UTC))

    with pytest.raises(ObjectStoreError) as exc_info:
        await store.read_bytes(
            key,
            expected_checksum=expected_checksum,
            max_bytes=max_bytes,
        )

    assert exc_info.value.code == "OBJECT_VERIFICATION_FAILED"


@pytest.mark.asyncio
async def test_stream_upload_is_byte_bounded_nonblocking_and_hashes_while_streaming() -> None:
    client = BlockingFakeMinio(read_size=17, read_delay=0.0005)
    store = make_store(client, buffer_bytes=64)
    payload = b"0123456789abcdef" * 256
    stop = asyncio.Event()
    ticks = [0]
    heartbeat_task = asyncio.create_task(heartbeat(stop, ticks))
    try:
        stored = await store.upload_stream(
            "knowledge-bases/k/documents/d/versions/v/source/source.txt",
            chunks(payload[:2000], b"", payload[2000:]),
            content_type="text/plain",
            max_bytes=len(payload),
        )
    finally:
        stop.set()
        await heartbeat_task

    assert ticks[0] > 10
    assert stored.size == len(payload)
    assert stored.checksum_sha256 == hashlib.sha256(payload).hexdigest()
    assert stored.max_pipe_buffered_bytes <= 64
    assert stored.producer_slice_bytes <= 64
    assert stored.adapter_memory_bound_bytes == (
        64 + (2 * ((5 * 1024 * 1024) + 1)) + stored.producer_slice_bytes
    )
    assert stored.object_key not in repr(stored)
    assert client.objects[stored.object_key].body == payload
    assert client.objects[stored.object_key].metadata == {"sha256": stored.checksum_sha256}
    assert client.objects[stored.object_key].content_type == "text/plain"
    assert client.put_calls[0]["length"] == -1
    assert client.put_calls[0]["part_size"] == 5 * 1024 * 1024
    assert client.put_calls[0]["parallel"] == 1
    assert all(not name.startswith("tmp/uploads/") for name in client.objects)


@pytest.mark.asyncio
async def test_concurrent_multichunk_uploads_do_not_depend_on_the_default_executor() -> None:
    loop = asyncio.get_running_loop()
    default_executor = ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="test-starved-default-executor",
    )
    loop.set_default_executor(default_executor)
    client = BlockingFakeMinio(read_size=2)
    store = MinioObjectStore(
        client=client,
        bucket="private-rag-bucket",
        buffer_bytes=4,
        part_size_bytes=5 * 1024 * 1024,
        operation_timeout_seconds=0.2,
        cancel_grace_seconds=0.01,
    )
    payloads = (b"abcdefghijklmno", b"123456789abcdef")

    try:
        stored = await asyncio.wait_for(
            asyncio.gather(
                *(
                    store.upload_stream(
                        f"knowledge-bases/k/documents/d/versions/v/source/{index}.txt",
                        chunks(payload[:3], payload[3:8], payload[8:]),
                        content_type="text/plain",
                        max_bytes=len(payload),
                    )
                    for index, payload in enumerate(payloads)
                )
            ),
            timeout=1,
        )
    finally:
        await store.aclose()
        default_executor.shutdown(wait=True, cancel_futures=True)

    assert [item.size for item in stored] == [len(payload) for payload in payloads]
    assert all(
        client.objects[item.object_key].body == payload
        for item, payload in zip(stored, payloads, strict=True)
    )


@pytest.mark.asyncio
async def test_queued_upload_does_not_consume_stream_before_an_io_slot_is_available() -> None:
    client = BlockingAfterReadFakeMinio()
    store = MinioObjectStore(
        client=client,
        bucket="private-rag-bucket",
        buffer_bytes=4,
        part_size_bytes=5 * 1024 * 1024,
        operation_timeout_seconds=1,
        cancel_grace_seconds=0.01,
        io_max_workers=1,
    )
    first = asyncio.create_task(
        store.upload_stream(
            "knowledge-bases/k/documents/d/versions/v/source/first.txt",
            chunks(b"first upload"),
            content_type="text/plain",
            max_bytes=100,
        )
    )
    second_stream = ObservedClosingStream(b"second upload")
    second: asyncio.Task[minio_store.StoredObject] | None = None

    try:
        assert await asyncio.to_thread(client.network_hang_started.wait, 1)
        second = asyncio.create_task(
            store.upload_stream(
                "knowledge-bases/k/documents/d/versions/v/source/second.txt",
                second_stream,
                content_type="text/plain",
                max_bytes=100,
            )
        )
        await asyncio.sleep(0)
        assert second in store._active_operations
        await asyncio.sleep(0)

        assert store._io_executor.active_count == 1
        assert second_stream.consumed_bytes == 0
        assert second_stream.close_calls == 0

        client.network_release.set()
        first_stored, second_stored = await asyncio.wait_for(
            asyncio.gather(first, second),
            timeout=2,
        )

        assert client.objects[first_stored.object_key].body == b"first upload"
        assert client.objects[second_stored.object_key].body == b"second upload"
        assert second_stream.consumed_bytes == len(b"second upload")
        assert second_stream.close_calls == 1
    finally:
        client.network_release.set()
        pending = [task for task in (first, second) if task is not None and not task.done()]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        with suppress(ObjectStoreError):
            await store.aclose()


@pytest.mark.asyncio
async def test_queued_upload_timeout_closes_an_unconsumed_stream_without_leaks() -> None:
    client = BlockingOperationFakeMinio("delete")
    store = MinioObjectStore(
        client=client,
        bucket="private-rag-bucket",
        buffer_bytes=4,
        part_size_bytes=5 * 1024 * 1024,
        operation_timeout_seconds=0.05,
        cancel_grace_seconds=0.01,
        io_max_workers=1,
    )
    blocker = asyncio.create_task(store.delete_best_effort("tmp/jobs/blocker.txt"))
    stream = ObservedClosingStream(b"must remain unconsumed")

    try:
        assert await asyncio.to_thread(client.operation_started.wait, 1)
        started = asyncio.get_running_loop().time()
        with pytest.raises(ObjectStoreError) as exc_info:
            await asyncio.wait_for(
                store.upload_stream(
                    "knowledge-bases/k/documents/d/versions/v/source/timeout.txt",
                    stream,
                    content_type="text/plain",
                    max_bytes=100,
                ),
                timeout=0.2,
            )

        assert exc_info.value.code == "OBJECT_STORE_UNAVAILABLE"
        assert asyncio.get_running_loop().time() - started < 0.15
        assert stream.consumed_bytes == 0
        assert stream.close_calls == 1
        assert store._io_executor.active_count == 1
        assert store.pending_operation_count == 1
        assert [
            task.get_name()
            for task in asyncio.all_tasks()
            if task is not asyncio.current_task()
            and not task.done()
            and task.get_name().startswith("minio-")
        ] == ["minio-sdk-operation"]
    finally:
        client.operation_release.set()
        await asyncio.gather(blocker, return_exceptions=True)
        with suppress(ObjectStoreError):
            await store.aclose()


@pytest.mark.asyncio
async def test_queued_upload_cancellation_closes_input_without_cleanup_task_leaks() -> None:
    client = BlockingOperationFakeMinio("delete")
    store = MinioObjectStore(
        client=client,
        bucket="private-rag-bucket",
        buffer_bytes=4,
        part_size_bytes=5 * 1024 * 1024,
        operation_timeout_seconds=1,
        cancel_grace_seconds=0.01,
        io_max_workers=1,
    )
    blocker = asyncio.create_task(store.delete_best_effort("tmp/jobs/blocker.txt"))
    stream = ObservedClosingStream(b"must remain unconsumed")
    upload: asyncio.Task[minio_store.StoredObject] | None = None

    try:
        assert await asyncio.to_thread(client.operation_started.wait, 1)
        upload = asyncio.create_task(
            store.upload_stream(
                "knowledge-bases/k/documents/d/versions/v/source/cancelled.txt",
                stream,
                content_type="text/plain",
                max_bytes=100,
            )
        )
        await asyncio.sleep(0)
        assert upload in store._active_operations
        await asyncio.sleep(0)

        upload.cancel("client disconnected")
        with pytest.raises(asyncio.CancelledError) as cancellation:
            await asyncio.wait_for(upload, timeout=0.2)

        assert cancellation.value.args == ("client disconnected",)
        assert stream.consumed_bytes == 0
        assert stream.close_calls == 1
        assert store._io_executor.active_count == 1
        assert store.pending_operation_count == 1
        assert [
            task.get_name()
            for task in asyncio.all_tasks()
            if task is not asyncio.current_task()
            and not task.done()
            and task.get_name().startswith("minio-")
        ] == ["minio-sdk-operation"]
    finally:
        client.operation_release.set()
        pending = [task for task in (blocker, upload) if task is not None and not task.done()]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        with suppress(ObjectStoreError):
            await store.aclose()


@pytest.mark.asyncio
async def test_close_joins_owned_minio_executor_threads() -> None:
    client = BlockingFakeMinio()
    store = make_store(client)
    stored = await store.upload_stream(
        "knowledge-bases/k/documents/d/versions/v/source/thread-lifecycle.txt",
        chunks(b"thread lifecycle"),
        content_type="text/plain",
        max_bytes=100,
    )
    io_thread_ids = {
        thread.ident
        for thread in threading.enumerate()
        if thread.ident is not None
        and thread.name.startswith(store._io_executor.thread_name_prefix)
    }

    assert client.objects[stored.object_key].body == b"thread lifecycle"
    assert io_thread_ids

    await store.aclose()

    assert all(thread.ident not in io_thread_ids for thread in threading.enumerate())


@pytest.mark.asyncio
async def test_cancelled_close_bounds_cleanup_and_keeps_failure_sticky() -> None:
    client = BlockingHeadAndRemoveFakeMinio()
    store = MinioObjectStore(
        client=client,
        bucket="private-rag-bucket",
        buffer_bytes=64,
        part_size_bytes=5 * 1024 * 1024,
        operation_timeout_seconds=1,
        cancel_grace_seconds=0.01,
        io_max_workers=1,
    )
    object_key = "tmp/jobs/job/cancel-close/parsed/text.txt"
    client.objects[object_key] = FakeObject(b"cancel close", {}, datetime.now(UTC))
    delete = asyncio.create_task(store.delete_best_effort(object_key))
    assert await asyncio.to_thread(client.delete_started.wait, 1)
    thread_prefix = store._io_executor.thread_name_prefix
    io_threads = [
        thread for thread in threading.enumerate() if thread.name.startswith(thread_prefix)
    ]
    assert io_threads
    close = asyncio.create_task(store.aclose())
    await asyncio.sleep(0)

    started = asyncio.get_running_loop().time()
    close.cancel("shutdown timeout")
    with pytest.raises(asyncio.CancelledError) as cancellation:
        await asyncio.wait_for(close, timeout=0.2)

    assert cancellation.value.args == ("shutdown timeout",)
    assert asyncio.get_running_loop().time() - started < 0.2
    assert store._closed is True
    assert store._close_failed is True
    assert store._io_executor._shutdown is True
    assert store._io_executor.active_count == 1
    assert store.pending_operation_count == 0
    assert delete.done()
    assert delete.cancelled()
    assert [
        task
        for task in asyncio.all_tasks()
        if task is not asyncio.current_task()
        and not task.done()
        and task.get_name().startswith("minio-")
    ] == []
    assert any(thread.is_alive() for thread in io_threads)

    with pytest.raises(ObjectStoreError) as second_close_error:
        await asyncio.wait_for(store.aclose(), timeout=0.2)
    assert second_close_error.value.code == "OBJECT_STORE_UNAVAILABLE"
    assert store._io_executor.active_count == 1
    assert any(thread.is_alive() for thread in io_threads)

    client.delete_release.set()
    assert await asyncio.to_thread(client.delete_finished.wait, 1)
    await asyncio.gather(*(asyncio.to_thread(thread.join, 1) for thread in io_threads))
    await asyncio.sleep(0)

    with pytest.raises(ObjectStoreError) as settled_close_error:
        await store.aclose()
    assert settled_close_error.value.code == "OBJECT_STORE_UNAVAILABLE"
    assert store.pending_operation_count == 0
    assert store._io_executor.active_count == 0
    assert all(not thread.is_alive() for thread in io_threads)


@pytest.mark.asyncio
async def test_cancelled_close_while_waiting_for_lifecycle_lock_closes_admission() -> None:
    store = make_store(BlockingFakeMinio())
    await store._lifecycle_lock.acquire()
    close = asyncio.create_task(store.aclose())
    await asyncio.sleep(0)

    close.cancel()
    store._lifecycle_lock.release()
    with pytest.raises(asyncio.CancelledError):
        await close

    assert store._closed is True
    assert store._close_failed is True
    assert store._io_executor._shutdown is True
    assert store._io_executor.active_count == 0
    with pytest.raises(ObjectStoreError) as second_close_error:
        await store.aclose()
    assert second_close_error.value.code == "OBJECT_STORE_UNAVAILABLE"


@pytest.mark.asyncio
async def test_successful_upload_waits_for_slow_stream_close_within_operation_timeout() -> None:
    client = BlockingFakeMinio()
    store = MinioObjectStore(
        client=client,
        bucket="private-rag-bucket",
        buffer_bytes=64,
        part_size_bytes=5 * 1024 * 1024,
        operation_timeout_seconds=1,
        cancel_grace_seconds=0.01,
    )
    stream = SlowClosingStream(b"ok", close_delay=0.05)
    final_key = "knowledge-bases/k/documents/d/versions/v/source/slow-close.txt"

    stored = await store.upload_stream(
        final_key,
        stream,
        content_type="text/plain",
        max_bytes=2,
    )

    assert stream.close_finished.is_set()
    assert stored.size == 2
    assert client.objects[final_key].body == b"ok"


@pytest.mark.asyncio
async def test_close_waits_for_an_admitted_upload_after_sdk_io_finishes() -> None:
    client = BlockingFakeMinio()
    store = MinioObjectStore(
        client=client,
        bucket="private-rag-bucket",
        buffer_bytes=64,
        part_size_bytes=5 * 1024 * 1024,
        operation_timeout_seconds=1,
        cancel_grace_seconds=0.01,
    )
    stream = SlowClosingStream(b"close-race", close_delay=0.1)
    final_key = "knowledge-bases/k/documents/d/versions/v/source/close-race.txt"
    upload = asyncio.create_task(
        store.upload_stream(
            final_key,
            stream,
            content_type="text/plain",
            max_bytes=100,
        )
    )
    close: asyncio.Task[None] | None = None

    try:
        await asyncio.wait_for(stream.close_started.wait(), timeout=1)
        assert await asyncio.to_thread(client.reader_finished.wait, 1)
        sdk_wrappers = tuple(store._blocking_operations)
        if sdk_wrappers:
            await asyncio.wait_for(
                asyncio.gather(*(asyncio.shield(task) for task in sdk_wrappers)),
                timeout=1,
            )
        await asyncio.sleep(0)
        assert store.pending_operation_count == 0
        assert not upload.done()

        close = asyncio.create_task(store.aclose())
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(asyncio.shield(close), timeout=0.02)

        await asyncio.wait_for(close, timeout=1)
        stored = await asyncio.wait_for(upload, timeout=1)
        assert stored.object_key == final_key
        assert client.objects[final_key].body == b"close-race"
        assert all(not key.startswith("tmp/uploads/") for key in client.objects)

        objects_after_close = dict(client.objects)
        mutations_after_close = (len(client.copy_calls), len(client.remove_calls))
        await asyncio.sleep(0.05)
        assert client.objects == objects_after_close
        assert (len(client.copy_calls), len(client.remove_calls)) == mutations_after_close
    finally:
        pending = [task for task in (close, upload) if task is not None and not task.done()]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)


@pytest.mark.asyncio
async def test_external_upload_cancellation_during_graceful_close_still_cleans_staging() -> None:
    client = BlockingAfterReadFakeMinio()
    store = MinioObjectStore(
        client=client,
        bucket="private-rag-bucket",
        buffer_bytes=64,
        part_size_bytes=5 * 1024 * 1024,
        operation_timeout_seconds=1,
        cancel_grace_seconds=0.02,
    )
    upload = asyncio.create_task(
        store.upload_stream(
            "knowledge-bases/k/documents/d/versions/v/source/graceful-close.txt",
            chunks(b"graceful close cancellation"),
            content_type="text/plain",
            max_bytes=100,
        )
    )
    await asyncio.to_thread(client.network_hang_started.wait, 1)
    close = asyncio.create_task(store.aclose())
    await asyncio.sleep(0)
    assert store._closed is True

    upload.cancel()
    try:
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(asyncio.shield(upload), timeout=0.2)
    finally:
        client.network_release.set()

    await asyncio.wait_for(close, timeout=1)
    assert any(key.startswith("tmp/uploads/") for key in client.remove_calls)
    assert all(not key.startswith("tmp/uploads/") for key in client.objects)


@pytest.mark.asyncio
async def test_stream_upload_rejects_same_size_staging_tampering_before_publish() -> None:
    client = BlockingFakeMinio()
    client.tamper_staging_before_stat = b"evil"
    store = make_store(client)
    final_key = "knowledge-bases/k/documents/d/versions/v/source/source.txt"

    with pytest.raises(ObjectStoreError) as exc_info:
        await store.upload_stream(
            final_key,
            chunks(b"good"),
            content_type="text/plain",
            max_bytes=4,
        )

    assert exc_info.value.code == "OBJECT_VERIFICATION_FAILED"
    assert final_key not in client.objects
    assert all(not key.startswith("tmp/uploads/") for key in client.objects)


@pytest.mark.asyncio
async def test_publish_rejects_a_same_size_source_etag_race() -> None:
    client = BlockingFakeMinio()
    store = make_store(client)
    temp_key = "tmp/jobs/job/source-race/parsed/text.txt"
    final_key = "knowledge-bases/k/documents/d/versions/v/parsed/source-race.txt"
    body = b"good"
    checksum = hashlib.sha256(body).hexdigest()
    client.objects[temp_key] = FakeObject(body, {"sha256": checksum}, datetime.now(UTC))
    client.tamper_source_before_copy = b"evil"

    with pytest.raises(ObjectStoreError) as exc_info:
        await store.publish_temp(
            temp_key,
            final_key,
            expected_size=len(body),
            expected_checksum=checksum,
        )

    assert exc_info.value.code == "OBJECT_VERIFICATION_FAILED"
    assert final_key not in client.objects


@pytest.mark.asyncio
async def test_sdk_sized_aggregation_keeps_each_producer_handoff_bounded() -> None:
    client = SdkAggregatingFakeMinio()
    store = make_store(client, buffer_bytes=64 * 1024)
    payload = b"x" * ((5 * 1024 * 1024) + 257)

    stored = await store.upload_stream(
        "knowledge-bases/k/documents/d/versions/v/source/source.txt",
        chunks(payload),
        content_type="text/plain",
        max_bytes=len(payload),
    )

    assert client.max_single_pipe_read <= stored.producer_slice_bytes
    assert stored.max_pipe_buffered_bytes <= 64 * 1024
    assert stored.adapter_memory_bound_bytes == (
        (64 * 1024) + (2 * ((5 * 1024 * 1024) + 1)) + stored.producer_slice_bytes
    )


@pytest.mark.asyncio
async def test_oversize_stream_aborts_reader_upload_and_removes_staging_object() -> None:
    client = BlockingFakeMinio(read_size=4)
    store = make_store(client, buffer_bytes=8)

    with pytest.raises(UploadLimitExceeded) as exc_info:
        await store.upload_stream(
            "knowledge-bases/k/documents/d/versions/v/source/source.txt",
            chunks(b"12345678", b"9"),
            content_type="text/plain",
            max_bytes=8,
        )

    assert exc_info.value.code == "FILE_TOO_LARGE"
    assert client.abort_count == 1
    assert client.reader_finished.wait(1)
    assert client.objects == {}


@pytest.mark.asyncio
async def test_request_stream_failure_aborts_and_exposes_only_a_sanitized_error() -> None:
    client = BlockingFakeMinio(read_size=4)
    store = make_store(client, buffer_bytes=8)
    final_key = "knowledge-bases/private/documents/d/versions/v/source/source.txt"

    async def disconnected() -> AsyncIterator[bytes]:
        yield b"started"
        raise RuntimeError("client-disconnect-secret-body")

    with pytest.raises(ObjectStoreError) as exc_info:
        await store.upload_stream(
            final_key,
            disconnected(),
            content_type="text/plain",
            max_bytes=100,
        )

    error = exc_info.value
    assert error.code == "UPLOAD_STREAM_FAILED"
    surface = f"{error!s} {error!r}"
    assert "client-disconnect-secret-body" not in surface
    assert final_key not in surface
    assert "private-rag-bucket" not in surface
    assert client.abort_count == 1
    assert client.reader_finished.wait(1)


@pytest.mark.asyncio
async def test_cancellation_unblocks_the_sync_reader_aborts_and_reclaims_tasks() -> None:
    client = BlockingFakeMinio(read_size=1, read_delay=0.005)
    store = make_store(client, buffer_bytes=2)

    async def endless() -> AsyncIterator[bytes]:
        while True:
            yield b"x" * 64
            await asyncio.sleep(0)

    upload = asyncio.create_task(
        store.upload_stream(
            "knowledge-bases/k/documents/d/versions/v/source/source.txt",
            endless(),
            content_type="text/plain",
            max_bytes=10_000_000,
        )
    )
    await asyncio.to_thread(client.reader_started.wait, 1)
    upload.cancel()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(upload, timeout=2)

    assert client.reader_finished.wait(1)
    assert client.abort_count == 1
    await store.aclose()
    await asyncio.sleep(0)
    orphaned = [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]
    assert all(task.done() or "minio" not in repr(task.get_coro()).lower() for task in orphaned)


@pytest.mark.asyncio
async def test_cancellation_detaches_a_network_hang_and_eventually_cleans_staging() -> None:
    client = BlockingAfterReadFakeMinio()
    store = MinioObjectStore(
        client=client,
        bucket="private-rag-bucket",
        buffer_bytes=64,
        part_size_bytes=5 * 1024 * 1024,
        operation_timeout_seconds=1,
        cancel_grace_seconds=0.02,
    )
    upload = asyncio.create_task(
        store.upload_stream(
            "knowledge-bases/k/documents/d/versions/v/source/source.txt",
            chunks(b"network-hang"),
            content_type="text/plain",
            max_bytes=100,
        )
    )
    await asyncio.to_thread(client.network_hang_started.wait, 1)

    started = asyncio.get_running_loop().time()
    upload.cancel()
    try:
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(asyncio.shield(upload), timeout=0.2)
        assert asyncio.get_running_loop().time() - started < 0.2
        assert store.pending_operation_count >= 1
    finally:
        client.network_release.set()

    await store.aclose()
    assert store.pending_operation_count == 0
    assert all(not key.startswith("tmp/uploads/") for key in client.objects)


@pytest.mark.asyncio
async def test_upload_cancellation_does_not_wait_for_a_hung_staging_delete() -> None:
    client = BlockingPutAndRemoveFakeMinio()
    store = MinioObjectStore(
        client=client,
        bucket="private-rag-bucket",
        buffer_bytes=64,
        part_size_bytes=5 * 1024 * 1024,
        operation_timeout_seconds=1,
        cancel_grace_seconds=0.02,
    )
    upload = asyncio.create_task(
        store.upload_stream(
            "knowledge-bases/k/documents/d/versions/v/source/source.txt",
            chunks(b"hung-put-and-delete"),
            content_type="text/plain",
            max_bytes=100,
        )
    )
    await asyncio.to_thread(client.network_hang_started.wait, 1)

    started = asyncio.get_running_loop().time()
    upload.cancel()
    try:
        assert await asyncio.to_thread(client.delete_started.wait, 1)
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(asyncio.shield(upload), timeout=0.2)
        assert asyncio.get_running_loop().time() - started < 0.2
    finally:
        client.network_release.set()
        client.delete_release.set()
        with suppress(asyncio.CancelledError):
            await upload

    await store.aclose()
    assert store.pending_operation_count == 0
    assert all(not key.startswith("tmp/uploads/") for key in client.objects)


@pytest.mark.asyncio
async def test_close_retires_async_wrappers_when_sdk_calls_remain_blocked() -> None:
    client = BlockingPutAndRemoveFakeMinio()
    store = MinioObjectStore(
        client=client,
        bucket="private-rag-bucket",
        buffer_bytes=64,
        part_size_bytes=5 * 1024 * 1024,
        operation_timeout_seconds=0.05,
        cancel_grace_seconds=0.01,
    )
    upload = asyncio.create_task(
        store.upload_stream(
            "knowledge-bases/k/documents/d/versions/v/source/permanent-hang.txt",
            chunks(b"permanent-hang"),
            content_type="text/plain",
            max_bytes=100,
        )
    )
    await asyncio.to_thread(client.network_hang_started.wait, 1)
    upload.cancel()
    with suppress(asyncio.CancelledError):
        await asyncio.wait_for(asyncio.shield(upload), timeout=0.2)
    assert await asyncio.to_thread(client.delete_started.wait, 1)

    started = asyncio.get_running_loop().time()
    try:
        with pytest.raises(ObjectStoreError) as close_error:
            await asyncio.wait_for(store.aclose(), timeout=0.2)
        assert close_error.value.code == "OBJECT_STORE_UNAVAILABLE"
        assert asyncio.get_running_loop().time() - started < 0.2
        assert store.pending_operation_count == 0
        await asyncio.sleep(0)
        live_minio_tasks = [
            task
            for task in asyncio.all_tasks()
            if task is not asyncio.current_task()
            and not task.done()
            and task.get_name().startswith("minio-")
        ]
        assert live_minio_tasks == []

        with pytest.raises(ObjectStoreError) as closed_error:
            await store.verify_object(
                "knowledge-bases/k/documents/d/versions/v/source/permanent-hang.txt",
                expected_size=1,
                expected_checksum="0" * 64,
            )
        assert closed_error.value.code == "OBJECT_STORE_UNAVAILABLE"
        assert closed_error.value.retryable is False
    finally:
        client.network_release.set()
        client.delete_release.set()
        await asyncio.to_thread(client.delete_finished.wait, 1)
        with suppress(ObjectStoreError):
            await store.aclose()


@pytest.mark.asyncio
async def test_close_deadline_retires_cleanup_spawned_by_cancelled_publish() -> None:
    client = BlockingHeadAndRemoveFakeMinio()
    store = MinioObjectStore(
        client=client,
        bucket="private-rag-bucket",
        buffer_bytes=64,
        part_size_bytes=5 * 1024 * 1024,
        operation_timeout_seconds=0.1,
        cancel_grace_seconds=0.01,
    )
    temp_key = "tmp/jobs/job/close-deadline/parsed/text.txt"
    final_key = "knowledge-bases/k/documents/d/versions/v/parsed/close-deadline.txt"
    body = b"close deadline"
    checksum = hashlib.sha256(body).hexdigest()
    client.objects[temp_key] = FakeObject(
        body,
        {"sha256": checksum},
        datetime.now(UTC),
        "text/plain",
    )
    publish: asyncio.Task[PublishedObject] | None = None
    close: asyncio.Task[None] | None = None
    lifecycle_lock_held = False
    try:
        await store._lifecycle_lock.acquire()
        lifecycle_lock_held = True
        publish = asyncio.create_task(
            store.publish_temp(
                temp_key,
                final_key,
                expected_size=len(body),
                expected_checksum=checksum,
            )
        )
        await asyncio.sleep(0)
        close = asyncio.create_task(store.aclose())
        await asyncio.sleep(0.05)
        store._lifecycle_lock.release()
        lifecycle_lock_held = False
        assert await asyncio.to_thread(client.operation_started.wait, 1)

        started = asyncio.get_running_loop().time()
        with pytest.raises(ObjectStoreError) as close_error:
            await asyncio.wait_for(close, timeout=0.2)
        assert close_error.value.code == "OBJECT_STORE_UNAVAILABLE"
        assert asyncio.get_running_loop().time() - started < 0.2
        assert store.pending_operation_count == 0
        live_minio_tasks = [
            task
            for task in asyncio.all_tasks()
            if task is not asyncio.current_task()
            and not task.done()
            and task.get_name().startswith("minio-")
        ]
        assert live_minio_tasks == []
    finally:
        if lifecycle_lock_held:
            store._lifecycle_lock.release()
        client.operation_release.set()
        client.delete_release.set()
        pending = [task for task in (publish, close) if task is not None]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        if client.delete_started.is_set():
            await asyncio.to_thread(client.delete_finished.wait, 1)
        with suppress(ObjectStoreError):
            await store.aclose()


@pytest.mark.asyncio
async def test_failed_close_keeps_a_cancellation_resistant_producer_tracked() -> None:
    client = BlockingFakeMinio()
    store = MinioObjectStore(
        client=client,
        bucket="private-rag-bucket",
        buffer_bytes=64,
        part_size_bytes=5 * 1024 * 1024,
        operation_timeout_seconds=0.05,
        cancel_grace_seconds=0.01,
    )
    stream = CancellationResistantClosingStream(b"resist close")
    upload = asyncio.create_task(
        store.upload_stream(
            "knowledge-bases/k/documents/d/versions/v/source/resist-close.txt",
            stream,
            content_type="text/plain",
            max_bytes=100,
        )
    )
    await asyncio.wait_for(stream.close_started.wait(), timeout=1)

    try:
        with pytest.raises(ObjectStoreError) as close_error:
            await asyncio.wait_for(store.aclose(), timeout=0.2)
        assert close_error.value.code == "OBJECT_STORE_UNAVAILABLE"
        tracked_producers = [
            task
            for task in store._background_tasks
            if not task.done() and task.get_name() == "minio-upload-producer"
        ]
        assert len(tracked_producers) == 1
        assert tracked_producers[0] in asyncio.all_tasks()
    finally:
        stream.close_release.set()
        await asyncio.wait_for(stream.close_finished.wait(), timeout=1)
        await asyncio.gather(upload, return_exceptions=True)
        with suppress(ObjectStoreError):
            await store.aclose()


@pytest.mark.asyncio
async def test_close_failure_is_sticky_while_a_cancelled_sdk_call_can_still_mutate() -> None:
    client = BlockingPutAndRemoveFakeMinio()
    store = MinioObjectStore(
        client=client,
        bucket="private-rag-bucket",
        buffer_bytes=64,
        part_size_bytes=5 * 1024 * 1024,
        operation_timeout_seconds=0.05,
        cancel_grace_seconds=0.01,
    )
    object_key = "tmp/jobs/job/sticky-close/parsed/text.txt"
    client.objects[object_key] = FakeObject(b"sticky", {}, datetime.now(UTC))
    delete = asyncio.create_task(store.delete_best_effort(object_key))
    await asyncio.to_thread(client.delete_started.wait, 1)

    try:
        with pytest.raises(ObjectStoreError) as first_close_error:
            await asyncio.wait_for(store.aclose(), timeout=0.2)
        assert first_close_error.value.code == "OBJECT_STORE_UNAVAILABLE"
        assert object_key in client.objects

        with pytest.raises(ObjectStoreError) as second_close_error:
            await asyncio.wait_for(store.aclose(), timeout=0.2)
        assert second_close_error.value.code == "OBJECT_STORE_UNAVAILABLE"
        assert object_key in client.objects
    finally:
        client.delete_release.set()
        await asyncio.to_thread(client.delete_finished.wait, 1)
        await asyncio.gather(delete, return_exceptions=True)
        with suppress(ObjectStoreError):
            await store.aclose()


@pytest.mark.asyncio
async def test_eventual_cleanup_runs_when_the_detached_sdk_call_later_fails() -> None:
    client = BlockingAfterReadFakeMinio()
    client.raise_after_store = True
    store = MinioObjectStore(
        client=client,
        bucket="private-rag-bucket",
        buffer_bytes=64,
        part_size_bytes=5 * 1024 * 1024,
        operation_timeout_seconds=1,
        cancel_grace_seconds=0.02,
    )
    upload = asyncio.create_task(
        store.upload_stream(
            "knowledge-bases/k/documents/d/versions/v/source/source.txt",
            chunks(b"late-failure"),
            content_type="text/plain",
            max_bytes=100,
        )
    )
    await asyncio.to_thread(client.network_hang_started.wait, 1)
    upload.cancel()
    try:
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(asyncio.shield(upload), timeout=0.2)
    finally:
        client.network_release.set()

    await store.aclose()
    assert all(not key.startswith("tmp/uploads/") for key in client.objects)


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["stat", "delete", "list"])
async def test_other_blocking_sdk_operations_cancel_within_the_bound_and_are_drained(
    operation: str,
) -> None:
    client = BlockingOperationFakeMinio(operation)
    store = MinioObjectStore(
        client=client,
        bucket="private-rag-bucket",
        buffer_bytes=64,
        part_size_bytes=5 * 1024 * 1024,
        operation_timeout_seconds=1,
        cancel_grace_seconds=0.02,
    )
    key = "tmp/jobs/job/blocking/parsed/text.txt"
    body = b"blocking"
    checksum = hashlib.sha256(body).hexdigest()
    client.objects[key] = FakeObject(body, {"sha256": checksum}, datetime.now(UTC))
    coroutine: Coroutine[Any, Any, object]
    if operation == "stat":
        coroutine = store.verify_object(
            key,
            expected_size=len(body),
            expected_checksum=checksum,
        )
    elif operation == "delete":
        coroutine = store.delete_best_effort(key)
    else:
        coroutine = store.list_older_than(
            prefix="tmp/jobs/",
            older_than=datetime.now(UTC) + timedelta(seconds=1),
            limit=1,
        )
    task = asyncio.create_task(coroutine)
    await asyncio.to_thread(client.operation_started.wait, 1)
    task.cancel()
    try:
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(asyncio.shield(task), timeout=0.2)
        assert store.pending_operation_count == 1
    finally:
        client.operation_release.set()

    await store.aclose()
    assert store.pending_operation_count == 0


@pytest.mark.asyncio
async def test_cancellation_during_conditional_copy_is_not_reclassified_as_storage_failure() -> (
    None
):
    client = BlockingConditionalCopyFakeMinio()
    store = make_store(client)
    temp_key = "tmp/jobs/job/cancel-copy/parsed/text.txt"
    final_key = "knowledge-bases/k/documents/d/versions/v/parsed/cancel-copy.txt"
    body = b"cancel copy"
    checksum = hashlib.sha256(body).hexdigest()
    client.objects[temp_key] = FakeObject(body, {"sha256": checksum}, datetime.now(UTC))
    publish = asyncio.create_task(
        store.publish_temp(
            temp_key,
            final_key,
            expected_size=len(body),
            expected_checksum=checksum,
        )
    )
    await asyncio.to_thread(client.copy_started.wait, 1)

    publish.cancel()
    started = asyncio.get_running_loop().time()
    try:
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(asyncio.shield(publish), timeout=0.2)
        assert asyncio.get_running_loop().time() - started < 0.2
        assert store.pending_operation_count >= 1
    finally:
        client.copy_release.set()

    await store.aclose()
    assert client.copy_finished.wait(1)
    assert temp_key not in client.objects
    assert final_key not in client.objects


@pytest.mark.asyncio
async def test_timed_out_conditional_copy_removes_a_late_successful_destination() -> None:
    client = BlockingConditionalCopyFakeMinio()
    store = MinioObjectStore(
        client=client,
        bucket="private-rag-bucket",
        buffer_bytes=64,
        part_size_bytes=5 * 1024 * 1024,
        operation_timeout_seconds=0.05,
        cancel_grace_seconds=0.01,
    )
    temp_key = "tmp/jobs/job/timeout-copy/parsed/text.txt"
    final_key = "knowledge-bases/k/documents/d/versions/v/parsed/timeout-copy.txt"
    body = b"timeout copy"
    checksum = hashlib.sha256(body).hexdigest()
    client.objects[temp_key] = FakeObject(body, {"sha256": checksum}, datetime.now(UTC))
    publish = asyncio.create_task(
        store.publish_temp(
            temp_key,
            final_key,
            expected_size=len(body),
            expected_checksum=checksum,
        )
    )
    await asyncio.to_thread(client.copy_started.wait, 1)

    try:
        with pytest.raises(ObjectStoreError) as exc_info:
            await publish
        assert exc_info.value.code == "OBJECT_STORE_UNAVAILABLE"
    finally:
        client.copy_release.set()

    await store.aclose()
    assert client.copy_finished.wait(1)
    assert temp_key not in client.objects
    assert final_key not in client.objects


@pytest.mark.asyncio
async def test_cancelled_copy_never_removes_a_destination_created_by_a_racer() -> None:
    client = BlockingConditionalCopyFakeMinio()
    store = make_store(client)
    temp_key = "tmp/jobs/job/cancel-copy-race/parsed/text.txt"
    final_key = "knowledge-bases/k/documents/d/versions/v/parsed/cancel-copy-race.txt"
    body = b"cancel copy race"
    checksum = hashlib.sha256(body).hexdigest()
    racer = FakeObject(
        b"racer",
        {"sha256": hashlib.sha256(b"racer").hexdigest()},
        datetime.now(UTC),
    )
    client.objects[temp_key] = FakeObject(body, {"sha256": checksum}, datetime.now(UTC))
    client.copy_race = racer
    publish = asyncio.create_task(
        store.publish_temp(
            temp_key,
            final_key,
            expected_size=len(body),
            expected_checksum=checksum,
        )
    )
    await asyncio.to_thread(client.copy_started.wait, 1)

    publish.cancel()
    try:
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(asyncio.shield(publish), timeout=0.2)
    finally:
        client.copy_release.set()

    await store.aclose()
    assert client.objects[final_key] is racer
    assert final_key not in client.remove_calls
    assert temp_key not in client.objects


@pytest.mark.asyncio
async def test_publish_cancellation_does_not_wait_for_a_hung_temp_delete() -> None:
    client = BlockingCopyAndRemoveFakeMinio()
    store = MinioObjectStore(
        client=client,
        bucket="private-rag-bucket",
        buffer_bytes=64,
        part_size_bytes=5 * 1024 * 1024,
        operation_timeout_seconds=1,
        cancel_grace_seconds=0.02,
    )
    temp_key = "tmp/jobs/job/cancel-copy-delete/parsed/text.txt"
    final_key = "knowledge-bases/k/documents/d/versions/v/parsed/cancel-copy-delete.txt"
    body = b"cancel copy and delete"
    checksum = hashlib.sha256(body).hexdigest()
    client.objects[temp_key] = FakeObject(body, {"sha256": checksum}, datetime.now(UTC))
    publish = asyncio.create_task(
        store.publish_temp(
            temp_key,
            final_key,
            expected_size=len(body),
            expected_checksum=checksum,
        )
    )
    await asyncio.to_thread(client.copy_started.wait, 1)

    started = asyncio.get_running_loop().time()
    publish.cancel()
    try:
        assert await asyncio.to_thread(client.delete_started.wait, 1)
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(asyncio.shield(publish), timeout=0.2)
        assert asyncio.get_running_loop().time() - started < 0.2
    finally:
        client.copy_release.set()
        client.delete_release.set()
        with suppress(asyncio.CancelledError):
            await publish

    await store.aclose()
    assert store.pending_operation_count == 0
    assert temp_key not in client.objects


@pytest.mark.asyncio
async def test_sync_consumer_failure_stops_the_producer_without_deadlock() -> None:
    client = BlockingFakeMinio(read_size=2)
    client.put_error_after_reads = 2
    store = make_store(client, buffer_bytes=4)
    producer_closed = asyncio.Event()

    async def producing() -> AsyncIterator[bytes]:
        try:
            while True:
                yield b"abcdefgh"
                await asyncio.sleep(0)
        finally:
            producer_closed.set()

    with pytest.raises(ObjectStoreError) as exc_info:
        await asyncio.wait_for(
            store.upload_stream(
                "knowledge-bases/k/documents/d/versions/v/source/source.txt",
                producing(),
                content_type="text/plain",
                max_bytes=10_000,
            ),
            timeout=2,
        )

    assert exc_info.value.code == "OBJECT_STORE_UNAVAILABLE"
    assert "minio-secret-from-consumer" not in repr(exc_info.value)
    assert producer_closed.is_set()


@pytest.mark.asyncio
async def test_normal_early_consumer_stop_closes_the_producer_and_rejects_partial_upload() -> None:
    client = EarlyStopFakeMinio()
    store = make_store(client, buffer_bytes=4)
    producer_closed = asyncio.Event()

    async def producing() -> AsyncIterator[bytes]:
        try:
            yield b"abcdefgh"
            yield b"ijklmnop"
        finally:
            producer_closed.set()

    with pytest.raises(ObjectStoreError):
        await asyncio.wait_for(
            store.upload_stream(
                "knowledge-bases/k/documents/d/versions/v/source/source.txt",
                producing(),
                content_type="text/plain",
                max_bytes=100,
            ),
            timeout=1,
        )

    assert producer_closed.is_set()
    assert all(not key.startswith("tmp/uploads/") for key in client.objects)


@pytest.mark.asyncio
async def test_real_sdk_aborts_created_multipart_upload_after_reader_failure() -> None:
    client = MultipartAbortProbeMinio()
    store = MinioObjectStore(
        client=cast(Any, client),
        bucket="private-rag-bucket",
        buffer_bytes=64 * 1024,
        part_size_bytes=5 * 1024 * 1024,
    )

    async def disconnected_after_first_part() -> AsyncIterator[bytes]:
        yield b"x" * ((5 * 1024 * 1024) + 1)
        raise ConnectionError("client disconnected")

    with pytest.raises(ObjectStoreError) as exc_info:
        await store.upload_stream(
            "knowledge-bases/k/documents/d/versions/v/source/source.txt",
            disconnected_after_first_part(),
            content_type="text/plain",
            max_bytes=6 * 1024 * 1024,
        )

    assert exc_info.value.code == "UPLOAD_STREAM_FAILED"
    assert client.created_upload_id == "real-sdk-upload-id"
    assert client.uploaded_parts == [1]
    assert client.aborted_upload_ids == ["real-sdk-upload-id"]


@pytest.mark.asyncio
async def test_upload_stream_accepts_exact_50_mib_and_rejects_the_next_byte() -> None:
    client = BlockingFakeMinio(read_size=64 * 1024)
    store = make_store(client, buffer_bytes=64 * 1024)
    mebibyte = b"x" * (1024 * 1024)
    exact_key = "knowledge-bases/k/documents/d/versions/v/source/exact.txt"

    exact = await store.upload_stream(
        exact_key,
        chunks(*(mebibyte for _ in range(50))),
        content_type="text/plain",
        max_bytes=50 * 1024 * 1024,
    )

    assert exact.size == 50 * 1024 * 1024
    assert client.objects[exact_key].body == mebibyte * 50
    oversized_key = "knowledge-bases/k/documents/d/versions/v/source/oversized.txt"
    with pytest.raises(UploadLimitExceeded):
        await store.upload_stream(
            oversized_key,
            chunks(*(mebibyte for _ in range(50)), b"!"),
            content_type="text/plain",
            max_bytes=50 * 1024 * 1024,
        )

    assert oversized_key not in client.objects
    assert all(not key.startswith("tmp/uploads/") for key in client.objects)


@pytest.mark.asyncio
async def test_head_verification_requires_exact_size_and_sha256_metadata() -> None:
    client = BlockingFakeMinio()
    store = make_store(client)
    key = "tmp/jobs/job/1/parsed/text.txt"
    body = b"verified"
    checksum = hashlib.sha256(body).hexdigest()
    client.objects[key] = FakeObject(body, {"sha256": checksum}, datetime.now(UTC))

    verified = await store.verify_object(key, expected_size=len(body), expected_checksum=checksum)

    assert verified.size == len(body)
    assert verified.checksum_sha256 == checksum

    client.objects[key].metadata["sha256"] = "0" * 64
    with pytest.raises(ObjectStoreError) as exc_info:
        await store.verify_object(key, expected_size=len(body), expected_checksum=checksum)
    assert exc_info.value.code == "OBJECT_VERIFICATION_FAILED"
    assert key not in repr(exc_info.value)


@pytest.mark.asyncio
async def test_publish_copies_if_absent_verifies_and_deletes_temp() -> None:
    client = BlockingFakeMinio()
    store = make_store(client)
    temp_key = "tmp/jobs/job/2/parsed/text.txt"
    final_key = "knowledge-bases/k/documents/d/versions/v/parsed/text.txt"
    body = b"canonical parsed text\n"
    checksum = hashlib.sha256(body).hexdigest()
    client.objects[temp_key] = FakeObject(body, {"sha256": checksum}, datetime.now(UTC))

    published = await store.publish_temp(
        temp_key,
        final_key,
        expected_size=len(body),
        expected_checksum=checksum,
    )

    assert published.object_key == final_key
    assert published.reused is False
    assert final_key not in repr(published)
    assert client.objects[final_key].body == body
    assert client.objects[final_key].metadata == {"sha256": checksum}
    assert temp_key not in client.objects
    assert (
        client.copy_calls[0]["source_if_match"]
        == hashlib.md5(
            body,
            usedforsecurity=False,
        ).hexdigest()
    )
    assert client.copy_calls[-1]["if_none_match"] == "*"


@pytest.mark.asyncio
async def test_publish_preserves_the_verified_source_content_type() -> None:
    client = BlockingFakeMinio()
    store = make_store(client)
    temp_key = "tmp/jobs/job/content-type/parsed/text.md"
    final_key = "knowledge-bases/k/documents/d/versions/v/parsed/text.md"
    body = b"# heading"
    checksum = hashlib.sha256(body).hexdigest()
    client.objects[temp_key] = FakeObject(
        body,
        {"sha256": checksum},
        datetime.now(UTC),
        "text/markdown",
    )

    await store.publish_temp(
        temp_key,
        final_key,
        expected_size=len(body),
        expected_checksum=checksum,
    )

    assert client.objects[final_key].content_type == "text/markdown"


@pytest.mark.asyncio
async def test_publish_reuses_same_checksum_and_conflicts_without_overwrite() -> None:
    client = BlockingFakeMinio()
    store = make_store(client)
    temp_key = "tmp/jobs/job/3/chunks/recursive_text_v1.jsonl"
    final_key = "knowledge-bases/k/documents/d/versions/v/chunks/recursive_text_v1.jsonl"
    body = b'{"chunk_count":0}\n'
    checksum = hashlib.sha256(body).hexdigest()
    client.objects[temp_key] = FakeObject(body, {"sha256": checksum}, datetime.now(UTC))
    client.objects[final_key] = FakeObject(body, {"sha256": checksum}, datetime.now(UTC))

    reused = await store.publish_temp(
        temp_key,
        final_key,
        expected_size=len(body),
        expected_checksum=checksum,
    )

    assert reused.reused is True
    assert client.copy_calls == []
    assert temp_key not in client.objects

    conflicting_temp = "tmp/jobs/job/4/chunks/recursive_text_v1.jsonl"
    client.objects[conflicting_temp] = FakeObject(
        body,
        {"sha256": checksum},
        datetime.now(UTC),
    )
    client.objects[final_key] = FakeObject(
        b"different",
        {"sha256": hashlib.sha256(b"different").hexdigest()},
        datetime.now(UTC),
    )

    with pytest.raises(ArtifactChecksumConflict) as exc_info:
        await store.publish_temp(
            conflicting_temp,
            final_key,
            expected_size=len(body),
            expected_checksum=checksum,
        )

    assert exc_info.value.code == "ARTIFACT_CHECKSUM_CONFLICT"
    assert client.objects[final_key].body == b"different"
    assert conflicting_temp not in client.objects
    assert final_key not in repr(exc_info.value)


@pytest.mark.asyncio
async def test_publish_rechecks_a_precondition_race_and_reports_checksum_conflict() -> None:
    client = BlockingFakeMinio()
    store = make_store(client)
    temp_key = "tmp/jobs/job/5/parsed/text.txt"
    final_key = "knowledge-bases/k/documents/d/versions/v/parsed/text.txt"
    body = b"ours"
    checksum = hashlib.sha256(body).hexdigest()
    client.objects[temp_key] = FakeObject(body, {"sha256": checksum}, datetime.now(UTC))
    client.copy_race = FakeObject(
        b"theirs",
        {"sha256": hashlib.sha256(b"theirs").hexdigest()},
        datetime.now(UTC),
    )

    with pytest.raises(ArtifactChecksumConflict):
        await store.publish_temp(
            temp_key,
            final_key,
            expected_size=len(body),
            expected_checksum=checksum,
        )

    assert client.objects[final_key].body == b"theirs"
    assert temp_key not in client.objects


@pytest.mark.asyncio
async def test_real_minio_private_copy_path_is_version_guarded_and_conditionally_writes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = PrivateExecuteFakeMinio()
    monkeypatch.setattr(minio_store.minio, "__version__", "7.2.20")
    store = make_store(client)
    temp_key = "tmp/jobs/job/private/parsed/text.txt"
    final_key = "knowledge-bases/k/documents/d/versions/v/parsed/private.txt"
    body = b"private execute"
    checksum = hashlib.sha256(body).hexdigest()
    client.objects[temp_key] = FakeObject(body, {"sha256": checksum}, datetime.now(UTC))

    published = await store.publish_temp(
        temp_key,
        final_key,
        expected_size=len(body),
        expected_checksum=checksum,
    )

    assert published.reused is False
    assert client.execute_headers is not None
    assert client.execute_headers["If-None-Match"] == "*"
    assert client.execute_query is not None
    assert set(client.execute_query) == {"uploadId"}
    assert (
        client.copy_calls[0]["source_if_match"]
        == hashlib.md5(
            body,
            usedforsecurity=False,
        ).hexdigest()
    )
    assert client.copy_calls[-1]["if_none_match"] == "*"


@pytest.mark.asyncio
async def test_real_minio_private_copy_path_fails_closed_on_api_version_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = PrivateExecuteFakeMinio()
    monkeypatch.setattr(minio_store.minio, "__version__", "7.2.20.post1")
    store = make_store(client)
    temp_key = "tmp/jobs/job/drift/parsed/text.txt"
    final_key = "knowledge-bases/k/documents/d/versions/v/parsed/drift.txt"
    body = b"api drift"
    checksum = hashlib.sha256(body).hexdigest()
    client.objects[temp_key] = FakeObject(body, {"sha256": checksum}, datetime.now(UTC))

    with pytest.raises(ObjectStoreError) as exc_info:
        await store.publish_temp(
            temp_key,
            final_key,
            expected_size=len(body),
            expected_checksum=checksum,
        )

    assert exc_info.value.code == "OBJECT_STORE_API_INCOMPATIBLE"
    assert client.execute_headers is None


@pytest.mark.asyncio
async def test_non_atomic_public_copy_fallback_is_rejected() -> None:
    final_key = "knowledge-bases/k/documents/d/versions/v/parsed/fallback.txt"
    client = PublicOnlyRaceFakeMinio(final_key)
    store = make_store(client)
    body = b"same canonical artifact"
    checksum = hashlib.sha256(body).hexdigest()
    temp_key = "tmp/jobs/job/fallback-1/parsed/text.txt"
    client.objects[temp_key] = FakeObject(body, {"sha256": checksum}, datetime.now(UTC))

    with pytest.raises(ObjectStoreError) as exc_info:
        await store.publish_temp(
            temp_key,
            final_key,
            expected_size=len(body),
            expected_checksum=checksum,
        )

    assert exc_info.value.code == "OBJECT_STORE_API_INCOMPATIBLE"
    assert final_key not in client.objects


@pytest.mark.asyncio
async def test_private_api_guard_checks_parameter_kind_order_and_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = WrongSignatureFakeMinio()
    monkeypatch.setattr(minio_store.minio, "__version__", "7.2.20")
    store = make_store(client)
    temp_key = "tmp/jobs/job/signature/parsed/text.txt"
    final_key = "knowledge-bases/k/documents/d/versions/v/parsed/signature.txt"
    body = b"signature guard"
    checksum = hashlib.sha256(body).hexdigest()
    client.objects[temp_key] = FakeObject(body, {"sha256": checksum}, datetime.now(UTC))

    with pytest.raises(ObjectStoreError) as exc_info:
        await store.publish_temp(
            temp_key,
            final_key,
            expected_size=len(body),
            expected_checksum=checksum,
        )

    assert exc_info.value.code == "OBJECT_STORE_API_INCOMPATIBLE"
    assert inspect.signature(client._execute).parameters["preload_content"].default is False


@pytest.mark.asyncio
async def test_concurrent_fake_publish_uses_server_side_precondition_not_an_instance_lock() -> None:
    final_key = "knowledge-bases/k/documents/d/versions/v/parsed/fallback.txt"
    client = BlockingFakeMinio()
    first_store = make_store(client)
    second_store = make_store(client)
    body = b"same canonical artifact"
    checksum = hashlib.sha256(body).hexdigest()
    first_temp = "tmp/jobs/job/fallback-1/parsed/text.txt"
    second_temp = "tmp/jobs/job/fallback-2/parsed/text.txt"
    client.objects[first_temp] = FakeObject(body, {"sha256": checksum}, datetime.now(UTC))
    client.objects[second_temp] = FakeObject(body, {"sha256": checksum}, datetime.now(UTC))

    first, second = await asyncio.gather(
        first_store.publish_temp(
            first_temp,
            final_key,
            expected_size=len(body),
            expected_checksum=checksum,
        ),
        second_store.publish_temp(
            second_temp,
            final_key,
            expected_size=len(body),
            expected_checksum=checksum,
        ),
    )

    assert {first.reused, second.reused} == {False, True}
    assert client.objects[final_key].body == body


@pytest.mark.asyncio
async def test_temp_cleanup_is_best_effort_after_successful_publish() -> None:
    client = BlockingFakeMinio()
    store = make_store(client)
    temp_key = "tmp/jobs/job/6/parsed/text.txt"
    final_key = "knowledge-bases/k/documents/d/versions/v/parsed/text.txt"
    body = b"published"
    checksum = hashlib.sha256(body).hexdigest()
    client.objects[temp_key] = FakeObject(body, {"sha256": checksum}, datetime.now(UTC))
    client.remove_error = PermissionError("cleanup-secret")

    published = await store.publish_temp(
        temp_key,
        final_key,
        expected_size=len(body),
        expected_checksum=checksum,
    )

    assert published.object_key == final_key
    assert final_key in client.objects
    assert temp_key in client.objects


@pytest.mark.asyncio
async def test_orphan_listing_is_prefix_age_and_page_bounded() -> None:
    client = BlockingFakeMinio()
    store = make_store(client)
    now = datetime.now(UTC)
    for index in range(5):
        client.objects[f"tmp/jobs/job/{index}/artifact"] = FakeObject(
            str(index).encode(),
            {"sha256": hashlib.sha256(str(index).encode()).hexdigest()},
            now - timedelta(hours=30 + index),
        )
    client.objects["tmp/jobs/job/recent/artifact"] = FakeObject(
        b"recent",
        {"sha256": hashlib.sha256(b"recent").hexdigest()},
        now - timedelta(minutes=5),
    )
    client.objects["knowledge-bases/not-temp"] = FakeObject(
        b"other",
        {"sha256": hashlib.sha256(b"other").hexdigest()},
        now - timedelta(days=3),
    )

    page = await store.list_older_than(
        prefix="tmp/jobs/",
        older_than=now - timedelta(hours=24),
        limit=2,
    )

    assert len(page.items) == 2
    assert all(item.object_key.startswith("tmp/jobs/") for item in page.items)
    assert all(item.last_modified < now - timedelta(hours=24) for item in page.items)
    assert page.next_start_after == page.items[-1].object_key


@pytest.mark.asyncio
async def test_orphan_listing_advances_after_a_bounded_page_with_no_old_objects() -> None:
    client = BlockingFakeMinio()
    store = make_store(client)
    now = datetime.now(UTC)
    for index in range(9):
        client.objects[f"tmp/jobs/recent/{index:02d}/artifact"] = FakeObject(
            str(index).encode(),
            {"sha256": hashlib.sha256(str(index).encode()).hexdigest()},
            now - timedelta(minutes=5),
        )

    page = await store.list_older_than(
        prefix="tmp/jobs/",
        older_than=now - timedelta(hours=24),
        limit=1,
    )

    assert page.items == ()
    assert page.next_start_after == "tmp/jobs/recent/07/artifact"
    assert page.next_start_after not in repr(page)


@pytest.mark.asyncio
async def test_object_store_rejects_control_characters_in_object_keys() -> None:
    client = BlockingFakeMinio()
    store = make_store(client)

    with pytest.raises(ValueError, match="upload request is invalid"):
        await store.upload_stream(
            "tmp/jobs/job/unsafe\nkey",
            chunks(b"content"),
            content_type="text/plain",
            max_bytes=100,
        )

    assert client.put_calls == []


def test_private_conditional_copy_header_source_is_canonical() -> None:
    source = CopySource("private-rag-bucket", "tmp/jobs/job/7/parsed/text.txt")
    header = source.gen_copy_headers()["x-amz-copy-source"]

    assert unquote(header) == "/private-rag-bucket/tmp/jobs/job/7/parsed/text.txt"
