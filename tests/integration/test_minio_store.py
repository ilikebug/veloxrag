from __future__ import annotations

import asyncio
import hashlib
import threading
from collections.abc import AsyncIterator, Iterator
from io import BytesIO
from typing import Any, Protocol, cast
from uuid import uuid4

import pytest
from minio import Minio
from minio.datatypes import Object
from minio.sse import SseCustomerKey

from rag_service.infrastructure.minio_store import (
    ArtifactChecksumConflict,
    MinioObjectStore,
    PublishedObject,
)

pytestmark = pytest.mark.integration

type MinioHeaderValue = str | list[str] | tuple[str]
type MinioHeaders = dict[str, MinioHeaderValue]


class _MinioConnection(Protocol):
    endpoint: str
    access_key: str
    secret_key: str
    secure: bool


class _BarrierMinio(Minio):
    def __init__(
        self,
        endpoint: str,
        *,
        access_key: str,
        secret_key: str,
        secure: bool,
        final_key: str,
        initial_head_barrier: threading.Barrier,
    ) -> None:
        super().__init__(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure,
        )
        self._test_final_key = final_key
        self._test_initial_head_barrier = initial_head_barrier
        self._test_head_count = 0
        self._test_head_lock = threading.Lock()

    def stat_object(
        self,
        bucket_name: str,
        object_name: str,
        ssec: SseCustomerKey | None = None,
        version_id: str | None = None,
        extra_headers: MinioHeaders | None = None,
        extra_query_params: MinioHeaders | None = None,
    ) -> Object:
        if object_name == self._test_final_key:
            with self._test_head_lock:
                self._test_head_count += 1
                initial_head = self._test_head_count == 1
            if initial_head:
                self._test_initial_head_barrier.wait(timeout=15)
        return super().stat_object(
            bucket_name,
            object_name,
            ssec,
            version_id,
            extra_headers,
            extra_query_params,
        )


class _BlindInitialHeadMinio(Minio):
    def __init__(
        self,
        endpoint: str,
        *,
        access_key: str,
        secret_key: str,
        secure: bool,
        final_key: str,
    ) -> None:
        super().__init__(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure,
        )
        self._test_final_key = final_key
        self._test_blind = True

    def stat_object(
        self,
        bucket_name: str,
        object_name: str,
        ssec: SseCustomerKey | None = None,
        version_id: str | None = None,
        extra_headers: MinioHeaders | None = None,
        extra_query_params: MinioHeaders | None = None,
    ) -> Object:
        if object_name == self._test_final_key and self._test_blind:
            self._test_blind = False
            return super().stat_object(
                bucket_name,
                f"missing-{uuid4().hex}",
                ssec,
                version_id,
                extra_headers,
                extra_query_params,
            )
        return super().stat_object(
            bucket_name,
            object_name,
            ssec,
            version_id,
            extra_headers,
            extra_query_params,
        )


@pytest.fixture
def real_minio(
    minio_connection: _MinioConnection,
) -> Iterator[tuple[Minio, str]]:
    client = Minio(
        minio_connection.endpoint,
        access_key=minio_connection.access_key,
        secret_key=minio_connection.secret_key,
        secure=minio_connection.secure,
    )
    bucket = f"task7-{uuid4().hex}"
    client.make_bucket(bucket)
    try:
        yield client, bucket
    finally:
        for item in client.list_objects(bucket, recursive=True):
            client.remove_object(bucket, item.object_name)
        client.remove_bucket(bucket)


def _put(client: Minio, bucket: str, key: str, body: bytes) -> str:
    checksum = hashlib.sha256(body).hexdigest()
    client.put_object(
        bucket,
        key,
        BytesIO(body),
        len(body),
        metadata={"sha256": checksum},
    )
    return checksum


def _read(client: Minio, bucket: str, key: str) -> bytes:
    response = client.get_object(bucket, key)
    try:
        return response.read()
    finally:
        response.close()
        response.release_conn()


async def _chunks(*values: bytes) -> AsyncIterator[bytes]:
    for value in values:
        yield value


@pytest.mark.asyncio
async def test_real_minio_publish_never_overwrites_an_existing_destination(
    real_minio: tuple[Minio, str],
    minio_connection: _MinioConnection,
) -> None:
    setup_client, bucket = real_minio
    temp_key = "tmp/jobs/atomic-existing/parsed/text.txt"
    final_key = "knowledge-bases/k/documents/d/versions/v/parsed/text.txt"
    ours = b"new canonical artifact"
    existing = b"existing immutable artifact"
    ours_checksum = _put(setup_client, bucket, temp_key, ours)
    existing_checksum = _put(setup_client, bucket, final_key, existing)
    client = _BlindInitialHeadMinio(
        minio_connection.endpoint,
        access_key=minio_connection.access_key,
        secret_key=minio_connection.secret_key,
        secure=minio_connection.secure,
        final_key=final_key,
    )
    store = MinioObjectStore(
        client=cast(Any, client),
        bucket=bucket,
        buffer_bytes=64 * 1024,
        part_size_bytes=5 * 1024 * 1024,
    )

    with pytest.raises(ArtifactChecksumConflict):
        await store.publish_temp(
            temp_key,
            final_key,
            expected_size=len(ours),
            expected_checksum=ours_checksum,
        )

    stat = setup_client.stat_object(bucket, final_key)
    assert _read(setup_client, bucket, final_key) == existing
    assert stat.metadata is not None
    assert stat.metadata["x-amz-meta-sha256"] == existing_checksum
    assert setup_client._list_multipart_uploads(bucket, prefix=final_key).uploads == []


@pytest.mark.asyncio
async def test_real_minio_concurrent_publish_has_one_create_only_winner(
    real_minio: tuple[Minio, str],
    minio_connection: _MinioConnection,
) -> None:
    setup_client, bucket = real_minio
    final_key = "knowledge-bases/k/documents/d/versions/v/parsed/race.txt"
    first_temp = "tmp/jobs/atomic-race-1/parsed/text.txt"
    second_temp = "tmp/jobs/atomic-race-2/parsed/text.txt"
    first_body = b"first contender"
    second_body = b"second contender"
    first_checksum = _put(setup_client, bucket, first_temp, first_body)
    second_checksum = _put(setup_client, bucket, second_temp, second_body)
    barrier = threading.Barrier(2)
    first_client = _BarrierMinio(
        minio_connection.endpoint,
        access_key=minio_connection.access_key,
        secret_key=minio_connection.secret_key,
        secure=minio_connection.secure,
        final_key=final_key,
        initial_head_barrier=barrier,
    )
    second_client = _BarrierMinio(
        minio_connection.endpoint,
        access_key=minio_connection.access_key,
        secret_key=minio_connection.secret_key,
        secure=minio_connection.secure,
        final_key=final_key,
        initial_head_barrier=barrier,
    )
    first_store = MinioObjectStore(
        client=cast(Any, first_client),
        bucket=bucket,
        buffer_bytes=64 * 1024,
        part_size_bytes=5 * 1024 * 1024,
    )
    second_store = MinioObjectStore(
        client=cast(Any, second_client),
        bucket=bucket,
        buffer_bytes=64 * 1024,
        part_size_bytes=5 * 1024 * 1024,
    )
    results = await asyncio.gather(
        first_store.publish_temp(
            first_temp,
            final_key,
            expected_size=len(first_body),
            expected_checksum=first_checksum,
        ),
        second_store.publish_temp(
            second_temp,
            final_key,
            expected_size=len(second_body),
            expected_checksum=second_checksum,
        ),
        return_exceptions=True,
    )

    winners = [result for result in results if isinstance(result, PublishedObject)]
    conflicts = [result for result in results if isinstance(result, ArtifactChecksumConflict)]
    assert len(winners) == 1
    assert len(conflicts) == 1
    winner = winners[0]
    expected_body = first_body if winner.checksum_sha256 == first_checksum else second_body
    assert _read(setup_client, bucket, final_key) == expected_body
    assert setup_client._list_multipart_uploads(bucket, prefix=final_key).uploads == []


@pytest.mark.asyncio
async def test_real_minio_server_side_multipart_copy_preserves_large_content(
    real_minio: tuple[Minio, str],
) -> None:
    client, bucket = real_minio
    temp_key = "tmp/jobs/multipart/parsed/text.txt"
    final_key = "knowledge-bases/k/documents/d/versions/v/parsed/multipart.txt"
    body = b"m" * ((5 * 1024 * 1024) + 257)
    checksum = _put(client, bucket, temp_key, body)
    store = MinioObjectStore(
        client=cast(Any, client),
        bucket=bucket,
        buffer_bytes=64 * 1024,
        part_size_bytes=5 * 1024 * 1024,
    )

    published = await store.publish_temp(
        temp_key,
        final_key,
        expected_size=len(body),
        expected_checksum=checksum,
    )

    assert published.reused is False
    assert _read(client, bucket, final_key) == body
    metadata = client.stat_object(bucket, final_key).metadata
    assert metadata is not None
    assert metadata["x-amz-meta-sha256"] == checksum
    assert client._list_multipart_uploads(bucket, prefix=final_key).uploads == []


@pytest.mark.asyncio
async def test_real_minio_upload_stream_binds_raw_put_etag_before_atomic_publish(
    real_minio: tuple[Minio, str],
) -> None:
    client, bucket = real_minio
    final_key = "knowledge-bases/k/documents/d/versions/v/source/source.txt"
    body = b"s" * ((5 * 1024 * 1024) + 257)
    checksum = hashlib.sha256(body).hexdigest()
    store = MinioObjectStore(
        client=cast(Any, client),
        bucket=bucket,
        buffer_bytes=64 * 1024,
        part_size_bytes=5 * 1024 * 1024,
    )

    stored = await store.upload_stream(
        final_key,
        _chunks(body),
        content_type="text/plain",
        max_bytes=len(body),
    )

    assert stored.checksum_sha256 == checksum
    assert _read(client, bucket, final_key) == body
    final_stat = client.stat_object(bucket, final_key)
    assert final_stat.metadata is not None
    assert final_stat.metadata["x-amz-meta-sha256"] == checksum
    assert final_stat.content_type == "text/plain"
    assert client._list_multipart_uploads(bucket, prefix="").uploads == []
    assert all(
        not item.object_name.startswith("tmp/uploads/")
        for item in client.list_objects(bucket, recursive=True)
    )


@pytest.mark.asyncio
async def test_real_minio_bounded_read_verifies_the_persisted_checksum(
    real_minio: tuple[Minio, str],
) -> None:
    client, bucket = real_minio
    key = "knowledge-bases/k/documents/d/versions/v/source/read.txt"
    body = "有界读取🙂\n".encode()
    checksum = _put(client, bucket, key, body)
    store = MinioObjectStore(
        client=cast(Any, client),
        bucket=bucket,
        buffer_bytes=64 * 1024,
        part_size_bytes=5 * 1024 * 1024,
    )

    assert (
        await store.read_bytes(
            key,
            expected_checksum=checksum,
            max_bytes=len(body),
        )
        == body
    )
