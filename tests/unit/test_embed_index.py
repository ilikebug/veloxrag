from __future__ import annotations

import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import cast
from uuid import UUID, uuid4

import httpx
import pytest
from qdrant_client import AsyncQdrantClient
from qdrant_client.common.client_exceptions import ResourceExhaustedResponse
from qdrant_client.http import models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from rag_service.indexing import qdrant as qdrant_module
from rag_service.indexing.identities import canonical_json_bytes, canonical_sha256, point_id
from rag_service.indexing.qdrant import (
    AsyncQdrantCollectionClient,
    CollectionSpec,
    FakeQdrantClient,
    QdrantConfigurationError,
    QdrantPoint,
    QdrantRetrievedPoint,
    QdrantTransientError,
)
from rag_service.ingestion.artifacts import chunk_manifest_header_bytes, chunk_manifest_record_bytes
from rag_service.ingestion.chunkers import Chunk, RecursiveTextChunker
from rag_service.ingestion.parsers import parser_for_extension
from rag_service.ingestion.pipeline import ManifestExpectation, iter_verified_manifest_batches


async def _stream(value: bytes, size: int = 17) -> AsyncIterator[bytes]:
    for offset in range(0, len(value), size):
        yield value[offset : offset + size]


def _manifest() -> tuple[bytes, ManifestExpectation, tuple[Chunk, ...]]:
    source = b"alpha beta gamma delta"
    parsed = parser_for_extension(".txt").parse(source)
    chunker = RecursiveTextChunker(max_chunk_codepoints=8, target_overlap_codepoints=1)
    chunks = tuple(chunker.chunk(parsed))
    created_at = datetime(2026, 1, 1, tzinfo=UTC)
    value = chunk_manifest_header_bytes(
        source_checksum_sha256="a" * 64,
        parsed=parsed,
        chunker=chunker,
        document_version_created_at=created_at,
        chunk_count=len(chunks),
    ) + b"".join(chunk_manifest_record_bytes(chunk) for chunk in chunks)
    expected = ManifestExpectation(
        source_checksum_sha256="a" * 64,
        parsed_checksum_sha256=parsed.checksum_sha256,
        parser_name=parsed.parser_name,
        parser_version=parsed.parser_version,
        parser_config=dict(parsed.parser_config),
        chunker_name=chunker.name,
        chunker_version=chunker.version,
        chunker_config=dict(chunker.config),
        chunk_config_hash=chunker.config_hash,
        document_version_created_at=created_at,
        chunk_count=len(chunks),
    )
    return value, expected, chunks


@pytest.mark.asyncio
async def test_manifest_reader_validates_then_batches_from_exclusive_checkpoint() -> None:
    value, expected, chunks = _manifest()

    batches = [
        batch
        async for batch in iter_verified_manifest_batches(
            _stream(value),
            expected=expected,
            next_chunk_index=1,
            batch_size=2,
        )
    ]

    assert [[chunk.chunk_index for chunk in batch] for batch in batches] == [
        list(range(start, min(start + 2, len(chunks)))) for start in range(1, len(chunks), 2)
    ]


@pytest.mark.asyncio
async def test_manifest_reader_rejects_noncanonical_or_noncontiguous_records() -> None:
    value, expected, _chunks = _manifest()
    lines = value.splitlines()
    record = json.loads(lines[2])
    record["chunk_index"] += 1
    lines[2] = canonical_json_bytes(record)
    invalid = b"\n".join(lines) + b"\n"

    with pytest.raises(ValueError, match="chunk manifest is invalid"):
        _ = [
            batch
            async for batch in iter_verified_manifest_batches(
                _stream(invalid),
                expected=expected,
                next_chunk_index=0,
                batch_size=2,
            )
        ]


@pytest.mark.asyncio
async def test_manifest_reader_closes_source_when_full_final_batch_is_returned_early() -> None:
    value, expected, chunks = _manifest()
    closed = False

    async def tracked_stream() -> AsyncIterator[bytes]:
        nonlocal closed
        try:
            yield value
        finally:
            closed = True

    reader = iter_verified_manifest_batches(
        tracked_stream(),
        expected=expected,
        next_chunk_index=0,
        batch_size=len(chunks),
    )

    batch = await anext(reader)
    assert len(batch) == len(chunks)
    assert closed is False
    await reader.aclose()

    assert closed is True


def test_qdrant_point_payload_is_a_strict_content_safe_allowlist() -> None:
    version_id = uuid4()
    chunk_hash = "b" * 64
    point = QdrantPoint(
        id=point_id(version_id, 3, chunk_hash),
        vector=(0.1, 0.2, 0.3),
        payload={
            "knowledge_base_id": str(uuid4()),
            "document_id": str(uuid4()),
            "version_id": str(version_id),
            "chunk_index": 3,
            "chunk_hash": chunk_hash,
            "text": "safe source text",
            "title_path": ["Title"],
            "start_offset": 1,
            "end_offset": 17,
            "metadata": {"f_1234567890abcdef1234567890abcdef": "approved"},
        },
    )

    assert set(point.payload) == {
        "knowledge_base_id",
        "document_id",
        "version_id",
        "chunk_index",
        "chunk_hash",
        "text",
        "title_path",
        "start_offset",
        "end_offset",
        "metadata",
    }
    with pytest.raises(ValueError, match="Qdrant point is invalid"):
        QdrantPoint(
            id=point.id,
            vector=point.vector,
            payload={**dict(point.payload), "object_key": "secret/minio/key"},
        )
    with pytest.raises(ValueError, match="Qdrant point is invalid"):
        QdrantPoint(id=uuid4(), vector=point.vector, payload=point.payload)
    with pytest.raises(ValueError, match="Qdrant point is invalid"):
        QdrantPoint(
            id=point.id,
            vector=point.vector,
            payload={**dict(point.payload), "version_id": version_id.hex},
        )
    with pytest.raises(ValueError, match="Qdrant point is invalid"):
        QdrantPoint(
            id=point.id,
            vector=point.vector,
            payload={
                **dict(point.payload),
                "metadata": {"f_1234567890abcdef1234567890abcdef": {"secret": "x"}},
            },
        )
    with pytest.raises(ValueError, match="Qdrant point is invalid"):
        QdrantPoint(
            id=point.id,
            vector=point.vector,
            payload={
                **dict(point.payload),
                "metadata": {"f_1234567890abcdef1234567890abcdef": 2**63},
            },
        )


@pytest.mark.asyncio
async def test_qdrant_upsert_requires_completed_wait_result() -> None:
    class RawClient:
        def __init__(self) -> None:
            self.status = models.UpdateStatus.ACKNOWLEDGED

        async def get_collection(self, _collection: str) -> object:
            return object()

        async def upsert(self, **kwargs: object) -> models.UpdateResult:
            assert kwargs["wait"] is True
            return models.UpdateResult(status=self.status)

    class Adapter(AsyncQdrantCollectionClient):
        async def _managed_info(self, _collection: str) -> models.CollectionInfo:
            return cast(models.CollectionInfo, object())

    raw = RawClient()
    adapter = Adapter(cast(AsyncQdrantClient, raw))
    version_id = uuid4()
    chunk_hash = "c" * 64
    point = QdrantPoint(
        id=point_id(version_id, 0, chunk_hash),
        vector=(0.1,),
        payload={
            "knowledge_base_id": str(uuid4()),
            "document_id": str(uuid4()),
            "version_id": str(version_id),
            "chunk_index": 0,
            "chunk_hash": chunk_hash,
            "text": "x",
            "title_path": [],
            "start_offset": 0,
            "end_offset": 1,
            "metadata": {},
        },
    )

    with pytest.raises(QdrantTransientError, match="did not complete"):
        await adapter.upsert_points(
            "rag_kb_" + uuid4().hex + "_g_" + uuid4().hex,
            (point,),
        )
    raw.status = models.UpdateStatus.COMPLETED
    await adapter.upsert_points(
        "rag_kb_" + uuid4().hex + "_g_" + uuid4().hex,
        (point,),
    )


@pytest.mark.asyncio
async def test_qdrant_version_inspection_counts_exactly_then_retrieves_one_bounded_batch() -> None:
    version_id = uuid4()
    collection = "rag_kb_" + uuid4().hex + "_g_" + uuid4().hex
    points = tuple(
        QdrantPoint(
            id=point_id(version_id, index, f"{index + 1:064x}"),
            vector=(0.1, 0.2, 0.3),
            payload={
                "knowledge_base_id": str(uuid4()),
                "document_id": str(uuid4()),
                "version_id": str(version_id),
                "chunk_index": index,
                "chunk_hash": f"{index + 1:064x}",
                "text": f"point-{index}",
                "title_path": [],
                "start_offset": 0,
                "end_offset": len(f"point-{index}"),
                "metadata": {},
            },
        )
        for index in range(3)
    )

    class RawClient:
        def __init__(self) -> None:
            self.retrieve_calls: list[dict[str, object]] = []

        async def get_collection(self, _collection: str) -> object:
            return object()

        async def count(self, **kwargs: object) -> models.CountResult:
            assert kwargs["exact"] is True
            must = cast(models.Filter, kwargs["count_filter"]).must
            assert isinstance(must, list)
            condition = must[0]
            assert isinstance(condition, models.FieldCondition)
            assert condition.key == "version_id"
            assert isinstance(condition.match, models.MatchValue)
            assert condition.match.value == str(version_id)
            return models.CountResult(count=3)

        async def retrieve(self, **kwargs: object) -> list[models.Record]:
            self.retrieve_calls.append(kwargs)
            assert kwargs["with_payload"] is True
            assert kwargs["with_vectors"] is True
            assert kwargs["ids"] == [str(point.id) for point in points]
            return [
                models.Record(id=point.id, payload=dict(point.payload), vector=list(point.vector))
                for point in reversed(points)
            ]

    class Adapter(AsyncQdrantCollectionClient):
        async def _managed_info(self, _collection: str) -> models.CollectionInfo:
            return cast(models.CollectionInfo, object())

    raw = RawClient()
    adapter = Adapter(cast(AsyncQdrantClient, raw))
    assert await adapter.count_version_points(collection, version_id) == 3
    actual = await adapter.retrieve_version_points(
        collection,
        version_id,
        tuple(point.id for point in points),
    )

    assert tuple(point.id for point in actual) == tuple(point.id for point in points)
    assert {point.vector_dimension for point in actual} == {3}
    assert len(raw.retrieve_calls) == 1


@pytest.mark.asyncio
async def test_qdrant_version_retrieve_rejects_missing_or_duplicate_ids() -> None:
    version_id = uuid4()
    point = QdrantPoint(
        id=point_id(version_id, 0, "d" * 64),
        vector=(0.1,),
        payload={
            "knowledge_base_id": str(uuid4()),
            "document_id": str(uuid4()),
            "version_id": str(version_id),
            "chunk_index": 0,
            "chunk_hash": "d" * 64,
            "text": "x",
            "title_path": [],
            "start_offset": 0,
            "end_offset": 1,
            "metadata": {},
        },
    )

    class RawClient:
        async def get_collection(self, _collection: str) -> object:
            return object()

        async def retrieve(self, **_kwargs: object) -> list[models.Record]:
            return [
                models.Record(id=point.id, payload=dict(point.payload), vector=list(point.vector)),
                models.Record(id=point.id, payload=dict(point.payload), vector=list(point.vector)),
            ]

    class Adapter(AsyncQdrantCollectionClient):
        async def _managed_info(self, _collection: str) -> models.CollectionInfo:
            return cast(models.CollectionInfo, object())

    adapter = Adapter(cast(AsyncQdrantClient, RawClient()))
    with pytest.raises(QdrantConfigurationError, match="response is invalid"):
        await adapter.retrieve_version_points(
            "rag_kb_" + uuid4().hex + "_g_" + uuid4().hex,
            version_id,
            (point.id, uuid4()),
        )


@pytest.mark.asyncio
async def test_fake_qdrant_version_inspection_is_strict_and_version_scoped() -> None:
    version_id = uuid4()
    other_version_id = uuid4()
    collection = "rag_kb_" + uuid4().hex + "_g_" + uuid4().hex
    spec = CollectionSpec(collection, 3, "cosine", ())
    fake = FakeQdrantClient()
    await fake.seed_collection(spec, created_at=datetime.now(UTC))
    point = QdrantPoint(
        id=point_id(version_id, 0, "e" * 64),
        vector=(0.1, 0.2, 0.3),
        payload={
            "knowledge_base_id": str(uuid4()),
            "document_id": str(uuid4()),
            "version_id": str(version_id),
            "chunk_index": 0,
            "chunk_hash": "e" * 64,
            "text": "point",
            "title_path": [],
            "start_offset": 0,
            "end_offset": 5,
            "metadata": {},
        },
    )
    await fake.upsert_points(collection, (point,))

    assert await fake.count_version_points(collection, version_id) == 1
    assert await fake.count_version_points(collection, other_version_id) == 0
    inspected = await fake.retrieve_version_points(collection, version_id, (point.id,))
    assert inspected == (QdrantRetrievedPoint(point.id, 3, canonical_sha256(dict(point.payload))),)

    with pytest.raises(ValueError, match="version identity is invalid"):
        await fake.count_version_points("invalid", version_id)
    with pytest.raises(ValueError, match="version identity is invalid"):
        await fake.count_version_points(collection, cast(UUID, "invalid"))
    with pytest.raises(QdrantConfigurationError, match="does not exist"):
        await fake.count_version_points(
            "rag_kb_" + uuid4().hex + "_g_" + uuid4().hex,
            version_id,
        )
    with pytest.raises(QdrantConfigurationError, match="response is invalid"):
        await fake.retrieve_version_points(collection, version_id, (uuid4(),))
    with pytest.raises(QdrantConfigurationError, match="response is invalid"):
        await fake.retrieve_version_points(collection, other_version_id, (point.id,))


@pytest.mark.asyncio
async def test_qdrant_retrieve_request_validation_rejects_invalid_shapes() -> None:
    class RawClient:
        async def get_collection(self, _collection: str) -> object:
            return object()

    collection = "rag_kb_" + uuid4().hex + "_g_" + uuid4().hex
    adapter = AsyncQdrantCollectionClient(cast(AsyncQdrantClient, RawClient()))
    cases: tuple[tuple[str, object, tuple[object, ...]], ...] = (
        ("invalid", uuid4(), (uuid4(),)),
        (collection, "invalid", (uuid4(),)),
        (collection, uuid4(), ()),
        (collection, uuid4(), (uuid4(), "invalid")),
    )
    duplicate = uuid4()
    cases += ((collection, uuid4(), (duplicate, duplicate)),)
    for requested_collection, version_id, requested in cases:
        with pytest.raises(ValueError, match="retrieve request is invalid"):
            await adapter.retrieve_version_points(
                requested_collection,
                cast(UUID, version_id),
                cast(tuple[UUID, ...], requested),
            )


@pytest.mark.asyncio
async def test_qdrant_version_count_request_validation_matches_fake() -> None:
    class RawClient:
        async def get_collection(self, _collection: str) -> object:
            return object()

    collection = "rag_kb_" + uuid4().hex + "_g_" + uuid4().hex
    real = AsyncQdrantCollectionClient(cast(AsyncQdrantClient, RawClient()))
    fake = FakeQdrantClient()
    cases: tuple[tuple[str, object], ...] = (
        ("invalid", uuid4()),
        (collection, "invalid"),
    )
    for client in (real, fake):
        for requested_collection, version_id in cases:
            with pytest.raises(ValueError, match="version identity is invalid"):
                await client.count_version_points(
                    requested_collection,
                    cast(UUID, version_id),
                )


@pytest.mark.asyncio
@pytest.mark.parametrize("count", [-1, True, 10_000_001])
async def test_qdrant_version_count_rejects_invalid_provider_counts(count: object) -> None:
    class RawClient:
        async def get_collection(self, _collection: str) -> object:
            return object()

        async def count(self, **_kwargs: object) -> object:
            return type("Count", (), {"count": count})()

    class Adapter(AsyncQdrantCollectionClient):
        async def _managed_info(self, _collection: str) -> models.CollectionInfo:
            return cast(models.CollectionInfo, object())

    adapter = Adapter(cast(AsyncQdrantClient, RawClient()))
    with pytest.raises(QdrantConfigurationError, match="point count is invalid"):
        await adapter.count_version_points(
            "rag_kb_" + uuid4().hex + "_g_" + uuid4().hex,
            uuid4(),
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["count", "retrieve"])
async def test_qdrant_version_inspection_maps_network_failures_to_transient(
    operation: str,
) -> None:
    class RawClient:
        async def get_collection(self, _collection: str) -> object:
            return object()

        async def count(self, **_kwargs: object) -> object:
            raise httpx.ReadTimeout("offline")

        async def retrieve(self, **_kwargs: object) -> object:
            raise OSError("offline")

    class Adapter(AsyncQdrantCollectionClient):
        async def _managed_info(self, _collection: str) -> models.CollectionInfo:
            return cast(models.CollectionInfo, object())

    adapter = Adapter(cast(AsyncQdrantClient, RawClient()))
    collection = "rag_kb_" + uuid4().hex + "_g_" + uuid4().hex
    with pytest.raises(QdrantTransientError, match="unavailable"):
        if operation == "count":
            await adapter.count_version_points(collection, uuid4())
        else:
            await adapter.retrieve_version_points(collection, uuid4(), (uuid4(),))


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["count", "retrieve"])
@pytest.mark.parametrize(
    ("error", "expected_type"),
    [
        (
            UnexpectedResponse(400, "bad request", b"{}", httpx.Headers()),
            QdrantConfigurationError,
        ),
        (
            UnexpectedResponse(503, "unavailable", b"{}", httpx.Headers()),
            QdrantTransientError,
        ),
        (
            UnexpectedResponse(None, "offline", b"{}", httpx.Headers()),
            QdrantTransientError,
        ),
        (
            UnexpectedResponse(429, "rate limited", b"{}", httpx.Headers()),
            QdrantTransientError,
        ),
        (ResourceExhaustedResponse("busy", 1), QdrantTransientError),
        (ResponseHandlingException(ValueError("bad response")), QdrantConfigurationError),
        (ResponseHandlingException(OSError("offline")), QdrantTransientError),
        (ValueError("invalid provider result"), QdrantConfigurationError),
    ],
)
async def test_qdrant_version_inspection_maps_provider_failures_safely(
    operation: str,
    error: Exception,
    expected_type: type[Exception],
) -> None:
    class RawClient:
        async def get_collection(self, _collection: str) -> object:
            return object()

        async def count(self, **_kwargs: object) -> object:
            raise error

        async def retrieve(self, **_kwargs: object) -> object:
            raise error

    class Adapter(AsyncQdrantCollectionClient):
        async def _managed_info(self, _collection: str) -> models.CollectionInfo:
            return cast(models.CollectionInfo, object())

    adapter = Adapter(cast(AsyncQdrantClient, RawClient()))
    collection = "rag_kb_" + uuid4().hex + "_g_" + uuid4().hex
    with pytest.raises(expected_type):
        if operation == "count":
            await adapter.count_version_points(collection, uuid4())
        else:
            await adapter.retrieve_version_points(collection, uuid4(), (uuid4(),))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode",
    [
        "not-list",
        "wrong-count",
        "not-record",
        "noncanonical-id",
        "empty-vector",
        "integer-vector",
        "missing-payload",
        "invalid-payload",
        "wrong-version",
        "unexpected-id",
        "duplicate-id",
    ],
)
async def test_qdrant_version_retrieve_rejects_malformed_provider_records(mode: str) -> None:
    version_id = uuid4()
    requested_id = point_id(version_id, 0, "f" * 64)
    payload = {
        "knowledge_base_id": str(uuid4()),
        "document_id": str(uuid4()),
        "version_id": str(version_id),
        "chunk_index": 0,
        "chunk_hash": "f" * 64,
        "text": "x",
        "title_path": [],
        "start_offset": 0,
        "end_offset": 1,
        "metadata": {},
    }
    record: object = models.Record(
        id=requested_id,
        payload=payload,
        vector=[0.1],
    )
    requested_ids: tuple[UUID, ...] = (requested_id,)
    records: object = [record]
    if mode == "not-list":
        records = (record,)
    elif mode == "wrong-count":
        records = []
    elif mode == "not-record":
        records = [object()]
    elif mode == "noncanonical-id":
        records = [models.Record(id=str(requested_id).upper(), payload=payload, vector=[0.1])]
    elif mode == "empty-vector":
        records = [models.Record(id=requested_id, payload=payload, vector=[])]
    elif mode == "integer-vector":
        integer_vector_record = models.Record(
            id=requested_id,
            payload=payload,
            vector=[0.1],
        )
        object.__setattr__(integer_vector_record, "vector", [1])
        records = [integer_vector_record]
    elif mode == "missing-payload":
        records = [models.Record(id=requested_id, payload=None, vector=[0.1])]
    elif mode == "invalid-payload":
        records = [
            models.Record(
                id=requested_id,
                payload={**payload, "chunk_hash": "e" * 64},
                vector=[0.1],
            )
        ]
    elif mode == "wrong-version":
        other_version = uuid4()
        other_id = point_id(other_version, 0, "f" * 64)
        records = [
            models.Record(
                id=other_id,
                payload={**payload, "version_id": str(other_version)},
                vector=[0.1],
            )
        ]
        requested_id = other_id
    elif mode == "unexpected-id":
        other_id = point_id(version_id, 1, "e" * 64)
        records = [
            models.Record(
                id=other_id,
                payload={**payload, "chunk_index": 1, "chunk_hash": "e" * 64},
                vector=[0.1],
            )
        ]
    else:
        other_id = point_id(version_id, 1, "e" * 64)
        records = [record, record]
        requested_ids = (requested_id, other_id)

    class RawClient:
        async def get_collection(self, _collection: str) -> object:
            return object()

        async def retrieve(self, **_kwargs: object) -> object:
            return records

    class Adapter(AsyncQdrantCollectionClient):
        async def _managed_info(self, _collection: str) -> models.CollectionInfo:
            return cast(models.CollectionInfo, object())

    adapter = Adapter(cast(AsyncQdrantClient, RawClient()))
    with pytest.raises(QdrantConfigurationError, match="response is invalid"):
        await adapter.retrieve_version_points(
            "rag_kb_" + uuid4().hex + "_g_" + uuid4().hex,
            version_id,
            requested_ids,
        )


@pytest.mark.parametrize(
    ("point_id_value", "dimension", "digest"),
    [
        ("invalid", 1, "a" * 64),
        (uuid4(), 0, "a" * 64),
        (uuid4(), 10_000_001, "a" * 64),
        (uuid4(), 1, "invalid"),
    ],
)
def test_retrieved_point_value_object_rejects_invalid_fields(
    point_id_value: object,
    dimension: int,
    digest: str,
) -> None:
    with pytest.raises(ValueError, match="retrieved point is invalid"):
        QdrantRetrievedPoint(cast(UUID, point_id_value), dimension, digest)


def test_qdrant_record_uuid_rejects_non_uuid_and_noncanonical_strings() -> None:
    with pytest.raises(QdrantConfigurationError, match="response is invalid"):
        qdrant_module._record_uuid("invalid")
    value = uuid4()
    assert qdrant_module._record_uuid(str(value)) == value
    with pytest.raises(QdrantConfigurationError, match="response is invalid"):
        qdrant_module._record_uuid(str(value).upper())
    with pytest.raises(QdrantConfigurationError, match="response is invalid"):
        qdrant_module._record_uuid(123)


@pytest.mark.asyncio
async def test_qdrant_collection_existence_probe_has_real_and_fake_parity() -> None:
    existing = "rag_kb_" + uuid4().hex + "_g_" + uuid4().hex
    missing = "rag_kb_" + uuid4().hex + "_g_" + uuid4().hex

    class RawClient:
        async def get_collection(self, _collection: str) -> object:
            return object()

        async def collection_exists(self, collection: str) -> bool:
            return collection == existing

    adapter = AsyncQdrantCollectionClient(cast(AsyncQdrantClient, RawClient()))
    fake = FakeQdrantClient()
    await fake.seed_collection(
        CollectionSpec(existing, 3, "cosine", ()),
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )

    assert callable(getattr(adapter, "collection_exists", None))
    assert callable(getattr(fake, "collection_exists", None))
    assert await adapter.collection_exists(existing) is True
    assert await adapter.collection_exists(missing) is False
    assert await fake.collection_exists(existing) is True
    assert await fake.collection_exists(missing) is False
