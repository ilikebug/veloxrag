from __future__ import annotations

import math
from collections.abc import Awaitable, Callable
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace
from typing import Any, cast
from uuid import UUID

import httpx
import pytest
from qdrant_client import AsyncQdrantClient
from qdrant_client.common.client_exceptions import ResourceExhaustedResponse
from qdrant_client.http import models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

import rag_service.indexing.generation_services as generation_services
import rag_service.indexing.qdrant as qdrant_module
from rag_service.indexing.generation_services import (
    build_embedding_configuration,
    build_filter_snapshot,
    canonical_empty_validation,
    payload_indexes_for_filter_snapshot,
)
from rag_service.indexing.identities import (
    canonical_json_bytes,
    canonical_sha256,
    collection_name,
    point_id,
)
from rag_service.indexing.qdrant import (
    AsyncQdrantCollectionClient,
    CollectionSpec,
    FakeQdrantClient,
    PayloadIndex,
    QdrantConfigurationError,
    QdrantTransientError,
    qdrant_client_from_url,
    qdrant_distance,
)
from rag_service.providers.repositories import (
    ModelProfileRecord,
    ProviderConfigRecord,
    ProviderCredentialRecord,
)

KB_ID = UUID("11111111-1111-4111-8111-111111111111")
GENERATION_ID = UUID("22222222-2222-4222-8222-222222222222")
VERSION_ID = UUID("33333333-3333-4333-8333-333333333333")
PROFILE_ID = UUID("44444444-4444-4444-8444-444444444444")
PROVIDER_ID = UUID("55555555-5555-4555-8555-555555555555")
CREDENTIAL_ID = UUID("66666666-6666-4666-8666-666666666666")
NOW = datetime(2026, 7, 28, 8, 0, tzinfo=UTC)


@pytest.mark.parametrize(
    "cursor",
    (
        "orphan-cleanup-v1:expired:not-base64!",
        "orphan-cleanup-v1:unknown:-",
        "orphan-cleanup-v1:expired:",
    ),
)
def test_orphan_reconciliation_cursor_rejects_malformed_internal_state(cursor: str) -> None:
    with pytest.raises(ValueError, match="cursor"):
        generation_services._decode_orphan_cursor(cursor)


def _credential(identifier: UUID = CREDENTIAL_ID) -> ProviderCredentialRecord:
    return ProviderCredentialRecord(
        id=identifier,
        name="embedding credential",
        key_version="2026-07",
        resource_revision=1,
        created_at=NOW,
        updated_at=NOW,
        rotated_at=None,
    )


def _provider(
    *,
    credential_id: UUID | None = CREDENTIAL_ID,
    enabled: bool = True,
) -> ProviderConfigRecord:
    return ProviderConfigRecord(
        id=PROVIDER_ID,
        name="embedding provider",
        provider_type="openai_compatible",
        base_url="https://Provider.Example:443/v1/",
        credential_id=credential_id,
        default_headers={"X-Title": "RAG"},
        routing_options={},
        timeout_seconds=Decimal("12.500"),
        max_concurrency=3,
        requests_per_minute=120,
        enabled=enabled,
        resource_revision=2,
        endpoint_policy_version="provider-endpoint-v1",
        endpoint_validated_at=NOW,
        created_at=NOW,
        updated_at=NOW,
    )


def _profile(
    *,
    capability: str = "embedding",
    dimension: int | None = 3,
    enabled: bool = True,
) -> ModelProfileRecord:
    return ModelProfileRecord(
        id=PROFILE_ID,
        name="embedding profile",
        capability=capability,
        provider_config_id=PROVIDER_ID,
        model_name="text-embedding-test",
        dimension=dimension,
        max_input_tokens=8192,
        batch_size=16,
        timeout_seconds=Decimal("10.000"),
        vector_config={},
        enabled=enabled,
        resource_revision=4,
        created_at=NOW,
        updated_at=NOW,
    )


def _filter_schema() -> dict[str, object]:
    return {
        "fields": [
            {
                "name": "department",
                "source_path": "attributes.department",
                "type": "keyword",
                "operators": ["eq", "in"],
                "field_id": "fld_ERERERERQRGBEREREREREQ",
                "payload_path": "metadata.f_11111111111141118111111111111111",
            },
            {
                "name": "priority",
                "source_path": "attributes.priority",
                "type": "integer",
                "operators": ["eq", "gte", "lte"],
                "field_id": "fld_IiIiIiIiQiKCIiIiIiIiIg",
                "payload_path": "metadata.f_22222222222242228222222222222222",
            },
            {
                "name": "confidence",
                "source_path": "attributes.confidence",
                "type": "float",
                "operators": ["eq", "gte", "lte"],
                "field_id": "fld_MzMzMzMzQzODMzMzMzMzMw",
                "payload_path": "metadata.f_33333333333343338333333333333333",
            },
            {
                "name": "approved",
                "source_path": "attributes.approved",
                "type": "boolean",
                "operators": ["eq"],
                "field_id": "fld_REREREREQESERERERERERA",
                "payload_path": "metadata.f_44444444444444448444444444444444",
            },
            {
                "name": "publishedAt",
                "source_path": "system.published_at",
                "type": "datetime",
                "operators": ["eq", "gte", "lte"],
                "field_id": "fld_VVVVVVVVRVWVVVVVVVVVVQ",
                "payload_path": "metadata.f_55555555555545559555555555555555",
            },
        ]
    }


def test_collection_and_point_identities_are_stable_and_strict() -> None:
    assert collection_name(KB_ID, GENERATION_ID) == (
        "rag_kb_11111111111141118111111111111111_g_22222222222242228222222222222222"
    )
    expected = point_id(VERSION_ID, 7, "a" * 64)
    assert expected == point_id(VERSION_ID, 7, "a" * 64)
    assert expected != point_id(VERSION_ID, 8, "a" * 64)
    assert expected != point_id(VERSION_ID, 7, "b" * 64)

    for invalid_index in (-1, True, 1.5):
        with pytest.raises(ValueError, match="chunk index"):
            point_id(VERSION_ID, invalid_index, "a" * 64)  # type: ignore[arg-type]
    for invalid_hash in ("A" * 64, "a" * 63, "g" * 64, 1):
        with pytest.raises(ValueError, match="chunk hash"):
            point_id(VERSION_ID, 0, invalid_hash)  # type: ignore[arg-type]


def test_canonical_json_and_sha256_are_order_independent_and_reject_noncanonical_values() -> None:
    left = {"z": [3, {"é": "文档"}], "a": {"false": False, "none": None}}
    right = {"a": {"none": None, "false": False}, "z": [3, {"é": "文档"}]}

    assert canonical_json_bytes(left) == (
        b'{"a":{"false":false,"none":null},"z":[3,{"\xc3\xa9":"\xe6\x96\x87\xe6\xa1\xa3"}]}'
    )
    assert canonical_json_bytes(left) == canonical_json_bytes(right)
    assert canonical_sha256(left) == canonical_sha256(right)
    assert len(canonical_sha256(left)) == 64

    invalid_values = [
        math.nan,
        math.inf,
        -math.inf,
        Decimal("1.0"),
        {1: "non-string-key"},
        {"tuple": (1, 2)},
        {"set": {1}},
    ]
    for invalid in invalid_values:
        with pytest.raises(ValueError, match="canonical JSON"):
            canonical_json_bytes(invalid)


@pytest.mark.parametrize(
    ("distance", "qdrant_value"),
    [
        ("cosine", "Cosine"),
        ("dot", "Dot"),
        ("euclid", "Euclid"),
        ("manhattan", "Manhattan"),
    ],
)
def test_distance_mapping_matches_database_and_qdrant(
    distance: str,
    qdrant_value: str,
) -> None:
    assert qdrant_distance(distance).value == qdrant_value


def test_embedding_snapshot_is_authoritative_safe_and_hash_excludes_only_credential_identity() -> (
    None
):
    built = build_embedding_configuration(
        _profile(),
        _provider(),
        _credential(),
        distance="manhattan",
    )

    assert built.snapshot == {
        "adapter_schema_version": "openai-embeddings-v1",
        "provider_type": "openai_compatible",
        "base_url": "https://provider.example/v1",
        "provider_config_id": str(PROVIDER_ID),
        "credential_id": str(CREDENTIAL_ID),
        "default_headers": {"X-Title": "RAG"},
        "routing_options": {},
        "model_name": "text-embedding-test",
        "dimension": 3,
        "distance": "manhattan",
        "max_input_tokens": 8192,
        "vector_config": {},
    }
    assert "secret" not in canonical_json_bytes(built.snapshot).decode("utf-8").lower()
    assert built.gateway_snapshot.credential_id == CREDENTIAL_ID
    assert built.operational.provider_config_id == PROVIDER_ID
    assert len(built.semantic_hash) == 64

    rotated_identity = UUID("77777777-7777-4777-8777-777777777777")
    other = build_embedding_configuration(
        _profile(),
        _provider(credential_id=rotated_identity),
        _credential(rotated_identity),
        distance="manhattan",
    )
    assert other.snapshot != built.snapshot
    assert other.semantic_hash == built.semantic_hash


@pytest.mark.parametrize(
    ("profile", "provider", "credential", "message"),
    [
        (_profile(capability="chat", dimension=None), _provider(), _credential(), "embedding"),
        (_profile(enabled=False), _provider(), _credential(), "disabled"),
        (_profile(), _provider(enabled=False), _credential(), "disabled"),
        (_profile(), _provider(credential_id=None), _credential(), "credential"),
        (_profile(), _provider(), None, "credential"),
    ],
)
def test_embedding_snapshot_rejects_non_embedding_disabled_and_missing_credential(
    profile: ModelProfileRecord,
    provider: ProviderConfigRecord,
    credential: ProviderCredentialRecord | None,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=f"(?i){message}"):
        build_embedding_configuration(
            profile,
            provider,
            credential,
            distance="cosine",
        )


def test_filter_snapshot_payload_indexes_and_empty_manifest_are_canonical() -> None:
    source = _filter_schema()
    snapshot = build_filter_snapshot(source)
    assert snapshot == source
    assert snapshot is not source
    assert snapshot["fields"] is not source["fields"]

    assert payload_indexes_for_filter_snapshot(snapshot) == (
        PayloadIndex("knowledge_base_id", "keyword"),
        PayloadIndex("document_id", "keyword"),
        PayloadIndex("version_id", "keyword"),
        PayloadIndex("chunk_index", "integer"),
        PayloadIndex("chunk_hash", "keyword"),
        PayloadIndex("metadata.f_11111111111141118111111111111111", "keyword"),
        PayloadIndex("metadata.f_22222222222242228222222222222222", "integer"),
        PayloadIndex("metadata.f_33333333333343338333333333333333", "float"),
        PayloadIndex("metadata.f_44444444444444448444444444444444", "bool"),
        PayloadIndex("metadata.f_55555555555545559555555555555555", "datetime"),
    )

    collection = collection_name(KB_ID, GENERATION_ID)
    manifest, manifest_hash = canonical_empty_validation(
        knowledge_base_id=KB_ID,
        generation_id=GENERATION_ID,
        collection=collection,
        revision=9,
        actual_point_count=0,
    )
    assert manifest == {
        "schema_version": "empty-generation-validation-v1",
        "knowledge_base_id": str(KB_ID),
        "generation_id": str(GENERATION_ID),
        "collection": collection,
        "revision": 9,
        "expected_point_count": 0,
        "actual_point_count": 0,
        "point_ids": [],
    }
    assert manifest_hash == canonical_sha256(manifest)


@pytest.mark.asyncio
async def test_qdrant_fake_is_idempotent_and_verifies_exact_vector_and_index_schema() -> None:
    fake = FakeQdrantClient(clock=lambda: NOW)
    spec = CollectionSpec(
        name=collection_name(KB_ID, GENERATION_ID),
        dimension=3,
        distance="cosine",
        payload_indexes=(PayloadIndex("document_id", "keyword"),),
    )

    await fake.ensure_collection(spec.vector_only())
    await fake.ensure_collection(spec.vector_only())
    await fake.ensure_payload_indexes(spec.name, spec.payload_indexes)
    await fake.ensure_payload_indexes(spec.name, spec.payload_indexes)
    await fake.verify_collection(spec)

    assert fake.create_calls == 1
    assert fake.payload_index_create_calls == 1
    assert await fake.count_points(spec.name) == 0
    assert tuple(fake.collection_names) == (spec.name,)

    for mismatch in (
        CollectionSpec(spec.name, 4, "cosine", spec.payload_indexes),
        CollectionSpec(spec.name, 3, "dot", spec.payload_indexes),
        CollectionSpec(
            spec.name,
            3,
            "cosine",
            (*spec.payload_indexes, PayloadIndex("chunk_index", "integer")),
        ),
    ):
        with pytest.raises(QdrantConfigurationError, match="does not match"):
            await fake.verify_collection(mismatch)

    with pytest.raises(QdrantConfigurationError, match="does not match"):
        await fake.ensure_collection(CollectionSpec(spec.name, 4, "cosine", ()))

    for unsafe in (
        replace(spec, datatype="uint8"),
        replace(spec, vector_on_disk=True),
        replace(spec, vector_hnsw_config=True),
        replace(spec, vector_quantization=True),
        replace(spec, multivector=True),
        replace(spec, sparse_vectors=("unexpected",)),
        replace(spec, on_disk_payload=False),
        replace(spec, collection_quantization=True),
        replace(spec, hnsw_config=replace(spec.hnsw_config, payload_m=0)),
        replace(
            spec,
            payload_indexes=(PayloadIndex("document_id", "keyword", (("on_disk", True),)),),
        ),
    ):
        with pytest.raises(QdrantConfigurationError, match="does not match"):
            await fake.verify_collection(unsafe)


@pytest.mark.asyncio
async def test_qdrant_fake_records_managed_creation_time_for_orphan_reconciliation() -> None:
    fake = FakeQdrantClient(clock=lambda: NOW)
    old_id = UUID("88888888-8888-4888-8888-888888888888")
    old_spec = CollectionSpec(
        collection_name(KB_ID, old_id),
        3,
        "cosine",
        (),
    )
    await fake.seed_collection(old_spec, created_at=NOW - timedelta(hours=25))

    page = await fake.list_managed_collections(limit=10, cursor=None)
    assert [(item.name, item.created_at) for item in page.items] == [
        (old_spec.name, NOW - timedelta(hours=25))
    ]
    assert page.next_cursor is None


@pytest.mark.asyncio
async def test_qdrant_fake_rechecks_managed_ownership_before_delete() -> None:
    fake = FakeQdrantClient(clock=lambda: NOW)
    spec = CollectionSpec(collection_name(KB_ID, GENERATION_ID), 3, "cosine", ())
    await fake.seed_collection(spec, created_at=NOW, managed=False)

    assert (await fake.list_managed_collections(limit=10, cursor=None)).items == ()
    with pytest.raises(QdrantConfigurationError, match="not managed"):
        await fake.delete_collection(spec.name)
    assert fake.collection_names == (spec.name,)


@pytest.mark.asyncio
async def test_qdrant_fake_rejects_unowned_collection_before_any_generation_operation() -> None:
    fake = FakeQdrantClient(clock=lambda: NOW)
    spec = CollectionSpec(
        collection_name(KB_ID, GENERATION_ID),
        3,
        "cosine",
        (PayloadIndex("document_id", "keyword"),),
    )
    await fake.seed_collection(
        spec,
        created_at=NOW,
        managed=False,
    )

    operations = (
        lambda: fake.ensure_collection(spec.vector_only()),
        lambda: fake.ensure_payload_indexes(spec.name, spec.payload_indexes),
        lambda: fake.verify_collection(spec),
        lambda: fake.count_points(spec.name),
    )
    for operation in operations:
        with pytest.raises(QdrantConfigurationError, match="not managed"):
            await operation()

    assert fake.payload_index_create_calls == 0
    assert fake.collection_names == (spec.name,)


@pytest.mark.asyncio
async def test_qdrant_fake_managed_listing_is_bounded_and_advances_past_unowned_prefix() -> None:
    fake = FakeQdrantClient(clock=lambda: NOW)
    identifiers = [UUID(f"00000000-0000-4000-8000-{index:012d}") for index in range(1, 6)]
    names = [collection_name(KB_ID, identifier) for identifier in identifiers]
    for index, name in enumerate(names):
        await fake.seed_collection(
            CollectionSpec(name, 3, "cosine", ()),
            created_at=NOW,
            managed=index >= 2,
        )

    cursor: str | None = None
    seen: list[str] = []
    page_count = 0
    while True:
        page = await fake.list_managed_collections(limit=2, cursor=cursor)
        assert len(page.items) <= 2
        seen.extend(item.name for item in page.items)
        page_count += 1
        if page.next_cursor is None:
            break
        assert page.next_cursor != cursor
        cursor = page.next_cursor

    assert page_count == 3
    assert seen == names[2:]

    with pytest.raises(ValueError, match="limit"):
        await fake.list_managed_collections(limit=0, cursor=None)
    with pytest.raises(ValueError, match="limit"):
        await fake.list_managed_collections(limit=101, cursor=None)


@pytest.mark.asyncio
async def test_qdrant_payload_index_creation_tolerates_concurrent_already_exists_race() -> None:
    spec = CollectionSpec(
        collection_name(KB_ID, GENERATION_ID),
        3,
        "cosine",
        (PayloadIndex("document_id", "keyword"),),
    )

    class _RaceClient:
        def __init__(self) -> None:
            self.create_calls = 0

        async def get_collection(self, _collection: str) -> None:
            return None

        async def create_payload_index(self, **_kwargs: object) -> None:
            self.create_calls += 1
            raise UnexpectedResponse(
                409,
                "Conflict",
                b"already exists",
                httpx.Headers(),
            )

    race_client = _RaceClient()

    class _RaceAdapter(AsyncQdrantCollectionClient):
        async def describe_collection(self, collection: str) -> CollectionSpec:
            assert collection == spec.name
            return spec if race_client.create_calls else spec.vector_only()

    adapter = _RaceAdapter(cast(AsyncQdrantClient, race_client))

    await adapter.ensure_payload_indexes(spec.name, spec.payload_indexes)

    assert race_client.create_calls == 1


@pytest.mark.asyncio
async def test_qdrant_adapter_creates_explicit_safe_vector_and_collection_defaults() -> None:
    spec = CollectionSpec(collection_name(KB_ID, GENERATION_ID), 3, "cosine", ())
    captured: dict[str, object] = {}

    class _Client:
        async def collection_exists(self, _collection: str) -> bool:
            return False

        async def create_collection(self, **kwargs: object) -> None:
            captured.update(kwargs)

        async def get_collection(self, collection: str) -> object:
            parts = collection.split("_")
            return SimpleNamespace(
                config=SimpleNamespace(
                    params=SimpleNamespace(
                        vectors=models.VectorParams(
                            size=3,
                            distance=models.Distance.COSINE,
                            datatype=models.Datatype.FLOAT32,
                            on_disk=False,
                        ),
                        sparse_vectors={},
                        on_disk_payload=True,
                    ),
                    quantization_config=None,
                    metadata={
                        "managed_by": "rag-service",
                        "schema_version": "rag-index-generation-v1",
                        "collection_name": collection,
                        "knowledge_base_id": parts[2],
                        "generation_id": parts[4],
                        "created_at": NOW.isoformat(),
                    },
                ),
                payload_schema={},
            )

    adapter = AsyncQdrantCollectionClient(cast(AsyncQdrantClient, _Client()), clock=lambda: NOW)

    await adapter.ensure_collection(spec)

    vectors = cast(models.VectorParams, captured["vectors_config"])
    assert vectors.datatype == models.Datatype.FLOAT32
    assert vectors.on_disk is False
    assert vectors.hnsw_config is None
    assert vectors.quantization_config is None
    assert vectors.multivector_config is None
    assert captured["sparse_vectors_config"] == {}
    assert captured["on_disk_payload"] is True
    assert captured["quantization_config"] is None
    assert captured["hnsw_config"] == models.HnswConfigDiff(
        m=16,
        ef_construct=100,
        full_scan_threshold=10_000,
        max_indexing_threads=0,
        on_disk=False,
        payload_m=16,
        inline_storage=False,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("schema", "expected_type"),
    [
        ("keyword", models.KeywordIndexParams),
        ("integer", models.IntegerIndexParams),
        ("float", models.FloatIndexParams),
        ("bool", models.BoolIndexParams),
        ("datetime", models.DatetimeIndexParams),
    ],
)
async def test_qdrant_adapter_creates_explicit_safe_payload_index_defaults(
    schema: str,
    expected_type: type[object],
) -> None:
    index = PayloadIndex("metadata.value", schema)
    captured: dict[str, object] = {}

    class _Client:
        async def get_collection(self, _collection: str) -> None:
            return None

        async def create_payload_index(self, **kwargs: object) -> None:
            captured.update(kwargs)

    class _Adapter(AsyncQdrantCollectionClient):
        async def describe_collection(self, collection: str) -> CollectionSpec:
            return CollectionSpec(
                collection,
                3,
                "cosine",
                (index,) if captured else (),
            )

    adapter = _Adapter(cast(AsyncQdrantClient, _Client()))
    collection = collection_name(KB_ID, GENERATION_ID)

    await adapter.ensure_payload_indexes(collection, (index,))

    field_schema = captured["field_schema"]
    assert type(field_schema) is expected_type
    dumped = cast(Any, field_schema).model_dump(exclude_none=False)
    assert dumped["on_disk"] is False
    assert dumped["enable_hnsw"] is True
    if schema == "keyword":
        assert dumped["is_tenant"] is False
    if schema == "integer":
        assert dumped["lookup"] is True
        assert dumped["range"] is True
        assert dumped["is_principal"] is False
    if schema in {"float", "datetime"}:
        assert dumped["is_principal"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "unsafe_overrides",
    [
        {"datatype": models.Datatype.UINT8},
        {"on_disk": True},
        {"hnsw_config": models.HnswConfigDiff(m=32)},
        {
            "quantization_config": models.ScalarQuantization(
                scalar=models.ScalarQuantizationConfig(type=models.ScalarType.INT8)
            )
        },
        {
            "multivector_config": models.MultiVectorConfig(
                comparator=models.MultiVectorComparator.MAX_SIM
            )
        },
    ],
)
async def test_qdrant_adapter_rejects_unsafe_dense_vector_configuration(
    unsafe_overrides: dict[str, object],
) -> None:
    spec = CollectionSpec(collection_name(KB_ID, GENERATION_ID), 3, "cosine", ())
    vector_kwargs: dict[str, object] = {
        "size": 3,
        "distance": models.Distance.COSINE,
        "datatype": models.Datatype.FLOAT32,
        "on_disk": False,
    }
    vector_kwargs.update(unsafe_overrides)

    class _Client:
        async def get_collection(self, collection: str) -> object:
            parts = collection.split("_")
            return SimpleNamespace(
                config=SimpleNamespace(
                    params=SimpleNamespace(
                        vectors=models.VectorParams(**vector_kwargs),
                        sparse_vectors={},
                        on_disk_payload=True,
                    ),
                    quantization_config=None,
                    metadata={
                        "managed_by": "rag-service",
                        "schema_version": "rag-index-generation-v1",
                        "collection_name": collection,
                        "knowledge_base_id": parts[2],
                        "generation_id": parts[4],
                        "created_at": NOW.isoformat(),
                    },
                ),
                payload_schema={},
            )

    adapter = AsyncQdrantCollectionClient(cast(AsyncQdrantClient, _Client()))

    with pytest.raises(QdrantConfigurationError, match="does not match"):
        await adapter.verify_collection(spec)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("sparse_vectors", "on_disk_payload", "collection_quantization", "payload_params"),
    [
        ({"sparse": models.SparseVectorParams()}, True, None, None),
        ({}, False, None, None),
        ({}, True, object(), None),
        (
            {},
            True,
            None,
            models.KeywordIndexParams(
                type=models.KeywordIndexType.KEYWORD,
                is_tenant=False,
                on_disk=True,
                enable_hnsw=True,
            ),
        ),
    ],
)
async def test_qdrant_adapter_rejects_unsafe_collection_or_payload_configuration(
    sparse_vectors: dict[str, models.SparseVectorParams],
    on_disk_payload: bool,
    collection_quantization: object | None,
    payload_params: models.KeywordIndexParams | None,
) -> None:
    index = PayloadIndex("document_id", "keyword")
    spec = CollectionSpec(
        collection_name(KB_ID, GENERATION_ID),
        3,
        "cosine",
        (index,),
    )

    class _Client:
        async def get_collection(self, collection: str) -> object:
            parts = collection.split("_")
            return SimpleNamespace(
                config=SimpleNamespace(
                    params=SimpleNamespace(
                        vectors=models.VectorParams(
                            size=3,
                            distance=models.Distance.COSINE,
                            datatype=models.Datatype.FLOAT32,
                            on_disk=False,
                        ),
                        sparse_vectors=sparse_vectors,
                        on_disk_payload=on_disk_payload,
                    ),
                    quantization_config=collection_quantization,
                    metadata={
                        "managed_by": "rag-service",
                        "schema_version": "rag-index-generation-v1",
                        "collection_name": collection,
                        "knowledge_base_id": parts[2],
                        "generation_id": parts[4],
                        "created_at": NOW.isoformat(),
                    },
                ),
                payload_schema={
                    "document_id": models.PayloadIndexInfo(
                        data_type=models.PayloadSchemaType.KEYWORD,
                        params=payload_params,
                        points=0,
                    )
                },
            )

    adapter = AsyncQdrantCollectionClient(cast(AsyncQdrantClient, _Client()))

    with pytest.raises(QdrantConfigurationError, match="does not match"):
        await adapter.verify_collection(spec)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "hnsw_config",
    [
        models.HnswConfig(
            m=32,
            ef_construct=100,
            full_scan_threshold=10_000,
            max_indexing_threads=0,
            on_disk=False,
            payload_m=16,
            inline_storage=False,
        ),
        models.HnswConfig(
            m=16,
            ef_construct=100,
            full_scan_threshold=10_000,
            max_indexing_threads=0,
            on_disk=False,
            payload_m=0,
            inline_storage=False,
        ),
        models.HnswConfig(
            m=16,
            ef_construct=100,
            full_scan_threshold=10_000,
            max_indexing_threads=0,
            on_disk=True,
            payload_m=16,
            inline_storage=False,
        ),
    ],
)
async def test_qdrant_adapter_rejects_unsafe_effective_collection_hnsw_defaults(
    hnsw_config: models.HnswConfig,
) -> None:
    spec = CollectionSpec(collection_name(KB_ID, GENERATION_ID), 3, "cosine", ())

    class _Client:
        async def get_collection(self, collection: str) -> object:
            parts = collection.split("_")
            return SimpleNamespace(
                config=SimpleNamespace(
                    params=SimpleNamespace(
                        vectors=models.VectorParams(
                            size=3,
                            distance=models.Distance.COSINE,
                            datatype=models.Datatype.FLOAT32,
                            on_disk=False,
                        ),
                        sparse_vectors={},
                        on_disk_payload=True,
                    ),
                    hnsw_config=hnsw_config,
                    quantization_config=None,
                    metadata={
                        "managed_by": "rag-service",
                        "schema_version": "rag-index-generation-v1",
                        "collection_name": collection,
                        "knowledge_base_id": parts[2],
                        "generation_id": parts[4],
                        "created_at": NOW.isoformat(),
                    },
                ),
                payload_schema={},
            )

    adapter = AsyncQdrantCollectionClient(cast(AsyncQdrantClient, _Client()))

    with pytest.raises(QdrantConfigurationError, match="does not match"):
        await adapter.verify_collection(spec)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "wrapped_error",
    [
        ResponseHandlingException(
            httpx.ConnectError(
                "provider-network-secret",
                request=httpx.Request("GET", "http://qdrant.invalid"),
            )
        ),
        ResponseHandlingException(
            httpx.ReadTimeout(
                "provider-timeout-secret",
                request=httpx.Request("GET", "http://qdrant.invalid"),
            )
        ),
        ResponseHandlingException(
            httpx.RemoteProtocolError(
                "provider-remote-protocol-secret",
                request=httpx.Request("GET", "http://qdrant.invalid"),
            )
        ),
        ResponseHandlingException(
            httpx.ProxyError(
                "provider-proxy-secret",
                request=httpx.Request("GET", "http://qdrant.invalid"),
            )
        ),
        ResponseHandlingException(
            httpx.DecodingError(
                "provider-decoding-secret",
                request=httpx.Request("GET", "http://qdrant.invalid"),
            )
        ),
        ResourceExhaustedResponse("provider-rate-limit-secret", 3),
    ],
)
async def test_qdrant_adapter_maps_real_transport_wrappers_and_429_to_safe_transient_error(
    wrapped_error: Exception,
) -> None:
    spec = CollectionSpec(collection_name(KB_ID, GENERATION_ID), 3, "cosine", ())

    class _Client:
        async def get_collection(self, _collection: str) -> None:
            return None

        async def collection_exists(self, _collection: str) -> bool:
            raise wrapped_error

    adapter = AsyncQdrantCollectionClient(cast(AsyncQdrantClient, _Client()))

    with pytest.raises(QdrantTransientError, match="^Qdrant unavailable$") as captured:
        await adapter.ensure_collection(spec)

    assert "secret" not in str(captured.value).lower()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "source",
    [
        ValueError("protocol-secret"),
        httpx.LocalProtocolError(
            "local-protocol-secret",
            request=httpx.Request("GET", "http://qdrant.invalid"),
        ),
        httpx.UnsupportedProtocol(
            "unsupported-protocol-secret",
            request=httpx.Request("GET", "qdrant://invalid"),
        ),
    ],
)
async def test_qdrant_adapter_maps_local_or_invalid_wrapped_errors_to_safe_permanent_error(
    source: Exception,
) -> None:
    spec = CollectionSpec(collection_name(KB_ID, GENERATION_ID), 3, "cosine", ())

    class _Client:
        async def get_collection(self, _collection: str) -> None:
            return None

        async def collection_exists(self, _collection: str) -> bool:
            raise ResponseHandlingException(source)

    adapter = AsyncQdrantCollectionClient(cast(AsyncQdrantClient, _Client()))

    with pytest.raises(QdrantConfigurationError) as captured:
        await adapter.ensure_collection(spec)

    assert "secret" not in str(captured.value).lower()


def test_qdrant_factory_preserves_fractional_request_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class _Client:
        def __init__(self, *, url: str, timeout: float) -> None:
            captured.update(url=url, timeout=timeout)

        async def get_collection(self, _collection: str) -> None:
            return None

        async def close(self) -> None:
            return None

    monkeypatch.setattr(qdrant_module, "AsyncQdrantClient", _Client)

    client = qdrant_client_from_url(
        "http://qdrant.example:6333",
        timeout_seconds=2.75,
    )

    assert isinstance(client, AsyncQdrantCollectionClient)
    assert captured == {
        "url": "http://qdrant.example:6333",
        "timeout": 2.75,
    }


@pytest.mark.asyncio
async def test_qdrant_adapter_parses_exact_count_and_rechecks_delete_ownership() -> None:
    class _Client:
        def __init__(self) -> None:
            self.deleted = False

        async def get_collection(self, _collection: str) -> object:
            return SimpleNamespace(
                config=SimpleNamespace(metadata={"managed_by": "another-service"})
            )

        async def count(self, **_kwargs: object) -> object:
            return SimpleNamespace(count=7)

        async def delete_collection(self, _collection: str) -> None:
            self.deleted = True

    raw = _Client()
    adapter = AsyncQdrantCollectionClient(cast(AsyncQdrantClient, raw))
    name = collection_name(KB_ID, GENERATION_ID)

    with pytest.raises(QdrantConfigurationError, match="not managed"):
        await adapter.count_points(name)
    with pytest.raises(QdrantConfigurationError, match="not managed"):
        await adapter.delete_collection(name)
    assert raw.deleted is False


@pytest.mark.asyncio
async def test_qdrant_adapter_rejects_legacy_managed_metadata_without_generation_identity() -> None:
    spec = CollectionSpec(collection_name(KB_ID, GENERATION_ID), 3, "cosine", ())

    class _Client:
        def __init__(self) -> None:
            self.mutated = False
            self.counted = False

        async def collection_exists(self, _collection: str) -> bool:
            return True

        async def get_collection(self, _collection: str) -> object:
            return SimpleNamespace(
                config=SimpleNamespace(
                    params=SimpleNamespace(
                        vectors=models.VectorParams(
                            size=3,
                            distance=models.Distance.COSINE,
                        )
                    ),
                    metadata={
                        "managed_by": "rag-service",
                        "created_at": NOW.isoformat(),
                    },
                ),
                payload_schema={},
            )

        async def create_payload_index(self, **_kwargs: object) -> None:
            self.mutated = True

        async def count(self, **_kwargs: object) -> object:
            self.counted = True
            return SimpleNamespace(count=0)

    operations: tuple[
        Callable[[AsyncQdrantCollectionClient], Awaitable[object]],
        ...,
    ] = (
        lambda adapter: adapter.ensure_collection(spec),
        lambda adapter: adapter.ensure_payload_indexes(spec.name, ()),
        lambda adapter: adapter.verify_collection(spec),
        lambda adapter: adapter.count_points(spec.name),
    )
    for operation in operations:
        raw = _Client()
        adapter = AsyncQdrantCollectionClient(cast(AsyncQdrantClient, raw))
        with pytest.raises(QdrantConfigurationError, match="not managed"):
            await operation(adapter)
        assert raw.mutated is False
        assert raw.counted is False


@pytest.mark.asyncio
async def test_qdrant_adapter_managed_listing_pages_raw_names_without_starvation() -> None:
    identifiers = [UUID(f"99999999-9999-4999-8999-{index:012d}") for index in range(1, 5)]
    managed_names = [collection_name(KB_ID, identifier) for identifier in identifiers]
    raw_names = ["aaa_unrelated", *managed_names]

    class _Client:
        async def get_collection(self, collection: str) -> object:
            parts = collection.split("_")
            knowledge_base_hex = parts[2]
            generation_hex = parts[4]
            return SimpleNamespace(
                config=SimpleNamespace(
                    metadata={
                        "managed_by": "rag-service",
                        "schema_version": "rag-index-generation-v1",
                        "collection_name": collection,
                        "knowledge_base_id": knowledge_base_hex,
                        "generation_id": generation_hex,
                        "created_at": NOW.isoformat(),
                    }
                )
            )

        async def get_collections(self) -> object:
            return SimpleNamespace(
                collections=[SimpleNamespace(name=name) for name in reversed(raw_names)]
            )

    adapter = AsyncQdrantCollectionClient(cast(AsyncQdrantClient, _Client()))
    cursor: str | None = None
    seen: list[str] = []
    pages = 0
    while True:
        page = await adapter.list_managed_collections(limit=2, cursor=cursor)
        assert len(page.items) <= 2
        seen.extend(item.name for item in page.items)
        pages += 1
        if page.next_cursor is None:
            break
        cursor = page.next_cursor

    assert pages == 3
    assert seen == managed_names
