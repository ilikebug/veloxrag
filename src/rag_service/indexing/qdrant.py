"""Qdrant collection protocol, async adapter, and deterministic fake."""

from __future__ import annotations

import asyncio
import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from types import MappingProxyType
from typing import Literal, Protocol, cast
from uuid import UUID

import httpx
from qdrant_client import AsyncQdrantClient
from qdrant_client.common.client_exceptions import ResourceExhaustedResponse
from qdrant_client.http import models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from rag_service.indexing.identities import canonical_sha256, point_id

_COLLECTION_PATTERN = re.compile(
    r"^rag_kb_(?P<knowledge_base_id>[0-9a-f]{32})_g_(?P<generation_id>[0-9a-f]{32})$"
)
_PAYLOAD_PATH_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_.]{0,254}$")
_PAYLOAD_SCHEMAS = frozenset({"keyword", "integer", "float", "bool", "datetime"})
_PAYLOAD_INDEX_DEFAULTS: dict[str, tuple[tuple[str, bool], ...]] = {
    "keyword": (("enable_hnsw", True), ("is_tenant", False), ("on_disk", False)),
    "integer": (
        ("enable_hnsw", True),
        ("is_principal", False),
        ("lookup", True),
        ("on_disk", False),
        ("range", True),
    ),
    "float": (("enable_hnsw", True), ("is_principal", False), ("on_disk", False)),
    "bool": (("enable_hnsw", True), ("on_disk", False)),
    "datetime": (("enable_hnsw", True), ("is_principal", False), ("on_disk", False)),
}
_DISTANCES = frozenset({"cosine", "dot", "euclid", "manhattan"})
_DATATYPES = frozenset({"float32", "float16", "uint8"})
_MANAGED_BY = "rag-service"
_MANAGED_SCHEMA_VERSION = "rag-index-generation-v1"
_MAX_COLLECTION_PAGE_SIZE = 100
_MAX_VERSION_POINTS = 10_000_000
_MAX_POINT_RETRIEVE_BATCH = 10_000
_POINT_PAYLOAD_KEYS = frozenset(
    {
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
)


def _safe_metadata_value(value: object) -> bool:
    if type(value) is str:
        return len(value) <= 4096 and "\x00" not in value
    if type(value) is int:
        return -(2**63) <= value <= 2**63 - 1
    if type(value) is bool:
        return True
    if type(value) is float:
        return math.isfinite(value)
    return False


_PayloadIndexParams = (
    models.KeywordIndexParams
    | models.IntegerIndexParams
    | models.FloatIndexParams
    | models.BoolIndexParams
    | models.DatetimeIndexParams
)


class QdrantConfigurationError(Exception):
    """A stable permanent mismatch between PostgreSQL facts and Qdrant state."""


class QdrantTransientError(Exception):
    """A stable retryable Qdrant availability failure."""


def _validated_point_payload(point: UUID, payload_value: Mapping[str, object]) -> dict[str, object]:
    try:
        payload = dict(payload_value)
        title_path = payload.get("title_path")
        metadata = payload.get("metadata")
        knowledge_base_id = UUID(cast(str, payload.get("knowledge_base_id")))
        document_id = UUID(cast(str, payload.get("document_id")))
        version_id = UUID(cast(str, payload.get("version_id")))
        chunk_index = cast(int, payload.get("chunk_index"))
        chunk_hash = cast(str, payload.get("chunk_hash"))
        if (
            type(point) is not UUID
            or set(payload) != _POINT_PAYLOAD_KEYS
            or payload["knowledge_base_id"] != str(knowledge_base_id)
            or payload["document_id"] != str(document_id)
            or payload["version_id"] != str(version_id)
            or type(chunk_index) is not int
            or chunk_index < 0
            or type(chunk_hash) is not str
            or re.fullmatch(r"[0-9a-f]{64}", chunk_hash) is None
            or point != point_id(version_id, chunk_index, chunk_hash)
            or type(payload["text"]) is not str
            or not payload["text"]
            or "\x00" in payload["text"]
            or type(title_path) is not list
            or any(type(value) is not str or not value or "\x00" in value for value in title_path)
            or type(payload["start_offset"]) is not int
            or type(payload["end_offset"]) is not int
            or payload["start_offset"] < 0
            or payload["end_offset"] <= payload["start_offset"]
            or payload["end_offset"] - payload["start_offset"] != len(payload["text"])
            or type(metadata) is not dict
            or any(
                type(key) is not str
                or re.fullmatch(r"f_[0-9a-f]{32}", key) is None
                or not _safe_metadata_value(value)
                for key, value in metadata.items()
            )
        ):
            raise ValueError
        copied = json.loads(json.dumps(payload, ensure_ascii=False, allow_nan=False))
        if type(copied) is not dict:
            raise ValueError
        return cast(dict[str, object], copied)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("Qdrant point is invalid") from None


@dataclass(frozen=True, slots=True)
class QdrantRetrievedPoint:
    id: UUID
    vector_dimension: int
    payload_digest_sha256: str

    def __post_init__(self) -> None:
        if (
            type(self.id) is not UUID
            or type(self.vector_dimension) is not int
            or not 1 <= self.vector_dimension <= 10_000_000
            or re.fullmatch(r"[0-9a-f]{64}", self.payload_digest_sha256) is None
        ):
            raise ValueError("Qdrant retrieved point is invalid")


@dataclass(frozen=True, slots=True, repr=False)
class QdrantPoint:
    """One deterministic point with an explicitly allowlisted payload."""

    id: UUID
    vector: tuple[float, ...]
    payload: Mapping[str, object]

    def __post_init__(self) -> None:
        try:
            if (
                type(self.id) is not UUID
                or type(self.vector) is not tuple
                or not self.vector
                or any(
                    type(value) is not float or not math.isfinite(value) for value in self.vector
                )
            ):
                raise ValueError
            copied = _validated_point_payload(self.id, self.payload)
            object.__setattr__(self, "payload", MappingProxyType(copied))
        except (TypeError, ValueError, OverflowError):
            raise ValueError("Qdrant point is invalid") from None

    def __repr__(self) -> str:
        return f"QdrantPoint(id={self.id!r}, vector_dimension={len(self.vector)})"


def _valid_search_value(field_type: str, value: object) -> bool:
    if field_type == "keyword":
        return type(value) is str and len(value) <= 4096 and "\x00" not in value
    if field_type == "integer":
        return type(value) is int and -(2**63) <= value <= 2**63 - 1
    if field_type == "float":
        try:
            return (
                type(value) in {int, float}
                and not isinstance(value, bool)
                and math.isfinite(float(cast(int | float, value)))
            )
        except OverflowError:
            return False
    if field_type == "boolean":
        return type(value) is bool
    if field_type == "datetime" and type(value) is str:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return False
        return parsed.tzinfo is not None
    return False


def _exact_qdrant_range_integer(value: object) -> bool:
    if type(value) is not int:
        return False
    try:
        converted = float(value)
    except OverflowError:
        return False
    return math.isfinite(converted) and int(converted) == value


@dataclass(frozen=True, slots=True)
class QdrantFilterCondition:
    path: str
    field_type: Literal["keyword", "integer", "float", "boolean", "datetime"]
    operator: Literal["eq", "in", "gte", "lte"]
    value: object

    def __post_init__(self) -> None:
        allowed = {
            "keyword": frozenset({"eq", "in"}),
            "integer": frozenset({"eq", "in", "gte", "lte"}),
            "float": frozenset({"eq", "in", "gte", "lte"}),
            "boolean": frozenset({"eq", "in"}),
            "datetime": frozenset({"eq", "in", "gte", "lte"}),
        }
        value = self.value
        if self.operator == "in":
            valid_value = (
                type(value) is tuple
                and 1 <= len(value) <= 100
                and all(_valid_search_value(self.field_type, item) for item in value)
            )
        else:
            valid_value = _valid_search_value(self.field_type, value)
        if (
            self.field_type == "integer"
            and self.operator in {"gte", "lte"}
            and not _exact_qdrant_range_integer(value)
        ):
            valid_value = False
        if (
            re.fullmatch(r"metadata\.f_[0-9a-f]{32}", self.path) is None
            or self.field_type not in allowed
            or self.operator not in allowed[self.field_type]
            or not valid_value
        ):
            raise ValueError("Qdrant filter condition is invalid")


@dataclass(frozen=True, slots=True)
class QdrantSearchFilter:
    document_ids: tuple[UUID, ...] = ()
    conditions: tuple[QdrantFilterCondition, ...] = ()

    def __post_init__(self) -> None:
        if (
            type(self.document_ids) is not tuple
            or len(self.document_ids) > 200
            or any(type(value) is not UUID for value in self.document_ids)
            or len(set(self.document_ids)) != len(self.document_ids)
            or type(self.conditions) is not tuple
            or len(self.conditions) > 64
            or any(type(value) is not QdrantFilterCondition for value in self.conditions)
            or len({value.path for value in self.conditions}) != len(self.conditions)
        ):
            raise ValueError("Qdrant search filter is invalid")


@dataclass(frozen=True, slots=True, repr=False)
class QdrantSearchPoint:
    id: UUID
    score: float
    payload: Mapping[str, object]

    def __post_init__(self) -> None:
        try:
            score = float(self.score)
            if (
                type(self.id) is not UUID
                or isinstance(self.score, bool)
                or not math.isfinite(score)
            ):
                raise ValueError
            payload = _validated_point_payload(self.id, self.payload)
            object.__setattr__(self, "score", score)
            object.__setattr__(self, "payload", MappingProxyType(payload))
        except (TypeError, ValueError, OverflowError):
            raise ValueError("Qdrant search point is invalid") from None

    def __repr__(self) -> str:
        return f"QdrantSearchPoint(id={self.id!r}, score={self.score!r})"


def qdrant_distance(distance: str) -> models.Distance:
    mapping = {
        "cosine": models.Distance.COSINE,
        "dot": models.Distance.DOT,
        "euclid": models.Distance.EUCLID,
        "manhattan": models.Distance.MANHATTAN,
    }
    if type(distance) is not str or distance not in mapping:
        raise ValueError("distance is invalid")
    return mapping[distance]


@dataclass(frozen=True, slots=True, order=True)
class PayloadIndex:
    path: str
    schema: str
    params: tuple[tuple[str, bool], ...] = ()

    def __post_init__(self) -> None:
        defaults = dict(_PAYLOAD_INDEX_DEFAULTS.get(self.schema, ()))
        provided = self.params
        if (
            type(self.path) is not str
            or _PAYLOAD_PATH_PATTERN.fullmatch(self.path) is None
            or self.path.startswith(".")
            or self.path.endswith(".")
            or ".." in self.path
            or self.schema not in _PAYLOAD_SCHEMAS
            or type(provided) is not tuple
            or any(
                type(item) is not tuple
                or len(item) != 2
                or type(item[0]) is not str
                or item[0] not in defaults
                or type(item[1]) is not bool
                for item in provided
            )
            or len({item[0] for item in provided}) != len(provided)
        ):
            raise ValueError("payload index is invalid")
        defaults.update(provided)
        object.__setattr__(self, "params", tuple(sorted(defaults.items())))


@dataclass(frozen=True, slots=True)
class CollectionHnswSpec:
    m: int = 16
    ef_construct: int = 100
    full_scan_threshold: int = 10_000
    max_indexing_threads: int = 0
    on_disk: bool = False
    payload_m: int = 16
    inline_storage: bool = False

    def __post_init__(self) -> None:
        if (
            type(self.m) is not int
            or self.m < 0
            or type(self.ef_construct) is not int
            or self.ef_construct < 0
            or type(self.full_scan_threshold) is not int
            or self.full_scan_threshold < 0
            or type(self.max_indexing_threads) is not int
            or self.max_indexing_threads < 0
            or type(self.on_disk) is not bool
            or type(self.payload_m) is not int
            or self.payload_m < 0
            or type(self.inline_storage) is not bool
        ):
            raise ValueError("collection HNSW specification is invalid")


@dataclass(frozen=True, slots=True)
class CollectionSpec:
    name: str
    dimension: int
    distance: str
    payload_indexes: tuple[PayloadIndex, ...]
    datatype: str = "float32"
    vector_on_disk: bool = False
    vector_hnsw_config: bool = False
    vector_quantization: bool = False
    multivector: bool = False
    sparse_vectors: tuple[str, ...] = ()
    on_disk_payload: bool = True
    collection_quantization: bool = False
    hnsw_config: CollectionHnswSpec = CollectionHnswSpec()

    def __post_init__(self) -> None:
        canonical_indexes = tuple(sorted(self.payload_indexes))
        canonical_sparse = tuple(sorted(self.sparse_vectors))
        if (
            type(self.name) is not str
            or _COLLECTION_PATTERN.fullmatch(self.name) is None
            or type(self.dimension) is not int
            or not 1 <= self.dimension <= 10_000_000
            or self.distance not in _DISTANCES
            or self.datatype not in _DATATYPES
            or type(self.vector_on_disk) is not bool
            or type(self.vector_hnsw_config) is not bool
            or type(self.vector_quantization) is not bool
            or type(self.multivector) is not bool
            or type(self.sparse_vectors) is not tuple
            or any(type(name) is not str or not name or "\x00" in name for name in canonical_sparse)
            or len(set(canonical_sparse)) != len(canonical_sparse)
            or type(self.on_disk_payload) is not bool
            or type(self.collection_quantization) is not bool
            or type(self.hnsw_config) is not CollectionHnswSpec
            or any(type(index) is not PayloadIndex for index in canonical_indexes)
            or len({index.path for index in canonical_indexes}) != len(canonical_indexes)
        ):
            raise ValueError("collection specification is invalid")
        object.__setattr__(self, "payload_indexes", canonical_indexes)
        object.__setattr__(self, "sparse_vectors", canonical_sparse)

    def vector_only(self) -> CollectionSpec:
        return CollectionSpec(
            self.name,
            self.dimension,
            self.distance,
            (),
            datatype=self.datatype,
            vector_on_disk=self.vector_on_disk,
            vector_hnsw_config=self.vector_hnsw_config,
            vector_quantization=self.vector_quantization,
            multivector=self.multivector,
            sparse_vectors=self.sparse_vectors,
            on_disk_payload=self.on_disk_payload,
            collection_quantization=self.collection_quantization,
            hnsw_config=self.hnsw_config,
        )


@dataclass(frozen=True, slots=True)
class ManagedCollection:
    name: str
    created_at: datetime

    def __post_init__(self) -> None:
        if (
            _COLLECTION_PATTERN.fullmatch(self.name) is None
            or self.created_at.tzinfo is None
            or self.created_at.utcoffset() != UTC.utcoffset(self.created_at)
        ):
            raise ValueError("managed collection is invalid")


@dataclass(frozen=True, slots=True)
class ManagedCollectionPage:
    items: tuple[ManagedCollection, ...]
    next_cursor: str | None

    def __post_init__(self) -> None:
        if (
            any(type(item) is not ManagedCollection for item in self.items)
            or len(self.items) > _MAX_COLLECTION_PAGE_SIZE
            or (
                self.next_cursor is not None
                and (
                    type(self.next_cursor) is not str
                    or not self.next_cursor
                    or len(self.next_cursor) > 255
                    or "\x00" in self.next_cursor
                )
            )
        ):
            raise ValueError("managed collection page is invalid")


class QdrantClient(Protocol):
    async def collection_exists(self, collection: str) -> bool: ...

    async def ensure_collection(self, spec: CollectionSpec) -> None: ...

    async def ensure_payload_indexes(
        self,
        collection: str,
        indexes: Sequence[PayloadIndex],
    ) -> None: ...

    async def verify_collection(self, spec: CollectionSpec) -> None: ...

    async def upsert_points(self, collection: str, points: Sequence[QdrantPoint]) -> None: ...

    async def count_points(self, collection: str) -> int: ...

    async def count_version_points(self, collection: str, version_id: UUID) -> int: ...

    async def retrieve_version_points(
        self,
        collection: str,
        version_id: UUID,
        point_ids: Sequence[UUID],
    ) -> tuple[QdrantRetrievedPoint, ...]: ...

    async def search_points(
        self,
        collection: str,
        vector: Sequence[float],
        *,
        limit: int,
        query_filter: QdrantSearchFilter,
    ) -> tuple[QdrantSearchPoint, ...]: ...

    async def describe_collection(self, collection: str) -> CollectionSpec: ...

    async def list_managed_collections(
        self,
        *,
        limit: int,
        cursor: str | None,
    ) -> ManagedCollectionPage: ...

    async def delete_collection(self, collection: str) -> None: ...

    async def aclose(self) -> None: ...


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _version_filter(version_id: UUID) -> models.Filter:
    if type(version_id) is not UUID:
        raise ValueError("Qdrant version identity is invalid")
    return models.Filter(
        must=[
            models.FieldCondition(
                key="version_id",
                match=models.MatchValue(value=str(version_id)),
            )
        ]
    )


def _search_datetime(value: object) -> datetime:
    if type(value) is not str:
        raise ValueError("Qdrant search filter is invalid")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise ValueError("Qdrant search filter is invalid") from None
    if parsed.tzinfo is None:
        raise ValueError("Qdrant search filter is invalid")
    return parsed


def _exact_range_condition(
    condition: QdrantFilterCondition,
    value: object,
) -> models.FieldCondition:
    if condition.field_type == "datetime":
        parsed = _search_datetime(value)
        return models.FieldCondition(
            key=condition.path,
            range=models.DatetimeRange(gte=parsed, lte=parsed),
        )
    numeric = float(cast(int | float, value))
    return models.FieldCondition(
        key=condition.path,
        range=models.Range(gte=numeric, lte=numeric),
    )


def _search_condition(
    condition: QdrantFilterCondition,
) -> models.FieldCondition | models.Filter:
    value = condition.value
    if condition.operator == "eq":
        if condition.field_type in {"float", "datetime"}:
            return _exact_range_condition(condition, value)
        return models.FieldCondition(
            key=condition.path,
            match=models.MatchValue(value=cast(bool | int | str, value)),
        )
    if condition.operator == "in":
        values = cast(tuple[object, ...], value)
        if condition.field_type in {"float", "datetime", "boolean"}:
            return models.Filter(
                should=[
                    (
                        models.FieldCondition(
                            key=condition.path,
                            match=models.MatchValue(value=cast(bool, item)),
                        )
                        if condition.field_type == "boolean"
                        else _exact_range_condition(condition, item)
                    )
                    for item in values
                ]
            )
        return models.FieldCondition(
            key=condition.path,
            match=models.MatchAny(any=cast(list[bool | int | str], list(values))),
        )
    if condition.field_type == "datetime":
        parsed = _search_datetime(value)
        return models.FieldCondition(
            key=condition.path,
            range=models.DatetimeRange(
                gte=parsed if condition.operator == "gte" else None,
                lte=parsed if condition.operator == "lte" else None,
            ),
        )
    numeric = float(cast(int | float, value))
    return models.FieldCondition(
        key=condition.path,
        range=models.Range(
            gte=numeric if condition.operator == "gte" else None,
            lte=numeric if condition.operator == "lte" else None,
        ),
    )


def _search_filter(value: QdrantSearchFilter) -> models.Filter | None:
    if type(value) is not QdrantSearchFilter:
        raise ValueError("Qdrant search filter is invalid")
    must: list[models.Condition] = []
    if value.document_ids:
        must.append(
            models.FieldCondition(
                key="document_id",
                match=models.MatchAny(any=[str(identifier) for identifier in value.document_ids]),
            )
        )
    must.extend(_search_condition(condition) for condition in value.conditions)
    return None if not must else models.Filter(must=must)


def _record_uuid(value: object) -> UUID:
    if type(value) is UUID:
        return value
    if type(value) is str:
        try:
            parsed = UUID(value)
        except ValueError:
            raise QdrantConfigurationError("Qdrant point retrieve response is invalid") from None
        if value != str(parsed):
            raise QdrantConfigurationError("Qdrant point retrieve response is invalid")
        return parsed
    raise QdrantConfigurationError("Qdrant point retrieve response is invalid")


def _retrieved_point(record: object, version_id: UUID) -> QdrantRetrievedPoint:
    if not isinstance(record, models.Record):
        raise QdrantConfigurationError("Qdrant point retrieve response is invalid")
    point = _record_uuid(record.id)
    vector = record.vector
    if (
        type(vector) is not list
        or not vector
        or any(type(value) is not float or not math.isfinite(value) for value in vector)
        or not isinstance(record.payload, dict)
    ):
        raise QdrantConfigurationError("Qdrant point retrieve response is invalid")
    try:
        payload = _validated_point_payload(point, record.payload)
    except ValueError:
        raise QdrantConfigurationError("Qdrant point retrieve response is invalid") from None
    if payload["version_id"] != str(version_id):
        raise QdrantConfigurationError("Qdrant point retrieve response is invalid")
    return QdrantRetrievedPoint(
        id=point,
        vector_dimension=len(vector),
        payload_digest_sha256=canonical_sha256(payload),
    )


def _collection_identity(collection: str) -> tuple[str, str]:
    match = _COLLECTION_PATTERN.fullmatch(collection)
    if match is None:
        raise ValueError("collection identity is invalid")
    return match.group("knowledge_base_id"), match.group("generation_id")


def managed_collection_identity(collection: str) -> tuple[UUID, UUID]:
    knowledge_base_id, generation_id = _collection_identity(collection)
    return UUID(hex=knowledge_base_id), UUID(hex=generation_id)


def _metadata(collection: str, created_at: datetime) -> dict[str, str]:
    knowledge_base_id, generation_id = _collection_identity(collection)
    return {
        "managed_by": _MANAGED_BY,
        "schema_version": _MANAGED_SCHEMA_VERSION,
        "collection_name": collection,
        "knowledge_base_id": knowledge_base_id,
        "generation_id": generation_id,
        "created_at": created_at.isoformat().replace("+00:00", "Z"),
    }


def _parse_created_at(value: object) -> datetime | None:
    if type(value) is not str:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        return None
    return parsed


def _managed_created_at(collection: str, metadata: object) -> datetime:
    try:
        knowledge_base_id, generation_id = _collection_identity(collection)
    except ValueError:
        raise QdrantConfigurationError("Qdrant collection is not managed") from None
    if not isinstance(metadata, dict):
        raise QdrantConfigurationError("Qdrant collection is not managed")
    expected = {
        "managed_by": _MANAGED_BY,
        "schema_version": _MANAGED_SCHEMA_VERSION,
        "collection_name": collection,
        "knowledge_base_id": knowledge_base_id,
        "generation_id": generation_id,
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise QdrantConfigurationError("Qdrant collection is not managed")
    created_at = _parse_created_at(metadata.get("created_at"))
    if created_at is None:
        raise QdrantConfigurationError("Qdrant collection is not managed")
    return created_at


def _validate_page(limit: int, cursor: str | None) -> None:
    if (
        type(limit) is not int
        or not 1 <= limit <= _MAX_COLLECTION_PAGE_SIZE
        or (
            cursor is not None
            and (type(cursor) is not str or not cursor or len(cursor) > 255 or "\x00" in cursor)
        )
    ):
        raise ValueError("managed collection page limit or cursor is invalid")


def _payload_schema(index: PayloadIndex) -> models.PayloadFieldSchema:
    params = dict(index.params)
    if index.schema == "keyword":
        return models.KeywordIndexParams(
            type=models.KeywordIndexType.KEYWORD,
            is_tenant=params["is_tenant"],
            on_disk=params["on_disk"],
            enable_hnsw=params["enable_hnsw"],
        )
    if index.schema == "integer":
        return models.IntegerIndexParams(
            type=models.IntegerIndexType.INTEGER,
            lookup=params["lookup"],
            range=params["range"],
            is_principal=params["is_principal"],
            on_disk=params["on_disk"],
            enable_hnsw=params["enable_hnsw"],
        )
    if index.schema == "float":
        return models.FloatIndexParams(
            type=models.FloatIndexType.FLOAT,
            is_principal=params["is_principal"],
            on_disk=params["on_disk"],
            enable_hnsw=params["enable_hnsw"],
        )
    if index.schema == "bool":
        return models.BoolIndexParams(
            type=models.BoolIndexType.BOOL,
            on_disk=params["on_disk"],
            enable_hnsw=params["enable_hnsw"],
        )
    if index.schema == "datetime":
        return models.DatetimeIndexParams(
            type=models.DatetimeIndexType.DATETIME,
            is_principal=params["is_principal"],
            on_disk=params["on_disk"],
            enable_hnsw=params["enable_hnsw"],
        )
    raise ValueError("payload schema is invalid")


def _model_has_values(value: object) -> bool:
    if value is None:
        return False
    dump = getattr(value, "model_dump", None)
    return bool(dump(exclude_none=True)) if callable(dump) else True


def _payload_index(path: str, payload: object) -> PayloadIndex:
    data_type = getattr(payload, "data_type", None)
    value = getattr(data_type, "value", None)
    if type(value) is not str:
        raise QdrantConfigurationError("Qdrant collection does not match requested configuration")
    schema = value.lower()
    model_by_schema: dict[str, type[object]] = {
        "keyword": models.KeywordIndexParams,
        "integer": models.IntegerIndexParams,
        "float": models.FloatIndexParams,
        "bool": models.BoolIndexParams,
        "datetime": models.DatetimeIndexParams,
    }
    expected_model = model_by_schema.get(schema)
    if expected_model is None:
        raise QdrantConfigurationError("Qdrant collection does not match requested configuration")
    params = getattr(payload, "params", None)
    if params is None:
        return PayloadIndex(path, schema)
    if type(params) is not expected_model:
        raise QdrantConfigurationError("Qdrant collection does not match requested configuration")
    dumped = cast(_PayloadIndexParams, params).model_dump(exclude_none=False)
    expected_defaults = dict(_PAYLOAD_INDEX_DEFAULTS[schema])
    actual = tuple(
        (name, expected_defaults[name] if dumped[name] is None else dumped[name])
        for name in expected_defaults
    )
    return PayloadIndex(path, schema, actual)


def _safe_unexpected(error: UnexpectedResponse) -> Exception:
    status_code = error.status_code
    if status_code is None or status_code >= 500 or status_code in {408, 429}:
        return QdrantTransientError("Qdrant unavailable")
    return QdrantConfigurationError("Qdrant collection does not match requested configuration")


def _safe_response_handling(error: ResponseHandlingException) -> Exception:
    if isinstance(
        error.source,
        (
            httpx.TimeoutException,
            httpx.NetworkError,
            httpx.RemoteProtocolError,
            httpx.ProxyError,
            httpx.DecodingError,
            TimeoutError,
            ConnectionError,
            OSError,
        ),
    ):
        return QdrantTransientError("Qdrant unavailable")
    return QdrantConfigurationError("Qdrant response is invalid")


class AsyncQdrantCollectionClient:
    """Async Qdrant adapter that never deletes a mismatched collection."""

    def __init__(
        self,
        client: AsyncQdrantClient,
        *,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        if not callable(getattr(client, "get_collection", None)) or not callable(clock):
            raise ValueError("Qdrant dependencies are invalid")
        self._client = client
        self._clock = clock

    async def collection_exists(self, collection: str) -> bool:
        if _COLLECTION_PATTERN.fullmatch(collection) is None:
            raise ValueError("collection identity is invalid")
        try:
            exists = await self._client.collection_exists(collection)
            if type(exists) is not bool:
                raise QdrantConfigurationError("Qdrant collection existence response is invalid")
            return exists
        except (QdrantConfigurationError, QdrantTransientError):
            raise
        except UnexpectedResponse as error:
            raise _safe_unexpected(error) from None
        except ResourceExhaustedResponse:
            raise QdrantTransientError("Qdrant unavailable") from None
        except ResponseHandlingException as error:
            raise _safe_response_handling(error) from None
        except (httpx.TimeoutException, httpx.NetworkError, OSError):
            raise QdrantTransientError("Qdrant unavailable") from None
        except (AttributeError, TypeError, ValueError):
            raise QdrantConfigurationError(
                "Qdrant collection existence response is invalid"
            ) from None

    async def ensure_collection(self, spec: CollectionSpec) -> None:
        if spec.payload_indexes:
            raise ValueError("ensure_collection requires a vector-only specification")
        try:
            exists = await self._client.collection_exists(spec.name)
            if not exists:
                created_at = self._clock()
                if created_at.tzinfo is None or created_at.utcoffset() != UTC.utcoffset(created_at):
                    raise ValueError("Qdrant clock is invalid")
                try:
                    await self._client.create_collection(
                        collection_name=spec.name,
                        vectors_config=models.VectorParams(
                            size=spec.dimension,
                            distance=qdrant_distance(spec.distance),
                            datatype=models.Datatype.FLOAT32,
                            on_disk=False,
                            hnsw_config=None,
                            quantization_config=None,
                            multivector_config=None,
                        ),
                        sparse_vectors_config={},
                        on_disk_payload=True,
                        hnsw_config=models.HnswConfigDiff(
                            m=spec.hnsw_config.m,
                            ef_construct=spec.hnsw_config.ef_construct,
                            full_scan_threshold=spec.hnsw_config.full_scan_threshold,
                            max_indexing_threads=spec.hnsw_config.max_indexing_threads,
                            on_disk=spec.hnsw_config.on_disk,
                            payload_m=spec.hnsw_config.payload_m,
                            inline_storage=spec.hnsw_config.inline_storage,
                        ),
                        quantization_config=None,
                        metadata=_metadata(spec.name, created_at),
                    )
                except UnexpectedResponse as error:
                    if error.status_code != 409:
                        raise _safe_unexpected(error) from None
            described = await self.describe_collection(spec.name)
            if described.vector_only() != spec:
                raise QdrantConfigurationError(
                    "Qdrant collection does not match requested configuration"
                )
        except (QdrantConfigurationError, QdrantTransientError):
            raise
        except UnexpectedResponse as error:
            raise _safe_unexpected(error) from None
        except ResourceExhaustedResponse:
            raise QdrantTransientError("Qdrant unavailable") from None
        except ResponseHandlingException as error:
            raise _safe_response_handling(error) from None
        except (httpx.TimeoutException, httpx.NetworkError, OSError):
            raise QdrantTransientError("Qdrant unavailable") from None

    async def ensure_payload_indexes(
        self,
        collection: str,
        indexes: Sequence[PayloadIndex],
    ) -> None:
        expected = tuple(sorted(indexes))
        if len({index.path for index in expected}) != len(expected):
            raise ValueError("payload index specification is invalid")
        try:
            current = await self.describe_collection(collection)
            current_by_path = {index.path: index for index in current.payload_indexes}
            expected_by_path = {index.path: index for index in expected}
            for path, index in current_by_path.items():
                if path not in expected_by_path or expected_by_path[path] != index:
                    raise QdrantConfigurationError(
                        "Qdrant collection does not match requested configuration"
                    )
            for path, index in expected_by_path.items():
                if path in current_by_path:
                    continue
                try:
                    await self._client.create_payload_index(
                        collection_name=collection,
                        field_name=path,
                        field_schema=_payload_schema(index),
                        wait=True,
                    )
                except UnexpectedResponse as error:
                    if error.status_code != 409:
                        raise
            actual = await self.describe_collection(collection)
            if actual.payload_indexes != expected:
                raise QdrantConfigurationError(
                    "Qdrant collection does not match requested configuration"
                )
        except (QdrantConfigurationError, QdrantTransientError):
            raise
        except UnexpectedResponse as error:
            raise _safe_unexpected(error) from None
        except ResourceExhaustedResponse:
            raise QdrantTransientError("Qdrant unavailable") from None
        except ResponseHandlingException as error:
            raise _safe_response_handling(error) from None
        except (httpx.TimeoutException, httpx.NetworkError, OSError):
            raise QdrantTransientError("Qdrant unavailable") from None

    async def verify_collection(self, spec: CollectionSpec) -> None:
        actual = await self.describe_collection(spec.name)
        if actual != spec:
            raise QdrantConfigurationError(
                "Qdrant collection does not match requested configuration"
            )

    async def upsert_points(self, collection: str, points: Sequence[QdrantPoint]) -> None:
        batch = tuple(points)
        if (
            _COLLECTION_PATTERN.fullmatch(collection) is None
            or not 1 <= len(batch) <= 10_000
            or any(type(point) is not QdrantPoint for point in batch)
            or len({point.id for point in batch}) != len(batch)
        ):
            raise ValueError("Qdrant point batch is invalid")
        try:
            await self._managed_info(collection)
            result = await self._client.upsert(
                collection_name=collection,
                points=[
                    models.PointStruct(
                        id=str(point.id),
                        vector=list(point.vector),
                        payload=dict(point.payload),
                    )
                    for point in batch
                ],
                wait=True,
            )
            if (
                not isinstance(result, models.UpdateResult)
                or result.status != models.UpdateStatus.COMPLETED
            ):
                if isinstance(result, models.UpdateResult):
                    raise QdrantTransientError("Qdrant upsert did not complete")
                raise QdrantConfigurationError("Qdrant point upsert response is invalid")
        except (QdrantConfigurationError, QdrantTransientError):
            raise
        except UnexpectedResponse as error:
            raise _safe_unexpected(error) from None
        except ResourceExhaustedResponse:
            raise QdrantTransientError("Qdrant unavailable") from None
        except ResponseHandlingException as error:
            raise _safe_response_handling(error) from None
        except (httpx.TimeoutException, httpx.NetworkError, OSError):
            raise QdrantTransientError("Qdrant unavailable") from None
        except (AttributeError, TypeError, ValueError):
            raise QdrantConfigurationError("Qdrant point upsert is invalid") from None

    async def count_points(self, collection: str) -> int:
        try:
            await self._managed_info(collection)
            result = await self._client.count(
                collection_name=collection,
                exact=True,
            )
            count = result.count
            if type(count) is not int or count < 0:
                raise QdrantConfigurationError("Qdrant point count is invalid")
            return count
        except (QdrantConfigurationError, QdrantTransientError):
            raise
        except UnexpectedResponse as error:
            raise _safe_unexpected(error) from None
        except ResourceExhaustedResponse:
            raise QdrantTransientError("Qdrant unavailable") from None
        except ResponseHandlingException as error:
            raise _safe_response_handling(error) from None
        except (httpx.TimeoutException, httpx.NetworkError, OSError):
            raise QdrantTransientError("Qdrant unavailable") from None
        except (AttributeError, TypeError, ValueError):
            raise QdrantConfigurationError("Qdrant point count is invalid") from None

    async def count_version_points(self, collection: str, version_id: UUID) -> int:
        if _COLLECTION_PATTERN.fullmatch(collection) is None or type(version_id) is not UUID:
            raise ValueError("Qdrant version identity is invalid")
        try:
            await self._managed_info(collection)
            result = await self._client.count(
                collection_name=collection,
                count_filter=_version_filter(version_id),
                exact=True,
            )
            count = result.count
            if type(count) is not int or not 0 <= count <= _MAX_VERSION_POINTS:
                raise QdrantConfigurationError("Qdrant point count is invalid")
            return count
        except (QdrantConfigurationError, QdrantTransientError):
            raise
        except UnexpectedResponse as error:
            raise _safe_unexpected(error) from None
        except ResourceExhaustedResponse:
            raise QdrantTransientError("Qdrant unavailable") from None
        except ResponseHandlingException as error:
            raise _safe_response_handling(error) from None
        except (httpx.TimeoutException, httpx.NetworkError, OSError):
            raise QdrantTransientError("Qdrant unavailable") from None
        except (AttributeError, TypeError, ValueError):
            raise QdrantConfigurationError("Qdrant point count is invalid") from None

    async def retrieve_version_points(
        self,
        collection: str,
        version_id: UUID,
        point_ids: Sequence[UUID],
    ) -> tuple[QdrantRetrievedPoint, ...]:
        requested = tuple(point_ids)
        if (
            _COLLECTION_PATTERN.fullmatch(collection) is None
            or type(version_id) is not UUID
            or not 1 <= len(requested) <= _MAX_POINT_RETRIEVE_BATCH
            or any(type(point) is not UUID for point in requested)
            or len(set(requested)) != len(requested)
        ):
            raise ValueError("Qdrant point retrieve request is invalid")
        try:
            await self._managed_info(collection)
            records = await self._client.retrieve(
                collection_name=collection,
                ids=[str(point) for point in requested],
                with_payload=True,
                with_vectors=True,
            )
            if type(records) is not list or len(records) != len(requested):
                raise QdrantConfigurationError("Qdrant point retrieve response is invalid")
            by_id: dict[UUID, QdrantRetrievedPoint] = {}
            expected_ids = set(requested)
            for record in records:
                inspected = _retrieved_point(record, version_id)
                if inspected.id not in expected_ids or inspected.id in by_id:
                    raise QdrantConfigurationError("Qdrant point retrieve response is invalid")
                by_id[inspected.id] = inspected
            if set(by_id) != expected_ids:
                raise QdrantConfigurationError("Qdrant point retrieve response is invalid")
            return tuple(by_id[point] for point in requested)
        except (QdrantConfigurationError, QdrantTransientError):
            raise
        except UnexpectedResponse as error:
            raise _safe_unexpected(error) from None
        except ResourceExhaustedResponse:
            raise QdrantTransientError("Qdrant unavailable") from None
        except ResponseHandlingException as error:
            raise _safe_response_handling(error) from None
        except (httpx.TimeoutException, httpx.NetworkError, OSError):
            raise QdrantTransientError("Qdrant unavailable") from None
        except (AttributeError, TypeError, ValueError):
            raise QdrantConfigurationError("Qdrant point retrieve response is invalid") from None

    async def search_points(
        self,
        collection: str,
        vector: Sequence[float],
        *,
        limit: int,
        query_filter: QdrantSearchFilter,
    ) -> tuple[QdrantSearchPoint, ...]:
        query_vector = tuple(vector)
        if (
            _COLLECTION_PATTERN.fullmatch(collection) is None
            or not query_vector
            or any(type(value) is not float or not math.isfinite(value) for value in query_vector)
            or type(limit) is not int
            or not 1 <= limit <= 200
            or type(query_filter) is not QdrantSearchFilter
        ):
            raise ValueError("Qdrant search request is invalid")
        expected_knowledge_base_id, _generation_id = managed_collection_identity(collection)
        try:
            await self._managed_info(collection)
            response = await self._client.query_points(
                collection_name=collection,
                query=list(query_vector),
                query_filter=_search_filter(query_filter),
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )
            if not isinstance(response, models.QueryResponse) or not isinstance(
                response.points, list
            ):
                raise QdrantConfigurationError("Qdrant search response is invalid")
            if len(response.points) > limit:
                raise QdrantConfigurationError("Qdrant search response is invalid")
            results: list[QdrantSearchPoint] = []
            seen: set[UUID] = set()
            for raw in response.points:
                if not isinstance(raw, models.ScoredPoint) or not isinstance(raw.payload, dict):
                    raise QdrantConfigurationError("Qdrant search response is invalid")
                identifier = _record_uuid(raw.id)
                try:
                    point = QdrantSearchPoint(identifier, raw.score, raw.payload)
                except ValueError:
                    raise QdrantConfigurationError("Qdrant search response is invalid") from None
                if identifier in seen or point.payload["knowledge_base_id"] != str(
                    expected_knowledge_base_id
                ):
                    raise QdrantConfigurationError("Qdrant search response is invalid")
                seen.add(identifier)
                results.append(point)
            return tuple(results)
        except (QdrantConfigurationError, QdrantTransientError):
            raise
        except UnexpectedResponse as error:
            raise _safe_unexpected(error) from None
        except ResourceExhaustedResponse:
            raise QdrantTransientError("Qdrant unavailable") from None
        except ResponseHandlingException as error:
            raise _safe_response_handling(error) from None
        except (httpx.TimeoutException, httpx.NetworkError, OSError):
            raise QdrantTransientError("Qdrant unavailable") from None
        except (AttributeError, TypeError, ValueError):
            raise QdrantConfigurationError("Qdrant search response is invalid") from None

    async def describe_collection(self, collection: str) -> CollectionSpec:
        try:
            info = await self._managed_info(collection)
            vectors = info.config.params.vectors
            if not isinstance(vectors, models.VectorParams):
                raise QdrantConfigurationError(
                    "Qdrant collection does not match requested configuration"
                )
            reverse_distance = {
                models.Distance.COSINE: "cosine",
                models.Distance.DOT: "dot",
                models.Distance.EUCLID: "euclid",
                models.Distance.MANHATTAN: "manhattan",
            }
            try:
                distance = reverse_distance[vectors.distance]
            except KeyError:
                raise QdrantConfigurationError(
                    "Qdrant collection does not match requested configuration"
                ) from None
            datatype = vectors.datatype
            if datatype is None or datatype == models.Datatype.FLOAT32:
                normalized_datatype = "float32"
            else:
                normalized_datatype = datatype.value
            sparse_vectors = getattr(info.config.params, "sparse_vectors", None) or {}
            if not isinstance(sparse_vectors, dict):
                raise QdrantConfigurationError(
                    "Qdrant collection does not match requested configuration"
                )
            indexes = tuple(
                _payload_index(path, payload)
                for path, payload in sorted(info.payload_schema.items())
            )
            raw_hnsw = getattr(info.config, "hnsw_config", None)
            if raw_hnsw is None:
                hnsw_config = CollectionHnswSpec()
            elif isinstance(raw_hnsw, models.HnswConfig):
                hnsw_config = CollectionHnswSpec(
                    m=raw_hnsw.m,
                    ef_construct=raw_hnsw.ef_construct,
                    full_scan_threshold=raw_hnsw.full_scan_threshold,
                    max_indexing_threads=raw_hnsw.max_indexing_threads or 0,
                    on_disk=raw_hnsw.on_disk is True,
                    payload_m=raw_hnsw.m if raw_hnsw.payload_m is None else raw_hnsw.payload_m,
                    inline_storage=raw_hnsw.inline_storage is True,
                )
            else:
                raise QdrantConfigurationError(
                    "Qdrant collection does not match requested configuration"
                )
            return CollectionSpec(
                collection,
                vectors.size,
                distance,
                indexes,
                datatype=normalized_datatype,
                vector_on_disk=vectors.on_disk is True,
                vector_hnsw_config=_model_has_values(vectors.hnsw_config),
                vector_quantization=vectors.quantization_config is not None,
                multivector=vectors.multivector_config is not None,
                sparse_vectors=tuple(sparse_vectors),
                on_disk_payload=getattr(info.config.params, "on_disk_payload", None) is not False,
                collection_quantization=(
                    getattr(info.config, "quantization_config", None) is not None
                ),
                hnsw_config=hnsw_config,
            )
        except (QdrantConfigurationError, QdrantTransientError):
            raise
        except UnexpectedResponse as error:
            raise _safe_unexpected(error) from None
        except ResourceExhaustedResponse:
            raise QdrantTransientError("Qdrant unavailable") from None
        except ResponseHandlingException as error:
            raise _safe_response_handling(error) from None
        except (httpx.TimeoutException, httpx.NetworkError, OSError):
            raise QdrantTransientError("Qdrant unavailable") from None
        except (AttributeError, TypeError, ValueError):
            raise QdrantConfigurationError(
                "Qdrant collection does not match requested configuration"
            ) from None

    async def _managed_info(self, collection: str) -> models.CollectionInfo:
        info = await self._client.get_collection(collection)
        try:
            metadata = info.config.metadata
        except AttributeError:
            raise QdrantConfigurationError("Qdrant collection is not managed") from None
        _managed_created_at(collection, metadata)
        return info

    async def list_managed_collections(
        self,
        *,
        limit: int,
        cursor: str | None,
    ) -> ManagedCollectionPage:
        _validate_page(limit, cursor)
        managed: list[ManagedCollection] = []
        try:
            # Qdrant's collection-list API has no server-side cursor. Keep this to
            # one full-name fetch per reconciliation page, then bound metadata reads
            # and downstream database work to the caller's hard page limit.
            response = await self._client.get_collections()
            descriptions = [
                description
                for description in sorted(response.collections, key=lambda item: item.name)
                if cursor is None or description.name > cursor
            ]
            page = descriptions[:limit]
            for description in page:
                if _COLLECTION_PATTERN.fullmatch(description.name) is None:
                    continue
                info = await self._client.get_collection(description.name)
                try:
                    created_at = _managed_created_at(
                        description.name,
                        info.config.metadata,
                    )
                except (AttributeError, QdrantConfigurationError):
                    continue
                managed.append(ManagedCollection(description.name, created_at))
            next_cursor = page[-1].name if len(descriptions) > limit and page else None
            return ManagedCollectionPage(tuple(managed), next_cursor)
        except UnexpectedResponse as error:
            raise _safe_unexpected(error) from None
        except ResourceExhaustedResponse:
            raise QdrantTransientError("Qdrant unavailable") from None
        except ResponseHandlingException as error:
            raise _safe_response_handling(error) from None
        except (httpx.TimeoutException, httpx.NetworkError, OSError):
            raise QdrantTransientError("Qdrant unavailable") from None

    async def delete_collection(self, collection: str) -> None:
        if _COLLECTION_PATTERN.fullmatch(collection) is None:
            raise ValueError("collection is not managed")
        try:
            await self._managed_info(collection)
            await self._client.delete_collection(collection)
        except QdrantConfigurationError:
            raise
        except UnexpectedResponse as error:
            if error.status_code != 404:
                raise _safe_unexpected(error) from None
        except ResourceExhaustedResponse:
            raise QdrantTransientError("Qdrant unavailable") from None
        except ResponseHandlingException as error:
            raise _safe_response_handling(error) from None
        except (httpx.TimeoutException, httpx.NetworkError, OSError):
            raise QdrantTransientError("Qdrant unavailable") from None

    async def aclose(self) -> None:
        await self._client.close()


@dataclass(slots=True)
class _FakeCollection:
    dimension: int
    distance: str
    datatype: str
    vector_on_disk: bool
    vector_hnsw_config: bool
    vector_quantization: bool
    multivector: bool
    sparse_vectors: tuple[str, ...]
    on_disk_payload: bool
    collection_quantization: bool
    hnsw_config: CollectionHnswSpec
    payload_indexes: dict[str, PayloadIndex]
    created_at: datetime
    point_count: int
    managed: bool
    identity_name: str
    schema_version: str
    points: dict[UUID, QdrantPoint]


def _fake_condition_value(payload: Mapping[str, object], path: str) -> object:
    current: object = payload
    for segment in path.split("."):
        if not isinstance(current, Mapping) or segment not in current:
            return None
        current = current[segment]
    return current


def _fake_comparable(field_type: str, value: object) -> object:
    if field_type == "datetime":
        return _search_datetime(value)
    if field_type == "float":
        return float(cast(int | float, value))
    return value


def _fake_matches_condition(
    payload: Mapping[str, object],
    condition: QdrantFilterCondition,
) -> bool:
    actual = _fake_condition_value(payload, condition.path)
    if actual is None or not _valid_search_value(condition.field_type, actual):
        return False
    expected = condition.value
    if condition.operator == "in":
        return _fake_comparable(condition.field_type, actual) in {
            _fake_comparable(condition.field_type, item)
            for item in cast(tuple[object, ...], expected)
        }
    left = _fake_comparable(condition.field_type, actual)
    right = _fake_comparable(condition.field_type, expected)
    if condition.operator == "eq":
        return left == right
    if condition.field_type == "datetime":
        datetime_left = cast(datetime, left)
        datetime_right = cast(datetime, right)
        return (
            datetime_left >= datetime_right
            if condition.operator == "gte"
            else datetime_left <= datetime_right
        )
    if condition.field_type == "integer":
        integer_left = cast(int, left)
        integer_right = cast(int, right)
        return (
            integer_left >= integer_right
            if condition.operator == "gte"
            else integer_left <= integer_right
        )
    if condition.field_type == "float":
        float_left = cast(float, left)
        float_right = cast(float, right)
        return (
            float_left >= float_right if condition.operator == "gte" else float_left <= float_right
        )
    raise QdrantConfigurationError("Qdrant search filter is invalid")


def _fake_matches_filter(payload: Mapping[str, object], value: QdrantSearchFilter) -> bool:
    if value.document_ids and payload.get("document_id") not in {
        str(identifier) for identifier in value.document_ids
    }:
        return False
    return all(_fake_matches_condition(payload, condition) for condition in value.conditions)


def _fake_score(distance: str, query: tuple[float, ...], stored: tuple[float, ...]) -> float:
    if len(query) != len(stored):
        raise QdrantConfigurationError("Qdrant point dimension does not match collection")
    if distance == "dot":
        return sum(left * right for left, right in zip(query, stored, strict=True))
    if distance == "cosine":
        dot = sum(left * right for left, right in zip(query, stored, strict=True))
        query_norm = math.sqrt(sum(value * value for value in query))
        stored_norm = math.sqrt(sum(value * value for value in stored))
        if query_norm == 0 or stored_norm == 0:
            return 0.0
        return max(-1.0, min(1.0, dot / (query_norm * stored_norm)))
    if distance == "euclid":
        return -math.sqrt(
            sum((left - right) ** 2 for left, right in zip(query, stored, strict=True))
        )
    return -sum(abs(left - right) for left, right in zip(query, stored, strict=True))


class FakeQdrantClient:
    """Concurrency-safe deterministic Qdrant fake used by unit and API tests."""

    def __init__(self, *, clock: Callable[[], datetime] = _utc_now) -> None:
        self._clock = clock
        self._collections: dict[str, _FakeCollection] = {}
        self._lock = asyncio.Lock()
        self.create_calls = 0
        self.payload_index_create_calls = 0

    @property
    def collection_names(self) -> tuple[str, ...]:
        return tuple(sorted(self._collections))

    async def collection_exists(self, collection: str) -> bool:
        if _COLLECTION_PATTERN.fullmatch(collection) is None:
            raise ValueError("collection identity is invalid")
        async with self._lock:
            return collection in self._collections

    async def seed_collection(
        self,
        spec: CollectionSpec,
        *,
        created_at: datetime,
        point_count: int = 0,
        managed: bool = True,
    ) -> None:
        if type(point_count) is not int or point_count < 0 or type(managed) is not bool:
            raise ValueError("fake collection state is invalid")
        async with self._lock:
            self._collections[spec.name] = _FakeCollection(
                spec.dimension,
                spec.distance,
                spec.datatype,
                spec.vector_on_disk,
                spec.vector_hnsw_config,
                spec.vector_quantization,
                spec.multivector,
                spec.sparse_vectors,
                spec.on_disk_payload,
                spec.collection_quantization,
                spec.hnsw_config,
                {index.path: index for index in spec.payload_indexes},
                created_at,
                point_count,
                managed,
                spec.name,
                _MANAGED_SCHEMA_VERSION,
                {},
            )

    @staticmethod
    def _require_managed(collection: str, current: _FakeCollection) -> None:
        if (
            not current.managed
            or current.identity_name != collection
            or current.schema_version != _MANAGED_SCHEMA_VERSION
            or current.created_at.tzinfo is None
            or current.created_at.utcoffset() != UTC.utcoffset(current.created_at)
        ):
            raise QdrantConfigurationError("Qdrant collection is not managed")

    async def ensure_collection(self, spec: CollectionSpec) -> None:
        if spec.payload_indexes:
            raise ValueError("ensure_collection requires a vector-only specification")
        async with self._lock:
            current = self._collections.get(spec.name)
            if current is None:
                now = self._clock()
                self._collections[spec.name] = _FakeCollection(
                    spec.dimension,
                    spec.distance,
                    spec.datatype,
                    spec.vector_on_disk,
                    spec.vector_hnsw_config,
                    spec.vector_quantization,
                    spec.multivector,
                    spec.sparse_vectors,
                    spec.on_disk_payload,
                    spec.collection_quantization,
                    spec.hnsw_config,
                    {},
                    now,
                    0,
                    True,
                    spec.name,
                    _MANAGED_SCHEMA_VERSION,
                    {},
                )
                self.create_calls += 1
                return
            self._require_managed(spec.name, current)
            current_spec = CollectionSpec(
                spec.name,
                current.dimension,
                current.distance,
                (),
                datatype=current.datatype,
                vector_on_disk=current.vector_on_disk,
                vector_hnsw_config=current.vector_hnsw_config,
                vector_quantization=current.vector_quantization,
                multivector=current.multivector,
                sparse_vectors=current.sparse_vectors,
                on_disk_payload=current.on_disk_payload,
                collection_quantization=current.collection_quantization,
                hnsw_config=current.hnsw_config,
            )
            if current_spec != spec:
                raise QdrantConfigurationError(
                    "Qdrant collection does not match requested configuration"
                )

    async def ensure_payload_indexes(
        self,
        collection: str,
        indexes: Sequence[PayloadIndex],
    ) -> None:
        expected = {index.path: index for index in indexes}
        if len(expected) != len(indexes):
            raise ValueError("payload index specification is invalid")
        async with self._lock:
            current = self._collections.get(collection)
            if current is None:
                raise QdrantConfigurationError("Qdrant collection does not exist")
            self._require_managed(collection, current)
            for path, index in current.payload_indexes.items():
                if expected.get(path) != index:
                    raise QdrantConfigurationError(
                        "Qdrant collection does not match requested configuration"
                    )
            for path, index in expected.items():
                if path not in current.payload_indexes:
                    current.payload_indexes[path] = index
                    self.payload_index_create_calls += 1
            if current.payload_indexes != expected:
                raise QdrantConfigurationError(
                    "Qdrant collection does not match requested configuration"
                )

    async def verify_collection(self, spec: CollectionSpec) -> None:
        actual = await self.describe_collection(spec.name)
        if actual != spec:
            raise QdrantConfigurationError(
                "Qdrant collection does not match requested configuration"
            )

    async def upsert_points(self, collection: str, points: Sequence[QdrantPoint]) -> None:
        batch = tuple(points)
        if (
            not batch
            or len(batch) > 10_000
            or any(type(point) is not QdrantPoint for point in batch)
            or len({point.id for point in batch}) != len(batch)
        ):
            raise ValueError("Qdrant point batch is invalid")
        async with self._lock:
            current = self._collections.get(collection)
            if current is None:
                raise QdrantConfigurationError("Qdrant collection does not exist")
            self._require_managed(collection, current)
            if any(len(point.vector) != current.dimension for point in batch):
                raise QdrantConfigurationError("Qdrant point dimension does not match collection")
            for point in batch:
                if point.id not in current.points:
                    current.point_count += 1
                current.points[point.id] = point

    async def stored_points(self, collection: str) -> tuple[QdrantPoint, ...]:
        async with self._lock:
            current = self._collections.get(collection)
            if current is None:
                raise QdrantConfigurationError("Qdrant collection does not exist")
            self._require_managed(collection, current)
            return tuple(current.points[point_id] for point_id in sorted(current.points, key=str))

    async def count_points(self, collection: str) -> int:
        async with self._lock:
            current = self._collections.get(collection)
            if current is None:
                raise QdrantConfigurationError("Qdrant collection does not exist")
            self._require_managed(collection, current)
            return current.point_count

    async def count_version_points(self, collection: str, version_id: UUID) -> int:
        if _COLLECTION_PATTERN.fullmatch(collection) is None or type(version_id) is not UUID:
            raise ValueError("Qdrant version identity is invalid")
        async with self._lock:
            current = self._collections.get(collection)
            if current is None:
                raise QdrantConfigurationError("Qdrant collection does not exist")
            self._require_managed(collection, current)
            count = sum(
                point.payload["version_id"] == str(version_id) for point in current.points.values()
            )
            if count > _MAX_VERSION_POINTS:
                raise QdrantConfigurationError("Qdrant point count is invalid")
            return count

    async def retrieve_version_points(
        self,
        collection: str,
        version_id: UUID,
        point_ids: Sequence[UUID],
    ) -> tuple[QdrantRetrievedPoint, ...]:
        requested = tuple(point_ids)
        if (
            _COLLECTION_PATTERN.fullmatch(collection) is None
            or type(version_id) is not UUID
            or not 1 <= len(requested) <= _MAX_POINT_RETRIEVE_BATCH
            or any(type(point) is not UUID for point in requested)
            or len(set(requested)) != len(requested)
        ):
            raise ValueError("Qdrant point retrieve request is invalid")
        async with self._lock:
            current = self._collections.get(collection)
            if current is None:
                raise QdrantConfigurationError("Qdrant collection does not exist")
            self._require_managed(collection, current)
            if any(point not in current.points for point in requested):
                raise QdrantConfigurationError("Qdrant point retrieve response is invalid")
            result: list[QdrantRetrievedPoint] = []
            for point_id_value in requested:
                point = current.points[point_id_value]
                if point.payload["version_id"] != str(version_id):
                    raise QdrantConfigurationError("Qdrant point retrieve response is invalid")
                result.append(
                    QdrantRetrievedPoint(
                        point.id,
                        len(point.vector),
                        canonical_sha256(dict(point.payload)),
                    )
                )
            return tuple(result)

    async def search_points(
        self,
        collection: str,
        vector: Sequence[float],
        *,
        limit: int,
        query_filter: QdrantSearchFilter,
    ) -> tuple[QdrantSearchPoint, ...]:
        query_vector = tuple(vector)
        if (
            _COLLECTION_PATTERN.fullmatch(collection) is None
            or not query_vector
            or any(type(value) is not float or not math.isfinite(value) for value in query_vector)
            or type(limit) is not int
            or not 1 <= limit <= 200
            or type(query_filter) is not QdrantSearchFilter
        ):
            raise ValueError("Qdrant search request is invalid")
        async with self._lock:
            current = self._collections.get(collection)
            if current is None:
                raise QdrantConfigurationError("Qdrant collection does not exist")
            self._require_managed(collection, current)
            if len(query_vector) != current.dimension:
                raise QdrantConfigurationError("Qdrant point dimension does not match collection")
            ranked = [
                QdrantSearchPoint(
                    point.id,
                    _fake_score(current.distance, query_vector, point.vector),
                    point.payload,
                )
                for point in current.points.values()
                if _fake_matches_filter(point.payload, query_filter)
            ]
            ranked.sort(key=lambda point: (-point.score, str(point.id)))
            return tuple(ranked[:limit])

    async def describe_collection(self, collection: str) -> CollectionSpec:
        async with self._lock:
            current = self._collections.get(collection)
            if current is None:
                raise QdrantConfigurationError("Qdrant collection does not exist")
            self._require_managed(collection, current)
            return CollectionSpec(
                collection,
                current.dimension,
                current.distance,
                tuple(current.payload_indexes.values()),
                datatype=current.datatype,
                vector_on_disk=current.vector_on_disk,
                vector_hnsw_config=current.vector_hnsw_config,
                vector_quantization=current.vector_quantization,
                multivector=current.multivector,
                sparse_vectors=current.sparse_vectors,
                on_disk_payload=current.on_disk_payload,
                collection_quantization=current.collection_quantization,
                hnsw_config=current.hnsw_config,
            )

    async def list_managed_collections(
        self,
        *,
        limit: int,
        cursor: str | None,
    ) -> ManagedCollectionPage:
        _validate_page(limit, cursor)
        async with self._lock:
            candidates = [
                (name, collection)
                for name, collection in sorted(self._collections.items())
                if cursor is None or name > cursor
            ]
            page = candidates[:limit]
            managed: list[ManagedCollection] = []
            for name, collection in page:
                try:
                    self._require_managed(name, collection)
                except QdrantConfigurationError:
                    continue
                managed.append(ManagedCollection(name, collection.created_at))
            next_cursor = page[-1][0] if len(candidates) > limit and page else None
            return ManagedCollectionPage(tuple(managed), next_cursor)

    async def delete_collection(self, collection: str) -> None:
        async with self._lock:
            current = self._collections.get(collection)
            if current is None:
                return
            self._require_managed(collection, current)
            self._collections.pop(collection)

    async def aclose(self) -> None:
        return None


def qdrant_client_from_url(
    url: str,
    *,
    timeout_seconds: float,
    clock: Callable[[], datetime] = _utc_now,
) -> AsyncQdrantCollectionClient:
    if not 0 < timeout_seconds <= 600:
        raise ValueError("Qdrant request timeout is invalid")
    return AsyncQdrantCollectionClient(
        AsyncQdrantClient(url=url, timeout=timeout_seconds),  # type: ignore[arg-type]
        clock=clock,
    )


__all__ = [
    "AsyncQdrantCollectionClient",
    "CollectionHnswSpec",
    "CollectionSpec",
    "FakeQdrantClient",
    "ManagedCollection",
    "ManagedCollectionPage",
    "PayloadIndex",
    "QdrantRetrievedPoint",
    "QdrantPoint",
    "QdrantClient",
    "QdrantConfigurationError",
    "QdrantFilterCondition",
    "QdrantSearchFilter",
    "QdrantSearchPoint",
    "QdrantTransientError",
    "managed_collection_identity",
    "qdrant_client_from_url",
    "qdrant_distance",
]
