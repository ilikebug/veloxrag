import base64
import binascii
import hashlib
import json
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Annotated, Literal
from uuid import RFC_4122, UUID, uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from rag_service.api.errors import BusinessError
from rag_service.api.etags import knowledge_base_etag
from rag_service.api.validation import JSONValue, validate_bounded_json

KnowledgeBaseName = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=120),
]
KnowledgeBaseDescription = Annotated[str, StringConstraints(max_length=2000)]
KnowledgeBaseStatus = Literal["active", "reindexing", "disabled", "deleting"]
DocumentStatus = Literal["processing", "active", "failed", "deleting", "deleted"]
DocumentVersionStatus = Literal[
    "uploaded",
    "parsing",
    "chunking",
    "embedding",
    "indexing",
    "ready",
    "failed",
    "conflicted",
    "cancelled",
    "ocr_required",
    "superseded",
]
FilterFieldType = Literal["keyword", "integer", "float", "boolean", "datetime"]
FilterOperator = Literal["eq", "in", "gte", "lte"]

ALLOWED_OPERATORS: dict[FilterFieldType, frozenset[FilterOperator]] = {
    "keyword": frozenset({"eq", "in"}),
    "integer": frozenset({"eq", "in", "gte", "lte"}),
    "float": frozenset({"eq", "in", "gte", "lte"}),
    "boolean": frozenset({"eq", "in"}),
    "datetime": frozenset({"eq", "in", "gte", "lte"}),
}

_FILTER_SEGMENT_PATTERN = r"^[A-Za-z][A-Za-z0-9_]{0,63}$"
_FILTER_SOURCE_PATH_PATTERN = r"^[A-Za-z][A-Za-z0-9_]{0,63}(\.[A-Za-z][A-Za-z0-9_]{0,63}){0,3}$"
_FILTER_FIELD_ID_PATTERN = r"^fld_[A-Za-z0-9_-]{22}$"
_FILTER_PAYLOAD_PATH_PATTERN = r"^metadata\.f_[0-9a-f]{32}$"

_CANONICAL_JSON_ENCODER = json.JSONEncoder(
    ensure_ascii=False,
    separators=(",", ":"),
    sort_keys=True,
    allow_nan=False,
)


class _Schema(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        hide_input_in_errors=True,
    )


def _validated_request_metadata(value: object) -> object:
    if value is None or not isinstance(value, dict):
        return value
    validated = validate_bounded_json(value)
    if not isinstance(validated, dict):
        raise ValueError("metadata must be an object")
    return validated


def _validated_stored_metadata(value: object) -> object:
    if not isinstance(value, dict):
        return value
    try:
        validated = validate_bounded_json(value)
    except BusinessError:
        raise ValueError("stored metadata is invalid") from None
    if not isinstance(validated, dict):
        raise ValueError("stored metadata must be an object")
    return validated


def _require_aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
        raise ValueError("timestamp must use UTC")
    return value


class KnowledgeBaseCreate(_Schema):
    name: KnowledgeBaseName
    description: KnowledgeBaseDescription | None = None
    metadata: dict[str, JSONValue] = Field(default_factory=dict)

    _validate_metadata = field_validator("metadata", mode="before")(_validated_request_metadata)


class KnowledgeBasePatch(_Schema):
    name: KnowledgeBaseName | None = None
    description: KnowledgeBaseDescription | None = None
    metadata: dict[str, JSONValue] | None = None
    status: Literal["active", "disabled"] | None = None
    # Explicit null clears it, which is how reranking is turned back off.
    rerank_profile_id: UUID | None = None

    _validate_metadata = field_validator("metadata", mode="before")(_validated_request_metadata)

    @field_validator("rerank_profile_id", mode="before")
    @classmethod
    def validate_rerank_profile_id(cls, value: object) -> object:
        # This model is strict, so a JSON string is not coerced to UUID on its
        # own. Parsing here also rejects non-canonical spellings such as braced
        # or uppercase forms, which would otherwise reach storage as a different
        # string than they came in as.
        if value is None or type(value) is UUID:
            return value
        if type(value) is not str:
            raise ValueError("rerank profile id is invalid")
        try:
            parsed = UUID(value)
        except ValueError:
            raise ValueError("rerank profile id is invalid") from None
        if value != str(parsed):
            raise ValueError("rerank profile id is invalid")
        return parsed

    @model_validator(mode="after")
    def validate_patch(self) -> "KnowledgeBasePatch":
        if not self.model_fields_set:
            raise ValueError("at least one field is required")
        # Both exempt fields are genuinely clearable: a description can be
        # removed, and a null rerank profile is the only way to turn reranking
        # back off once a knowledge base has one.
        for field_name in self.model_fields_set - {"description", "rerank_profile_id"}:
            if getattr(self, field_name) is None:
                raise ValueError(f"{field_name} must not be null")
        return self


class SafeKnowledgeBase(_Schema):
    id: UUID
    name: str = Field(min_length=1, max_length=120)
    description: str | None = Field(default=None, max_length=2000)
    status: KnowledgeBaseStatus
    metadata: dict[str, JSONValue]
    resource_revision: int = Field(ge=1)
    mutation_revision: int = Field(ge=0)
    filter_schema_revision: int = Field(ge=0)
    active_index_generation_id: UUID | None
    pending_index_generation_id: UUID | None
    rerank_profile_id: UUID | None = None
    created_at: datetime
    updated_at: datetime
    etag: str

    _validate_metadata = field_validator("metadata", mode="before")(_validated_stored_metadata)
    _validate_created_at = field_validator("created_at")(_require_aware_utc)
    _validate_updated_at = field_validator("updated_at")(_require_aware_utc)

    @model_validator(mode="after")
    def validate_etag(self) -> "SafeKnowledgeBase":
        if self.etag != knowledge_base_etag(self.id, self.resource_revision):
            raise ValueError("etag must match the resource revision")
        return self


class KnowledgeBaseCreateResult(_Schema):
    knowledge_base: SafeKnowledgeBase
    created: bool


class KnowledgeBasePage(_Schema):
    items: tuple[SafeKnowledgeBase, ...]
    next_cursor: str | None = None


class SafeDocument(_Schema):
    id: UUID
    knowledge_base_id: UUID
    display_name: str = Field(min_length=1, max_length=255)
    mime_type: str | None = Field(default=None, min_length=1, max_length=255)
    checksum_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    current_version_id: UUID | None
    pending_version_id: UUID | None
    status: DocumentStatus
    tags: tuple[str, ...] = Field(max_length=64)
    metadata: dict[str, JSONValue]
    created_at: datetime
    updated_at: datetime

    _validate_metadata = field_validator("metadata", mode="before")(_validated_stored_metadata)
    _validate_created_at = field_validator("created_at")(_require_aware_utc)
    _validate_updated_at = field_validator("updated_at")(_require_aware_utc)


class DocumentPage(_Schema):
    items: tuple[SafeDocument, ...]
    next_cursor: str | None = None


class SafeVersion(_Schema):
    id: UUID
    document_id: UUID
    version_number: int = Field(ge=1)
    source_checksum_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    parsed_object_checksum_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    declared_mime_type: str | None = Field(default=None, min_length=1, max_length=255)
    detected_mime_type: str | None = Field(default=None, min_length=1, max_length=255)
    source_extension: str | None = Field(default=None, min_length=1, max_length=32)
    base_version_id: UUID | None
    parser_name: str | None = Field(default=None, min_length=1, max_length=120)
    parser_version: str | None = Field(default=None, min_length=1, max_length=64)
    parser_config: dict[str, JSONValue]
    chunker_name: str | None = Field(default=None, min_length=1, max_length=120)
    chunker_version: str | None = Field(default=None, min_length=1, max_length=64)
    chunker_config: dict[str, JSONValue]
    chunk_count: int | None = Field(default=None, ge=0)
    status: DocumentVersionStatus
    activated_at: datetime | None
    created_at: datetime

    _validate_parser_config = field_validator("parser_config", mode="before")(
        _validated_stored_metadata
    )
    _validate_chunker_config = field_validator("chunker_config", mode="before")(
        _validated_stored_metadata
    )
    _validate_activated_at = field_validator("activated_at")(
        lambda value: None if value is None else _require_aware_utc(value)
    )
    _validate_created_at = field_validator("created_at")(_require_aware_utc)


class DocumentVersionPage(_Schema):
    items: tuple[SafeVersion, ...]
    next_cursor: str | None = None


class FilterSchemaField(_Schema):
    name: Annotated[str, StringConstraints(pattern=_FILTER_SEGMENT_PATTERN)]
    source_path: Annotated[str, StringConstraints(pattern=_FILTER_SOURCE_PATH_PATTERN)]
    type: FilterFieldType
    operators: tuple[FilterOperator, ...] = Field(min_length=1, max_length=4)

    @field_validator("operators", mode="before")
    @classmethod
    def canonicalize_operators(cls, value: object) -> object:
        if not isinstance(value, (list, tuple)):
            return value
        if not 1 <= len(value) <= 4:
            raise ValueError("operators must contain between one and four values")
        if any(type(operator) is not str for operator in value):
            return value
        if len(set(value)) != len(value):
            raise ValueError("operators must be unique")
        return tuple(sorted(value))

    @model_validator(mode="after")
    def validate_operator_matrix(self) -> "FilterSchemaField":
        if not set(self.operators) <= ALLOWED_OPERATORS[self.type]:
            raise ValueError("operators are not supported for this field type")
        return self


class FilterSchemaReplacement(_Schema):
    fields: tuple[FilterSchemaField, ...] = Field(max_length=64)

    @field_validator("fields", mode="before")
    @classmethod
    def accept_json_array(cls, value: object) -> object:
        if isinstance(value, (list, tuple)):
            if len(value) > 64:
                raise ValueError("filter schema must contain at most 64 fields")
            return tuple(value)
        return value

    @model_validator(mode="after")
    def validate_unique_fields(self) -> "FilterSchemaReplacement":
        names = [field.name for field in self.fields]
        source_paths = [field.source_path for field in self.fields]
        if len(names) != len(set(names)):
            raise ValueError("filter field names must be unique")
        if len(source_paths) != len(set(source_paths)):
            raise ValueError("filter field source paths must be unique")
        return self


class SafeFilterSchema(_Schema):
    fields: tuple[FilterSchemaField, ...]
    resource_revision: int = Field(ge=1)
    mutation_revision: int = Field(ge=0)
    filter_schema_revision: int = Field(ge=0)
    etag: str


class _StoredFilterSchemaField(FilterSchemaField):
    field_id: Annotated[str, StringConstraints(pattern=_FILTER_FIELD_ID_PATTERN)]
    payload_path: Annotated[str, StringConstraints(pattern=_FILTER_PAYLOAD_PATH_PATTERN)]

    @model_validator(mode="after")
    def validate_identifier_pair(self) -> "_StoredFilterSchemaField":
        encoded = self.field_id.removeprefix("fld_")
        try:
            identifier_bytes = base64.b64decode(
                f"{encoded}==",
                altchars=b"-_",
                validate=True,
            )
        except (binascii.Error, ValueError):
            raise ValueError("stored filter field identifier is invalid") from None
        if len(identifier_bytes) != 16:
            raise ValueError("stored filter field identifier is invalid")
        canonical = base64.urlsafe_b64encode(identifier_bytes).rstrip(b"=").decode("ascii")
        if encoded != canonical:
            raise ValueError("stored filter field identifier is not canonical")
        identifier = UUID(bytes=identifier_bytes)
        if identifier.variant != RFC_4122 or identifier.version != 4:
            raise ValueError("stored filter field identifier must be a UUIDv4")
        if self.payload_path != f"metadata.f_{identifier.hex}":
            raise ValueError("stored filter field identifier pair does not match")
        return self


class _StoredFilterSchema(_Schema):
    fields: tuple[_StoredFilterSchemaField, ...] = Field(max_length=64)

    @field_validator("fields", mode="before")
    @classmethod
    def accept_json_array(cls, value: object) -> object:
        if isinstance(value, (list, tuple)):
            if len(value) > 64:
                raise ValueError("stored filter schema must contain at most 64 fields")
            return tuple(value)
        return value

    @model_validator(mode="after")
    def validate_unique_fields(self) -> "_StoredFilterSchema":
        names = [field.name for field in self.fields]
        source_paths = [field.source_path for field in self.fields]
        field_ids = [field.field_id for field in self.fields]
        payload_paths = [field.payload_path for field in self.fields]
        for values in (names, source_paths, field_ids, payload_paths):
            if len(values) != len(set(values)):
                raise ValueError("stored filter schema fields must be unique")
        return self


def _new_stored_filter_field(
    field: FilterSchemaField,
    id_factory: Callable[[], UUID],
) -> _StoredFilterSchemaField:
    identifier = id_factory()
    if not isinstance(identifier, UUID):
        raise TypeError("id_factory must return UUID values")
    encoded = base64.urlsafe_b64encode(identifier.bytes).rstrip(b"=").decode("ascii")
    return _StoredFilterSchemaField(
        **field.model_dump(),
        field_id=f"fld_{encoded}",
        payload_path=f"metadata.f_{identifier.hex}",
    )


def canonicalize_filter_schema(
    command: FilterSchemaReplacement,
    current_schema: object,
    *,
    id_factory: Callable[[], UUID] = uuid4,
) -> dict[str, object]:
    if type(command) is not FilterSchemaReplacement:
        raise TypeError("command must be a FilterSchemaReplacement")
    current = _StoredFilterSchema.model_validate(current_schema)
    retained_by_name = {field.name: field for field in current.fields}
    canonical_fields: list[_StoredFilterSchemaField] = []
    for field in command.fields:
        retained = retained_by_name.get(field.name)
        if retained is not None:
            if retained.type != field.type or retained.source_path != field.source_path:
                raise BusinessError(
                    422,
                    "VALIDATION_ERROR",
                    "Invalid knowledge base request",
                )
            canonical_fields.append(retained.model_copy(update={"operators": field.operators}))
            continue
        canonical_fields.append(_new_stored_filter_field(field, id_factory))
    canonical = _StoredFilterSchema(fields=tuple(canonical_fields))
    return canonical.model_dump(mode="json")


def _canonical_create_payload(command: KnowledgeBaseCreate) -> bytes:
    if type(command) is not KnowledgeBaseCreate:
        raise TypeError("command must be a KnowledgeBaseCreate")
    document = command.model_dump(mode="json")
    return _CANONICAL_JSON_ENCODER.encode(document).encode("utf-8")


def knowledge_base_create_fingerprint(command: KnowledgeBaseCreate) -> bytes:
    return hashlib.sha256(_canonical_create_payload(command)).digest()


class DocumentContent(_Schema):
    """A slice of a document's normalized text, addressed by codepoint offsets.

    The offsets are the same ones a search result carries, so a consumer can take
    a hit and widen it without a second addressing scheme. `total_codepoints`
    lets the caller tell a clamped range from an exhausted one.
    """

    document_id: UUID
    version_id: UUID
    start_offset: int = Field(ge=0)
    end_offset: int = Field(ge=0)
    total_codepoints: int = Field(ge=0)
    text: str
