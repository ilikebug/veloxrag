import json
import math
import re
from copy import deepcopy
from datetime import UTC, datetime
from decimal import Decimal
from typing import Annotated, Literal, cast
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    StringConstraints,
    WithJsonSchema,
    field_validator,
    model_validator,
)
from pydantic.json_schema import SkipJsonSchema

CredentialName = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=120),
]
ProviderConfigName = CredentialName
ModelProfileName = CredentialName
WritableProviderType = Literal["openai_compatible", "openrouter"]
ProviderTimeoutSeconds = Annotated[
    Decimal,
    Field(gt=0, le=Decimal("600"), decimal_places=3),
]
ProviderMaxConcurrency = Annotated[int, Field(ge=1, le=10000)]
ProviderRequestsPerMinute = Annotated[int, Field(ge=1, le=1000000)]
ModelProfileDimension = Annotated[int, Field(ge=1, le=10000000)]
ModelProfileMaxInputTokens = Annotated[int, Field(ge=1, le=10000000)]
ModelProfileBatchSize = Annotated[int, Field(ge=1, le=10000)]
ModelProfileTimeoutSeconds = ProviderTimeoutSeconds
_MAX_SECRET_BYTES = 8192
_OPENROUTER_ROUTING_KEYS = frozenset(
    {
        "order",
        "allow_fallbacks",
        "require_parameters",
        "data_collection",
        "zdr",
        "enforce_distillable_text",
        "only",
        "ignore",
        "quantizations",
        "sort",
        "preferred_min_throughput",
        "preferred_max_latency",
        "max_price",
    }
)
_OPENROUTER_BOOLEAN_KEYS = frozenset(
    {"allow_fallbacks", "require_parameters", "zdr", "enforce_distillable_text"}
)
_OPENROUTER_PROVIDER_ARRAY_KEYS = frozenset({"order", "only", "ignore"})
_OPENROUTER_QUANTIZATIONS = frozenset(
    {"int4", "int8", "fp4", "fp6", "fp8", "fp16", "bf16", "fp32", "unknown"}
)
_OPENROUTER_SORT_VALUES = frozenset({"price", "throughput", "latency", "exacto"})
_OPENROUTER_PERCENTILES = frozenset({"p50", "p75", "p90", "p99"})
_OPENROUTER_PRICE_KEYS = frozenset({"audio", "prompt", "completion", "request", "image"})
_OPENROUTER_PRICE_PATTERN = re.compile(r"^(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?$")

_DEFAULT_HEADERS_JSON_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "maxProperties": 3,
    "properties": {
        "HTTP-Referer": {
            "type": "string",
            "minLength": 1,
            "maxLength": 2048,
            "format": "uri",
        },
        "X-OpenRouter-Title": {
            "type": "string",
            "minLength": 1,
            "maxLength": 120,
        },
        "X-Title": {
            "type": "string",
            "minLength": 1,
            "maxLength": 120,
        },
    },
}
_ROUTING_PROVIDER_ARRAY_JSON_SCHEMA: dict[str, object] = {
    "type": ["array", "null"],
    "maxItems": 100,
    "items": {"type": "string", "minLength": 1, "maxLength": 255},
}
_ROUTING_PERCENTILE_JSON_SCHEMA: dict[str, object] = {
    "additionalProperties": False,
    "properties": {
        percentile: {"type": ["number", "null"], "minimum": 0}
        for percentile in sorted(_OPENROUTER_PERCENTILES)
    },
    "anyOf": [
        {"type": "number", "minimum": 0},
        {"type": "object"},
        {"type": "null"},
    ],
}
_ROUTING_SORT_JSON_SCHEMA: dict[str, object] = {
    "additionalProperties": False,
    "properties": {
        "by": {"enum": [*sorted(_OPENROUTER_SORT_VALUES), None]},
        "partition": {"enum": ["model", "none", None]},
    },
    "anyOf": [
        {"type": "string", "enum": sorted(_OPENROUTER_SORT_VALUES)},
        {"type": "object"},
        {"type": "null"},
    ],
}
_ROUTING_MAX_PRICE_JSON_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "maxProperties": len(_OPENROUTER_PRICE_KEYS),
    "properties": {
        key: {
            "type": "string",
            "minLength": 1,
            "maxLength": 64,
            "pattern": _OPENROUTER_PRICE_PATTERN.pattern,
        }
        for key in sorted(_OPENROUTER_PRICE_KEYS)
    },
}
_ROUTING_OPTIONS_JSON_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "maxProperties": len(_OPENROUTER_ROUTING_KEYS),
    "properties": {
        "order": _ROUTING_PROVIDER_ARRAY_JSON_SCHEMA,
        "allow_fallbacks": {"type": ["boolean", "null"]},
        "require_parameters": {"type": ["boolean", "null"]},
        "data_collection": {"enum": ["allow", "deny", None]},
        "zdr": {"type": ["boolean", "null"]},
        "enforce_distillable_text": {"type": ["boolean", "null"]},
        "only": _ROUTING_PROVIDER_ARRAY_JSON_SCHEMA,
        "ignore": _ROUTING_PROVIDER_ARRAY_JSON_SCHEMA,
        "quantizations": {
            "type": ["array", "null"],
            "maxItems": 32,
            "items": {"type": "string", "enum": sorted(_OPENROUTER_QUANTIZATIONS)},
        },
        "sort": _ROUTING_SORT_JSON_SCHEMA,
        "preferred_min_throughput": _ROUTING_PERCENTILE_JSON_SCHEMA,
        "preferred_max_latency": _ROUTING_PERCENTILE_JSON_SCHEMA,
        "max_price": _ROUTING_MAX_PRICE_JSON_SCHEMA,
    },
}
ProviderDefaultHeaders = Annotated[
    dict[str, str],
    WithJsonSchema(_DEFAULT_HEADERS_JSON_SCHEMA),
]
ProviderRoutingOptions = Annotated[
    dict[str, object],
    WithJsonSchema(_ROUTING_OPTIONS_JSON_SCHEMA),
]
WritableVectorConfig = Annotated[
    dict[str, object],
    WithJsonSchema(
        {
            "type": "object",
            "additionalProperties": False,
            "maxProperties": 0,
        }
    ),
]


class _Schema(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
    )


def _validate_secret(value: SecretStr) -> SecretStr:
    secret = value.get_secret_value()
    encoded = b""
    try:
        encoded = secret.encode("utf-8")
        if not secret.strip() or len(encoded) > _MAX_SECRET_BYTES:
            raise ValueError("secret must contain between 1 and 8192 UTF-8 bytes")
        return value
    except UnicodeEncodeError:
        raise ValueError("secret must be valid UTF-8") from None
    finally:
        secret = "<redacted>"
        encoded = b""


def _require_aware_utc(value: datetime | None) -> datetime | None:
    if value is not None and (value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value)):
        raise ValueError("timestamp must use UTC")
    return value


def _validate_bounded_int(value: object, *, minimum: int, maximum: int) -> object:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError("integer is outside the supported range")
    return value


def _validate_timeout(value: object) -> object:
    if type(value) is bool:
        raise ValueError("timeout is invalid")
    return value


def _validate_provider_array(value: object, *, maximum: int) -> list[str] | None:
    if value is None:
        return None
    if type(value) is not list or len(value) > maximum:
        raise ValueError("routing array is invalid")
    for item in value:
        if (
            type(item) is not str
            or not 1 <= len(item) <= 255
            or any(ord(character) < 32 or ord(character) == 127 for character in item)
        ):
            raise ValueError("routing array is invalid")
    return cast(list[str], value)


def _validate_nonnegative_number(value: object) -> None:
    if type(value) not in {int, float} or type(value) is bool:
        raise ValueError("routing number is invalid")
    numeric_value = cast(int | float, value)
    if isinstance(numeric_value, float) and not math.isfinite(numeric_value):
        raise ValueError("routing number is invalid")
    if numeric_value < 0:
        raise ValueError("routing number is invalid")


def _validate_latency_or_throughput(value: object) -> None:
    if value is None:
        return
    if type(value) is dict:
        if not set(value).issubset(_OPENROUTER_PERCENTILES):
            raise ValueError("routing percentile object is invalid")
        for member in value.values():
            if member is not None:
                _validate_nonnegative_number(member)
        return
    _validate_nonnegative_number(value)


def _validate_routing_options(value: dict[str, object]) -> dict[str, object]:
    if type(value) is not dict or not set(value).issubset(_OPENROUTER_ROUTING_KEYS):
        raise ValueError("routing options are invalid")
    if len(value) > len(_OPENROUTER_ROUTING_KEYS):
        raise ValueError("routing options are invalid")
    for key in _OPENROUTER_PROVIDER_ARRAY_KEYS:
        if key in value:
            _validate_provider_array(value[key], maximum=100)
    for key in _OPENROUTER_BOOLEAN_KEYS:
        if key in value and value[key] is not None and type(value[key]) is not bool:
            raise ValueError("routing boolean is invalid")
    if value.get("data_collection") not in {None, "allow", "deny"}:
        raise ValueError("routing data collection value is invalid")
    if "quantizations" in value:
        quantizations = _validate_provider_array(value["quantizations"], maximum=32)
        if quantizations is not None and not set(quantizations).issubset(_OPENROUTER_QUANTIZATIONS):
            raise ValueError("routing quantizations are invalid")
    if "sort" in value:
        sort = value["sort"]
        if sort is not None and type(sort) is str and sort not in _OPENROUTER_SORT_VALUES:
            raise ValueError("routing sort is invalid")
        if sort is not None and type(sort) is dict:
            if not set(sort).issubset({"by", "partition"}):
                raise ValueError("routing sort is invalid")
            if sort.get("by") is not None and sort.get("by") not in _OPENROUTER_SORT_VALUES:
                raise ValueError("routing sort is invalid")
            if sort.get("partition") is not None and sort.get("partition") not in {
                "model",
                "none",
            }:
                raise ValueError("routing sort is invalid")
        elif sort is not None and type(sort) is not str:
            raise ValueError("routing sort is invalid")
    for key in ("preferred_min_throughput", "preferred_max_latency"):
        if key in value:
            _validate_latency_or_throughput(value[key])
    if "max_price" in value:
        prices = value["max_price"]
        if type(prices) is not dict or not set(prices).issubset(_OPENROUTER_PRICE_KEYS):
            raise ValueError("routing max price is invalid")
        for price in prices.values():
            if (
                type(price) is not str
                or not 1 <= len(price) <= 64
                or _OPENROUTER_PRICE_PATTERN.fullmatch(price) is None
            ):
                raise ValueError("routing max price is invalid")
    try:
        encoded = json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError):
        raise ValueError("routing options are invalid") from None
    if len(encoded) > 16000:
        raise ValueError("routing options are invalid")
    return deepcopy(value)


def _validate_provider_type_and_routing(
    provider_type: str | None,
    routing_options: dict[str, object] | None,
) -> None:
    if provider_type == "vendor_specific":
        raise ValueError("vendor_specific embedding providers are not supported")
    if provider_type == "openai_compatible" and routing_options:
        raise ValueError("routing options require openrouter")


def _validate_empty_vector_config(value: dict[str, object]) -> dict[str, object]:
    if type(value) is not dict or value:
        raise ValueError("vector config is not supported")
    return {}


class ProviderCredentialCreate(_Schema):
    name: CredentialName
    secret: SecretStr

    _validate_secret_value = field_validator("secret")(_validate_secret)


class ProviderCredentialPatch(_Schema):
    name: CredentialName | None = None
    secret: SecretStr | None = None

    @field_validator("secret")
    @classmethod
    def validate_secret_value(cls, value: SecretStr | None) -> SecretStr | None:
        return None if value is None else _validate_secret(value)

    @model_validator(mode="after")
    def validate_patch(self) -> "ProviderCredentialPatch":
        if not self.model_fields_set:
            raise ValueError("at least one field is required")
        for field_name in self.model_fields_set:
            if getattr(self, field_name) is None:
                raise ValueError(f"{field_name} must not be null")
        return self


class SafeProviderCredential(_Schema):
    id: UUID
    name: CredentialName
    credential_configured: Literal[True]
    key_version: Annotated[str, StringConstraints(min_length=1, max_length=64)]
    resource_revision: int = Field(ge=1)
    created_at: datetime
    updated_at: datetime
    rotated_at: datetime | None

    _validate_created_at = field_validator("created_at")(_require_aware_utc)
    _validate_updated_at = field_validator("updated_at")(_require_aware_utc)
    _validate_rotated_at = field_validator("rotated_at")(_require_aware_utc)


class ProviderCredentialCreateResult(_Schema):
    credential: SafeProviderCredential
    created: bool


class ProviderCredentialPage(_Schema):
    items: tuple[SafeProviderCredential, ...]
    next_cursor: str | None = None


class ProviderConfigCreate(_Schema):
    name: ProviderConfigName
    provider_type: WritableProviderType
    base_url: Annotated[str, StringConstraints(min_length=1, max_length=2048)]
    credential_id: UUID
    default_headers: ProviderDefaultHeaders = Field(default_factory=dict)
    routing_options: ProviderRoutingOptions = Field(default_factory=dict)
    timeout_seconds: ProviderTimeoutSeconds
    max_concurrency: ProviderMaxConcurrency
    requests_per_minute: ProviderRequestsPerMinute
    enabled: bool = True

    _validate_timeout_value = field_validator("timeout_seconds", mode="before")(_validate_timeout)

    @field_validator("max_concurrency", mode="before")
    @classmethod
    def validate_max_concurrency(cls, value: object) -> object:
        return _validate_bounded_int(value, minimum=1, maximum=10000)

    @field_validator("requests_per_minute", mode="before")
    @classmethod
    def validate_requests_per_minute(cls, value: object) -> object:
        return _validate_bounded_int(value, minimum=1, maximum=1000000)

    @field_validator("routing_options")
    @classmethod
    def validate_routing_options(cls, value: dict[str, object]) -> dict[str, object]:
        return _validate_routing_options(value)

    @model_validator(mode="after")
    def validate_protocol(self) -> "ProviderConfigCreate":
        _validate_provider_type_and_routing(self.provider_type, self.routing_options)
        return self


class ProviderConfigPatch(_Schema):
    name: ProviderConfigName | SkipJsonSchema[None] = None
    provider_type: WritableProviderType | SkipJsonSchema[None] = None
    base_url: (
        Annotated[str, StringConstraints(min_length=1, max_length=2048)] | SkipJsonSchema[None]
    ) = None
    credential_id: UUID | SkipJsonSchema[None] = None
    default_headers: ProviderDefaultHeaders | SkipJsonSchema[None] = None
    routing_options: ProviderRoutingOptions | SkipJsonSchema[None] = None
    timeout_seconds: ProviderTimeoutSeconds | SkipJsonSchema[None] = None
    max_concurrency: ProviderMaxConcurrency | SkipJsonSchema[None] = None
    requests_per_minute: ProviderRequestsPerMinute | SkipJsonSchema[None] = None
    enabled: bool | SkipJsonSchema[None] = None

    @field_validator("timeout_seconds", mode="before")
    @classmethod
    def validate_timeout_value(cls, value: object) -> object:
        return _validate_timeout(value)

    @field_validator("max_concurrency", mode="before")
    @classmethod
    def validate_max_concurrency(cls, value: object) -> object:
        if value is None:
            return value
        return _validate_bounded_int(value, minimum=1, maximum=10000)

    @field_validator("requests_per_minute", mode="before")
    @classmethod
    def validate_requests_per_minute(cls, value: object) -> object:
        if value is None:
            return value
        return _validate_bounded_int(value, minimum=1, maximum=1000000)

    @field_validator("routing_options")
    @classmethod
    def validate_routing_options(
        cls,
        value: dict[str, object] | None,
    ) -> dict[str, object] | None:
        return None if value is None else _validate_routing_options(value)

    @model_validator(mode="after")
    def validate_patch(self) -> "ProviderConfigPatch":
        if not self.model_fields_set:
            raise ValueError("at least one field is required")
        for field_name in self.model_fields_set:
            if getattr(self, field_name) is None:
                raise ValueError(f"{field_name} must not be null")
        _validate_provider_type_and_routing(self.provider_type, self.routing_options)
        return self


class SafeProviderConfig(_Schema):
    id: UUID
    name: ProviderConfigName
    provider_type: Literal["openai_compatible", "openrouter", "vendor_specific"]
    base_url: Annotated[str, StringConstraints(min_length=1, max_length=2048)]
    credential_id: UUID | None
    default_headers: ProviderDefaultHeaders
    routing_options: ProviderRoutingOptions
    timeout_seconds: Decimal = Field(gt=0, le=Decimal("600"), decimal_places=3)
    max_concurrency: int = Field(ge=1, le=10000)
    requests_per_minute: int = Field(ge=1, le=1000000)
    enabled: bool
    resource_revision: int = Field(ge=1)
    endpoint_policy_version: Annotated[str, StringConstraints(min_length=1, max_length=64)] | None
    endpoint_validated_at: datetime | None
    created_at: datetime
    updated_at: datetime

    _validate_endpoint_validated_at = field_validator("endpoint_validated_at")(_require_aware_utc)
    _validate_created_at = field_validator("created_at")(_require_aware_utc)
    _validate_updated_at = field_validator("updated_at")(_require_aware_utc)


class ProviderConfigCreateResult(_Schema):
    provider_config: SafeProviderConfig
    created: bool


class ProviderConfigPage(_Schema):
    items: tuple[SafeProviderConfig, ...]
    next_cursor: str | None = None


class EmbeddingProbeCreate(_Schema):
    model_name: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=255),
    ]

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, value: str) -> str:
        if any(ord(character) < 32 or 127 <= ord(character) <= 159 for character in value):
            raise ValueError("model name contains control characters")
        return value


class SafeEmbeddingProbe(_Schema):
    provider_config_id: UUID
    model_name: str
    dimension: int = Field(ge=1, le=10_000_000)

    @field_validator("dimension", mode="before")
    @classmethod
    def validate_dimension(cls, value: object) -> object:
        return _validate_bounded_int(value, minimum=1, maximum=10_000_000)


class ModelProfileCreate(_Schema):
    name: ModelProfileName
    # `chat` stays closed: answer generation is the consuming agent's job, not
    # this service's, so a profile for it would have nothing to drive.
    capability: Literal["embedding", "rerank"]
    provider_config_id: UUID
    model_name: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=255)
    ]
    dimension: ModelProfileDimension | None
    max_input_tokens: ModelProfileMaxInputTokens
    batch_size: ModelProfileBatchSize
    timeout_seconds: ModelProfileTimeoutSeconds
    vector_config: WritableVectorConfig
    enabled: bool

    _validate_timeout_value = field_validator("timeout_seconds", mode="before")(_validate_timeout)

    @field_validator("dimension", mode="before")
    @classmethod
    def validate_dimension(cls, value: object) -> object:
        if value is None:
            return None
        return _validate_bounded_int(value, minimum=1, maximum=10000000)

    @field_validator("max_input_tokens", mode="before")
    @classmethod
    def validate_max_input_tokens(cls, value: object) -> object:
        return _validate_bounded_int(value, minimum=1, maximum=10000000)

    @field_validator("batch_size", mode="before")
    @classmethod
    def validate_batch_size(cls, value: object) -> object:
        return _validate_bounded_int(value, minimum=1, maximum=10000)

    _validate_vector_config = field_validator("vector_config")(_validate_empty_vector_config)

    @model_validator(mode="after")
    def validate_capability_dimension(self) -> "ModelProfileCreate":
        # Same pairing the database enforces (ck_model_profiles_dimension): a
        # reranker scores query/passage pairs and has no vector width, and an
        # embedding profile without one cannot size a collection.
        if (self.capability == "embedding") != (self.dimension is not None):
            raise ValueError("model profile capability and dimension are inconsistent")
        return self


class ModelProfilePatch(_Schema):
    name: ModelProfileName | SkipJsonSchema[None] = None
    provider_config_id: UUID | SkipJsonSchema[None] = None
    model_name: (
        Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=255)]
        | SkipJsonSchema[None]
    ) = None
    dimension: ModelProfileDimension | SkipJsonSchema[None] = None
    max_input_tokens: ModelProfileMaxInputTokens | SkipJsonSchema[None] = None
    batch_size: ModelProfileBatchSize | SkipJsonSchema[None] = None
    timeout_seconds: ModelProfileTimeoutSeconds | SkipJsonSchema[None] = None
    vector_config: WritableVectorConfig | SkipJsonSchema[None] = None
    enabled: bool | SkipJsonSchema[None] = None

    @field_validator("timeout_seconds", mode="before")
    @classmethod
    def validate_timeout_value(cls, value: object) -> object:
        return _validate_timeout(value)

    @field_validator("dimension", mode="before")
    @classmethod
    def validate_dimension(cls, value: object) -> object:
        if value is None:
            return value
        return _validate_bounded_int(value, minimum=1, maximum=10000000)

    @field_validator("max_input_tokens", mode="before")
    @classmethod
    def validate_max_input_tokens(cls, value: object) -> object:
        if value is None:
            return value
        return _validate_bounded_int(value, minimum=1, maximum=10000000)

    @field_validator("batch_size", mode="before")
    @classmethod
    def validate_batch_size(cls, value: object) -> object:
        if value is None:
            return value
        return _validate_bounded_int(value, minimum=1, maximum=10000)

    @field_validator("vector_config")
    @classmethod
    def validate_vector_config(
        cls,
        value: dict[str, object] | None,
    ) -> dict[str, object] | None:
        return None if value is None else _validate_empty_vector_config(value)

    @model_validator(mode="after")
    def validate_patch(self) -> "ModelProfilePatch":
        if not self.model_fields_set:
            raise ValueError("at least one field is required")
        for field_name in self.model_fields_set:
            if getattr(self, field_name) is None:
                raise ValueError(f"{field_name} must not be null")
        return self


class InternalSafeModelProfile(_Schema):
    id: UUID
    name: ModelProfileName
    capability: Literal["embedding", "rerank", "chat"]
    provider_config_id: UUID
    model_name: Annotated[str, StringConstraints(min_length=1, max_length=255)]
    dimension: int | None = Field(default=None, ge=1, le=10000000)
    max_input_tokens: int = Field(ge=1, le=10000000)
    batch_size: int = Field(ge=1, le=10000)
    timeout_seconds: Decimal = Field(gt=0, le=Decimal("600"), decimal_places=3)
    vector_config: dict[str, object]
    enabled: bool
    resource_revision: int = Field(ge=1)
    created_at: datetime
    updated_at: datetime

    _validate_created_at = field_validator("created_at")(_require_aware_utc)
    _validate_updated_at = field_validator("updated_at")(_require_aware_utc)

    @model_validator(mode="after")
    def validate_capability_dimension(self) -> "InternalSafeModelProfile":
        if (self.capability == "embedding") != (self.dimension is not None):
            raise ValueError("model profile capability and dimension are inconsistent")
        return self


class SafeModelProfile(_Schema):
    id: UUID
    name: ModelProfileName
    capability: Literal["embedding", "rerank"]
    provider_config_id: UUID
    model_name: Annotated[str, StringConstraints(min_length=1, max_length=255)]
    dimension: int | None = Field(default=None, ge=1, le=10000000)
    max_input_tokens: int = Field(ge=1, le=10000000)
    batch_size: int = Field(ge=1, le=10000)
    timeout_seconds: Decimal = Field(gt=0, le=Decimal("600"), decimal_places=3)
    vector_config: dict[str, object]
    enabled: bool
    resource_revision: int = Field(ge=1)
    created_at: datetime
    updated_at: datetime

    _validate_created_at = field_validator("created_at")(_require_aware_utc)
    _validate_updated_at = field_validator("updated_at")(_require_aware_utc)


class ModelProfileCreateResult(_Schema):
    model_profile: SafeModelProfile
    created: bool


class ModelProfilePage(_Schema):
    items: tuple[SafeModelProfile, ...]
    next_cursor: str | None = None
