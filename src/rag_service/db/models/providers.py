"""Provider, model, sparse retrieval, and query profile mappings."""

from datetime import datetime
from decimal import Decimal
from typing import Any
from uuid import UUID

from sqlalchemy import (
    CHAR,
    BigInteger,
    Boolean,
    CheckConstraint,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    Numeric,
    String,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP
from sqlalchemy.dialects.postgresql import UUID as PostgreSQLUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from rag_service.db.base import Base, CreatedAt, UpdatedAt, UUIDPrimaryKey


def _empty_dict() -> dict[str, Any]:
    return {}


_HTTP_SITE_URL_PATTERN = (
    r"^https?://(\[[0-9A-Fa-f:.]+\]|[^:/?#@[:space:][:cntrl:]]+)"
    r"(:[0-9]{1,5})?(/[^?#[:space:][:cntrl:]]*)?$"
)
_HTTP_SITE_URL_EXPLICIT_PORT_PATTERN = (
    r"^https?://(\[[0-9A-Fa-f:.]+\]|[^:/?#@[:space:][:cntrl:]]+)"
    r":([0-9]{1,5})(/|$)"
)
_DEFAULT_HEADER_KEYS = ("HTTP-Referer", "X-OpenRouter-Title", "X-Title")
_OPENROUTER_ROUTING_KEYS = (
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
)
_OPENROUTER_SORT_VALUES = ("price", "throughput", "latency", "exacto")
_OPENROUTER_PERCENTILES = ("p50", "p75", "p90", "p99")
_OPENROUTER_MAX_PRICE_KEYS = ("audio", "prompt", "completion", "request", "image")
_OPENROUTER_PRICE_PATTERN = r"^(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?$"


def _sql_string_list(values: tuple[str, ...]) -> str:
    return ", ".join(f"'{value}'" for value in values)


def _jsonb_allowed_keys_check(expression: str, keys: tuple[str, ...]) -> str:
    return f"({expression} - ARRAY[{_sql_string_list(keys)}]::text[]) = '{{}}'::jsonb"


def _jsonb_object_key_count(expression: str) -> str:
    return (
        f"jsonb_array_length(jsonb_path_query_array({expression}, "
        "'$ ? (@.type() == \"object\").keyvalue()'))"
    )


def _optional_jsonb_type_check(
    column: str,
    key: str,
    expected_type: str,
    *,
    nullable: bool = False,
) -> str:
    if nullable:
        return (
            f"(NOT ({column} ? '{key}') OR "
            f"jsonb_typeof({column} -> '{key}') IN ('{expected_type}', 'null'))"
        )
    return f"(NOT ({column} ? '{key}') OR jsonb_typeof({column} -> '{key}') = '{expected_type}')"


def _http_url_port_check(expression: str) -> str:
    return (
        f"CASE WHEN {expression} !~* '{_HTTP_SITE_URL_EXPLICIT_PORT_PATTERN}' "
        f"THEN TRUE ELSE "
        f"((regexp_match({expression}, '{_HTTP_SITE_URL_EXPLICIT_PORT_PATTERN}', 'i'))[2])"
        f"::integer BETWEEN 1 AND 65535 END"
    )


def _jsonb_string_array_check(
    column: str,
    key: str,
    *,
    max_items: int,
    value_pattern: str,
) -> str:
    value = f"({column} -> '{key}')"
    return (
        f"CASE WHEN NOT ({column} ? '{key}') THEN TRUE "
        f"WHEN jsonb_typeof({value}) = 'null' THEN TRUE "
        f"WHEN jsonb_typeof({value}) <> 'array' THEN FALSE "
        f"WHEN jsonb_array_length({value}) > {max_items} THEN FALSE "
        f"ELSE NOT jsonb_path_exists({value}, "
        f'\'$[*] ? (@.type() != "string" || !(@ like_regex "{value_pattern}"))\') END'
    )


def _jsonb_nonnegative_object_member_check(expression: str, key: str) -> str:
    return (
        f"CASE WHEN NOT ({expression} ? '{key}') THEN TRUE "
        f"WHEN jsonb_typeof({expression} -> '{key}') = 'null' THEN TRUE "
        f"WHEN jsonb_typeof({expression} -> '{key}') <> 'number' THEN FALSE "
        f"ELSE ({expression} ->> '{key}')::numeric >= 0 END"
    )


def _jsonb_nonnegative_number_or_percentiles_check(column: str, key: str) -> str:
    value = f"({column} -> '{key}')"
    member_checks = " AND ".join(
        _jsonb_nonnegative_object_member_check(value, percentile)
        for percentile in _OPENROUTER_PERCENTILES
    )
    return (
        f"CASE WHEN NOT ({column} ? '{key}') THEN TRUE "
        f"WHEN jsonb_typeof({value}) = 'null' THEN TRUE "
        f"WHEN jsonb_typeof({value}) = 'number' "
        f"THEN ({column} ->> '{key}')::numeric >= 0 "
        f"WHEN jsonb_typeof({value}) = 'object' THEN "
        f"{_jsonb_object_key_count(value)} BETWEEN 0 AND {len(_OPENROUTER_PERCENTILES)} AND "
        f"{_jsonb_allowed_keys_check(value, _OPENROUTER_PERCENTILES)} AND {member_checks} "
        f"ELSE FALSE END"
    )


def _jsonb_price_object_member_check(expression: str, key: str) -> str:
    return (
        f"CASE WHEN NOT ({expression} ? '{key}') THEN TRUE "
        f"WHEN jsonb_typeof({expression} -> '{key}') = 'string' THEN "
        f"char_length({expression} ->> '{key}') BETWEEN 1 AND 64 "
        f"AND {expression} ->> '{key}' ~ '{_OPENROUTER_PRICE_PATTERN}' "
        f"ELSE FALSE END"
    )


def _jsonb_price_object_check(column: str, key: str) -> str:
    value = f"({column} -> '{key}')"
    member_checks = " AND ".join(
        _jsonb_price_object_member_check(value, price_key)
        for price_key in _OPENROUTER_MAX_PRICE_KEYS
    )
    return (
        f"CASE WHEN NOT ({column} ? '{key}') THEN TRUE "
        f"WHEN jsonb_typeof({value}) <> 'object' THEN FALSE "
        f"ELSE {_jsonb_object_key_count(value)} BETWEEN 0 AND "
        f"{len(_OPENROUTER_MAX_PRICE_KEYS)} "
        f"AND {_jsonb_allowed_keys_check(value, _OPENROUTER_MAX_PRICE_KEYS)} "
        f"AND {member_checks} END"
    )


def _openrouter_sort_check(column: str) -> str:
    value = f"({column} -> 'sort')"
    allowed_sort_values = _sql_string_list(_OPENROUTER_SORT_VALUES)
    return (
        f"CASE WHEN NOT ({column} ? 'sort') THEN TRUE "
        f"WHEN jsonb_typeof({value}) = 'null' THEN TRUE "
        f"WHEN jsonb_typeof({value}) = 'string' "
        f"THEN {column} ->> 'sort' IN ({allowed_sort_values}) "
        f"WHEN jsonb_typeof({value}) = 'object' THEN "
        f"{_jsonb_object_key_count(value)} BETWEEN 0 AND 2 "
        f"AND {_jsonb_allowed_keys_check(value, ('by', 'partition'))} "
        f"AND (NOT ({value} ? 'by') OR "
        f"jsonb_typeof({value} -> 'by') = 'null' OR "
        f"(jsonb_typeof({value} -> 'by') = 'string' "
        f"AND {value} ->> 'by' IN ({allowed_sort_values}))) "
        f"AND (NOT ({value} ? 'partition') OR "
        f"jsonb_typeof({value} -> 'partition') = 'null' OR "
        f"(jsonb_typeof({value} -> 'partition') = 'string' "
        f"AND {value} ->> 'partition' IN ('model', 'none'))) "
        f"ELSE FALSE END"
    )


class ProviderCredential(Base):
    __tablename__ = "provider_credentials"
    __table_args__ = (
        CheckConstraint(
            "char_length(name) BETWEEN 1 AND 120",
            name="ck_provider_credentials_name_length",
        ),
        CheckConstraint(
            "octet_length(nonce) = 12",
            name="ck_provider_credentials_nonce_length",
        ),
        CheckConstraint(
            "algorithm = 'AES-256-GCM'",
            name="ck_provider_credentials_algorithm",
        ),
        CheckConstraint(
            "char_length(key_version) BETWEEN 1 AND 64",
            name="ck_provider_credentials_key_version_length",
        ),
        CheckConstraint(
            "resource_revision >= 1",
            name="ck_provider_credentials_revision_positive",
        ),
        UniqueConstraint("name", name="uq_provider_credentials_name"),
    )

    id: Mapped[UUIDPrimaryKey]
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    ciphertext: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    nonce: Mapped[bytes] = mapped_column(LargeBinary(12), nullable=False)
    algorithm: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="AES-256-GCM",
        server_default=text("'AES-256-GCM'"),
    )
    key_version: Mapped[str] = mapped_column(String(64), nullable=False)
    resource_revision: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        default=1,
        server_default=text("1"),
    )
    created_at: Mapped[CreatedAt]
    updated_at: Mapped[UpdatedAt]
    rotated_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True))


class ProviderConfig(Base):
    __tablename__ = "provider_configs"
    __table_args__ = (
        CheckConstraint(
            "char_length(name) BETWEEN 1 AND 120",
            name="ck_provider_configs_name_length",
        ),
        CheckConstraint(
            "provider_type IN ('openai_compatible', 'openrouter', 'vendor_specific')",
            name="ck_provider_configs_provider_type",
        ),
        CheckConstraint(
            "char_length(base_url) BETWEEN 1 AND 2048",
            name="ck_provider_configs_base_url_length",
        ),
        CheckConstraint(
            f"base_url ~* '{_HTTP_SITE_URL_PATTERN}'",
            name="ck_provider_configs_base_url_http",
        ),
        CheckConstraint(
            _http_url_port_check("base_url"),
            name="ck_provider_configs_base_url_port",
        ),
        CheckConstraint(
            "char_length(secret_ref) BETWEEN 1 AND 255",
            name="ck_provider_configs_secret_ref_length",
        ),
        CheckConstraint(
            # Persistence syntax only; the provider service will own locator resolution.
            "secret_ref ~ '^(env|file|docker-secret|vault|aws-secrets-manager|"
            "gcp-secret-manager|azure-key-vault):[^[:space:][:cntrl:]]+$'",
            name="ck_provider_configs_secret_ref_scheme",
        ),
        CheckConstraint(
            "jsonb_typeof(default_headers) = 'object'",
            name="ck_provider_configs_default_headers_object",
        ),
        CheckConstraint(
            "pg_column_size(default_headers) <= 4096",
            name="ck_provider_configs_default_headers_size",
        ),
        CheckConstraint(
            "CASE WHEN jsonb_typeof(default_headers) = 'object' THEN "
            f"{_jsonb_object_key_count('default_headers')} <= 3 ELSE FALSE END",
            name="ck_provider_configs_default_headers_key_count",
        ),
        CheckConstraint(
            _jsonb_allowed_keys_check("default_headers", _DEFAULT_HEADER_KEYS),
            name="ck_provider_configs_default_headers_keys",
        ),
        CheckConstraint(
            " AND ".join(
                _optional_jsonb_type_check("default_headers", key, "string")
                for key in _DEFAULT_HEADER_KEYS
            ),
            name="ck_provider_configs_default_headers_value_types",
        ),
        CheckConstraint(
            "NOT (default_headers ? 'HTTP-Referer') OR "
            "(char_length(default_headers ->> 'HTTP-Referer') BETWEEN 1 AND 2048 "
            f"AND default_headers ->> 'HTTP-Referer' ~* '{_HTTP_SITE_URL_PATTERN}' "
            "AND " + _http_url_port_check("default_headers ->> 'HTTP-Referer'") + ")",
            name="ck_provider_configs_default_headers_http_referer",
        ),
        CheckConstraint(
            " AND ".join(
                f"(NOT (default_headers ? '{key}') OR "
                f"(char_length(default_headers ->> '{key}') BETWEEN 1 AND 120 "
                f"AND default_headers ->> '{key}' !~ '[[:cntrl:]]'))"
                for key in ("X-OpenRouter-Title", "X-Title")
            ),
            name="ck_provider_configs_default_headers_titles",
        ),
        CheckConstraint(
            "jsonb_typeof(routing_options) = 'object'",
            name="ck_provider_configs_routing_options_object",
        ),
        CheckConstraint(
            "pg_column_size(routing_options) <= 16384",
            name="ck_provider_configs_routing_options_size",
        ),
        CheckConstraint(
            "CASE WHEN jsonb_typeof(routing_options) = 'object' THEN "
            f"{_jsonb_object_key_count('routing_options')} "
            f"<= {len(_OPENROUTER_ROUTING_KEYS)} "
            "ELSE FALSE END",
            name="ck_provider_configs_routing_options_key_count",
        ),
        CheckConstraint(
            "provider_type = 'openrouter' OR routing_options = '{}'::jsonb",
            name="ck_provider_configs_routing_options_provider_scope",
        ),
        CheckConstraint(
            _jsonb_allowed_keys_check("routing_options", _OPENROUTER_ROUTING_KEYS),
            name="ck_provider_configs_routing_options_keys",
        ),
        CheckConstraint(
            " AND ".join(
                _jsonb_string_array_check(
                    "routing_options",
                    key,
                    max_items=100,
                    value_pattern=r"^[^\u0001-\u001F\u007F]{1,255}$",
                )
                for key in ("order", "only", "ignore")
            ),
            name="ck_provider_configs_routing_options_provider_arrays",
        ),
        CheckConstraint(
            " AND ".join(
                _optional_jsonb_type_check(
                    "routing_options",
                    key,
                    "boolean",
                    nullable=True,
                )
                for key in (
                    "allow_fallbacks",
                    "require_parameters",
                    "zdr",
                    "enforce_distillable_text",
                )
            ),
            name="ck_provider_configs_routing_options_booleans",
        ),
        CheckConstraint(
            "NOT (routing_options ? 'data_collection') OR "
            "jsonb_typeof(routing_options -> 'data_collection') = 'null' OR "
            "(jsonb_typeof(routing_options -> 'data_collection') = 'string' AND "
            "routing_options ->> 'data_collection' IN ('allow', 'deny'))",
            name="ck_provider_configs_routing_options_data_collection",
        ),
        CheckConstraint(
            _jsonb_string_array_check(
                "routing_options",
                "quantizations",
                max_items=32,
                value_pattern="^(int4|int8|fp4|fp6|fp8|fp16|bf16|fp32|unknown)$",
            ),
            name="ck_provider_configs_routing_options_quantizations",
        ),
        CheckConstraint(
            _openrouter_sort_check("routing_options"),
            name="ck_provider_configs_routing_options_sort",
        ),
        CheckConstraint(
            _jsonb_nonnegative_number_or_percentiles_check(
                "routing_options",
                "preferred_min_throughput",
            ),
            name="ck_provider_configs_routing_options_min_throughput",
        ),
        CheckConstraint(
            _jsonb_nonnegative_number_or_percentiles_check(
                "routing_options",
                "preferred_max_latency",
            ),
            name="ck_provider_configs_routing_options_max_latency",
        ),
        CheckConstraint(
            _jsonb_price_object_check("routing_options", "max_price"),
            name="ck_provider_configs_routing_options_max_price",
        ),
        CheckConstraint(
            "timeout_seconds > 0 AND timeout_seconds <= 600",
            name="ck_provider_configs_timeout",
        ),
        CheckConstraint(
            "max_concurrency BETWEEN 1 AND 10000",
            name="ck_provider_configs_max_concurrency",
        ),
        CheckConstraint(
            "requests_per_minute BETWEEN 1 AND 1000000",
            name="ck_provider_configs_requests_per_minute",
        ),
        CheckConstraint(
            "(secret_ref IS NULL) <> (credential_id IS NULL)",
            name="ck_provider_configs_credential_source_exactly_one",
        ),
        CheckConstraint(
            "(credential_id IS NULL AND endpoint_policy_version IS NULL "
            "AND endpoint_validated_at IS NULL) OR "
            "(credential_id IS NOT NULL AND endpoint_policy_version IS NOT NULL "
            "AND endpoint_validated_at IS NOT NULL)",
            name="ck_provider_configs_credential_endpoint_validation",
        ),
        CheckConstraint(
            "endpoint_policy_version IS NULL "
            "OR char_length(endpoint_policy_version) BETWEEN 1 AND 64",
            name="ck_provider_configs_endpoint_policy_version_length",
        ),
        CheckConstraint(
            "resource_revision >= 1",
            name="ck_provider_configs_revision_positive",
        ),
        UniqueConstraint("name", name="uq_provider_configs_name"),
        Index("ix_provider_configs_credential_id", "credential_id"),
    )

    id: Mapped[UUIDPrimaryKey]
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    provider_type: Mapped[str] = mapped_column(String(32), nullable=False)
    base_url: Mapped[str] = mapped_column(String(2048), nullable=False)
    secret_ref: Mapped[str | None] = mapped_column(String(255))
    default_headers: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=_empty_dict,
        server_default=text("'{}'::jsonb"),
    )
    routing_options: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=_empty_dict,
        server_default=text("'{}'::jsonb"),
    )
    timeout_seconds: Mapped[Decimal] = mapped_column(Numeric(8, 3), nullable=False)
    max_concurrency: Mapped[int] = mapped_column(Integer, nullable=False)
    requests_per_minute: Mapped[int] = mapped_column(Integer, nullable=False)
    enabled: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        server_default=text("true"),
    )
    created_at: Mapped[CreatedAt]
    updated_at: Mapped[UpdatedAt]
    credential_id: Mapped[UUID | None] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey(
            "provider_credentials.id",
            name="fk_provider_configs_credential",
            ondelete="RESTRICT",
        ),
    )
    resource_revision: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        default=1,
        server_default=text("1"),
    )
    endpoint_policy_version: Mapped[str | None] = mapped_column(String(64))
    endpoint_validated_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True))


class ModelProfile(Base):
    __tablename__ = "model_profiles"
    __table_args__ = (
        CheckConstraint(
            "char_length(name) BETWEEN 1 AND 120",
            name="ck_model_profiles_name_length",
        ),
        CheckConstraint(
            "capability IN ('embedding', 'rerank', 'chat')",
            name="ck_model_profiles_capability",
        ),
        CheckConstraint(
            "char_length(model_name) BETWEEN 1 AND 255",
            name="ck_model_profiles_model_name_length",
        ),
        CheckConstraint(
            "(capability = 'embedding' AND dimension IS NOT NULL AND dimension > 0) "
            "OR (capability IN ('rerank', 'chat') AND dimension IS NULL)",
            name="ck_model_profiles_dimension",
        ),
        CheckConstraint(
            "max_input_tokens BETWEEN 1 AND 10000000",
            name="ck_model_profiles_max_input_tokens",
        ),
        CheckConstraint(
            "batch_size BETWEEN 1 AND 10000",
            name="ck_model_profiles_batch_size",
        ),
        CheckConstraint(
            "timeout_seconds > 0 AND timeout_seconds <= 600",
            name="ck_model_profiles_timeout",
        ),
        CheckConstraint(
            "resource_revision >= 1",
            name="ck_model_profiles_revision_positive",
        ),
        CheckConstraint(
            "jsonb_typeof(vector_config) = 'object'",
            name="ck_model_profiles_vector_config_object",
        ),
        UniqueConstraint("name", name="uq_model_profiles_name"),
    )

    id: Mapped[UUIDPrimaryKey]
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    capability: Mapped[str] = mapped_column(String(16), nullable=False)
    provider_config_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey(
            "provider_configs.id",
            name="fk_model_profiles_provider_config",
            ondelete="RESTRICT",
        ),
        nullable=False,
    )
    model_name: Mapped[str] = mapped_column(String(255), nullable=False)
    dimension: Mapped[int | None] = mapped_column(Integer)
    max_input_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    batch_size: Mapped[int] = mapped_column(Integer, nullable=False)
    timeout_seconds: Mapped[Decimal] = mapped_column(Numeric(8, 3), nullable=False)
    enabled: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        server_default=text("true"),
    )
    created_at: Mapped[CreatedAt]
    updated_at: Mapped[UpdatedAt]
    resource_revision: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        default=1,
        server_default=text("1"),
    )
    vector_config: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=_empty_dict,
        server_default=text("'{}'::jsonb"),
    )


class ModelProfileFallback(Base):
    __tablename__ = "model_profile_fallbacks"
    __table_args__ = (
        CheckConstraint(
            "priority BETWEEN 1 AND 100",
            name="ck_model_profile_fallbacks_priority",
        ),
        CheckConstraint(
            "profile_id <> fallback_profile_id",
            name="ck_model_profile_fallbacks_distinct",
        ),
        UniqueConstraint(
            "profile_id",
            "priority",
            name="uq_model_profile_fallbacks_profile_priority",
        ),
    )

    profile_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey(
            "model_profiles.id",
            name="fk_model_profile_fallbacks_profile",
            ondelete="CASCADE",
        ),
        primary_key=True,
    )
    fallback_profile_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey(
            "model_profiles.id",
            name="fk_model_profile_fallbacks_fallback",
            ondelete="RESTRICT",
        ),
        primary_key=True,
    )
    priority: Mapped[int] = mapped_column(Integer, nullable=False)


class SparseProfile(Base):
    __tablename__ = "sparse_profiles"
    __table_args__ = (
        CheckConstraint(
            "char_length(name) BETWEEN 1 AND 120",
            name="ck_sparse_profiles_name_length",
        ),
        CheckConstraint(
            "algorithm = 'qdrant_bm25_v1'",
            name="ck_sparse_profiles_algorithm",
        ),
        CheckConstraint(
            "char_length(encoder_package) BETWEEN 1 AND 255",
            name="ck_sparse_profiles_encoder_package_length",
        ),
        CheckConstraint(
            "char_length(encoder_version) BETWEEN 1 AND 64",
            name="ck_sparse_profiles_encoder_version_length",
        ),
        CheckConstraint(
            "char_length(tokenizer_name) BETWEEN 1 AND 120",
            name="ck_sparse_profiles_tokenizer_name_length",
        ),
        CheckConstraint(
            "char_length(tokenizer_version) BETWEEN 1 AND 64",
            name="ck_sparse_profiles_tokenizer_version_length",
        ),
        CheckConstraint(
            "char_length(idf_object_key) BETWEEN 1 AND 1024",
            name="ck_sparse_profiles_idf_object_key_length",
        ),
        CheckConstraint(
            "idf_checksum_sha256 ~ '^[0-9a-f]{64}$'",
            name="ck_sparse_profiles_idf_checksum",
        ),
        CheckConstraint(
            "config_hash ~ '^[0-9a-f]{64}$'",
            name="ck_sparse_profiles_config_hash",
        ),
        UniqueConstraint("name", name="uq_sparse_profiles_name"),
        UniqueConstraint("config_hash", name="uq_sparse_profiles_config_hash"),
    )

    id: Mapped[UUIDPrimaryKey]
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    algorithm: Mapped[str] = mapped_column(String(64), nullable=False)
    encoder_package: Mapped[str] = mapped_column(String(255), nullable=False)
    encoder_version: Mapped[str] = mapped_column(String(64), nullable=False)
    tokenizer_name: Mapped[str] = mapped_column(String(120), nullable=False)
    tokenizer_version: Mapped[str] = mapped_column(String(64), nullable=False)
    language_config: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=_empty_dict,
        server_default=text("'{}'::jsonb"),
    )
    term_frequency_config: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=_empty_dict,
        server_default=text("'{}'::jsonb"),
    )
    length_normalization_config: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=_empty_dict,
        server_default=text("'{}'::jsonb"),
    )
    idf_object_key: Mapped[str] = mapped_column(String(1024), nullable=False)
    idf_checksum_sha256: Mapped[str] = mapped_column(CHAR(64), nullable=False)
    oov_config: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=_empty_dict,
        server_default=text("'{}'::jsonb"),
    )
    config_hash: Mapped[str] = mapped_column(CHAR(64), nullable=False)
    enabled: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        server_default=text("true"),
    )
    created_at: Mapped[CreatedAt]
    updated_at: Mapped[UpdatedAt]


class QueryProfile(Base):
    __tablename__ = "query_profiles"
    __table_args__ = (
        CheckConstraint(
            "char_length(name) BETWEEN 1 AND 120",
            name="ck_query_profiles_name_length",
        ),
        CheckConstraint(
            "dense_candidate_limit BETWEEN 1 AND 1000 "
            "AND sparse_candidate_limit BETWEEN 1 AND 1000 "
            "AND rrf_candidate_limit BETWEEN 1 AND 1000 "
            "AND rerank_candidate_limit BETWEEN 1 AND 1000 "
            "AND rerank_candidate_limit <= rrf_candidate_limit",
            name="ck_query_profiles_candidate_limits",
        ),
        CheckConstraint(
            "top_k_limit BETWEEN 1 AND 100 AND top_k_limit <= rerank_candidate_limit",
            name="ck_query_profiles_top_k_limit",
        ),
        CheckConstraint(
            "min_rerank_score BETWEEN -1 AND 1",
            name="ck_query_profiles_min_rerank_score",
        ),
        CheckConstraint(
            "min_rrf_score_when_degraded BETWEEN 0 AND 1",
            name="ck_query_profiles_min_rrf_degraded",
        ),
        CheckConstraint(
            "context_token_budget BETWEEN 1 AND 1000000",
            name="ck_query_profiles_context_budget",
        ),
        Index(
            "uq_query_profiles_enabled_system_default",
            "is_system_default",
            unique=True,
            postgresql_where=text("enabled AND is_system_default"),
        ),
        UniqueConstraint("name", name="uq_query_profiles_name"),
    )

    id: Mapped[UUIDPrimaryKey]
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    rerank_profile_id: Mapped[UUID | None] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey(
            "model_profiles.id",
            name="fk_query_profiles_rerank_profile",
            ondelete="RESTRICT",
        ),
    )
    chat_profile_id: Mapped[UUID | None] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey(
            "model_profiles.id",
            name="fk_query_profiles_chat_profile",
            ondelete="RESTRICT",
        ),
    )
    dense_candidate_limit: Mapped[int] = mapped_column(Integer, nullable=False)
    sparse_candidate_limit: Mapped[int] = mapped_column(Integer, nullable=False)
    rrf_candidate_limit: Mapped[int] = mapped_column(Integer, nullable=False)
    rerank_candidate_limit: Mapped[int] = mapped_column(Integer, nullable=False)
    top_k_limit: Mapped[int] = mapped_column(Integer, nullable=False)
    min_rerank_score: Mapped[Decimal] = mapped_column(Numeric(12, 8), nullable=False)
    min_rrf_score_when_degraded: Mapped[Decimal] = mapped_column(
        Numeric(12, 8),
        nullable=False,
    )
    context_token_budget: Mapped[int] = mapped_column(Integer, nullable=False)
    enabled: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        server_default=text("true"),
    )
    is_system_default: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default=text("false"),
    )
    created_at: Mapped[CreatedAt]
    updated_at: Mapped[UpdatedAt]

    rerank_profile: Mapped[ModelProfile | None] = relationship(foreign_keys=[rerank_profile_id])
    chat_profile: Mapped[ModelProfile | None] = relationship(foreign_keys=[chat_profile_id])
