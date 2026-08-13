"""Create the authentication and metadata persistence schema.

Revision ID: 20260724_0002
Revises: 20260723_0001
Create Date: 2026-07-24 16:49:17.958034

"""

# ruff: noqa: E501

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "20260724_0002"
down_revision: str | Sequence[str] | None = "20260723_0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""

    op.create_table(
        "provider_configs",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("name", sa.String(length=120), nullable=False),
        sa.Column("provider_type", sa.String(length=32), nullable=False),
        sa.Column("base_url", sa.String(length=2048), nullable=False),
        sa.Column("secret_ref", sa.String(length=255), nullable=False),
        sa.Column(
            "default_headers",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "routing_options",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column("timeout_seconds", sa.Numeric(precision=8, scale=3), nullable=False),
        sa.Column("max_concurrency", sa.Integer(), nullable=False),
        sa.Column("requests_per_minute", sa.Integer(), nullable=False),
        sa.Column("enabled", sa.Boolean(), server_default=sa.text("true"), nullable=False),
        sa.Column(
            "created_at",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "(NOT (default_headers ? 'HTTP-Referer') OR jsonb_typeof(default_headers -> 'HTTP-Referer') = 'string') AND (NOT (default_headers ? 'X-OpenRouter-Title') OR jsonb_typeof(default_headers -> 'X-OpenRouter-Title') = 'string') AND (NOT (default_headers ? 'X-Title') OR jsonb_typeof(default_headers -> 'X-Title') = 'string')",
            name="ck_provider_configs_default_headers_value_types",
        ),
        sa.CheckConstraint(
            "(NOT (default_headers ? 'X-OpenRouter-Title') OR (char_length(default_headers ->> 'X-OpenRouter-Title') BETWEEN 1 AND 120 AND default_headers ->> 'X-OpenRouter-Title' !~ '[[:cntrl:]]')) AND (NOT (default_headers ? 'X-Title') OR (char_length(default_headers ->> 'X-Title') BETWEEN 1 AND 120 AND default_headers ->> 'X-Title' !~ '[[:cntrl:]]'))",
            name="ck_provider_configs_default_headers_titles",
        ),
        sa.CheckConstraint(
            "(NOT (routing_options ? 'allow_fallbacks') OR jsonb_typeof(routing_options -> 'allow_fallbacks') IN ('boolean', 'null')) AND (NOT (routing_options ? 'require_parameters') OR jsonb_typeof(routing_options -> 'require_parameters') IN ('boolean', 'null')) AND (NOT (routing_options ? 'zdr') OR jsonb_typeof(routing_options -> 'zdr') IN ('boolean', 'null')) AND (NOT (routing_options ? 'enforce_distillable_text') OR jsonb_typeof(routing_options -> 'enforce_distillable_text') IN ('boolean', 'null'))",
            name="ck_provider_configs_routing_options_booleans",
        ),
        sa.CheckConstraint(
            "(default_headers - ARRAY['HTTP-Referer', 'X-OpenRouter-Title', 'X-Title']::text[]) = '{}'::jsonb",
            name="ck_provider_configs_default_headers_keys",
        ),
        sa.CheckConstraint(
            "(routing_options - ARRAY['order', 'allow_fallbacks', 'require_parameters', 'data_collection', 'zdr', 'enforce_distillable_text', 'only', 'ignore', 'quantizations', 'sort', 'preferred_min_throughput', 'preferred_max_latency', 'max_price']::text[]) = '{}'::jsonb",
            name="ck_provider_configs_routing_options_keys",
        ),
        sa.CheckConstraint(
            "CASE WHEN base_url !~* '^https?://(\\[[0-9A-Fa-f:.]+\\]|[^:/?#@[:space:][:cntrl:]]+):([0-9]{1,5})(/|$)' THEN TRUE ELSE ((regexp_match(base_url, '^https?://(\\[[0-9A-Fa-f:.]+\\]|[^:/?#@[:space:][:cntrl:]]+):([0-9]{1,5})(/|$)', 'i'))[2])::integer BETWEEN 1 AND 65535 END",
            name="ck_provider_configs_base_url_port",
        ),
        sa.CheckConstraint(
            "NOT (default_headers ? 'HTTP-Referer') OR (char_length(default_headers ->> 'HTTP-Referer') BETWEEN 1 AND 2048 AND default_headers ->> 'HTTP-Referer' ~* '^https?://(\\[[0-9A-Fa-f:.]+\\]|[^:/?#@[:space:][:cntrl:]]+)(:[0-9]{1,5})?(/[^?#[:space:][:cntrl:]]*)?$' AND CASE WHEN default_headers ->> 'HTTP-Referer' !~* '^https?://(\\[[0-9A-Fa-f:.]+\\]|[^:/?#@[:space:][:cntrl:]]+):([0-9]{1,5})(/|$)' THEN TRUE ELSE ((regexp_match(default_headers ->> 'HTTP-Referer', '^https?://(\\[[0-9A-Fa-f:.]+\\]|[^:/?#@[:space:][:cntrl:]]+):([0-9]{1,5})(/|$)', 'i'))[2])::integer BETWEEN 1 AND 65535 END)",
            name="ck_provider_configs_default_headers_http_referer",
        ),
        sa.CheckConstraint(
            "NOT (routing_options ? 'data_collection') OR jsonb_typeof(routing_options -> 'data_collection') = 'null' OR (jsonb_typeof(routing_options -> 'data_collection') = 'string' AND routing_options ->> 'data_collection' IN ('allow', 'deny'))",
            name="ck_provider_configs_routing_options_data_collection",
        ),
        sa.CheckConstraint(
            "base_url ~* '^https?://(\\[[0-9A-Fa-f:.]+\\]|[^:/?#@[:space:][:cntrl:]]+)(:[0-9]{1,5})?(/[^?#[:space:][:cntrl:]]*)?$'",
            name="ck_provider_configs_base_url_http",
        ),
        sa.CheckConstraint(
            "jsonb_typeof(default_headers) = 'object'",
            name="ck_provider_configs_default_headers_object",
        ),
        sa.CheckConstraint(
            "jsonb_typeof(routing_options) = 'object'",
            name="ck_provider_configs_routing_options_object",
        ),
        sa.CheckConstraint(
            "provider_type = 'openrouter' OR routing_options = '{}'::jsonb",
            name="ck_provider_configs_routing_options_provider_scope",
        ),
        sa.CheckConstraint(
            "provider_type IN ('openai_compatible', 'openrouter', 'vendor_specific')",
            name="ck_provider_configs_provider_type",
        ),
        sa.CheckConstraint(
            "secret_ref ~ '^(env|file|docker-secret|vault|aws-secrets-manager|gcp-secret-manager|azure-key-vault):[^[:space:][:cntrl:]]+$'",
            name="ck_provider_configs_secret_ref_scheme",
        ),
        sa.CheckConstraint(
            "CASE WHEN NOT (routing_options ? 'max_price') THEN TRUE WHEN jsonb_typeof((routing_options -> 'max_price')) <> 'object' THEN FALSE ELSE jsonb_array_length(jsonb_path_query_array((routing_options -> 'max_price'), '$ ? (@.type() == \"object\").keyvalue()')) BETWEEN 0 AND 5 AND ((routing_options -> 'max_price') - ARRAY['audio', 'prompt', 'completion', 'request', 'image']::text[]) = '{}'::jsonb AND CASE WHEN NOT ((routing_options -> 'max_price') ? 'audio') THEN TRUE WHEN jsonb_typeof((routing_options -> 'max_price') -> 'audio') = 'string' THEN char_length((routing_options -> 'max_price') ->> 'audio') BETWEEN 1 AND 64 AND (routing_options -> 'max_price') ->> 'audio' ~ '^(0|[1-9][0-9]*)(\\.[0-9]+)?([eE][+-]?[0-9]+)?$' ELSE FALSE END AND CASE WHEN NOT ((routing_options -> 'max_price') ? 'prompt') THEN TRUE WHEN jsonb_typeof((routing_options -> 'max_price') -> 'prompt') = 'string' THEN char_length((routing_options -> 'max_price') ->> 'prompt') BETWEEN 1 AND 64 AND (routing_options -> 'max_price') ->> 'prompt' ~ '^(0|[1-9][0-9]*)(\\.[0-9]+)?([eE][+-]?[0-9]+)?$' ELSE FALSE END AND CASE WHEN NOT ((routing_options -> 'max_price') ? 'completion') THEN TRUE WHEN jsonb_typeof((routing_options -> 'max_price') -> 'completion') = 'string' THEN char_length((routing_options -> 'max_price') ->> 'completion') BETWEEN 1 AND 64 AND (routing_options -> 'max_price') ->> 'completion' ~ '^(0|[1-9][0-9]*)(\\.[0-9]+)?([eE][+-]?[0-9]+)?$' ELSE FALSE END AND CASE WHEN NOT ((routing_options -> 'max_price') ? 'request') THEN TRUE WHEN jsonb_typeof((routing_options -> 'max_price') -> 'request') = 'string' THEN char_length((routing_options -> 'max_price') ->> 'request') BETWEEN 1 AND 64 AND (routing_options -> 'max_price') ->> 'request' ~ '^(0|[1-9][0-9]*)(\\.[0-9]+)?([eE][+-]?[0-9]+)?$' ELSE FALSE END AND CASE WHEN NOT ((routing_options -> 'max_price') ? 'image') THEN TRUE WHEN jsonb_typeof((routing_options -> 'max_price') -> 'image') = 'string' THEN char_length((routing_options -> 'max_price') ->> 'image') BETWEEN 1 AND 64 AND (routing_options -> 'max_price') ->> 'image' ~ '^(0|[1-9][0-9]*)(\\.[0-9]+)?([eE][+-]?[0-9]+)?$' ELSE FALSE END END",
            name="ck_provider_configs_routing_options_max_price",
        ),
        sa.CheckConstraint(
            "CASE WHEN NOT (routing_options ? 'order') THEN TRUE WHEN jsonb_typeof((routing_options -> 'order')) = 'null' THEN TRUE WHEN jsonb_typeof((routing_options -> 'order')) <> 'array' THEN FALSE WHEN jsonb_array_length((routing_options -> 'order')) > 100 THEN FALSE ELSE NOT jsonb_path_exists((routing_options -> 'order'), '$[*] ? (@.type() != \"string\" || !(@ like_regex \"^[^\\u0001-\\u001F\\u007F]{1,255}$\"))') END AND CASE WHEN NOT (routing_options ? 'only') THEN TRUE WHEN jsonb_typeof((routing_options -> 'only')) = 'null' THEN TRUE WHEN jsonb_typeof((routing_options -> 'only')) <> 'array' THEN FALSE WHEN jsonb_array_length((routing_options -> 'only')) > 100 THEN FALSE ELSE NOT jsonb_path_exists((routing_options -> 'only'), '$[*] ? (@.type() != \"string\" || !(@ like_regex \"^[^\\u0001-\\u001F\\u007F]{1,255}$\"))') END AND CASE WHEN NOT (routing_options ? 'ignore') THEN TRUE WHEN jsonb_typeof((routing_options -> 'ignore')) = 'null' THEN TRUE WHEN jsonb_typeof((routing_options -> 'ignore')) <> 'array' THEN FALSE WHEN jsonb_array_length((routing_options -> 'ignore')) > 100 THEN FALSE ELSE NOT jsonb_path_exists((routing_options -> 'ignore'), '$[*] ? (@.type() != \"string\" || !(@ like_regex \"^[^\\u0001-\\u001F\\u007F]{1,255}$\"))') END",
            name="ck_provider_configs_routing_options_provider_arrays",
        ),
        sa.CheckConstraint(
            "CASE WHEN NOT (routing_options ? 'preferred_max_latency') THEN TRUE WHEN jsonb_typeof((routing_options -> 'preferred_max_latency')) = 'null' THEN TRUE WHEN jsonb_typeof((routing_options -> 'preferred_max_latency')) = 'number' THEN (routing_options ->> 'preferred_max_latency')::numeric >= 0 WHEN jsonb_typeof((routing_options -> 'preferred_max_latency')) = 'object' THEN jsonb_array_length(jsonb_path_query_array((routing_options -> 'preferred_max_latency'), '$ ? (@.type() == \"object\").keyvalue()')) BETWEEN 0 AND 4 AND ((routing_options -> 'preferred_max_latency') - ARRAY['p50', 'p75', 'p90', 'p99']::text[]) = '{}'::jsonb AND CASE WHEN NOT ((routing_options -> 'preferred_max_latency') ? 'p50') THEN TRUE WHEN jsonb_typeof((routing_options -> 'preferred_max_latency') -> 'p50') = 'null' THEN TRUE WHEN jsonb_typeof((routing_options -> 'preferred_max_latency') -> 'p50') <> 'number' THEN FALSE ELSE ((routing_options -> 'preferred_max_latency') ->> 'p50')::numeric >= 0 END AND CASE WHEN NOT ((routing_options -> 'preferred_max_latency') ? 'p75') THEN TRUE WHEN jsonb_typeof((routing_options -> 'preferred_max_latency') -> 'p75') = 'null' THEN TRUE WHEN jsonb_typeof((routing_options -> 'preferred_max_latency') -> 'p75') <> 'number' THEN FALSE ELSE ((routing_options -> 'preferred_max_latency') ->> 'p75')::numeric >= 0 END AND CASE WHEN NOT ((routing_options -> 'preferred_max_latency') ? 'p90') THEN TRUE WHEN jsonb_typeof((routing_options -> 'preferred_max_latency') -> 'p90') = 'null' THEN TRUE WHEN jsonb_typeof((routing_options -> 'preferred_max_latency') -> 'p90') <> 'number' THEN FALSE ELSE ((routing_options -> 'preferred_max_latency') ->> 'p90')::numeric >= 0 END AND CASE WHEN NOT ((routing_options -> 'preferred_max_latency') ? 'p99') THEN TRUE WHEN jsonb_typeof((routing_options -> 'preferred_max_latency') -> 'p99') = 'null' THEN TRUE WHEN jsonb_typeof((routing_options -> 'preferred_max_latency') -> 'p99') <> 'number' THEN FALSE ELSE ((routing_options -> 'preferred_max_latency') ->> 'p99')::numeric >= 0 END ELSE FALSE END",
            name="ck_provider_configs_routing_options_max_latency",
        ),
        sa.CheckConstraint(
            "CASE WHEN NOT (routing_options ? 'preferred_min_throughput') THEN TRUE WHEN jsonb_typeof((routing_options -> 'preferred_min_throughput')) = 'null' THEN TRUE WHEN jsonb_typeof((routing_options -> 'preferred_min_throughput')) = 'number' THEN (routing_options ->> 'preferred_min_throughput')::numeric >= 0 WHEN jsonb_typeof((routing_options -> 'preferred_min_throughput')) = 'object' THEN jsonb_array_length(jsonb_path_query_array((routing_options -> 'preferred_min_throughput'), '$ ? (@.type() == \"object\").keyvalue()')) BETWEEN 0 AND 4 AND ((routing_options -> 'preferred_min_throughput') - ARRAY['p50', 'p75', 'p90', 'p99']::text[]) = '{}'::jsonb AND CASE WHEN NOT ((routing_options -> 'preferred_min_throughput') ? 'p50') THEN TRUE WHEN jsonb_typeof((routing_options -> 'preferred_min_throughput') -> 'p50') = 'null' THEN TRUE WHEN jsonb_typeof((routing_options -> 'preferred_min_throughput') -> 'p50') <> 'number' THEN FALSE ELSE ((routing_options -> 'preferred_min_throughput') ->> 'p50')::numeric >= 0 END AND CASE WHEN NOT ((routing_options -> 'preferred_min_throughput') ? 'p75') THEN TRUE WHEN jsonb_typeof((routing_options -> 'preferred_min_throughput') -> 'p75') = 'null' THEN TRUE WHEN jsonb_typeof((routing_options -> 'preferred_min_throughput') -> 'p75') <> 'number' THEN FALSE ELSE ((routing_options -> 'preferred_min_throughput') ->> 'p75')::numeric >= 0 END AND CASE WHEN NOT ((routing_options -> 'preferred_min_throughput') ? 'p90') THEN TRUE WHEN jsonb_typeof((routing_options -> 'preferred_min_throughput') -> 'p90') = 'null' THEN TRUE WHEN jsonb_typeof((routing_options -> 'preferred_min_throughput') -> 'p90') <> 'number' THEN FALSE ELSE ((routing_options -> 'preferred_min_throughput') ->> 'p90')::numeric >= 0 END AND CASE WHEN NOT ((routing_options -> 'preferred_min_throughput') ? 'p99') THEN TRUE WHEN jsonb_typeof((routing_options -> 'preferred_min_throughput') -> 'p99') = 'null' THEN TRUE WHEN jsonb_typeof((routing_options -> 'preferred_min_throughput') -> 'p99') <> 'number' THEN FALSE ELSE ((routing_options -> 'preferred_min_throughput') ->> 'p99')::numeric >= 0 END ELSE FALSE END",
            name="ck_provider_configs_routing_options_min_throughput",
        ),
        sa.CheckConstraint(
            "CASE WHEN NOT (routing_options ? 'quantizations') THEN TRUE WHEN jsonb_typeof((routing_options -> 'quantizations')) = 'null' THEN TRUE WHEN jsonb_typeof((routing_options -> 'quantizations')) <> 'array' THEN FALSE WHEN jsonb_array_length((routing_options -> 'quantizations')) > 32 THEN FALSE ELSE NOT jsonb_path_exists((routing_options -> 'quantizations'), '$[*] ? (@.type() != \"string\" || !(@ like_regex \"^(int4|int8|fp4|fp6|fp8|fp16|bf16|fp32|unknown)$\"))') END",
            name="ck_provider_configs_routing_options_quantizations",
        ),
        sa.CheckConstraint(
            "CASE WHEN NOT (routing_options ? 'sort') THEN TRUE WHEN jsonb_typeof((routing_options -> 'sort')) = 'null' THEN TRUE WHEN jsonb_typeof((routing_options -> 'sort')) = 'string' THEN routing_options ->> 'sort' IN ('price', 'throughput', 'latency', 'exacto') WHEN jsonb_typeof((routing_options -> 'sort')) = 'object' THEN jsonb_array_length(jsonb_path_query_array((routing_options -> 'sort'), '$ ? (@.type() == \"object\").keyvalue()')) BETWEEN 0 AND 2 AND ((routing_options -> 'sort') - ARRAY['by', 'partition']::text[]) = '{}'::jsonb AND (NOT ((routing_options -> 'sort') ? 'by') OR jsonb_typeof((routing_options -> 'sort') -> 'by') = 'null' OR (jsonb_typeof((routing_options -> 'sort') -> 'by') = 'string' AND (routing_options -> 'sort') ->> 'by' IN ('price', 'throughput', 'latency', 'exacto'))) AND (NOT ((routing_options -> 'sort') ? 'partition') OR jsonb_typeof((routing_options -> 'sort') -> 'partition') = 'null' OR (jsonb_typeof((routing_options -> 'sort') -> 'partition') = 'string' AND (routing_options -> 'sort') ->> 'partition' IN ('model', 'none'))) ELSE FALSE END",
            name="ck_provider_configs_routing_options_sort",
        ),
        sa.CheckConstraint(
            "CASE WHEN jsonb_typeof(default_headers) = 'object' THEN jsonb_array_length(jsonb_path_query_array(default_headers, '$ ? (@.type() == \"object\").keyvalue()')) <= 3 ELSE FALSE END",
            name="ck_provider_configs_default_headers_key_count",
        ),
        sa.CheckConstraint(
            "CASE WHEN jsonb_typeof(routing_options) = 'object' THEN jsonb_array_length(jsonb_path_query_array(routing_options, '$ ? (@.type() == \"object\").keyvalue()')) <= 13 ELSE FALSE END",
            name="ck_provider_configs_routing_options_key_count",
        ),
        sa.CheckConstraint(
            "char_length(base_url) BETWEEN 1 AND 2048", name="ck_provider_configs_base_url_length"
        ),
        sa.CheckConstraint(
            "char_length(name) BETWEEN 1 AND 120", name="ck_provider_configs_name_length"
        ),
        sa.CheckConstraint(
            "char_length(secret_ref) BETWEEN 1 AND 255",
            name="ck_provider_configs_secret_ref_length",
        ),
        sa.CheckConstraint(
            "max_concurrency BETWEEN 1 AND 10000", name="ck_provider_configs_max_concurrency"
        ),
        sa.CheckConstraint(
            "pg_column_size(default_headers) <= 4096",
            name="ck_provider_configs_default_headers_size",
        ),
        sa.CheckConstraint(
            "pg_column_size(routing_options) <= 16384",
            name="ck_provider_configs_routing_options_size",
        ),
        sa.CheckConstraint(
            "requests_per_minute BETWEEN 1 AND 1000000",
            name="ck_provider_configs_requests_per_minute",
        ),
        sa.CheckConstraint(
            "timeout_seconds > 0 AND timeout_seconds <= 600", name="ck_provider_configs_timeout"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name", name="uq_provider_configs_name"),
    )

    op.create_table(
        "model_profiles",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("name", sa.String(length=120), nullable=False),
        sa.Column("capability", sa.String(length=16), nullable=False),
        sa.Column("provider_config_id", sa.UUID(), nullable=False),
        sa.Column("model_name", sa.String(length=255), nullable=False),
        sa.Column("dimension", sa.Integer(), nullable=True),
        sa.Column("max_input_tokens", sa.Integer(), nullable=False),
        sa.Column("batch_size", sa.Integer(), nullable=False),
        sa.Column("timeout_seconds", sa.Numeric(precision=8, scale=3), nullable=False),
        sa.Column("enabled", sa.Boolean(), server_default=sa.text("true"), nullable=False),
        sa.Column(
            "created_at",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "(capability = 'embedding' AND dimension IS NOT NULL AND dimension > 0) OR (capability IN ('rerank', 'chat') AND dimension IS NULL)",
            name="ck_model_profiles_dimension",
        ),
        sa.CheckConstraint(
            "capability IN ('embedding', 'rerank', 'chat')", name="ck_model_profiles_capability"
        ),
        sa.CheckConstraint("batch_size BETWEEN 1 AND 10000", name="ck_model_profiles_batch_size"),
        sa.CheckConstraint(
            "char_length(model_name) BETWEEN 1 AND 255", name="ck_model_profiles_model_name_length"
        ),
        sa.CheckConstraint(
            "char_length(name) BETWEEN 1 AND 120", name="ck_model_profiles_name_length"
        ),
        sa.CheckConstraint(
            "max_input_tokens BETWEEN 1 AND 10000000", name="ck_model_profiles_max_input_tokens"
        ),
        sa.CheckConstraint(
            "timeout_seconds > 0 AND timeout_seconds <= 600", name="ck_model_profiles_timeout"
        ),
        sa.ForeignKeyConstraint(
            ["provider_config_id"],
            ["provider_configs.id"],
            name="fk_model_profiles_provider_config",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name", name="uq_model_profiles_name"),
    )

    op.create_table(
        "model_profile_fallbacks",
        sa.Column("profile_id", sa.UUID(), nullable=False),
        sa.Column("fallback_profile_id", sa.UUID(), nullable=False),
        sa.Column("priority", sa.Integer(), nullable=False),
        sa.CheckConstraint(
            "priority BETWEEN 1 AND 100", name="ck_model_profile_fallbacks_priority"
        ),
        sa.CheckConstraint(
            "profile_id <> fallback_profile_id", name="ck_model_profile_fallbacks_distinct"
        ),
        sa.ForeignKeyConstraint(
            ["fallback_profile_id"],
            ["model_profiles.id"],
            name="fk_model_profile_fallbacks_fallback",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["profile_id"],
            ["model_profiles.id"],
            name="fk_model_profile_fallbacks_profile",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("profile_id", "fallback_profile_id"),
        sa.UniqueConstraint(
            "profile_id", "priority", name="uq_model_profile_fallbacks_profile_priority"
        ),
    )

    op.create_table(
        "sparse_profiles",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("name", sa.String(length=120), nullable=False),
        sa.Column("algorithm", sa.String(length=64), nullable=False),
        sa.Column("encoder_package", sa.String(length=255), nullable=False),
        sa.Column("encoder_version", sa.String(length=64), nullable=False),
        sa.Column("tokenizer_name", sa.String(length=120), nullable=False),
        sa.Column("tokenizer_version", sa.String(length=64), nullable=False),
        sa.Column(
            "language_config",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "term_frequency_config",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "length_normalization_config",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column("idf_object_key", sa.String(length=1024), nullable=False),
        sa.Column("idf_checksum_sha256", sa.CHAR(length=64), nullable=False),
        sa.Column(
            "oov_config",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column("config_hash", sa.CHAR(length=64), nullable=False),
        sa.Column("enabled", sa.Boolean(), server_default=sa.text("true"), nullable=False),
        sa.Column(
            "created_at",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint("algorithm = 'qdrant_bm25_v1'", name="ck_sparse_profiles_algorithm"),
        sa.CheckConstraint("config_hash ~ '^[0-9a-f]{64}$'", name="ck_sparse_profiles_config_hash"),
        sa.CheckConstraint(
            "idf_checksum_sha256 ~ '^[0-9a-f]{64}$'", name="ck_sparse_profiles_idf_checksum"
        ),
        sa.CheckConstraint(
            "char_length(encoder_package) BETWEEN 1 AND 255",
            name="ck_sparse_profiles_encoder_package_length",
        ),
        sa.CheckConstraint(
            "char_length(encoder_version) BETWEEN 1 AND 64",
            name="ck_sparse_profiles_encoder_version_length",
        ),
        sa.CheckConstraint(
            "char_length(idf_object_key) BETWEEN 1 AND 1024",
            name="ck_sparse_profiles_idf_object_key_length",
        ),
        sa.CheckConstraint(
            "char_length(name) BETWEEN 1 AND 120", name="ck_sparse_profiles_name_length"
        ),
        sa.CheckConstraint(
            "char_length(tokenizer_name) BETWEEN 1 AND 120",
            name="ck_sparse_profiles_tokenizer_name_length",
        ),
        sa.CheckConstraint(
            "char_length(tokenizer_version) BETWEEN 1 AND 64",
            name="ck_sparse_profiles_tokenizer_version_length",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("config_hash", name="uq_sparse_profiles_config_hash"),
        sa.UniqueConstraint("name", name="uq_sparse_profiles_name"),
    )

    op.create_table(
        "query_profiles",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("name", sa.String(length=120), nullable=False),
        sa.Column("rerank_profile_id", sa.UUID(), nullable=True),
        sa.Column("chat_profile_id", sa.UUID(), nullable=True),
        sa.Column("dense_candidate_limit", sa.Integer(), nullable=False),
        sa.Column("sparse_candidate_limit", sa.Integer(), nullable=False),
        sa.Column("rrf_candidate_limit", sa.Integer(), nullable=False),
        sa.Column("rerank_candidate_limit", sa.Integer(), nullable=False),
        sa.Column("top_k_limit", sa.Integer(), nullable=False),
        sa.Column("min_rerank_score", sa.Numeric(precision=12, scale=8), nullable=False),
        sa.Column("min_rrf_score_when_degraded", sa.Numeric(precision=12, scale=8), nullable=False),
        sa.Column("context_token_budget", sa.Integer(), nullable=False),
        sa.Column("enabled", sa.Boolean(), server_default=sa.text("true"), nullable=False),
        sa.Column(
            "is_system_default", sa.Boolean(), server_default=sa.text("false"), nullable=False
        ),
        sa.Column(
            "created_at",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "char_length(name) BETWEEN 1 AND 120", name="ck_query_profiles_name_length"
        ),
        sa.CheckConstraint(
            "context_token_budget BETWEEN 1 AND 1000000", name="ck_query_profiles_context_budget"
        ),
        sa.CheckConstraint(
            "dense_candidate_limit BETWEEN 1 AND 1000 AND sparse_candidate_limit BETWEEN 1 AND 1000 AND rrf_candidate_limit BETWEEN 1 AND 1000 AND rerank_candidate_limit BETWEEN 1 AND 1000 AND rerank_candidate_limit <= rrf_candidate_limit",
            name="ck_query_profiles_candidate_limits",
        ),
        sa.CheckConstraint(
            "min_rerank_score BETWEEN -1 AND 1", name="ck_query_profiles_min_rerank_score"
        ),
        sa.CheckConstraint(
            "min_rrf_score_when_degraded BETWEEN 0 AND 1", name="ck_query_profiles_min_rrf_degraded"
        ),
        sa.CheckConstraint(
            "top_k_limit BETWEEN 1 AND 100 AND top_k_limit <= rerank_candidate_limit",
            name="ck_query_profiles_top_k_limit",
        ),
        sa.ForeignKeyConstraint(
            ["chat_profile_id"],
            ["model_profiles.id"],
            name="fk_query_profiles_chat_profile",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["rerank_profile_id"],
            ["model_profiles.id"],
            name="fk_query_profiles_rerank_profile",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name", name="uq_query_profiles_name"),
    )

    op.create_index(
        "uq_query_profiles_enabled_system_default",
        "query_profiles",
        ["is_system_default"],
        unique=True,
        postgresql_where=sa.text("enabled AND is_system_default"),
    )

    op.create_table(
        "knowledge_bases",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("name", sa.String(length=120), nullable=False),
        sa.Column("description", sa.String(length=2000), nullable=True),
        sa.Column(
            "status", sa.String(length=16), server_default=sa.text("'active'"), nullable=False
        ),
        sa.Column(
            "metadata",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "filter_schema",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{\"fields\": []}'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "resource_revision", sa.BigInteger(), server_default=sa.text("1"), nullable=False
        ),
        sa.Column(
            "mutation_revision", sa.BigInteger(), server_default=sa.text("0"), nullable=False
        ),
        sa.Column(
            "filter_schema_revision", sa.BigInteger(), server_default=sa.text("0"), nullable=False
        ),
        sa.Column("active_index_generation_id", sa.UUID(), nullable=True),
        sa.Column("pending_index_generation_id", sa.UUID(), nullable=True),
        sa.Column(
            "created_at",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "status IN ('active', 'reindexing', 'disabled', 'deleting')",
            name="ck_knowledge_bases_status",
        ),
        sa.CheckConstraint(
            "active_index_generation_id IS NULL OR pending_index_generation_id IS NULL OR active_index_generation_id <> pending_index_generation_id",
            name="ck_knowledge_bases_generation_pointers_distinct",
        ),
        sa.CheckConstraint(
            "char_length(name) BETWEEN 1 AND 120", name="ck_knowledge_bases_name_length"
        ),
        sa.CheckConstraint(
            "filter_schema_revision >= 0",
            name="ck_knowledge_bases_filter_schema_revision_nonnegative",
        ),
        sa.CheckConstraint(
            "mutation_revision >= 0", name="ck_knowledge_bases_mutation_revision_nonnegative"
        ),
        sa.CheckConstraint("resource_revision >= 1", name="ck_knowledge_bases_revision_positive"),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "knowledge_base_index_generations",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("knowledge_base_id", sa.UUID(), nullable=False),
        sa.Column("embedding_profile_id", sa.UUID(), nullable=True),
        sa.Column("sparse_profile_id", sa.UUID(), nullable=True),
        sa.Column("index_profile_hash", sa.CHAR(length=64), nullable=False),
        sa.Column("qdrant_collection_name", sa.String(length=255), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("rebuild_snapshot_at", postgresql.TIMESTAMP(timezone=True), nullable=False),
        sa.Column(
            "caught_up_revision", sa.BigInteger(), server_default=sa.text("0"), nullable=False
        ),
        sa.Column("validated_revision", sa.BigInteger(), nullable=True),
        sa.Column("validation_manifest_hash", sa.CHAR(length=64), nullable=True),
        sa.Column("expected_point_count", sa.BigInteger(), nullable=True),
        sa.Column("actual_point_count", sa.BigInteger(), nullable=True),
        sa.Column(
            "created_at",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("validated_at", postgresql.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("activated_at", postgresql.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("retired_at", postgresql.TIMESTAMP(timezone=True), nullable=True),
        sa.CheckConstraint(
            "index_profile_hash ~ '^[0-9a-f]{64}$'",
            name="ck_kb_index_generations_index_profile_hash",
        ),
        sa.CheckConstraint(
            "status IN ('building', 'active', 'retiring', 'failed')",
            name="ck_kb_index_generations_status",
        ),
        sa.CheckConstraint(
            "validation_manifest_hash IS NULL OR validation_manifest_hash ~ '^[0-9a-f]{64}$'",
            name="ck_kb_index_generations_validation_manifest_hash",
        ),
        sa.CheckConstraint(
            "actual_point_count IS NULL OR actual_point_count >= 0",
            name="ck_kb_index_generations_actual_count_nonnegative",
        ),
        sa.CheckConstraint(
            "caught_up_revision >= 0", name="ck_kb_index_generations_caught_up_revision_nonnegative"
        ),
        sa.CheckConstraint(
            "expected_point_count IS NULL OR expected_point_count >= 0",
            name="ck_kb_index_generations_expected_count_nonnegative",
        ),
        sa.CheckConstraint(
            "validated_revision IS NULL OR validated_revision >= 0",
            name="ck_kb_index_generations_validated_revision_nonnegative",
        ),
        sa.ForeignKeyConstraint(
            ["embedding_profile_id"],
            ["model_profiles.id"],
            name="fk_kb_index_generations_embedding_profile",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["knowledge_base_id"],
            ["knowledge_bases.id"],
            name="fk_kb_index_generations_knowledge_base",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["sparse_profile_id"],
            ["sparse_profiles.id"],
            name="fk_kb_index_generations_sparse_profile",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "id", "knowledge_base_id", name="uq_kb_index_generations_id_knowledge_base"
        ),
        sa.UniqueConstraint(
            "qdrant_collection_name", name="uq_kb_index_generations_qdrant_collection"
        ),
    )

    op.create_index(
        "uq_kb_index_generations_one_active",
        "knowledge_base_index_generations",
        ["knowledge_base_id"],
        unique=True,
        postgresql_where=sa.text("status = 'active'"),
    )

    op.create_index(
        "uq_kb_index_generations_one_building",
        "knowledge_base_index_generations",
        ["knowledge_base_id"],
        unique=True,
        postgresql_where=sa.text("status = 'building'"),
    )

    op.create_table(
        "knowledge_base_mutations",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("knowledge_base_id", sa.UUID(), nullable=False),
        sa.Column("revision", sa.BigInteger(), nullable=False),
        sa.Column("mutation_type", sa.String(length=40), nullable=False),
        sa.Column("target_type", sa.String(length=32), nullable=False),
        sa.Column("target_id", sa.UUID(), nullable=False),
        sa.Column(
            "payload",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "mutation_type IN ('document_version_created', 'document_activated', 'document_deleted', 'metadata_changed', 'filter_schema_changed', 'index_config_changed')",
            name="ck_knowledge_base_mutations_mutation_type",
        ),
        sa.CheckConstraint(
            "target_type IN ('knowledge_base', 'document', 'document_version', 'index_generation', 'filter_schema_revision')",
            name="ck_knowledge_base_mutations_target_type",
        ),
        sa.CheckConstraint("revision > 0", name="ck_knowledge_base_mutations_revision_positive"),
        sa.ForeignKeyConstraint(
            ["knowledge_base_id"],
            ["knowledge_bases.id"],
            name="fk_knowledge_base_mutations_knowledge_base",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "knowledge_base_id", "revision", name="uq_knowledge_base_mutations_kb_revision"
        ),
    )

    op.create_table(
        "documents",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("knowledge_base_id", sa.UUID(), nullable=False),
        sa.Column("display_name", sa.String(length=255), nullable=False),
        sa.Column("mime_type", sa.String(length=255), nullable=True),
        sa.Column("checksum_sha256", sa.CHAR(length=64), nullable=True),
        sa.Column("current_version_id", sa.UUID(), nullable=True),
        sa.Column("pending_version_id", sa.UUID(), nullable=True),
        sa.Column(
            "status", sa.String(length=16), server_default=sa.text("'processing'"), nullable=False
        ),
        sa.Column(
            "tags",
            postgresql.ARRAY(sa.Text()),
            server_default=sa.text("'{}'::text[]"),
            nullable=False,
        ),
        sa.Column(
            "metadata",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("deleted_at", postgresql.TIMESTAMP(timezone=True), nullable=True),
        sa.CheckConstraint(
            "checksum_sha256 IS NULL OR checksum_sha256 ~ '^[0-9a-f]{64}$'",
            name="ck_documents_checksum",
        ),
        sa.CheckConstraint(
            "status IN ('processing', 'active', 'failed', 'deleting', 'deleted')",
            name="ck_documents_status",
        ),
        sa.CheckConstraint(
            "cardinality(tags) <= 64 AND array_position(tags, NULL) IS NULL",
            name="ck_documents_tags",
        ),
        sa.CheckConstraint(
            "char_length(display_name) BETWEEN 1 AND 255", name="ck_documents_display_name_length"
        ),
        sa.CheckConstraint(
            "current_version_id IS NULL OR pending_version_id IS NULL OR current_version_id <> pending_version_id",
            name="ck_documents_version_pointers_distinct",
        ),
        sa.CheckConstraint(
            "mime_type IS NULL OR char_length(mime_type) BETWEEN 1 AND 255",
            name="ck_documents_mime_type_length",
        ),
        sa.ForeignKeyConstraint(
            ["knowledge_base_id"],
            ["knowledge_bases.id"],
            name="fk_documents_knowledge_base",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("id", "knowledge_base_id", name="uq_documents_id_knowledge_base"),
    )

    op.create_index(
        "uq_documents_kb_checksum_live",
        "documents",
        ["knowledge_base_id", "checksum_sha256"],
        unique=True,
        postgresql_where=sa.text("deleted_at IS NULL AND checksum_sha256 IS NOT NULL"),
    )

    op.create_table(
        "document_versions",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("document_id", sa.UUID(), nullable=False),
        sa.Column("version_number", sa.Integer(), nullable=False),
        sa.Column("source_object_key", sa.String(length=1024), nullable=False),
        sa.Column("parsed_object_key", sa.String(length=1024), nullable=True),
        sa.Column("source_checksum_sha256", sa.CHAR(length=64), nullable=False),
        sa.Column("parsed_object_checksum_sha256", sa.CHAR(length=64), nullable=True),
        sa.Column("declared_mime_type", sa.String(length=255), nullable=True),
        sa.Column("detected_mime_type", sa.String(length=255), nullable=True),
        sa.Column("source_extension", sa.String(length=32), nullable=True),
        sa.Column("base_version_id", sa.UUID(), nullable=True),
        sa.Column("parser_name", sa.String(length=120), nullable=True),
        sa.Column("parser_version", sa.String(length=64), nullable=True),
        sa.Column(
            "parser_config",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column("chunker_name", sa.String(length=120), nullable=True),
        sa.Column("chunker_version", sa.String(length=64), nullable=True),
        sa.Column(
            "chunker_config",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column("chunk_count", sa.Integer(), nullable=True),
        sa.Column(
            "status", sa.String(length=24), server_default=sa.text("'uploaded'"), nullable=False
        ),
        sa.Column("activated_at", postgresql.TIMESTAMP(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "parsed_object_checksum_sha256 IS NULL OR parsed_object_checksum_sha256 ~ '^[0-9a-f]{64}$'",
            name="ck_document_versions_parsed_object_checksum",
        ),
        sa.CheckConstraint(
            "source_checksum_sha256 ~ '^[0-9a-f]{64}$'", name="ck_document_versions_source_checksum"
        ),
        sa.CheckConstraint(
            "status IN ('uploaded', 'parsing', 'chunking', 'embedding', 'indexing', 'ready', 'failed', 'conflicted', 'cancelled', 'ocr_required', 'superseded')",
            name="ck_document_versions_status",
        ),
        sa.CheckConstraint(
            "char_length(source_object_key) BETWEEN 1 AND 1024",
            name="ck_document_versions_source_object_key_length",
        ),
        sa.CheckConstraint(
            "chunk_count IS NULL OR chunk_count >= 0",
            name="ck_document_versions_chunk_count_nonnegative",
        ),
        sa.CheckConstraint(
            "chunker_name IS NULL OR char_length(chunker_name) BETWEEN 1 AND 120",
            name="ck_document_versions_chunker_name_length",
        ),
        sa.CheckConstraint(
            "chunker_version IS NULL OR char_length(chunker_version) BETWEEN 1 AND 64",
            name="ck_document_versions_chunker_version_length",
        ),
        sa.CheckConstraint(
            "declared_mime_type IS NULL OR char_length(declared_mime_type) BETWEEN 1 AND 255",
            name="ck_document_versions_declared_mime_length",
        ),
        sa.CheckConstraint(
            "detected_mime_type IS NULL OR char_length(detected_mime_type) BETWEEN 1 AND 255",
            name="ck_document_versions_detected_mime_length",
        ),
        sa.CheckConstraint(
            "parsed_object_key IS NULL OR char_length(parsed_object_key) BETWEEN 1 AND 1024",
            name="ck_document_versions_parsed_object_key_length",
        ),
        sa.CheckConstraint(
            "parser_name IS NULL OR char_length(parser_name) BETWEEN 1 AND 120",
            name="ck_document_versions_parser_name_length",
        ),
        sa.CheckConstraint(
            "parser_version IS NULL OR char_length(parser_version) BETWEEN 1 AND 64",
            name="ck_document_versions_parser_version_length",
        ),
        sa.CheckConstraint(
            "source_extension IS NULL OR char_length(source_extension) BETWEEN 1 AND 32",
            name="ck_document_versions_source_extension_length",
        ),
        sa.CheckConstraint("version_number > 0", name="ck_document_versions_version_positive"),
        sa.ForeignKeyConstraint(
            ["base_version_id", "document_id"],
            ["document_versions.id", "document_versions.document_id"],
            name="fk_document_versions_base_same_document",
            ondelete="RESTRICT",
            initially="DEFERRED",
            deferrable=True,
        ),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["documents.id"],
            name="fk_document_versions_document",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "document_id", "version_number", name="uq_document_versions_document_number"
        ),
        sa.UniqueConstraint("id", "document_id", name="uq_document_versions_id_document"),
    )

    op.create_foreign_key(
        "fk_knowledge_bases_active_generation_same_parent",
        "knowledge_bases",
        "knowledge_base_index_generations",
        ["active_index_generation_id", "id"],
        ["id", "knowledge_base_id"],
        ondelete="RESTRICT",
        deferrable=True,
        initially="DEFERRED",
    )

    op.create_foreign_key(
        "fk_knowledge_bases_pending_generation_same_parent",
        "knowledge_bases",
        "knowledge_base_index_generations",
        ["pending_index_generation_id", "id"],
        ["id", "knowledge_base_id"],
        ondelete="RESTRICT",
        deferrable=True,
        initially="DEFERRED",
    )

    op.create_foreign_key(
        "fk_documents_current_version_same_parent",
        "documents",
        "document_versions",
        ["current_version_id", "id"],
        ["id", "document_id"],
        ondelete="RESTRICT",
        deferrable=True,
        initially="DEFERRED",
    )

    op.create_foreign_key(
        "fk_documents_pending_version_same_parent",
        "documents",
        "document_versions",
        ["pending_version_id", "id"],
        ["id", "document_id"],
        ondelete="RESTRICT",
        deferrable=True,
        initially="DEFERRED",
    )

    op.create_table(
        "document_index_states",
        sa.Column("document_version_id", sa.UUID(), nullable=False),
        sa.Column("index_generation_id", sa.UUID(), nullable=False),
        sa.Column(
            "status", sa.String(length=16), server_default=sa.text("'queued'"), nullable=False
        ),
        sa.Column("expected_point_count", sa.BigInteger(), nullable=True),
        sa.Column("actual_point_count", sa.BigInteger(), nullable=True),
        sa.Column("error_code", sa.String(length=64), nullable=True),
        sa.Column(
            "created_at",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("validated_at", postgresql.TIMESTAMP(timezone=True), nullable=True),
        sa.CheckConstraint(
            "status IN ('queued', 'embedding', 'indexing', 'validated', 'failed', 'retired')",
            name="ck_document_index_states_status",
        ),
        sa.CheckConstraint(
            "actual_point_count IS NULL OR actual_point_count >= 0",
            name="ck_document_index_states_actual_count",
        ),
        sa.CheckConstraint(
            "error_code IS NULL OR char_length(error_code) BETWEEN 1 AND 64",
            name="ck_document_index_states_error_code_length",
        ),
        sa.CheckConstraint(
            "expected_point_count IS NULL OR expected_point_count >= 0",
            name="ck_document_index_states_expected_count",
        ),
        sa.ForeignKeyConstraint(
            ["document_version_id"],
            ["document_versions.id"],
            name="fk_document_index_states_document_version",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["index_generation_id"],
            ["knowledge_base_index_generations.id"],
            name="fk_document_index_states_index_generation",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("document_version_id", "index_generation_id"),
    )

    op.create_table(
        "jobs",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("knowledge_base_id", sa.UUID(), nullable=True),
        sa.Column("target_type", sa.String(length=32), nullable=False),
        sa.Column("target_id", sa.UUID(), nullable=False),
        sa.Column("target_revision", sa.BigInteger(), nullable=True),
        sa.Column("index_generation_id", sa.UUID(), nullable=True),
        sa.Column("mutation_id", sa.UUID(), nullable=True),
        sa.Column("parent_job_id", sa.UUID(), nullable=True),
        sa.Column("root_job_id", sa.UUID(), nullable=True),
        sa.Column("idempotency_key", sa.String(length=128), nullable=True),
        sa.Column("operation", sa.String(length=48), nullable=False),
        sa.Column("stage", sa.String(length=64), nullable=True),
        sa.Column(
            "status", sa.String(length=16), server_default=sa.text("'queued'"), nullable=False
        ),
        sa.Column("progress_current", sa.BigInteger(), server_default=sa.text("0"), nullable=False),
        sa.Column("progress_total", sa.BigInteger(), nullable=True),
        sa.Column("attempt_count", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("max_attempts", sa.Integer(), server_default=sa.text("5"), nullable=False),
        sa.Column("next_retry_at", postgresql.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("worker_heartbeat_at", postgresql.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("cancel_requested_at", postgresql.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("error_code", sa.String(length=64), nullable=True),
        sa.Column("error_message_sanitized", sa.String(length=500), nullable=True),
        sa.Column(
            "created_at",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("started_at", postgresql.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("finished_at", postgresql.TIMESTAMP(timezone=True), nullable=True),
        sa.CheckConstraint(
            "idempotency_key IS NULL OR (char_length(idempotency_key) BETWEEN 1 AND 128 AND idempotency_key ~ '^[!-~]+$')",
            name="ck_jobs_idempotency_key",
        ),
        sa.CheckConstraint(
            "operation IN ('ingest_document', 'index_document', 'delete_document', 'rebuild_generation', 'apply_filter_schema', 'cleanup_generation', 'cleanup_document_version')",
            name="ck_jobs_operation",
        ),
        sa.CheckConstraint(
            "status IN ('queued', 'running', 'retry_wait', 'succeeded', 'failed', 'cancelled')",
            name="ck_jobs_status",
        ),
        sa.CheckConstraint(
            "target_type IN ('document_version', 'index_generation', 'filter_schema_revision', 'knowledge_base')",
            name="ck_jobs_target_type",
        ),
        sa.CheckConstraint(
            "attempt_count >= 0 AND attempt_count <= max_attempts", name="ck_jobs_attempt_count"
        ),
        sa.CheckConstraint(
            "error_code IS NULL OR char_length(error_code) BETWEEN 1 AND 64",
            name="ck_jobs_error_code_length",
        ),
        sa.CheckConstraint(
            "error_message_sanitized IS NULL OR char_length(error_message_sanitized) BETWEEN 1 AND 500",
            name="ck_jobs_error_message_sanitized_length",
        ),
        sa.CheckConstraint("max_attempts BETWEEN 1 AND 100", name="ck_jobs_max_attempts"),
        sa.CheckConstraint(
            "parent_job_id IS NULL OR parent_job_id <> id", name="ck_jobs_parent_not_self"
        ),
        sa.CheckConstraint(
            "progress_current >= 0 AND (progress_total IS NULL OR (progress_total >= 0 AND progress_current <= progress_total))",
            name="ck_jobs_progress",
        ),
        sa.CheckConstraint(
            "root_job_id IS NULL OR root_job_id <> id", name="ck_jobs_root_not_self"
        ),
        sa.CheckConstraint(
            "stage IS NULL OR char_length(stage) BETWEEN 1 AND 64", name="ck_jobs_stage_length"
        ),
        sa.CheckConstraint(
            "target_revision IS NULL OR target_revision >= 0", name="ck_jobs_target_revision"
        ),
        sa.ForeignKeyConstraint(
            ["index_generation_id"],
            ["knowledge_base_index_generations.id"],
            name="fk_jobs_index_generation",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["knowledge_base_id"],
            ["knowledge_bases.id"],
            name="fk_jobs_knowledge_base",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["mutation_id"],
            ["knowledge_base_mutations.id"],
            name="fk_jobs_mutation",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["parent_job_id"], ["jobs.id"], name="fk_jobs_parent", ondelete="RESTRICT"
        ),
        sa.ForeignKeyConstraint(
            ["root_job_id"], ["jobs.id"], name="fk_jobs_root", ondelete="RESTRICT"
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_index(
        "uq_jobs_active_target",
        "jobs",
        ["operation", "target_type", "target_id", "target_revision", "index_generation_id"],
        unique=True,
        postgresql_where=sa.text("status IN ('queued', 'running', 'retry_wait')"),
        postgresql_nulls_not_distinct=True,
    )

    op.create_table(
        "api_keys",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("public_id", sa.String(length=64), nullable=False),
        sa.Column("secret_digest", sa.LargeBinary(length=32), nullable=False),
        sa.Column("key_type", sa.String(length=16), nullable=False),
        sa.Column("name", sa.String(length=120), nullable=False),
        sa.Column(
            "status", sa.String(length=16), server_default=sa.text("'active'"), nullable=False
        ),
        sa.Column(
            "capabilities",
            postgresql.ARRAY(sa.Text()),
            server_default=sa.text("'{}'::text[]"),
            nullable=False,
        ),
        sa.Column("raw_file_read", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("requests_per_minute", sa.Integer(), nullable=True),
        sa.Column("max_concurrency", sa.Integer(), nullable=True),
        sa.Column("not_before", postgresql.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("expires_at", postgresql.TIMESTAMP(timezone=True), nullable=True),
        sa.Column(
            "resource_revision", sa.BigInteger(), server_default=sa.text("1"), nullable=False
        ),
        sa.Column("created_by_api_key_id", sa.UUID(), nullable=True),
        sa.Column("revoked_by_api_key_id", sa.UUID(), nullable=True),
        sa.Column(
            "created_at",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("revoked_at", postgresql.TIMESTAMP(timezone=True), nullable=True),
        sa.CheckConstraint(
            "(status = 'revoked') = (revoked_at IS NOT NULL)", name="ck_api_keys_revocation_state"
        ),
        sa.CheckConstraint(
            "cardinality(capabilities) <= 4 AND array_position(capabilities, NULL) IS NULL AND capabilities <@ ARRAY['ingest', 'retrieve', 'answer', 'manage']::text[]",
            name="ck_api_keys_capabilities",
        ),
        sa.CheckConstraint(
            "key_type <> 'admin' OR (cardinality(capabilities) = 0 AND raw_file_read = false AND requests_per_minute IS NULL AND max_concurrency IS NULL)",
            name="ck_api_keys_admin_policy",
        ),
        sa.CheckConstraint(
            "key_type <> 'agent' OR (requests_per_minute IS NOT NULL AND max_concurrency IS NOT NULL AND requests_per_minute BETWEEN 1 AND 10000 AND max_concurrency BETWEEN 1 AND 1000)",
            name="ck_api_keys_policy_positive",
        ),
        sa.CheckConstraint("key_type IN ('admin', 'agent')", name="ck_api_keys_key_type"),
        sa.CheckConstraint(
            "status IN ('active', 'disabled', 'revoked')", name="ck_api_keys_status"
        ),
        sa.CheckConstraint("char_length(name) BETWEEN 1 AND 120", name="ck_api_keys_name_length"),
        sa.CheckConstraint(
            "char_length(public_id) BETWEEN 16 AND 64", name="ck_api_keys_public_id_length"
        ),
        sa.CheckConstraint(
            "not_before IS NULL OR expires_at IS NULL OR expires_at > not_before",
            name="ck_api_keys_validity_window",
        ),
        sa.CheckConstraint(
            "octet_length(secret_digest) = 32", name="ck_api_keys_secret_digest_length"
        ),
        sa.CheckConstraint("resource_revision >= 1", name="ck_api_keys_revision_positive"),
        sa.ForeignKeyConstraint(
            ["created_by_api_key_id"],
            ["api_keys.id"],
            name="fk_api_keys_created_by",
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["revoked_by_api_key_id"],
            ["api_keys.id"],
            name="fk_api_keys_revoked_by",
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("public_id", name="uq_api_keys_public_id"),
    )

    op.create_table(
        "api_key_knowledge_base_scopes",
        sa.Column("api_key_id", sa.UUID(), nullable=False),
        sa.Column("knowledge_base_id", sa.UUID(), nullable=False),
        sa.Column(
            "created_at",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["api_key_id"],
            ["api_keys.id"],
            name="fk_api_key_kb_scopes_api_key",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["knowledge_base_id"],
            ["knowledge_bases.id"],
            name="fk_api_key_kb_scopes_knowledge_base",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("api_key_id", "knowledge_base_id"),
    )

    op.create_table(
        "api_key_query_profile_scopes",
        sa.Column("api_key_id", sa.UUID(), nullable=False),
        sa.Column("query_profile_id", sa.UUID(), nullable=False),
        sa.Column("is_default", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column(
            "created_at",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["api_key_id"],
            ["api_keys.id"],
            name="fk_api_key_query_profile_scopes_api_key",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["query_profile_id"],
            ["query_profiles.id"],
            name="fk_api_key_query_profile_scopes_query_profile",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("api_key_id", "query_profile_id"),
    )

    op.create_index(
        "uq_api_key_query_profile_default",
        "api_key_query_profile_scopes",
        ["api_key_id"],
        unique=True,
        postgresql_where=sa.text("is_default"),
    )

    op.create_table(
        "audit_events",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("request_id", sa.String(length=128), nullable=False),
        sa.Column("actor_api_key_id", sa.UUID(), nullable=True),
        sa.Column("actor_kind", sa.String(length=16), nullable=False),
        sa.Column("action", sa.String(length=64), nullable=False),
        sa.Column("target_type", sa.String(length=64), nullable=False),
        sa.Column("target_id", sa.UUID(), nullable=True),
        sa.Column(
            "metadata",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "actor_kind IN ('admin_key', 'agent_key', 'local_cli', 'system')",
            name="ck_audit_events_actor_kind",
        ),
        sa.CheckConstraint(
            "char_length(action) BETWEEN 1 AND 64", name="ck_audit_events_action_length"
        ),
        sa.CheckConstraint(
            "char_length(request_id) BETWEEN 1 AND 128", name="ck_audit_events_request_id_length"
        ),
        sa.CheckConstraint(
            "char_length(target_type) BETWEEN 1 AND 64", name="ck_audit_events_target_type_length"
        ),
        sa.ForeignKeyConstraint(
            ["actor_api_key_id"],
            ["api_keys.id"],
            name="fk_audit_events_actor_api_key",
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "idempotency_records",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("actor_key_id", sa.UUID(), nullable=False),
        sa.Column("operation", sa.String(length=64), nullable=False),
        sa.Column("idempotency_key", sa.String(length=128), nullable=False),
        sa.Column("request_fingerprint", sa.LargeBinary(length=32), nullable=False),
        sa.Column("result_resource_type", sa.String(length=64), nullable=False),
        sa.Column("result_resource_id", sa.UUID(), nullable=False),
        sa.Column("http_status", sa.SmallInteger(), nullable=False),
        sa.Column(
            "created_at",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "char_length(idempotency_key) BETWEEN 1 AND 128 AND idempotency_key ~ '^[!-~]+$'",
            name="ck_idempotency_records_key_visible_ascii",
        ),
        sa.CheckConstraint(
            "char_length(operation) BETWEEN 1 AND 64",
            name="ck_idempotency_records_operation_length",
        ),
        sa.CheckConstraint(
            "char_length(result_resource_type) BETWEEN 1 AND 64",
            name="ck_idempotency_records_result_type_length",
        ),
        sa.CheckConstraint(
            "http_status BETWEEN 100 AND 599", name="ck_idempotency_records_http_status"
        ),
        sa.CheckConstraint(
            "octet_length(request_fingerprint) = 32",
            name="ck_idempotency_records_fingerprint_length",
        ),
        sa.ForeignKeyConstraint(
            ["actor_key_id"],
            ["api_keys.id"],
            name="fk_idempotency_records_actor_key",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "actor_key_id",
            "operation",
            "idempotency_key",
            name="uq_idempotency_actor_operation_key",
        ),
    )

    op.create_table(
        "query_logs",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("request_id", sa.String(length=128), nullable=False),
        sa.Column("actor_api_key_id", sa.UUID(), nullable=True),
        sa.Column("knowledge_base_ids", postgresql.ARRAY(sa.UUID()), nullable=False),
        sa.Column("query_profile_id", sa.UUID(), nullable=True),
        sa.Column("latency_ms", sa.BigInteger(), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("degraded", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column(
            "created_at",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "status IN ('succeeded', 'failed', 'rejected')", name="ck_query_logs_status"
        ),
        sa.CheckConstraint(
            "cardinality(knowledge_base_ids) BETWEEN 1 AND 64 AND array_position(knowledge_base_ids, NULL) IS NULL",
            name="ck_query_logs_kb_ids",
        ),
        sa.CheckConstraint(
            "char_length(request_id) BETWEEN 1 AND 128", name="ck_query_logs_request_id_length"
        ),
        sa.CheckConstraint("latency_ms >= 0", name="ck_query_logs_latency"),
        sa.ForeignKeyConstraint(
            ["actor_api_key_id"],
            ["api_keys.id"],
            name="fk_query_logs_actor_api_key",
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["query_profile_id"],
            ["query_profiles.id"],
            name="fk_query_logs_query_profile",
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "provider_usage",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("request_id", sa.String(length=128), nullable=False),
        sa.Column("actor_api_key_id", sa.UUID(), nullable=True),
        sa.Column("provider_config_id", sa.UUID(), nullable=True),
        sa.Column("model_profile_id", sa.UUID(), nullable=True),
        sa.Column("capability", sa.String(length=16), nullable=False),
        sa.Column("provider_identifier", sa.String(length=120), nullable=False),
        sa.Column("model_identifier", sa.String(length=255), nullable=False),
        sa.Column("route_identifier", sa.String(length=255), nullable=True),
        sa.Column("provider_request_id", sa.String(length=255), nullable=True),
        sa.Column("input_tokens", sa.BigInteger(), server_default=sa.text("0"), nullable=False),
        sa.Column("output_tokens", sa.BigInteger(), server_default=sa.text("0"), nullable=False),
        sa.Column("cost_micros", sa.BigInteger(), server_default=sa.text("0"), nullable=False),
        sa.Column("currency", sa.CHAR(length=3), nullable=False),
        sa.Column("latency_ms", sa.BigInteger(), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("error_code", sa.String(length=64), nullable=True),
        sa.Column("degraded", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column(
            "created_at",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "capability IN ('embedding', 'rerank', 'chat')", name="ck_provider_usage_capability"
        ),
        sa.CheckConstraint("currency ~ '^[A-Z]{3}$'", name="ck_provider_usage_currency"),
        sa.CheckConstraint(
            "status IN ('succeeded', 'failed', 'rate_limited', 'timeout', 'cancelled')",
            name="ck_provider_usage_status",
        ),
        sa.CheckConstraint(
            "char_length(model_identifier) BETWEEN 1 AND 255",
            name="ck_provider_usage_model_identifier_length",
        ),
        sa.CheckConstraint(
            "char_length(provider_identifier) BETWEEN 1 AND 120",
            name="ck_provider_usage_provider_identifier_length",
        ),
        sa.CheckConstraint(
            "char_length(request_id) BETWEEN 1 AND 128", name="ck_provider_usage_request_id_length"
        ),
        sa.CheckConstraint("cost_micros >= 0", name="ck_provider_usage_cost"),
        sa.CheckConstraint(
            "error_code IS NULL OR char_length(error_code) BETWEEN 1 AND 64",
            name="ck_provider_usage_error_code_length",
        ),
        sa.CheckConstraint(
            "input_tokens >= 0 AND output_tokens >= 0", name="ck_provider_usage_tokens"
        ),
        sa.CheckConstraint("latency_ms >= 0", name="ck_provider_usage_latency"),
        sa.CheckConstraint(
            "provider_request_id IS NULL OR char_length(provider_request_id) BETWEEN 1 AND 255",
            name="ck_provider_usage_provider_request_id_length",
        ),
        sa.CheckConstraint(
            "route_identifier IS NULL OR char_length(route_identifier) BETWEEN 1 AND 255",
            name="ck_provider_usage_route_identifier_length",
        ),
        sa.ForeignKeyConstraint(
            ["actor_api_key_id"],
            ["api_keys.id"],
            name="fk_provider_usage_actor_api_key",
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["model_profile_id"],
            ["model_profiles.id"],
            name="fk_provider_usage_model_profile",
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["provider_config_id"],
            ["provider_configs.id"],
            name="fk_provider_usage_provider_config",
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("query_logs")
    op.drop_index(
        "uq_jobs_active_target",
        table_name="jobs",
        postgresql_where=sa.text("status IN ('queued', 'running', 'retry_wait')"),
        postgresql_nulls_not_distinct=True,
    )
    op.drop_table("jobs")
    op.drop_table("document_index_states")
    op.drop_index(
        "uq_api_key_query_profile_default",
        table_name="api_key_query_profile_scopes",
        postgresql_where=sa.text("is_default"),
    )
    op.drop_table("api_key_query_profile_scopes")
    op.drop_index(
        "uq_query_profiles_enabled_system_default",
        table_name="query_profiles",
        postgresql_where=sa.text("enabled AND is_system_default"),
    )
    op.drop_table("query_profiles")
    op.drop_table("provider_usage")
    op.drop_table("model_profile_fallbacks")
    op.drop_constraint("fk_documents_pending_version_same_parent", "documents", type_="foreignkey")
    op.drop_constraint("fk_documents_current_version_same_parent", "documents", type_="foreignkey")
    op.drop_constraint(
        "fk_knowledge_bases_pending_generation_same_parent",
        "knowledge_bases",
        type_="foreignkey",
    )
    op.drop_constraint(
        "fk_knowledge_bases_active_generation_same_parent",
        "knowledge_bases",
        type_="foreignkey",
    )
    op.drop_index(
        "uq_kb_index_generations_one_building",
        table_name="knowledge_base_index_generations",
        postgresql_where=sa.text("status = 'building'"),
    )
    op.drop_index(
        "uq_kb_index_generations_one_active",
        table_name="knowledge_base_index_generations",
        postgresql_where=sa.text("status = 'active'"),
    )
    op.drop_table("knowledge_base_index_generations")
    op.drop_table("document_versions")
    op.drop_table("model_profiles")
    op.drop_table("knowledge_base_mutations")
    op.drop_table("idempotency_records")
    op.drop_index(
        "uq_documents_kb_checksum_live",
        table_name="documents",
        postgresql_where=sa.text("deleted_at IS NULL AND checksum_sha256 IS NOT NULL"),
    )
    op.drop_table("documents")
    op.drop_table("audit_events")
    op.drop_table("api_key_knowledge_base_scopes")
    op.drop_table("sparse_profiles")
    op.drop_table("provider_configs")
    op.drop_table("knowledge_bases")
    op.drop_table("api_keys")
