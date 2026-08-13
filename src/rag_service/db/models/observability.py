"""Content-free query and provider usage observability mappings."""

from uuid import UUID

from sqlalchemy import CHAR, BigInteger, Boolean, CheckConstraint, ForeignKey, String, text
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.dialects.postgresql import UUID as PostgreSQLUUID
from sqlalchemy.orm import Mapped, mapped_column

from rag_service.db.base import Base, CreatedAt, UUIDPrimaryKey


class QueryLog(Base):
    __tablename__ = "query_logs"
    __table_args__ = (
        CheckConstraint(
            "char_length(request_id) BETWEEN 1 AND 128",
            name="ck_query_logs_request_id_length",
        ),
        CheckConstraint(
            "cardinality(knowledge_base_ids) BETWEEN 1 AND 64 "
            "AND array_position(knowledge_base_ids, NULL) IS NULL",
            name="ck_query_logs_kb_ids",
        ),
        CheckConstraint(
            "latency_ms >= 0",
            name="ck_query_logs_latency",
        ),
        CheckConstraint(
            "status IN ('succeeded', 'failed', 'rejected')",
            name="ck_query_logs_status",
        ),
    )

    id: Mapped[UUIDPrimaryKey]
    request_id: Mapped[str] = mapped_column(String(128), nullable=False)
    actor_api_key_id: Mapped[UUID | None] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey(
            "api_keys.id",
            name="fk_query_logs_actor_api_key",
            ondelete="SET NULL",
        ),
    )
    knowledge_base_ids: Mapped[list[UUID]] = mapped_column(
        ARRAY(PostgreSQLUUID(as_uuid=True)),
        nullable=False,
    )
    query_profile_id: Mapped[UUID | None] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey(
            "query_profiles.id",
            name="fk_query_logs_query_profile",
            ondelete="SET NULL",
        ),
    )
    latency_ms: Mapped[int] = mapped_column(BigInteger, nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    degraded: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default=text("false"),
    )
    created_at: Mapped[CreatedAt]


class ProviderUsage(Base):
    __tablename__ = "provider_usage"
    __table_args__ = (
        CheckConstraint(
            "char_length(request_id) BETWEEN 1 AND 128",
            name="ck_provider_usage_request_id_length",
        ),
        CheckConstraint(
            "capability IN ('embedding', 'rerank', 'chat')",
            name="ck_provider_usage_capability",
        ),
        CheckConstraint(
            "char_length(provider_identifier) BETWEEN 1 AND 120",
            name="ck_provider_usage_provider_identifier_length",
        ),
        CheckConstraint(
            "char_length(model_identifier) BETWEEN 1 AND 255",
            name="ck_provider_usage_model_identifier_length",
        ),
        CheckConstraint(
            "route_identifier IS NULL OR char_length(route_identifier) BETWEEN 1 AND 255",
            name="ck_provider_usage_route_identifier_length",
        ),
        CheckConstraint(
            "provider_request_id IS NULL OR char_length(provider_request_id) BETWEEN 1 AND 255",
            name="ck_provider_usage_provider_request_id_length",
        ),
        CheckConstraint(
            "input_tokens >= 0 AND output_tokens >= 0",
            name="ck_provider_usage_tokens",
        ),
        CheckConstraint(
            "cost_micros >= 0",
            name="ck_provider_usage_cost",
        ),
        CheckConstraint(
            "currency ~ '^[A-Z]{3}$'",
            name="ck_provider_usage_currency",
        ),
        CheckConstraint(
            "latency_ms >= 0",
            name="ck_provider_usage_latency",
        ),
        CheckConstraint(
            "status IN ('succeeded', 'failed', 'rate_limited', 'timeout', 'cancelled')",
            name="ck_provider_usage_status",
        ),
        CheckConstraint(
            "error_code IS NULL OR char_length(error_code) BETWEEN 1 AND 64",
            name="ck_provider_usage_error_code_length",
        ),
    )

    id: Mapped[UUIDPrimaryKey]
    request_id: Mapped[str] = mapped_column(String(128), nullable=False)
    actor_api_key_id: Mapped[UUID | None] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey(
            "api_keys.id",
            name="fk_provider_usage_actor_api_key",
            ondelete="SET NULL",
        ),
    )
    provider_config_id: Mapped[UUID | None] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey(
            "provider_configs.id",
            name="fk_provider_usage_provider_config",
            ondelete="SET NULL",
        ),
    )
    model_profile_id: Mapped[UUID | None] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey(
            "model_profiles.id",
            name="fk_provider_usage_model_profile",
            ondelete="SET NULL",
        ),
    )
    capability: Mapped[str] = mapped_column(String(16), nullable=False)
    provider_identifier: Mapped[str] = mapped_column(String(120), nullable=False)
    model_identifier: Mapped[str] = mapped_column(String(255), nullable=False)
    route_identifier: Mapped[str | None] = mapped_column(String(255))
    provider_request_id: Mapped[str | None] = mapped_column(String(255))
    input_tokens: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        default=0,
        server_default=text("0"),
    )
    output_tokens: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        default=0,
        server_default=text("0"),
    )
    cost_micros: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        default=0,
        server_default=text("0"),
    )
    currency: Mapped[str] = mapped_column(CHAR(3), nullable=False)
    latency_ms: Mapped[int] = mapped_column(BigInteger, nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    error_code: Mapped[str | None] = mapped_column(String(64))
    degraded: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default=text("false"),
    )
    created_at: Mapped[CreatedAt]
