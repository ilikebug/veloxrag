"""Authentication, authorization-scope, audit, and idempotency mappings."""

from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, TIMESTAMP
from sqlalchemy.dialects.postgresql import UUID as PostgreSQLUUID
from sqlalchemy.orm import Mapped, mapped_column

from rag_service.db.base import Base, CreatedAt, UpdatedAt, UUIDPrimaryKey


def _empty_dict() -> dict[str, Any]:
    return {}


def _empty_string_list() -> list[str]:
    return []


class ApiKey(Base):
    __tablename__ = "api_keys"
    __table_args__ = (
        CheckConstraint(
            "char_length(public_id) BETWEEN 16 AND 64",
            name="ck_api_keys_public_id_length",
        ),
        CheckConstraint(
            "octet_length(secret_digest) = 32",
            name="ck_api_keys_secret_digest_length",
        ),
        CheckConstraint(
            "key_type IN ('admin', 'agent')",
            name="ck_api_keys_key_type",
        ),
        CheckConstraint(
            "char_length(name) BETWEEN 1 AND 120",
            name="ck_api_keys_name_length",
        ),
        CheckConstraint(
            "status IN ('active', 'disabled', 'revoked')",
            name="ck_api_keys_status",
        ),
        CheckConstraint(
            "cardinality(capabilities) <= 4 "
            "AND array_position(capabilities, NULL) IS NULL "
            "AND capabilities <@ ARRAY['ingest', 'retrieve', 'answer', 'manage']::text[]",
            name="ck_api_keys_capabilities",
        ),
        CheckConstraint(
            "not_before IS NULL OR expires_at IS NULL OR expires_at > not_before",
            name="ck_api_keys_validity_window",
        ),
        CheckConstraint(
            "key_type <> 'admin' OR "
            "(cardinality(capabilities) = 0 AND raw_file_read = false "
            "AND requests_per_minute IS NULL AND max_concurrency IS NULL)",
            name="ck_api_keys_admin_policy",
        ),
        CheckConstraint(
            "key_type <> 'agent' OR "
            "(requests_per_minute IS NOT NULL "
            "AND max_concurrency IS NOT NULL "
            "AND requests_per_minute BETWEEN 1 AND 10000 "
            "AND max_concurrency BETWEEN 1 AND 1000)",
            name="ck_api_keys_policy_positive",
        ),
        CheckConstraint(
            "resource_revision >= 1",
            name="ck_api_keys_revision_positive",
        ),
        CheckConstraint(
            "(status = 'revoked') = (revoked_at IS NOT NULL)",
            name="ck_api_keys_revocation_state",
        ),
    )

    id: Mapped[UUIDPrimaryKey]
    public_id: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    secret_digest: Mapped[bytes] = mapped_column(LargeBinary(32), nullable=False)
    key_type: Mapped[str] = mapped_column(String(16), nullable=False)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    status: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        default="active",
        server_default=text("'active'"),
    )
    capabilities: Mapped[list[str]] = mapped_column(
        ARRAY(Text()),
        nullable=False,
        default=_empty_string_list,
        server_default=text("'{}'::text[]"),
    )
    raw_file_read: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default=text("false"),
    )
    requests_per_minute: Mapped[int | None] = mapped_column(Integer)
    max_concurrency: Mapped[int | None] = mapped_column(Integer)
    not_before: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True))
    expires_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True))
    resource_revision: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        default=1,
        server_default=text("1"),
    )
    created_by_api_key_id: Mapped[UUID | None] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey("api_keys.id", ondelete="SET NULL"),
    )
    revoked_by_api_key_id: Mapped[UUID | None] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey("api_keys.id", ondelete="SET NULL"),
    )
    created_at: Mapped[CreatedAt]
    updated_at: Mapped[UpdatedAt]
    revoked_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True))


class ApiKeyKnowledgeBaseScope(Base):
    __tablename__ = "api_key_knowledge_base_scopes"

    api_key_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey("api_keys.id", ondelete="CASCADE"),
        primary_key=True,
    )
    knowledge_base_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey("knowledge_bases.id", ondelete="RESTRICT"),
        primary_key=True,
    )
    created_at: Mapped[CreatedAt]


class ApiKeyQueryProfileScope(Base):
    __tablename__ = "api_key_query_profile_scopes"
    __table_args__ = (
        Index(
            "uq_api_key_query_profile_default",
            "api_key_id",
            unique=True,
            postgresql_where=text("is_default"),
        ),
    )

    api_key_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey("api_keys.id", ondelete="CASCADE"),
        primary_key=True,
    )
    query_profile_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey("query_profiles.id", ondelete="RESTRICT"),
        primary_key=True,
    )
    is_default: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default=text("false"),
    )
    created_at: Mapped[CreatedAt]


class AuditEvent(Base):
    __tablename__ = "audit_events"
    __table_args__ = (
        CheckConstraint(
            "char_length(request_id) BETWEEN 1 AND 128",
            name="ck_audit_events_request_id_length",
        ),
        CheckConstraint(
            "actor_kind IN ('admin_key', 'agent_key', 'local_cli', 'system')",
            name="ck_audit_events_actor_kind",
        ),
        CheckConstraint(
            "char_length(action) BETWEEN 1 AND 64",
            name="ck_audit_events_action_length",
        ),
        CheckConstraint(
            "char_length(target_type) BETWEEN 1 AND 64",
            name="ck_audit_events_target_type_length",
        ),
    )

    id: Mapped[UUIDPrimaryKey]
    request_id: Mapped[str] = mapped_column(String(128), nullable=False)
    actor_api_key_id: Mapped[UUID | None] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey("api_keys.id", ondelete="SET NULL"),
    )
    actor_kind: Mapped[str] = mapped_column(String(16), nullable=False)
    action: Mapped[str] = mapped_column(String(64), nullable=False)
    target_type: Mapped[str] = mapped_column(String(64), nullable=False)
    target_id: Mapped[UUID | None] = mapped_column(PostgreSQLUUID(as_uuid=True))
    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        nullable=False,
        default=_empty_dict,
        server_default=text("'{}'::jsonb"),
    )
    created_at: Mapped[CreatedAt]


class IdempotencyRecord(Base):
    __tablename__ = "idempotency_records"
    __table_args__ = (
        CheckConstraint(
            "char_length(operation) BETWEEN 1 AND 64",
            name="ck_idempotency_records_operation_length",
        ),
        CheckConstraint(
            "char_length(idempotency_key) BETWEEN 1 AND 128 AND idempotency_key ~ '^[!-~]+$'",
            name="ck_idempotency_records_key_visible_ascii",
        ),
        CheckConstraint(
            "octet_length(request_fingerprint) = 32",
            name="ck_idempotency_records_fingerprint_length",
        ),
        CheckConstraint(
            "char_length(result_resource_type) BETWEEN 1 AND 64",
            name="ck_idempotency_records_result_type_length",
        ),
        CheckConstraint(
            "http_status BETWEEN 100 AND 599",
            name="ck_idempotency_records_http_status",
        ),
        UniqueConstraint(
            "actor_key_id",
            "operation",
            "idempotency_key",
            name="uq_idempotency_actor_operation_key",
        ),
    )

    id: Mapped[UUIDPrimaryKey]
    actor_key_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey("api_keys.id", ondelete="RESTRICT"),
        nullable=False,
    )
    operation: Mapped[str] = mapped_column(String(64), nullable=False)
    idempotency_key: Mapped[str] = mapped_column(String(128), nullable=False)
    request_fingerprint: Mapped[bytes] = mapped_column(LargeBinary(32), nullable=False)
    result_resource_type: Mapped[str] = mapped_column(String(64), nullable=False)
    result_resource_id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), nullable=False)
    http_status: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    created_at: Mapped[CreatedAt]
    updated_at: Mapped[UpdatedAt]
