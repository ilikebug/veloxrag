"""Knowledge-base, index-generation, and mutation mappings."""

from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import (
    CHAR,
    BigInteger,
    CheckConstraint,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    LargeBinary,
    String,
    UniqueConstraint,
    and_,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP
from sqlalchemy.dialects.postgresql import UUID as PostgreSQLUUID
from sqlalchemy.orm import Mapped, foreign, mapped_column, relationship

from rag_service.db.base import Base, CreatedAt, UpdatedAt, UUIDPrimaryKey


def _empty_dict() -> dict[str, Any]:
    return {}


def _empty_filter_schema() -> dict[str, list[object]]:
    return {"fields": []}


class KnowledgeBase(Base):
    __tablename__ = "knowledge_bases"
    __table_args__ = (
        CheckConstraint(
            "char_length(name) BETWEEN 1 AND 120",
            name="ck_knowledge_bases_name_length",
        ),
        CheckConstraint(
            "status IN ('active', 'reindexing', 'disabled', 'deleting')",
            name="ck_knowledge_bases_status",
        ),
        CheckConstraint(
            "resource_revision >= 1",
            name="ck_knowledge_bases_revision_positive",
        ),
        CheckConstraint(
            "mutation_revision >= 0",
            name="ck_knowledge_bases_mutation_revision_nonnegative",
        ),
        CheckConstraint(
            "filter_schema_revision >= 0",
            name="ck_knowledge_bases_filter_schema_revision_nonnegative",
        ),
        CheckConstraint(
            "active_index_generation_id IS NULL "
            "OR pending_index_generation_id IS NULL "
            "OR active_index_generation_id <> pending_index_generation_id",
            name="ck_knowledge_bases_generation_pointers_distinct",
        ),
    )

    id: Mapped[UUIDPrimaryKey]
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    description: Mapped[str | None] = mapped_column(String(2000))
    status: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        default="active",
        server_default=text("'active'"),
    )
    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        nullable=False,
        default=_empty_dict,
        server_default=text("'{}'::jsonb"),
    )
    filter_schema: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=_empty_filter_schema,
        server_default=text("'{\"fields\": []}'::jsonb"),
    )
    resource_revision: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        default=1,
        server_default=text("1"),
    )
    mutation_revision: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        default=0,
        server_default=text("0"),
    )
    filter_schema_revision: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        default=0,
        server_default=text("0"),
    )
    active_index_generation_id: Mapped[UUID | None] = mapped_column(PostgreSQLUUID(as_uuid=True))
    pending_index_generation_id: Mapped[UUID | None] = mapped_column(PostgreSQLUUID(as_uuid=True))
    # Which reranker a search may use, chosen by an operator rather than named
    # per request: picking the model spends a provider's quota, while deciding
    # whether a given query is worth the extra latency belongs to the caller.
    # Unlike the embedding profile this is not frozen into a generation — the
    # reranker reads live configuration because it does not shape the index, so
    # changing it takes effect without a rebuild.
    rerank_profile_id: Mapped[UUID | None] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey(
            "model_profiles.id",
            name="fk_knowledge_bases_rerank_profile",
            ondelete="RESTRICT",
        ),
    )
    created_at: Mapped[CreatedAt]
    updated_at: Mapped[UpdatedAt]

    active_index_generation: Mapped["KnowledgeBaseIndexGeneration | None"] = relationship(
        primaryjoin=lambda: and_(
            foreign(KnowledgeBase.active_index_generation_id) == KnowledgeBaseIndexGeneration.id,
            KnowledgeBase.id == KnowledgeBaseIndexGeneration.knowledge_base_id,
        ),
        foreign_keys=lambda: [KnowledgeBase.active_index_generation_id],
        post_update=True,
    )
    pending_index_generation: Mapped["KnowledgeBaseIndexGeneration | None"] = relationship(
        primaryjoin=lambda: and_(
            foreign(KnowledgeBase.pending_index_generation_id) == KnowledgeBaseIndexGeneration.id,
            KnowledgeBase.id == KnowledgeBaseIndexGeneration.knowledge_base_id,
        ),
        foreign_keys=lambda: [KnowledgeBase.pending_index_generation_id],
        post_update=True,
    )
    index_generations: Mapped[list["KnowledgeBaseIndexGeneration"]] = relationship(
        back_populates="knowledge_base",
        foreign_keys=lambda: [KnowledgeBaseIndexGeneration.knowledge_base_id],
    )


class KnowledgeBaseIndexGeneration(Base):
    __tablename__ = "knowledge_base_index_generations"
    __table_args__ = (
        CheckConstraint(
            "index_profile_hash ~ '^[0-9a-f]{64}$'",
            name="ck_kb_index_generations_index_profile_hash",
        ),
        CheckConstraint(
            "status IN ('building', 'active', 'retiring', 'failed')",
            name="ck_kb_index_generations_status",
        ),
        CheckConstraint(
            "caught_up_revision >= 0",
            name="ck_kb_index_generations_caught_up_revision_nonnegative",
        ),
        CheckConstraint(
            "validated_revision IS NULL OR validated_revision >= 0",
            name="ck_kb_index_generations_validated_revision_nonnegative",
        ),
        CheckConstraint(
            "validation_manifest_hash IS NULL OR validation_manifest_hash ~ '^[0-9a-f]{64}$'",
            name="ck_kb_index_generations_validation_manifest_hash",
        ),
        CheckConstraint(
            "expected_point_count IS NULL OR expected_point_count >= 0",
            name="ck_kb_index_generations_expected_count_nonnegative",
        ),
        CheckConstraint(
            "actual_point_count IS NULL OR actual_point_count >= 0",
            name="ck_kb_index_generations_actual_count_nonnegative",
        ),
        CheckConstraint(
            "distance IS NULL OR distance IN ('cosine', 'dot', 'euclid', 'manhattan')",
            name="ck_kb_index_generations_distance",
        ),
        CheckConstraint(
            "embedding_config_snapshot IS NULL "
            "OR jsonb_typeof(embedding_config_snapshot) = 'object'",
            name="ck_kb_index_generations_embedding_config_snapshot_object",
        ),
        CheckConstraint(
            "filter_schema_snapshot IS NULL OR jsonb_typeof(filter_schema_snapshot) = 'object'",
            name="ck_kb_index_generations_filter_schema_snapshot_object",
        ),
        CheckConstraint(
            "applied_filter_schema_revision IS NULL OR applied_filter_schema_revision >= 0",
            name="ck_kb_index_generations_filter_schema_revision_nonnegative",
        ),
        CheckConstraint(
            "embedding_config_hash IS NULL OR embedding_config_hash ~ '^[0-9a-f]{64}$'",
            name="ck_kb_index_generations_embedding_config_hash",
        ),
        CheckConstraint(
            "safe_error_code IS NULL OR char_length(safe_error_code) BETWEEN 1 AND 64",
            name="ck_kb_index_generations_safe_error_code_length",
        ),
        CheckConstraint(
            "safe_error_message IS NULL OR char_length(safe_error_message) BETWEEN 1 AND 500",
            name="ck_kb_index_generations_safe_error_message_length",
        ),
        CheckConstraint(
            "status <> 'active' OR ("
            "distance IS NOT NULL AND embedding_config_snapshot IS NOT NULL "
            "AND filter_schema_snapshot IS NOT NULL "
            "AND applied_filter_schema_revision IS NOT NULL "
            "AND embedding_config_hash IS NOT NULL "
            "AND validated_revision IS NOT NULL "
            "AND validation_manifest_hash IS NOT NULL "
            "AND expected_point_count IS NOT NULL AND actual_point_count IS NOT NULL "
            "AND expected_point_count = actual_point_count "
            "AND validated_revision = caught_up_revision "
            "AND validated_at IS NOT NULL AND activated_at IS NOT NULL)",
            name="ck_kb_index_generations_active_validation_complete",
        ),
        UniqueConstraint(
            "id",
            "knowledge_base_id",
            name="uq_kb_index_generations_id_knowledge_base",
        ),
        Index(
            "uq_kb_index_generations_one_active",
            "knowledge_base_id",
            unique=True,
            postgresql_where=text("status = 'active'"),
        ),
        Index(
            "uq_kb_index_generations_one_building",
            "knowledge_base_id",
            unique=True,
            postgresql_where=text("status = 'building'"),
        ),
    )

    id: Mapped[UUIDPrimaryKey]
    knowledge_base_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey("knowledge_bases.id", ondelete="RESTRICT"),
        nullable=False,
    )
    embedding_profile_id: Mapped[UUID | None] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey("model_profiles.id", ondelete="RESTRICT"),
    )
    sparse_profile_id: Mapped[UUID | None] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey("sparse_profiles.id", ondelete="RESTRICT"),
    )
    index_profile_hash: Mapped[str] = mapped_column(CHAR(64), nullable=False)
    qdrant_collection_name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    rebuild_snapshot_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False)
    caught_up_revision: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        default=0,
        server_default=text("0"),
    )
    validated_revision: Mapped[int | None] = mapped_column(BigInteger)
    validation_manifest_hash: Mapped[str | None] = mapped_column(CHAR(64))
    expected_point_count: Mapped[int | None] = mapped_column(BigInteger)
    actual_point_count: Mapped[int | None] = mapped_column(BigInteger)
    created_at: Mapped[CreatedAt]
    validated_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True))
    activated_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True))
    retired_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True))
    distance: Mapped[str | None] = mapped_column(String(16))
    embedding_config_snapshot: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    filter_schema_snapshot: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    applied_filter_schema_revision: Mapped[int | None] = mapped_column(BigInteger)
    embedding_config_hash: Mapped[str | None] = mapped_column(CHAR(64))
    safe_error_code: Mapped[str | None] = mapped_column(String(64))
    safe_error_message: Mapped[str | None] = mapped_column(String(500))

    knowledge_base: Mapped[KnowledgeBase] = relationship(
        back_populates="index_generations",
        foreign_keys=[knowledge_base_id],
    )


class IndexGenerationCleanupClaim(Base):
    __tablename__ = "index_generation_cleanup_claims"
    __table_args__ = (
        CheckConstraint(
            "char_length(collection_name) BETWEEN 1 AND 255",
            name="ck_generation_cleanup_claims_collection_name_length",
        ),
        CheckConstraint(
            "collection_name = 'rag_kb_' || "
            "replace(knowledge_base_id::text, '-', '') || '_g_' || "
            "replace(generation_id::text, '-', '')",
            name="ck_generation_cleanup_claims_collection_identity",
        ),
        CheckConstraint(
            "lease_epoch >= 1",
            name="ck_generation_cleanup_claims_lease_epoch_positive",
        ),
        Index(
            "ix_generation_cleanup_claims_expired",
            "lease_expires_at",
            "collection_name",
            postgresql_where=text("completed_at IS NULL"),
        ),
    )

    collection_name: Mapped[str] = mapped_column(String(255), primary_key=True)
    # These identities intentionally have no foreign keys: a managed Qdrant
    # collection and its durable cleanup claim may outlive deleted database rows.
    knowledge_base_id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), nullable=False)
    generation_id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), nullable=False)
    lease_owner: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), nullable=False)
    lease_epoch: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        default=1,
        server_default=text("1"),
    )
    lease_expires_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False)
    # Completed rows are permanent tombstones: their collection names are never reused.
    completed_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True))
    created_at: Mapped[CreatedAt]
    updated_at: Mapped[UpdatedAt]


class IndexGenerationCreationRequest(Base):
    __tablename__ = "index_generation_creation_requests"
    __table_args__ = (
        CheckConstraint(
            "char_length(idempotency_key) BETWEEN 1 AND 128 AND idempotency_key ~ '^[!-~]+$'",
            name="ck_index_generation_requests_idempotency_key",
        ),
        CheckConstraint(
            "octet_length(request_fingerprint) = 32",
            name="ck_index_generation_requests_fingerprint_length",
        ),
        CheckConstraint(
            "state IN ('building', 'succeeded', 'failed')",
            name="ck_index_generation_requests_state",
        ),
        CheckConstraint(
            "(state = 'building' AND final_http_status IS NULL) OR "
            "(state IN ('succeeded', 'failed') AND final_http_status IS NOT NULL "
            "AND final_http_status BETWEEN 100 AND 599)",
            name="ck_index_generation_requests_terminal_http_status",
        ),
        CheckConstraint(
            "safe_result IS NULL OR jsonb_typeof(safe_result) = 'object'",
            name="ck_index_generation_requests_safe_result_object",
        ),
        CheckConstraint(
            "safe_error_code IS NULL OR char_length(safe_error_code) BETWEEN 1 AND 64",
            name="ck_index_generation_requests_safe_error_code_length",
        ),
        CheckConstraint(
            "safe_error_message IS NULL OR char_length(safe_error_message) BETWEEN 1 AND 500",
            name="ck_index_generation_requests_safe_error_message_length",
        ),
        UniqueConstraint(
            "actor_api_key_id",
            "knowledge_base_id",
            "idempotency_key",
            name="uq_index_generation_requests_actor_kb_key",
        ),
        ForeignKeyConstraint(
            ["generation_id", "knowledge_base_id"],
            [
                "knowledge_base_index_generations.id",
                "knowledge_base_index_generations.knowledge_base_id",
            ],
            name="fk_index_generation_requests_generation_same_kb",
            ondelete="RESTRICT",
            deferrable=True,
            initially="DEFERRED",
        ),
        Index("ix_index_generation_requests_generation_id", "generation_id"),
        Index(
            "ix_index_generation_requests_reconciliation",
            "state",
            "updated_at",
            postgresql_where=text("state = 'building'"),
        ),
    )

    id: Mapped[UUIDPrimaryKey]
    actor_api_key_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey("api_keys.id", name="fk_index_generation_requests_actor", ondelete="RESTRICT"),
        nullable=False,
    )
    knowledge_base_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey(
            "knowledge_bases.id",
            name="fk_index_generation_requests_knowledge_base",
            ondelete="RESTRICT",
        ),
        nullable=False,
    )
    idempotency_key: Mapped[str] = mapped_column(String(128), nullable=False)
    request_fingerprint: Mapped[bytes] = mapped_column(LargeBinary(32), nullable=False)
    generation_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        nullable=False,
    )
    state: Mapped[str] = mapped_column(
        String(16), nullable=False, default="building", server_default=text("'building'")
    )
    final_http_status: Mapped[int | None] = mapped_column(Integer)
    safe_result: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    safe_error_code: Mapped[str | None] = mapped_column(String(64))
    safe_error_message: Mapped[str | None] = mapped_column(String(500))
    created_at: Mapped[CreatedAt]
    updated_at: Mapped[UpdatedAt]


class KnowledgeBaseMutation(Base):
    __tablename__ = "knowledge_base_mutations"
    __table_args__ = (
        CheckConstraint(
            "revision > 0",
            name="ck_knowledge_base_mutations_revision_positive",
        ),
        CheckConstraint(
            "mutation_type IN ("
            "'document_version_created', 'document_activated', 'document_deleted', "
            "'metadata_changed', 'filter_schema_changed', 'index_config_changed')",
            name="ck_knowledge_base_mutations_mutation_type",
        ),
        CheckConstraint(
            "target_type IN ("
            "'knowledge_base', 'document', 'document_version', "
            "'index_generation', 'filter_schema_revision')",
            name="ck_knowledge_base_mutations_target_type",
        ),
        UniqueConstraint(
            "knowledge_base_id",
            "revision",
            name="uq_knowledge_base_mutations_kb_revision",
        ),
    )

    id: Mapped[UUIDPrimaryKey]
    knowledge_base_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey("knowledge_bases.id", ondelete="RESTRICT"),
        nullable=False,
    )
    revision: Mapped[int] = mapped_column(BigInteger, nullable=False)
    mutation_type: Mapped[str] = mapped_column(String(40), nullable=False)
    target_type: Mapped[str] = mapped_column(String(32), nullable=False)
    target_id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=_empty_dict,
        server_default=text("'{}'::jsonb"),
    )
    created_at: Mapped[CreatedAt]
