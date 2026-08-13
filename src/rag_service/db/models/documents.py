"""Document lifecycle, indexing state, and background job mappings."""

from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import (
    CHAR,
    BigInteger,
    Boolean,
    CheckConstraint,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
    and_,
    text,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, TIMESTAMP
from sqlalchemy.dialects.postgresql import UUID as PostgreSQLUUID
from sqlalchemy.orm import Mapped, foreign, mapped_column, relationship

from rag_service.db.base import Base, CreatedAt, UpdatedAt, UUIDPrimaryKey


def _empty_dict() -> dict[str, Any]:
    return {}


def _empty_string_list() -> list[str]:
    return []


class Document(Base):
    __tablename__ = "documents"
    __table_args__ = (
        CheckConstraint(
            "char_length(display_name) BETWEEN 1 AND 255",
            name="ck_documents_display_name_length",
        ),
        CheckConstraint(
            "mime_type IS NULL OR char_length(mime_type) BETWEEN 1 AND 255",
            name="ck_documents_mime_type_length",
        ),
        CheckConstraint(
            "checksum_sha256 IS NULL OR checksum_sha256 ~ '^[0-9a-f]{64}$'",
            name="ck_documents_checksum",
        ),
        CheckConstraint(
            "status IN ('processing', 'active', 'failed', 'deleting', 'deleted')",
            name="ck_documents_status",
        ),
        CheckConstraint(
            "cardinality(tags) <= 64 AND array_position(tags, NULL) IS NULL",
            name="ck_documents_tags",
        ),
        CheckConstraint(
            "current_version_id IS NULL OR pending_version_id IS NULL "
            "OR current_version_id <> pending_version_id",
            name="ck_documents_version_pointers_distinct",
        ),
        UniqueConstraint(
            "id",
            "knowledge_base_id",
            name="uq_documents_id_knowledge_base",
        ),
        ForeignKeyConstraint(
            ["current_version_id", "id"],
            ["document_versions.id", "document_versions.document_id"],
            name="fk_documents_current_version_same_parent",
            ondelete="RESTRICT",
            deferrable=True,
            initially="DEFERRED",
            use_alter=True,
        ),
        ForeignKeyConstraint(
            ["pending_version_id", "id"],
            ["document_versions.id", "document_versions.document_id"],
            name="fk_documents_pending_version_same_parent",
            ondelete="RESTRICT",
            deferrable=True,
            initially="DEFERRED",
            use_alter=True,
        ),
        Index(
            "uq_documents_kb_checksum_live",
            "knowledge_base_id",
            "checksum_sha256",
            unique=True,
            postgresql_where=text("deleted_at IS NULL AND checksum_sha256 IS NOT NULL"),
        ),
    )

    id: Mapped[UUIDPrimaryKey]
    knowledge_base_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey(
            "knowledge_bases.id",
            name="fk_documents_knowledge_base",
            ondelete="RESTRICT",
        ),
        nullable=False,
    )
    display_name: Mapped[str] = mapped_column(String(255), nullable=False)
    mime_type: Mapped[str | None] = mapped_column(String(255))
    checksum_sha256: Mapped[str | None] = mapped_column(CHAR(64))
    current_version_id: Mapped[UUID | None] = mapped_column(PostgreSQLUUID(as_uuid=True))
    pending_version_id: Mapped[UUID | None] = mapped_column(PostgreSQLUUID(as_uuid=True))
    status: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        default="processing",
        server_default=text("'processing'"),
    )
    tags: Mapped[list[str]] = mapped_column(
        ARRAY(Text()),
        nullable=False,
        default=_empty_string_list,
        server_default=text("'{}'::text[]"),
    )
    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        nullable=False,
        default=_empty_dict,
        server_default=text("'{}'::jsonb"),
    )
    created_at: Mapped[CreatedAt]
    updated_at: Mapped[UpdatedAt]
    deleted_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True))

    current_version: Mapped["DocumentVersion | None"] = relationship(
        primaryjoin=lambda: and_(
            foreign(Document.current_version_id) == DocumentVersion.id,
            Document.id == DocumentVersion.document_id,
        ),
        foreign_keys=lambda: [Document.current_version_id],
        post_update=True,
    )
    pending_version: Mapped["DocumentVersion | None"] = relationship(
        primaryjoin=lambda: and_(
            foreign(Document.pending_version_id) == DocumentVersion.id,
            Document.id == DocumentVersion.document_id,
        ),
        foreign_keys=lambda: [Document.pending_version_id],
        post_update=True,
    )


class DocumentVersion(Base):
    __tablename__ = "document_versions"
    __table_args__ = (
        CheckConstraint(
            "version_number > 0",
            name="ck_document_versions_version_positive",
        ),
        CheckConstraint(
            "char_length(source_object_key) BETWEEN 1 AND 1024",
            name="ck_document_versions_source_object_key_length",
        ),
        CheckConstraint(
            "parsed_object_key IS NULL OR char_length(parsed_object_key) BETWEEN 1 AND 1024",
            name="ck_document_versions_parsed_object_key_length",
        ),
        CheckConstraint(
            "source_checksum_sha256 ~ '^[0-9a-f]{64}$'",
            name="ck_document_versions_source_checksum",
        ),
        CheckConstraint(
            "parsed_object_checksum_sha256 IS NULL "
            "OR parsed_object_checksum_sha256 ~ '^[0-9a-f]{64}$'",
            name="ck_document_versions_parsed_object_checksum",
        ),
        CheckConstraint(
            "declared_mime_type IS NULL OR char_length(declared_mime_type) BETWEEN 1 AND 255",
            name="ck_document_versions_declared_mime_length",
        ),
        CheckConstraint(
            "detected_mime_type IS NULL OR char_length(detected_mime_type) BETWEEN 1 AND 255",
            name="ck_document_versions_detected_mime_length",
        ),
        CheckConstraint(
            "source_extension IS NULL OR char_length(source_extension) BETWEEN 1 AND 32",
            name="ck_document_versions_source_extension_length",
        ),
        CheckConstraint(
            "parser_name IS NULL OR char_length(parser_name) BETWEEN 1 AND 120",
            name="ck_document_versions_parser_name_length",
        ),
        CheckConstraint(
            "parser_version IS NULL OR char_length(parser_version) BETWEEN 1 AND 64",
            name="ck_document_versions_parser_version_length",
        ),
        CheckConstraint(
            "chunker_name IS NULL OR char_length(chunker_name) BETWEEN 1 AND 120",
            name="ck_document_versions_chunker_name_length",
        ),
        CheckConstraint(
            "chunker_version IS NULL OR char_length(chunker_version) BETWEEN 1 AND 64",
            name="ck_document_versions_chunker_version_length",
        ),
        CheckConstraint(
            "chunk_count IS NULL OR chunk_count >= 0",
            name="ck_document_versions_chunk_count_nonnegative",
        ),
        CheckConstraint(
            "chunk_manifest_object_key IS NULL "
            "OR char_length(chunk_manifest_object_key) BETWEEN 1 AND 1024",
            name="ck_document_versions_chunk_manifest_object_key_length",
        ),
        CheckConstraint(
            "chunk_manifest_checksum_sha256 IS NULL "
            "OR chunk_manifest_checksum_sha256 ~ '^[0-9a-f]{64}$'",
            name="ck_document_versions_chunk_manifest_checksum",
        ),
        CheckConstraint(
            "chunk_config_hash IS NULL OR chunk_config_hash ~ '^[0-9a-f]{64}$'",
            name="ck_document_versions_chunk_config_hash",
        ),
        CheckConstraint(
            "num_nonnulls(chunk_manifest_object_key, chunk_manifest_checksum_sha256, "
            "chunk_config_hash) IN (0, 3)",
            name="ck_document_versions_chunk_manifest_complete",
        ),
        CheckConstraint(
            "status IN ('uploaded', 'parsing', 'chunking', 'embedding', 'indexing', "
            "'ready', 'failed', 'conflicted', 'cancelled', 'ocr_required', 'superseded')",
            name="ck_document_versions_status",
        ),
        UniqueConstraint(
            "id",
            "document_id",
            name="uq_document_versions_id_document",
        ),
        UniqueConstraint(
            "document_id",
            "version_number",
            name="uq_document_versions_document_number",
        ),
        ForeignKeyConstraint(
            ["base_version_id", "document_id"],
            ["document_versions.id", "document_versions.document_id"],
            name="fk_document_versions_base_same_document",
            ondelete="RESTRICT",
            deferrable=True,
            initially="DEFERRED",
        ),
    )

    id: Mapped[UUIDPrimaryKey]
    document_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey(
            "documents.id",
            name="fk_document_versions_document",
            ondelete="RESTRICT",
        ),
        nullable=False,
    )
    version_number: Mapped[int] = mapped_column(Integer, nullable=False)
    source_object_key: Mapped[str] = mapped_column(String(1024), nullable=False)
    parsed_object_key: Mapped[str | None] = mapped_column(String(1024))
    source_checksum_sha256: Mapped[str] = mapped_column(CHAR(64), nullable=False)
    parsed_object_checksum_sha256: Mapped[str | None] = mapped_column(CHAR(64))
    declared_mime_type: Mapped[str | None] = mapped_column(String(255))
    detected_mime_type: Mapped[str | None] = mapped_column(String(255))
    source_extension: Mapped[str | None] = mapped_column(String(32))
    base_version_id: Mapped[UUID | None] = mapped_column(PostgreSQLUUID(as_uuid=True))
    parser_name: Mapped[str | None] = mapped_column(String(120))
    parser_version: Mapped[str | None] = mapped_column(String(64))
    parser_config: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=_empty_dict,
        server_default=text("'{}'::jsonb"),
    )
    chunker_name: Mapped[str | None] = mapped_column(String(120))
    chunker_version: Mapped[str | None] = mapped_column(String(64))
    chunker_config: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=_empty_dict,
        server_default=text("'{}'::jsonb"),
    )
    chunk_count: Mapped[int | None] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(
        String(24),
        nullable=False,
        default="uploaded",
        server_default=text("'uploaded'"),
    )
    activated_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True))
    created_at: Mapped[CreatedAt]
    chunk_manifest_object_key: Mapped[str | None] = mapped_column(String(1024))
    chunk_manifest_checksum_sha256: Mapped[str | None] = mapped_column(CHAR(64))
    chunk_config_hash: Mapped[str | None] = mapped_column(CHAR(64))


class DocumentIndexState(Base):
    __tablename__ = "document_index_states"
    __table_args__ = (
        CheckConstraint(
            "status IN ('queued', 'embedding', 'indexing', 'validated', 'failed', 'retired')",
            name="ck_document_index_states_status",
        ),
        CheckConstraint(
            "expected_point_count IS NULL OR expected_point_count >= 0",
            name="ck_document_index_states_expected_count",
        ),
        CheckConstraint(
            "actual_point_count IS NULL OR actual_point_count >= 0",
            name="ck_document_index_states_actual_count",
        ),
        CheckConstraint(
            "error_code IS NULL OR char_length(error_code) BETWEEN 1 AND 64",
            name="ck_document_index_states_error_code_length",
        ),
        CheckConstraint(
            "chunk_manifest_checksum_sha256 IS NULL "
            "OR chunk_manifest_checksum_sha256 ~ '^[0-9a-f]{64}$'",
            name="ck_document_index_states_manifest_checksum",
        ),
        CheckConstraint(
            "embedding_config_hash IS NULL OR embedding_config_hash ~ '^[0-9a-f]{64}$'",
            name="ck_document_index_states_embedding_config_hash",
        ),
        CheckConstraint(
            "next_chunk_index >= 0",
            name="ck_document_index_states_next_chunk_index_nonnegative",
        ),
        CheckConstraint(
            "safe_error_message IS NULL OR char_length(safe_error_message) BETWEEN 1 AND 500",
            name="ck_document_index_states_safe_error_message_length",
        ),
    )

    document_version_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey(
            "document_versions.id",
            name="fk_document_index_states_document_version",
            ondelete="CASCADE",
        ),
        primary_key=True,
    )
    index_generation_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey(
            "knowledge_base_index_generations.id",
            name="fk_document_index_states_index_generation",
            ondelete="CASCADE",
        ),
        primary_key=True,
    )
    status: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        default="queued",
        server_default=text("'queued'"),
    )
    expected_point_count: Mapped[int | None] = mapped_column(BigInteger)
    actual_point_count: Mapped[int | None] = mapped_column(BigInteger)
    error_code: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[CreatedAt]
    validated_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True))
    chunk_manifest_checksum_sha256: Mapped[str | None] = mapped_column(CHAR(64))
    embedding_config_hash: Mapped[str | None] = mapped_column(CHAR(64))
    next_chunk_index: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        default=0,
        server_default=text("0"),
    )
    safe_error_message: Mapped[str | None] = mapped_column(String(500))


class Job(Base):
    __tablename__ = "jobs"
    __table_args__ = (
        CheckConstraint(
            "target_type IN ('document_version', 'index_generation', "
            "'filter_schema_revision', 'knowledge_base')",
            name="ck_jobs_target_type",
        ),
        CheckConstraint(
            "target_revision IS NULL OR target_revision >= 0",
            name="ck_jobs_target_revision",
        ),
        CheckConstraint(
            "parent_job_id IS NULL OR parent_job_id <> id",
            name="ck_jobs_parent_not_self",
        ),
        CheckConstraint(
            "root_job_id IS NULL OR root_job_id <> id",
            name="ck_jobs_root_not_self",
        ),
        CheckConstraint(
            "idempotency_key IS NULL OR (char_length(idempotency_key) BETWEEN 1 AND 128 "
            "AND idempotency_key ~ '^[!-~]+$')",
            name="ck_jobs_idempotency_key",
        ),
        CheckConstraint(
            "operation IN ('ingest_document', 'index_document', 'delete_document', "
            "'rebuild_generation', 'apply_filter_schema', 'cleanup_generation', "
            "'cleanup_document_version', 'purge_knowledge_base')",
            name="ck_jobs_operation",
        ),
        CheckConstraint(
            "stage IS NULL OR char_length(stage) BETWEEN 1 AND 64",
            name="ck_jobs_stage_length",
        ),
        CheckConstraint(
            "status IN ('queued', 'running', 'retry_wait', 'succeeded', 'failed', 'cancelled')",
            name="ck_jobs_status",
        ),
        CheckConstraint(
            "progress_current >= 0 AND (progress_total IS NULL OR "
            "(progress_total >= 0 AND progress_current <= progress_total))",
            name="ck_jobs_progress",
        ),
        CheckConstraint(
            "attempt_count >= 0 AND attempt_count <= max_attempts",
            name="ck_jobs_attempt_count",
        ),
        CheckConstraint(
            "max_attempts BETWEEN 1 AND 100",
            name="ck_jobs_max_attempts",
        ),
        CheckConstraint(
            "error_code IS NULL OR char_length(error_code) BETWEEN 1 AND 64",
            name="ck_jobs_error_code_length",
        ),
        CheckConstraint(
            "error_message_sanitized IS NULL "
            "OR char_length(error_message_sanitized) BETWEEN 1 AND 500",
            name="ck_jobs_error_message_sanitized_length",
        ),
        CheckConstraint(
            "lease_epoch >= 0",
            name="ck_jobs_lease_epoch_nonnegative",
        ),
        CheckConstraint(
            "lease_owner IS NULL OR char_length(lease_owner) BETWEEN 1 AND 255",
            name="ck_jobs_lease_owner_length",
        ),
        CheckConstraint(
            "resume_stage IS NULL OR char_length(resume_stage) BETWEEN 1 AND 64",
            name="ck_jobs_resume_stage_length",
        ),
        CheckConstraint(
            "(status = 'running' AND lease_owner IS NOT NULL "
            "AND lease_expires_at IS NOT NULL AND worker_heartbeat_at IS NOT NULL) OR "
            "(status <> 'running' AND lease_owner IS NULL AND lease_expires_at IS NULL)",
            name="ck_jobs_lease_state_invariant",
        ),
        Index(
            "uq_jobs_active_target",
            "operation",
            "target_type",
            "target_id",
            "target_revision",
            "index_generation_id",
            unique=True,
            postgresql_where=text("status IN ('queued', 'running', 'retry_wait')"),
            postgresql_nulls_not_distinct=True,
        ),
        Index(
            "ix_jobs_polling",
            "status",
            "next_retry_at",
            "created_at",
            postgresql_where=text("status IN ('queued', 'retry_wait')"),
        ),
        Index(
            "ix_jobs_expired_leases",
            "lease_expires_at",
            postgresql_where=text("status = 'running'"),
        ),
    )

    id: Mapped[UUIDPrimaryKey]
    knowledge_base_id: Mapped[UUID | None] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey(
            "knowledge_bases.id",
            name="fk_jobs_knowledge_base",
            ondelete="RESTRICT",
        ),
    )
    target_type: Mapped[str] = mapped_column(String(32), nullable=False)
    target_id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), nullable=False)
    target_revision: Mapped[int | None] = mapped_column(BigInteger)
    index_generation_id: Mapped[UUID | None] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey(
            "knowledge_base_index_generations.id",
            name="fk_jobs_index_generation",
            ondelete="RESTRICT",
        ),
    )
    mutation_id: Mapped[UUID | None] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey(
            "knowledge_base_mutations.id",
            name="fk_jobs_mutation",
            ondelete="RESTRICT",
        ),
    )
    parent_job_id: Mapped[UUID | None] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey("jobs.id", name="fk_jobs_parent", ondelete="RESTRICT"),
    )
    root_job_id: Mapped[UUID | None] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey("jobs.id", name="fk_jobs_root", ondelete="RESTRICT"),
    )
    idempotency_key: Mapped[str | None] = mapped_column(String(128))
    operation: Mapped[str] = mapped_column(String(48), nullable=False)
    stage: Mapped[str | None] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        default="queued",
        server_default=text("'queued'"),
    )
    progress_current: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        default=0,
        server_default=text("0"),
    )
    progress_total: Mapped[int | None] = mapped_column(BigInteger)
    attempt_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default=text("0"),
    )
    max_attempts: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=5,
        server_default=text("5"),
    )
    next_retry_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True))
    worker_heartbeat_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True))
    cancel_requested_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True))
    error_code: Mapped[str | None] = mapped_column(String(64))
    error_message_sanitized: Mapped[str | None] = mapped_column(String(500))
    created_at: Mapped[CreatedAt]
    updated_at: Mapped[UpdatedAt]
    started_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True))
    lease_owner: Mapped[str | None] = mapped_column(String(255))
    lease_epoch: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        default=0,
        server_default=text("0"),
    )
    lease_expires_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True))
    retryable: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        server_default=text("true"),
    )
    resume_stage: Mapped[str | None] = mapped_column(String(64))
    actor_api_key_id: Mapped[UUID | None] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey(
            "api_keys.id",
            name="fk_jobs_actor_api_key",
            ondelete="RESTRICT",
        ),
    )


class DocumentUploadIdempotency(Base):
    __tablename__ = "document_upload_idempotency"
    __table_args__ = (
        CheckConstraint(
            "char_length(idempotency_key) BETWEEN 1 AND 128 AND idempotency_key ~ '^[!-~]+$'",
            name="ck_document_upload_idempotency_key",
        ),
        CheckConstraint(
            "octet_length(request_fingerprint) = 32",
            name="ck_document_upload_fingerprint_length",
        ),
        CheckConstraint(
            "result_status IN ('pending', 'accepted', 'failed')",
            name="ck_document_upload_result_status",
        ),
        UniqueConstraint(
            "actor_api_key_id",
            "knowledge_base_id",
            "idempotency_key",
            name="uq_document_upload_idempotency_actor_kb_key",
        ),
        ForeignKeyConstraint(
            ["document_id", "knowledge_base_id"],
            ["documents.id", "documents.knowledge_base_id"],
            name="fk_document_upload_idempotency_document_same_kb",
            ondelete="RESTRICT",
            deferrable=True,
            initially="DEFERRED",
        ),
        ForeignKeyConstraint(
            ["document_version_id", "document_id"],
            ["document_versions.id", "document_versions.document_id"],
            name="fk_document_upload_idempotency_version_same_document",
            ondelete="RESTRICT",
            deferrable=True,
            initially="DEFERRED",
        ),
        ForeignKeyConstraint(
            ["job_id"],
            ["jobs.id"],
            name="fk_document_upload_idempotency_job",
            ondelete="RESTRICT",
            deferrable=True,
            initially="DEFERRED",
        ),
        Index("ix_document_upload_idempotency_document_id", "document_id"),
        Index(
            "ix_document_upload_idempotency_reconciliation",
            "result_status",
            "updated_at",
            postgresql_where=text("result_status = 'pending'"),
        ),
    )

    id: Mapped[UUIDPrimaryKey]
    actor_api_key_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey(
            "api_keys.id",
            name="fk_document_upload_idempotency_actor",
            ondelete="RESTRICT",
        ),
        nullable=False,
    )
    knowledge_base_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey(
            "knowledge_bases.id",
            name="fk_document_upload_idempotency_knowledge_base",
            ondelete="RESTRICT",
        ),
        nullable=False,
    )
    idempotency_key: Mapped[str] = mapped_column(String(128), nullable=False)
    request_fingerprint: Mapped[bytes] = mapped_column(LargeBinary(32), nullable=False)
    document_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        nullable=False,
    )
    document_version_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        nullable=False,
    )
    job_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        nullable=False,
    )
    result_status: Mapped[str] = mapped_column(
        String(16), nullable=False, default="pending", server_default=text("'pending'")
    )
    created_at: Mapped[CreatedAt]
    updated_at: Mapped[UpdatedAt]
