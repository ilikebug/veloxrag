"""Add authoritative ingestion and retrieval persistence schema.

Revision ID: 20260726_0003
Revises: 20260724_0002
"""

from collections.abc import Sequence
from typing import Any

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260726_0003"
down_revision: str | Sequence[str] | None = "20260724_0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "provider_credentials",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("name", sa.String(length=120), nullable=False),
        sa.Column("ciphertext", sa.LargeBinary(), nullable=False),
        sa.Column("nonce", sa.LargeBinary(length=12), nullable=False),
        sa.Column(
            "algorithm",
            sa.String(length=32),
            server_default=sa.text("'AES-256-GCM'"),
            nullable=False,
        ),
        sa.Column("key_version", sa.String(length=64), nullable=False),
        sa.Column(
            "resource_revision", sa.BigInteger(), server_default=sa.text("1"), nullable=False
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
        sa.Column("rotated_at", postgresql.TIMESTAMP(timezone=True), nullable=True),
        sa.CheckConstraint(
            "char_length(name) BETWEEN 1 AND 120",
            name="ck_provider_credentials_name_length",
        ),
        sa.CheckConstraint("octet_length(nonce) = 12", name="ck_provider_credentials_nonce_length"),
        sa.CheckConstraint("algorithm = 'AES-256-GCM'", name="ck_provider_credentials_algorithm"),
        sa.CheckConstraint(
            "char_length(key_version) BETWEEN 1 AND 64",
            name="ck_provider_credentials_key_version_length",
        ),
        sa.CheckConstraint(
            "resource_revision >= 1", name="ck_provider_credentials_revision_positive"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name", name="uq_provider_credentials_name"),
    )

    op.add_column("provider_configs", sa.Column("credential_id", sa.UUID(), nullable=True))
    op.add_column(
        "provider_configs",
        sa.Column(
            "resource_revision", sa.BigInteger(), server_default=sa.text("1"), nullable=False
        ),
    )
    op.add_column("provider_configs", sa.Column("endpoint_policy_version", sa.String(length=64)))
    op.add_column(
        "provider_configs",
        sa.Column("endpoint_validated_at", postgresql.TIMESTAMP(timezone=True)),
    )
    op.alter_column("provider_configs", "secret_ref", existing_type=sa.String(255), nullable=True)
    op.create_foreign_key(
        "fk_provider_configs_credential",
        "provider_configs",
        "provider_credentials",
        ["credential_id"],
        ["id"],
        ondelete="RESTRICT",
    )
    op.create_check_constraint(
        "ck_provider_configs_credential_source_exactly_one",
        "provider_configs",
        "(secret_ref IS NULL) <> (credential_id IS NULL)",
    )
    op.create_check_constraint(
        "ck_provider_configs_credential_endpoint_validation",
        "provider_configs",
        "(credential_id IS NULL AND endpoint_policy_version IS NULL "
        "AND endpoint_validated_at IS NULL) OR "
        "(credential_id IS NOT NULL AND endpoint_policy_version IS NOT NULL "
        "AND endpoint_validated_at IS NOT NULL)",
    )
    op.create_check_constraint(
        "ck_provider_configs_endpoint_policy_version_length",
        "provider_configs",
        "endpoint_policy_version IS NULL OR char_length(endpoint_policy_version) BETWEEN 1 AND 64",
    )
    op.create_check_constraint(
        "ck_provider_configs_revision_positive",
        "provider_configs",
        "resource_revision >= 1",
    )
    op.create_index("ix_provider_configs_credential_id", "provider_configs", ["credential_id"])

    op.add_column(
        "model_profiles",
        sa.Column(
            "resource_revision", sa.BigInteger(), server_default=sa.text("1"), nullable=False
        ),
    )
    op.add_column(
        "model_profiles",
        sa.Column(
            "vector_config",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
    )
    op.create_check_constraint(
        "ck_model_profiles_revision_positive", "model_profiles", "resource_revision >= 1"
    )
    op.create_check_constraint(
        "ck_model_profiles_vector_config_object",
        "model_profiles",
        "jsonb_typeof(vector_config) = 'object'",
    )

    generation_columns = (
        sa.Column("distance", sa.String(length=16)),
        sa.Column("embedding_config_snapshot", postgresql.JSONB(astext_type=sa.Text())),
        sa.Column("filter_schema_snapshot", postgresql.JSONB(astext_type=sa.Text())),
        sa.Column("applied_filter_schema_revision", sa.BigInteger()),
        sa.Column("embedding_config_hash", sa.CHAR(length=64)),
        sa.Column("safe_error_code", sa.String(length=64)),
        sa.Column("safe_error_message", sa.String(length=500)),
    )
    for column in generation_columns:
        op.add_column("knowledge_base_index_generations", column)
    # Revision 0002 could mark a generation active without the immutable
    # configuration snapshots introduced here. Those facts cannot be inferred
    # safely, so fail closed: preserve the generation and its validation audit
    # columns, remove routable pointers, and require a fresh rebuild.
    op.execute(
        """
        UPDATE knowledge_bases
        SET active_index_generation_id = CASE
                WHEN active_index_generation_id IN (
                    SELECT id FROM knowledge_base_index_generations WHERE status = 'active'
                ) THEN NULL
                ELSE active_index_generation_id
            END,
            pending_index_generation_id = CASE
                WHEN pending_index_generation_id IN (
                    SELECT id FROM knowledge_base_index_generations WHERE status = 'active'
                ) THEN NULL
                ELSE pending_index_generation_id
            END
        WHERE active_index_generation_id IN (
                SELECT id FROM knowledge_base_index_generations WHERE status = 'active'
              )
           OR pending_index_generation_id IN (
                SELECT id FROM knowledge_base_index_generations WHERE status = 'active'
              )
        """
    )
    op.execute(
        """
        UPDATE knowledge_base_index_generations
        SET status = 'failed',
            safe_error_code = 'migration_revalidation_required',
            safe_error_message =
                'Generation deactivated during schema upgrade; rebuild required'
        WHERE status = 'active'
        """
    )
    generation_checks = (
        (
            "ck_kb_index_generations_distance",
            "distance IS NULL OR distance IN ('cosine', 'dot', 'euclid', 'manhattan')",
        ),
        (
            "ck_kb_index_generations_embedding_config_snapshot_object",
            "embedding_config_snapshot IS NULL "
            "OR jsonb_typeof(embedding_config_snapshot) = 'object'",
        ),
        (
            "ck_kb_index_generations_filter_schema_snapshot_object",
            "filter_schema_snapshot IS NULL OR jsonb_typeof(filter_schema_snapshot) = 'object'",
        ),
        (
            "ck_kb_index_generations_filter_schema_revision_nonnegative",
            "applied_filter_schema_revision IS NULL OR applied_filter_schema_revision >= 0",
        ),
        (
            "ck_kb_index_generations_embedding_config_hash",
            "embedding_config_hash IS NULL OR embedding_config_hash ~ '^[0-9a-f]{64}$'",
        ),
        (
            "ck_kb_index_generations_safe_error_code_length",
            "safe_error_code IS NULL OR char_length(safe_error_code) BETWEEN 1 AND 64",
        ),
        (
            "ck_kb_index_generations_safe_error_message_length",
            "safe_error_message IS NULL OR char_length(safe_error_message) BETWEEN 1 AND 500",
        ),
        (
            "ck_kb_index_generations_active_validation_complete",
            "status <> 'active' OR (distance IS NOT NULL "
            "AND embedding_config_snapshot IS NOT NULL "
            "AND filter_schema_snapshot IS NOT NULL "
            "AND applied_filter_schema_revision IS NOT NULL "
            "AND embedding_config_hash IS NOT NULL "
            "AND validated_revision IS NOT NULL "
            "AND validation_manifest_hash IS NOT NULL "
            "AND expected_point_count IS NOT NULL AND actual_point_count IS NOT NULL "
            "AND expected_point_count = actual_point_count "
            "AND validated_revision = caught_up_revision "
            "AND validated_at IS NOT NULL AND activated_at IS NOT NULL)",
        ),
    )
    for name, condition in generation_checks:
        op.create_check_constraint(name, "knowledge_base_index_generations", condition)

    op.create_table(
        "index_generation_creation_requests",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("actor_api_key_id", sa.UUID(), nullable=False),
        sa.Column("knowledge_base_id", sa.UUID(), nullable=False),
        sa.Column("idempotency_key", sa.String(length=128), nullable=False),
        sa.Column("request_fingerprint", sa.LargeBinary(length=32), nullable=False),
        sa.Column("generation_id", sa.UUID(), nullable=False),
        sa.Column(
            "state", sa.String(length=16), server_default=sa.text("'building'"), nullable=False
        ),
        sa.Column("final_http_status", sa.Integer()),
        sa.Column("safe_result", postgresql.JSONB(astext_type=sa.Text())),
        sa.Column("safe_error_code", sa.String(length=64)),
        sa.Column("safe_error_message", sa.String(length=500)),
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
            name="ck_index_generation_requests_idempotency_key",
        ),
        sa.CheckConstraint(
            "octet_length(request_fingerprint) = 32",
            name="ck_index_generation_requests_fingerprint_length",
        ),
        sa.CheckConstraint(
            "state IN ('building', 'succeeded', 'failed')",
            name="ck_index_generation_requests_state",
        ),
        sa.CheckConstraint(
            "(state = 'building' AND final_http_status IS NULL) OR "
            "(state IN ('succeeded', 'failed') AND final_http_status IS NOT NULL "
            "AND final_http_status BETWEEN 100 AND 599)",
            name="ck_index_generation_requests_terminal_http_status",
        ),
        sa.CheckConstraint(
            "safe_result IS NULL OR jsonb_typeof(safe_result) = 'object'",
            name="ck_index_generation_requests_safe_result_object",
        ),
        sa.CheckConstraint(
            "safe_error_code IS NULL OR char_length(safe_error_code) BETWEEN 1 AND 64",
            name="ck_index_generation_requests_safe_error_code_length",
        ),
        sa.CheckConstraint(
            "safe_error_message IS NULL OR char_length(safe_error_message) BETWEEN 1 AND 500",
            name="ck_index_generation_requests_safe_error_message_length",
        ),
        sa.ForeignKeyConstraint(
            ["actor_api_key_id"],
            ["api_keys.id"],
            name="fk_index_generation_requests_actor",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["knowledge_base_id"],
            ["knowledge_bases.id"],
            name="fk_index_generation_requests_knowledge_base",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
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
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "actor_api_key_id",
            "knowledge_base_id",
            "idempotency_key",
            name="uq_index_generation_requests_actor_kb_key",
        ),
    )
    op.create_index(
        "ix_index_generation_requests_generation_id",
        "index_generation_creation_requests",
        ["generation_id"],
    )
    op.create_index(
        "ix_index_generation_requests_reconciliation",
        "index_generation_creation_requests",
        ["state", "updated_at"],
        postgresql_where=sa.text("state = 'building'"),
    )

    document_version_columns = (
        sa.Column("chunk_manifest_object_key", sa.String(length=1024)),
        sa.Column("chunk_manifest_checksum_sha256", sa.CHAR(length=64)),
        sa.Column("chunk_config_hash", sa.CHAR(length=64)),
    )
    for column in document_version_columns:
        op.add_column("document_versions", column)
    document_version_checks = (
        (
            "ck_document_versions_chunk_manifest_object_key_length",
            "chunk_manifest_object_key IS NULL "
            "OR char_length(chunk_manifest_object_key) BETWEEN 1 AND 1024",
        ),
        (
            "ck_document_versions_chunk_manifest_checksum",
            "chunk_manifest_checksum_sha256 IS NULL "
            "OR chunk_manifest_checksum_sha256 ~ '^[0-9a-f]{64}$'",
        ),
        (
            "ck_document_versions_chunk_config_hash",
            "chunk_config_hash IS NULL OR chunk_config_hash ~ '^[0-9a-f]{64}$'",
        ),
        (
            "ck_document_versions_chunk_manifest_complete",
            "num_nonnulls(chunk_manifest_object_key, chunk_manifest_checksum_sha256, "
            "chunk_config_hash) IN (0, 3)",
        ),
    )
    for name, condition in document_version_checks:
        op.create_check_constraint(name, "document_versions", condition)

    document_state_columns = (
        sa.Column("chunk_manifest_checksum_sha256", sa.CHAR(length=64)),
        sa.Column("embedding_config_hash", sa.CHAR(length=64)),
        sa.Column("next_chunk_index", sa.BigInteger(), server_default=sa.text("0"), nullable=False),
        sa.Column("safe_error_message", sa.String(length=500)),
    )
    for column in document_state_columns:
        op.add_column("document_index_states", column)
    document_state_checks = (
        (
            "ck_document_index_states_manifest_checksum",
            "chunk_manifest_checksum_sha256 IS NULL "
            "OR chunk_manifest_checksum_sha256 ~ '^[0-9a-f]{64}$'",
        ),
        (
            "ck_document_index_states_embedding_config_hash",
            "embedding_config_hash IS NULL OR embedding_config_hash ~ '^[0-9a-f]{64}$'",
        ),
        (
            "ck_document_index_states_next_chunk_index_nonnegative",
            "next_chunk_index >= 0",
        ),
        (
            "ck_document_index_states_safe_error_message_length",
            "safe_error_message IS NULL OR char_length(safe_error_message) BETWEEN 1 AND 500",
        ),
    )
    for name, condition in document_state_checks:
        op.create_check_constraint(name, "document_index_states", condition)

    job_columns: tuple[sa.Column[Any], ...] = (
        sa.Column("lease_owner", sa.String(length=255)),
        sa.Column("lease_epoch", sa.BigInteger(), server_default=sa.text("0"), nullable=False),
        sa.Column("lease_expires_at", postgresql.TIMESTAMP(timezone=True)),
        sa.Column("retryable", sa.Boolean(), server_default=sa.text("true"), nullable=False),
        sa.Column("resume_stage", sa.String(length=64)),
    )
    for column in job_columns:
        op.add_column("jobs", column)
    # A 0002 running job has no owner/expiry fencing facts. Converting it to an
    # immediately pollable retry preserves progress, attempts, stage, and the
    # diagnostic heartbeat without pretending that its old worker still owns it.
    op.execute(
        """
        UPDATE jobs
        SET status = 'retry_wait',
            next_retry_at = COALESCE(next_retry_at, now()),
            resume_stage = COALESCE(resume_stage, stage)
        WHERE status = 'running'
        """
    )
    job_checks = (
        ("ck_jobs_lease_epoch_nonnegative", "lease_epoch >= 0"),
        (
            "ck_jobs_lease_owner_length",
            "lease_owner IS NULL OR char_length(lease_owner) BETWEEN 1 AND 255",
        ),
        (
            "ck_jobs_resume_stage_length",
            "resume_stage IS NULL OR char_length(resume_stage) BETWEEN 1 AND 64",
        ),
        (
            "ck_jobs_lease_state_invariant",
            "(status = 'running' AND lease_owner IS NOT NULL "
            "AND lease_expires_at IS NOT NULL AND worker_heartbeat_at IS NOT NULL) OR "
            "(status <> 'running' AND lease_owner IS NULL AND lease_expires_at IS NULL)",
        ),
    )
    for name, condition in job_checks:
        op.create_check_constraint(name, "jobs", condition)
    op.create_index(
        "ix_jobs_polling",
        "jobs",
        ["status", "next_retry_at", "created_at"],
        postgresql_where=sa.text("status IN ('queued', 'retry_wait')"),
    )
    op.create_index(
        "ix_jobs_expired_leases",
        "jobs",
        ["lease_expires_at"],
        postgresql_where=sa.text("status = 'running'"),
    )

    op.create_table(
        "document_upload_idempotency",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("actor_api_key_id", sa.UUID(), nullable=False),
        sa.Column("knowledge_base_id", sa.UUID(), nullable=False),
        sa.Column("idempotency_key", sa.String(length=128), nullable=False),
        sa.Column("request_fingerprint", sa.LargeBinary(length=32), nullable=False),
        sa.Column("document_id", sa.UUID(), nullable=False),
        sa.Column("document_version_id", sa.UUID(), nullable=False),
        sa.Column("job_id", sa.UUID(), nullable=False),
        sa.Column(
            "result_status",
            sa.String(length=16),
            server_default=sa.text("'pending'"),
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
        sa.CheckConstraint(
            "char_length(idempotency_key) BETWEEN 1 AND 128 AND idempotency_key ~ '^[!-~]+$'",
            name="ck_document_upload_idempotency_key",
        ),
        sa.CheckConstraint(
            "octet_length(request_fingerprint) = 32",
            name="ck_document_upload_fingerprint_length",
        ),
        sa.CheckConstraint(
            "result_status IN ('pending', 'accepted', 'failed')",
            name="ck_document_upload_result_status",
        ),
        sa.ForeignKeyConstraint(
            ["actor_api_key_id"],
            ["api_keys.id"],
            name="fk_document_upload_idempotency_actor",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["knowledge_base_id"],
            ["knowledge_bases.id"],
            name="fk_document_upload_idempotency_knowledge_base",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["document_id", "knowledge_base_id"],
            ["documents.id", "documents.knowledge_base_id"],
            name="fk_document_upload_idempotency_document_same_kb",
            ondelete="RESTRICT",
            deferrable=True,
            initially="DEFERRED",
        ),
        sa.ForeignKeyConstraint(
            ["document_version_id", "document_id"],
            ["document_versions.id", "document_versions.document_id"],
            name="fk_document_upload_idempotency_version_same_document",
            ondelete="RESTRICT",
            deferrable=True,
            initially="DEFERRED",
        ),
        sa.ForeignKeyConstraint(
            ["job_id"],
            ["jobs.id"],
            name="fk_document_upload_idempotency_job",
            ondelete="RESTRICT",
            deferrable=True,
            initially="DEFERRED",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "actor_api_key_id",
            "knowledge_base_id",
            "idempotency_key",
            name="uq_document_upload_idempotency_actor_kb_key",
        ),
    )
    op.create_index(
        "ix_document_upload_idempotency_document_id",
        "document_upload_idempotency",
        ["document_id"],
    )
    op.create_index(
        "ix_document_upload_idempotency_reconciliation",
        "document_upload_idempotency",
        ["result_status", "updated_at"],
        postgresql_where=sa.text("result_status = 'pending'"),
    )


def downgrade() -> None:
    op.execute("LOCK TABLE provider_configs, provider_credentials IN ACCESS EXCLUSIVE MODE")
    credential_count = (
        op.get_bind().execute(sa.text("SELECT count(*) FROM provider_credentials")).scalar_one()
    )
    if credential_count:
        raise RuntimeError(
            "credential-backed provider configs cannot be represented by revision "
            "20260724_0002; replace credential_id with secret_ref before retrying the "
            "downgrade, then explicitly remove unreferenced provider credentials"
        )

    op.drop_index(
        "ix_document_upload_idempotency_reconciliation",
        table_name="document_upload_idempotency",
    )
    op.drop_index(
        "ix_document_upload_idempotency_document_id",
        table_name="document_upload_idempotency",
    )
    op.drop_table("document_upload_idempotency")

    op.drop_index("ix_jobs_expired_leases", table_name="jobs")
    op.drop_index("ix_jobs_polling", table_name="jobs")
    for name in (
        "ck_jobs_lease_state_invariant",
        "ck_jobs_resume_stage_length",
        "ck_jobs_lease_owner_length",
        "ck_jobs_lease_epoch_nonnegative",
    ):
        op.drop_constraint(name, "jobs", type_="check")
    for column in ("resume_stage", "retryable", "lease_expires_at", "lease_epoch", "lease_owner"):
        op.drop_column("jobs", column)

    for name in (
        "ck_document_index_states_safe_error_message_length",
        "ck_document_index_states_next_chunk_index_nonnegative",
        "ck_document_index_states_embedding_config_hash",
        "ck_document_index_states_manifest_checksum",
    ):
        op.drop_constraint(name, "document_index_states", type_="check")
    for column in (
        "safe_error_message",
        "next_chunk_index",
        "embedding_config_hash",
        "chunk_manifest_checksum_sha256",
    ):
        op.drop_column("document_index_states", column)

    for name in (
        "ck_document_versions_chunk_manifest_complete",
        "ck_document_versions_chunk_config_hash",
        "ck_document_versions_chunk_manifest_checksum",
        "ck_document_versions_chunk_manifest_object_key_length",
    ):
        op.drop_constraint(name, "document_versions", type_="check")
    for column in (
        "chunk_config_hash",
        "chunk_manifest_checksum_sha256",
        "chunk_manifest_object_key",
    ):
        op.drop_column("document_versions", column)

    op.drop_index(
        "ix_index_generation_requests_reconciliation",
        table_name="index_generation_creation_requests",
    )
    op.drop_index(
        "ix_index_generation_requests_generation_id",
        table_name="index_generation_creation_requests",
    )
    op.drop_table("index_generation_creation_requests")

    for name in (
        "ck_kb_index_generations_active_validation_complete",
        "ck_kb_index_generations_safe_error_message_length",
        "ck_kb_index_generations_safe_error_code_length",
        "ck_kb_index_generations_embedding_config_hash",
        "ck_kb_index_generations_filter_schema_revision_nonnegative",
        "ck_kb_index_generations_filter_schema_snapshot_object",
        "ck_kb_index_generations_embedding_config_snapshot_object",
        "ck_kb_index_generations_distance",
    ):
        op.drop_constraint(name, "knowledge_base_index_generations", type_="check")
    for column in (
        "safe_error_message",
        "safe_error_code",
        "embedding_config_hash",
        "applied_filter_schema_revision",
        "filter_schema_snapshot",
        "embedding_config_snapshot",
        "distance",
    ):
        op.drop_column("knowledge_base_index_generations", column)

    op.drop_constraint("ck_model_profiles_vector_config_object", "model_profiles", type_="check")
    op.drop_constraint("ck_model_profiles_revision_positive", "model_profiles", type_="check")
    op.drop_column("model_profiles", "vector_config")
    op.drop_column("model_profiles", "resource_revision")

    op.drop_index("ix_provider_configs_credential_id", table_name="provider_configs")
    op.drop_constraint(
        "ck_provider_configs_credential_endpoint_validation", "provider_configs", type_="check"
    )
    op.drop_constraint(
        "ck_provider_configs_endpoint_policy_version_length",
        "provider_configs",
        type_="check",
    )
    op.drop_constraint(
        "ck_provider_configs_credential_source_exactly_one", "provider_configs", type_="check"
    )
    op.drop_constraint("ck_provider_configs_revision_positive", "provider_configs", type_="check")
    op.drop_constraint("fk_provider_configs_credential", "provider_configs", type_="foreignkey")
    op.drop_column("provider_configs", "endpoint_validated_at")
    op.drop_column("provider_configs", "endpoint_policy_version")
    op.drop_column("provider_configs", "resource_revision")
    op.drop_column("provider_configs", "credential_id")
    op.alter_column("provider_configs", "secret_ref", existing_type=sa.String(255), nullable=False)
    op.drop_table("provider_credentials")
