"""Add durable index-generation cleanup claims.

Revision ID: 20260728_0004
Revises: 20260726_0003
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260728_0004"
down_revision: str | Sequence[str] | None = "20260726_0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "index_generation_cleanup_claims",
        sa.Column("collection_name", sa.String(length=255), nullable=False),
        sa.Column("knowledge_base_id", sa.UUID(), nullable=False),
        sa.Column("generation_id", sa.UUID(), nullable=False),
        sa.Column("lease_owner", sa.UUID(), nullable=False),
        sa.Column("lease_epoch", sa.BigInteger(), server_default=sa.text("1"), nullable=False),
        sa.Column("lease_expires_at", postgresql.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("completed_at", postgresql.TIMESTAMP(timezone=True), nullable=True),
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
            "char_length(collection_name) BETWEEN 1 AND 255",
            name="ck_generation_cleanup_claims_collection_name_length",
        ),
        sa.CheckConstraint(
            "collection_name = 'rag_kb_' || "
            "replace(knowledge_base_id::text, '-', '') || '_g_' || "
            "replace(generation_id::text, '-', '')",
            name="ck_generation_cleanup_claims_collection_identity",
        ),
        sa.CheckConstraint(
            "lease_epoch >= 1",
            name="ck_generation_cleanup_claims_lease_epoch_positive",
        ),
        sa.PrimaryKeyConstraint("collection_name"),
    )
    op.create_index(
        "ix_generation_cleanup_claims_expired",
        "index_generation_cleanup_claims",
        ["lease_expires_at", "collection_name"],
        postgresql_where=sa.text("completed_at IS NULL"),
    )


def downgrade() -> None:
    connection = op.get_bind()
    connection.execute(
        sa.text("LOCK TABLE index_generation_cleanup_claims IN ACCESS EXCLUSIVE MODE")
    )
    if connection.scalar(sa.text("SELECT EXISTS (SELECT 1 FROM index_generation_cleanup_claims)")):
        raise RuntimeError(
            "index-generation cleanup claims cannot be represented by revision "
            "20260726_0003; wait for all cleanup operations to quiesce and remove "
            "every claim before retrying the downgrade"
        )
    op.drop_index(
        "ix_generation_cleanup_claims_expired",
        table_name="index_generation_cleanup_claims",
    )
    op.drop_table("index_generation_cleanup_claims")
