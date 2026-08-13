"""Allow a job that destroys a deleted knowledge base's data.

Revision ID: 20260813_0007
Revises: 20260813_0006
"""

from collections.abc import Sequence

from alembic import op

revision: str = "20260813_0007"
down_revision: str | Sequence[str] | None = "20260813_0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_OPERATIONS_WITHOUT_PURGE = (
    "operation IN ('ingest_document', 'index_document', 'delete_document', "
    "'rebuild_generation', 'apply_filter_schema', 'cleanup_generation', "
    "'cleanup_document_version')"
)
_OPERATIONS_WITH_PURGE = (
    "operation IN ('ingest_document', 'index_document', 'delete_document', "
    "'rebuild_generation', 'apply_filter_schema', 'cleanup_generation', "
    "'cleanup_document_version', 'purge_knowledge_base')"
)


def upgrade() -> None:
    op.drop_constraint("ck_jobs_operation", "jobs", type_="check")
    op.create_check_constraint("ck_jobs_operation", "jobs", _OPERATIONS_WITH_PURGE)


def downgrade() -> None:
    # Rows naming the new operation would violate the narrower constraint, and a
    # purge job that has already destroyed data cannot be meaningfully replayed,
    # so they are dropped rather than blocking the downgrade.
    op.execute("DELETE FROM jobs WHERE operation = 'purge_knowledge_base'")
    op.drop_constraint("ck_jobs_operation", "jobs", type_="check")
    op.create_check_constraint("ck_jobs_operation", "jobs", _OPERATIONS_WITHOUT_PURGE)
