"""Persist the API-key actor that created an ingestion job.

Revision ID: 20260730_0005
Revises: 20260728_0004
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260730_0005"
down_revision: str | Sequence[str] | None = "20260728_0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("jobs", sa.Column("actor_api_key_id", sa.UUID(), nullable=True))
    op.create_foreign_key(
        "fk_jobs_actor_api_key",
        "jobs",
        "api_keys",
        ["actor_api_key_id"],
        ["id"],
        ondelete="RESTRICT",
    )


def downgrade() -> None:
    connection = op.get_bind()
    connection.execute(sa.text("LOCK TABLE jobs IN ACCESS EXCLUSIVE MODE"))
    if connection.scalar(
        sa.text("SELECT EXISTS (SELECT 1 FROM jobs WHERE actor_api_key_id IS NOT NULL)")
    ):
        raise RuntimeError(
            "job actors cannot be represented by revision 20260728_0004; "
            "remove every job actor before retrying the downgrade"
        )
    op.drop_constraint("fk_jobs_actor_api_key", "jobs", type_="foreignkey")
    op.drop_column("jobs", "actor_api_key_id")
