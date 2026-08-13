"""Record which reranker a knowledge base may use.

Revision ID: 20260813_0006
Revises: 20260730_0005
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260813_0006"
down_revision: str | Sequence[str] | None = "20260730_0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("knowledge_bases", sa.Column("rerank_profile_id", sa.UUID(), nullable=True))
    # RESTRICT rather than SET NULL: a profile still selected by a knowledge base
    # must not be deletable out from under an in-flight search, and silently
    # dropping the pointer would turn reranking off without anyone asking.
    op.create_foreign_key(
        "fk_knowledge_bases_rerank_profile",
        "knowledge_bases",
        "model_profiles",
        ["rerank_profile_id"],
        ["id"],
        ondelete="RESTRICT",
    )


def downgrade() -> None:
    op.drop_constraint("fk_knowledge_bases_rerank_profile", "knowledge_bases", type_="foreignkey")
    op.drop_column("knowledge_bases", "rerank_profile_id")
