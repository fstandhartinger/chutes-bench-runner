"""Add model_comparisons table for RLM vs Base model benchmark comparisons.

This supports spec 003: systematically comparing base models vs RLM variants
on long-context benchmarks (S-NIAH, OOLONG, OOLONG-Pairs).

Revision ID: 011_add_model_comparisons
Revises: 010_add_long_context_benchmarks
Create Date: 2026-01-18
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "011_add_model_comparisons"
down_revision: Union[str, None] = "010_add_long_context_benchmarks"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "model_comparisons",
        sa.Column("id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("base_run_id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("base_model_slug", sa.String(255), nullable=False),
        sa.Column("rlm_run_id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("rlm_model_slug", sa.String(255), nullable=False),
        sa.Column("name", sa.String(255), nullable=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("subset_seed", sa.String(128), nullable=True),
        sa.Column("benchmarks", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("status", sa.String(50), nullable=False, server_default="pending"),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("results", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("markdown_report", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.text("now()")),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["base_run_id"], ["benchmark_runs.id"]),
        sa.ForeignKeyConstraint(["rlm_run_id"], ["benchmark_runs.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_model_comparisons_status", "model_comparisons", ["status"])
    op.create_index("ix_model_comparisons_created_at", "model_comparisons", ["created_at"])


def downgrade() -> None:
    op.drop_index("ix_model_comparisons_created_at", table_name="model_comparisons")
    op.drop_index("ix_model_comparisons_status", table_name="model_comparisons")
    op.drop_table("model_comparisons")
