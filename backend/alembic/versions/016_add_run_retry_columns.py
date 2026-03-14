"""Add retry_count and max_retries columns to benchmark_runs.

Revision ID: 016_add_run_retry_columns
Revises: 015_add_item_results_created_at_index
Create Date: 2026-03-14
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "016_run_retry_cols"
down_revision: Union[str, None] = "015_item_created_idx"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "benchmark_runs",
        sa.Column("retry_count", sa.Integer(), server_default="0", nullable=False),
    )
    op.add_column(
        "benchmark_runs",
        sa.Column("max_retries", sa.Integer(), server_default="3", nullable=False),
    )


def downgrade() -> None:
    op.drop_column("benchmark_runs", "max_retries")
    op.drop_column("benchmark_runs", "retry_count")
