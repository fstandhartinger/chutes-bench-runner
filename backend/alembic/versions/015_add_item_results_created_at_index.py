"""Add created_at index for benchmark item results.

Revision ID: 015_add_item_results_created_at_index
Revises: 014_add_deepresearch_bench
Create Date: 2026-03-02
"""
from typing import Sequence, Union

from alembic import op

revision: str = "015_add_item_results_created_at_index"
down_revision: Union[str, None] = "014_add_deepresearch_bench"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

INDEX_NAME = "ix_benchmark_item_results_created_at"
TABLE_NAME = "benchmark_item_results"


def upgrade() -> None:
    # CONCURRENTLY avoids a full table lock on production-sized datasets.
    with op.get_context().autocommit_block():
        op.execute(
            f"CREATE INDEX CONCURRENTLY IF NOT EXISTS {INDEX_NAME} "
            f"ON {TABLE_NAME} (created_at)"
        )


def downgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {INDEX_NAME}")

