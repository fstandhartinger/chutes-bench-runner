"""Add immutable execution provenance to benchmark runs.

Revision ID: 021_run_provenance
Revises: 020_run_cancellation
Create Date: 2026-08-07
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "021_run_provenance"
down_revision: str | None = "020_run_cancellation"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "benchmark_runs",
        sa.Column(
            "provenance",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
    )


def downgrade() -> None:
    op.drop_column("benchmark_runs", "provenance")
