"""Add durable per-item CLI-agent evidence provenance.

Revision ID: 019_agent_evidence
Revises: 018_add_deepswe
Create Date: 2026-08-07
"""
from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "019_agent_evidence"
down_revision: str | None = "018_add_deepswe"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    table = "benchmark_item_results"
    op.add_column(table, sa.Column("agent_evidence_status", sa.String(32), nullable=True))
    op.add_column(table, sa.Column("agent_evidence_path", sa.Text(), nullable=True))
    op.add_column(table, sa.Column("agent_evidence_sha256", sa.String(64), nullable=True))
    op.add_column(table, sa.Column("agent_evidence_size_bytes", sa.BigInteger(), nullable=True))
    op.add_column(table, sa.Column("agent_evidence_error", sa.Text(), nullable=True))
    op.add_column(
        table,
        sa.Column("token_usage_samples", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )


def downgrade() -> None:
    table = "benchmark_item_results"
    op.drop_column(table, "token_usage_samples")
    op.drop_column(table, "agent_evidence_error")
    op.drop_column(table, "agent_evidence_size_bytes")
    op.drop_column(table, "agent_evidence_sha256")
    op.drop_column(table, "agent_evidence_path")
    op.drop_column(table, "agent_evidence_status")
