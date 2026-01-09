"""Add provider fields for Gremium support.

Revision ID: 008_add_provider_fields
Revises: 007_add_new_benchmarks
Create Date: 2026-01-09
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "008_add_provider_fields"
down_revision = "007_add_new_benchmarks"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("models", sa.Column("provider", sa.String(length=32), nullable=False, server_default="chutes"))
    op.add_column("benchmark_runs", sa.Column("provider", sa.String(length=32), nullable=False, server_default="chutes"))
    op.add_column(
        "benchmark_runs",
        sa.Column("provider_metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )

    op.execute("UPDATE models SET provider = 'chutes' WHERE provider IS NULL")
    op.execute("UPDATE benchmark_runs SET provider = 'chutes' WHERE provider IS NULL")

    op.alter_column("models", "provider", server_default=None)
    op.alter_column("benchmark_runs", "provider", server_default=None)


def downgrade() -> None:
    op.drop_column("benchmark_runs", "provider_metadata")
    op.drop_column("benchmark_runs", "provider")
    op.drop_column("models", "provider")
