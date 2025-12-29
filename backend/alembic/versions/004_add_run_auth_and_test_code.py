"""Add run auth fields and test_code column.

Revision ID: 004
Revises: 003
Create Date: 2025-12-29
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers
revision = "004"
down_revision = "003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("benchmark_runs", sa.Column("auth_mode", sa.String(50), nullable=True))
    op.add_column("benchmark_runs", sa.Column("auth_session_id", sa.String(64), nullable=True))
    op.add_column("benchmark_runs", sa.Column("auth_api_key", sa.Text(), nullable=True))
    op.add_column("benchmark_item_results", sa.Column("test_code", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("benchmark_item_results", "test_code")
    op.drop_column("benchmark_runs", "auth_api_key")
    op.drop_column("benchmark_runs", "auth_session_id")
    op.drop_column("benchmark_runs", "auth_mode")
