"""Add subset count/seed and worker heartbeat tables.

Revision ID: 006
Revises: 005
Create Date: 2026-01-08
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision = "006"
down_revision = "005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("benchmark_runs", sa.Column("subset_count", sa.Integer(), nullable=True))
    op.add_column("benchmark_runs", sa.Column("subset_seed", sa.String(length=128), nullable=True))

    op.create_table(
        "worker_heartbeats",
        sa.Column("id", postgresql.UUID(as_uuid=False), primary_key=True, nullable=False),
        sa.Column("worker_id", sa.String(length=128), nullable=False, unique=True),
        sa.Column("hostname", sa.String(length=255), nullable=True),
        sa.Column("running_runs", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("max_concurrent_runs", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("item_concurrency", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("last_seen", sa.DateTime(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_worker_heartbeats_last_seen", "worker_heartbeats", ["last_seen"])

    op.create_table(
        "worker_heartbeat_logs",
        sa.Column("id", postgresql.UUID(as_uuid=False), primary_key=True, nullable=False),
        sa.Column("worker_id", sa.String(length=128), nullable=False),
        sa.Column("hostname", sa.String(length=255), nullable=True),
        sa.Column("running_runs", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("max_concurrent_runs", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("item_concurrency", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_worker_heartbeat_logs_created_at", "worker_heartbeat_logs", ["created_at"])


def downgrade() -> None:
    op.drop_index("ix_worker_heartbeat_logs_created_at", table_name="worker_heartbeat_logs")
    op.drop_table("worker_heartbeat_logs")

    op.drop_index("ix_worker_heartbeats_last_seen", table_name="worker_heartbeats")
    op.drop_table("worker_heartbeats")

    op.drop_column("benchmark_runs", "subset_seed")
    op.drop_column("benchmark_runs", "subset_count")
