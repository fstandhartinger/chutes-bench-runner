"""Initial migration

Revision ID: 001
Revises:
Create Date: 2025-01-01

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Models table
    op.create_table(
        "models",
        sa.Column("id", postgresql.UUID(as_uuid=False), primary_key=True),
        sa.Column("slug", sa.String(255), unique=True, nullable=False, index=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("tagline", sa.Text, nullable=True),
        sa.Column("user", sa.String(255), nullable=True),
        sa.Column("logo", sa.Text, nullable=True),
        sa.Column("chute_id", sa.String(255), nullable=True),
        sa.Column("instance_count", sa.Integer, default=0),
        sa.Column("is_active", sa.Boolean, default=True),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime, server_default=sa.func.now()),
    )

    # Benchmarks table
    op.create_table(
        "benchmarks",
        sa.Column("id", postgresql.UUID(as_uuid=False), primary_key=True),
        sa.Column("name", sa.String(100), unique=True, nullable=False),
        sa.Column("display_name", sa.String(255), nullable=False),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("version", sa.String(50), default="1.0.0"),
        sa.Column("adapter_class", sa.String(255), nullable=False),
        sa.Column("is_enabled", sa.Boolean, default=True),
        sa.Column("supports_subset", sa.Boolean, default=True),
        sa.Column("requires_setup", sa.Boolean, default=False),
        sa.Column("setup_notes", sa.Text, nullable=True),
        sa.Column("config", postgresql.JSONB, nullable=True),
        sa.Column("total_items", sa.Integer, default=0),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime, server_default=sa.func.now()),
    )

    # Benchmark items table
    op.create_table(
        "benchmark_items",
        sa.Column("id", postgresql.UUID(as_uuid=False), primary_key=True),
        sa.Column("benchmark_id", postgresql.UUID(as_uuid=False), sa.ForeignKey("benchmarks.id"), nullable=False),
        sa.Column("item_id", sa.String(255), nullable=False),
        sa.Column("item_hash", sa.String(64), nullable=True),
        sa.Column("data", postgresql.JSONB, nullable=True),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
    )

    # Benchmark runs table
    op.create_table(
        "benchmark_runs",
        sa.Column("id", postgresql.UUID(as_uuid=False), primary_key=True),
        sa.Column("model_id", postgresql.UUID(as_uuid=False), sa.ForeignKey("models.id"), nullable=False),
        sa.Column("model_slug", sa.String(255), nullable=False),
        sa.Column("subset_pct", sa.Integer, default=100),
        sa.Column("status", sa.String(50), default="queued"),
        sa.Column("config", postgresql.JSONB, nullable=True),
        sa.Column("selected_benchmarks", postgresql.JSONB, nullable=True),
        sa.Column("overall_score", sa.Float, nullable=True),
        sa.Column("error_message", sa.Text, nullable=True),
        sa.Column("started_at", sa.DateTime, nullable=True),
        sa.Column("completed_at", sa.DateTime, nullable=True),
        sa.Column("canceled_at", sa.DateTime, nullable=True),
        sa.Column("code_version", sa.String(100), nullable=True),
        sa.Column("git_sha", sa.String(40), nullable=True),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime, server_default=sa.func.now()),
    )
    op.create_index("ix_benchmark_runs_status", "benchmark_runs", ["status"])
    op.create_index("ix_benchmark_runs_created_at", "benchmark_runs", ["created_at"])

    # Benchmark run benchmarks table
    op.create_table(
        "benchmark_run_benchmarks",
        sa.Column("id", postgresql.UUID(as_uuid=False), primary_key=True),
        sa.Column("run_id", postgresql.UUID(as_uuid=False), sa.ForeignKey("benchmark_runs.id"), nullable=False),
        sa.Column("benchmark_id", postgresql.UUID(as_uuid=False), sa.ForeignKey("benchmarks.id"), nullable=False),
        sa.Column("benchmark_name", sa.String(100), nullable=False),
        sa.Column("status", sa.String(50), default="pending"),
        sa.Column("total_items", sa.Integer, default=0),
        sa.Column("completed_items", sa.Integer, default=0),
        sa.Column("sampled_items", sa.Integer, default=0),
        sa.Column("sampled_item_ids", postgresql.JSONB, nullable=True),
        sa.Column("metrics", postgresql.JSONB, nullable=True),
        sa.Column("score", sa.Float, nullable=True),
        sa.Column("error_message", sa.Text, nullable=True),
        sa.Column("started_at", sa.DateTime, nullable=True),
        sa.Column("completed_at", sa.DateTime, nullable=True),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime, server_default=sa.func.now()),
    )
    op.create_index("ix_benchmark_run_benchmarks_run_id", "benchmark_run_benchmarks", ["run_id"])

    # Benchmark item results table
    op.create_table(
        "benchmark_item_results",
        sa.Column("id", postgresql.UUID(as_uuid=False), primary_key=True),
        sa.Column("run_benchmark_id", postgresql.UUID(as_uuid=False), sa.ForeignKey("benchmark_run_benchmarks.id"), nullable=False),
        sa.Column("item_id", sa.String(255), nullable=False),
        sa.Column("item_hash", sa.String(64), nullable=True),
        sa.Column("prompt", sa.Text, nullable=True),
        sa.Column("response", sa.Text, nullable=True),
        sa.Column("expected", sa.Text, nullable=True),
        sa.Column("is_correct", sa.Boolean, nullable=True),
        sa.Column("score", sa.Float, nullable=True),
        sa.Column("judge_output", postgresql.JSONB, nullable=True),
        sa.Column("latency_ms", sa.Integer, nullable=True),
        sa.Column("input_tokens", sa.Integer, nullable=True),
        sa.Column("output_tokens", sa.Integer, nullable=True),
        sa.Column("error", sa.Text, nullable=True),
        sa.Column("metadata", postgresql.JSONB, nullable=True),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
    )
    op.create_index("ix_benchmark_item_results_run_benchmark_id", "benchmark_item_results", ["run_benchmark_id"])

    # Run events table
    op.create_table(
        "run_events",
        sa.Column("id", postgresql.UUID(as_uuid=False), primary_key=True),
        sa.Column("run_id", postgresql.UUID(as_uuid=False), sa.ForeignKey("benchmark_runs.id"), nullable=False),
        sa.Column("event_type", sa.String(50), nullable=False),
        sa.Column("benchmark_name", sa.String(100), nullable=True),
        sa.Column("message", sa.Text, nullable=True),
        sa.Column("data", postgresql.JSONB, nullable=True),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now(), index=True),
    )
    op.create_index("ix_run_events_run_id", "run_events", ["run_id"])

    # Exports table
    op.create_table(
        "exports",
        sa.Column("id", postgresql.UUID(as_uuid=False), primary_key=True),
        sa.Column("run_id", postgresql.UUID(as_uuid=False), sa.ForeignKey("benchmark_runs.id"), nullable=False),
        sa.Column("format", sa.String(10), nullable=False),
        sa.Column("filename", sa.String(255), nullable=False),
        sa.Column("content_path", sa.Text, nullable=True),
        sa.Column("content_size", sa.Integer, nullable=True),
        sa.Column("generated_at", sa.DateTime, server_default=sa.func.now()),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
    )


def downgrade() -> None:
    op.drop_table("exports")
    op.drop_table("run_events")
    op.drop_table("benchmark_item_results")
    op.drop_table("benchmark_run_benchmarks")
    op.drop_table("benchmark_runs")
    op.drop_table("benchmark_items")
    op.drop_table("benchmarks")
    op.drop_table("models")

