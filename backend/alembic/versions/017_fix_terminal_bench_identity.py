"""Register version-pinned Terminal-Bench adapters.

Revision ID: 017_tb_identity
Revises: 016_run_retry_cols
Create Date: 2026-08-06
"""

from collections.abc import Sequence
from uuid import uuid4

import sqlalchemy as sa

from alembic import op

revision: str = "017_tb_identity"
down_revision: str | None = "016_run_retry_cols"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


BENCHMARKS = [
    {
        "name": "terminal_bench",
        "display_name": "Terminal-Bench (current: 2.1)",
        "adapter_class": "TerminalBenchAdapter",
        "version": "2.1",
        "total_items": 89,
    },
    {
        "name": "terminal_bench_1",
        "display_name": "Terminal-Bench 1.0 Core",
        "adapter_class": "TerminalBench1Adapter",
        "version": "1.0 / core 0.1.1",
        "total_items": 80,
    },
    {
        "name": "terminal_bench_2",
        "display_name": "Terminal-Bench 2.x (current: 2.1)",
        "adapter_class": "TerminalBench2Adapter",
        "version": "2.1",
        "total_items": 89,
    },
    {
        "name": "terminal_bench_2_0",
        "display_name": "Terminal-Bench 2.0",
        "adapter_class": "TerminalBench20Adapter",
        "version": "2.0",
        "total_items": 89,
    },
    {
        "name": "terminal_bench_2_1",
        "display_name": "Terminal-Bench 2.1",
        "adapter_class": "TerminalBench21Adapter",
        "version": "2.1",
        "total_items": 89,
    },
]


def upgrade() -> None:
    benchmarks = sa.table(
        "benchmarks",
        sa.column("id", sa.String),
        sa.column("name", sa.String),
        sa.column("display_name", sa.String),
        sa.column("description", sa.Text),
        sa.column("adapter_class", sa.String),
        sa.column("is_enabled", sa.Boolean),
        sa.column("supports_subset", sa.Boolean),
        sa.column("requires_setup", sa.Boolean),
        sa.column("setup_notes", sa.Text),
        sa.column("version", sa.String),
        sa.column("total_items", sa.Integer),
        sa.column("config", sa.JSON),
    )

    for benchmark in BENCHMARKS:
        op.execute(
            benchmarks.insert().values(
                id=str(uuid4()),
                name=benchmark["name"],
                display_name=benchmark["display_name"],
                description=(
                    "Version-pinned official Terminal-Bench task set with an "
                    "exact task manifest and startup identity check."
                ),
                adapter_class=benchmark["adapter_class"],
                is_enabled=True,
                supports_subset=True,
                requires_setup=True,
                setup_notes="Requires Sandy with Docker socket access.",
                version=benchmark["version"],
                total_items=benchmark["total_items"],
                config={"category": "Agentic Coding", "default_selected": False},
            )
        )

    op.execute(
        benchmarks.update()
        .where(benchmarks.c.name == "terminal_bench_hard")
        .values(
            display_name="Terminal-Bench Hard (47-task leaderboard subset)",
            description=(
                "Reproducible 47-task Terminal-Bench Hard leaderboard subset, "
                "pinned to the reported upstream task revision."
            ),
            setup_notes="Requires Sandy with Docker socket access.",
            version="1.0 / 47-task hard subset",
            total_items=47,
        )
    )


def downgrade() -> None:
    benchmarks = sa.table(
        "benchmarks",
        sa.column("name", sa.String),
        sa.column("display_name", sa.String),
        sa.column("description", sa.Text),
        sa.column("setup_notes", sa.Text),
        sa.column("version", sa.String),
        sa.column("total_items", sa.Integer),
    )
    for benchmark in BENCHMARKS:
        op.execute(benchmarks.delete().where(benchmarks.c.name == benchmark["name"]))
    op.execute(
        benchmarks.update()
        .where(benchmarks.c.name == "terminal_bench_hard")
        .values(
            display_name="Terminal-Bench Hard",
            description="Challenging terminal/CLI interaction benchmark",
            setup_notes=(
                "Terminal-Bench requires Docker or isolated shell environment for execution."
            ),
            version="1.0.0",
            total_items=0,
        )
    )
