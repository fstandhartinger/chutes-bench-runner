"""Add AA-Omniscience, CritPt, and GDPval-AA benchmarks.

Revision ID: 007
Revises: 006
Create Date: 2026-01-07
"""
from typing import Sequence, Union
from uuid import uuid4

from alembic import op
import sqlalchemy as sa

revision: str = "007"
down_revision: Union[str, None] = "006"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

CATEGORY = "Core Benchmarks"

BENCHMARKS = [
    {
        "name": "aa_omniscience",
        "display_name": "AA-Omniscience",
        "description": "Artificial Analysis Omniscience benchmark (public 10% set).",
        "adapter_class": "AAOmniscienceAdapter",
        "is_enabled": True,
        "supports_subset": True,
        "requires_setup": False,
        "config": {"category": CATEGORY, "default_selected": False},
        "total_items": 600,
    },
    {
        "name": "critpt",
        "display_name": "CritPt",
        "description": "CritPt physics research coding benchmark (external evaluation server).",
        "adapter_class": "CritPtAdapter",
        "is_enabled": True,
        "supports_subset": False,
        "requires_setup": True,
        "setup_notes": "Requires CRITPT_EVAL_URL (and CRITPT_API_KEY if protected).",
        "config": {"category": CATEGORY, "default_selected": False},
        "total_items": 70,
    },
    {
        "name": "gdpval_aa",
        "display_name": "GDPval-AA",
        "description": "GDPval tasks with LLM-based grading against reference documents.",
        "adapter_class": "GDPvalAAAdapter",
        "is_enabled": True,
        "supports_subset": True,
        "requires_setup": False,
        "config": {"category": CATEGORY, "default_selected": False},
        "total_items": 220,
    },
]


def upgrade() -> None:
    benchmarks_table = sa.table(
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

    conn = op.get_bind()
    existing = {row[0] for row in conn.execute(sa.select(benchmarks_table.c.name))}

    for benchmark in BENCHMARKS:
        if benchmark["name"] in existing:
            continue
        op.execute(
            benchmarks_table.insert().values(
                id=str(uuid4()),
                name=benchmark["name"],
                display_name=benchmark["display_name"],
                description=benchmark.get("description"),
                adapter_class=benchmark["adapter_class"],
                is_enabled=benchmark.get("is_enabled", True),
                supports_subset=benchmark.get("supports_subset", True),
                requires_setup=benchmark.get("requires_setup", False),
                setup_notes=benchmark.get("setup_notes"),
                version="1.0.0",
                total_items=benchmark.get("total_items", 0),
                config=benchmark.get("config"),
            )
        )


def downgrade() -> None:
    names = [benchmark["name"] for benchmark in BENCHMARKS]
    placeholders = ", ".join([f"'{name}'" for name in names])
    op.execute(f"DELETE FROM benchmarks WHERE name IN ({placeholders})")
