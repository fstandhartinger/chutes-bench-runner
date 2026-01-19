"""Add S-NIAH, OOLONG, and OOLONG-Pairs long-context benchmarks.

These benchmarks are based on the RLM paper (arxiv:2512.24601v1) and evaluate
long-context reasoning and aggregation capabilities.

Revision ID: 010_add_long_context_benchmarks
Revises: 009_default_select_benchmarks
Create Date: 2026-01-18
"""
from typing import Sequence, Union
from uuid import uuid4

from alembic import op
import sqlalchemy as sa

revision: str = "010_add_long_context_benchmarks"
down_revision: Union[str, None] = "009_default_select_benchmarks"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

CATEGORY = "Long Context Benchmarks"

BENCHMARKS = [
    {
        "name": "s_niah",
        "display_name": "S-NIAH (Needle-in-Haystack)",
        "description": (
            "Single Needle-in-a-Haystack benchmark based on RULER. "
            "Tests retrieval of specific information from long distractor texts. "
            "Processing costs scale roughly constant with input length."
        ),
        "adapter_class": "SNIAHAdapter",
        "is_enabled": True,
        "supports_subset": True,
        "requires_setup": False,
        "config": {
            "category": CATEGORY,
            "default_selected": False,
            "context_sizes": [8192, 16384, 32768, 65536, 131072, 262144],
        },
        "total_items": 60,  # 6 context sizes × 10 tasks each
    },
    {
        "name": "oolong",
        "display_name": "OOLONG",
        "description": (
            "Long-context reasoning benchmark requiring semantic classification "
            "and aggregation across dataset entries. Uses oolongbench/oolong-synth. "
            "Processing costs scale linearly with input length."
        ),
        "adapter_class": "OolongAdapter",
        "is_enabled": True,
        "supports_subset": True,
        "requires_setup": False,
        "config": {"category": CATEGORY, "default_selected": False},
        "total_items": 10600,  # From oolong-synth test split
    },
    {
        "name": "oolong_pairs",
        "display_name": "OOLONG-Pairs",
        "description": (
            "Extension of OOLONG requiring pairwise aggregation. "
            "Uses D&D transcripts from oolongbench/oolong-real. "
            "Processing costs scale quadratically with input length."
        ),
        "adapter_class": "OolongPairsAdapter",
        "is_enabled": True,
        "supports_subset": True,
        "requires_setup": False,
        "config": {"category": CATEGORY, "default_selected": False},
        "total_items": 6130,  # From oolong-real dnd test split
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
