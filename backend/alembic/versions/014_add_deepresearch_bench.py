"""Add DeepResearch-Bench (RACE) benchmark.

Revision ID: 014_add_deepresearch_bench
Revises: 013_add_kimi_vendor_verifier
Create Date: 2026-02-27
"""
from typing import Sequence, Union
from uuid import uuid4

from alembic import op
import sqlalchemy as sa

revision: str = "014_add_deepresearch_bench"
down_revision: Union[str, None] = "013_add_kimi_vendor_verifier"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

BENCHMARKS = [
    {
        "name": "deepresearch_bench",
        "display_name": "DeepResearch-Bench (RACE)",
        "description": (
            "Evaluates deep research API quality using RACE methodology. "
            "Compares search.chutes.ai reports against reference articles "
            "across comprehensiveness, insight, instruction following, and "
            "readability dimensions. Score > 0.5 means target outperforms reference."
        ),
        "adapter_class": "DeepResearchBenchAdapter",
        "total_items": 50,
        "config": {
            "category": "Deep Research",
            "default_selected": False,
            "search_api": "search.chutes.ai",
            "scoring_method": "RACE (static criteria)",
        },
    }
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

    for benchmark in BENCHMARKS:
        op.execute(
            benchmarks_table.insert().values(
                id=str(uuid4()),
                name=benchmark["name"],
                display_name=benchmark["display_name"],
                description=benchmark.get("description"),
                adapter_class=benchmark["adapter_class"],
                is_enabled=True,
                supports_subset=True,
                requires_setup=False,
                setup_notes="Requires CHUTES_API_KEY environment variable for search.chutes.ai access.",
                version="1.0.0",
                total_items=benchmark.get("total_items", 0),
                config=benchmark.get("config"),
            )
        )


def downgrade() -> None:
    op.execute("DELETE FROM benchmarks WHERE name = 'deepresearch_bench'")
