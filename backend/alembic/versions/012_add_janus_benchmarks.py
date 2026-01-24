"""Add Janus benchmark entries.

Revision ID: 012
Revises: 011
Create Date: 2025-01-15
"""
from typing import Sequence, Union
from uuid import uuid4

from alembic import op
import sqlalchemy as sa

revision: str = "012"
down_revision: Union[str, None] = "011"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

JANUS_CATEGORY = "Janus Intelligence"

BENCHMARKS = [
    {
        "name": "janus_research",
        "display_name": "Janus Research",
        "description": "Web research, search, and synthesis capabilities",
        "adapter_class": "JanusResearchAdapter",
        "total_items": 100,
        "config": {
            "category": JANUS_CATEGORY,
            "default_selected": False,
            "janus_scoring_weight": 0.20,
        },
    },
    {
        "name": "janus_tool_use",
        "display_name": "Janus Tool Use",
        "description": "Tool-use and multi-step reasoning tasks",
        "adapter_class": "JanusToolUseAdapter",
        "total_items": 80,
        "config": {
            "category": JANUS_CATEGORY,
            "default_selected": False,
            "janus_scoring_weight": 0.20,
        },
    },
    {
        "name": "janus_multimodal",
        "display_name": "Janus Multimodal",
        "description": "Multimodal understanding and reasoning tasks",
        "adapter_class": "JanusMultimodalAdapter",
        "total_items": 60,
        "config": {
            "category": JANUS_CATEGORY,
            "default_selected": False,
            "janus_scoring_weight": 0.10,
        },
    },
    {
        "name": "janus_streaming",
        "display_name": "Janus Streaming",
        "description": "Streaming responsiveness and continuity checks",
        "adapter_class": "JanusStreamingAdapter",
        "total_items": 50,
        "config": {
            "category": JANUS_CATEGORY,
            "default_selected": False,
            "janus_scoring_weight": 0.15,
            "janus_metrics": ["ttft", "tps", "continuity"],
        },
    },
    {
        "name": "janus_cost",
        "display_name": "Janus Cost Efficiency",
        "description": "Cost efficiency and token usage optimization",
        "adapter_class": "JanusCostAdapter",
        "total_items": 40,
        "config": {
            "category": JANUS_CATEGORY,
            "default_selected": False,
            "janus_scoring_weight": 0.15,
            "janus_metrics": ["token_savings_pct"],
        },
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
                setup_notes=None,
                version="1.0.0",
                total_items=benchmark.get("total_items", 0),
                config=benchmark.get("config"),
            )
        )


def downgrade() -> None:
    op.execute(
        "DELETE FROM benchmarks WHERE name LIKE 'janus_%'"
    )
