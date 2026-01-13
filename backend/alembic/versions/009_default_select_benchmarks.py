"""Default-select AA-Omniscience, CritPt, and GDPval-AA.

Revision ID: 009_default_select_benchmarks
Revises: 008_add_provider_fields
Create Date: 2026-01-14
"""
from alembic import op
import sqlalchemy as sa

revision = "009_default_select_benchmarks"
down_revision = "008_add_provider_fields"
branch_labels = None
depends_on = None

BENCHMARKS = ["aa_omniscience", "critpt", "gdpval_aa"]
DEFAULT_CATEGORY = "Core Benchmarks"


def _update_default_selected(default_selected: bool) -> None:
    conn = op.get_bind()
    benchmarks_table = sa.table(
        "benchmarks",
        sa.column("name", sa.String),
        sa.column("config", sa.JSON),
    )
    rows = conn.execute(
        sa.select(benchmarks_table.c.name, benchmarks_table.c.config).where(
            benchmarks_table.c.name.in_(BENCHMARKS)
        )
    ).fetchall()
    for name, config in rows:
        config = dict(config or {})
        config["default_selected"] = default_selected
        if "category" not in config:
            config["category"] = DEFAULT_CATEGORY
        conn.execute(
            benchmarks_table.update()
            .where(benchmarks_table.c.name == name)
            .values(config=config)
        )


def upgrade() -> None:
    _update_default_selected(True)


def downgrade() -> None:
    _update_default_selected(False)
