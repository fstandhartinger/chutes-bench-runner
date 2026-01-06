"""Add Affine environment benchmarks.

Revision ID: 004
Revises: 003
Create Date: 2026-01-05
"""
from typing import Sequence, Union
from uuid import uuid4

from alembic import op
import sqlalchemy as sa

revision: str = "004"
down_revision: Union[str, None] = "003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

CATEGORY = "Affine Environments"

BENCHMARKS = [
    {
        "name": "affine_print",
        "display_name": "Affine PRINT",
        "description": "Affine environment: PRINT (evaluation environment for AI model testing).",
        "adapter_class": "AffineEnvAdapter",
        "is_enabled": True,
        "supports_subset": True,
        "requires_setup": False,
        "config": {"category": CATEGORY, "default_selected": False},
        "total_items": 11000,
    },
    {
        "name": "affine_lgc_v2",
        "display_name": "Affine LGC-V2",
        "description": "Affine environment: LGC-V2 logic evaluation.",
        "adapter_class": "AffineEnvAdapter",
        "is_enabled": True,
        "supports_subset": True,
        "requires_setup": False,
        "config": {"category": CATEGORY, "default_selected": False},
        "total_items": 10600,
    },
    {
        "name": "affine_game",
        "display_name": "Affine GAME",
        "description": "Affine environment: GAME (OpenSpiel evaluation).",
        "adapter_class": "AffineEnvAdapter",
        "is_enabled": True,
        "supports_subset": True,
        "requires_setup": False,
        "config": {"category": CATEGORY, "default_selected": False},
        "total_items": 7300,
    },
    {
        "name": "affine_ded",
        "display_name": "Affine DED",
        "description": "Affine environment: DED (program deduction).",
        "adapter_class": "AffineEnvAdapter",
        "is_enabled": True,
        "supports_subset": True,
        "requires_setup": False,
        "config": {"category": CATEGORY, "default_selected": False},
        "total_items": 23300,
    },
    {
        "name": "affine_cde",
        "display_name": "Affine CDE",
        "description": "Affine environment: CDE code evaluation.",
        "adapter_class": "AffineEnvAdapter",
        "is_enabled": True,
        "supports_subset": True,
        "requires_setup": False,
        "config": {"category": CATEGORY, "default_selected": False},
        "total_items": 8600,
    },
    {
        "name": "affine_lgc",
        "display_name": "Affine LGC",
        "description": "Affine environment: LGC logic evaluation.",
        "adapter_class": "AffineEnvAdapter",
        "is_enabled": True,
        "supports_subset": True,
        "requires_setup": False,
        "config": {"category": CATEGORY, "default_selected": False},
        "total_items": 1100000,
    },
    {
        "name": "affine_abd",
        "display_name": "Affine ABD",
        "description": "Affine environment: ABD (program abduction).",
        "adapter_class": "AffineEnvAdapter",
        "is_enabled": True,
        "supports_subset": True,
        "requires_setup": False,
        "config": {"category": CATEGORY, "default_selected": False},
        "total_items": 23300,
    },
    {
        "name": "affine_swe_pro",
        "display_name": "Affine SWE-PRO",
        "description": "Affine environment: SWE-Pro (software engineering).",
        "adapter_class": "AffineEnvAdapter",
        "is_enabled": True,
        "supports_subset": True,
        "requires_setup": False,
        "config": {"category": CATEGORY, "default_selected": False},
        "total_items": 731,
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
