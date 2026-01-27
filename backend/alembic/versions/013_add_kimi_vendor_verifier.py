"""Add Kimi Vendor Verifier benchmark.

Revision ID: 013_add_kimi_vendor_verifier
Revises: 012_add_janus_benchmarks
Create Date: 2026-01-27
"""
from typing import Sequence, Union
from uuid import uuid4

from alembic import op
import sqlalchemy as sa

revision: str = "013_add_kimi_vendor_verifier"
down_revision: Union[str, None] = "012_add_janus_benchmarks"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

KIMI_CATEGORY = "Kimi Vendor Verifier"

BENCHMARKS = [
    {
        "name": "kimi_vendor_verifier",
        "display_name": "Kimi Vendor Verifier",
        "description": "Tool-call trigger similarity and schema accuracy vs Moonshot reference",
        "adapter_class": "KimiVendorVerifierAdapter",
        "total_items": 2000,
        "config": {
            "category": KIMI_CATEGORY,
            "default_selected": False,
            "k2vv_reference_model": "kimi-k2-0905-preview",
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
                setup_notes=None,
                version="1.0.0",
                total_items=benchmark.get("total_items", 0),
                config=benchmark.get("config"),
            )
        )


def downgrade() -> None:
    op.execute("DELETE FROM benchmarks WHERE name = 'kimi_vendor_verifier'")
