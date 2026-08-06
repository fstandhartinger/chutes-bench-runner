"""Register the pinned DeepSWE v1.1 Sandy CLI adapter.

Revision ID: 018_add_deepswe
Revises: 017_tb_identity
Create Date: 2026-08-07
"""

from collections.abc import Sequence
from uuid import uuid4

import sqlalchemy as sa

from alembic import op

revision: str = "018_add_deepswe"
down_revision: str | None = "017_tb_identity"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


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
    op.execute(
        benchmarks.insert().values(
            id=str(uuid4()),
            name="deepswe",
            display_name="DeepSWE v1.1 (Sandy CLI scaffold)",
            description=(
                "Pinned 113-task DeepSWE v1.1 corpus with selectable Sandy CLI "
                "agent and a separate pristine verifier container."
            ),
            adapter_class="DeepSWEAdapter",
            is_enabled=True,
            supports_subset=True,
            requires_setup=True,
            setup_notes=(
                "Requires Sandy with Docker socket access. Pulls one unique task "
                "image at a time; each task declares 2 CPUs, 8 GiB RAM, and "
                "20 GiB storage."
            ),
            version="1.1 / source 435ee89e",
            total_items=113,
            config={"category": "Agentic Coding", "default_selected": False},
        )
    )


def downgrade() -> None:
    benchmarks = sa.table("benchmarks", sa.column("name", sa.String))
    op.execute(benchmarks.delete().where(benchmarks.c.name == "deepswe"))
