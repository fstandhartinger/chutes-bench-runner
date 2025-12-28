"""Seed benchmarks

Revision ID: 002
Revises: 001
Create Date: 2025-01-01

"""
from typing import Sequence, Union
from uuid import uuid4

from alembic import op
import sqlalchemy as sa

revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

BENCHMARKS = [
    {
        "name": "mmlu_pro",
        "display_name": "MMLU-Pro",
        "description": "Massive Multitask Language Understanding - Professional level questions across 57 subjects",
        "adapter_class": "MMLUProAdapter",
        "is_enabled": True,
        "supports_subset": True,
        "requires_setup": False,
    },
    {
        "name": "gpqa_diamond",
        "display_name": "GPQA Diamond",
        "description": "Graduate-level science questions requiring expert-level knowledge",
        "adapter_class": "GPQADiamondAdapter",
        "is_enabled": True,
        "supports_subset": True,
        "requires_setup": False,
    },
    {
        "name": "hle",
        "display_name": "Humanity's Last Exam",
        "description": "A challenging benchmark of expert-level questions across many domains",
        "adapter_class": "HLEAdapter",
        "is_enabled": True,
        "supports_subset": True,
        "requires_setup": True,
        "setup_notes": "HLE dataset may require authentication. Check Hugging Face access.",
    },
    {
        "name": "livecodebench",
        "display_name": "LiveCodeBench",
        "description": "A coding benchmark with live programming problems",
        "adapter_class": "LiveCodeBenchAdapter",
        "is_enabled": True,
        "supports_subset": True,
        "requires_setup": True,
        "setup_notes": "LiveCodeBench requires local code execution environment for verification.",
    },
    {
        "name": "scicode",
        "display_name": "SciCode",
        "description": "Scientific computing code generation benchmark",
        "adapter_class": "SciCodeAdapter",
        "is_enabled": True,
        "supports_subset": True,
        "requires_setup": True,
        "setup_notes": "SciCode requires scientific computing libraries (numpy, scipy, etc.) for execution.",
    },
    {
        "name": "aime_2025",
        "display_name": "AIME 2025",
        "description": "American Invitational Mathematics Examination problems",
        "adapter_class": "AIME2025Adapter",
        "is_enabled": True,
        "supports_subset": True,
        "requires_setup": False,
    },
    {
        "name": "ifbench",
        "display_name": "IFBench",
        "description": "Instruction Following benchmark evaluating model's ability to follow complex instructions",
        "adapter_class": "IFBenchAdapter",
        "is_enabled": True,
        "supports_subset": True,
        "requires_setup": False,
    },
    {
        "name": "aa_lcr",
        "display_name": "AA-LCR",
        "description": "Adversarial attacks on code reasoning capabilities",
        "adapter_class": "AALCRAdapter",
        "is_enabled": True,
        "supports_subset": True,
        "requires_setup": True,
        "setup_notes": "AA-LCR dataset may require special access or local setup.",
    },
    {
        "name": "terminal_bench_hard",
        "display_name": "Terminal-Bench Hard",
        "description": "Challenging terminal/CLI interaction benchmark",
        "adapter_class": "TerminalBenchHardAdapter",
        "is_enabled": True,
        "supports_subset": True,
        "requires_setup": True,
        "setup_notes": "Terminal-Bench requires Docker or isolated shell environment for execution.",
    },
    {
        "name": "tau_bench_telecom",
        "display_name": "τ²-Bench Telecom",
        "description": "Telecommunications domain agent benchmark",
        "adapter_class": "TauBenchTelecomAdapter",
        "is_enabled": True,
        "supports_subset": True,
        "requires_setup": True,
        "setup_notes": "τ²-Bench requires the official evaluation harness and environment setup.",
    },
    {
        "name": "swe_bench_pro",
        "display_name": "SWE-Bench Pro",
        "description": "Software engineering benchmark with real GitHub issues",
        "adapter_class": "SWEBenchProAdapter",
        "is_enabled": True,
        "supports_subset": True,
        "requires_setup": True,
        "setup_notes": "SWE-Bench Pro requires Docker, Git credentials, and the official SWE-Bench harness.",
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
    )

    for benchmark in BENCHMARKS:
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
                total_items=0,
            )
        )


def downgrade() -> None:
    op.execute("DELETE FROM benchmarks")


