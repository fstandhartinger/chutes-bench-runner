"""Enforce parent-to-child benchmark run cancellation.

Revision ID: 020_run_cancellation
Revises: 019_agent_evidence
Create Date: 2026-08-07
"""
from collections.abc import Sequence

from alembic import op

revision: str = "020_run_cancellation"
down_revision: str | None = "019_agent_evidence"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Repair the inconsistent rows that motivated this migration before
    # installing the invariant for future parent updates.
    op.execute(
        """
        UPDATE benchmark_run_benchmarks AS child
        SET status = 'skipped',
            error_message = 'Run canceled',
            completed_at = COALESCE(child.completed_at, NOW()),
            updated_at = NOW()
        FROM benchmark_runs AS parent
        WHERE child.run_id = parent.id
          AND parent.status = 'canceled'
          AND child.status NOT IN ('succeeded', 'failed', 'skipped')
        """
    )
    op.execute(
        """
        UPDATE benchmark_runs
        SET canceled_at = COALESCE(canceled_at, completed_at, NOW()),
            completed_at = COALESCE(completed_at, canceled_at, NOW())
        WHERE status = 'canceled'
        """
    )
    op.execute(
        """
        CREATE FUNCTION propagate_benchmark_run_cancellation()
        RETURNS trigger
        LANGUAGE plpgsql
        AS $$
        BEGIN
            IF NEW.status = 'canceled' THEN
                NEW.canceled_at := COALESCE(NEW.canceled_at, NEW.completed_at, NOW());
                NEW.completed_at := COALESCE(NEW.completed_at, NEW.canceled_at, NOW());

                UPDATE benchmark_run_benchmarks
                SET status = 'skipped',
                    error_message = 'Run canceled',
                    completed_at = COALESCE(completed_at, NOW()),
                    updated_at = NOW()
                WHERE run_id = NEW.id
                  AND status NOT IN ('succeeded', 'failed', 'skipped');
            END IF;
            RETURN NEW;
        END;
        $$
        """
    )
    op.execute(
        """
        CREATE TRIGGER benchmark_runs_propagate_cancellation
        BEFORE UPDATE OF status ON benchmark_runs
        FOR EACH ROW
        EXECUTE FUNCTION propagate_benchmark_run_cancellation()
        """
    )


def downgrade() -> None:
    op.execute(
        "DROP TRIGGER IF EXISTS benchmark_runs_propagate_cancellation ON benchmark_runs"
    )
    op.execute("DROP FUNCTION IF EXISTS propagate_benchmark_run_cancellation()")
