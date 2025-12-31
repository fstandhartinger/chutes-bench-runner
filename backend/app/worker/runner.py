"""Background worker for running benchmarks."""
import asyncio
from datetime import datetime
from typing import Optional

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import noload

from app.benchmarks import get_adapter
from app.benchmarks.base import ItemResult
from app.core.config import get_settings
from app.core.logging import get_logger
from app.db.session import async_session_maker
from app.models.benchmark import Benchmark
from app.models.run import (
    BenchmarkRun,
    BenchmarkRunBenchmark,
    BenchmarkRunStatus,
    RunStatus,
)
from app.services import auth_service
from app.services.chutes_client import ChutesClient, get_chutes_client
from app.services.run_service import (
    add_run_event,
    save_item_result,
    update_benchmark_status,
    update_run_status,
)

logger = get_logger(__name__)
settings = get_settings()


class BenchmarkWorker:
    """Worker that processes benchmark runs."""

    def __init__(self):
        self.running = False
        self.current_run_id: Optional[str] = None
        self.client = get_chutes_client()

    async def _get_client_for_run(self, db: AsyncSession, run: BenchmarkRun) -> ChutesClient:
        if run.auth_mode == "idp" and run.auth_session_id:
            session = await auth_service.get_session(db, run.auth_session_id)
            if not session:
                raise Exception("Chutes session not found")
            if not session.can_invoke_chutes():
                raise Exception("Chutes session missing chutes:invoke scope")
            access_token = await auth_service.get_valid_access_token(db, session)
            if not access_token:
                raise Exception("Chutes session expired or invalid")
            return get_chutes_client(user_access_token=access_token)

        if run.auth_mode == "api_key" and run.auth_api_key:
            return get_chutes_client(api_key=run.auth_api_key)

        return self.client

    async def start(self) -> None:
        """Start the worker loop."""
        self.running = True
        logger.info("Worker started")

        while self.running:
            try:
                await self.process_next_run()
            except Exception as e:
                logger.error("Worker error", error=str(e))

            await asyncio.sleep(settings.worker_poll_interval)

    async def stop(self) -> None:
        """Stop the worker."""
        self.running = False
        logger.info("Worker stopping")

    async def process_next_run(self) -> bool:
        """
        Claim and process the next queued run.
        
        Uses SKIP LOCKED to prevent multiple workers from claiming same run.
        
        Returns:
            True if a run was processed, False otherwise
        """
        async with async_session_maker() as db:
            # Claim a queued run with row lock
            # Use noload to prevent JOIN that's incompatible with FOR UPDATE
            result = await db.execute(
                select(BenchmarkRun)
                .options(noload(BenchmarkRun.model))
                .where(BenchmarkRun.status == RunStatus.QUEUED.value)
                .order_by(BenchmarkRun.created_at)
                .limit(1)
                .with_for_update(skip_locked=True)
            )
            run = result.scalar_one_or_none()

            if not run:
                return False

            self.current_run_id = run.id
            logger.info("Claimed run", run_id=run.id, model=run.model_slug)

            # Update status to running
            await update_run_status(db, run.id, RunStatus.RUNNING)
            await add_run_event(
                db, run.id, "run_started",
                message=f"Starting benchmark run for {run.model_slug}"
            )

            try:
                await self.execute_run(db, run)
            except Exception as e:
                logger.error("Run failed", run_id=run.id, error=str(e))
                await update_run_status(db, run.id, RunStatus.FAILED, error_message=str(e))
                await add_run_event(
                    db, run.id, "run_failed",
                    message=f"Run failed: {str(e)}"
                )
            finally:
                self.current_run_id = None

            return True

    async def execute_run(self, db: AsyncSession, run: BenchmarkRun) -> None:
        """Execute all benchmarks in a run."""
        # Get benchmarks for this run
        result = await db.execute(
            select(BenchmarkRunBenchmark).where(BenchmarkRunBenchmark.run_id == run.id)
        )
        run_benchmarks = list(result.scalars().all())

        total_score = 0.0
        completed_benchmarks = 0
        client = await self._get_client_for_run(db, run)

        try:
            for rb in run_benchmarks:
                # Check for cancellation
                await db.refresh(run)
                if run.canceled_at:
                    logger.info("Run canceled", run_id=run.id)
                    await update_run_status(db, run.id, RunStatus.CANCELED)
                    return

                try:
                    score = await self.execute_benchmark(db, run, rb, client)
                    if score is not None:
                        total_score += score
                        completed_benchmarks += 1
                except Exception as e:
                    logger.error(
                        "Benchmark failed",
                        run_id=run.id,
                        benchmark=rb.benchmark_name,
                        error=str(e),
                    )
                    await update_benchmark_status(
                        db, rb.id, BenchmarkRunStatus.FAILED,
                        error_message=str(e)
                    )
                    await add_run_event(
                        db, run.id, "benchmark_failed",
                        benchmark_name=rb.benchmark_name,
                        message=f"Benchmark failed: {str(e)}"
                    )
        finally:
            if client is not self.client:
                await client.close()

        # Compute overall score
        overall_score = total_score / completed_benchmarks if completed_benchmarks > 0 else None

        await update_run_status(
            db, run.id, RunStatus.SUCCEEDED,
            overall_score=overall_score
        )
        await add_run_event(
            db, run.id, "run_completed",
            message=f"Run completed with overall score: {overall_score:.2%}" if overall_score else "Run completed",
            data={"overall_score": overall_score, "completed_benchmarks": completed_benchmarks}
        )

        logger.info(
            "Run completed",
            run_id=run.id,
            overall_score=overall_score,
            completed=completed_benchmarks,
        )

    async def execute_benchmark(
        self,
        db: AsyncSession,
        run: BenchmarkRun,
        rb: BenchmarkRunBenchmark,
        client: ChutesClient,
    ) -> Optional[float]:
        """Execute a single benchmark."""
        logger.info("Starting benchmark", run_id=run.id, benchmark=rb.benchmark_name)

        # Get adapter
        adapter = get_adapter(rb.benchmark_name, client, run.model_slug)
        if not adapter:
            await update_benchmark_status(
                db, rb.id, BenchmarkRunStatus.SKIPPED,
                error_message=f"No adapter found for {rb.benchmark_name}"
            )
            return None

        # Check if setup is required
        if adapter.requires_setup():
            notes = adapter.get_setup_notes()
            # For now, still try to run - real implementation would check dependencies
            logger.warning(
                "Benchmark requires setup",
                benchmark=rb.benchmark_name,
                notes=notes,
            )

        await update_benchmark_status(db, rb.id, BenchmarkRunStatus.RUNNING)
        await add_run_event(
            db, run.id, "benchmark_started",
            benchmark_name=rb.benchmark_name,
            message=f"Starting {adapter.get_display_name()}"
        )

        try:
            # Get items and apply subset
            seed = f"{run.id}_{rb.benchmark_name}"
            total_items, items_to_evaluate = await adapter.get_items_for_evaluation(run.subset_pct, seed)

            if total_items <= 0 or not items_to_evaluate:
                await update_benchmark_status(
                    db, rb.id, BenchmarkRunStatus.NEEDS_SETUP,
                    error_message="No items found - dataset may require setup"
                )
                return None

            await db.execute(
                update(Benchmark)
                .where(Benchmark.id == rb.benchmark_id)
                .values(total_items=total_items)
            )

            await update_benchmark_status(
                db, rb.id, BenchmarkRunStatus.RUNNING,
                total_items=total_items,
                sampled_items=len(items_to_evaluate),
                sampled_item_ids=items_to_evaluate[:1000],  # Store up to 1000 IDs
            )

            # Evaluate items
            results: list[ItemResult] = []
            correct = 0

            for i, item_id in enumerate(items_to_evaluate):
                # Check for cancellation periodically
                if i % 10 == 0:
                    await db.refresh(run)
                    if run.canceled_at:
                        raise Exception("Run canceled")

                result = await adapter.evaluate_item(item_id)
                results.append(result)

                # Save result
                await save_item_result(
                    db, rb.id,
                    item_id=result.item_id,
                    item_hash=result.item_hash,
                    prompt=result.prompt,
                    response=result.response,
                    expected=result.expected,
                    is_correct=result.is_correct,
                    score=result.score,
                    judge_output=result.judge_output,
                    latency_ms=result.latency_ms,
                    input_tokens=result.input_tokens,
                    output_tokens=result.output_tokens,
                    error=result.error,
                    test_code=result.test_code,
                    item_metadata=result.metadata,
                )

                if result.is_correct:
                    correct += 1

                # Update progress
                await update_benchmark_status(
                    db, rb.id, BenchmarkRunStatus.RUNNING,
                    completed_items=i + 1,
                )

                # Emit progress event every 5 items
                if (i + 1) % 5 == 0 or i == len(items_to_evaluate) - 1:
                    await add_run_event(
                        db, run.id, "benchmark_progress",
                        benchmark_name=rb.benchmark_name,
                        message=f"Progress: {i + 1}/{len(items_to_evaluate)}",
                        data={
                            "completed": i + 1,
                            "total": len(items_to_evaluate),
                            "current_accuracy": correct / (i + 1) if i >= 0 else 0,
                        }
                    )

            # Compute final metrics
            score = correct / len(results) if results else 0.0
            additional_metrics = await adapter.postprocess(results)

            metrics = {
                "accuracy": score,
                "total_items": len(all_items),
                "sampled_items": len(items_to_evaluate),
                "sampled_pct": run.subset_pct,
                "correct": correct,
                **additional_metrics,
            }

            await update_benchmark_status(
                db, rb.id, BenchmarkRunStatus.SUCCEEDED,
                score=score,
                metrics=metrics,
                completed_items=len(results),
            )

            await add_run_event(
                db, run.id, "benchmark_completed",
                benchmark_name=rb.benchmark_name,
                message=f"Completed {adapter.get_display_name()} with score {score:.2%}",
                data=metrics,
            )

            logger.info(
                "Benchmark completed",
                run_id=run.id,
                benchmark=rb.benchmark_name,
                score=score,
                items=len(results),
            )

            return score

        except Exception as e:
            raise
        finally:
            try:
                await adapter.cleanup()
            except Exception as cleanup_error:
                logger.warning(
                    "Adapter cleanup failed",
                    benchmark=rb.benchmark_name,
                    error=str(cleanup_error),
                )


async def run_worker() -> None:
    """Entry point for worker process."""
    from app.core.logging import setup_logging
    setup_logging()

    worker = BenchmarkWorker()

    try:
        await worker.start()
    except KeyboardInterrupt:
        await worker.stop()


if __name__ == "__main__":
    asyncio.run(run_worker())
