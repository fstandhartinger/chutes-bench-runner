"""Background worker for running benchmarks."""
import asyncio
import time
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import noload

from app.benchmarks import get_adapter
from app.benchmarks.base import BenchmarkAdapter, ItemResult
from app.core.config import get_settings
from app.core.logging import get_logger
from app.db.session import async_session_maker
from app.models.benchmark import Benchmark
from app.models.run import (
    BenchmarkRun,
    BenchmarkRunBenchmark,
    BenchmarkItemResult,
    BenchmarkRunStatus,
    RunStatus,
)
from app.services import auth_service
from app.services.chutes_client import ChutesClient, get_chutes_client
from app.services.run_service import (
    add_run_event,
    get_run,
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
        self.current_run_ids: set[str] = set()
        self.run_tasks: dict[str, asyncio.Task] = {}
        self.last_progress_at: dict[str, datetime] = {}
        self._benchmark_timeout_cache: dict[str, int] = {}
        self.client = get_chutes_client()
        self._last_stale_check = 0.0
        self._last_heartbeat = 0.0

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

    def _get_benchmark_timeout(self, benchmark_name: str, model_slug: str) -> int:
        cached = self._benchmark_timeout_cache.get(benchmark_name)
        if cached is not None:
            return cached

        adapter = get_adapter(benchmark_name, self.client, model_slug)
        timeout = adapter.get_item_timeout_seconds() if adapter else None
        if not timeout or timeout <= 0:
            timeout = settings.worker_item_timeout_seconds
        self._benchmark_timeout_cache[benchmark_name] = timeout
        return timeout

    async def start(self) -> None:
        """Start the worker loop."""
        self.running = True
        logger.info("Worker started")

        while self.running:
            try:
                now = time.monotonic()
                if now - self._last_stale_check >= settings.worker_stale_check_interval:
                    await self.requeue_stale_runs()
                    self._last_stale_check = now
                if now - self._last_heartbeat >= settings.worker_heartbeat_seconds:
                    await self.touch_active_runs()
                    self._last_heartbeat = now
                await self.reap_completed_runs()
                await self.launch_runs()
            except Exception:
                logger.exception("Worker error")

            await asyncio.sleep(settings.worker_poll_interval)

    async def stop(self) -> None:
        """Stop the worker."""
        self.running = False
        logger.info("Worker stopping")

    async def reap_completed_runs(self) -> None:
        """Remove completed tasks from tracking."""
        completed_run_ids = [run_id for run_id, task in self.run_tasks.items() if task.done()]
        for run_id in completed_run_ids:
            task = self.run_tasks.pop(run_id, None)
            self.current_run_ids.discard(run_id)
            self.last_progress_at.pop(run_id, None)
            if task:
                try:
                    task.result()
                except asyncio.CancelledError:
                    logger.warning("Run task canceled", run_id=run_id)
                except Exception:
                    logger.exception("Run task failed", run_id=run_id)

    async def launch_runs(self) -> None:
        """Launch new runs up to the concurrency limit."""
        while len(self.run_tasks) < settings.worker_max_concurrent:
            claimed = await self.claim_next_run()
            if not claimed:
                break

    async def claim_next_run(self) -> bool:
        """
        Claim the next queued run and launch it.
        
        Uses SKIP LOCKED to prevent multiple workers from claiming same run.
        
        Returns:
            True if a run was claimed, False otherwise
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

            self.current_run_ids.add(run.id)
            model_slug = run.model_slug
            was_started = run.started_at is not None
            self.last_progress_at[run.id] = datetime.utcnow()
            logger.info("Claimed run", run_id=run.id, model=model_slug)

            # Update status to running
            await update_run_status(db, run.id, RunStatus.RUNNING)
            event_type = "run_resumed" if was_started else "run_started"
            message = (
                f"Resuming benchmark run for {model_slug}"
                if was_started
                else f"Starting benchmark run for {model_slug}"
            )
            await add_run_event(db, run.id, event_type, message=message)
            task = asyncio.create_task(self.execute_claimed_run(run.id, model_slug))
            self.run_tasks[run.id] = task

            return True

    async def execute_claimed_run(self, run_id: str, model_slug: str) -> None:
        """Execute a claimed run in its own session."""
        async with async_session_maker() as db:
            run = await get_run(db, run_id)
            if not run:
                logger.warning("Run not found after claim", run_id=run_id)
                return

            try:
                await self.execute_run(db, run)
            except asyncio.CancelledError:
                logger.warning("Run execution canceled", run_id=run_id)
                raise
            except Exception as e:
                logger.exception("Run failed", run_id=run_id)
                await update_run_status(db, run_id, RunStatus.FAILED, error_message=str(e))
                await add_run_event(
                    db,
                    run_id,
                    "run_failed",
                    message=f"Run failed: {str(e)}",
                )

    async def touch_active_runs(self) -> None:
        """Touch all active runs for this worker to avoid stale requeue."""
        return

    async def requeue_stale_runs(self) -> None:
        """Requeue stale running runs after a worker restart or stall."""
        async with async_session_maker() as db:
            result = await db.execute(
                select(BenchmarkRun)
                .options(noload(BenchmarkRun.model))
                .where(BenchmarkRun.status == RunStatus.RUNNING.value)
            )
            running_runs = list(result.scalars().all())
            if not running_runs:
                return

            run_ids = [run.id for run in running_runs]
            benchmark_result = await db.execute(
                select(
                    BenchmarkRunBenchmark.run_id,
                    BenchmarkRunBenchmark.benchmark_name,
                    BenchmarkRunBenchmark.status,
                    BenchmarkRunBenchmark.updated_at,
                ).where(BenchmarkRunBenchmark.run_id.in_(run_ids))
            )
            benchmarks_by_run: dict[str, list[tuple[str, Optional[datetime]]]] = {}
            for run_id, benchmark_name, status, updated_at in benchmark_result.all():
                if status in (
                    BenchmarkRunStatus.SUCCEEDED.value,
                    BenchmarkRunStatus.FAILED.value,
                    BenchmarkRunStatus.SKIPPED.value,
                ):
                    continue
                benchmarks_by_run.setdefault(run_id, []).append((benchmark_name, updated_at))

            now = datetime.utcnow()
            base_seconds = settings.worker_stale_run_minutes * 60
            buffer_seconds = max(settings.worker_heartbeat_seconds * 2, 60)
            for run in running_runs:
                benchmark_entries = benchmarks_by_run.get(run.id, [])
                max_timeout = max(settings.worker_item_timeout_seconds, 0)
                benchmark_updated_at: list[datetime] = []
                for benchmark_name, benchmark_updated in benchmark_entries:
                    max_timeout = max(
                        max_timeout,
                        self._get_benchmark_timeout(benchmark_name, run.model_slug),
                    )
                    if benchmark_updated:
                        benchmark_updated_at.append(benchmark_updated)
                stale_after_seconds = max(base_seconds, max_timeout + buffer_seconds)
                cutoff = now - timedelta(seconds=stale_after_seconds)
                last_update_candidates = list(benchmark_updated_at)
                if run.updated_at:
                    last_update_candidates.append(run.updated_at)
                elif run.started_at:
                    last_update_candidates.append(run.started_at)
                else:
                    last_update_candidates.append(run.created_at)
                last_update = max(last_update_candidates) if last_update_candidates else None
                if last_update and last_update >= cutoff:
                    continue

                if run.id in self.current_run_ids:
                    last_progress = self.last_progress_at.get(run.id)
                    if last_progress and last_progress >= cutoff:
                        continue
                    task = self.run_tasks.pop(run.id, None)
                    if task:
                        task.cancel()
                        await asyncio.gather(task, return_exceptions=True)
                    self.current_run_ids.discard(run.id)
                    self.last_progress_at.pop(run.id, None)
                logger.warning(
                    "Requeuing stale run",
                    run_id=run.id,
                    updated_at=run.updated_at,
                )
                await db.execute(
                    update(BenchmarkRunBenchmark)
                    .where(BenchmarkRunBenchmark.run_id == run.id)
                    .where(
                        BenchmarkRunBenchmark.status.in_(
                            [BenchmarkRunStatus.RUNNING.value]
                        )
                    )
                    .values(
                        status=BenchmarkRunStatus.PENDING.value,
                        error_message=None,
                        updated_at=now,
                    )
                )
                await db.execute(
                    update(BenchmarkRun)
                    .where(BenchmarkRun.id == run.id)
                    .values(
                        status=RunStatus.QUEUED.value,
                        error_message=None,
                        updated_at=now,
                    )
                )
                await db.commit()
                await add_run_event(
                    db,
                    run.id,
                    "run_requeued",
                    message="Run requeued after worker restart or stall detected",
                )

    async def execute_run(self, db: AsyncSession, run: BenchmarkRun) -> None:
        """Execute all benchmarks in a run."""
        # Get benchmarks for this run
        result = await db.execute(
            select(BenchmarkRunBenchmark).where(BenchmarkRunBenchmark.run_id == run.id)
        )
        run_benchmarks = list(result.scalars().all())

        total_score = 0.0
        completed_benchmarks = 0
        failed_benchmarks = 0
        client = await self._get_client_for_run(db, run)

        try:
            for rb in run_benchmarks:
                # Check for cancellation
                await db.refresh(run)
                if run.canceled_at:
                    logger.info("Run canceled", run_id=run.id)
                    await update_run_status(db, run.id, RunStatus.CANCELED)
                    return

                if rb.status == BenchmarkRunStatus.SUCCEEDED.value:
                    if rb.score is not None:
                        total_score += rb.score
                        completed_benchmarks += 1
                    continue

                if rb.status in (BenchmarkRunStatus.FAILED.value, BenchmarkRunStatus.SKIPPED.value):
                    failed_benchmarks += 1
                    continue

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
                    failed_benchmarks += 1
        finally:
            if client is not self.client:
                await client.close()

        # Compute overall score
        if completed_benchmarks == 0 and failed_benchmarks > 0:
            await update_run_status(
                db,
                run.id,
                RunStatus.FAILED,
                error_message="All benchmarks failed",
            )
            await add_run_event(
                db,
                run.id,
                "run_failed",
                message="Run failed: all benchmarks failed",
                data={"failed_benchmarks": failed_benchmarks},
            )
            logger.error("Run failed", run_id=run.id, failed=failed_benchmarks)
            return

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

            sampled_item_ids = list(rb.sampled_item_ids or [])
            if sampled_item_ids and len(sampled_item_ids) == len(items_to_evaluate):
                items_to_evaluate = list(sampled_item_ids)
            else:
                sampled_item_ids = list(items_to_evaluate)

            needs_postprocess = adapter.__class__.postprocess is not BenchmarkAdapter.postprocess
            existing_results: list[ItemResult] = []

            if needs_postprocess:
                result = await db.execute(
                    select(BenchmarkItemResult)
                    .where(BenchmarkItemResult.run_benchmark_id == rb.id)
                )
                existing_rows = list(result.scalars().all())
                completed_item_ids = {row.item_id for row in existing_rows}
                correct = sum(1 for row in existing_rows if row.is_correct)
                existing_results = [
                    ItemResult(
                        item_id=row.item_id,
                        item_hash=row.item_hash,
                        prompt=row.prompt,
                        response=row.response,
                        expected=row.expected,
                        is_correct=row.is_correct,
                        score=row.score,
                        judge_output=row.judge_output,
                        latency_ms=row.latency_ms,
                        input_tokens=row.input_tokens,
                        output_tokens=row.output_tokens,
                        error=row.error,
                        test_code=row.test_code,
                        metadata=row.item_metadata,
                    )
                    for row in existing_rows
                ]
            else:
                result = await db.execute(
                    select(BenchmarkItemResult.item_id, BenchmarkItemResult.is_correct)
                    .where(BenchmarkItemResult.run_benchmark_id == rb.id)
                )
                existing_rows = list(result.all())
                completed_item_ids = {row.item_id for row in existing_rows}
                correct = sum(1 for row in existing_rows if row.is_correct)

            completed_items = len(completed_item_ids)
            pending_item_ids = [item_id for item_id in items_to_evaluate if item_id not in completed_item_ids]

            await update_benchmark_status(
                db, rb.id, BenchmarkRunStatus.RUNNING,
                total_items=total_items,
                sampled_items=len(items_to_evaluate),
                sampled_item_ids=sampled_item_ids,
                completed_items=completed_items,
            )

            event_type = "benchmark_resumed" if completed_items else "benchmark_started"
            message = (
                f"Resuming {adapter.get_display_name()} ({completed_items}/{len(items_to_evaluate)})"
                if completed_items
                else f"Starting {adapter.get_display_name()}"
            )
            await add_run_event(
                db, run.id, event_type,
                benchmark_name=rb.benchmark_name,
                message=message,
                data={
                    "completed": completed_items,
                    "total": len(items_to_evaluate),
                },
            )

            if not pending_item_ids:
                score = correct / len(items_to_evaluate) if items_to_evaluate else 0.0
                additional_metrics = await adapter.postprocess(existing_results) if needs_postprocess else {}
                metrics = {
                    "accuracy": score,
                    "total_items": total_items,
                    "sampled_items": len(items_to_evaluate),
                    "sampled_pct": run.subset_pct,
                    "correct": correct,
                    **additional_metrics,
                }
                await update_benchmark_status(
                    db, rb.id, BenchmarkRunStatus.SUCCEEDED,
                    score=score,
                    metrics=metrics,
                    completed_items=completed_items,
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
                    items=completed_items,
                )
                return score

            # Evaluate items
            new_results: list[ItemResult] = []
            max_concurrency = (
                settings.worker_item_concurrency
                if adapter.supports_parallel_items()
                else 1
            )
            item_timeout = adapter.get_item_timeout_seconds()
            if item_timeout is None:
                item_timeout = settings.worker_item_timeout_seconds
            if item_timeout is not None and item_timeout <= 0:
                item_timeout = None

            result_lock = asyncio.Lock()

            async def evaluate_item(item_id: str) -> ItemResult:
                try:
                    if item_timeout:
                        return await asyncio.wait_for(
                            adapter.evaluate_item(item_id),
                            timeout=item_timeout,
                        )
                    return await adapter.evaluate_item(item_id)
                except asyncio.TimeoutError:
                    return ItemResult(item_id=item_id, error=f"Item evaluation timed out after {item_timeout}s")

            async def record_result(result: ItemResult) -> None:
                nonlocal correct, completed_items

                async with result_lock:
                    if needs_postprocess:
                        new_results.append(result)

                    if result.is_correct:
                        correct += 1

                    completed_items += 1
                    current_completed = completed_items
                    current_correct = correct
                    self.last_progress_at[run.id] = datetime.utcnow()

                async with async_session_maker() as item_db:
                    await save_item_result(
                        item_db, rb.id,
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

                    await update_benchmark_status(
                        item_db, rb.id, BenchmarkRunStatus.RUNNING,
                        completed_items=current_completed,
                    )

                    if current_completed % 5 == 0 or current_completed == len(items_to_evaluate):
                        await add_run_event(
                            item_db, run.id, "benchmark_progress",
                            benchmark_name=rb.benchmark_name,
                            message=f"Progress: {current_completed}/{len(items_to_evaluate)}",
                            data={
                                "completed": current_completed,
                                "total": len(items_to_evaluate),
                                "current_accuracy": current_correct / current_completed if current_completed else 0,
                            }
                        )

            if max_concurrency > 1 and len(pending_item_ids) > 1:
                if db.in_transaction():
                    await db.commit()

                semaphore = asyncio.Semaphore(max_concurrency)

                async def run_item(item_id: str) -> ItemResult:
                    async with semaphore:
                        return await evaluate_item(item_id)

                pending_iter = iter(pending_item_ids)
                pending_tasks: set[asyncio.Task[ItemResult]] = set()

                async def schedule_next() -> bool:
                    try:
                        item_id = next(pending_iter)
                    except StopIteration:
                        return False
                    pending_tasks.add(asyncio.create_task(run_item(item_id)))
                    return True

                for _ in range(min(max_concurrency, len(pending_item_ids))):
                    await schedule_next()

                completed_since_check = 0
                while pending_tasks:
                    done, _ = await asyncio.wait(
                        pending_tasks, return_when=asyncio.FIRST_COMPLETED
                    )
                    for task in done:
                        pending_tasks.discard(task)
                        result = await task
                        await record_result(result)
                        completed_since_check += 1
                        if completed_since_check % 10 == 0:
                            await db.refresh(run)
                            if run.canceled_at:
                                for pending in pending_tasks:
                                    pending.cancel()
                                await asyncio.gather(*pending_tasks, return_exceptions=True)
                                raise Exception("Run canceled")
                        await schedule_next()
            else:
                for i, item_id in enumerate(pending_item_ids):
                    if i % 10 == 0:
                        await db.refresh(run)
                        if run.canceled_at:
                            raise Exception("Run canceled")
                    if db.in_transaction():
                        await db.commit()

                    result = await evaluate_item(item_id)
                    await record_result(result)

            all_results = existing_results + new_results if needs_postprocess else []
            score = correct / len(items_to_evaluate) if items_to_evaluate else 0.0
            additional_metrics = await adapter.postprocess(all_results) if needs_postprocess else {}

            metrics = {
                "accuracy": score,
                "total_items": total_items,
                "sampled_items": len(items_to_evaluate),
                "sampled_pct": run.subset_pct,
                "correct": correct,
                **additional_metrics,
            }

            await update_benchmark_status(
                db, rb.id, BenchmarkRunStatus.SUCCEEDED,
                score=score,
                metrics=metrics,
                completed_items=completed_items,
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
                items=completed_items,
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
