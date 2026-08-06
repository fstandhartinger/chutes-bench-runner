"""Background worker for running benchmarks."""
import asyncio
import os
import socket
import sys
import time
from datetime import datetime, timedelta
from typing import Any, Optional

from sqlalchemy import delete, func, select, update
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import noload

from app.benchmarks import get_adapter
from app.benchmarks.base import BenchmarkAdapter, ItemResult
from app.core.config import get_settings
from app.core.logging import get_logger
from app.db.session import async_session_maker
from app.models.benchmark import Benchmark
from app.models.model import Model
from app.models.run import (
    BenchmarkRun,
    BenchmarkRunBenchmark,
    BenchmarkItemResult,
    BenchmarkRunStatus,
    RunStatus,
)
from app.services import auth_service
from app.services.chutes_client import get_chutes_client
from app.services.janus_client import get_janus_client
from app.services.gremium_client import GremiumClient
from app.services.rlm_client import RLMClient
from app.services.inference_client import InferenceClient
from app.services.run_service import (
    add_run_event,
    get_run,
    save_item_result,
    update_benchmark_status,
    update_run_status,
)
from app.services.worker_service import record_worker_heartbeat
from app.worker.watchdog import WATCHDOG_EXIT_CODE, LoopWatchdog

logger = get_logger(__name__)
settings = get_settings()

PROGRESS_PERSIST_INTERVAL = 5
ITEM_TIMEOUT_EXCLUSION_REASON = "infrastructure_item_timeout"
ITEM_TIMEOUT_ERROR_PREFIX = "Item evaluation timed out"


class _RunnerItemTimeoutError(TimeoutError):
    """The worker's outer item deadline expired, not an adapter timeout."""


async def _evaluate_adapter_item(
    adapter: BenchmarkAdapter,
    item_id: str,
    timeout_seconds: Optional[float],
) -> ItemResult:
    """Evaluate an item while distinguishing the outer cap from inner timeouts."""
    if timeout_seconds is None:
        return await adapter.evaluate_item(item_id)

    task = asyncio.create_task(adapter.evaluate_item(item_id))
    done, _ = await asyncio.wait({task}, timeout=timeout_seconds)
    if task in done:
        return task.result()

    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    raise _RunnerItemTimeoutError


def _item_timeout_result(item_id: str, timeout_seconds: Optional[int]) -> ItemResult:
    error = ITEM_TIMEOUT_ERROR_PREFIX
    if timeout_seconds:
        error = f"{error} after {timeout_seconds}s"
    return ItemResult(
        item_id=item_id,
        error=error,
        metadata={
            "exclusion_reason": ITEM_TIMEOUT_EXCLUSION_REASON,
            "worker_item_timeout_seconds": timeout_seconds,
        },
    )


def _is_retryable_item_error(error: Optional[str]) -> bool:
    if not error:
        return False
    message = error.lower()
    # The worker's outer cap is deterministic for this item. Retrying under
    # the same cap cannot succeed and only adds backoff/cancellation churn.
    if message.startswith(ITEM_TIMEOUT_ERROR_PREFIX.lower()):
        return False
    if "currently disabled" in message or ("chute" in message and "disabled" in message):
        return False
    if "http 402" in message or "zero balance" in message:
        return False
    # Sandy sometimes returns transient 503s while the sandbox cluster is under load.
    # Retry the current item a few times; if it still fails, we abort the benchmark
    # (see _is_fatal_item_error) so we don't burn through the entire item set.
    if "all upstreams failed to create sandbox" in message:
        return True
    if "sandy api key is not configured" in message:
        return False
    if "sandbox not found" in message or "sandbox missing" in message:
        return True
    if "sandbox expired" in message or "sandbox terminated" in message:
        return True
    if "all connection attempts failed" in message:
        return True
    if "failed to establish a new connection" in message:
        return True
    if "connection refused" in message:
        return True
    if "sandbox" in message and "preempt" in message:
        return True
    if "http 404" in message and ("sandbox" in message or "sandy" in message):
        return True
    if "network error contacting chutes" in message:
        return True
    if "timeout" in message or "timed out" in message:
        return True
    if "http 429" in message:
        return True
    if "http 5" in message:
        return True
    if "empty response" in message:
        return True
    if "response truncated" in message:
        return True
    if "no instances available" in message:
        return True
    if "service unavailable" in message or "temporarily unavailable" in message:
        return True
    if "connection reset" in message or "connection aborted" in message:
        return True
    if "sandbox" in message and (
        "could not create" in message
        or "failed to create" in message
        or "failed to write file" in message
        or "failed to execute command" in message
        or "failed to terminate" in message
    ):
        return True
    return False


def _is_fatal_item_error(error: Optional[str]) -> bool:
    if not error:
        return False
    message = error.lower()
    if "currently disabled" in message or ("chute" in message and "disabled" in message):
        return True
    if "all upstreams failed to create sandbox" in message:
        return True
    if "sandy api key is not configured" in message:
        return True
    if "all connection attempts failed" in message:
        return True
    if "failed to establish a new connection" in message:
        return True
    if "connection refused" in message:
        return True
    if "model not found" in message or "no such model" in message or "http 404" in message:
        if "sandbox" in message or "sandy" in message:
            return False
        return True
    if "http 401" in message or "http 403" in message:
        return True
    # HTTP 402 = zero balance / payment required — model creator has no credits.
    # Retrying is pointless; fail fast so worker capacity is freed for other runs.
    if "http 402" in message or "zero balance" in message:
        return True
    if "unauthorized" in message or "forbidden" in message:
        return True
    if "invalid api key" in message or "invalid api-key" in message:
        return True
    return False


def _is_run_retryable(error: Optional[str]) -> bool:
    """Determine whether a failed *run* should be auto-retried.

    Returns True for transient infrastructure errors (sandbox failures,
    service unavailability, timeouts).  Returns False for permanent errors
    (model not found, auth failures, zero balance) where retrying is futile.
    """
    if not error:
        return True  # no error detail → assume transient
    message = error.lower()
    # --- Permanent / fatal errors — never retry ---
    if "model not found" in message or "no such model" in message:
        return False
    if "http 401" in message or "http 403" in message:
        return False
    if "http 402" in message or "zero balance" in message:
        return False
    if "unauthorized" in message or "forbidden" in message:
        return False
    if "invalid api key" in message or "invalid api-key" in message:
        return False
    if "invalid token" in message:
        return False
    if "currently disabled" in message or ("chute" in message and "disabled" in message):
        return False
    if "sandy api key is not configured" in message:
        return False
    # --- Everything else is considered transient → retry ---
    return True


def _apply_error_score_defaults(result: ItemResult) -> ItemResult:
    exclusion_reason = (result.metadata or {}).get("exclusion_reason")
    if result.error and result.score is None and not exclusion_reason:
        result.score = 0.0
    return result


def _is_excluded_item(result: ItemResult) -> bool:
    return bool((result.metadata or {}).get("exclusion_reason"))


def _accuracy_excluding_infrastructure(
    correct_items: int,
    sampled_items: int,
    excluded_items: int,
) -> Optional[float]:
    scored_items = max(0, sampled_items - excluded_items)
    if not scored_items:
        return None
    return correct_items / scored_items


def _compute_run_stale_seconds(
    base_seconds: int,
    max_timeout: int,
    has_started_work: bool,
) -> int:
    if has_started_work and max_timeout:
        return max(base_seconds, max_timeout)
    return base_seconds


def _is_retryable_db_write_error(exc: Exception) -> bool:
    if isinstance(exc, DBAPIError) and getattr(exc, "connection_invalidated", False):
        return True
    message = str(exc).lower()
    return (
        "deadlock" in message
        or "connection was closed in the middle of operation" in message
        or "connectiondoesnotexisterror" in message
        or "authentication timed out" in message
    )


async def _try_transition_stale_run(
    db: AsyncSession,
    run_id: str,
    expected_updated_at: Optional[datetime],
    values: dict[str, Any],
) -> bool:
    conditions = [
        BenchmarkRun.id == run_id,
        BenchmarkRun.status == RunStatus.RUNNING.value,
    ]
    if expected_updated_at is None:
        conditions.append(BenchmarkRun.updated_at.is_(None))
    else:
        conditions.append(BenchmarkRun.updated_at == expected_updated_at)
    result = await db.execute(
        update(BenchmarkRun)
        .where(*conditions)
        .values(**values)
        .returning(BenchmarkRun.id)
    )
    return result.scalar_one_or_none() is not None


def _load_direct_adapter(
    name: str,
    client: InferenceClient,
    model_slug: str,
    judge_client: Optional[InferenceClient] = None,
) -> Optional[BenchmarkAdapter]:
    """
    Load long-context benchmark adapters directly.
    
    This provides a fallback for the new long-context benchmarks in case
    the registry didn't load them properly.
    """
    adapter_imports = {
        "s_niah": ("app.benchmarks.adapters.s_niah", "SNIAHAdapter"),
        "oolong": ("app.benchmarks.adapters.oolong", "OolongAdapter"),
        "oolong_pairs": ("app.benchmarks.adapters.oolong_pairs", "OolongPairsAdapter"),
    }
    
    if name not in adapter_imports:
        return None
    
    module_name, class_name = adapter_imports[name]
    
    try:
        import importlib
        module = importlib.import_module(module_name)
        adapter_class = getattr(module, class_name)
        return adapter_class(client, model_slug, judge_client=judge_client)
    except Exception as exc:
        logger.warning(
            "Direct adapter import failed",
            adapter=name,
            module=module_name,
            error=str(exc) or exc.__class__.__name__,
        )
        return None


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
        # Tasks whose timeout expired and whose cancellation we refuse to await.
        # Held so they are not garbage collected mid-flight (and so we can count them).
        self._abandoned_ops: set[asyncio.Task] = set()
        self.watchdog: Optional[LoopWatchdog] = None
        self.worker_id = (
            os.getenv("SANDY_SANDBOX_ID")
            or os.getenv("WORKER_INSTANCE_ID")
            or socket.gethostname()
        )
        self.hostname = socket.gethostname()

    async def _is_run_canceled(self, run_id: str) -> bool:
        for attempt in range(3):
            try:
                async with async_session_maker() as db:
                    result = await db.execute(
                        select(BenchmarkRun.canceled_at).where(BenchmarkRun.id == run_id)
                    )
                    return result.scalar_one_or_none() is not None
            except Exception as exc:
                if _is_retryable_db_write_error(exc) and attempt < 2:
                    logger.warning(
                        "Transient DB error while checking cancellation, retrying",
                        run_id=run_id,
                        attempt=attempt + 1,
                        error=str(exc),
                    )
                    await asyncio.sleep(0.25 * (attempt + 1))
                    continue
                logger.warning(
                    "Failed to check cancellation state; assuming run is active",
                    run_id=run_id,
                    error=str(exc),
                )
                return False
        return False

    async def _safe_update_run_status(
        self,
        run_id: str,
        status: RunStatus,
        error_message: Optional[str] = None,
        overall_score: Optional[float] = None,
    ) -> None:
        async with async_session_maker() as db:
            await update_run_status(
                db,
                run_id,
                status,
                error_message=error_message,
                overall_score=overall_score,
            )

    async def _safe_update_benchmark_status(
        self,
        run_benchmark_id: str,
        status: BenchmarkRunStatus,
        metrics: Optional[dict[str, Any]] = None,
        score: Optional[float] = None,
        error_message: Optional[str] = None,
        completed_items: Optional[int] = None,
        total_items: Optional[int] = None,
        sampled_items: Optional[int] = None,
        sampled_item_ids: Optional[list[str]] = None,
    ) -> None:
        for attempt in range(3):
            try:
                async with async_session_maker() as db:
                    await update_benchmark_status(
                        db,
                        run_benchmark_id,
                        status,
                        metrics=metrics,
                        score=score,
                        error_message=error_message,
                        completed_items=completed_items,
                        total_items=total_items,
                        sampled_items=sampled_items,
                        sampled_item_ids=sampled_item_ids,
                    )
                return
            except Exception as exc:
                if "deadlock" in str(exc).lower() and attempt < 2:
                    logger.warning("Deadlock on benchmark status update, retrying", attempt=attempt + 1)
                    await asyncio.sleep(0.5 * (attempt + 1))
                    continue
                raise

    async def _safe_add_run_event(
        self,
        run_id: str,
        event_type: str,
        benchmark_name: Optional[str] = None,
        message: Optional[str] = None,
        data: Optional[dict[str, Any]] = None,
    ) -> None:
        async with async_session_maker() as db:
            await add_run_event(
                db,
                run_id,
                event_type,
                benchmark_name=benchmark_name,
                message=message,
                data=data,
            )

    async def _persist_item_result_progress(
        self,
        run_id: str,
        run_benchmark_id: str,
        result: ItemResult,
        benchmark_name: str,
        current_completed: int,
        total_items: int,
        current_correct: int,
        should_persist_progress: bool,
    ) -> None:
        for attempt in range(3):
            try:
                async with async_session_maker() as item_db:
                    await save_item_result(
                        item_db, run_benchmark_id,
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

                    if should_persist_progress:
                        await update_benchmark_status(
                            item_db,
                            run_benchmark_id,
                            BenchmarkRunStatus.RUNNING,
                            completed_items=current_completed,
                        )
                    else:
                        # Save the item row even when we are between progress checkpoints.
                        await item_db.commit()
                break
            except Exception as exc:
                if _is_retryable_db_write_error(exc) and attempt < 2:
                    logger.warning(
                        "Transient DB error while persisting item progress, retrying",
                        run_id=run_id,
                        benchmark=benchmark_name,
                        attempt=attempt + 1,
                        error=str(exc),
                    )
                    await asyncio.sleep(0.5 * (attempt + 1))
                    continue
                raise

        if should_persist_progress:
            try:
                await self._safe_add_run_event(
                    run_id,
                    "benchmark_progress",
                    benchmark_name=benchmark_name,
                    message=f"Progress: {current_completed}/{total_items}",
                    data={
                        "completed": current_completed,
                        "total": total_items,
                        "current_accuracy": current_correct / current_completed if current_completed else 0,
                    },
                )
            except Exception as exc:
                logger.warning(
                    "Failed to persist benchmark progress event",
                    run_id=run_id,
                    benchmark=benchmark_name,
                    error=str(exc),
                )

    async def _fail_run_for_model_access(
        self,
        run: BenchmarkRun,
        run_benchmarks: list[BenchmarkRunBenchmark],
        message: str,
    ) -> None:
        for rb in run_benchmarks:
            if rb.status in (
                BenchmarkRunStatus.SUCCEEDED.value,
                BenchmarkRunStatus.FAILED.value,
                BenchmarkRunStatus.SKIPPED.value,
            ):
                continue
            await self._safe_update_benchmark_status(
                rb.id,
                BenchmarkRunStatus.FAILED,
                error_message=message,
            )
        await self._safe_update_run_status(
            run.id,
            RunStatus.FAILED,
            error_message=message,
        )
        await self._safe_add_run_event(
            run.id,
            "run_failed",
            message=message,
        )
        logger.error("Run failed", run_id=run.id, error=message)

    async def _get_client_for_run(self, db: AsyncSession, run: BenchmarkRun) -> InferenceClient:
        if run.provider and run.provider.startswith("gremium"):
            return GremiumClient(
                api_key=settings.gremium_api_key or settings.chutes_api_key,
                provider=run.provider,
                base_url=settings.gremium_api_base_url,
            )
        if run.provider == "rlm":
            return RLMClient(
                api_key=settings.rlm_api_key or settings.chutes_api_key,
                base_url=settings.rlm_api_base_url,
            )
        if run.provider == "janus":
            return get_janus_client(api_key=run.auth_api_key)
        if run.auth_mode == "idp":
            if not run.auth_session_id:
                raise RuntimeError("Run is missing Chutes session credentials")
            session = await auth_service.get_session(db, run.auth_session_id)
            if not session:
                raise RuntimeError("Chutes session not found for run")
            if not session.can_invoke_chutes():
                raise RuntimeError("Chutes session missing chutes:invoke scope")
            access_token = await auth_service.get_valid_access_token(db, session)
            if not access_token:
                raise RuntimeError("Chutes session expired or invalid")
            return get_chutes_client(user_access_token=access_token)

        if run.auth_mode == "api_key":
            if not run.auth_api_key:
                raise RuntimeError("Run is missing API key credentials")
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

    async def run_guarded(self, coro, timeout: float, label: str) -> bool:
        """Run a DB coroutine with a timeout that can never wedge the main loop.

        This deliberately does NOT use ``asyncio.wait_for``. On timeout,
        ``wait_for`` calls ``_cancel_and_wait()``, which awaits the cancelled
        task's completion *with no timeout of its own*. If that task cannot be
        cancelled -- which is exactly what happens when SQLAlchemy's
        ``connectors/asyncio.py::terminate()`` runs
        ``await_(asyncio.shield(self._terminate_graceful_close()))`` on a dead
        asyncpg connection -- then ``wait_for`` blocks forever and takes the whole
        single-task event loop with it. That is what killed this worker for 15
        days on 2026-07-21 (see docs/bench_runner_incident_2026_08_06.md).

        ``asyncio.wait`` returns as soon as the timeout expires and never awaits
        cancellation, so we can request the cancel and then *abandon* the task.
        An orphaned task leaks at worst one pool slot; a wedged loop kills the
        whole worker silently.

        Returns True if the coroutine completed within the timeout.
        """
        task = asyncio.ensure_future(coro)
        done, pending = await asyncio.wait({task}, timeout=timeout)

        if task in done:
            self._abandoned_ops.discard(task)
            try:
                task.result()
                return True
            except asyncio.CancelledError:
                logger.warning("Worker op canceled", op=label)
                return False
            except Exception:
                logger.exception("Worker op failed", op=label)
                return False

        # Timed out. Ask for cancellation but do NOT await it.
        task.cancel()
        self._abandoned_ops.add(task)
        task.add_done_callback(self._abandoned_ops.discard)
        logger.warning(
            "Worker op timed out and was abandoned",
            op=label,
            timeout=timeout,
            abandoned_ops=len(self._abandoned_ops),
        )
        return False

    async def start(self) -> None:
        """Start the worker loop."""
        self.running = True
        logger.info("Worker started")

        while self.running:
            if self.watchdog is not None:
                self.watchdog.tick("loop")
            try:
                now = time.monotonic()
                if now - self._last_stale_check >= settings.worker_stale_check_interval:
                    await self.run_guarded(
                        self.requeue_stale_runs(), timeout=30, label="requeue_stale_runs"
                    )
                    self._last_stale_check = now
                if now - self._last_heartbeat >= settings.worker_heartbeat_seconds:
                    await self.run_guarded(
                        self.touch_active_runs(), timeout=15, label="touch_active_runs"
                    )
                    await self.run_guarded(
                        self.touch_worker_heartbeat(), timeout=15, label="touch_worker_heartbeat"
                    )
                    self._last_heartbeat = now
                await self.reap_completed_runs()
                await self.run_guarded(self.launch_runs(), timeout=30, label="launch_runs")
            except Exception:
                logger.exception("Worker error")

            if self._abandoned_ops:
                # Abandoned tasks still hold their DB pool slots. The watchdog
                # cannot catch this on its own: the loop keeps ticking happily
                # while every DB op times out, so the worker would become a
                # zombie that is "alive" but does no work. Trip out explicitly
                # once too many have piled up and let the restart policy give us
                # a clean pool.
                logger.warning(
                    "Abandoned worker ops still pending",
                    count=len(self._abandoned_ops),
                )
                if len(self._abandoned_ops) >= settings.worker_max_abandoned_ops:
                    logger.error(
                        "Too many abandoned worker ops; exiting for a clean restart",
                        count=len(self._abandoned_ops),
                        limit=settings.worker_max_abandoned_ops,
                    )
                    sys.stderr.write(
                        f"FATAL: {len(self._abandoned_ops)} abandoned DB ops are holding "
                        f"pool slots (limit {settings.worker_max_abandoned_ops}). "
                        f"Exiting so the container restart policy can recover.\n"
                    )
                    sys.stderr.flush()
                    os._exit(WATCHDOG_EXIT_CODE)

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
            exclusive = [b for b in settings.worker_exclusive_benchmarks if b]
            active_exclusive: set[str] = set()
            if exclusive:
                active_result = await db.execute(
                    select(BenchmarkRunBenchmark.benchmark_name)
                    .join(BenchmarkRun, BenchmarkRunBenchmark.run_id == BenchmarkRun.id)
                    .where(BenchmarkRun.status == RunStatus.RUNNING.value)
                    .where(BenchmarkRunBenchmark.status == BenchmarkRunStatus.RUNNING.value)
                    .where(BenchmarkRunBenchmark.benchmark_name.in_(exclusive))
                )
                active_exclusive = {row[0] for row in active_result.all()}

            # Claim a queued run with row lock
            # Use noload to prevent JOIN that's incompatible with FOR UPDATE
            result = await db.execute(
                select(BenchmarkRun)
                .options(noload(BenchmarkRun.model))
                .where(BenchmarkRun.status == RunStatus.QUEUED.value)
                .where(
                    BenchmarkRun.auth_mode == settings.worker_only_auth_mode
                    if settings.worker_only_auth_mode
                    else True
                )
                .where(
                    BenchmarkRun.auth_api_key == settings.worker_only_api_key
                    if settings.worker_only_api_key
                    else True
                )
                .order_by(BenchmarkRun.created_at)
                .limit(10)
                .with_for_update(skip_locked=True)
            )
            runs = list(result.scalars().all())
            if not runs:
                return False

            run = None
            for candidate in runs:
                selected = candidate.selected_benchmarks or []
                if active_exclusive and any(b in active_exclusive for b in selected):
                    continue
                run = candidate
                break

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
                await self._safe_update_run_status(
                    run_id,
                    RunStatus.FAILED,
                    error_message=str(e),
                )
                await self._safe_add_run_event(
                    run_id,
                    "run_failed",
                    message=f"Run failed: {str(e)}",
                )

    async def touch_active_runs(self) -> None:
        """Touch all active runs for this worker to avoid stale requeue."""
        if not self.current_run_ids:
            return
        now = datetime.utcnow()
        async with async_session_maker() as db:
            result = await db.execute(
                select(
                    BenchmarkRunBenchmark.run_id,
                    BenchmarkRunBenchmark.benchmark_name,
                    BenchmarkRun.model_slug,
                )
                .join(BenchmarkRun, BenchmarkRunBenchmark.run_id == BenchmarkRun.id)
                .where(BenchmarkRunBenchmark.run_id.in_(self.current_run_ids))
                .where(BenchmarkRunBenchmark.status == BenchmarkRunStatus.RUNNING.value)
            )
            active_ids: list[str] = []
            for run_id, benchmark_name, model_slug in result.all():
                timeout = self._get_benchmark_timeout(benchmark_name, model_slug)
                threshold_seconds = max(timeout, settings.worker_stale_run_minutes * 60)
                last_progress = self.last_progress_at.get(run_id)
                if not last_progress or last_progress >= now - timedelta(seconds=threshold_seconds):
                    active_ids.append(run_id)

            if not active_ids:
                return

            await db.execute(
                update(BenchmarkRun)
                .where(BenchmarkRun.id.in_(active_ids))
                .values(updated_at=now)
            )
            await db.execute(
                update(BenchmarkRunBenchmark)
                .where(BenchmarkRunBenchmark.run_id.in_(active_ids))
                .where(BenchmarkRunBenchmark.status == BenchmarkRunStatus.RUNNING.value)
                .values(updated_at=now)
            )
            await db.commit()

    async def touch_worker_heartbeat(self) -> None:
        """Record worker liveness and capacity for ops monitoring."""
        async with async_session_maker() as db:
            await record_worker_heartbeat(
                db,
                worker_id=self.worker_id,
                hostname=self.hostname,
                running_runs=len(self.run_tasks),
                max_concurrent_runs=settings.worker_max_concurrent,
                item_concurrency=settings.worker_item_concurrency,
            )

    async def _preload_adapter(self, run_id: str, adapter: BenchmarkAdapter) -> None:
        """Preload adapter data while keeping the run heartbeat fresh."""
        if adapter.__class__.preload is BenchmarkAdapter.preload:
            return
        heartbeat_interval = max(settings.worker_heartbeat_seconds, 30)
        preload_task = asyncio.create_task(adapter.preload())
        while True:
            done, _ = await asyncio.wait({preload_task}, timeout=heartbeat_interval)
            self.last_progress_at[run_id] = datetime.utcnow()
            if preload_task in done:
                await preload_task
                return

    async def requeue_stale_runs(self) -> None:
        """Requeue stale running runs after a worker restart or stall."""
        async with async_session_maker() as db:
            inactive_result = await db.execute(
                select(BenchmarkRun.id, BenchmarkRun.status)
                .where(BenchmarkRun.status.in_([RunStatus.QUEUED.value, RunStatus.RUNNING.value]))
                .where(BenchmarkRun.updated_at > datetime.utcnow() - timedelta(hours=24))
            )
            inactive_runs = list(inactive_result.all())
            if inactive_runs:
                now = datetime.utcnow()
                queued_ids = [
                    run_id
                    for run_id, status in inactive_runs
                    if status == RunStatus.QUEUED.value
                ]
                skip_ids = [
                    run_id
                    for run_id, status in inactive_runs
                    if status
                    in (
                        RunStatus.CANCELED.value,
                        RunStatus.FAILED.value,
                        RunStatus.SUCCEEDED.value,
                    )
                ]
                if queued_ids:
                    await db.execute(
                        update(BenchmarkRunBenchmark)
                        .where(BenchmarkRunBenchmark.run_id.in_(queued_ids))
                        .where(BenchmarkRunBenchmark.status == BenchmarkRunStatus.RUNNING.value)
                        .values(
                            status=BenchmarkRunStatus.PENDING.value,
                            error_message=None,
                            started_at=None,
                            completed_at=None,
                            updated_at=now,
                        )
                    )
                if skip_ids:
                    await db.execute(
                        update(BenchmarkRunBenchmark)
                        .where(BenchmarkRunBenchmark.run_id.in_(skip_ids))
                        .where(BenchmarkRunBenchmark.status == BenchmarkRunStatus.RUNNING.value)
                        .values(
                            status=BenchmarkRunStatus.SKIPPED.value,
                            error_message="Run no longer active",
                            completed_at=now,
                            updated_at=now,
                        )
                    )
                if queued_ids or skip_ids:
                    await db.commit()

            result = await db.execute(
                select(
                    BenchmarkRun.id,
                    BenchmarkRun.updated_at,
                    BenchmarkRun.started_at,
                    BenchmarkRun.created_at,
                    BenchmarkRun.model_slug,
                    BenchmarkRun.retry_count,
                    BenchmarkRun.max_retries,
                )
                .where(BenchmarkRun.status == RunStatus.RUNNING.value)
            )
            running_runs = list(result.all())
            if not running_runs:
                return

            run_ids = [run_id for run_id, *_ in running_runs]
            benchmark_result = await db.execute(
                select(
                    BenchmarkRunBenchmark.run_id,
                    BenchmarkRunBenchmark.status,
                    BenchmarkRunBenchmark.started_at,
                    BenchmarkRunBenchmark.sampled_items,
                    BenchmarkRunBenchmark.benchmark_name,
                ).where(BenchmarkRunBenchmark.run_id.in_(run_ids))
            )
            benchmarks_by_run: dict[str, list[tuple[Optional[datetime], Optional[int], Optional[str]]]] = {}
            for run_id, status, started_at, sampled_items, benchmark_name in benchmark_result.all():
                if status in (
                    BenchmarkRunStatus.SUCCEEDED.value,
                    BenchmarkRunStatus.FAILED.value,
                    BenchmarkRunStatus.SKIPPED.value,
                ):
                    continue
                benchmarks_by_run.setdefault(run_id, []).append(
                    (started_at, sampled_items, benchmark_name)
                )

            now = datetime.utcnow()
            base_seconds = settings.worker_stale_run_minutes * 60
            buffer_seconds = max(settings.worker_heartbeat_seconds * 2, 60)
            stale_after_seconds = max(base_seconds, buffer_seconds)
            default_item_timeout = max(settings.worker_item_timeout_seconds, 0)
            item_result = await db.execute(
                select(
                    BenchmarkRunBenchmark.run_id,
                    func.max(BenchmarkItemResult.created_at),
                )
                .join(
                    BenchmarkItemResult,
                    BenchmarkItemResult.run_benchmark_id == BenchmarkRunBenchmark.id,
                )
                .where(BenchmarkRunBenchmark.run_id.in_(run_ids))
                .group_by(BenchmarkRunBenchmark.run_id)
            )
            last_item_by_run = {run_id: created_at for run_id, created_at in item_result.all()}
            for (
                run_id,
                run_updated_at,
                run_started_at,
                run_created_at,
                run_model_slug,
                run_retry_count,
                run_max_retries,
            ) in running_runs:
                benchmark_entries = benchmarks_by_run.get(run_id, [])
                last_item_at = last_item_by_run.get(run_id)
                last_update_candidates: list[datetime] = []
                if last_item_at:
                    last_update_candidates.append(last_item_at)
                if run_updated_at:
                    last_update_candidates.append(run_updated_at)
                if not last_item_at:
                    started_candidates: list[datetime] = [
                        started_at
                        for started_at, _, _ in benchmark_entries
                        if started_at
                    ]
                    if run_started_at:
                        started_candidates.append(run_started_at)
                    if started_candidates:
                        last_update_candidates.append(max(started_candidates))
                    else:
                        last_update_candidates.append(run_created_at)

                has_started_work = any(
                    started_at is not None for started_at, _, _ in benchmark_entries
                )
                has_sampled_items = any(
                    (sampled_items or 0) > 0 for _, sampled_items, _ in benchmark_entries
                )
                max_timeout = default_item_timeout
                if benchmark_entries:
                    timeouts: list[int] = []
                    for _, _, benchmark_name in benchmark_entries:
                        if benchmark_name:
                            timeout = self._get_benchmark_timeout(benchmark_name, run_model_slug)
                            if timeout:
                                timeouts.append(timeout)
                    if timeouts:
                        max_timeout = max(max_timeout, max(timeouts))
                run_stale_seconds = _compute_run_stale_seconds(
                    stale_after_seconds,
                    max_timeout,
                    has_started_work or has_sampled_items,
                )
                cutoff = now - timedelta(seconds=run_stale_seconds)
                last_update = max(last_update_candidates) if last_update_candidates else None
                if last_update and last_update >= cutoff:
                    continue

                if run_id in self.current_run_ids:
                    last_progress = self.last_progress_at.get(run_id)
                    if last_progress and last_progress >= cutoff:
                        continue
                    task = self.run_tasks.pop(run_id, None)
                    if task:
                        task.cancel()
                        await asyncio.gather(task, return_exceptions=True)
                    self.current_run_ids.discard(run_id)
                    self.last_progress_at.pop(run_id, None)
                retry_count = run_retry_count or 0
                max_retries = run_max_retries or 3
                new_retry = retry_count + 1

                if new_retry > max_retries:
                    # Exhausted retries — fail permanently instead of infinite requeue loop
                    logger.error(
                        "Failing stale run (retries exhausted)",
                        run_id=run_id,
                        retry_count=retry_count,
                        max_retries=max_retries,
                    )
                    claimed = await _try_transition_stale_run(
                        db,
                        run_id,
                        run_updated_at,
                        {
                            "status": RunStatus.FAILED.value,
                            "error_message": (
                                f"All benchmarks failed (stale requeue retries exhausted: {retry_count}/{max_retries})"
                            ),
                            "completed_at": now,
                            "updated_at": now,
                        },
                    )
                    if not claimed:
                        await db.rollback()
                        continue
                    await db.execute(
                        update(BenchmarkRunBenchmark)
                        .where(BenchmarkRunBenchmark.run_id == run_id)
                        .where(
                            BenchmarkRunBenchmark.status.in_(
                                [BenchmarkRunStatus.RUNNING.value, BenchmarkRunStatus.PENDING.value]
                            )
                        )
                        .values(
                            status=BenchmarkRunStatus.FAILED.value,
                            error_message="Run stalled and retries exhausted",
                            completed_at=now,
                            updated_at=now,
                        )
                    )
                    await db.commit()
                    await add_run_event(
                        db,
                        run_id,
                        "run_failed",
                        message=f"Run failed: stale requeue retries exhausted ({retry_count}/{max_retries})",
                    )
                else:
                    logger.warning(
                        "Requeuing stale run",
                        run_id=run_id,
                        updated_at=run_updated_at,
                        retry=new_retry,
                        max_retries=max_retries,
                    )
                    claimed = await _try_transition_stale_run(
                        db,
                        run_id,
                        run_updated_at,
                        {
                            "status": RunStatus.QUEUED.value,
                            "retry_count": new_retry,
                            "error_message": None,
                            "started_at": None,
                            "completed_at": None,
                            "updated_at": now,
                        },
                    )
                    if not claimed:
                        await db.rollback()
                        continue
                    await db.execute(
                        update(BenchmarkRunBenchmark)
                        .where(BenchmarkRunBenchmark.run_id == run_id)
                        .where(
                            BenchmarkRunBenchmark.status.in_(
                                [BenchmarkRunStatus.RUNNING.value]
                            )
                        )
                        .values(
                            status=BenchmarkRunStatus.PENDING.value,
                            error_message=None,
                            started_at=None,
                            completed_at=None,
                            updated_at=now,
                        )
                    )
                    await db.commit()
                    await add_run_event(
                        db,
                        run_id,
                        "run_requeued",
                        message=f"Stale run requeued (attempt {new_retry}/{max_retries})",
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
        judge_client: Optional[InferenceClient] = None
        if run.provider != "chutes":
            judge_client = get_chutes_client()

        try:
            if run.provider == "chutes":
                model = await db.get(Model, run.model_id)
                chute_id = model.chute_id if model else None
                available = await client.is_model_available(run.model_slug, chute_id)
                if available is False:
                    ok, status_code, detail = await client.probe_model_access(run.model_slug)
                    if ok:
                        available = True
                    elif status_code in (401, 403):
                        message = f"Chutes credentials are not authorized for {run.model_slug}"
                        await self._fail_run_for_model_access(run, run_benchmarks, message)
                        return
                    elif status_code == 402:
                        message = f"Chutes model {run.model_slug} is unavailable because the creator has zero balance"
                        await self._fail_run_for_model_access(run, run_benchmarks, message)
                        return
                    elif status_code == 404:
                        message = f"Model {run.model_slug} not found on Chutes"
                        await self._fail_run_for_model_access(run, run_benchmarks, message)
                        return
                    else:
                        logger.warning(
                            "Model availability check failed; continuing",
                            run_id=run.id,
                            model=run.model_slug,
                            status_code=status_code,
                            detail=detail,
                        )

            for rb in run_benchmarks:
                # Check for cancellation
                if await self._is_run_canceled(run.id):
                    logger.info("Run canceled", run_id=run.id)
                    await self._safe_update_run_status(run.id, RunStatus.CANCELED)
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
                    score = await self.execute_benchmark(db, run, rb, client, judge_client=judge_client)
                    if score is not None:
                        total_score += score
                        completed_benchmarks += 1
                except Exception as e:
                    error_detail = str(e)
                    if error_detail:
                        error_detail = f"{e.__class__.__name__}: {error_detail}"
                    else:
                        error_detail = repr(e)
                    logger.exception(
                        "Benchmark failed",
                        run_id=run.id,
                        benchmark=rb.benchmark_name,
                        error=error_detail,
                    )
                    await self._safe_update_benchmark_status(
                        rb.id,
                        BenchmarkRunStatus.FAILED,
                        error_message=error_detail,
                    )
                    await self._safe_add_run_event(
                        run.id,
                        "benchmark_failed",
                        benchmark_name=rb.benchmark_name,
                        message=f"Benchmark failed: {error_detail}",
                    )
                    failed_benchmarks += 1
        finally:
            if client is not self.client:
                await client.close()
            if judge_client:
                await judge_client.close()

        # Compute overall score
        if completed_benchmarks == 0 and failed_benchmarks > 0:
            # Collect all benchmark error messages to decide retryability
            all_errors: list[str] = []
            async with async_session_maker() as db:
                result = await db.execute(
                    select(BenchmarkRunBenchmark).where(BenchmarkRunBenchmark.run_id == run.id)
                )
                for rb_row in result.scalars():
                    if rb_row.error_message:
                        all_errors.append(rb_row.error_message)
            combined_error = " | ".join(all_errors) if all_errors else "All benchmarks failed"

            # Auto-retry: requeue if failure is transient and retries remain
            retry_count = getattr(run, "retry_count", 0) or 0
            max_retries = getattr(run, "max_retries", 3) or 3
            if retry_count < max_retries and _is_run_retryable(combined_error):
                new_retry = retry_count + 1
                # Exponential backoff delay: 30s, 60s, 120s
                backoff = min(30 * (2 ** retry_count), 300)
                logger.warning(
                    "Auto-retrying failed run",
                    run_id=run.id,
                    retry=new_retry,
                    max_retries=max_retries,
                    backoff_seconds=backoff,
                    error=combined_error[:200],
                )
                await self._safe_add_run_event(
                    run.id,
                    "run_auto_retry",
                    message=f"Auto-retrying (attempt {new_retry}/{max_retries}) after {backoff}s backoff",
                    data={
                        "retry_count": new_retry,
                        "max_retries": max_retries,
                        "backoff_seconds": backoff,
                        "error": combined_error[:500],
                    },
                )
                await asyncio.sleep(backoff)
                # Reset run and benchmark statuses to QUEUED/PENDING
                async with async_session_maker() as db:
                    await db.execute(
                        update(BenchmarkRun)
                        .where(BenchmarkRun.id == run.id)
                        .values(
                            status=RunStatus.QUEUED.value,
                            retry_count=new_retry,
                            error_message=None,
                            started_at=None,
                            completed_at=None,
                        )
                    )
                    await db.execute(
                        update(BenchmarkRunBenchmark)
                        .where(BenchmarkRunBenchmark.run_id == run.id)
                        .values(
                            status=BenchmarkRunStatus.PENDING.value,
                            error_message=None,
                            score=None,
                            metrics=None,
                            completed_items=0,
                            started_at=None,
                            completed_at=None,
                        )
                    )
                    # Delete item results from the failed attempt so they don't
                    # pollute the retry.
                    sub = select(BenchmarkRunBenchmark.id).where(
                        BenchmarkRunBenchmark.run_id == run.id
                    )
                    await db.execute(
                        delete(BenchmarkItemResult).where(
                            BenchmarkItemResult.run_benchmark_id.in_(sub)
                        )
                    )
                    await db.commit()
                logger.info("Run requeued for retry", run_id=run.id, retry=new_retry)
                return

            await self._safe_update_run_status(
                run.id,
                RunStatus.FAILED,
                error_message=combined_error[:2000],
            )
            await self._safe_add_run_event(
                run.id,
                "run_failed",
                message=f"Run failed: all benchmarks failed (retries exhausted: {retry_count}/{max_retries})",
                data={"failed_benchmarks": failed_benchmarks, "retry_count": retry_count},
            )
            logger.error("Run failed", run_id=run.id, failed=failed_benchmarks, retries_exhausted=retry_count)
            return

        overall_score = total_score / completed_benchmarks if completed_benchmarks > 0 else None

        if failed_benchmarks > 0:
            all_errors: list[str] = []
            async with async_session_maker() as db:
                result = await db.execute(
                    select(BenchmarkRunBenchmark).where(BenchmarkRunBenchmark.run_id == run.id)
                )
                for rb_row in result.scalars():
                    if rb_row.status != BenchmarkRunStatus.SUCCEEDED.value and rb_row.error_message:
                        all_errors.append(f"{rb_row.benchmark_name}: {rb_row.error_message}")
            combined_error = " | ".join(all_errors) if all_errors else f"{failed_benchmarks} benchmark(s) failed"

            await self._safe_update_run_status(
                run.id,
                RunStatus.FAILED,
                error_message=combined_error[:2000],
                overall_score=overall_score,
            )
            await self._safe_add_run_event(
                run.id,
                "run_failed",
                message=(
                    f"Run failed: {failed_benchmarks} benchmark(s) failed after "
                    f"{completed_benchmarks} succeeded"
                ),
                data={
                    "overall_score": overall_score,
                    "completed_benchmarks": completed_benchmarks,
                    "failed_benchmarks": failed_benchmarks,
                },
            )
            logger.error(
                "Run failed with partial benchmark failures",
                run_id=run.id,
                completed=completed_benchmarks,
                failed=failed_benchmarks,
            )
            return

        await self._safe_update_run_status(
            run.id,
            RunStatus.SUCCEEDED,
            overall_score=overall_score,
        )
        await self._safe_add_run_event(
            run.id,
            "run_completed",
            message=f"Run completed with overall score: {overall_score:.2%}" if overall_score else "Run completed",
            data={"overall_score": overall_score, "completed_benchmarks": completed_benchmarks},
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
        client: InferenceClient,
        judge_client: Optional[InferenceClient] = None,
    ) -> Optional[float]:
        """Execute a single benchmark."""
        logger.info("Starting benchmark", run_id=run.id, benchmark=rb.benchmark_name)

        # Get adapter
        adapter = get_adapter(rb.benchmark_name, client, run.model_slug, judge_client=judge_client)
        if not adapter:
            try:
                from app.benchmarks import adapters as _adapters  # noqa: F401
            except Exception as exc:
                logger.warning(
                    "Failed to import benchmark adapters",
                    benchmark=rb.benchmark_name,
                    error=str(exc) or exc.__class__.__name__,
                )
            else:
                adapter = get_adapter(
                    rb.benchmark_name,
                    client,
                    run.model_slug,
                    judge_client=judge_client,
                )
        if not adapter:
            try:
                adapter = _load_direct_adapter(
                    rb.benchmark_name,
                    client,
                    run.model_slug,
                    judge_client=judge_client,
                )
            except Exception as exc:
                logger.warning(
                    "Direct adapter load failed",
                    benchmark=rb.benchmark_name,
                    error=str(exc) or exc.__class__.__name__,
                )
        if not adapter:
            await self._safe_update_benchmark_status(
                rb.id,
                BenchmarkRunStatus.SKIPPED,
                error_message=f"No adapter found for {rb.benchmark_name}",
            )
            return None

        # Allow adapters to consume run-level config overrides.
        if isinstance(run.config, dict):
            setattr(adapter, "run_config", run.config)

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
            self.last_progress_at[run.id] = datetime.utcnow()
            await self._safe_update_benchmark_status(
                rb.id,
                BenchmarkRunStatus.RUNNING,
                completed_items=rb.completed_items,
            )
            # Preload adapter data (e.g. HF dataset download) with heartbeat
            # refresh BEFORE get_items_for_evaluation, since some adapters
            # (e.g. livecodebench) trigger preload() inside get_total_items().
            await self._preload_adapter(run.id, adapter)
            # Get items and apply subset
            seed_base = run.subset_seed or run.id
            seed = f"{seed_base}:{rb.benchmark_name}"
            total_items, items_to_evaluate = await adapter.get_items_for_evaluation(
                run.subset_pct,
                seed,
                run.subset_count,
            )

            if total_items <= 0 or not items_to_evaluate:
                await self._safe_update_benchmark_status(
                    rb.id,
                    BenchmarkRunStatus.NEEDS_SETUP,
                    error_message="No items found - dataset may require setup",
                )
                return None

            async with async_session_maker() as update_db:
                await update_db.execute(
                    update(Benchmark)
                    .where(Benchmark.id == rb.benchmark_id)
                    .values(total_items=total_items)
                )
                await update_db.commit()

            sampled_item_ids = list(rb.sampled_item_ids or [])
            if sampled_item_ids and len(sampled_item_ids) == len(items_to_evaluate):
                items_to_evaluate = list(sampled_item_ids)
            else:
                sampled_item_ids = list(items_to_evaluate)

            needs_postprocess = adapter.__class__.postprocess is not BenchmarkAdapter.postprocess
            existing_results: list[ItemResult] = []

            if needs_postprocess:
                async with async_session_maker() as read_db:
                    result = await read_db.execute(
                        select(BenchmarkItemResult)
                        .where(BenchmarkItemResult.run_benchmark_id == rb.id)
                    )
                    existing_rows = list(result.scalars().all())
                completed_item_ids = {row.item_id for row in existing_rows}
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
                excluded_items = sum(
                    1 for item_result in existing_results if _is_excluded_item(item_result)
                )
                correct = sum(
                    1
                    for item_result in existing_results
                    if item_result.is_correct and not _is_excluded_item(item_result)
                )
            else:
                async with async_session_maker() as read_db:
                    result = await read_db.execute(
                        select(
                            BenchmarkItemResult.item_id,
                            BenchmarkItemResult.is_correct,
                            BenchmarkItemResult.item_metadata,
                        )
                        .where(BenchmarkItemResult.run_benchmark_id == rb.id)
                    )
                    existing_rows = list(result.all())
                completed_item_ids = {row.item_id for row in existing_rows}
                excluded_items = sum(
                    1
                    for row in existing_rows
                    if (row.item_metadata or {}).get("exclusion_reason")
                )
                correct = sum(
                    1
                    for row in existing_rows
                    if row.is_correct
                    and not (row.item_metadata or {}).get("exclusion_reason")
                )

            completed_items = len(completed_item_ids)
            pending_item_ids = [item_id for item_id in items_to_evaluate if item_id not in completed_item_ids]

            await self._safe_update_benchmark_status(
                rb.id,
                BenchmarkRunStatus.RUNNING,
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
            await self._safe_add_run_event(
                run.id,
                event_type,
                benchmark_name=rb.benchmark_name,
                message=message,
                data={
                    "completed": completed_items,
                    "total": len(items_to_evaluate),
                },
            )

            if not pending_item_ids:
                accuracy = _accuracy_excluding_infrastructure(
                    correct,
                    len(items_to_evaluate),
                    excluded_items,
                )
                additional_metrics = await adapter.postprocess(existing_results) if needs_postprocess else {}
                accuracy_override_present = "accuracy_override" in additional_metrics
                accuracy_override = additional_metrics.pop("accuracy_override", None)
                if accuracy_override_present:
                    accuracy = float(accuracy_override or 0.0)
                correct_override_present = "correct_count_override" in additional_metrics
                correct_override = additional_metrics.pop("correct_count_override", None)
                if correct_override_present:
                    correct = float(correct_override or 0.0)
                score_override_present = "score_override" in additional_metrics
                score_override = additional_metrics.pop("score_override", None)
                score = score_override if score_override_present else accuracy
                metrics = {
                    "accuracy": accuracy,
                    "total_items": total_items,
                    "sampled_items": len(items_to_evaluate),
                    "scored_items": len(items_to_evaluate) - excluded_items,
                    "excluded_items": excluded_items,
                    "sampled_pct": run.subset_pct,
                    "subset_count": run.subset_count,
                    "subset_seed": run.subset_seed,
                    "correct": correct,
                    "score": score,
                    **additional_metrics,
                }
                await self._safe_update_benchmark_status(
                    rb.id,
                    BenchmarkRunStatus.SUCCEEDED,
                    score=score,
                    metrics=metrics,
                    completed_items=completed_items,
                )
                await self._safe_add_run_event(
                    run.id,
                    "benchmark_completed",
                    benchmark_name=rb.benchmark_name,
                    message=(
                        f"Completed {adapter.get_display_name()} with score {score:.2%}"
                        if score is not None
                        else f"Completed {adapter.get_display_name()}"
                    ),
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
            if adapter.supports_parallel_items():
                max_concurrency = settings.worker_item_concurrency
                override_concurrency = adapter.get_item_concurrency()
                if override_concurrency is not None:
                    max_concurrency = max(1, override_concurrency)
            else:
                max_concurrency = 1
            is_gremium_run = (run.provider or "").startswith("gremium")

            def get_effective_item_timeout(item_id: str) -> Optional[int]:
                item_timeout = adapter.get_item_timeout_seconds(item_id)
                if item_timeout is None:
                    item_timeout = settings.worker_item_timeout_seconds
                if is_gremium_run:
                    # A provider-specific default may extend an adapter budget,
                    # but it must never shorten the adapter's declared cap.
                    item_timeout = max(
                        item_timeout or 0,
                        settings.gremium_item_timeout_seconds,
                    )
                if item_timeout is not None and item_timeout <= 0:
                    return None
                return item_timeout

            resolved_item_timeouts = [
                timeout
                for item_id in items_to_evaluate
                if (timeout := get_effective_item_timeout(item_id)) is not None
            ]
            if resolved_item_timeouts:
                cached_timeout = self._benchmark_timeout_cache.get(rb.benchmark_name, 0)
                self._benchmark_timeout_cache[rb.benchmark_name] = max(
                    cached_timeout,
                    max(resolved_item_timeouts),
                )
            item_attempts = settings.worker_item_attempts
            if is_gremium_run:
                item_attempts = settings.gremium_item_attempts

            result_lock = asyncio.Lock()
            abort_event = asyncio.Event()
            abort_error: Optional[str] = None

            async def evaluate_item(item_id: str) -> ItemResult:
                item_timeout = get_effective_item_timeout(item_id)
                attempt = 0
                delay_seconds = 1
                last_result: Optional[ItemResult] = None
                max_attempts = max(item_attempts, 1)
                deadline = None
                if item_timeout:
                    deadline = time.monotonic() + item_timeout

                while attempt < max_attempts:
                    attempt += 1
                    self.last_progress_at[run.id] = datetime.utcnow()
                    try:
                        timeout_remaining = None
                        if deadline is not None:
                            timeout_remaining = max(0.0, deadline - time.monotonic())
                            if timeout_remaining <= 0:
                                raise _RunnerItemTimeoutError
                        result = await _evaluate_adapter_item(
                            adapter,
                            item_id,
                            timeout_remaining,
                        )
                    except _RunnerItemTimeoutError:
                        result = _item_timeout_result(item_id, item_timeout)
                    except Exception as exc:
                        detail = str(exc) or exc.__class__.__name__
                        result = ItemResult(item_id=item_id, error=detail)

                    if result.metadata is None:
                        result.metadata = {}
                    result.metadata["worker_attempt"] = attempt
                    result.metadata["worker_attempts"] = max_attempts
                    result = _apply_error_score_defaults(result)
                    last_result = result

                    if not _is_retryable_item_error(result.error):
                        return result

                    if attempt < max_attempts:
                        await asyncio.sleep(delay_seconds)
                        delay_seconds = min(delay_seconds * 2, 10)

                if last_result and last_result.error and max_attempts > 1:
                    last_result.error = (
                        f"{last_result.error} (after {max_attempts} attempts)"
                    )
                return last_result or ItemResult(item_id=item_id, error="Item evaluation failed")

            async def record_result(result: ItemResult) -> None:
                nonlocal correct, excluded_items, completed_items, abort_error

                async with result_lock:
                    if needs_postprocess:
                        new_results.append(result)

                    is_excluded = _is_excluded_item(result)
                    if is_excluded:
                        excluded_items += 1
                    elif result.is_correct:
                        correct += 1

                    completed_items += 1
                    current_completed = completed_items
                    current_correct = correct
                    if result.error and _is_fatal_item_error(result.error):
                        abort_error = result.error
                        abort_event.set()
                    self.last_progress_at[run.id] = datetime.utcnow()

                should_persist_progress = (
                    current_completed == len(items_to_evaluate)
                    or current_completed % PROGRESS_PERSIST_INTERVAL == 0
                )
                await self._persist_item_result_progress(
                    run.id,
                    rb.id,
                    result,
                    rb.benchmark_name,
                    current_completed,
                    len(items_to_evaluate),
                    current_correct,
                    should_persist_progress,
                )

            if max_concurrency > 1 and len(pending_item_ids) > 1:
                semaphore = asyncio.Semaphore(max_concurrency)

                async def run_item(item_id: str) -> ItemResult:
                    async with semaphore:
                        return await evaluate_item(item_id)

                pending_iter = iter(pending_item_ids)
                pending_tasks: set[asyncio.Task[ItemResult]] = set()

                async def schedule_next() -> bool:
                    if abort_event.is_set():
                        return False
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
                        if abort_event.is_set():
                            for pending in pending_tasks:
                                pending.cancel()
                            await asyncio.gather(*pending_tasks, return_exceptions=True)
                            raise Exception(abort_error or "Fatal benchmark error")
                        completed_since_check += 1
                        if completed_since_check % 10 == 0:
                            if await self._is_run_canceled(run.id):
                                for pending in pending_tasks:
                                    pending.cancel()
                                await asyncio.gather(*pending_tasks, return_exceptions=True)
                                raise Exception("Run canceled")
                        await schedule_next()
            else:
                for i, item_id in enumerate(pending_item_ids):
                    if i % 10 == 0:
                        if await self._is_run_canceled(run.id):
                            raise Exception("Run canceled")

                    result = await evaluate_item(item_id)
                    await record_result(result)
                    if abort_event.is_set():
                        raise Exception(abort_error or "Fatal benchmark error")

            all_results = existing_results + new_results if needs_postprocess else []
            accuracy = _accuracy_excluding_infrastructure(
                correct,
                len(items_to_evaluate),
                excluded_items,
            )
            additional_metrics = await adapter.postprocess(all_results) if needs_postprocess else {}
            accuracy_override_present = "accuracy_override" in additional_metrics
            accuracy_override = additional_metrics.pop("accuracy_override", None)
            if accuracy_override_present:
                accuracy = float(accuracy_override or 0.0)
            correct_override_present = "correct_count_override" in additional_metrics
            correct_override = additional_metrics.pop("correct_count_override", None)
            if correct_override_present:
                correct = float(correct_override or 0.0)
            score_override_present = "score_override" in additional_metrics
            score_override = additional_metrics.pop("score_override", None)
            score = score_override if score_override_present else accuracy

            metrics = {
                "accuracy": accuracy,
                "total_items": total_items,
                "sampled_items": len(items_to_evaluate),
                "scored_items": len(items_to_evaluate) - excluded_items,
                "excluded_items": excluded_items,
                "sampled_pct": run.subset_pct,
                "subset_count": run.subset_count,
                "subset_seed": run.subset_seed,
                "correct": correct,
                "score": score,
                **additional_metrics,
            }

            await self._safe_update_benchmark_status(
                rb.id,
                BenchmarkRunStatus.SUCCEEDED,
                score=score,
                metrics=metrics,
                completed_items=completed_items,
            )

            await self._safe_add_run_event(
                run.id,
                "benchmark_completed",
                benchmark_name=rb.benchmark_name,
                message=(
                    f"Completed {adapter.get_display_name()} with score {score:.2%}"
                    if score is not None
                    else f"Completed {adapter.get_display_name()}"
                ),
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

    if settings.worker_disabled:
        logger.warning("Worker disabled via WORKER_DISABLED, skipping run loop.")
        while True:
            await asyncio.sleep(3600)

    worker = BenchmarkWorker()

    watchdog = None
    if settings.worker_watchdog_enabled:
        watchdog = LoopWatchdog(
            timeout_seconds=settings.worker_watchdog_timeout_seconds,
            check_interval_seconds=settings.worker_watchdog_check_interval_seconds,
            heartbeat_file=settings.worker_watchdog_heartbeat_file,
            logger=logger,
        )
        worker.watchdog = watchdog
        watchdog.start()

    try:
        await worker.start()
    except KeyboardInterrupt:
        await worker.stop()
    finally:
        if watchdog is not None:
            watchdog.stop()


if __name__ == "__main__":
    asyncio.run(run_worker())
