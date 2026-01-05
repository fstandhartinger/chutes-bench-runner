"""Benchmark run management service."""
from datetime import datetime
from typing import Any, Optional

from sqlalchemy import func, select, update
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.models.benchmark import Benchmark
from app.models.run import (
    BenchmarkItemResult,
    BenchmarkRun,
    BenchmarkRunBenchmark,
    BenchmarkRunStatus,
    RunEvent,
    RunStatus,
)

logger = get_logger(__name__)


async def create_run(
    db: AsyncSession,
    model_id: str,
    model_slug: str,
    subset_pct: int,
    selected_benchmarks: Optional[list[str]] = None,
    config: Optional[dict[str, Any]] = None,
    auth_mode: Optional[str] = None,
    auth_session_id: Optional[str] = None,
    auth_api_key: Optional[str] = None,
) -> BenchmarkRun:
    """
    Create a new benchmark run.
    
    Args:
        db: Database session
        model_id: Model UUID
        model_slug: Model slug for display
        subset_pct: Percentage of items to sample (1, 5, 10, 25, 50, 100)
        selected_benchmarks: Optional list of benchmark names to run (None = all enabled)
        config: Optional additional configuration
        
    Returns:
        Created BenchmarkRun
    """
    run = BenchmarkRun(
        model_id=model_id,
        model_slug=model_slug,
        subset_pct=subset_pct,
        selected_benchmarks=selected_benchmarks,
        config=config,
        auth_mode=auth_mode,
        auth_session_id=auth_session_id,
        auth_api_key=auth_api_key,
        status=RunStatus.QUEUED.value,
    )
    db.add(run)
    await db.flush()

    # Get enabled benchmarks
    query = select(Benchmark).where(Benchmark.is_enabled == True)  # noqa: E712
    if selected_benchmarks:
        query = query.where(Benchmark.name.in_(selected_benchmarks))
    result = await db.execute(query)
    benchmarks = result.scalars().all()

    # Create benchmark run entries
    for benchmark in benchmarks:
        run_benchmark = BenchmarkRunBenchmark(
            run_id=run.id,
            benchmark_id=benchmark.id,
            benchmark_name=benchmark.name,
            status=BenchmarkRunStatus.PENDING.value,
            total_items=benchmark.total_items,
        )
        db.add(run_benchmark)

    await db.commit()
    await db.refresh(run)

    # Create initial event
    await add_run_event(
        db,
        run.id,
        "run_created",
        message=f"Run created for model {model_slug} with {subset_pct}% subset",
        data={"model_slug": model_slug, "subset_pct": subset_pct, "benchmark_count": len(benchmarks)},
    )

    logger.info("Created run", run_id=run.id, model=model_slug, benchmarks=len(benchmarks))
    return run


async def get_run(db: AsyncSession, run_id: str) -> Optional[BenchmarkRun]:
    """Get a run by ID with all related data."""
    result = await db.execute(select(BenchmarkRun).where(BenchmarkRun.id == run_id))
    return result.scalar_one_or_none()


async def list_runs(
    db: AsyncSession,
    status: Optional[str] = None,
    model_id: Optional[str] = None,
    model_slug: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> list[BenchmarkRun]:
    """List runs with optional filtering."""
    query = select(BenchmarkRun)

    if status:
        query = query.where(BenchmarkRun.status == status)
    if model_id:
        query = query.where(BenchmarkRun.model_id == model_id)
    if model_slug:
        query = query.where(BenchmarkRun.model_slug.ilike(f"%{model_slug}%"))

    query = query.order_by(BenchmarkRun.created_at.desc()).offset(offset).limit(limit)

    result = await db.execute(query)
    return list(result.scalars().all())


async def update_run_status(
    db: AsyncSession,
    run_id: str,
    status: RunStatus,
    error_message: Optional[str] = None,
    overall_score: Optional[float] = None,
) -> None:
    """Update run status."""
    update_data: dict[str, Any] = {"status": status.value, "updated_at": datetime.utcnow()}

    if status == RunStatus.RUNNING:
        update_data["started_at"] = func.coalesce(BenchmarkRun.started_at, datetime.utcnow())
    elif status in (RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.CANCELED):
        update_data["completed_at"] = datetime.utcnow()

    if error_message:
        update_data["error_message"] = error_message
    if overall_score is not None:
        update_data["overall_score"] = overall_score

    await db.execute(
        update(BenchmarkRun)
        .where(BenchmarkRun.id == run_id)
        .values(**update_data)
        .execution_options(synchronize_session=False)
    )
    await db.commit()


async def cancel_run(db: AsyncSession, run_id: str) -> bool:
    """Cancel a run if it's still running or queued."""
    run = await get_run(db, run_id)
    if not run or run.status not in (RunStatus.QUEUED.value, RunStatus.RUNNING.value):
        return False

    now = datetime.utcnow()
    await db.execute(
        update(BenchmarkRunBenchmark)
        .where(BenchmarkRunBenchmark.run_id == run_id)
        .where(BenchmarkRunBenchmark.status == BenchmarkRunStatus.RUNNING.value)
        .values(
            status=BenchmarkRunStatus.SKIPPED.value,
            error_message="Run canceled",
            completed_at=now,
            updated_at=now,
        )
    )
    await db.execute(
        update(BenchmarkRunBenchmark)
        .where(BenchmarkRunBenchmark.run_id == run_id)
        .where(BenchmarkRunBenchmark.status == BenchmarkRunStatus.PENDING.value)
        .values(
            status=BenchmarkRunStatus.SKIPPED.value,
            error_message="Run canceled",
            completed_at=now,
            updated_at=now,
        )
    )
    await db.execute(
        update(BenchmarkRun)
        .where(BenchmarkRun.id == run_id)
        .values(
            status=RunStatus.CANCELED.value,
            canceled_at=now,
            completed_at=now,
            updated_at=now,
        )
    )
    await db.commit()

    await add_run_event(db, run_id, "run_canceled", message="Run canceled by user")
    return True


async def requeue_run(db: AsyncSession, run_id: str) -> bool:
    """Requeue a failed or canceled run so remaining benchmarks resume."""
    run = await get_run(db, run_id)
    if not run or run.status not in (RunStatus.FAILED.value, RunStatus.CANCELED.value):
        return False

    now = datetime.utcnow()
    await db.execute(
        update(BenchmarkRunBenchmark)
        .where(BenchmarkRunBenchmark.run_id == run_id)
        .where(BenchmarkRunBenchmark.status != BenchmarkRunStatus.SUCCEEDED.value)
        .values(
            status=BenchmarkRunStatus.PENDING.value,
            error_message=None,
            completed_at=None,
            started_at=None,
            updated_at=now,
        )
    )
    await db.execute(
        update(BenchmarkRun)
        .where(BenchmarkRun.id == run_id)
        .values(
            status=RunStatus.QUEUED.value,
            error_message=None,
            completed_at=None,
            started_at=None,
            canceled_at=None,
            updated_at=now,
        )
    )
    await db.commit()
    await add_run_event(db, run_id, "run_requeued", message="Run manually requeued")
    return True


async def update_benchmark_status(
    db: AsyncSession,
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
    """Update benchmark status within a run."""
    now = datetime.utcnow()
    update_data: dict[str, Any] = {"status": status.value, "updated_at": now}

    if status == BenchmarkRunStatus.RUNNING:
        started_result = await db.execute(
            select(BenchmarkRunBenchmark.started_at).where(BenchmarkRunBenchmark.id == run_benchmark_id)
        )
        if started_result.scalar_one_or_none() is None:
            update_data["started_at"] = now
    elif status in (BenchmarkRunStatus.SUCCEEDED, BenchmarkRunStatus.FAILED, BenchmarkRunStatus.SKIPPED):
        update_data["completed_at"] = now

    if metrics:
        update_data["metrics"] = metrics
    if score is not None:
        update_data["score"] = score
    if error_message:
        update_data["error_message"] = error_message
    if completed_items is not None:
        update_data["completed_items"] = completed_items
    if total_items is not None:
        update_data["total_items"] = total_items
    if sampled_items is not None:
        update_data["sampled_items"] = sampled_items
    if sampled_item_ids is not None:
        update_data["sampled_item_ids"] = sampled_item_ids

    run_id_result = await db.execute(
        select(BenchmarkRunBenchmark.run_id).where(BenchmarkRunBenchmark.id == run_benchmark_id)
    )
    run_id = run_id_result.scalar_one_or_none()
    if run_id:
        await db.execute(
            update(BenchmarkRun)
            .where(BenchmarkRun.id == run_id)
            .values(updated_at=now)
            .execution_options(synchronize_session=False)
        )
    await db.execute(
        update(BenchmarkRunBenchmark)
        .where(BenchmarkRunBenchmark.id == run_benchmark_id)
        .values(**update_data)
        .execution_options(synchronize_session=False)
    )
    await db.commit()


async def save_item_result(
    db: AsyncSession,
    run_benchmark_id: str,
    item_id: str,
    prompt: Optional[str] = None,
    response: Optional[str] = None,
    expected: Optional[str] = None,
    is_correct: Optional[bool] = None,
    score: Optional[float] = None,
    judge_output: Optional[dict[str, Any]] = None,
    latency_ms: Optional[int] = None,
    input_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
    error: Optional[str] = None,
    test_code: Optional[str] = None,
    item_metadata: Optional[dict[str, Any]] = None,
    item_hash: Optional[str] = None,
) -> BenchmarkItemResult:
    """Save a single item result."""
    result = BenchmarkItemResult(
        run_benchmark_id=run_benchmark_id,
        item_id=item_id,
        item_hash=item_hash,
        prompt=prompt,
        response=response,
        expected=expected,
        is_correct=is_correct,
        score=score,
        judge_output=judge_output,
        latency_ms=latency_ms,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        error=error,
        test_code=test_code,
        item_metadata=item_metadata,
    )
    db.add(result)
    try:
        await db.flush()
    except DBAPIError as exc:
        await db.rollback()
        if getattr(exc, "connection_invalidated", False):
            logger.warning("DB connection invalidated while saving item result; retrying", error=str(exc))
            db.add(result)
            await db.flush()
        else:
            raise
    return result


async def get_run_benchmarks(db: AsyncSession, run_id: str) -> list[BenchmarkRunBenchmark]:
    """Get all benchmarks for a run."""
    result = await db.execute(
        select(BenchmarkRunBenchmark).where(BenchmarkRunBenchmark.run_id == run_id)
    )
    return list(result.scalars().all())


async def get_item_results(
    db: AsyncSession,
    run_benchmark_id: str,
    limit: int = 100,
    offset: int = 0,
) -> list[BenchmarkItemResult]:
    """Get item results for a benchmark run."""
    result = await db.execute(
        select(BenchmarkItemResult)
        .where(BenchmarkItemResult.run_benchmark_id == run_benchmark_id)
        .order_by(BenchmarkItemResult.created_at)
        .offset(offset)
        .limit(limit)
    )
    return list(result.scalars().all())


async def add_run_event(
    db: AsyncSession,
    run_id: str,
    event_type: str,
    message: Optional[str] = None,
    benchmark_name: Optional[str] = None,
    data: Optional[dict[str, Any]] = None,
) -> RunEvent:
    """Add a run event for progress/log streaming."""
    event = RunEvent(
        run_id=run_id,
        event_type=event_type,
        message=message,
        benchmark_name=benchmark_name,
        data=data,
    )
    db.add(event)
    await db.commit()
    return event


async def get_run_events(
    db: AsyncSession,
    run_id: str,
    after_id: Optional[str] = None,
    limit: int = 100,
) -> list[RunEvent]:
    """Get run events, optionally after a specific event ID."""
    query = select(RunEvent).where(RunEvent.run_id == run_id)

    if after_id:
        # Get the timestamp of the reference event
        ref_result = await db.execute(select(RunEvent.created_at).where(RunEvent.id == after_id))
        ref_time = ref_result.scalar_one_or_none()
        if ref_time:
            query = query.where(RunEvent.created_at > ref_time)

    query = query.order_by(RunEvent.created_at).limit(limit)
    result = await db.execute(query)
    return list(result.scalars().all())
