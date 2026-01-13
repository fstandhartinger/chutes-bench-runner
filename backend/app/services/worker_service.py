"""Worker heartbeat tracking and ops helpers."""
from __future__ import annotations

from datetime import datetime, timedelta
from sqlalchemy import select, func, or_
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.worker import WorkerHeartbeat, WorkerHeartbeatLog
from app.models.run import BenchmarkRun


async def record_worker_heartbeat(
    db: AsyncSession,
    worker_id: str,
    hostname: str | None,
    running_runs: int,
    max_concurrent_runs: int,
    item_concurrency: int,
) -> None:
    """Upsert the latest heartbeat and append a time-series entry."""
    now = datetime.utcnow()
    stmt = insert(WorkerHeartbeat).values(
        worker_id=worker_id,
        hostname=hostname,
        running_runs=running_runs,
        max_concurrent_runs=max_concurrent_runs,
        item_concurrency=item_concurrency,
        last_seen=now,
        created_at=now,
        updated_at=now,
    )
    stmt = stmt.on_conflict_do_update(
        index_elements=[WorkerHeartbeat.worker_id],
        set_={
            "hostname": hostname,
            "running_runs": running_runs,
            "max_concurrent_runs": max_concurrent_runs,
            "item_concurrency": item_concurrency,
            "last_seen": now,
            "updated_at": now,
        },
    )
    await db.execute(stmt)
    await db.execute(
        insert(WorkerHeartbeatLog).values(
            worker_id=worker_id,
            hostname=hostname,
            running_runs=running_runs,
            max_concurrent_runs=max_concurrent_runs,
            item_concurrency=item_concurrency,
            created_at=now,
        )
    )
    await db.commit()


async def list_active_workers(
    db: AsyncSession,
    stale_seconds: int = 180,
) -> list[WorkerHeartbeat]:
    """Return workers seen within the stale window."""
    cutoff = datetime.utcnow() - timedelta(seconds=stale_seconds)
    result = await db.execute(
        select(WorkerHeartbeat).where(WorkerHeartbeat.last_seen >= cutoff)
    )
    return list(result.scalars().all())


async def get_worker_timeseries(
    db: AsyncSession,
    minutes: int = 180,
) -> list[dict[str, int | datetime]]:
    """Return aggregated worker counts by minute for a lookback window."""
    cutoff = datetime.utcnow() - timedelta(minutes=minutes)
    result = await db.execute(
        select(
            func.date_trunc("minute", WorkerHeartbeatLog.created_at).label("bucket"),
            func.count(func.distinct(WorkerHeartbeatLog.worker_id)).label("worker_count"),
            func.sum(WorkerHeartbeatLog.running_runs).label("running_runs"),
        )
        .where(WorkerHeartbeatLog.created_at >= cutoff)
        .group_by("bucket")
        .order_by("bucket")
    )
    rows = result.all()
    return [
        {
            "timestamp": row.bucket,
            "worker_count": int(row.worker_count or 0),
            "running_runs": int(row.running_runs or 0),
        }
        for row in rows
    ]


async def get_queue_timeseries(
    db: AsyncSession,
    buckets: list[datetime],
) -> list[dict[str, int | datetime]]:
    """Return queued/running counts aligned to the provided buckets."""
    if not buckets:
        return []
    cutoff = buckets[0]
    result = await db.execute(
        select(
            BenchmarkRun.created_at,
            BenchmarkRun.started_at,
            BenchmarkRun.completed_at,
        ).where(
            or_(
                BenchmarkRun.created_at >= cutoff,
                BenchmarkRun.started_at >= cutoff,
                BenchmarkRun.completed_at >= cutoff,
                BenchmarkRun.completed_at.is_(None),
            )
        )
    )
    runs = result.all()

    series: list[dict[str, int | datetime]] = []
    for bucket in buckets:
        queued = 0
        running = 0
        for created_at, started_at, completed_at in runs:
            if not created_at or created_at > bucket:
                continue
            if completed_at and completed_at <= bucket:
                continue
            if started_at and started_at <= bucket:
                running += 1
            else:
                queued += 1
        series.append(
            {
                "timestamp": bucket,
                "queued_runs": queued,
                "running_runs": running,
            }
        )
    return series
