"""Worker heartbeat tracking and ops helpers."""
from __future__ import annotations

from datetime import datetime, timedelta
from sqlalchemy import select, func
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.worker import WorkerHeartbeat, WorkerHeartbeatLog


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
