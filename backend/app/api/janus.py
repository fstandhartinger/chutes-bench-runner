"""Janus-specific API routes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.benchmarks.janus_scoring import calculate_janus_composite_score
from app.db.session import async_session_maker
from app.services.run_service import get_run

router = APIRouter(prefix="/api/janus", tags=["janus"])


@router.get("/composite-score/{run_id}")
async def get_composite_score(run_id: str) -> dict[str, float]:
    """Calculate Janus composite score for a run."""
    async with async_session_maker() as db:
        run = await get_run(db, run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")

        benchmark_results: dict[str, dict[str, object]] = {}
        for rb in run.benchmarks:
            if rb.benchmark_name.startswith("janus_"):
                benchmark_results[rb.benchmark_name] = {
                    "score": rb.score or 0.0,
                    "metrics": rb.metrics or {},
                }

    return calculate_janus_composite_score(benchmark_results)
