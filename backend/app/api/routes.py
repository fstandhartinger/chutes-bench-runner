"""API routes."""
import asyncio
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from sqlalchemy import func, select

from app.api.deps import AdminDep, SessionDep
from app.api.schemas import (
    BenchmarkInfo,
    BenchmarksListResponse,
    CancelRunResponse,
    CreateRunRequest,
    ItemResultResponse,
    ItemResultsResponse,
    ModelResponse,
    ModelsListResponse,
    RunEventResponse,
    RunResponse,
    RunsListResponse,
    SyncModelsResponse,
)
from app.models.benchmark import Benchmark
from app.models.run import BenchmarkItemResult, BenchmarkRun, BenchmarkRunBenchmark, RunEvent
from app.services.export_service import generate_csv_export, generate_pdf_export
from app.services.model_service import get_model_by_id, get_models, sync_models
from app.services.run_service import (
    add_run_event,
    cancel_run,
    create_run,
    get_item_results,
    get_run,
    get_run_events,
    list_runs,
)

router = APIRouter(prefix="/api")


# Models endpoints
@router.get("/models", response_model=ModelsListResponse)
async def list_models(
    db: SessionDep,
    search: Optional[str] = None,
    limit: int = Query(default=100, le=500),
    offset: int = Query(default=0, ge=0),
):
    """List available Chutes models."""
    models = await get_models(db, search=search, limit=limit, offset=offset)
    return ModelsListResponse(
        models=[ModelResponse.model_validate(m) for m in models],
        total=len(models),
    )


@router.post("/admin/sync-models", response_model=SyncModelsResponse)
async def sync_models_endpoint(
    db: SessionDep,
    _: AdminDep,
):
    """Sync models from Chutes API (admin only)."""
    count = await sync_models(db)
    return SyncModelsResponse(synced=count, message=f"Synced {count} models")


# Benchmarks endpoints
@router.get("/benchmarks", response_model=BenchmarksListResponse)
async def list_benchmarks(db: SessionDep):
    """List all available benchmarks."""
    result = await db.execute(select(Benchmark).order_by(Benchmark.name))
    benchmarks = result.scalars().all()
    return BenchmarksListResponse(
        benchmarks=[
            BenchmarkInfo(
                name=b.name,
                display_name=b.display_name,
                description=b.description,
                is_enabled=b.is_enabled,
                supports_subset=b.supports_subset,
                requires_setup=b.requires_setup,
                setup_notes=b.setup_notes,
                total_items=b.total_items,
            )
            for b in benchmarks
        ]
    )


# Runs endpoints
@router.post("/runs", response_model=RunResponse, status_code=status.HTTP_201_CREATED)
async def create_benchmark_run(
    db: SessionDep,
    request: CreateRunRequest,
):
    """Create a new benchmark run."""
    # Validate model exists
    model = await get_model_by_id(db, request.model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    run = await create_run(
        db,
        model_id=model.id,
        model_slug=model.slug,
        subset_pct=request.subset_pct,
        selected_benchmarks=request.selected_benchmarks,
        config=request.config,
    )

    return RunResponse.model_validate(run)


@router.get("/runs", response_model=RunsListResponse)
async def list_benchmark_runs(
    db: SessionDep,
    status_filter: Optional[str] = Query(None, alias="status"),
    model_id: Optional[str] = None,
    limit: int = Query(default=50, le=200),
    offset: int = Query(default=0, ge=0),
):
    """List benchmark runs."""
    runs = await list_runs(db, status=status_filter, model_id=model_id, limit=limit, offset=offset)
    return RunsListResponse(
        runs=[RunResponse.model_validate(r) for r in runs],
        total=len(runs),
    )


@router.get("/runs/{run_id}", response_model=RunResponse)
async def get_benchmark_run(
    db: SessionDep,
    run_id: str,
):
    """Get a specific benchmark run."""
    run = await get_run(db, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return RunResponse.model_validate(run)


@router.post("/runs/{run_id}/cancel", response_model=CancelRunResponse)
async def cancel_benchmark_run(
    db: SessionDep,
    run_id: str,
):
    """Cancel a running or queued benchmark run."""
    success = await cancel_run(db, run_id)
    if not success:
        raise HTTPException(
            status_code=400,
            detail="Run cannot be canceled (may already be completed or not found)",
        )
    return CancelRunResponse(success=True, message="Run canceled")


@router.get("/runs/{run_id}/benchmarks/{benchmark_name}")
async def get_run_benchmark_details(
    db: SessionDep,
    run_id: str,
    benchmark_name: str,
    limit: int = Query(default=100, le=500),
    offset: int = Query(default=0, ge=0),
):
    """Get benchmark details and item results."""
    # Get run benchmark
    result = await db.execute(
        select(BenchmarkRunBenchmark)
        .where(
            BenchmarkRunBenchmark.run_id == run_id,
            BenchmarkRunBenchmark.benchmark_name == benchmark_name,
        )
    )
    rb = result.scalar_one_or_none()
    if not rb:
        raise HTTPException(status_code=404, detail="Benchmark not found in run")

    # Get item results
    items = await get_item_results(db, rb.id, limit=limit, offset=offset)

    # Get total count
    count_result = await db.execute(
        select(func.count())
        .select_from(BenchmarkItemResult)
        .where(BenchmarkItemResult.run_benchmark_id == rb.id)
    )
    total = count_result.scalar() or 0

    return {
        "benchmark": {
            "id": rb.id,
            "benchmark_name": rb.benchmark_name,
            "status": rb.status,
            "score": rb.score,
            "metrics": rb.metrics,
            "total_items": rb.total_items,
            "sampled_items": rb.sampled_items,
            "completed_items": rb.completed_items,
        },
        "items": ItemResultsResponse(
            items=[ItemResultResponse.model_validate(i) for i in items],
            total=total,
            limit=limit,
            offset=offset,
        ),
    }


@router.get("/runs/{run_id}/events")
async def stream_run_events(
    db: SessionDep,
    run_id: str,
    last_event_id: Optional[str] = Query(None, alias="Last-Event-ID"),
):
    """
    Stream run events via Server-Sent Events.
    
    Supports resumption via Last-Event-ID header.
    """
    run = await get_run(db, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    async def event_generator():
        """Generate SSE events."""
        current_last_id = last_event_id
        finished_statuses = {"succeeded", "failed", "canceled"}

        while True:
            events = await get_run_events(db, run_id, after_id=current_last_id, limit=50)

            for event in events:
                data = RunEventResponse.model_validate(event).model_dump_json()
                # Send as generic message (no event: field) so onmessage can catch it
                # The event_type is included in the data JSON
                yield f"id: {event.id}\ndata: {data}\n\n"
                current_last_id = event.id

            # Check if run is finished
            await db.refresh(run)
            if run.status in finished_statuses:
                # Send done event as named event for specific listener
                yield f"event: done\ndata: {{}}\n\n"
                break

            await asyncio.sleep(1)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/runs/{run_id}/export")
async def export_run(
    db: SessionDep,
    run_id: str,
    format: str = Query(default="csv", pattern="^(csv|pdf)$"),
):
    """Export run results as CSV or PDF."""
    run = await get_run(db, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    if run.status not in ("succeeded", "failed"):
        raise HTTPException(
            status_code=400,
            detail="Run must be completed before export",
        )

    if format == "csv":
        filename, content = await generate_csv_export(db, run)
        media_type = "text/csv"
    else:
        filename, content = await generate_pdf_export(db, run)
        media_type = "application/pdf"

    return StreamingResponse(
        iter([content]),
        media_type=media_type,
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


# Health endpoint
@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

