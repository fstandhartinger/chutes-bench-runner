"""API routes."""
import asyncio
from datetime import datetime, timedelta
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query, Request, UploadFile, File, status
from fastapi.responses import StreamingResponse
from sqlalchemy import case, func, select

from app.api.deps import AdminDep, ApiKeyDep, SessionDep
from app.benchmarks import get_adapter
from app.api.schemas import (
    BenchmarkInfo,
    BenchmarksListResponse,
    BenchmarkSummaryResponse,
    CancelRunResponse,
    CreateRunRequest,
    ItemResultResponse,
    ItemResultsResponse,
    ModelResponse,
    ModelsListResponse,
    PublicKeyResponse,
    MaintenanceStatusResponse,
    OpsOverviewResponse,
    SandyMetricsPoint,
    SandyResourcesResponse,
    SandySandboxStatsResponse,
    TokenUsageStats,
    TokenUsageWindow,
    RunBenchmarkDetailsResponse,
    RunEventResponse,
    RunResponse,
    RunSummaryResponse,
    RunsListResponse,
    SignedExportVerifyResponse,
    SyncModelsResponse,
    WorkerHeartbeatInfo,
    WorkerTimeseriesPoint,
)
from app.models.benchmark import Benchmark
from app.models.run import BenchmarkItemResult, BenchmarkRun, BenchmarkRunBenchmark, RunEvent
from app.services import auth_service
from app.services.export_service import generate_csv_export, generate_pdf_export
from app.core.config import get_settings
from app.core.logging import get_logger
from app.services.chutes_client import get_chutes_client
from app.services.gremium_client import GremiumClient
from app.services.model_service import (
    ensure_gremium_models,
    get_model_by_id,
    get_models,
    resolve_model_identifier,
    sync_models,
)
from app.services.run_service import (
    add_run_event,
    cancel_run,
    create_run,
    get_item_results,
    get_run,
    get_run_events,
    list_runs,
    requeue_run,
)
from app.services.worker_service import (
    get_queue_timeseries,
    get_worker_timeseries,
    list_active_workers,
)
from app.services.sandy_service import SandyService
from app.services.signed_export_service import (
    SigningKeyError,
    generate_signed_zip_export,
    get_public_key_info,
    verify_signed_zip_export,
)

logger = get_logger(__name__)
router = APIRouter(prefix="/api")


# Models endpoints
@router.get("/models", response_model=ModelsListResponse)
async def list_models(
    db: SessionDep,
    search: Optional[str] = None,
    provider: Optional[str] = Query(default=None),
    limit: int = Query(default=100, le=500),
    offset: int = Query(default=0, ge=0),
):
    """List available Chutes models."""
    settings = get_settings()
    if provider and provider.startswith("gremium"):
        if settings.enable_gremium_provider:
            await ensure_gremium_models(db)
            await db.commit()
        models = await get_models(db, search=search, provider=provider, limit=limit, offset=offset)
        return ModelsListResponse(
            models=[ModelResponse.model_validate(m) for m in models],
            total=len(models),
        )

    models = await get_models(db, search=search, provider=provider, limit=limit, offset=offset)
    llm_identifiers = await get_chutes_client().get_llm_identifiers()
    if llm_identifiers:
        models = [
            model
            for model in models
            if model.provider == "chutes"
            and (model.slug in llm_identifiers or (model.chute_id and model.chute_id in llm_identifiers))
        ]
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
    # Refresh total_items for benchmarks that haven't been computed yet.
    settings = get_settings()
    if not settings.skip_model_sync:
        client = get_chutes_client()
        totals_updated = False
        for benchmark in benchmarks:
            if benchmark.total_items > 0:
                continue
            adapter = get_adapter(benchmark.name, client, "system")
            if not adapter:
                continue
            try:
                total = await adapter.get_total_items()
            except Exception:
                continue
            if total > 0:
                benchmark.total_items = total
                totals_updated = True
        if totals_updated:
            await db.commit()
    # Sync setup metadata from adapters (lightweight).
    client = get_chutes_client()
    setup_updated = False
    for benchmark in benchmarks:
        adapter = get_adapter(benchmark.name, client, "system")
        if not adapter:
            continue
        requires_setup = adapter.requires_setup()
        setup_notes = adapter.get_setup_notes()
        display_name = adapter.get_display_name()
        if benchmark.requires_setup != requires_setup:
            benchmark.requires_setup = requires_setup
            setup_updated = True
        if (benchmark.setup_notes or "") != (setup_notes or ""):
            benchmark.setup_notes = setup_notes
            setup_updated = True
        if benchmark.display_name != display_name:
            benchmark.display_name = display_name
            setup_updated = True
    if setup_updated:
        await db.commit()
    latency_result = await db.execute(
        select(
            BenchmarkRunBenchmark.benchmark_id,
            func.avg(BenchmarkItemResult.latency_ms),
        )
        .join(
            BenchmarkItemResult,
            BenchmarkItemResult.run_benchmark_id == BenchmarkRunBenchmark.id,
        )
        .where(BenchmarkItemResult.latency_ms.is_not(None))
        .group_by(BenchmarkRunBenchmark.benchmark_id)
    )
    avg_latency_by_id = {
        row[0]: float(row[1]) if row[1] is not None else None
        for row in latency_result.all()
    }
    return BenchmarksListResponse(
        benchmarks=[
            BenchmarkInfo(
                name=b.name,
                display_name=b.display_name,
                description=b.description,
                category=(b.config or {}).get("category") or "Core Benchmarks",
                is_enabled=b.is_enabled,
                supports_subset=b.supports_subset,
                requires_setup=b.requires_setup,
                setup_notes=b.setup_notes,
                default_selected=(
                    (b.config or {}).get("default_selected")
                    if (b.config or {}).get("default_selected") is not None
                    else b.is_enabled
                ),
                total_items=b.total_items,
                avg_item_latency_ms=avg_latency_by_id.get(b.id),
            )
            for b in benchmarks
        ]
    )


# Runs endpoints
@router.post("/runs", response_model=RunResponse, status_code=status.HTTP_201_CREATED)
async def create_benchmark_run(
    db: SessionDep,
    request: CreateRunRequest,
    http_request: Request,
):
    """Create a new benchmark run."""
    settings = get_settings()
    if settings.maintenance_mode:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=settings.maintenance_message,
        )
    provider = request.provider or "chutes"
    allowed_providers = {"chutes", "gremium-openai", "gremium-anthropic"}
    if provider not in allowed_providers:
        raise HTTPException(status_code=400, detail="Unsupported provider")
    if provider.startswith("gremium") and not settings.enable_gremium_provider:
        raise HTTPException(status_code=400, detail="Gremium provider is not enabled.")

    # Validate model exists
    model = await get_model_by_id(db, request.model_id)
    if not model:
        model = await resolve_model_identifier(db, request.model_id, provider=provider)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    if model.provider != provider:
        raise HTTPException(status_code=400, detail="Model does not match selected provider")

    auth_mode = "system"
    auth_session_id = None
    access_token = None

    session_id = http_request.cookies.get(auth_service.SESSION_COOKIE_NAME)
    if session_id:
        session = await auth_service.get_session(db, session_id)
        if session:
            if not session.can_invoke_chutes():
                raise HTTPException(status_code=403, detail="Chutes session missing chutes:invoke scope")
            access_token = await auth_service.get_valid_access_token(db, session)
            if not access_token:
                raise HTTPException(status_code=401, detail="Chutes session expired or invalid")
            auth_mode = "idp"
            auth_session_id = session_id

    provider_metadata: Optional[dict[str, Any]] = None
    if provider == "chutes":
        client = (
            get_chutes_client(user_access_token=access_token)
            if access_token
            else get_chutes_client()
        )
        try:
            available = await client.is_model_available(model.slug, model.chute_id)
            if available is False:
                ok, status_code, detail = await client.probe_model_access(model.slug)
                if ok:
                    available = True
                elif status_code in (401, 403):
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Chutes credentials are not authorized for this model.",
                    )
                elif status_code == 404:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail="Model not available for inference on Chutes.",
                    )
                else:
                    logger.warning(
                        "Model access probe failed; allowing run creation",
                        model=model.slug,
                        status_code=status_code,
                        detail=detail,
                    )
        finally:
            if access_token:
                await client.close()
    else:
        gremium_client = GremiumClient(
            api_key=settings.gremium_api_key or settings.chutes_api_key,
            provider=provider,
            base_url=settings.gremium_api_base_url,
        )
        try:
            provider_metadata = await gremium_client.get_metadata()
            provider_metadata = {
                "provider": provider,
                "base_url": settings.gremium_api_base_url,
                "version": provider_metadata.get("version"),
                "participants": provider_metadata.get("participants"),
            }
        except Exception as exc:
            logger.warning("Failed to fetch Gremium metadata", error=str(exc))
        finally:
            await gremium_client.close()

    try:
        run = await create_run(
            db,
            model_id=model.id,
            model_slug=model.slug,
            provider=provider,
            subset_pct=request.subset_pct,
            subset_count=request.subset_count,
            subset_seed=request.subset_seed,
            selected_benchmarks=request.selected_benchmarks,
            config=request.config,
            provider_metadata=provider_metadata,
            auth_mode=auth_mode,
            auth_session_id=auth_session_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return RunResponse.model_validate(run)


@router.post("/runs/api", response_model=RunResponse, status_code=status.HTTP_201_CREATED)
async def create_benchmark_run_with_api_key(
    db: SessionDep,
    request: CreateRunRequest,
    api_key: ApiKeyDep,
):
    """Create a new benchmark run using a bearer API key."""
    settings = get_settings()
    if settings.maintenance_mode:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=settings.maintenance_message,
        )
    provider = request.provider or "chutes"
    allowed_providers = {"chutes", "gremium-openai", "gremium-anthropic"}
    if provider not in allowed_providers:
        raise HTTPException(status_code=400, detail="Unsupported provider")
    if provider.startswith("gremium") and not settings.enable_gremium_provider:
        raise HTTPException(status_code=400, detail="Gremium provider is not enabled.")

    model = await resolve_model_identifier(db, request.model_id, provider=provider)
    if not model:
        await sync_models(db)
        model = await resolve_model_identifier(db, request.model_id, provider=provider)
    if not model:
        raise HTTPException(
            status_code=404,
            detail="Model not found. Provide a bench runner model UUID, Chutes chute_id, or model slug.",
        )
    if model.provider != provider:
        raise HTTPException(status_code=400, detail="Model does not match selected provider")

    provider_metadata: Optional[dict[str, Any]] = None
    if provider == "chutes":
        client = get_chutes_client(api_key=api_key)
        try:
            available = await client.is_model_available(model.slug, model.chute_id)
            if available is False:
                ok, status_code, detail = await client.probe_model_access(model.slug)
                if ok:
                    available = True
                elif status_code in (401, 403):
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Chutes API key is not authorized for this model.",
                    )
                elif status_code == 404:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail="Model not available for inference on Chutes.",
                    )
                else:
                    logger.warning(
                        "Model access probe failed; allowing run creation",
                        model=model.slug,
                        status_code=status_code,
                        detail=detail,
                    )
        finally:
            await client.close()
    else:
        gremium_client = GremiumClient(
            api_key=settings.gremium_api_key or settings.chutes_api_key,
            provider=provider,
            base_url=settings.gremium_api_base_url,
        )
        try:
            provider_metadata = await gremium_client.get_metadata()
            provider_metadata = {
                "provider": provider,
                "base_url": settings.gremium_api_base_url,
                "version": provider_metadata.get("version"),
                "participants": provider_metadata.get("participants"),
            }
        except Exception as exc:
            logger.warning("Failed to fetch Gremium metadata", error=str(exc))
        finally:
            await gremium_client.close()

    try:
        run = await create_run(
            db,
            model_id=model.id,
            model_slug=model.slug,
            provider=provider,
            subset_pct=request.subset_pct,
            subset_count=request.subset_count,
            subset_seed=request.subset_seed,
            selected_benchmarks=request.selected_benchmarks,
            config=request.config,
            provider_metadata=provider_metadata,
            auth_mode="api_key",
            auth_api_key=api_key,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return RunResponse.model_validate(run)


@router.get("/runs", response_model=RunsListResponse)
async def list_benchmark_runs(
    db: SessionDep,
    status_filter: Optional[str] = Query(None, alias="status"),
    model_id: Optional[str] = None,
    model_slug: Optional[str] = None,
    limit: int = Query(default=50, le=200),
    offset: int = Query(default=0, ge=0),
):
    """List benchmark runs."""
    runs, total = await list_runs(
        db, status=status_filter, model_id=model_id, model_slug=model_slug, limit=limit, offset=offset
    )
    return RunsListResponse(
        runs=[RunResponse.model_validate(r) for r in runs],
        total=total,
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


@router.post("/admin/runs/{run_id}/requeue", response_model=CancelRunResponse)
async def requeue_benchmark_run(
    db: SessionDep,
    run_id: str,
    _: AdminDep,
):
    """Requeue a failed or canceled benchmark run (admin only)."""
    success = await requeue_run(db, run_id)
    if not success:
        raise HTTPException(
            status_code=400,
            detail="Run cannot be requeued (must be failed or canceled)",
        )
    return CancelRunResponse(success=True, message="Run requeued")


@router.get("/runs/{run_id}/benchmarks/{benchmark_name}", response_model=RunBenchmarkDetailsResponse)
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

    run = await get_run(db, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    # Get item results
    items = await get_item_results(db, rb.id, limit=limit, offset=offset)

    # Aggregate summary stats across all items
    summary_result = await db.execute(
        select(
            func.count(BenchmarkItemResult.id),
            func.coalesce(
                func.sum(case((BenchmarkItemResult.is_correct == True, 1), else_=0)),  # noqa: E712
                0,
            ),
            func.coalesce(
                func.sum(case((BenchmarkItemResult.is_correct == False, 1), else_=0)),  # noqa: E712
                0,
            ),
            func.coalesce(
                func.sum(case((BenchmarkItemResult.error.is_not(None), 1), else_=0)),
                0,
            ),
            func.avg(BenchmarkItemResult.latency_ms),
            func.coalesce(func.sum(BenchmarkItemResult.input_tokens), 0),
            func.coalesce(func.sum(BenchmarkItemResult.output_tokens), 0),
        ).where(BenchmarkItemResult.run_benchmark_id == rb.id)
    )
    (
        total_items,
        correct_count,
        incorrect_count,
        error_count,
        avg_latency,
        input_tokens,
        output_tokens,
    ) = summary_result.one()
    total_items = int(total_items or 0)
    input_tokens = int(input_tokens or 0)
    output_tokens = int(output_tokens or 0)
    total_tokens = input_tokens + output_tokens

    input_cost_usd = None
    output_cost_usd = None
    total_cost_usd = None
    pricing_input = None
    pricing_output = None
    client = get_chutes_client()
    pricing = await client.get_model_pricing(
        run.model_slug,
        run.model.chute_id if run.model else None,
    )
    if pricing:
        pricing_input, pricing_output = pricing
        input_cost_usd = (input_tokens / 1_000_000) * pricing_input
        output_cost_usd = (output_tokens / 1_000_000) * pricing_output
        total_cost_usd = input_cost_usd + output_cost_usd

    breakdown_result = await db.execute(
        select(
            BenchmarkItemResult.error,
            func.count(BenchmarkItemResult.id),
        )
        .where(
            BenchmarkItemResult.run_benchmark_id == rb.id,
            BenchmarkItemResult.error.is_not(None),
        )
        .group_by(BenchmarkItemResult.error)
        .order_by(func.count(BenchmarkItemResult.id).desc())
    )
    error_breakdown = [
        {"message": row[0], "count": int(row[1] or 0)}
        for row in breakdown_result.all()
        if row[0]
    ]

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
            "error_message": rb.error_message,
        },
        "items": ItemResultsResponse(
            items=[ItemResultResponse.model_validate(i) for i in items],
            total=total_items,
            limit=limit,
            offset=offset,
        ),
        "summary": BenchmarkSummaryResponse(
            total_items=total_items,
            correct=int(correct_count or 0),
            incorrect=int(incorrect_count or 0),
            errors=int(error_count or 0),
            avg_latency_ms=float(avg_latency) if avg_latency is not None else None,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            input_cost_usd=input_cost_usd,
            output_cost_usd=output_cost_usd,
            total_cost_usd=total_cost_usd,
            pricing_input_per_million_usd=pricing_input,
            pricing_output_per_million_usd=pricing_output,
            error_breakdown=error_breakdown,
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
    format: str = Query(default="csv", pattern="^(csv|pdf|zip)$"),
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
    elif format == "pdf":
        filename, content = await generate_pdf_export(db, run)
        media_type = "application/pdf"
    else:
        try:
            filename, content = await generate_signed_zip_export(db, run)
        except SigningKeyError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        media_type = "application/zip"

    return StreamingResponse(
        iter([content]),
        media_type=media_type,
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@router.get("/exports/public-key", response_model=PublicKeyResponse)
async def export_public_key():
    """Return the public key used to verify signed exports."""
    try:
        return get_public_key_info()
    except SigningKeyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/exports/verify", response_model=SignedExportVerifyResponse)
async def verify_signed_export(file: UploadFile = File(...)):
    """Verify a signed benchmark results zip."""
    content = await file.read()
    return SignedExportVerifyResponse.model_validate(verify_signed_zip_export(content))


# Ops endpoints
@router.get("/ops/overview", response_model=OpsOverviewResponse)
async def ops_overview(
    db: SessionDep,
    minutes: int = Query(default=360, ge=30, le=1440),
):
    """Get worker heartbeats, queue counts, and recent runs for ops dashboard."""
    settings = get_settings()
    stale_seconds = max(settings.worker_heartbeat_seconds * 3, 180)
    workers = await list_active_workers(db, stale_seconds=stale_seconds)
    timeseries = await get_worker_timeseries(db, minutes=minutes)
    if timeseries:
        buckets = [point["timestamp"] for point in timeseries]
        queue_series = await get_queue_timeseries(db, buckets)
        queue_map = {point["timestamp"]: point for point in queue_series}
        merged_timeseries = []
        for point in timeseries:
            queued = queue_map.get(point["timestamp"], {}).get("queued_runs", 0)
            merged_timeseries.append({**point, "queued_runs": int(queued or 0)})
        timeseries = merged_timeseries

    counts_result = await db.execute(
        select(BenchmarkRun.status, func.count()).group_by(BenchmarkRun.status)
    )
    queue_counts = {status: int(count) for status, count in counts_result.all()}
    for status_name in ["queued", "running", "succeeded", "failed", "canceled"]:
        queue_counts.setdefault(status_name, 0)

    queued_runs_result = await db.execute(
        select(BenchmarkRun)
        .where(BenchmarkRun.status == "queued")
        .order_by(BenchmarkRun.created_at.asc())
        .limit(25)
    )
    running_runs_result = await db.execute(
        select(BenchmarkRun)
        .where(BenchmarkRun.status == "running")
        .order_by(BenchmarkRun.started_at.desc().nullslast())
        .limit(25)
    )
    completed_runs_result = await db.execute(
        select(BenchmarkRun)
        .where(BenchmarkRun.status.in_(["succeeded", "failed", "canceled"]))
        .order_by(BenchmarkRun.completed_at.desc().nullslast(), BenchmarkRun.created_at.desc())
        .limit(25)
    )

    async def build_token_window(hours: int) -> TokenUsageWindow:
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        token_result = await db.execute(
            select(
                func.coalesce(func.sum(BenchmarkItemResult.input_tokens), 0),
                func.coalesce(func.sum(BenchmarkItemResult.output_tokens), 0),
            ).where(BenchmarkItemResult.created_at >= cutoff)
        )
        input_tokens, output_tokens = token_result.one()
        return TokenUsageWindow(
            window_hours=hours,
            input_tokens=int(input_tokens or 0),
            output_tokens=int(output_tokens or 0),
        )

    token_stats = TokenUsageStats(
        last_24h=await build_token_window(24),
        last_7d=await build_token_window(24 * 7),
    )

    def summarize_run(run: BenchmarkRun) -> RunSummaryResponse:
        benchmarks: Optional[list[str]]
        if run.selected_benchmarks:
            benchmarks = [str(name) for name in run.selected_benchmarks]
        elif run.benchmarks:
            benchmarks = [rb.benchmark_name for rb in run.benchmarks]
        else:
            benchmarks = None
        return RunSummaryResponse(
            id=run.id,
            model_slug=run.model_slug,
            provider=run.provider,
            subset_pct=run.subset_pct,
            subset_count=run.subset_count,
            subset_seed=run.subset_seed,
            status=run.status,
            created_at=run.created_at,
            started_at=run.started_at,
            completed_at=run.completed_at,
            benchmarks=benchmarks,
        )

    return OpsOverviewResponse(
        workers=[WorkerHeartbeatInfo.model_validate(worker) for worker in workers],
        timeseries=[WorkerTimeseriesPoint.model_validate(point) for point in timeseries],
        queue_counts=queue_counts,
        queued_runs=[summarize_run(run) for run in queued_runs_result.scalars().all()],
        running_runs=[summarize_run(run) for run in running_runs_result.scalars().all()],
        completed_runs=[summarize_run(run) for run in completed_runs_result.scalars().all()],
        worker_config={
            "worker_max_concurrent": settings.worker_max_concurrent,
            "worker_item_concurrency": settings.worker_item_concurrency,
        },
        token_stats=token_stats,
    )


@router.get("/ops/sandy/resources", response_model=SandyResourcesResponse)
async def sandy_resources():
    """Return current Sandy host resource usage."""
    sandy = SandyService()
    data = await sandy.get_resources()
    if not data:
        detail = sandy.last_error or "Unable to fetch Sandy resources"
        raise HTTPException(status_code=503, detail=detail)
    memory_total_bytes = data.get("memoryTotalBytes")
    memory_used_bytes = data.get("memoryUsedBytes")
    memory_total_gb = data.get("memory_total_gb")
    if memory_total_gb is None and memory_total_bytes:
        memory_total_gb = memory_total_bytes / (1024**3)
    memory_available_gb = data.get("memory_available_gb")
    if memory_available_gb is None and data.get("memoryAvailableBytes"):
        memory_available_gb = data["memoryAvailableBytes"] / (1024**3)
    memory_percent = data.get("memory_percent")
    if memory_percent is None and memory_total_bytes and memory_used_bytes is not None:
        memory_percent = (memory_used_bytes / memory_total_bytes) * 100 if memory_total_bytes else None
    disk_used_ratio = data.get("disk_used_ratio") or data.get("diskUsedRatio")

    normalized = {
        "canCreateSandbox": data.get("canCreateSandbox"),
        "rejectReason": data.get("rejectReason"),
        "limits": data.get("limits", {}),
        "priorityBreakdown": data.get("priorityBreakdown", {}),
        "cpu_percent": data.get("cpu_percent"),
        "memory_percent": memory_percent,
        "memory_total_gb": memory_total_gb,
        "memory_available_gb": memory_available_gb,
        "disk_used_ratio": disk_used_ratio,
        "cpuCount": data.get("cpuCount"),
        "load1": data.get("load1") or data.get("load_avg_1m"),
        "load5": data.get("load5") or data.get("load_avg_5m"),
        "load15": data.get("load15") or data.get("load_avg_15m"),
        "memoryTotalBytes": memory_total_bytes,
        "memoryUsedBytes": memory_used_bytes,
        "memoryAvailableBytes": data.get("memoryAvailableBytes"),
        "diskTotalBytes": data.get("diskTotalBytes"),
        "diskUsedBytes": data.get("diskUsedBytes"),
        "diskFreeBytes": data.get("diskFreeBytes"),
        "diskUsedRatio": data.get("diskUsedRatio"),
    }
    return SandyResourcesResponse.model_validate(normalized)


@router.get("/ops/sandy/metrics", response_model=list[SandyMetricsPoint])
async def sandy_metrics(
    hours: int = Query(default=12, ge=1, le=48),
):
    """Return Sandy system metrics time series."""
    sandy = SandyService()
    data = await sandy.get_metrics_timeseries(hours=hours)
    if data is None:
        detail = sandy.last_error or "Unable to fetch Sandy metrics"
        raise HTTPException(status_code=503, detail=detail)
    return [SandyMetricsPoint.model_validate(point) for point in data]


@router.get("/ops/sandy/sandboxes", response_model=list[SandySandboxStatsResponse])
async def sandy_sandbox_stats(
    ids: Optional[str] = Query(default=None),
):
    """Return per-sandbox resource stats from Sandy."""
    sandy = SandyService()
    sandbox_ids = None
    if ids:
        sandbox_ids = [value for value in ids.split(",") if value]
    data = await sandy.get_sandbox_stats(sandbox_ids)
    if data is None:
        detail = sandy.last_error or "Unable to fetch Sandy sandbox stats"
        raise HTTPException(status_code=503, detail=detail)
    return [SandySandboxStatsResponse.model_validate(point) for point in data]


# Health endpoint
@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@router.get("/status", response_model=MaintenanceStatusResponse)
async def service_status():
    """Service status endpoint (maintenance flag)."""
    settings = get_settings()
    return MaintenanceStatusResponse(
        maintenance_mode=settings.maintenance_mode,
        message=settings.maintenance_message,
    )
