import pytest
from sqlalchemy import select

from app.models.benchmark import Benchmark
from app.models.model import Model
from app.models.run import (
    BenchmarkItemResult,
    BenchmarkRun,
    BenchmarkRunBenchmark,
    BenchmarkRunStatus,
    RunEvent,
    RunStatus,
)
from app.services.run_service import (
    cancel_run,
    requeue_run,
    save_item_result,
    update_benchmark_status,
    update_run_status,
)


@pytest.mark.asyncio
async def test_requeue_allows_succeeded_run_with_failed_child_benchmark(test_session):
    model = Model(slug="rlm-gpt-4o", name="RLM GPT-4o", provider="rlm", is_active=True)
    succeeded_benchmark = Benchmark(
        name="s_niah",
        display_name="S-NIAH",
        adapter_class="SNIAHAdapter",
        is_enabled=True,
        supports_subset=True,
    )
    failed_benchmark = Benchmark(
        name="oolong_pairs",
        display_name="OOLONG-Pairs",
        adapter_class="OolongPairsAdapter",
        is_enabled=True,
        supports_subset=True,
    )
    test_session.add_all([model, succeeded_benchmark, failed_benchmark])
    await test_session.flush()

    run = BenchmarkRun(
        model_id=model.id,
        model_slug=model.slug,
        provider="rlm",
        subset_pct=100,
        status=RunStatus.SUCCEEDED.value,
    )
    test_session.add(run)
    await test_session.flush()
    succeeded = BenchmarkRunBenchmark(
        run_id=run.id,
        benchmark_id=succeeded_benchmark.id,
        benchmark_name=succeeded_benchmark.name,
        status=BenchmarkRunStatus.SUCCEEDED.value,
        completed_items=60,
    )
    failed = BenchmarkRunBenchmark(
        run_id=run.id,
        benchmark_id=failed_benchmark.id,
        benchmark_name=failed_benchmark.name,
        status=BenchmarkRunStatus.FAILED.value,
        completed_items=2950,
        error_message="Exception: Run canceled",
    )
    test_session.add_all([succeeded, failed])
    await test_session.commit()

    assert await requeue_run(test_session, run.id) is True

    await test_session.refresh(run)
    await test_session.refresh(succeeded)
    await test_session.refresh(failed)
    assert run.status == RunStatus.QUEUED.value
    assert run.error_message is None
    assert succeeded.status == BenchmarkRunStatus.SUCCEEDED.value
    assert failed.status == BenchmarkRunStatus.PENDING.value
    assert failed.error_message is None
    assert failed.completed_items == 2950


@pytest.mark.asyncio
async def test_requeue_rejects_clean_succeeded_run(test_session):
    model = Model(slug="rlm-gpt-4o", name="RLM GPT-4o", provider="rlm", is_active=True)
    benchmark = Benchmark(
        name="s_niah",
        display_name="S-NIAH",
        adapter_class="SNIAHAdapter",
        is_enabled=True,
        supports_subset=True,
    )
    test_session.add_all([model, benchmark])
    await test_session.flush()

    run = BenchmarkRun(
        model_id=model.id,
        model_slug=model.slug,
        provider="rlm",
        subset_pct=100,
        status=RunStatus.SUCCEEDED.value,
    )
    test_session.add(run)
    await test_session.flush()
    test_session.add(
        BenchmarkRunBenchmark(
            run_id=run.id,
            benchmark_id=benchmark.id,
            benchmark_name=benchmark.name,
            status=BenchmarkRunStatus.SUCCEEDED.value,
            completed_items=60,
        )
    )
    await test_session.commit()

    assert await requeue_run(test_session, run.id) is False


@pytest.mark.asyncio
async def test_cancel_run_atomically_finishes_every_nonterminal_child(test_session):
    model = Model(slug="cancel-model", name="Cancel Model", provider="chutes", is_active=True)
    benchmark = Benchmark(
        name="cancel-benchmark",
        display_name="Cancel Benchmark",
        adapter_class="CancelAdapter",
        is_enabled=True,
        supports_subset=True,
    )
    test_session.add_all([model, benchmark])
    await test_session.flush()

    run = BenchmarkRun(
        model_id=model.id,
        model_slug=model.slug,
        provider="chutes",
        subset_pct=100,
        status=RunStatus.RUNNING.value,
    )
    test_session.add(run)
    await test_session.flush()

    statuses = [
        BenchmarkRunStatus.RUNNING.value,
        BenchmarkRunStatus.PENDING.value,
        "queued",  # Legacy/live rows have used this value despite the current enum.
        BenchmarkRunStatus.NEEDS_SETUP.value,
        BenchmarkRunStatus.SUCCEEDED.value,
        BenchmarkRunStatus.FAILED.value,
        BenchmarkRunStatus.SKIPPED.value,
    ]
    children = [
        BenchmarkRunBenchmark(
            run_id=run.id,
            benchmark_id=benchmark.id,
            benchmark_name=f"cancel-benchmark-{index}",
            status=status,
        )
        for index, status in enumerate(statuses)
    ]
    test_session.add_all(children)
    await test_session.commit()

    assert await cancel_run(test_session, run.id) is True

    run_row = (
        await test_session.execute(
            select(BenchmarkRun.status, BenchmarkRun.canceled_at).where(BenchmarkRun.id == run.id)
        )
    ).one()
    child_rows = (
        await test_session.execute(
            select(BenchmarkRunBenchmark.id, BenchmarkRunBenchmark.status).where(
                BenchmarkRunBenchmark.run_id == run.id
            )
        )
    ).all()
    events = (
        await test_session.execute(
            select(RunEvent).where(
                RunEvent.run_id == run.id,
                RunEvent.event_type == "run_canceled",
            )
        )
    ).scalars().all()

    assert run_row.status == RunStatus.CANCELED.value
    assert run_row.canceled_at is not None
    status_by_id = {row.id: row.status for row in child_rows}
    for child in children[:4]:
        assert status_by_id[child.id] == BenchmarkRunStatus.SKIPPED.value
    for child, original_status in zip(children[4:], statuses[4:], strict=True):
        assert status_by_id[child.id] == original_status
    assert len(events) == 1


@pytest.mark.asyncio
async def test_late_worker_writes_cannot_resurrect_canceled_lifecycle(test_session):
    model = Model(slug="late-model", name="Late Model", provider="chutes", is_active=True)
    benchmark = Benchmark(
        name="late-benchmark",
        display_name="Late Benchmark",
        adapter_class="LateAdapter",
        is_enabled=True,
        supports_subset=True,
    )
    test_session.add_all([model, benchmark])
    await test_session.flush()
    run = BenchmarkRun(
        model_id=model.id,
        model_slug=model.slug,
        provider="chutes",
        subset_pct=100,
        status=RunStatus.RUNNING.value,
    )
    test_session.add(run)
    await test_session.flush()
    child = BenchmarkRunBenchmark(
        run_id=run.id,
        benchmark_id=benchmark.id,
        benchmark_name=benchmark.name,
        status=BenchmarkRunStatus.RUNNING.value,
    )
    test_session.add(child)
    await test_session.commit()
    run_id = run.id
    child_id = child.id

    assert await cancel_run(test_session, run_id) is True
    await update_benchmark_status(test_session, child_id, BenchmarkRunStatus.RUNNING)
    await update_run_status(test_session, run_id, RunStatus.SUCCEEDED, overall_score=1.0)
    saved = await save_item_result(
        test_session,
        child_id,
        item_id="too-late",
        is_correct=True,
        score=1.0,
    )
    await test_session.commit()

    lifecycle = (
        await test_session.execute(
            select(BenchmarkRun.status, BenchmarkRunBenchmark.status)
            .join(BenchmarkRunBenchmark, BenchmarkRunBenchmark.run_id == BenchmarkRun.id)
            .where(BenchmarkRun.id == run_id)
        )
    ).one()
    item_count = len(
        (
            await test_session.execute(
                select(BenchmarkItemResult).where(
                    BenchmarkItemResult.run_benchmark_id == child_id
                )
            )
        ).scalars().all()
    )

    assert lifecycle == (RunStatus.CANCELED.value, BenchmarkRunStatus.SKIPPED.value)
    assert saved is None
    assert item_count == 0
