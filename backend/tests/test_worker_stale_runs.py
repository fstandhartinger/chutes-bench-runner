from datetime import datetime, timedelta

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.models.benchmark import Benchmark
from app.models.model import Model
from app.models.run import BenchmarkRun, BenchmarkRunBenchmark, BenchmarkRunStatus, RunStatus
from app.worker.runner import (
    BenchmarkWorker,
    _compute_run_stale_seconds,
    _try_transition_stale_run,
)


def test_compute_run_stale_seconds_extends_for_started_work() -> None:
    assert _compute_run_stale_seconds(900, 1800, True) == 1800


def test_compute_run_stale_seconds_keeps_base_without_started_work() -> None:
    assert _compute_run_stale_seconds(900, 1800, False) == 900


@pytest.mark.asyncio
async def test_try_transition_stale_run_only_claims_once(test_session) -> None:
    model = Model(
        slug="test-model",
        name="Test Model",
        provider="chutes",
        is_active=True,
    )
    benchmark = Benchmark(
        name="livecodebench",
        display_name="LiveCodeBench",
        adapter_class="LiveCodeBenchAdapter",
        is_enabled=True,
        supports_subset=True,
    )
    test_session.add_all([model, benchmark])
    await test_session.flush()

    updated_at = datetime.utcnow() - timedelta(minutes=20)
    run = BenchmarkRun(
        model_id=model.id,
        model_slug=model.slug,
        provider="chutes",
        subset_pct=100,
        status=RunStatus.RUNNING.value,
        updated_at=updated_at,
        started_at=updated_at,
    )
    test_session.add(run)
    await test_session.flush()
    test_session.add(
        BenchmarkRunBenchmark(
            run_id=run.id,
            benchmark_id=benchmark.id,
            benchmark_name=benchmark.name,
            status=BenchmarkRunStatus.RUNNING.value,
            started_at=updated_at,
        )
    )
    await test_session.commit()

    first_claim = await _try_transition_stale_run(
        test_session,
        run.id,
        updated_at,
        {
            "status": RunStatus.QUEUED.value,
            "retry_count": 1,
            "updated_at": datetime.utcnow(),
        },
    )
    await test_session.commit()
    second_claim = await _try_transition_stale_run(
        test_session,
        run.id,
        updated_at,
        {
            "status": RunStatus.FAILED.value,
            "updated_at": datetime.utcnow(),
        },
    )

    assert first_claim is True
    assert second_claim is False


@pytest.mark.asyncio
async def test_requeue_stale_runs_respects_timeout_before_sampling(test_session, monkeypatch) -> None:
    model = Model(
        slug="test-model",
        name="Test Model",
        provider="chutes",
        is_active=True,
    )
    benchmark = Benchmark(
        name="livecodebench",
        display_name="LiveCodeBench",
        adapter_class="LiveCodeBenchAdapter",
        is_enabled=True,
        supports_subset=True,
    )
    test_session.add_all([model, benchmark])
    await test_session.flush()

    started_at = datetime.utcnow() - timedelta(minutes=20)
    run = BenchmarkRun(
        model_id=model.id,
        model_slug=model.slug,
        provider="chutes",
        subset_pct=100,
        status=RunStatus.RUNNING.value,
        updated_at=started_at,
        started_at=started_at,
        retry_count=0,
        max_retries=3,
    )
    test_session.add(run)
    await test_session.flush()
    test_session.add(
        BenchmarkRunBenchmark(
            run_id=run.id,
            benchmark_id=benchmark.id,
            benchmark_name=benchmark.name,
            status=BenchmarkRunStatus.RUNNING.value,
            started_at=started_at,
            sampled_items=0,
        )
    )
    await test_session.commit()

    worker = BenchmarkWorker()
    monkeypatch.setattr(worker, "_get_benchmark_timeout", lambda *_args: 1800)
    test_session_maker = async_sessionmaker(
        test_session.bind,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    monkeypatch.setattr("app.worker.runner.async_session_maker", test_session_maker)

    await worker.requeue_stale_runs()

    refreshed_run = await test_session.get(BenchmarkRun, run.id)
    assert refreshed_run is not None
    assert refreshed_run.status == RunStatus.RUNNING.value
    assert refreshed_run.retry_count == 0
