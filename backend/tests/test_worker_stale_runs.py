from datetime import datetime, timedelta
from unittest.mock import AsyncMock

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.benchmarks.base import ItemResult
from app.models.benchmark import Benchmark
from app.models.model import Model
from app.models.run import BenchmarkRun, BenchmarkRunBenchmark, BenchmarkRunStatus, RunStatus
from app.worker.runner import (
    BenchmarkWorker,
    _compute_run_stale_seconds,
    _is_retryable_db_write_error,
    _try_transition_stale_run,
)


def test_compute_run_stale_seconds_extends_for_started_work() -> None:
    assert _compute_run_stale_seconds(900, 1800, True) == 1800


def test_compute_run_stale_seconds_keeps_base_without_started_work() -> None:
    assert _compute_run_stale_seconds(900, 1800, False) == 900


def test_is_retryable_db_write_error_detects_deadlocks() -> None:
    assert _is_retryable_db_write_error(RuntimeError("deadlock detected")) is True
    assert _is_retryable_db_write_error(RuntimeError("connection was closed in the middle of operation")) is True
    assert _is_retryable_db_write_error(RuntimeError("other failure")) is False


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


@pytest.mark.asyncio
async def test_requeue_stale_runs_handles_commit_across_multiple_runs(test_session, monkeypatch) -> None:
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

    started_at = datetime.utcnow() - timedelta(minutes=40)
    run_ids: list[str] = []
    for _ in range(2):
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
        run_ids.append(str(run.id))
        test_session.add(
            BenchmarkRunBenchmark(
                run_id=run.id,
                benchmark_id=benchmark.id,
                benchmark_name=benchmark.name,
                status=BenchmarkRunStatus.RUNNING.value,
                started_at=started_at,
                sampled_items=1,
            )
        )
    await test_session.commit()

    worker = BenchmarkWorker()
    monkeypatch.setattr(worker, "_get_benchmark_timeout", lambda *_args: 60)
    test_session_maker = async_sessionmaker(
        test_session.bind,
        class_=AsyncSession,
        expire_on_commit=True,
    )
    monkeypatch.setattr("app.worker.runner.async_session_maker", test_session_maker)

    await worker.requeue_stale_runs()

    async with test_session_maker() as verify_session:
        for run_id in run_ids:
            refreshed_run = await verify_session.get(BenchmarkRun, run_id)
            assert refreshed_run is not None
            assert refreshed_run.status == RunStatus.QUEUED.value
            assert refreshed_run.retry_count == 1


@pytest.mark.asyncio
async def test_persist_item_result_progress_retries_deadlock(test_session, monkeypatch) -> None:
    worker = BenchmarkWorker()
    test_session_maker = async_sessionmaker(
        test_session.bind,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    monkeypatch.setattr("app.worker.runner.async_session_maker", test_session_maker)
    monkeypatch.setattr("app.worker.runner.save_item_result", AsyncMock())

    update_mock = AsyncMock(side_effect=[RuntimeError("deadlock detected"), None])
    monkeypatch.setattr("app.worker.runner.update_benchmark_status", update_mock)
    add_event_mock = AsyncMock()
    monkeypatch.setattr(worker, "_safe_add_run_event", add_event_mock)

    await worker._persist_item_result_progress(
        run_id="run-1",
        run_benchmark_id="rb-1",
        result=ItemResult(item_id="item-1", is_correct=True, score=1.0, metadata={}),
        benchmark_name="livecodebench",
        current_completed=5,
        total_items=10,
        current_correct=4,
        should_persist_progress=True,
    )

    assert update_mock.await_count == 2
    assert add_event_mock.await_count == 1


@pytest.mark.asyncio
async def test_is_run_canceled_fails_open_on_transient_db_error(monkeypatch) -> None:
    worker = BenchmarkWorker()

    class BrokenSessionFactory:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self):
            return self

        async def __aenter__(self):
            self.calls += 1
            raise RuntimeError("connection was closed in the middle of operation")

        async def __aexit__(self, exc_type, exc, tb):
            return False

    broken_factory = BrokenSessionFactory()
    monkeypatch.setattr("app.worker.runner.async_session_maker", broken_factory)

    assert await worker._is_run_canceled("run-1") is False
    assert broken_factory.calls == 3
