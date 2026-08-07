import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.benchmarks.base import ItemResult
from app.models.benchmark import Benchmark
from app.models.model import Model
from app.models.run import (
    BenchmarkItemResult,
    BenchmarkRun,
    BenchmarkRunBenchmark,
    BenchmarkRunStatus,
    RunStatus,
)
from app.worker.runner import (
    BenchmarkWorker,
    _compute_run_stale_seconds,
    _evaluate_adapter_item,
    _is_retryable_db_write_error,
    _try_transition_stale_run,
)

TEST_RUN_PROVENANCE = {
    "schema": "bench-runner-provenance-v1",
    "bench_runner_git_sha": "a" * 40,
    "code_version": "7" * 64,
    "worker_image_digest": "sha256:" + "1" * 64,
    "adapter_sha256": {"terminal_bench.py": "2" * 64},
    "adapter_set_sha256": "3" * 64,
    "sandy_runtime_image_digest": "sha256:" + "4" * 64,
    "agent_binaries": {},
}


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
async def test_execute_run_fails_fast_on_zero_balance_probe(test_session, monkeypatch) -> None:
    model = Model(
        slug="test-model",
        name="Test Model",
        provider="chutes",
        is_active=True,
        chute_id="test-chute",
    )
    benchmark = Benchmark(
        name="mmlu_pro",
        display_name="MMLU-Pro",
        adapter_class="MMLUProAdapter",
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
    run_benchmark = BenchmarkRunBenchmark(
        run_id=run.id,
        benchmark_id=benchmark.id,
        benchmark_name=benchmark.name,
        status=BenchmarkRunStatus.PENDING.value,
    )
    test_session.add(run_benchmark)
    await test_session.commit()
    await test_session.refresh(run)

    worker = BenchmarkWorker()
    test_session_maker = async_sessionmaker(
        test_session.bind,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    monkeypatch.setattr("app.worker.runner.async_session_maker", test_session_maker)
    monkeypatch.setattr(
        "app.worker.runner.collect_worker_provenance",
        AsyncMock(return_value=TEST_RUN_PROVENANCE),
    )

    fake_client = AsyncMock()
    fake_client.is_model_available.return_value = False
    fake_client.probe_model_access.return_value = (
        False,
        402,
        '{"detail":"Chute unavailable because the creator of this chute has zero balance."}',
    )
    monkeypatch.setattr(worker, "_get_client_for_run", AsyncMock(return_value=fake_client))
    execute_benchmark = AsyncMock(side_effect=AssertionError("execute_benchmark should not run"))
    monkeypatch.setattr(worker, "execute_benchmark", execute_benchmark)

    await worker.execute_run(test_session, run)

    async with test_session_maker() as verify_session:
        refreshed_run = await verify_session.get(BenchmarkRun, run.id)
        refreshed_rb = await verify_session.get(BenchmarkRunBenchmark, run_benchmark.id)
        assert refreshed_run is not None
        assert refreshed_rb is not None
        assert refreshed_run.status == RunStatus.FAILED.value
        assert "zero balance" in (refreshed_run.error_message or "").lower()
        assert refreshed_rb.status == BenchmarkRunStatus.FAILED.value
        assert "zero balance" in (refreshed_rb.error_message or "").lower()
    execute_benchmark.assert_not_awaited()
    fake_client.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_openrouter_preflight_failure_fails_run_before_items(
    test_session, monkeypatch
) -> None:
    model = Model(
        slug="deepseek/deepseek-v4-flash-0731",
        name="DeepSeek V4 Flash 0731",
        provider="openrouter",
        is_active=True,
    )
    benchmark = Benchmark(
        name="mmlu_pro",
        display_name="MMLU-Pro",
        adapter_class="MMLUProAdapter",
        is_enabled=True,
        supports_subset=True,
    )
    test_session.add_all([model, benchmark])
    await test_session.flush()
    run = BenchmarkRun(
        model_id=model.id,
        model_slug=model.slug,
        provider="openrouter",
        subset_pct=100,
        status=RunStatus.RUNNING.value,
    )
    test_session.add(run)
    await test_session.flush()
    run_benchmark = BenchmarkRunBenchmark(
        run_id=run.id,
        benchmark_id=benchmark.id,
        benchmark_name=benchmark.name,
        status=BenchmarkRunStatus.PENDING.value,
    )
    test_session.add(run_benchmark)
    await test_session.commit()
    await test_session.refresh(run)

    worker = BenchmarkWorker()
    test_session_maker = async_sessionmaker(
        test_session.bind,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    monkeypatch.setattr("app.worker.runner.async_session_maker", test_session_maker)
    monkeypatch.setattr(
        "app.worker.runner.collect_worker_provenance",
        AsyncMock(return_value=TEST_RUN_PROVENANCE),
    )
    fake_client = AsyncMock()
    fake_client.provider = "openrouter"
    fake_client.run_inference.side_effect = RuntimeError("provider unavailable")
    fake_judge = AsyncMock()
    monkeypatch.setattr(worker, "_get_client_for_run", AsyncMock(return_value=fake_client))
    monkeypatch.setattr("app.worker.runner.get_chutes_client", lambda: fake_judge)
    execute_benchmark = AsyncMock(side_effect=AssertionError("items must not run"))
    monkeypatch.setattr(worker, "execute_benchmark", execute_benchmark)

    await worker.execute_run(test_session, run)

    async with test_session_maker() as verify_session:
        refreshed_run = await verify_session.get(BenchmarkRun, run.id)
        refreshed_rb = await verify_session.get(BenchmarkRunBenchmark, run_benchmark.id)
        assert refreshed_run is not None
        assert refreshed_rb is not None
        assert refreshed_run.status == RunStatus.FAILED.value
        assert "provider preflight failed" in (refreshed_run.error_message or "").lower()
        assert refreshed_rb.status == BenchmarkRunStatus.FAILED.value
    execute_benchmark.assert_not_awaited()
    fake_client.close.assert_awaited_once()
    fake_judge.close.assert_awaited_once()


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
async def test_persist_item_result_progress_commits_non_checkpoint_items(
    test_session,
    monkeypatch,
) -> None:
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

    run = BenchmarkRun(
        model_id=model.id,
        model_slug=model.slug,
        provider="chutes",
        subset_pct=1,
        status=RunStatus.RUNNING.value,
    )
    test_session.add(run)
    await test_session.flush()

    run_benchmark = BenchmarkRunBenchmark(
        run_id=run.id,
        benchmark_id=benchmark.id,
        benchmark_name=benchmark.name,
        status=BenchmarkRunStatus.RUNNING.value,
        sampled_items=4,
    )
    test_session.add(run_benchmark)
    await test_session.commit()

    worker = BenchmarkWorker()
    test_session_maker = async_sessionmaker(
        test_session.bind,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    monkeypatch.setattr("app.worker.runner.async_session_maker", test_session_maker)
    add_event_mock = AsyncMock()
    monkeypatch.setattr(worker, "_safe_add_run_event", add_event_mock)

    await worker._persist_item_result_progress(
        run_id=str(run.id),
        run_benchmark_id=str(run_benchmark.id),
        result=ItemResult(item_id="item-1", is_correct=False, score=0.0, metadata={}),
        benchmark_name="livecodebench",
        current_completed=1,
        total_items=4,
        current_correct=0,
        should_persist_progress=False,
    )

    async with test_session_maker() as verify_session:
        result = await verify_session.execute(
            select(BenchmarkItemResult).where(BenchmarkItemResult.run_benchmark_id == run_benchmark.id)
        )
        saved_results = list(result.scalars().all())

    assert len(saved_results) == 1
    assert saved_results[0].item_id == "item-1"
    add_event_mock.assert_not_awaited()


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


@pytest.mark.asyncio
async def test_is_run_canceled_accepts_status_without_timestamp(test_session, monkeypatch) -> None:
    model = Model(
        slug="status-cancel-model",
        name="Status Cancel Model",
        provider="chutes",
        is_active=True,
    )
    test_session.add(model)
    await test_session.flush()
    run = BenchmarkRun(
        model_id=model.id,
        model_slug=model.slug,
        provider="chutes",
        subset_pct=100,
        status=RunStatus.CANCELED.value,
        canceled_at=None,
    )
    test_session.add(run)
    await test_session.commit()

    test_session_maker = async_sessionmaker(
        test_session.bind,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    monkeypatch.setattr("app.worker.runner.async_session_maker", test_session_maker)

    assert await BenchmarkWorker()._is_run_canceled(run.id) is True


@pytest.mark.asyncio
async def test_execute_claimed_run_polls_and_cancels_inflight_work(
    test_session,
    monkeypatch,
) -> None:
    model = Model(
        slug="cancel-model",
        name="Cancel Model",
        provider="chutes",
        is_active=True,
    )
    test_session.add(model)
    await test_session.flush()
    run = BenchmarkRun(
        model_id=model.id,
        model_slug=model.slug,
        provider="chutes",
        subset_pct=100,
        status=RunStatus.RUNNING.value,
    )
    test_session.add(run)
    await test_session.commit()

    test_session_maker = async_sessionmaker(
        test_session.bind,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    monkeypatch.setattr("app.worker.runner.async_session_maker", test_session_maker)
    monkeypatch.setattr("app.worker.runner.settings.worker_cancellation_poll_seconds", 0.01)

    worker = BenchmarkWorker()
    started = asyncio.Event()
    stopped = asyncio.Event()

    async def long_running_execute(_db, _run) -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    monkeypatch.setattr(worker, "execute_run", long_running_execute)
    monkeypatch.setattr(
        worker,
        "_is_run_canceled",
        AsyncMock(side_effect=[False, True]),
    )

    task = asyncio.create_task(worker.execute_claimed_run(run.id, run.model_slug))
    await asyncio.wait_for(started.wait(), timeout=1)
    await asyncio.wait_for(task, timeout=1)

    assert stopped.is_set()


@pytest.mark.asyncio
async def test_canceling_item_wrapper_cancels_adapter_work() -> None:
    started = asyncio.Event()
    stopped = asyncio.Event()

    class BlockingAdapter:
        async def evaluate_item(self, _item_id):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()

    task = asyncio.create_task(_evaluate_adapter_item(BlockingAdapter(), "item", 60))
    await asyncio.wait_for(started.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert stopped.is_set()
