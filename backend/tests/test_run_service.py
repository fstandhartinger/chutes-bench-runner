import pytest

from app.models.benchmark import Benchmark
from app.models.model import Model
from app.models.run import BenchmarkRun, BenchmarkRunBenchmark, BenchmarkRunStatus, RunStatus
from app.services.run_service import requeue_run


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
