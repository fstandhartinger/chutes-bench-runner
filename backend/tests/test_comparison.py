"""Tests for model comparison functionality (RLM vs Base model).

These tests verify the comparison service and API endpoints for spec 003.
"""
import pytest
from uuid import uuid4

from app.models.benchmark import Benchmark
from app.models.comparison import ComparisonStatus, ModelComparison
from app.models.model import Model
from app.models.run import (
    BenchmarkRun,
    BenchmarkRunBenchmark,
    BenchmarkItemResult,
    BenchmarkRunStatus,
    RunStatus,
)
from app.services import comparison_service


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
async def test_models(test_session):
    """Create test models (base and RLM variant)."""
    base_model = Model(
        id=str(uuid4()),
        slug="zai-org/GLM-4.7-TEE",
        name="GLM-4.7-TEE",
        provider="chutes",
        is_active=True,
    )
    rlm_model = Model(
        id=str(uuid4()),
        slug="zai-org/GLM-4.7-TEE-RLM",
        name="GLM-4.7-TEE-RLM",
        provider="chutes",
        is_active=True,
    )
    test_session.add(base_model)
    test_session.add(rlm_model)
    await test_session.commit()
    return base_model, rlm_model


@pytest.fixture
async def test_benchmarks(test_session):
    """Create test benchmarks."""
    s_niah = Benchmark(
        id=str(uuid4()),
        name="s_niah",
        display_name="S-NIAH",
        adapter_class="SNIAHAdapter",
        is_enabled=True,
        supports_subset=True,
        total_items=60,
    )
    oolong = Benchmark(
        id=str(uuid4()),
        name="oolong",
        display_name="OOLONG",
        adapter_class="OolongAdapter",
        is_enabled=True,
        supports_subset=True,
        total_items=100,
    )
    oolong_pairs = Benchmark(
        id=str(uuid4()),
        name="oolong_pairs",
        display_name="OOLONG-Pairs",
        adapter_class="OolongPairsAdapter",
        is_enabled=True,
        supports_subset=True,
        total_items=100,
    )
    test_session.add_all([s_niah, oolong, oolong_pairs])
    await test_session.commit()
    return s_niah, oolong, oolong_pairs


@pytest.fixture
async def completed_runs(test_session, test_models, test_benchmarks):
    """Create completed benchmark runs for base and RLM models."""
    base_model, rlm_model = test_models
    s_niah, oolong, oolong_pairs = test_benchmarks

    # Create base model run
    base_run = BenchmarkRun(
        id=str(uuid4()),
        model_id=base_model.id,
        model_slug=base_model.slug,
        provider="chutes",
        subset_pct=100,
        subset_seed="test-seed-123",
        status=RunStatus.SUCCEEDED.value,
        overall_score=0.65,
    )
    test_session.add(base_run)

    # Create RLM model run
    rlm_run = BenchmarkRun(
        id=str(uuid4()),
        model_id=rlm_model.id,
        model_slug=rlm_model.slug,
        provider="chutes",
        subset_pct=100,
        subset_seed="test-seed-123",
        status=RunStatus.SUCCEEDED.value,
        overall_score=0.85,
    )
    test_session.add(rlm_run)
    await test_session.flush()

    # Create benchmark run entries for base model
    base_s_niah = BenchmarkRunBenchmark(
        id=str(uuid4()),
        run_id=base_run.id,
        benchmark_id=s_niah.id,
        benchmark_name="s_niah",
        status=BenchmarkRunStatus.SUCCEEDED.value,
        total_items=60,
        completed_items=60,
        sampled_items=60,
        score=0.60,
        metrics={
            "accuracy_8k": 0.90,
            "accuracy_16k": 0.80,
            "accuracy_32k": 0.60,
            "accuracy_64k": 0.40,
            "accuracy_128k": 0.30,
            "accuracy_256k": 0.20,
        },
    )
    base_oolong = BenchmarkRunBenchmark(
        id=str(uuid4()),
        run_id=base_run.id,
        benchmark_id=oolong.id,
        benchmark_name="oolong",
        status=BenchmarkRunStatus.SUCCEEDED.value,
        total_items=100,
        completed_items=100,
        sampled_items=100,
        score=0.70,
        metrics={},
    )
    test_session.add_all([base_s_niah, base_oolong])

    # Create benchmark run entries for RLM model (better scores on longer contexts)
    rlm_s_niah = BenchmarkRunBenchmark(
        id=str(uuid4()),
        run_id=rlm_run.id,
        benchmark_id=s_niah.id,
        benchmark_name="s_niah",
        status=BenchmarkRunStatus.SUCCEEDED.value,
        total_items=60,
        completed_items=60,
        sampled_items=60,
        score=0.85,
        metrics={
            "accuracy_8k": 0.92,
            "accuracy_16k": 0.90,
            "accuracy_32k": 0.88,
            "accuracy_64k": 0.85,
            "accuracy_128k": 0.80,
            "accuracy_256k": 0.75,
        },
    )
    rlm_oolong = BenchmarkRunBenchmark(
        id=str(uuid4()),
        run_id=rlm_run.id,
        benchmark_id=oolong.id,
        benchmark_name="oolong",
        status=BenchmarkRunStatus.SUCCEEDED.value,
        total_items=100,
        completed_items=100,
        sampled_items=100,
        score=0.85,
        metrics={},
    )
    test_session.add_all([rlm_s_niah, rlm_oolong])

    # Add some item results for statistics
    for i in range(5):
        base_item = BenchmarkItemResult(
            id=str(uuid4()),
            run_benchmark_id=base_s_niah.id,
            item_id=str(i),
            is_correct=(i < 3),  # 3/5 correct
            score=1.0 if i < 3 else 0.0,
            latency_ms=1000 + i * 100,
            input_tokens=1000,
            output_tokens=50,
            item_metadata={"context_size": 8192, "needle_position": 0.5},
        )
        rlm_item = BenchmarkItemResult(
            id=str(uuid4()),
            run_benchmark_id=rlm_s_niah.id,
            item_id=str(i),
            is_correct=(i < 4),  # 4/5 correct
            score=1.0 if i < 4 else 0.0,
            latency_ms=800 + i * 50,  # RLM is faster
            input_tokens=500,  # RLM uses less tokens due to slicing
            output_tokens=50,
            item_metadata={"context_size": 8192, "needle_position": 0.5},
        )
        test_session.add_all([base_item, rlm_item])

    await test_session.commit()
    await test_session.refresh(base_run)
    await test_session.refresh(rlm_run)

    return base_run, rlm_run


# =============================================================================
# Comparison Service Tests
# =============================================================================

class TestComparisonService:
    """Tests for comparison service."""

    @pytest.mark.asyncio
    async def test_create_comparison_from_completed_runs(self, test_session, completed_runs):
        """Test creating a comparison from two completed runs."""
        base_run, rlm_run = completed_runs

        comparison = await comparison_service.create_comparison(
            test_session,
            base_run_id=base_run.id,
            rlm_run_id=rlm_run.id,
            name="Test Comparison",
            description="Testing RLM vs base model",
        )

        assert comparison is not None
        assert comparison.base_run_id == base_run.id
        assert comparison.rlm_run_id == rlm_run.id
        assert comparison.base_model_slug == "zai-org/GLM-4.7-TEE"
        assert comparison.rlm_model_slug == "zai-org/GLM-4.7-TEE-RLM"
        assert comparison.status == ComparisonStatus.SUCCEEDED.value
        assert comparison.results is not None
        assert comparison.markdown_report is not None

    @pytest.mark.asyncio
    async def test_comparison_has_correct_benchmarks(self, test_session, completed_runs):
        """Test that comparison finds common benchmarks."""
        base_run, rlm_run = completed_runs

        comparison = await comparison_service.create_comparison(
            test_session,
            base_run_id=base_run.id,
            rlm_run_id=rlm_run.id,
        )

        # Both runs have s_niah and oolong
        assert "s_niah" in comparison.benchmarks
        assert "oolong" in comparison.benchmarks

    @pytest.mark.asyncio
    async def test_comparison_results_structure(self, test_session, completed_runs):
        """Test that comparison results have expected structure."""
        base_run, rlm_run = completed_runs

        comparison = await comparison_service.create_comparison(
            test_session,
            base_run_id=base_run.id,
            rlm_run_id=rlm_run.id,
        )

        results = comparison.results
        assert "comparison_id" in results
        assert "base_model" in results
        assert "rlm_model" in results
        assert "benchmarks" in results
        assert "summary" in results

        # Check summary
        summary = results["summary"]
        assert "average_base_score" in summary
        assert "average_rlm_score" in summary
        assert "improvement" in summary
        assert "improvement_pct" in summary

    @pytest.mark.asyncio
    async def test_comparison_context_size_breakdown(self, test_session, completed_runs):
        """Test that S-NIAH comparison includes context size breakdown."""
        base_run, rlm_run = completed_runs

        comparison = await comparison_service.create_comparison(
            test_session,
            base_run_id=base_run.id,
            rlm_run_id=rlm_run.id,
        )

        s_niah_results = comparison.results["benchmarks"]["s_niah"]
        assert "by_context_size" in s_niah_results

        ctx_comparison = s_niah_results["by_context_size"]
        assert "8k" in ctx_comparison
        assert "256k" in ctx_comparison

        # RLM should show improvement especially on longer contexts
        for ctx_size, data in ctx_comparison.items():
            assert "base_accuracy" in data
            assert "rlm_accuracy" in data
            assert "improvement" in data

    @pytest.mark.asyncio
    async def test_comparison_shows_rlm_improvement(self, test_session, completed_runs):
        """Test that comparison shows RLM improvement."""
        base_run, rlm_run = completed_runs

        comparison = await comparison_service.create_comparison(
            test_session,
            base_run_id=base_run.id,
            rlm_run_id=rlm_run.id,
        )

        summary = comparison.results["summary"]
        # RLM should have higher score
        assert summary["average_rlm_score"] > summary["average_base_score"]
        assert summary["improvement"] > 0
        assert summary["improvement_pct"] > 0

    @pytest.mark.asyncio
    async def test_markdown_report_generated(self, test_session, completed_runs):
        """Test that markdown report is generated."""
        base_run, rlm_run = completed_runs

        comparison = await comparison_service.create_comparison(
            test_session,
            base_run_id=base_run.id,
            rlm_run_id=rlm_run.id,
        )

        markdown = comparison.markdown_report
        assert markdown is not None
        assert "# Model Comparison Report" in markdown
        assert comparison.base_model_slug in markdown
        assert comparison.rlm_model_slug in markdown
        assert "| Context Size |" in markdown  # Table header for context breakdown

    @pytest.mark.asyncio
    async def test_create_comparison_invalid_run(self, test_session):
        """Test creating comparison with invalid run ID."""
        with pytest.raises(ValueError, match="not found"):
            await comparison_service.create_comparison(
                test_session,
                base_run_id=str(uuid4()),
                rlm_run_id=str(uuid4()),
            )

    @pytest.mark.asyncio
    async def test_list_comparisons(self, test_session, completed_runs):
        """Test listing comparisons."""
        base_run, rlm_run = completed_runs

        # Create a comparison
        await comparison_service.create_comparison(
            test_session,
            base_run_id=base_run.id,
            rlm_run_id=rlm_run.id,
        )

        comparisons, total = await comparison_service.list_comparisons(test_session)
        assert total == 1
        assert len(comparisons) == 1

    @pytest.mark.asyncio
    async def test_get_comparison(self, test_session, completed_runs):
        """Test getting a comparison by ID."""
        base_run, rlm_run = completed_runs

        created = await comparison_service.create_comparison(
            test_session,
            base_run_id=base_run.id,
            rlm_run_id=rlm_run.id,
        )

        fetched = await comparison_service.get_comparison(test_session, created.id)
        assert fetched is not None
        assert fetched.id == created.id

    @pytest.mark.asyncio
    async def test_get_comparison_json(self, test_session, completed_runs):
        """Test getting comparison results as JSON."""
        base_run, rlm_run = completed_runs

        created = await comparison_service.create_comparison(
            test_session,
            base_run_id=base_run.id,
            rlm_run_id=rlm_run.id,
        )

        json_results = await comparison_service.get_comparison_json(test_session, created.id)
        assert json_results is not None
        assert json_results["comparison_id"] == created.id


# =============================================================================
# Context Size Comparison Tests
# =============================================================================

class TestContextSizeComparison:
    """Tests for accuracy vs context length comparison."""

    @pytest.mark.asyncio
    async def test_context_size_metrics_extracted(self, test_session, completed_runs):
        """Test that context size metrics are correctly extracted."""
        base_run, rlm_run = completed_runs

        comparison = await comparison_service.create_comparison(
            test_session,
            base_run_id=base_run.id,
            rlm_run_id=rlm_run.id,
        )

        ctx_comparison = comparison.results["benchmarks"]["s_niah"]["by_context_size"]

        # Verify all context sizes are present
        expected_sizes = ["8k", "16k", "32k", "64k", "128k", "256k"]
        for size in expected_sizes:
            assert size in ctx_comparison

    @pytest.mark.asyncio
    async def test_larger_context_shows_bigger_improvement(self, test_session, completed_runs):
        """Test that RLM shows bigger improvement on larger contexts."""
        base_run, rlm_run = completed_runs

        comparison = await comparison_service.create_comparison(
            test_session,
            base_run_id=base_run.id,
            rlm_run_id=rlm_run.id,
        )

        ctx_comparison = comparison.results["benchmarks"]["s_niah"]["by_context_size"]

        # RLM improvement should be larger on longer contexts
        improvement_8k = ctx_comparison["8k"]["improvement"]
        improvement_256k = ctx_comparison["256k"]["improvement"]

        # The test data is set up so RLM improves more on longer contexts
        assert improvement_256k > improvement_8k


# =============================================================================
# API Endpoint Tests
# =============================================================================

class TestComparisonAPI:
    """Tests for comparison API endpoints."""

    def test_create_comparison_endpoint(self, client, completed_runs):
        """Test POST /api/comparisons endpoint."""
        base_run, rlm_run = completed_runs

        response = client.post(
            "/api/comparisons",
            json={
                "base_run_id": base_run.id,
                "rlm_run_id": rlm_run.id,
                "name": "API Test Comparison",
            },
        )

        assert response.status_code == 201
        data = response.json()
        assert data["base_model_slug"] == "zai-org/GLM-4.7-TEE"
        assert data["rlm_model_slug"] == "zai-org/GLM-4.7-TEE-RLM"
        assert data["status"] == "succeeded"
        assert data["results"] is not None

    def test_list_comparisons_endpoint(self, client, completed_runs):
        """Test GET /api/comparisons endpoint."""
        base_run, rlm_run = completed_runs

        # Create a comparison first
        client.post(
            "/api/comparisons",
            json={
                "base_run_id": base_run.id,
                "rlm_run_id": rlm_run.id,
            },
        )

        response = client.get("/api/comparisons")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] >= 1
        assert len(data["comparisons"]) >= 1

    def test_get_comparison_endpoint(self, client, completed_runs):
        """Test GET /api/comparisons/{id} endpoint."""
        base_run, rlm_run = completed_runs

        # Create a comparison first
        create_response = client.post(
            "/api/comparisons",
            json={
                "base_run_id": base_run.id,
                "rlm_run_id": rlm_run.id,
            },
        )
        comparison_id = create_response.json()["id"]

        response = client.get(f"/api/comparisons/{comparison_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == comparison_id
        assert data["results"] is not None

    def test_get_comparison_json_endpoint(self, client, completed_runs):
        """Test GET /api/comparisons/{id}/json endpoint."""
        base_run, rlm_run = completed_runs

        # Create a comparison first
        create_response = client.post(
            "/api/comparisons",
            json={
                "base_run_id": base_run.id,
                "rlm_run_id": rlm_run.id,
            },
        )
        comparison_id = create_response.json()["id"]

        response = client.get(f"/api/comparisons/{comparison_id}/json")
        assert response.status_code == 200
        data = response.json()
        assert "benchmarks" in data
        assert "summary" in data

    def test_get_comparison_markdown_endpoint(self, client, completed_runs):
        """Test GET /api/comparisons/{id}/markdown endpoint."""
        base_run, rlm_run = completed_runs

        # Create a comparison first
        create_response = client.post(
            "/api/comparisons",
            json={
                "base_run_id": base_run.id,
                "rlm_run_id": rlm_run.id,
            },
        )
        comparison_id = create_response.json()["id"]

        response = client.get(f"/api/comparisons/{comparison_id}/markdown")
        assert response.status_code == 200
        assert "text/markdown" in response.headers.get("content-type", "")
        content = response.content.decode()
        assert "# Model Comparison Report" in content

    def test_create_comparison_invalid_run(self, client):
        """Test creating comparison with invalid run IDs."""
        response = client.post(
            "/api/comparisons",
            json={
                "base_run_id": str(uuid4()),
                "rlm_run_id": str(uuid4()),
            },
        )
        assert response.status_code == 400

    def test_get_comparison_not_found(self, client):
        """Test getting non-existent comparison."""
        response = client.get(f"/api/comparisons/{uuid4()}")
        assert response.status_code == 404
