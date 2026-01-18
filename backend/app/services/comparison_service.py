"""Comparison service for RLM vs Base model benchmark comparisons.

Supports spec 003: systematically comparing base models vs RLM variants
on long-context benchmarks (S-NIAH, OOLONG, OOLONG-Pairs).
"""
from datetime import datetime
from typing import Any, Optional
from uuid import uuid4

import sqlalchemy as sa
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.models.comparison import ComparisonStatus, ModelComparison
from app.models.run import BenchmarkRun, BenchmarkRunBenchmark, BenchmarkItemResult, RunStatus

logger = get_logger(__name__)

# Default long-context benchmarks for comparison
LONG_CONTEXT_BENCHMARKS = ["s_niah", "oolong", "oolong_pairs"]


async def create_comparison(
    db: AsyncSession,
    base_run_id: str,
    rlm_run_id: str,
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> ModelComparison:
    """
    Create a model comparison from two existing benchmark runs.

    Args:
        db: Database session
        base_run_id: UUID of the baseline model benchmark run
        rlm_run_id: UUID of the RLM model benchmark run
        name: Optional name for the comparison
        description: Optional description

    Returns:
        Created ModelComparison

    Raises:
        ValueError: If runs not found or invalid
    """
    # Fetch runs
    base_run = await db.get(BenchmarkRun, base_run_id)
    rlm_run = await db.get(BenchmarkRun, rlm_run_id)

    if not base_run:
        raise ValueError(f"Base run not found: {base_run_id}")
    if not rlm_run:
        raise ValueError(f"RLM run not found: {rlm_run_id}")

    # Get benchmarks from both runs
    base_benchmarks = {rb.benchmark_name for rb in base_run.benchmarks}
    rlm_benchmarks = {rb.benchmark_name for rb in rlm_run.benchmarks}
    common_benchmarks = sorted(base_benchmarks & rlm_benchmarks)

    if not common_benchmarks:
        raise ValueError("No common benchmarks between runs")

    # Determine status based on run statuses
    if base_run.status == RunStatus.SUCCEEDED.value and rlm_run.status == RunStatus.SUCCEEDED.value:
        status = ComparisonStatus.SUCCEEDED
    elif base_run.status in (RunStatus.FAILED.value, RunStatus.CANCELED.value) or \
         rlm_run.status in (RunStatus.FAILED.value, RunStatus.CANCELED.value):
        status = ComparisonStatus.FAILED
    elif base_run.status == RunStatus.RUNNING.value or rlm_run.status == RunStatus.RUNNING.value:
        status = ComparisonStatus.RUNNING
    else:
        status = ComparisonStatus.PENDING

    comparison = ModelComparison(
        id=str(uuid4()),
        base_run_id=base_run_id,
        base_model_slug=base_run.model_slug,
        rlm_run_id=rlm_run_id,
        rlm_model_slug=rlm_run.model_slug,
        name=name or f"{base_run.model_slug} vs {rlm_run.model_slug}",
        description=description,
        subset_seed=base_run.subset_seed or rlm_run.subset_seed,
        benchmarks=common_benchmarks,
        status=status.value,
    )

    db.add(comparison)
    await db.flush()

    # If both runs are complete, generate results immediately
    if status == ComparisonStatus.SUCCEEDED:
        await generate_comparison_results(db, comparison)

    await db.commit()
    await db.refresh(comparison)

    logger.info(
        "Created model comparison",
        comparison_id=comparison.id,
        base=base_run.model_slug,
        rlm=rlm_run.model_slug,
        benchmarks=common_benchmarks,
    )
    return comparison


async def get_comparison(db: AsyncSession, comparison_id: str) -> Optional[ModelComparison]:
    """Get a comparison by ID."""
    return await db.get(ModelComparison, comparison_id)


async def list_comparisons(
    db: AsyncSession,
    status: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> tuple[list[ModelComparison], int]:
    """List comparisons with optional filtering."""
    from sqlalchemy import func

    query = select(ModelComparison)
    count_query = select(func.count()).select_from(ModelComparison)

    if status:
        query = query.where(ModelComparison.status == status)
        count_query = count_query.where(ModelComparison.status == status)

    query = query.order_by(ModelComparison.created_at.desc()).offset(offset).limit(limit)

    result = await db.execute(query)
    count_result = await db.execute(count_query)
    total = int(count_result.scalar() or 0)

    return list(result.scalars().all()), total


async def update_comparison_status(db: AsyncSession, comparison_id: str) -> Optional[ModelComparison]:
    """
    Update comparison status based on underlying run statuses.
    Generate results if both runs are complete.
    """
    comparison = await db.get(ModelComparison, comparison_id)
    if not comparison:
        return None

    base_run = await db.get(BenchmarkRun, comparison.base_run_id)
    rlm_run = await db.get(BenchmarkRun, comparison.rlm_run_id)

    if not base_run or not rlm_run:
        comparison.status = ComparisonStatus.FAILED.value
        comparison.error_message = "One or both benchmark runs not found"
        await db.commit()
        return comparison

    # Determine new status
    if base_run.status == RunStatus.SUCCEEDED.value and rlm_run.status == RunStatus.SUCCEEDED.value:
        comparison.status = ComparisonStatus.SUCCEEDED.value
        await generate_comparison_results(db, comparison)
        comparison.completed_at = datetime.utcnow()
    elif base_run.status in (RunStatus.FAILED.value, RunStatus.CANCELED.value):
        comparison.status = ComparisonStatus.FAILED.value
        comparison.error_message = f"Base run failed: {base_run.error_message}"
    elif rlm_run.status in (RunStatus.FAILED.value, RunStatus.CANCELED.value):
        comparison.status = ComparisonStatus.FAILED.value
        comparison.error_message = f"RLM run failed: {rlm_run.error_message}"
    elif base_run.status == RunStatus.RUNNING.value or rlm_run.status == RunStatus.RUNNING.value:
        comparison.status = ComparisonStatus.RUNNING.value
    else:
        comparison.status = ComparisonStatus.PENDING.value

    comparison.updated_at = datetime.utcnow()
    await db.commit()
    await db.refresh(comparison)

    return comparison


async def generate_comparison_results(db: AsyncSession, comparison: ModelComparison) -> dict[str, Any]:
    """
    Generate comparison results and markdown report.

    This is the core function that computes accuracy vs context length comparisons.
    """
    results: dict[str, Any] = {
        "comparison_id": comparison.id,
        "base_model": comparison.base_model_slug,
        "rlm_model": comparison.rlm_model_slug,
        "subset_seed": comparison.subset_seed,
        "benchmarks": {},
        "summary": {},
    }

    total_base_score = 0.0
    total_rlm_score = 0.0
    total_benchmarks = 0

    for benchmark_name in comparison.benchmarks or []:
        benchmark_results = await _compute_benchmark_comparison(
            db,
            comparison.base_run_id,
            comparison.rlm_run_id,
            benchmark_name,
        )
        results["benchmarks"][benchmark_name] = benchmark_results

        if benchmark_results.get("base_score") is not None:
            total_base_score += benchmark_results["base_score"]
            total_rlm_score += benchmark_results.get("rlm_score", 0)
            total_benchmarks += 1

    # Compute summary
    if total_benchmarks > 0:
        avg_base = total_base_score / total_benchmarks
        avg_rlm = total_rlm_score / total_benchmarks
        results["summary"] = {
            "average_base_score": avg_base,
            "average_rlm_score": avg_rlm,
            "improvement": avg_rlm - avg_base,
            "improvement_pct": ((avg_rlm - avg_base) / avg_base * 100) if avg_base > 0 else 0,
            "total_benchmarks": total_benchmarks,
        }

    comparison.results = results
    comparison.markdown_report = _generate_markdown_report(results, comparison)

    return results


async def _compute_benchmark_comparison(
    db: AsyncSession,
    base_run_id: str,
    rlm_run_id: str,
    benchmark_name: str,
) -> dict[str, Any]:
    """Compute comparison metrics for a single benchmark."""
    # Get benchmark run records
    base_rb_result = await db.execute(
        select(BenchmarkRunBenchmark).where(
            BenchmarkRunBenchmark.run_id == base_run_id,
            BenchmarkRunBenchmark.benchmark_name == benchmark_name,
        )
    )
    base_rb = base_rb_result.scalar_one_or_none()

    rlm_rb_result = await db.execute(
        select(BenchmarkRunBenchmark).where(
            BenchmarkRunBenchmark.run_id == rlm_run_id,
            BenchmarkRunBenchmark.benchmark_name == benchmark_name,
        )
    )
    rlm_rb = rlm_rb_result.scalar_one_or_none()

    if not base_rb or not rlm_rb:
        return {"error": "Benchmark not found in one or both runs"}

    result: dict[str, Any] = {
        "benchmark_name": benchmark_name,
        "base_score": base_rb.score,
        "rlm_score": rlm_rb.score,
        "base_metrics": base_rb.metrics or {},
        "rlm_metrics": rlm_rb.metrics or {},
        "score_improvement": (rlm_rb.score or 0) - (base_rb.score or 0),
    }

    # Compute context size comparison if available (for S-NIAH)
    if base_rb.metrics and rlm_rb.metrics:
        context_comparison = _compute_context_size_comparison(
            base_rb.metrics,
            rlm_rb.metrics,
        )
        if context_comparison:
            result["by_context_size"] = context_comparison

    # Get item-level statistics
    base_items = await _get_item_stats(db, base_rb.id)
    rlm_items = await _get_item_stats(db, rlm_rb.id)

    result["base_item_stats"] = base_items
    result["rlm_item_stats"] = rlm_items

    return result


def _compute_context_size_comparison(
    base_metrics: dict[str, Any],
    rlm_metrics: dict[str, Any],
) -> Optional[dict[str, Any]]:
    """
    Compute accuracy comparison by context size.

    Looks for metrics like accuracy_8k, accuracy_16k, etc.
    """
    context_sizes = []

    # Find all context size metrics (e.g., accuracy_8k, accuracy_16k, etc.)
    for key in base_metrics:
        if key.startswith("accuracy_") and key.endswith("k"):
            try:
                size_str = key.replace("accuracy_", "").replace("k", "")
                size = int(size_str)
                context_sizes.append((size, key))
            except ValueError:
                continue

    if not context_sizes:
        return None

    # Sort by context size
    context_sizes.sort(key=lambda x: x[0])

    comparison = {}
    for size, key in context_sizes:
        base_acc = base_metrics.get(key)
        rlm_acc = rlm_metrics.get(key)

        if base_acc is not None or rlm_acc is not None:
            comparison[f"{size}k"] = {
                "base_accuracy": base_acc,
                "rlm_accuracy": rlm_acc,
                "improvement": (rlm_acc or 0) - (base_acc or 0) if base_acc is not None else None,
            }

    return comparison


async def _get_item_stats(db: AsyncSession, run_benchmark_id: str) -> dict[str, Any]:
    """Get aggregate statistics for benchmark items."""
    stats_result = await db.execute(
        select(
            func.count(BenchmarkItemResult.id).label("total"),
            func.coalesce(func.sum(
                func.cast(BenchmarkItemResult.is_correct == True, sa.Integer)  # noqa: E712
            ), 0).label("correct"),
            func.avg(BenchmarkItemResult.latency_ms).label("avg_latency"),
            func.sum(BenchmarkItemResult.input_tokens).label("total_input_tokens"),
            func.sum(BenchmarkItemResult.output_tokens).label("total_output_tokens"),
        ).where(BenchmarkItemResult.run_benchmark_id == run_benchmark_id)
    )
    row = stats_result.one()

    total = int(row.total or 0)
    correct = int(row.correct or 0)

    return {
        "total_items": total,
        "correct_items": correct,
        "accuracy": correct / total if total > 0 else 0,
        "avg_latency_ms": float(row.avg_latency) if row.avg_latency else None,
        "total_input_tokens": int(row.total_input_tokens or 0),
        "total_output_tokens": int(row.total_output_tokens or 0),
    }


def _generate_markdown_report(results: dict[str, Any], comparison: ModelComparison) -> str:
    """Generate a markdown report from comparison results."""
    lines = [
        f"# Model Comparison Report",
        f"",
        f"**Comparison ID:** {comparison.id}",
        f"",
        f"## Models",
        f"",
        f"| Model | Type | Slug |",
        f"|-------|------|------|",
        f"| Baseline | Base | {comparison.base_model_slug} |",
        f"| RLM | RLM Variant | {comparison.rlm_model_slug} |",
        f"",
    ]

    if comparison.subset_seed:
        lines.append(f"**Subset Seed:** {comparison.subset_seed}")
        lines.append("")

    # Summary
    summary = results.get("summary", {})
    if summary:
        lines.extend([
            "## Summary",
            "",
            f"| Metric | Base | RLM | Improvement |",
            f"|--------|------|-----|-------------|",
            f"| Average Score | {summary.get('average_base_score', 0):.2%} | {summary.get('average_rlm_score', 0):.2%} | {summary.get('improvement', 0):+.2%} ({summary.get('improvement_pct', 0):+.1f}%) |",
            "",
        ])

    # Per-benchmark results
    lines.extend([
        "## Benchmark Results",
        "",
    ])

    for benchmark_name, benchmark_data in results.get("benchmarks", {}).items():
        if isinstance(benchmark_data, dict) and "error" not in benchmark_data:
            base_score = benchmark_data.get("base_score", 0) or 0
            rlm_score = benchmark_data.get("rlm_score", 0) or 0
            improvement = benchmark_data.get("score_improvement", 0) or 0

            lines.extend([
                f"### {benchmark_name}",
                f"",
                f"| Metric | Base | RLM | Improvement |",
                f"|--------|------|-----|-------------|",
                f"| Score | {base_score:.2%} | {rlm_score:.2%} | {improvement:+.2%} |",
                "",
            ])

            # Context size breakdown if available
            ctx_comparison = benchmark_data.get("by_context_size")
            if ctx_comparison:
                lines.extend([
                    "#### Accuracy by Context Size",
                    "",
                    "| Context Size | Base | RLM | Improvement |",
                    "|--------------|------|-----|-------------|",
                ])
                for ctx_size, ctx_data in ctx_comparison.items():
                    base_acc = ctx_data.get("base_accuracy")
                    rlm_acc = ctx_data.get("rlm_accuracy")
                    ctx_imp = ctx_data.get("improvement")

                    base_str = f"{base_acc:.2%}" if base_acc is not None else "N/A"
                    rlm_str = f"{rlm_acc:.2%}" if rlm_acc is not None else "N/A"
                    imp_str = f"{ctx_imp:+.2%}" if ctx_imp is not None else "N/A"

                    lines.append(f"| {ctx_size} | {base_str} | {rlm_str} | {imp_str} |")
                lines.append("")

            # Item statistics
            base_stats = benchmark_data.get("base_item_stats", {})
            rlm_stats = benchmark_data.get("rlm_item_stats", {})

            if base_stats or rlm_stats:
                lines.extend([
                    "#### Resource Usage",
                    "",
                    "| Metric | Base | RLM |",
                    "|--------|------|-----|",
                ])
                if base_stats.get("avg_latency_ms") or rlm_stats.get("avg_latency_ms"):
                    base_lat = base_stats.get("avg_latency_ms")
                    rlm_lat = rlm_stats.get("avg_latency_ms")
                    lines.append(
                        f"| Avg Latency | {base_lat:.0f}ms | {rlm_lat:.0f}ms |"
                        if base_lat and rlm_lat else
                        f"| Avg Latency | {base_lat or 'N/A'} | {rlm_lat or 'N/A'} |"
                    )

                base_tokens = base_stats.get("total_input_tokens", 0) + base_stats.get("total_output_tokens", 0)
                rlm_tokens = rlm_stats.get("total_input_tokens", 0) + rlm_stats.get("total_output_tokens", 0)
                lines.append(f"| Total Tokens | {base_tokens:,} | {rlm_tokens:,} |")
                lines.append("")

    # Footer
    lines.extend([
        "---",
        f"*Generated at {datetime.utcnow().isoformat()}Z*",
        "",
        "*Based on RLM paper benchmarks: https://arxiv.org/html/2512.24601v1*",
    ])

    return "\n".join(lines)


async def get_comparison_json(db: AsyncSession, comparison_id: str) -> Optional[dict[str, Any]]:
    """Get comparison results as JSON."""
    comparison = await get_comparison(db, comparison_id)
    if not comparison:
        return None

    return comparison.results


async def get_comparison_markdown(db: AsyncSession, comparison_id: str) -> Optional[str]:
    """Get comparison markdown report."""
    comparison = await get_comparison(db, comparison_id)
    if not comparison:
        return None

    return comparison.markdown_report
