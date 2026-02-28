#!/usr/bin/env python3
"""Run DeepResearch-Bench matrix using the bench-runner adapter implementation.

This script executes the same adapter used by the worker and prints a summary in
the familiar format:

=== Evaluation Results Summary ===
Comprehensiveness
Insight
Instruction Following
Readability
Overall Score
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# Allow running as a standalone script from any working directory.
BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.benchmarks.base import ItemResult
from app.services.chutes_client import get_chutes_client

logger = logging.getLogger(__name__)


def _load_deepresearch_adapter_class():
    """Load DeepResearch adapter without importing all adapters package side effects."""
    module_path = BACKEND_ROOT / "app" / "benchmarks" / "adapters" / "deepresearch_bench.py"
    spec = importlib.util.spec_from_file_location("deepresearch_bench_local", module_path)
    if not spec or not spec.loader:
        raise RuntimeError(f"Unable to load adapter module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    adapter_class = getattr(module, "DeepResearchBenchAdapter", None)
    if adapter_class is None:
        raise RuntimeError("DeepResearchBenchAdapter class not found in module")
    return adapter_class


@dataclass
class CaseConfig:
    name: str
    search_api_url: str
    mode: str
    optimization_mode: str


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _serialize_item_result(result: ItemResult) -> dict[str, Any]:
    payload = asdict(result)
    payload["metadata"] = payload.get("metadata") or {}
    payload["judge_output"] = payload.get("judge_output") or {}
    return payload


async def _run_case(
    case: CaseConfig,
    judge_model: str,
    subset_count: int,
    seed: str,
    output_dir: Path,
) -> dict[str, Any]:
    deepresearch_adapter_class = _load_deepresearch_adapter_class()
    client = get_chutes_client(api_key=os.environ.get("CHUTES_API_KEY"))
    adapter = deepresearch_adapter_class(client, judge_model)
    adapter.run_config = {
        "deepresearch": {
            "search_api_url": case.search_api_url,
            "mode": case.mode,
            "optimization_mode": case.optimization_mode,
        }
    }

    started = time.time()
    await adapter.preload()
    total_items, items_to_evaluate = await adapter.get_items_for_evaluation(
        subset_pct=100,
        seed=f"{seed}:{case.name}",
        subset_count=subset_count,
    )

    concurrency = adapter.get_item_concurrency() or 1
    semaphore = asyncio.Semaphore(concurrency)
    results: list[ItemResult] = []
    completed = 0

    async def evaluate_one(item_id: str) -> ItemResult:
        async with semaphore:
            return await adapter.evaluate_item(item_id)

    tasks = [asyncio.create_task(evaluate_one(item_id)) for item_id in items_to_evaluate]
    for future in asyncio.as_completed(tasks):
        result = await future
        results.append(result)
        completed += 1
        if completed == 1 or completed % 5 == 0 or completed == len(items_to_evaluate):
            logger.info("[%s] Progress %d/%d", case.name, completed, len(items_to_evaluate))

    metrics = await adapter.postprocess(results)
    elapsed = time.time() - started

    case_dir = output_dir / case.name
    case_dir.mkdir(parents=True, exist_ok=True)
    raw_results_path = case_dir / "raw_results.jsonl"
    summary_path = case_dir / "summary.json"

    # Keep deterministic output ordering by numeric item_id.
    def item_sort_key(r: ItemResult) -> tuple[int, str]:
        try:
            return (int(r.item_id), r.item_id)
        except ValueError:
            return (10**9, r.item_id)

    ordered_results = sorted(results, key=item_sort_key)
    with raw_results_path.open("w", encoding="utf-8") as fh:
        for item in ordered_results:
            fh.write(json.dumps(_serialize_item_result(item), ensure_ascii=False) + "\n")

    summary_payload = {
        "case": asdict(case),
        "judge_model": judge_model,
        "total_items": total_items,
        "sampled_items": len(items_to_evaluate),
        "elapsed_seconds": elapsed,
        "metrics": metrics,
        "raw_results_file": str(raw_results_path),
    }
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary_payload, fh, indent=2, ensure_ascii=False)

    comprehensiveness = _safe_float(metrics.get("comprehensiveness"))
    insight = _safe_float(metrics.get("insight"))
    instruction_following = _safe_float(metrics.get("instruction_following"))
    readability = _safe_float(metrics.get("readability"))
    overall = _safe_float(metrics.get("overall_score", metrics.get("race_score")))

    logger.info("=== Evaluation Results Summary ===")
    logger.info("Comprehensiveness:      %.4f", comprehensiveness)
    logger.info("Insight:                %.4f", insight)
    logger.info("Instruction Following:  %.4f", instruction_following)
    logger.info("Readability:            %.4f", readability)
    logger.info("Overall Score:          %.4f", overall)
    logger.info("================================")
    logger.info("--- Run Summary ---")
    logger.info("Case: %s", case.name)
    logger.info("Target model: %s", judge_model)
    logger.info("Total tasks processed: %d", len(items_to_evaluate))
    logger.info("Results file: %s", raw_results_path)
    logger.info("Elapsed: %.1fs", elapsed)
    logger.info("-------------------")

    await adapter.cleanup()
    await client.close()

    return {
        "case": case.name,
        "comprehensiveness": comprehensiveness,
        "insight": insight,
        "instruction_following": instruction_following,
        "readability": readability,
        "overall_score": overall,
        "elapsed_seconds": elapsed,
        "sampled_items": len(items_to_evaluate),
        "raw_results_file": str(raw_results_path),
        "summary_file": str(summary_path),
    }


async def _main() -> int:
    parser = argparse.ArgumentParser(description="Run DeepResearch-Bench matrix")
    parser.add_argument(
        "--judge-model",
        default="moonshotai/Kimi-K2.5-TEE",
        help="Model slug used as the LLM judge",
    )
    parser.add_argument(
        "--subset-count",
        type=int,
        default=50,
        help="Number of EN tasks to run (default: 50)",
    )
    parser.add_argument(
        "--seed",
        default="deepresearch-matrix",
        help="Deterministic subset seed",
    )
    parser.add_argument(
        "--output-dir",
        default="results/race",
        help="Directory for raw results + summaries",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        default=["fixed_light_balanced", "fixed_max_quality"],
        help="Case names to run",
    )
    parser.add_argument(
        "--fixed-url",
        default="https://search.chutes.ai/api/v1/research",
        help="Search API URL for fixed deployment",
    )
    parser.add_argument(
        "--old-url",
        default="https://chutes-search.onrender.com/api/v1/research",
        help="Search API URL for old deployment (if still available)",
    )
    args = parser.parse_args()

    if not os.environ.get("CHUTES_API_KEY"):
        raise SystemExit("CHUTES_API_KEY is required")

    level = os.environ.get("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=getattr(logging, level, logging.INFO), format="%(levelname)s:%(name)s:%(message)s")

    cases_by_name: dict[str, CaseConfig] = {
        "fixed_light_balanced": CaseConfig(
            name="fixed_light_balanced",
            search_api_url=args.fixed_url,
            mode="light",
            optimization_mode="balanced",
        ),
        "fixed_max_quality": CaseConfig(
            name="fixed_max_quality",
            search_api_url=args.fixed_url,
            mode="max",
            optimization_mode="quality",
        ),
        "old_light_balanced": CaseConfig(
            name="old_light_balanced",
            search_api_url=args.old_url,
            mode="light",
            optimization_mode="balanced",
        ),
        "old_max_quality": CaseConfig(
            name="old_max_quality",
            search_api_url=args.old_url,
            mode="max",
            optimization_mode="quality",
        ),
    }

    selected_cases: list[CaseConfig] = []
    for case_name in args.cases:
        cfg = cases_by_name.get(case_name)
        if not cfg:
            raise SystemExit(f"Unknown case: {case_name}")
        selected_cases.append(cfg)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    matrix_results: list[dict[str, Any]] = []
    for case in selected_cases:
        logger.info("Starting case %s", case.name)
        result = await _run_case(
            case=case,
            judge_model=args.judge_model,
            subset_count=args.subset_count,
            seed=args.seed,
            output_dir=output_dir,
        )
        matrix_results.append(result)

    matrix_summary_path = output_dir / "matrix_summary.json"
    with matrix_summary_path.open("w", encoding="utf-8") as fh:
        json.dump(matrix_results, fh, indent=2, ensure_ascii=False)

    logger.info("Matrix run complete: %s", matrix_summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
