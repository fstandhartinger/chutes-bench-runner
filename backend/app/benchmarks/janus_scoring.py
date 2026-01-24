"""Janus composite score calculation helpers."""

from __future__ import annotations

from typing import Any


def normalize_ttft(
    ttft_ms: float | None,
    target_ms: float = 500.0,
    ceiling_ms: float = 5000.0,
) -> float:
    """Normalize TTFT (ms) to a 0-1 score."""
    if ttft_ms is None:
        return 0.0
    if ttft_ms <= target_ms:
        return 1.0
    if ttft_ms >= ceiling_ms:
        return 0.0
    return max(0.0, 1.0 - (ttft_ms - target_ms) / (ceiling_ms - target_ms))


def normalize_tps(
    tps: float | None,
    target_tps: float = 30.0,
    min_tps: float = 5.0,
) -> float:
    """Normalize tokens-per-second to a 0-1 score."""
    if tps is None:
        return 0.0
    if tps <= min_tps:
        return 0.0
    if tps >= target_tps:
        return 1.0
    return max(0.0, (tps - min_tps) / (target_tps - min_tps))


def calculate_janus_composite_score(run_results: dict[str, dict[str, Any]]) -> dict[str, float]:
    """Calculate Janus competition composite score (0-1 scale)."""
    scores: dict[str, float] = {}

    research_score = run_results.get("janus_research", {}).get("score", 0.0)
    tool_score = run_results.get("janus_tool_use", {}).get("score", 0.0)
    scores["quality"] = (research_score + tool_score) / 2

    streaming_data = run_results.get("janus_streaming", {})
    streaming_metrics = streaming_data.get("metrics", {})
    ttft_score = normalize_ttft(streaming_metrics.get("avg_ttft_ms"))
    tps_score = normalize_tps(streaming_metrics.get("avg_tps"))
    scores["speed"] = (ttft_score + tps_score) / 2

    cost_data = run_results.get("janus_cost", {})
    scores["cost"] = cost_data.get("score", 0.0)

    scores["streaming"] = streaming_metrics.get(
        "continuity_score",
        streaming_data.get("score", 0.0),
    )

    scores["modality"] = run_results.get("janus_multimodal", {}).get("score", 0.0)

    scores["composite"] = (
        scores["quality"] * 0.40
        + scores["speed"] * 0.20
        + scores["cost"] * 0.15
        + scores["streaming"] * 0.15
        + scores["modality"] * 0.10
    )

    return scores
