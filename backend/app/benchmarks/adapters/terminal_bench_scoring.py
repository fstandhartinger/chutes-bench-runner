"""Scoring audit for the pinned Terminal-Bench 2.1 verifier tests.

This is deliberately tied to one immutable task-content commit.  It is not a
heuristic classifier and must not be reused for another Terminal-Bench release
without inspecting that release's ``tests/`` directories.
"""

from __future__ import annotations

from typing import Any

TERMINAL_BENCH_2_1_SCORING_AUDIT_COMMIT = "5c8eadf1f393183288fa08b8f73ca9a469cc5e00"
TERMINAL_BENCH_2_1_SCORING_AUDIT_TASK_COUNT = 89

FUNCTIONAL = "functional"
PERFORMANCE_GATED = "performance_gated"
RESOURCE_GATED = "resource_gated"
NON_FUNCTIONAL_SCORING_CLASSES = frozenset({PERFORMANCE_GATED, RESOURCE_GATED})


def _gate(
    file: str,
    line: int,
    assertion: str,
    gate_kind: str,
) -> dict[str, Any]:
    return {
        "file": file,
        "line": line,
        "assertion": assertion,
        "gate_kind": gate_kind,
    }


# Every entry below is evidence from the top-level verifier ``tests/``
# directory in the pinned archive.  A task with both performance and resource
# gates is classified as performance_gated because host timing is the stronger
# source of non-determinism.  Unlisted tasks are functional only after the
# complete 89-task audit recorded in docs/TERMINAL_BENCH_2_1_SCORING_AUDIT.md.
TERMINAL_BENCH_2_1_GATED_TASKS: dict[str, dict[str, Any]] = {
    "circuit-fibsqrt": {
        "scoring_class": RESOURCE_GATED,
        "reason": "The submitted gate network has a hard line-count ceiling.",
        "evidence": (
            _gate(
                "tests/test_outputs.py",
                23,
                "assert line_count < 32000",
                "artifact_size",
            ),
        ),
    },
    "gpt2-codegolf": {
        "scoring_class": RESOURCE_GATED,
        "reason": "The submitted C source has a hard byte-size ceiling.",
        "evidence": (
            _gate(
                "tests/test_outputs.py",
                15,
                "assert gpt2_path.stat().st_size < 5000",
                "artifact_size",
            ),
        ),
    },
    "large-scale-text-editing": {
        "scoring_class": RESOURCE_GATED,
        "reason": "The submitted Vim macros have a hard keystroke-count ceiling.",
        "evidence": (
            _gate(
                "tests/test_outputs.py",
                215,
                "assert total < 200",
                "operation_count",
            ),
        ),
    },
    "largest-eigenval": {
        "scoring_class": PERFORMANCE_GATED,
        "reason": "Candidate and reference wall-clock medians are compared on the host.",
        "evidence": (
            _gate(
                "tests/test_outputs.py",
                111,
                "assert dt < ref_dt",
                "elapsed_time",
            ),
        ),
    },
    "llm-inference-batching-scheduler": {
        "scoring_class": RESOURCE_GATED,
        "reason": (
            "An analytical model imposes compute-cost, padding, latency, "
            "sequential-time, and shape-count ceilings."
        ),
        "evidence": (
            _gate(
                "tests/test_outputs.py",
                322,
                'assert m_b1["cost"] <= 3.0e11',
                "modeled_compute_cost",
            ),
            _gate(
                "tests/test_outputs.py",
                324,
                'assert m_b1["pad_ratio"] <= 0.055',
                "modeled_padding",
            ),
            _gate(
                "tests/test_outputs.py",
                326,
                'assert m_b1["p95_latency_ms"] <= 2.1e6',
                "modeled_latency",
            ),
            _gate(
                "tests/test_outputs.py",
                328,
                'assert m_b1["sequential_timecost"] <= 2.7e8',
                "modeled_time_cost",
            ),
            _gate(
                "tests/test_outputs.py",
                331,
                'assert m_b2["cost"] <= 4.8e10',
                "modeled_compute_cost",
            ),
            _gate(
                "tests/test_outputs.py",
                333,
                'assert m_b2["pad_ratio"] <= 0.15',
                "modeled_padding",
            ),
            _gate(
                "tests/test_outputs.py",
                335,
                'assert m_b2["p95_latency_ms"] <= 2.1e5',
                "modeled_latency",
            ),
            _gate(
                "tests/test_outputs.py",
                337,
                'assert m_b2["sequential_timecost"] <= 3.2e7',
                "modeled_time_cost",
            ),
            _gate(
                "tests/test_outputs.py",
                366,
                "assert len(unique_shapes_combined) <= 8",
                "shape_count",
            ),
        ),
    },
    "path-tracing": {
        "scoring_class": RESOURCE_GATED,
        "reason": "The compressed submitted source has a hard size ceiling.",
        "evidence": (
            _gate(
                "tests/test_outputs.py",
                37,
                'assert len(zlib.compress(open("/app/image.c").read().encode())) < 2100',
                "artifact_size",
            ),
        ),
    },
    "path-tracing-reverse": {
        "scoring_class": RESOURCE_GATED,
        "reason": "The compressed submitted source has a hard size ceiling.",
        "evidence": (
            _gate(
                "tests/test_outputs.py",
                39,
                'assert len(zlib.compress(open("/app/mystery.c").read().encode())) < 2100',
                "artifact_size",
            ),
        ),
    },
    "portfolio-optimization": {
        "scoring_class": PERFORMANCE_GATED,
        "reason": "Candidate wall-clock runtime must beat a Python baseline by 1.2x.",
        "evidence": (
            _gate(
                "tests/test_outputs.py",
                129,
                "assert speedup >= required_speedup  # required_speedup = 1.2",
                "elapsed_time",
            ),
        ),
    },
    "query-optimize": {
        "scoring_class": PERFORMANCE_GATED,
        "reason": (
            "The query's host-clock median is capped relative to the golden "
            "query, and its text also has a character ceiling."
        ),
        "evidence": (
            _gate(
                "tests/test_outputs.py",
                142,
                'assert summary["solution"]["median_s"] <= 1.05 * summary["golden"]["median_s"]',
                "elapsed_time",
            ),
            _gate(
                "tests/test_outputs.py",
                244,
                "assert char_count <= MAX_CHARS  # MAX_CHARS = 2000",
                "artifact_size",
            ),
        ),
    },
    "regex-chess": {
        "scoring_class": RESOURCE_GATED,
        "reason": "The JSON program has both byte-size and pair-count ceilings.",
        "evidence": (
            _gate(
                "tests/test_outputs.py",
                148,
                'assert len(open("/app/re.json").read()) < 10e6',
                "artifact_size",
            ),
            _gate(
                "tests/test_outputs.py",
                149,
                'assert len(json.loads(open("/app/re.json").read())) < 1e5',
                "artifact_count",
            ),
        ),
    },
    "reshard-c4-data": {
        "scoring_class": RESOURCE_GATED,
        "reason": "Output files and directory entries have hard storage-layout ceilings.",
        "evidence": (
            _gate(
                "tests/test_outputs.py",
                128,
                "assert file_size <= MAX_SIZE_PER_FILE  # 15 MiB",
                "artifact_size",
            ),
            _gate(
                "tests/test_outputs.py",
                135,
                "assert total_items <= MAX_FILES_PER_FOLDER  # 30",
                "artifact_count",
            ),
        ),
    },
    "train-fasttext": {
        "scoring_class": RESOURCE_GATED,
        "reason": "The trained model artifact has a hard byte-size ceiling.",
        "evidence": (
            _gate(
                "tests/test_outputs.py",
                41,
                "assert model_size < MAX_MODEL_SIZE  # 150 MiB",
                "artifact_size",
            ),
        ),
    },
    "tune-mjcf": {
        "scoring_class": PERFORMANCE_GATED,
        "reason": "Candidate and reference simulation wall-clock runtimes are compared.",
        "evidence": (
            _gate(
                "tests/test_outputs.py",
                111,
                "assert act_time_pctg <= pctg  # pctg = 0.6",
                "elapsed_time",
            ),
        ),
    },
    "write-compressor": {
        "scoring_class": RESOURCE_GATED,
        "reason": "The compressed data artifact has a hard byte-size ceiling.",
        "evidence": (
            _gate(
                "tests/test_outputs.py",
                70,
                "assert file_size <= max_size_bytes  # 2500 bytes",
                "artifact_size",
            ),
        ),
    },
}


def terminal_bench_2_1_scoring_classification(task_id: str) -> dict[str, Any]:
    """Return a fresh JSON-safe classification for one audited task."""
    gated = TERMINAL_BENCH_2_1_GATED_TASKS.get(task_id)
    if gated is None:
        return {
            "scoring_class": FUNCTIONAL,
            "reason": "No non-functional scoring gate found in the pinned verifier audit.",
            "evidence": [],
        }
    return {
        "scoring_class": gated["scoring_class"],
        "reason": gated["reason"],
        "evidence": [dict(item) for item in gated["evidence"]],
    }
