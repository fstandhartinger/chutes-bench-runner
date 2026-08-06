"""Identity, isolation, timeout, and exclusion tests for DeepSWE v1.1."""

from __future__ import annotations

import io
import tarfile
from dataclasses import replace

import pytest

from app.benchmarks.adapters.deepswe import (
    DeepSWEAdapter,
    classify_deepswe_agent_outcome,
    classify_deepswe_exception,
)
from app.benchmarks.adapters.deepswe_identity import DEEPSWE_V1_1
from app.benchmarks.adapters.terminal_bench import BenchmarkIdentityError


def _source_archive(files: dict[str, bytes]) -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as archive:
        for name, content in files.items():
            info = tarfile.TarInfo(f"deep-swe-commit/{name}")
            info.size = len(content)
            info.mode = 0o755 if name.endswith(".sh") else 0o644
            archive.addfile(info, io.BytesIO(content))
    return output.getvalue()


def _manifest(task_id: str) -> bytes:
    return f'''\
schema_version = "1.3"
artifacts = ["/logs/artifacts/model.patch"]
[metadata]
task_id = "{task_id}"
language = "python"
base_commit_hash = "abc123"
[verifier]
network_mode = "no-network"
environment_mode = "separate"
timeout_sec = 1800.0
[verifier.environment]
build_timeout_sec = 1800.0
[[verifier.collect]]
command = "cd /app && git diff --binary abc123 HEAD > /logs/artifacts/model.patch"
timeout_sec = 300.0
[agent]
network_mode = "no-network"
timeout_sec = 5400.0
[environment]
build_timeout_sec = 1800.0
docker_image = "public.ecr.aws/example/{task_id}:v1.1"
cpus = 2
memory_mb = 8192
storage_mb = 20480
'''.encode()


def test_pinned_manifest_has_exact_expected_count_and_unique_ids() -> None:
    assert DEEPSWE_V1_1.expected_count == 113
    assert len(DEEPSWE_V1_1.task_ids) == 113
    assert len(set(DEEPSWE_V1_1.task_ids)) == 113
    assert DEEPSWE_V1_1.task_ids[0] == "abs-module-cache-flags"
    assert DEEPSWE_V1_1.task_ids[-1] == "ytt-jsonpath-query-api"
    with pytest.raises(ValueError, match="named release requires 113"):
        replace(DEEPSWE_V1_1, task_ids=DEEPSWE_V1_1.task_ids[:-1])
    with pytest.raises(ValueError, match="duplicate task IDs"):
        replace(
            DEEPSWE_V1_1,
            expected_count=2,
            task_ids=("duplicate", "duplicate"),
        )


def test_loaded_count_assertion_fails_loudly() -> None:
    adapter = DeepSWEAdapter.__new__(DeepSWEAdapter)
    adapter.benchmark_spec = replace(
        DEEPSWE_V1_1,
        expected_count=2,
        task_ids=("task-a", "task-b"),
    )
    adapter._items = [{"task_id": "task-a"}]

    with pytest.raises(BenchmarkIdentityError, match="expected 2 items, loaded 1"):
        adapter._assert_benchmark_identity()


def test_source_packaging_never_puts_answer_key_in_agent_archive() -> None:
    spec = replace(
        DEEPSWE_V1_1,
        expected_count=2,
        task_ids=("task-b", "task-a"),
    )
    files = {}
    for task_id in ("task-a", "task-b"):
        files.update(
            {
                f"tasks/{task_id}/task.toml": _manifest(task_id),
                f"tasks/{task_id}/instruction.md": f"Implement {task_id}".encode(),
                f"tasks/{task_id}/environment/Dockerfile": b"FROM scratch\n",
                f"tasks/{task_id}/tests/Dockerfile": b"FROM example\n",
                f"tasks/{task_id}/tests/test.sh": b"#!/bin/bash\necho 1\n",
                f"tasks/{task_id}/tests/test.patch": b"hidden tests\n",
                f"tasks/{task_id}/solution/solve.sh": b"#!/bin/bash\nsolve\n",
                f"tasks/{task_id}/solution/solution.patch": b"gold patch\n",
            }
        )
    adapter = DeepSWEAdapter.__new__(DeepSWEAdapter)
    adapter.benchmark_spec = spec

    items = adapter._items_from_source_archive(_source_archive(files))
    adapter._items = items
    adapter._assert_benchmark_identity()

    assert [item["task_id"] for item in items] == ["task-b", "task-a"]
    with tarfile.open(fileobj=io.BytesIO(items[0]["agent_archive"]), mode="r:") as archive:
        names = archive.getnames()
        assert "task.toml" in names
        assert "instruction.md" in names
        assert all(not name.startswith(("tests", "solution")) for name in names)
    with tarfile.open(fileobj=io.BytesIO(items[0]["verifier_archive"]), mode="r:") as archive:
        names = archive.getnames()
        assert "Dockerfile" in names
        assert "test.patch" in names
        assert all(not name.startswith("solution") for name in names)
    assert all(entry["size"] > 0 for entry in items[0]["heldout_hashes"])


@pytest.mark.parametrize("agent", ["chutescoder", "chutescoder-baseline", "codex"])
def test_sandy_cli_agent_arms_are_selectable(agent: str) -> None:
    adapter = DeepSWEAdapter.__new__(DeepSWEAdapter)
    adapter.run_config = {"deepswe": {"agent": agent}}

    assert adapter._agent_name() == agent


def test_item_timeout_covers_every_declared_phase(monkeypatch) -> None:
    adapter = DeepSWEAdapter.__new__(DeepSWEAdapter)
    adapter._items = [
        {
            "id": "0",
            "max_agent_timeout_sec": 5400,
            "max_test_timeout_sec": 1800,
            "environment_build_timeout_sec": 1800,
            "verifier_build_timeout_sec": 1800,
            "collect_timeout_sec": 300,
        }
    ]
    monkeypatch.delenv("DEEPSWE_AGENT_TIMEOUT_MULTIPLIER", raising=False)

    timeout = adapter.get_item_timeout_seconds("0")

    assert timeout == 5400 + 1800 + 1800 + 1800 + 300 + 900
    assert timeout >= 5400


def test_infrastructure_exclusions_do_not_hide_live_cli_crashes() -> None:
    exclusion, note = classify_deepswe_agent_outcome({}, 5400, True)
    assert exclusion == "infrastructure_transport"
    assert "without a Sandy completion summary" in (note or "")

    early_crash = {"exitCode": 1, "duration": 30}
    exclusion, _ = classify_deepswe_agent_outcome(early_crash, 5400, False)
    assert exclusion == "infrastructure_sandbox_gone"

    exclusion, note = classify_deepswe_agent_outcome(early_crash, 5400, True)
    assert exclusion is None
    assert "Scored" in (note or "")

    budget_exit = {"exitCode": 124, "duration": 5390}
    exclusion, note = classify_deepswe_agent_outcome(budget_exit, 5400, True)
    assert exclusion is None
    assert note == "agent_exhausted_budget"


def test_transport_exception_without_summary_is_excluded() -> None:
    assert classify_deepswe_exception("peer closed connection", None) == "infrastructure_transport"
    assert classify_deepswe_exception("agent assertion failed", {"exitCode": 1}) is None
