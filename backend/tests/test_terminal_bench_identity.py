"""Identity and integrity tests for Terminal-Bench adapters."""

from __future__ import annotations

import io
import tarfile
from dataclasses import replace
from unittest.mock import AsyncMock

import pytest

from app.benchmarks.adapters.terminal_bench import (
    AGENT_NOT_TERMINATED_EXCLUSION_REASON,
    VERIFIER_NETWORK_EXCLUSION_REASON,
    VERIFIER_NOT_EXECUTED_EXCLUSION_REASON,
    BenchmarkIdentityError,
    TerminalBench1Adapter,
    TerminalBench2Adapter,
    TerminalBench20Adapter,
    TerminalBench21Adapter,
    TerminalBenchAdapter,
    TerminalBenchHardAdapter,
    classify_agent_exit,
    classify_verifier_network_failure,
)
from app.benchmarks.adapters.terminal_bench_identity import (
    TERMINAL_BENCH_1,
    TERMINAL_BENCH_2_0,
    TERMINAL_BENCH_2_1,
    TERMINAL_BENCH_HARD,
)


def _source_archive(files: dict[str, bytes]) -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as archive:
        for name, content in files.items():
            info = tarfile.TarInfo(f"release-commit/{name}")
            info.size = len(content)
            info.mode = 0o755 if name.endswith(".sh") else 0o644
            archive.addfile(info, io.BytesIO(content))
    return output.getvalue()


def test_pinned_manifests_have_expected_counts_and_unique_ids() -> None:
    assert TERMINAL_BENCH_1.expected_count == 80
    assert TERMINAL_BENCH_2_0.expected_count == 89
    assert TERMINAL_BENCH_2_1.expected_count == 89
    assert TERMINAL_BENCH_HARD.expected_count == 47
    assert len(set(TERMINAL_BENCH_1.task_ids)) == 80
    assert len(set(TERMINAL_BENCH_2_0.task_ids)) == 89
    assert TERMINAL_BENCH_2_0.task_ids == TERMINAL_BENCH_2_1.task_ids
    assert len(set(TERMINAL_BENCH_HARD.task_ids)) == 47
    assert "super-benchmark-upet" not in TERMINAL_BENCH_HARD.task_ids
    with pytest.raises(ValueError, match="named release requires 89"):
        replace(TERMINAL_BENCH_2_1, task_ids=TERMINAL_BENCH_2_1.task_ids[:-1])


def test_adapter_names_resolve_to_explicit_releases() -> None:
    assert TerminalBenchAdapter.benchmark_spec is TERMINAL_BENCH_2_1
    assert TerminalBench1Adapter.benchmark_spec is TERMINAL_BENCH_1
    assert TerminalBench2Adapter.benchmark_spec is TERMINAL_BENCH_2_1
    assert TerminalBench20Adapter.benchmark_spec is TERMINAL_BENCH_2_0
    assert TerminalBench21Adapter.benchmark_spec is TERMINAL_BENCH_2_1
    assert TerminalBenchHardAdapter.benchmark_spec is TERMINAL_BENCH_HARD


def test_item_timeout_covers_long_agent_budget(monkeypatch) -> None:
    adapter = TerminalBench21Adapter.__new__(TerminalBench21Adapter)
    adapter._items = [
        {
            "id": "long",
            "max_agent_timeout_sec": 3600,
            "max_test_timeout_sec": 600,
        },
        {
            "id": "short",
            "max_agent_timeout_sec": 300,
            "max_test_timeout_sec": 60,
        },
    ]
    monkeypatch.setenv("TERMINAL_BENCH_AGENT_TIMEOUT_MULTIPLIER", "2.0")

    long_timeout = adapter.get_item_timeout_seconds("long")

    assert long_timeout == 7200 + 600 + 900
    assert long_timeout > 1200
    assert long_timeout >= 7200
    assert adapter.get_item_timeout_seconds("short") == 600 + 60 + 900
    assert adapter.get_item_timeout_seconds() == long_timeout


def test_hard_manifest_is_the_reproduced_leaderboard_subset() -> None:
    assert TERMINAL_BENCH_HARD.commit == "74221fb0b6b5a7f88e53bed5726edaaf236348c9"
    assert TERMINAL_BENCH_HARD.manifest_repository == "NVIDIA-NeMo/Evaluator"
    assert TERMINAL_BENCH_HARD.manifest_commit == "bd952253260e7077973aadf5fc656e425d2758e1"
    assert TERMINAL_BENCH_HARD.task_ids[0] == "aimo-airline-departures"
    assert TERMINAL_BENCH_HARD.task_ids[-1] == "write-compressor"


def test_harbor_source_archive_is_packaged_in_manifest_order() -> None:
    spec = replace(
        TERMINAL_BENCH_2_1,
        expected_count=2,
        task_ids=("task-b", "task-a"),
    )
    manifest = b"""
[metadata]
difficulty = "hard"
[agent]
timeout_sec = 123
[verifier]
timeout_sec = 45
[environment]
docker_image = "example/task:pin"
cpus = 2
memory_mb = 4096
"""
    source = _source_archive(
        {
            "tasks/task-a/task.toml": manifest,
            "tasks/task-a/instruction.md": b"Do A",
            "tasks/task-a/tests/test.sh": b"#!/bin/bash\necho 1",
            "tasks/task-b/task.toml": manifest,
            "tasks/task-b/instruction.md": b"Do B",
            "tasks/task-b/solution/solve.sh": b"#!/bin/bash\ntrue",
        }
    )
    adapter = TerminalBench21Adapter.__new__(TerminalBench21Adapter)
    adapter.benchmark_spec = spec
    adapter.benchmark_name = "test_terminal_bench"

    items = adapter._items_from_source_archive(source)

    assert [item["task_id"] for item in items] == ["task-b", "task-a"]
    assert items[0]["id"] == "0"
    assert items[0]["instruction"] == "Do B"
    assert items[0]["task_format"] == "harbor"
    assert items[0]["docker_image"] == "example/task:pin"
    assert items[0]["max_agent_timeout_sec"] == 123
    assert items[0]["max_test_timeout_sec"] == 45
    with tarfile.open(fileobj=io.BytesIO(items[0]["archive"]), mode="r:") as task:
        assert "task.toml" in task.getnames()
        assert "solution/solve.sh" in task.getnames()
        assert all(not name.startswith("release-commit/") for name in task.getnames())


def test_source_archive_missing_canonical_task_fails() -> None:
    spec = replace(
        TERMINAL_BENCH_2_1,
        expected_count=2,
        task_ids=("present", "missing"),
    )
    source = _source_archive(
        {
            "tasks/present/task.toml": b"[environment]\ndocker_image='x/y:z'\n",
            "tasks/present/instruction.md": b"Present",
        }
    )
    adapter = TerminalBench21Adapter.__new__(TerminalBench21Adapter)
    adapter.benchmark_spec = spec
    adapter.benchmark_name = "test_terminal_bench"

    with pytest.raises(BenchmarkIdentityError, match="missing 1 canonical tasks: missing"):
        adapter._items_from_source_archive(source)


def test_loaded_count_self_check_fails_loudly() -> None:
    adapter = TerminalBench21Adapter.__new__(TerminalBench21Adapter)
    adapter.benchmark_spec = replace(
        TERMINAL_BENCH_2_1,
        expected_count=2,
        task_ids=("a", "b"),
    )
    adapter.benchmark_name = "test_terminal_bench"
    adapter._items = [{"task_id": "a"}]

    with pytest.raises(BenchmarkIdentityError, match="expected 2 items, loaded 1"):
        adapter._assert_benchmark_identity()


@pytest.mark.asyncio
async def test_versioned_item_ids_use_concrete_adapter_config() -> None:
    adapter = TerminalBench21Adapter.__new__(TerminalBench21Adapter)
    adapter._items = [{"id": "0"}, {"id": "1"}, {"id": "2"}]
    adapter.run_config = {
        "terminal_bench_2": {"item_ids": ["0"]},
        "terminal_bench_2_1": {"item_ids": ["2", "1"]},
    }

    total, selected = await adapter.get_items_for_evaluation(100, "seed")

    assert total == 3
    assert selected == ["2", "1"]


@pytest.mark.asyncio
async def test_hard_item_ids_still_use_terminal_bench_family_config() -> None:
    adapter = TerminalBenchHardAdapter.__new__(TerminalBenchHardAdapter)
    adapter._items = [{"id": "0"}, {"id": "1"}, {"id": "2"}]
    adapter.run_config = {"terminal_bench": {"item_ids": ["1"]}}

    total, selected = await adapter.get_items_for_evaluation(100, "seed")

    assert total == 3
    assert selected == ["1"]


def test_harbor_reward_parser_uses_verifier_output() -> None:
    assert TerminalBench21Adapter._parse_harbor_reward("1\n") == 1.0
    assert TerminalBench21Adapter._parse_harbor_reward('{"reward": 0.5}') == 0.5
    assert TerminalBench21Adapter._parse_harbor_reward('{"only_metric": 0}') == 0.0
    assert TerminalBench21Adapter._parse_harbor_reward("not a reward") is None


@pytest.mark.asyncio
async def test_answer_key_holdout_and_container_probe_are_preserved() -> None:
    adapter = TerminalBench21Adapter.__new__(TerminalBench21Adapter)
    adapter.sandy = type("Sandy", (), {})()
    adapter.sandy.execute_command = AsyncMock(
        side_effect=[
            {"exit_code": 0},
            {
                "exit_code": 0,
                "stdout": "LEFT=environment,instruction.md,task.toml ARCHIVES=0 "
                "HELD=solution,tests,",
            },
            {"exit_code": 0, "stdout": ""},
        ]
    )

    holdout = await adapter._withhold_answer_key("sandbox", "s123")
    clean = await adapter._verify_container_clean("sandbox", "container")

    assert holdout["withheld"] is True
    assert clean["clean"] is True
    holdout_command = adapter.sandy.execute_command.await_args_list[0].args[1]
    assert "*solution*" in holdout_command
    assert "*test*" in holdout_command
    assert "rm -f /workspace/archive.tar /workspace/archive.b64" in holdout_command
    container_probe = adapter.sandy.execute_command.await_args_list[2].args[1]
    assert "find /tests /solution -type f" in container_probe


def test_agent_exit_classification_still_excludes_only_dead_sandbox() -> None:
    summary = {"exitCode": 1, "duration": 10}
    exclusion, _ = classify_agent_exit(summary, 100, False)
    assert exclusion == "infrastructure_sandbox_gone"

    exclusion, note = classify_agent_exit(summary, 100, True)
    assert exclusion is None
    assert "Scored" in (note or "")


@pytest.mark.asyncio
async def test_unseal_proves_connectivity_inside_both_environments() -> None:
    adapter = TerminalBench21Adapter.__new__(TerminalBench21Adapter)
    adapter.sandy = type("Sandy", (), {})()
    adapter.sandy.execute_command = AsyncMock(
        side_effect=[
            # Exit 4 was universal in the affected runs, including passes. The
            # property probes, not these values, determine the verdict.
            {"exit_code": 4, "stderr": "benign historical signature"},
            {"exit_code": 4, "stderr": "benign historical signature"},
            {"exit_code": 0, "stdout": "HOSTS=0\nNETWORK=OPEN\n"},
            {"exit_code": 0, "stdout": "HOSTS=0\nNETWORK=OPEN\n"},
        ]
    )

    verdict = await adapter._unseal_network("sandbox", "task-container")

    assert verdict["restored"] is True
    assert verdict["sandbox_connected"] is True
    assert verdict["container_connected"] is True
    restore_command = adapter.sandy.execute_command.await_args_list[0].args[1]
    assert "cat \"$tmp\" > /etc/hosts" in restore_command
    assert "sed -i" not in restore_command
    container_probe = adapter.sandy.execute_command.await_args_list[3].args[1]
    assert "docker exec task-container" in container_probe
    assert "raw.githubusercontent.com" in container_probe


@pytest.mark.asyncio
async def test_unseal_rejects_successful_actions_without_restored_property() -> None:
    adapter = TerminalBench21Adapter.__new__(TerminalBench21Adapter)
    adapter.sandy = type("Sandy", (), {})()
    adapter.sandy.execute_command = AsyncMock(
        side_effect=[
            {"exit_code": 0},
            {"exit_code": 0},
            {"exit_code": 0, "stdout": "HOSTS=0\nNETWORK=OPEN\n"},
            {"exit_code": 0, "stdout": "HOSTS=1\nNETWORK=CLOSED:\n"},
        ]
    )

    verdict = await adapter._unseal_network("sandbox", "task-container")

    assert verdict["restored"] is False
    assert verdict["container_hosts_removed"] is False
    assert verdict["container_connected"] is False


@pytest.mark.asyncio
async def test_agent_must_emit_completion_and_be_stopped_before_unseal() -> None:
    adapter = TerminalBench21Adapter.__new__(TerminalBench21Adapter)
    adapter.sandy = type("Sandy", (), {})()
    adapter.sandy.execute_command = AsyncMock(
        return_value={
            "exit_code": 0,
            "stdout": "PID=42 STATE=gone RUNNING=no DONE=0\n",
        }
    )

    terminated = await adapter._verify_agent_terminated(
        "sandbox", {"type": "complete", "exitCode": 0}
    )
    missing_completion = await adapter._verify_agent_terminated(
        "sandbox", {"exitCode": 0}
    )

    assert terminated["terminated"] is True
    assert missing_completion["terminated"] is False
    assert AGENT_NOT_TERMINATED_EXCLUSION_REASON == (
        "infrastructure_agent_not_terminated"
    )


def test_verifier_network_failure_is_excluded_instead_of_scored_zero() -> None:
    test_result = {
        "exit_code": 127,
        "stderr": (
            "curl: (7) Failed to connect to github.com port 443\n"
            "failed to download uv-x86_64-unknown-linux-gnu.tar.gz\n"
            "/tests/test.sh: line 19: uvx: command not found"
        ),
    }
    reward_result = {"exit_code": 0, "stdout": "0\n"}

    outcome = TerminalBench21Adapter._harbor_verifier_outcome(
        test_result,
        reward_result,
        test_command_executed=True,
    )

    assert outcome["exclusion_reason"] == VERIFIER_NETWORK_EXCLUSION_REASON
    assert outcome["reward"] is None
    assert outcome["is_correct"] is None


def test_harbor_reward_requires_proof_test_command_executed() -> None:
    outcome = TerminalBench21Adapter._harbor_verifier_outcome(
        {"exit_code": -1, "error": "transport response lost"},
        {"exit_code": 0, "stdout": "0\n"},
        test_command_executed=False,
    )

    assert outcome["exclusion_reason"] == VERIFIER_NOT_EXECUTED_EXCLUSION_REASON
    assert outcome["reward"] is None
    assert outcome["is_correct"] is None


def test_executed_harbor_verifier_can_still_return_capability_zero() -> None:
    outcome = TerminalBench21Adapter._harbor_verifier_outcome(
        {"exit_code": 1, "stderr": "2 assertions failed"},
        {"exit_code": 0, "stdout": "0\n"},
        test_command_executed=True,
    )

    assert outcome["exclusion_reason"] is None
    assert outcome["reward"] == 0.0
    assert outcome["is_correct"] is False
    assert classify_verifier_network_failure(
        {"stderr": "failed to download artifact: HTTP 404"}
    ) is None
