"""Identity, holdout, verifier, and evidence tests for SWE-bench Verified."""

from __future__ import annotations

import hashlib
import inspect
from dataclasses import replace
from types import SimpleNamespace

import pytest

import app.benchmarks.adapters.swe_bench_verified as verified_module
from app.benchmarks.adapters.swe_bench_verified import (
    SWE_BENCH_VERIFIED_EVIDENCE_EXCLUSION_REASON,
    SWE_BENCH_VERIFIED_VERIFIER_NOT_EXECUTED,
    SWEBenchVerifiedAdapter,
    classify_swe_bench_verified_agent_outcome,
    classify_swe_bench_verified_verifier_outcome,
)
from app.benchmarks.adapters.swe_bench_verified_identity import SWE_BENCH_VERIFIED
from app.benchmarks.adapters.terminal_bench import BenchmarkIdentityError


def test_identity_pins_official_500_row_dataset_and_harness_release() -> None:
    assert SWE_BENCH_VERIFIED.expected_count == 500
    assert SWE_BENCH_VERIFIED.dataset_repository == "princeton-nlp/SWE-bench_Verified"
    assert SWE_BENCH_VERIFIED.dataset_commit == (
        "c104f840cc67f8b6eec6f759ebc8b2693d585d4a"
    )
    assert SWE_BENCH_VERIFIED.dataset_file_sha256 == (
        "a45b1fe4e2f0c8390b2b2938ac83e92ed5979000856808f3679c07812e9e6dcd"
    )
    assert SWE_BENCH_VERIFIED.harness_version == "4.1.0"
    assert SWE_BENCH_VERIFIED.harness_commit == (
        "726c5461e2ef52d83cf1ea2107870a8bb3328d57"
    )


def test_loaded_count_and_duplicate_assertions_fail_loudly() -> None:
    adapter = SWEBenchVerifiedAdapter.__new__(SWEBenchVerifiedAdapter)
    adapter.benchmark_spec = replace(SWE_BENCH_VERIFIED, expected_count=2)
    adapter._items = [{"instance_id": "one", "docker_image": "swebench/sweb.eval.x86_64.one"}]

    with pytest.raises(BenchmarkIdentityError, match="expected 2 items, loaded 1"):
        adapter._assert_benchmark_identity()

    adapter._items = [
        {"instance_id": "same", "docker_image": "swebench/sweb.eval.x86_64.same"},
        {"instance_id": "same", "docker_image": "swebench/sweb.eval.x86_64.same"},
    ]
    with pytest.raises(BenchmarkIdentityError, match="duplicate instance IDs"):
        adapter._assert_benchmark_identity()


def test_dataset_shard_hash_is_checked_before_parsing(tmp_path, monkeypatch) -> None:
    shard = tmp_path / "test.parquet"
    shard.write_bytes(b"not the pinned shard")
    adapter = SWEBenchVerifiedAdapter.__new__(SWEBenchVerifiedAdapter)
    adapter.benchmark_spec = SWE_BENCH_VERIFIED
    monkeypatch.setattr(adapter, "_dataset_cache_path", lambda: shard)

    with pytest.raises(BenchmarkIdentityError, match="dataset SHA-256 mismatch"):
        adapter._load_pinned_rows()


def test_normalization_keeps_answers_worker_only_and_records_hashes(monkeypatch) -> None:
    patch = "diff --git a/a.py b/a.py\n+fixed\n"
    test_patch = "diff --git a/test_a.py b/test_a.py\n+hidden\n"
    raw = {
        "instance_id": "owner__repo-1",
        "repo": "owner/repo",
        "base_commit": "a" * 40,
        "problem_statement": "Fix the bug",
        "version": "1.0",
        "patch": patch,
        "test_patch": test_patch,
        "FAIL_TO_PASS": '["test_regression"]',
        "PASS_TO_PASS": '["test_existing"]',
    }
    test_spec = SimpleNamespace(
        eval_script="#!/bin/bash\nhidden verifier script\n",
        instance_image_key="swebench/sweb.eval.x86_64.owner_1776_repo-1:latest",
    )
    adapter = SWEBenchVerifiedAdapter.__new__(SWEBenchVerifiedAdapter)
    adapter.benchmark_spec = SWE_BENCH_VERIFIED
    monkeypatch.setattr(adapter, "_make_official_test_spec", lambda _row: test_spec)

    item = adapter._normalize_row(7, raw)

    assert item["id"] == "7"
    assert item["problem_statement"] == "Fix the bug"
    assert item["_dataset_row"]["patch"] == patch
    assert item["_eval_script"] == test_spec.eval_script
    assert item["heldout_hashes"] == [
        {
            "name": "patch",
            "sha256": hashlib.sha256(patch.encode()).hexdigest(),
            "size": len(patch.encode()),
        },
        {
            "name": "test_patch",
            "sha256": hashlib.sha256(test_patch.encode()).hexdigest(),
            "size": len(test_patch.encode()),
        },
    ]


@pytest.mark.parametrize("agent", ["prime-agent", "chutescoder", "chutescoder-baseline", "codex"])
def test_sandy_cli_agent_arms_are_selectable(agent: str) -> None:
    adapter = SWEBenchVerifiedAdapter.__new__(SWEBenchVerifiedAdapter)
    adapter.run_config = {"swe_bench_verified": {"agent": agent}}

    assert adapter._agent_name() == agent


def test_item_timeout_covers_every_declared_phase(monkeypatch) -> None:
    monkeypatch.setenv("SWE_BENCH_VERIFIED_AGENT_TIMEOUT_SEC", "3600")
    monkeypatch.setenv("SWE_BENCH_VERIFIED_IMAGE_PULL_TIMEOUT_SEC", "1800")
    monkeypatch.setenv("SWE_BENCH_VERIFIED_VERIFIER_TIMEOUT_SEC", "1800")
    monkeypatch.setenv("SWE_BENCH_VERIFIED_COLLECT_TIMEOUT_SEC", "300")
    adapter = SWEBenchVerifiedAdapter.__new__(SWEBenchVerifiedAdapter)

    assert adapter.get_item_timeout_seconds("0") == 3600 + 1800 + 1800 + 300 + 900


def test_agent_never_receives_answers_socket_cache_or_eval_script() -> None:
    evaluate_source = inspect.getsource(SWEBenchVerifiedAdapter._evaluate_item)
    verifier_source = inspect.getsource(SWEBenchVerifiedAdapter._stage_and_run_verifier)

    assert "enable_docker_socket=False" in evaluate_source
    assert "enable_shared_cache=False" in evaluate_source
    assert "_start_task_gateway" in evaluate_source
    assert evaluate_source.index("_verify_workspace_clean") < evaluate_source.index(
        "prepare_sandy_agent_launch"
    )
    assert evaluate_source.index("_verify_agent_docker_boundary") < evaluate_source.index(
        "prepare_sandy_agent_launch"
    )
    assert evaluate_source.index("await asyncio.wait_for(asyncio.to_thread(_stop_agent_container)") < (
        evaluate_source.index("await self._stage_and_run_verifier")
    )
    assert "self.sandy.write_file" not in verifier_source
    assert "_put_file_outside" in verifier_source
    assert '"/eval.sh"' in verifier_source


@pytest.mark.asyncio
async def test_workspace_hash_scan_rejects_heldout_bytes() -> None:
    digest = "a" * 64

    class SandyStub:
        async def execute_command(self, *_args, **_kwargs):
            return {
                "exit_code": 0,
                "stdout": f"ANSWERS=0 ARCHIVES=0 CACHE_FILES=0\n{digest}  /tmp/answer\n",
            }

    adapter = SWEBenchVerifiedAdapter.__new__(SWEBenchVerifiedAdapter)
    adapter.sandy = SandyStub()
    proof = await adapter._verify_workspace_clean(
        "sandbox", [{"sha256": digest, "size": 123}]
    )

    assert proof["clean"] is False
    assert proof["heldout_hash_matches"] == [f"{digest}  /tmp/answer"]


def test_infrastructure_exclusions_do_not_hide_live_agent_crashes() -> None:
    exclusion, note = classify_swe_bench_verified_agent_outcome({}, 3600, True)
    assert exclusion == "infrastructure_transport"
    assert "without a Sandy completion summary" in (note or "")

    early_crash = {"exitCode": 1, "duration": 30}
    exclusion, _ = classify_swe_bench_verified_agent_outcome(early_crash, 3600, False)
    assert exclusion == "infrastructure_sandbox_gone"

    exclusion, note = classify_swe_bench_verified_agent_outcome(early_crash, 3600, True)
    assert exclusion is None
    assert "Scored" in (note or "")


def test_verifier_classifies_model_failures_and_infrastructure_separately() -> None:
    assert classify_swe_bench_verified_verifier_outcome(
        {"patch_applied": False, "patch_error": "bad diff"}
    ) == (0.0, None, "bad diff")

    reward, exclusion, _ = classify_swe_bench_verified_verifier_outcome(
        {"patch_applied": True, "test_command_executed": False}
    )
    assert reward is None
    assert exclusion == SWE_BENCH_VERIFIED_VERIFIER_NOT_EXECUTED

    assert classify_swe_bench_verified_verifier_outcome(
        {
            "patch_applied": True,
            "test_command_executed": True,
            "timed_out": False,
            "report": {"resolved": True},
        }
    ) == (1.0, None, None)
    assert classify_swe_bench_verified_verifier_outcome(
        {
            "patch_applied": True,
            "test_command_executed": True,
            "timed_out": False,
            "report": {"resolved": False},
        }
    ) == (0.0, None, None)


@pytest.mark.asyncio
async def test_network_seal_requires_failed_source_fetch(monkeypatch) -> None:
    class SandyStub:
        def __init__(self, probe: str):
            self.probe = probe
            self.calls = 0

        async def execute_command(self, *_args, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                return {"exit_code": 0, "stdout": ""}
            return {"exit_code": 0, "stdout": self.probe}

    async def outside_exec(*_args, **_kwargs):
        return {"exit_code": 0, "stdout": ""}

    class Container:
        attrs = {
            "Config": {"Labels": {"chutes.bench.sandbox_id": "sandbox"}},
            "HostConfig": {"NetworkMode": "none"},
        }

    class Containers:
        @staticmethod
        def get(_name):
            return Container()

    client = SimpleNamespace(containers=Containers())
    monkeypatch.setattr(verified_module.docker, "from_env", lambda: client)
    adapter = SWEBenchVerifiedAdapter.__new__(SWEBenchVerifiedAdapter)
    adapter._docker_exec_outside = outside_exec

    adapter.sandy = SandyStub("HOSTS=1\nCURL=yes\nFETCH=307")
    reachable = await adapter._seal_network("sandbox", "container")
    assert reachable["sealed"] is False

    adapter.sandy = SandyStub("HOSTS=1\nCURL=yes\nFETCH=000CURLFAIL")
    blocked = await adapter._seal_network("sandbox", "container")
    assert blocked["sealed"] is True


def test_missing_evidence_excludes_an_otherwise_scored_item() -> None:
    adapter = SWEBenchVerifiedAdapter.__new__(SWEBenchVerifiedAdapter)
    adapter._item_observability = {
        "task": {
            "agent_invoked": True,
            "evidence": {
                "status": "failed",
                "path": None,
                "sha256": None,
                "size_bytes": None,
                "error": "no rollout JSONL",
                "token_usage_samples": None,
            },
        }
    }
    result = adapter.attach_item_observability(
        verified_module.ItemResult(item_id="task", score=1.0, is_correct=True)
    )

    assert result.metadata["exclusion_reason"] == (
        SWE_BENCH_VERIFIED_EVIDENCE_EXCLUSION_REASON
    )
    assert result.agent_evidence_status == "failed"
    assert result.error == "Agent evidence was not retained: no rollout JSONL"


def test_finalizer_retains_evidence_before_destroying_agent_environment() -> None:
    source = inspect.getsource(SWEBenchVerifiedAdapter._evaluate_item)

    assert source.index("_finish_evidence_retention") < source.index(
        "_cleanup_owned_task_containers"
    )
    assert source.index("_finish_evidence_retention") < source.index("terminate_sandbox")

