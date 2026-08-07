"""Identity, isolation, timeout, and exclusion tests for DeepSWE v1.1."""

from __future__ import annotations

import inspect
import io
import json
import os
import subprocess
import tarfile
import tomllib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import app.benchmarks.adapters.deepswe as deepswe_module
from app.benchmarks.adapters.deepswe import (
    DEEPSWE_AGENT_NOT_TERMINATED_EXCLUSION_REASON,
    DEEPSWE_VERIFIER_NOT_EXECUTED_EXCLUSION_REASON,
    DeepSWEAdapter,
    classify_deepswe_agent_outcome,
    classify_deepswe_exception,
    classify_deepswe_verifier_outcome,
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


@pytest.mark.parametrize("agent", ["prime-agent", "chutescoder", "chutescoder-baseline", "codex"])
def test_sandy_cli_agent_arms_are_selectable(agent: str) -> None:
    adapter = DeepSWEAdapter.__new__(DeepSWEAdapter)
    adapter.run_config = {"deepswe": {"agent": agent}}

    assert adapter._agent_name() == agent


@pytest.mark.asyncio
@pytest.mark.parametrize("agent", ["chutescoder", "chutescoder-baseline", "codex"])
async def test_context_limit_reaches_effective_config_for_every_arm(agent: str) -> None:
    class Client:
        provider = "openrouter"

        @staticmethod
        def get_api_key() -> str:
            return "test-key"

        @staticmethod
        def get_api_base_url() -> str:
            return "https://openrouter.ai/api/v1"

        @staticmethod
        async def get_model_context_length(_model: str) -> int:
            return 1_048_576

        @staticmethod
        async def get_model_max_output_length(_model: str) -> int:
            return 65_536

    adapter = DeepSWEAdapter.__new__(DeepSWEAdapter)
    adapter.client = Client()
    adapter.sandy = AsyncMock()
    adapter.sandy.execute_command.return_value = {"exit_code": 0}
    adapter.model_slug = "deepseek/deepseek-v4-flash-0731"
    adapter.run_config = {"deepswe": {"agent": agent, "context_limit_tokens": 48_000}}

    launch = await adapter._prepare_agent_launch(
        "sandbox",
        adapter._agent_name(),
        adapter._context_limit_tokens(),
    )

    assert "self._prepare_agent_launch(" in inspect.getsource(DeepSWEAdapter._evaluate_item)
    assert launch.setup is not None
    config = tomllib.loads(launch.setup.config_toml)
    catalog = json.loads(launch.setup.model_catalog_json)
    assert config["model_context_window"] == 48_000
    assert catalog["models"][0]["context_window"] == 48_000
    assert catalog["models"][0]["max_context_window"] == 48_000
    assert launch.metadata["context_window"] == 48_000


@pytest.mark.parametrize("value", [True, 0, -1, 40_000.0, "40000"])
def test_context_limit_rejects_values_that_cannot_cap_the_agent(value) -> None:
    adapter = DeepSWEAdapter.__new__(DeepSWEAdapter)
    adapter.run_config = {"deepswe": {"context_limit_tokens": value}}

    with pytest.raises(ValueError, match="must be a positive integer"):
        adapter._context_limit_tokens()


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


def test_agent_never_receives_raw_docker_socket_or_verifier_archive() -> None:
    evaluate_source = inspect.getsource(DeepSWEAdapter._evaluate_item)
    verifier_source = inspect.getsource(DeepSWEAdapter._stage_and_run_verifier)

    assert "_start_task_gateway" in evaluate_source
    assert "enable_docker_socket=False" in evaluate_source
    assert "enable_shared_cache=False" in evaluate_source
    assert "_verify_agent_docker_boundary" in evaluate_source
    assert evaluate_source.index("_verify_agent_docker_boundary") < evaluate_source.index(
        "_prepare_agent_launch"
    )
    assert evaluate_source.index("_verify_workspace_clean") < evaluate_source.index(
        "_start_agent_container"
    )
    assert evaluate_source.index("_verify_shared_task_workspace") < evaluate_source.index(
        "_prepare_agent_launch"
    )
    assert "_upload_archive" not in verifier_source
    assert "self.sandy.write_file" not in verifier_source
    assert "docker.from_env" in verifier_source


def test_workspace_mount_is_resolved_from_the_sandbox_boundary(monkeypatch) -> None:
    sandbox = SimpleNamespace(
        attrs={
            "Mounts": [
                {
                    "Type": "bind",
                    "Source": "/var/lib/sandy/workspaces/sandbox",
                    "Destination": "/workspace",
                    "RW": True,
                }
            ]
        }
    )
    monkeypatch.setattr(deepswe_module, "sandbox_container", lambda _sandbox_id: sandbox)

    resolved = DeepSWEAdapter._agent_workspace_mount("sandbox")

    assert resolved == {
        "source": "/var/lib/sandy/workspaces/sandbox",
        "destination": "/workspace",
        "type": "bind",
        "read_write": True,
        "observed_from": "worker_docker_api",
    }


@pytest.mark.asyncio
async def test_task_runtime_bind_mounts_the_agent_checkout(monkeypatch) -> None:
    run_calls = []

    class TaskContainer:
        id = "task-container-id"
        attrs = {"HostConfig": {"NetworkMode": "none"}}

        @staticmethod
        def reload() -> None:
            return None

    class Containers:
        @staticmethod
        def get(_name):
            raise deepswe_module.docker.errors.NotFound("missing")

        @staticmethod
        def run(*args, **kwargs):
            run_calls.append((args, kwargs))
            return b"" if kwargs.get("remove") else TaskContainer()

    class Images:
        @staticmethod
        def pull(_image):
            return SimpleNamespace(
                id="sha256:image",
                attrs={"RepoDigests": ["example@sha256:digest"]},
            )

    class Client:
        images = Images()
        containers = Containers()

    async def outside_exec(*_args, **_kwargs):
        return {"exit_code": 0, "stdout": ""}

    monkeypatch.setattr(deepswe_module.docker, "from_env", lambda: Client())
    monkeypatch.setattr(
        deepswe_module.TerminalBenchAdapter,
        "_docker_exec_outside",
        outside_exec,
    )
    adapter = DeepSWEAdapter.__new__(DeepSWEAdapter)
    adapter.sandy = SimpleNamespace(
        execute_command=AsyncMock(return_value={"exit_code": 0, "stdout": ""})
    )

    result = await adapter._start_agent_container(
        "sandbox",
        {"docker_image": "example:task", "cpus": 2, "memory_mb": 1024},
        "task-container",
        30,
        {"source": "/host/sandbox", "destination": "/workspace"},
    )

    assert result["ok"] is True
    assert len(run_calls) == 2
    assert run_calls[0][1]["volumes"] == {
        "/host/sandbox/repo": {"bind": "/workspace/repo", "mode": "rw"}
    }
    assert "cp -a /app/. /workspace/repo/" in run_calls[0][1]["command"][1]
    assert run_calls[1][1]["volumes"] == {"/host/sandbox/repo": {"bind": "/app", "mode": "rw"}}
    assert result["workspace"] == {
        "mode": "shared_bind_mount",
        "copy_mount_scope": "repository_only",
        "agent_repository": "/workspace/repo",
        "task_repository": "/app",
        "read_write": True,
    }


@pytest.mark.asyncio
async def test_shared_workspace_probe_round_trips_through_task_runtime(monkeypatch) -> None:
    class SandyStub:
        value = ""

        async def execute_command(self, _sandbox_id, command, **_kwargs):
            if command.startswith("printf %s"):
                self.value = command.split("agent-", 1)[1].split(" ", 1)[0]
                self.value = "agent-" + self.value
                return {"exit_code": 0, "stdout": ""}
            if command.startswith("cat "):
                return {"exit_code": 0, "stdout": self.value}
            if command.startswith("rm -f "):
                self.value = ""
                return {"exit_code": 0, "stdout": ""}
            raise AssertionError(command)

    sandy = SandyStub()

    async def outside_exec(_adapter, _container, argv, **_kwargs):
        assert argv[:2] == ["sh", "-c"]
        assert "/app/.chutes-workspace-probe-" in argv[2]
        sandy.value = "task-" + sandy.value.removeprefix("agent-")
        return {"exit_code": 0, "stdout": ""}

    monkeypatch.setattr(
        deepswe_module.TerminalBenchAdapter,
        "_docker_exec_outside",
        outside_exec,
    )
    adapter = DeepSWEAdapter.__new__(DeepSWEAdapter)
    adapter.sandy = sandy

    proof = await adapter._verify_shared_task_workspace("sandbox", "task-container")

    assert proof["shared"] is True
    assert proof["agent_repository"] == "/workspace/repo"
    assert proof["task_repository"] == "/app"


@pytest.mark.parametrize("agent", ["chutescoder", "chutescoder-baseline", "codex"])
def test_every_codex_family_arm_enters_the_same_repository(
    agent: str,
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    observed = tmp_path / "observed.json"
    binary = tmp_path / "real-agent"
    binary.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "Path(os.environ['OBSERVED']).write_text(json.dumps({"
        "'cwd': os.getcwd(), 'args': sys.argv[1:]}))\n"
    )
    binary.chmod(0o755)
    wrapper = tmp_path / deepswe_module.DEEPSWE_AGENT_COMMANDS[agent]
    wrapper.write_text(DeepSWEAdapter._agent_workdir_wrapper(str(binary), str(repository)))
    wrapper.chmod(0o755)

    subprocess.run(
        [str(wrapper), "--cwd", "/workspace", "--model", "test-model"],
        check=True,
        env={**os.environ, "OBSERVED": str(observed)},
    )

    assert json.loads(observed.read_text()) == {
        "cwd": str(repository),
        "args": ["--cwd", str(repository), "--model", "test-model"],
    }
    env = DeepSWEAdapter._agent_workspace_env({"TOKEN": "value"})
    assert env["PATH"].split(":", 1)[0] == "/workspace/.chutes/bin"
    assert env["TOKEN"] == "value"


class _RetentionSandy:
    def __init__(self):
        self.commands: list[str] = []

    async def execute_command(self, _sandbox_id, command, timeout_ms=None):
        self.commands.append(command)
        return {"exit_code": 0, "stdout": ""}


@pytest.mark.asyncio
async def test_excluded_path_mirrors_rollout_before_evidence_archive(monkeypatch) -> None:
    calls = []

    async def retain_rollout(_sandy, sandbox_id, launch):
        calls.append(("rollout", sandbox_id, launch))

    async def retain_evidence(_sandy, sandbox_id, **kwargs):
        assert kwargs["require_rollout"] is True
        calls.append(("evidence", sandbox_id, kwargs["item_id"]))
        return {
            "status": "retained",
            "path": "/evidence/deepswe.tar.gz",
            "sha256": "a" * 64,
            "size_bytes": 123,
            "error": None,
            "token_usage_samples": {
                "events_seen": 2,
                "samples": [{"sequence": 1}, {"sequence": 2}],
            },
        }

    monkeypatch.setattr(deepswe_module, "retain_sandy_agent_rollout", retain_rollout)
    monkeypatch.setattr(deepswe_module, "retain_agent_evidence", retain_evidence)
    adapter = DeepSWEAdapter.__new__(DeepSWEAdapter)
    adapter.sandy = _RetentionSandy()
    adapter.run_id = "run"
    adapter._item_observability = {}
    state = adapter._new_item_observability("task")
    launch = SimpleNamespace(setup=object())
    state.update(agent_invoked=True, agent_launch=launch)

    await adapter._finish_evidence_retention("task", "sandbox")
    result = adapter.attach_item_observability(
        deepswe_module.ItemResult(
            item_id="task",
            score=0.0,
            is_correct=False,
            metadata={"exclusion_reason": "infrastructure_transport"},
        )
    )

    assert calls == [
        ("rollout", "sandbox", launch),
        ("evidence", "sandbox", "task"),
    ]
    assert result.score == 0.0
    assert result.metadata["exclusion_reason"] == "infrastructure_transport"
    assert result.agent_evidence_status == "retained"
    assert result.agent_evidence_sha256 == "a" * 64
    assert result.agent_evidence_size_bytes == 123
    assert [sample["sequence"] for sample in result.token_usage_samples["samples"]] == [1, 2]


def test_evidence_failure_is_recorded_without_changing_deepswe_score() -> None:
    adapter = DeepSWEAdapter.__new__(DeepSWEAdapter)
    adapter._item_observability = {
        "task": {
            "agent_invoked": True,
            "evidence": {
                "status": "failed",
                "path": None,
                "sha256": None,
                "size_bytes": None,
                "error": "SHA-256 mismatch",
                "token_usage_samples": None,
            },
        }
    }
    scored = deepswe_module.ItemResult(item_id="task", score=1.0, is_correct=True)

    result = adapter.attach_item_observability(scored)

    assert result.score == 1.0
    assert result.is_correct is True
    assert result.error is None
    assert result.agent_evidence_status == "failed"
    assert result.agent_evidence_error == "SHA-256 mismatch"


@pytest.mark.parametrize("agent", ["chutescoder", "chutescoder-baseline", "codex"])
def test_compaction_experiment_metrics_are_queryable_per_arm(agent: str) -> None:
    adapter = DeepSWEAdapter.__new__(DeepSWEAdapter)
    adapter._item_observability = {
        "task": {
            "agent": agent,
            "context_limit_tokens": 48_000,
            "configured_context_window": 48_000,
            "evidence": {
                "status": "retained",
                "path": "/evidence/deepswe.tar.gz",
                "sha256": "a" * 64,
                "size_bytes": 123,
                "error": None,
                "token_usage_samples": None,
                "rollout_metrics": {
                    "complete": True,
                    "compaction_events": 3,
                    "compaction_events_by_type": {"context_compacted": 3},
                    "rollout_line_count": 456,
                    "tool_calls_by_name": {"python": 88},
                    "rlm_native_helper_calls_by_name": {"read_file": 4, "grep": 2},
                    "rlm_native_helper_paths": ["src", "src/main.py"],
                    "rlm_python_cells_with_subprocess": 0,
                    "rlm_python_cells_with_docker": 0,
                },
            },
        }
    }

    result = adapter.attach_item_observability(
        deepswe_module.ItemResult(item_id="task", score=1.0, is_correct=True)
    )

    assert result.metadata["compaction_experiment"] == {
        "schema_version": 1,
        "arm": agent,
        "context_limit_tokens": 48_000,
        "configured_context_window": 48_000,
        "compaction_events": 3,
        "compaction_events_by_type": {"context_compacted": 3},
        "rollout_line_count": 456,
        "tool_calls_by_name": {"python": 88},
        "rollout_metrics_complete": True,
        "score": 1.0,
    }
    assert result.metadata["repository_access"] == {
        "agent_repository": "/workspace/repo",
        "task_repository": "/app",
        "rlm_native_helper_calls_by_name": {"read_file": 4, "grep": 2},
        "rlm_native_helper_paths": ["src", "src/main.py"],
        "rlm_python_cells_with_subprocess": 0,
        "rlm_python_cells_with_docker": 0,
    }


def test_deepswe_finalizer_waits_for_evidence_before_sandbox_teardown() -> None:
    source = inspect.getsource(DeepSWEAdapter._evaluate_item)

    assert source.index("_finish_evidence_retention") < source.index(
        "_cleanup_owned_task_containers"
    )
    assert source.index("_finish_evidence_retention") < source.index("terminate_sandbox")


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


@pytest.mark.asyncio
async def test_agent_must_be_proven_terminated_before_verifier_upload() -> None:
    class SandyStub:
        def __init__(self, *stdout: str):
            self.stdout = list(stdout)
            self.calls = 0

        async def execute_command(self, *_args, **_kwargs):
            value = self.stdout[min(self.calls, len(self.stdout) - 1)]
            self.calls += 1
            return {"exit_code": 0, "stdout": value}

    adapter = DeepSWEAdapter.__new__(DeepSWEAdapter)
    adapter.sandy = SandyStub("PID=123 STATE=S RUNNING=yes DONE_PRESENT=no DONE=missing")
    running = await adapter._verify_agent_terminated("sandbox", {"type": "complete"})
    assert running["terminated"] is False
    assert DEEPSWE_AGENT_NOT_TERMINATED_EXCLUSION_REASON == ("infrastructure_agent_not_terminated")

    adapter.sandy = SandyStub("PID=123 STATE=gone RUNNING=no DONE_PRESENT=yes DONE=0")
    stopped = await adapter._verify_agent_terminated("sandbox", {"type": "complete"})
    assert stopped["terminated"] is True
    assert stopped["done_value"] == 0

    missing_completion = await adapter._verify_agent_terminated("sandbox", {"exitCode": 0})
    assert missing_completion["terminated"] is False


@pytest.mark.asyncio
async def test_agent_termination_recovers_from_sandy_premature_complete() -> None:
    class SandyStub:
        def __init__(self):
            self.responses = iter(
                [
                    "PID=123 STATE=S RUNNING=yes DONE_PRESENT=no DONE=missing",
                    "PID=123 STATE=Z RUNNING=no DONE_PRESENT=yes DONE=0",
                ]
            )

        async def execute_command(self, *_args, **_kwargs):
            return {"exit_code": 0, "stdout": next(self.responses)}

    adapter = DeepSWEAdapter.__new__(DeepSWEAdapter)
    adapter.sandy = SandyStub()

    recovered = await adapter._verify_agent_terminated(
        "sandbox",
        {"type": "complete", "exitCode": 1},
        wait_timeout_seconds=1,
        poll_interval_seconds=0,
    )

    assert recovered["terminated"] is True
    assert recovered["attempts"] == 2
    assert "RUNNING=yes" in recovered["initial_probe"]
    assert recovered["done_value"] == 0


def test_verifier_requires_current_in_container_execution_proof() -> None:
    reward, _, exclusion, _ = classify_deepswe_verifier_outcome(
        {"exit_code": -1, "error": "transport closed"},
        {"exit_code": 0, "stdout": '{"reward": 1}'},
        test_command_executed=False,
    )
    assert reward is None
    assert exclusion == DEEPSWE_VERIFIER_NOT_EXECUTED_EXCLUSION_REASON


def test_valid_zero_is_scored_even_if_verifier_shell_exits_nonzero() -> None:
    reward, metrics, exclusion, error = classify_deepswe_verifier_outcome(
        {"exit_code": 1, "stderr": "tests failed"},
        {"exit_code": 0, "stdout": '{"reward": 0, "passed": 2}'},
        test_command_executed=True,
    )
    assert reward == 0.0
    assert metrics["passed"] == 2
    assert exclusion is None
    assert error is None


@pytest.mark.parametrize("raw_reward", ["-1", '{"reward": -1}', "not-json"])
def test_invalid_or_infrastructure_reward_is_excluded(raw_reward: str) -> None:
    reward, _, exclusion, _ = classify_deepswe_verifier_outcome(
        {"exit_code": 6},
        {"exit_code": 0, "stdout": raw_reward},
        test_command_executed=True,
    )
    assert reward is None
    assert exclusion == "infrastructure_verifier"


@pytest.mark.asyncio
async def test_network_seal_requires_a_real_sandbox_fetch_probe(monkeypatch) -> None:
    class SandyStub:
        def __init__(self, sandbox_probe: str):
            self.sandbox_probe = sandbox_probe
            self.calls = 0

        async def execute_command(self, *_args, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                return {"exit_code": 0, "stdout": ""}
            return {"exit_code": 0, "stdout": self.sandbox_probe}

    async def outside_exec(*_args, **_kwargs):
        return {"exit_code": 0, "stdout": "1\n", "stderr": ""}

    class Container:
        attrs = {
            "Config": {"Labels": {"chutes.bench.sandbox_id": "sandbox"}},
            "HostConfig": {"NetworkMode": "none"},
        }

    class Containers:
        @staticmethod
        def get(_name):
            return Container()

    class Client:
        containers = Containers()

    monkeypatch.setattr(
        deepswe_module.TerminalBenchAdapter,
        "_docker_exec_outside",
        outside_exec,
    )
    monkeypatch.setattr(deepswe_module.docker, "from_env", lambda: Client())

    adapter = DeepSWEAdapter.__new__(DeepSWEAdapter)
    adapter.sandy = SandyStub("HOSTS=1\nCURL=no")
    no_probe = await adapter._seal_network("sandbox", "container")
    assert no_probe["sealed"] is False

    adapter.sandy = SandyStub("HOSTS=1\nCURL=yes\nFETCH=200")
    fetched = await adapter._seal_network("sandbox", "container")
    assert fetched["sealed"] is False

    adapter.sandy = SandyStub("HOSTS=1\nCURL=yes\nFETCH=000CURLFAIL")
    blocked = await adapter._seal_network("sandbox", "container")
    assert blocked["sealed"] is True


@pytest.mark.asyncio
async def test_docker_boundary_attempts_fresh_container_source_fetch(monkeypatch) -> None:
    class ExecResult:
        exit_code = 0
        output = (
            b"SOCKET=ABSENT\nCACHE_MOUNT=ABSENT\nCACHE_FILES=1\n"
            b"RAW_DOCKER=BLOCKED\nSPAWN=BLOCKED\n"
            b"OTHER_CONTAINER=BLOCKED\nTASK_PATH=WORKS\n"
        )

    class Sandbox:
        @staticmethod
        def exec_run(argv, **_kwargs):
            assert argv[:2] == ["sh", "-lc"]
            assert "docker run --rm" in argv[2]
            assert deepswe_module.DEEPSWE_SOURCE_PROBE_URL in argv[2]
            return ExecResult()

    monkeypatch.setattr(deepswe_module, "sandbox_container", lambda _id: Sandbox())
    adapter = DeepSWEAdapter.__new__(DeepSWEAdapter)

    held = await adapter._verify_agent_docker_boundary("sandbox", "task")

    assert held["boundary_held"] is True
    assert held["observed_from"] == "worker_docker_api_into_agent_namespace"

    ExecResult.output = ExecResult.output.replace(b"SPAWN=BLOCKED", b"SPAWN=ESCAPED")
    escaped = await adapter._verify_agent_docker_boundary("sandbox", "task")
    assert escaped["boundary_held"] is False


@pytest.mark.asyncio
async def test_agent_view_hash_scan_fails_on_heldout_bytes() -> None:
    digest = "a" * 64

    class SandyStub:
        async def execute_command(self, *_args, **_kwargs):
            return {
                "exit_code": 0,
                "stdout": (
                    f"ANSWERS=0 ARCHIVES=0 CACHE_FILES=0\n{digest}  /tmp/copied-hidden-test\n"
                ),
            }

    adapter = DeepSWEAdapter.__new__(DeepSWEAdapter)
    adapter.sandy = SandyStub()
    proof = await adapter._verify_workspace_clean("sandbox", [{"sha256": digest, "size": 123}])

    assert proof["clean"] is False
    assert proof["heldout_hash_matches"] == [f"{digest}  /tmp/copied-hidden-test"]


@pytest.mark.asyncio
async def test_pre_image_holdout_proof_rejects_every_workspace_archive() -> None:
    commands = []

    class SandyStub:
        async def execute_command(self, _sandbox_id, command, **_kwargs):
            commands.append(command)
            return {"exit_code": 0, "stdout": "ANSWERS=0 ARCHIVES=0 CACHE_FILES=0\n"}

    adapter = DeepSWEAdapter.__new__(DeepSWEAdapter)
    adapter.sandy = SandyStub()

    proof = await adapter._verify_workspace_clean("sandbox")

    assert proof["clean"] is True
    assert "-name '*.tar'" in commands[0]
    assert "-name '*.b64'" in commands[0]
    assert "! -path '/workspace/repo/*'" not in commands[0]


@pytest.mark.asyncio
async def test_final_holdout_proof_only_exempts_archives_inside_public_task_repo() -> None:
    commands = []

    class SandyStub:
        async def execute_command(self, _sandbox_id, command, **_kwargs):
            commands.append(command)
            return {"exit_code": 0, "stdout": "ANSWERS=0 ARCHIVES=0 CACHE_FILES=0\n"}

    adapter = DeepSWEAdapter.__new__(DeepSWEAdapter)
    adapter.sandy = SandyStub()

    proof = await adapter._verify_workspace_clean(
        "sandbox",
        task_repository_present=True,
    )

    assert proof["clean"] is True
    assert "! -path '/workspace/repo/*'" in commands[0]


@pytest.mark.asyncio
async def test_agent_view_allows_private_image_cache_without_heldout_bytes() -> None:
    class SandyStub:
        async def execute_command(self, *_args, **_kwargs):
            return {
                "exit_code": 0,
                "stdout": "ANSWERS=0 ARCHIVES=0 CACHE_FILES=1\n",
            }

    adapter = DeepSWEAdapter.__new__(DeepSWEAdapter)
    adapter.sandy = SandyStub()

    proof = await adapter._verify_workspace_clean(
        "sandbox",
        [{"sha256": "a" * 64, "size": 123}],
    )

    assert proof["clean"] is True
    assert proof["heldout_hash_matches"] == []
