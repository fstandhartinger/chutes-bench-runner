"""Identity and integrity tests for Terminal-Bench adapters."""

from __future__ import annotations

import io
import tarfile
from collections import Counter
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.benchmarks.adapters.terminal_bench import (
    AGENT_LAUNCH_FAILED_EXCLUSION_REASON,
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
    classify_agent_launch_failure,
    classify_bare_failure,
    classify_verifier_network_failure,
)
from app.benchmarks.adapters.terminal_bench_identity import (
    TERMINAL_BENCH_1,
    TERMINAL_BENCH_2_0,
    TERMINAL_BENCH_2_1,
    TERMINAL_BENCH_HARD,
)
from app.benchmarks.adapters.terminal_bench_scoring import (
    FUNCTIONAL,
    PERFORMANCE_GATED,
    RESOURCE_GATED,
    TERMINAL_BENCH_2_1_GATED_TASKS,
    TERMINAL_BENCH_2_1_SCORING_AUDIT_COMMIT,
    terminal_bench_2_1_scoring_classification,
)
from app.benchmarks.base import ItemResult


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


def _audited_items() -> list[dict]:
    items = []
    for index, task_id in enumerate(TERMINAL_BENCH_2_1.task_ids):
        classification = terminal_bench_2_1_scoring_classification(task_id)
        items.append(
            {
                "id": str(index),
                "task_id": task_id,
                "scoring_class": classification["scoring_class"],
                "scoring_reason": classification["reason"],
                "scoring_evidence": classification["evidence"],
            }
        )
    return items


def test_terminal_bench_2_1_scoring_audit_covers_exact_release() -> None:
    assert TERMINAL_BENCH_2_1.commit == TERMINAL_BENCH_2_1_SCORING_AUDIT_COMMIT
    assert len(TERMINAL_BENCH_2_1.task_ids) == 89
    assert set(TERMINAL_BENCH_2_1_GATED_TASKS) < set(TERMINAL_BENCH_2_1.task_ids)

    classes = Counter(
        terminal_bench_2_1_scoring_classification(task_id)["scoring_class"]
        for task_id in TERMINAL_BENCH_2_1.task_ids
    )
    assert classes == {
        FUNCTIONAL: 75,
        PERFORMANCE_GATED: 4,
        RESOURCE_GATED: 10,
    }
    assert {
        task_id
        for task_id, classification in TERMINAL_BENCH_2_1_GATED_TASKS.items()
        if classification["scoring_class"] == PERFORMANCE_GATED
    } == {
        "largest-eigenval",
        "portfolio-optimization",
        "query-optimize",
        "tune-mjcf",
    }
    assert {
        task_id
        for task_id, classification in TERMINAL_BENCH_2_1_GATED_TASKS.items()
        if classification["scoring_class"] == RESOURCE_GATED
    } == {
        "circuit-fibsqrt",
        "gpt2-codegolf",
        "large-scale-text-editing",
        "llm-inference-batching-scheduler",
        "path-tracing",
        "path-tracing-reverse",
        "regex-chess",
        "reshard-c4-data",
        "train-fasttext",
        "write-compressor",
    }
    for classification in TERMINAL_BENCH_2_1_GATED_TASKS.values():
        assert classification["evidence"]
        assert all(evidence["assertion"] for evidence in classification["evidence"])


@pytest.mark.asyncio
async def test_capability_filter_is_opt_in_and_reports_every_exclusion() -> None:
    adapter = TerminalBench21Adapter.__new__(TerminalBench21Adapter)
    adapter._items = _audited_items()
    adapter.run_config = {}

    total, standard_items = await adapter.get_items_for_evaluation(100, "seed")
    standard_metrics = await adapter.postprocess([])

    assert total == 89
    assert len(standard_items) == 89
    standard_policy = standard_metrics["terminal_bench_scoring_policy"]
    assert standard_policy["mode"] == "standard"
    assert standard_policy["excluded_task_count"] == 0
    assert standard_policy["classified_gated_task_count"] == 14
    assert standard_policy["standard_terminal_bench_score"] is True

    adapter.run_config = {
        "terminal_bench_2_1": {
            "exclude_performance_and_resource_gated_tasks": True,
        }
    }
    total, capability_items = await adapter.get_items_for_evaluation(100, "seed")
    capability_metrics = await adapter.postprocess([])

    assert total == 89
    assert len(capability_items) == 75
    assert all(
        adapter._items[int(item_id)]["scoring_class"] == FUNCTIONAL for item_id in capability_items
    )
    capability_policy = capability_metrics["terminal_bench_scoring_policy"]
    assert capability_policy["mode"] == "capability_only"
    assert capability_policy["standard_terminal_bench_score"] is False
    assert capability_policy["thresholds_relaxed"] is False
    assert capability_policy["excluded_task_count"] == 14
    assert capability_policy["excluded_by_class"] == {
        PERFORMANCE_GATED: 4,
        RESOURCE_GATED: 10,
    }
    assert {entry["task_id"] for entry in capability_policy["excluded_tasks"]} == set(
        TERMINAL_BENCH_2_1_GATED_TASKS
    )
    assert "NON-STANDARD CAPABILITY-ONLY SCORE" in capability_policy["summary"]


@pytest.mark.asyncio
async def test_capability_filter_never_silently_drops_entire_explicit_selection() -> None:
    adapter = TerminalBench21Adapter.__new__(TerminalBench21Adapter)
    adapter._items = _audited_items()
    largest_eigenval_id = str(TERMINAL_BENCH_2_1.task_ids.index("largest-eigenval"))
    adapter.run_config = {
        "terminal_bench_2_1": {
            "item_ids": [largest_eigenval_id],
            "exclude_performance_and_resource_gated_tasks": True,
        }
    }

    with pytest.raises(ValueError, match="excluded every selected task.*largest-eigenval"):
        await adapter.get_items_for_evaluation(100, "seed")


def test_scoring_class_is_attached_even_to_worker_created_error_result() -> None:
    adapter = TerminalBench21Adapter.__new__(TerminalBench21Adapter)
    adapter._items = _audited_items()
    adapter._item_observability = {}
    item_id = str(TERMINAL_BENCH_2_1.task_ids.index("largest-eigenval"))

    result = adapter.attach_item_observability(ItemResult(item_id=item_id, error="worker timeout"))

    assert result.metadata["scoring_class"] == PERFORMANCE_GATED
    classification = result.metadata["scoring_classification"]
    assert classification["audit_commit"] == TERMINAL_BENCH_2_1.commit
    assert classification["evidence"][0]["assertion"] == "assert dt < ref_dt"


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
    assert TERMINAL_BENCH_HARD.repository == "harbor-framework/terminal-bench-1"
    assert TERMINAL_BENCH_HARD.commit == "74221fb0b6b5a7f88e53bed5726edaaf236348c9"
    assert TERMINAL_BENCH_HARD.manifest_repository == "NVIDIA-NeMo/Evaluator"
    assert TERMINAL_BENCH_HARD.manifest_commit == "bd952253260e7077973aadf5fc656e425d2758e1"
    assert TERMINAL_BENCH_HARD.task_ids[0] == "aimo-airline-departures"
    assert TERMINAL_BENCH_HARD.task_ids[-1] == "write-compressor"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("adapter_class", "spec"),
    (
        (TerminalBench1Adapter, TERMINAL_BENCH_1),
        (TerminalBench20Adapter, TERMINAL_BENCH_2_0),
        (TerminalBench21Adapter, TERMINAL_BENCH_2_1),
        (TerminalBenchHardAdapter, TERMINAL_BENCH_HARD),
    ),
)
async def test_each_pinned_source_archive_is_reachable_and_loadable(
    adapter_class, spec, tmp_path, monkeypatch
) -> None:
    """Exercise the real source URL, checksum, layout, and task membership."""
    # An empty per-test cache is intentional: a previously downloaded archive
    # must not hide a dead pin, which was the production failure this guards.
    monkeypatch.setenv("TERMINAL_BENCH_DATASET_CACHE", str(tmp_path))
    adapter = adapter_class.__new__(adapter_class)

    archive = await adapter._load_source_archive()
    adapter._items = adapter._items_from_source_archive(archive)
    adapter._assert_benchmark_identity()

    assert len(adapter._items) == spec.expected_count
    assert tuple(item["task_id"] for item in adapter._items) == spec.task_ids
    if spec.commit == TERMINAL_BENCH_2_1_SCORING_AUDIT_COMMIT:
        assert Counter(item["scoring_class"] for item in adapter._items) == {
            FUNCTIONAL: 75,
            PERFORMANCE_GATED: 4,
            RESOURCE_GATED: 10,
        }
        items_by_task = {item["task_id"]: item for item in adapter._items}
        for task_id, classification in TERMINAL_BENCH_2_1_GATED_TASKS.items():
            with tarfile.open(
                fileobj=io.BytesIO(items_by_task[task_id]["archive"]), mode="r:"
            ) as task_archive:
                for evidence in classification["evidence"]:
                    source = task_archive.extractfile(evidence["file"])
                    assert source is not None
                    lines = source.read().decode("utf-8").splitlines()
                    source_line = lines[evidence["line"] - 1].strip()
                    assertion = evidence["assertion"].split("  #", 1)[0]
                    assert assertion in source_line


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
        return_value={
            "exit_code": 0,
            "stdout": (
                "HOLDOUT_READ=BLOCKED\nCANDIDATES_BEGIN\nCANDIDATES_END\n"
                "SOURCE_ARCHIVE=ABSENT\n"
            ),
        }
    )
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w") as task:
        for name, content in {
            "instruction.md": b"Do the work",
            "solution/solve.sh": b"secret solution",
            "tests/test_outputs.py": b"secret tests",
        }.items():
            member = tarfile.TarInfo(name)
            member.size = len(content)
            task.addfile(member, io.BytesIO(content))
    partition = adapter._partition_answer_key(archive.getvalue())

    holdout = await adapter._withhold_answer_key("sandbox", "s123", partition)

    assert holdout["withheld"] is True
    holdout_command = adapter.sandy.execute_command.await_args_list[0].args[1]
    assert "test -r /opt/tb-holdout/s123" in holdout_command
    assert "find / -type f" in holdout_command
    assert partition["source_archive_sha256"] in holdout_command


def test_answer_key_is_partitioned_before_any_bytes_enter_sandy() -> None:
    adapter = TerminalBench21Adapter.__new__(TerminalBench21Adapter)
    archive = io.BytesIO()
    files = {
        "instruction.md": b"public instruction",
        "task.toml": b"[environment]",
        "solution/solve.sh": b"reference bytes",
        "tests/test_outputs.py": b"test bytes",
        "evaluation_tests_hidden/cases.json": b"hidden bytes",
    }
    with tarfile.open(fileobj=archive, mode="w") as task:
        for name, content in files.items():
            member = tarfile.TarInfo(name)
            member.size = len(content)
            task.addfile(member, io.BytesIO(content))

    partition = adapter._partition_answer_key(archive.getvalue())

    with tarfile.open(fileobj=io.BytesIO(partition["safe_archive"])) as safe:
        assert set(safe.getnames()) == {"instruction.md", "task.toml"}
    with tarfile.open(fileobj=io.BytesIO(partition["tests_archive"])) as tests:
        assert set(tests.getnames()) == {
            "tests/test_outputs.py",
            "evaluation_tests_hidden/cases.json",
        }
        assert b"reference bytes" not in partition["tests_archive"]
    assert [entry["path"] for entry in partition["reference_manifest"]] == [
        "solution/solve.sh"
    ]
    assert partition["source_archive_size_bytes"] == len(archive.getvalue())


def test_nested_archive_with_answer_key_is_rejected_before_sandy() -> None:
    nested = io.BytesIO()
    with tarfile.open(fileobj=nested, mode="w") as hidden:
        content = b"nested secret"
        member = tarfile.TarInfo("payload/solution/solve.sh")
        member.size = len(content)
        hidden.addfile(member, io.BytesIO(content))
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w") as task:
        public = b"public"
        instruction = tarfile.TarInfo("instruction.md")
        instruction.size = len(public)
        task.addfile(instruction, io.BytesIO(public))
        embedded = nested.getvalue()
        fixture = tarfile.TarInfo("fixtures/data.bin")
        fixture.size = len(embedded)
        task.addfile(fixture, io.BytesIO(embedded))
        tests = b"tests"
        test_member = tarfile.TarInfo("tests/test.py")
        test_member.size = len(tests)
        task.addfile(test_member, io.BytesIO(tests))

    with pytest.raises(BenchmarkIdentityError, match="Nested archive.*answer key"):
        TerminalBench21Adapter._partition_answer_key(archive.getvalue())


@pytest.mark.asyncio
async def test_answer_key_probe_fails_closed_on_holdout_or_archive_leak() -> None:
    adapter = TerminalBench21Adapter.__new__(TerminalBench21Adapter)
    adapter.sandy = type("Sandy", (), {})()
    adapter.sandy.execute_command = AsyncMock(
        return_value={
            "exit_code": 0,
            "stdout": (
                "HOLDOUT_READ=SUCCEEDED\nCANDIDATES_BEGIN\nCANDIDATES_END\n"
                "SOURCE_ARCHIVE=PRESENT\n"
            ),
        }
    )
    partition = {
        "source_archive_sha256": "a" * 64,
        "source_archive_size_bytes": 42,
        "tests_manifest": [],
        "reference_manifest": [],
    }

    verdict = await adapter._withhold_answer_key("sandbox", "s123", partition)

    assert verdict["withheld"] is False
    assert verdict["read_attempt"] == "succeeded"
    assert verdict["source_archive_absent"] is False


@pytest.mark.asyncio
async def test_cancellation_cleanup_targets_only_matching_sandy_owner_and_id(
    monkeypatch,
) -> None:
    sandbox_id = "123456781234"
    adapter = TerminalBench21Adapter.__new__(TerminalBench21Adapter)
    owned = MagicMock()
    owned.attrs = {
        "Config": {"Labels": {"chutes.bench.sandbox_id": sandbox_id}}
    }
    decoy = MagicMock()
    decoy.attrs = {
        "Config": {"Labels": {"chutes.bench.sandbox_id": "different"}}
    }
    image_tag = "tbench_s123456781234_task:latest"
    image = SimpleNamespace(tags=[image_tag])
    network = MagicMock()
    network.attrs = {"Labels": {"chutes.bench.sandbox_id": sandbox_id}}
    removed_containers: set[int] = set()
    removed_networks: set[int] = set()

    def remove_owned(*, force: bool) -> None:
        assert force is True
        removed_containers.add(id(owned))

    def list_containers(*, all: bool, filters: dict) -> list:
        assert all is True
        assert filters == {"label": f"chutes.bench.sandbox_id={sandbox_id}"}
        return [
            container
            for container in (owned, decoy)
            if id(container) not in removed_containers
        ]

    def remove_image(tag: str, *, force: bool) -> None:
        assert force is True
        image.tags.remove(tag)

    def remove_network() -> None:
        removed_networks.add(id(network))

    def list_networks(*, filters: dict) -> list:
        assert filters == {"label": f"chutes.bench.sandbox_id={sandbox_id}"}
        return [] if id(network) in removed_networks else [network]

    owned.remove.side_effect = remove_owned
    network.remove.side_effect = remove_network
    client = SimpleNamespace(
        containers=SimpleNamespace(list=MagicMock(side_effect=list_containers)),
        images=SimpleNamespace(
            list=MagicMock(return_value=[image]),
            remove=MagicMock(side_effect=remove_image),
        ),
        networks=SimpleNamespace(list=MagicMock(side_effect=list_networks)),
    )
    monkeypatch.setattr(
        "app.benchmarks.adapters.terminal_bench.docker.from_env",
        lambda: client,
    )

    assert await adapter._cleanup_owned_task_containers(sandbox_id) is True

    client.containers.list.assert_any_call(
        all=True,
        filters={"label": f"chutes.bench.sandbox_id={sandbox_id}"},
    )
    owned.remove.assert_called_once_with(force=True)
    decoy.remove.assert_not_called()
    client.images.remove.assert_called_once_with(image_tag, force=True)
    network.remove.assert_called_once_with()


@pytest.mark.asyncio
async def test_harbor_task_container_carries_sandbox_ownership_labels() -> None:
    sandbox_id = "123456781234"
    owner = "/var/lib/sandy/state"
    adapter = TerminalBench21Adapter.__new__(TerminalBench21Adapter)
    adapter.sandy = type("Sandy", (), {})()
    adapter.sandy.execute_command = AsyncMock(
        side_effect=[
            {"exit_code": 0},  # pull
            {"exit_code": 0, "stdout": f"{sandbox_id}|{owner}\n"},
            {"exit_code": 0},  # run
            {"exit_code": 0},  # mkdir
        ]
    )

    result = await adapter._run_harbor_task(
        sandbox_id,
        {"task_id": "task-one", "docker_image": "example/task:latest"},
    )

    assert result["container_name"].startswith("tbench_s123456781234_")
    run_command = adapter.sandy.execute_command.await_args_list[2].args[1]
    assert f"sandy.owner={owner}" in run_command
    assert f"chutes.bench.sandbox_id={sandbox_id}" in run_command


def test_agent_exit_classification_still_excludes_only_dead_sandbox() -> None:
    summary = {"exitCode": 1, "duration": 10}
    exclusion, _ = classify_agent_exit(summary, 100, False)
    assert exclusion == "infrastructure_sandbox_gone"

    exclusion, note = classify_agent_exit(summary, 100, True)
    assert exclusion is None
    assert "Scored" in (note or "")


def test_agent_run_http_400_is_excluded_as_launch_infrastructure() -> None:
    error = (
        "Client error '400 Bad Request' for url "
        "'http://host.docker.internal:7331/api/sandboxes/sbx/agent/run'"
    )

    assert (
        classify_bare_failure(error, None)
        == AGENT_LAUNCH_FAILED_EXCLUSION_REASON
    )


def test_unknown_agent_is_excluded_as_launch_infrastructure() -> None:
    assert (
        classify_bare_failure("Unknown agent: prime-agent", None)
        == AGENT_LAUNCH_FAILED_EXCLUSION_REASON
    )


def test_agent_invocation_without_summary_tokens_or_rollout_is_excluded() -> None:
    assert (
        classify_agent_launch_failure(
            "",
            {},
            {"error": "no agent usage file found"},
            agent_invoked=True,
        )
        == AGENT_LAUNCH_FAILED_EXCLUSION_REASON
    )
    assert (
        classify_agent_launch_failure(
            "",
            {},
            {"usage_source": "codex-token-count", "input_tokens": 10},
            agent_invoked=True,
        )
        is None
    )


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
