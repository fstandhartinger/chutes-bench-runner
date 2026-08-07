"""Pinned SWE-bench Verified driven by Sandy CLI coding agents.

The official dataset and evaluator are used, but the agent scaffold is the
bench-runner's selectable Sandy CLI arm. Scores therefore describe this
scaffold and must not be labelled as another lab's self-reported harness.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import hashlib
import importlib.metadata
import json
import math
import os
import shlex
import tempfile
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import docker
from huggingface_hub import hf_hub_download

from app.benchmarks.adapters.swe_bench_verified_identity import (
    SWE_BENCH_VERIFIED,
    SWEBenchVerifiedSpec,
)
from app.benchmarks.adapters.terminal_bench import (
    BenchmarkIdentityError,
    TerminalBenchAdapter,
    classify_agent_exit,
    classify_bare_failure,
)
from app.benchmarks.agent_evidence import retain_agent_evidence
from app.benchmarks.agent_provider_config import (
    prepare_sandy_agent_launch,
    retain_sandy_agent_rollout,
    validate_openrouter_agent_usage,
)
from app.benchmarks.agent_usage import collect_agent_usage
from app.benchmarks.base import BenchmarkAdapter, ItemResult
from app.benchmarks.registry import register_adapter
from app.benchmarks.utils import get_bench_data_dir
from app.core.logging import get_logger
from app.services.provenance_service import sandbox_container
from app.services.sandy_service import SandyService

logger = get_logger(__name__)

SWE_BENCH_VERIFIED_HARNESS = "sandy-cli-swebench-4.1.0-separate-verifier"
SWE_BENCH_VERIFIED_ITEM_TIMEOUT_MARGIN_SECONDS = 15 * 60
SWE_BENCH_VERIFIED_EVIDENCE_EXCLUSION_REASON = "infrastructure_evidence_retention"
SWE_BENCH_VERIFIED_VERIFIER_NOT_EXECUTED = "infrastructure_verifier_not_executed"
SWE_BENCH_VERIFIED_SOURCE_PROBE_URL = (
    "https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified/resolve/"
    f"{SWE_BENCH_VERIFIED.dataset_commit}/README.md"
)


def classify_swe_bench_verified_agent_outcome(
    agent_summary: dict | None,
    agent_timeout_sec: float,
    sandbox_alive: bool | None,
) -> tuple[str | None, str | None]:
    """Keep infrastructure loss out of the score without hiding live CLI crashes."""
    if not agent_summary:
        return (
            "infrastructure_transport",
            "Agent stream ended without a Sandy completion summary.",
        )
    return classify_agent_exit(agent_summary, agent_timeout_sec, sandbox_alive)


def classify_swe_bench_verified_verifier_outcome(
    verifier: dict[str, Any],
) -> tuple[float | None, str | None, str | None]:
    """Classify a model failure separately from an unproven verifier run."""
    if verifier.get("patch_applied") is False:
        return 0.0, None, verifier.get("patch_error") or "Agent patch did not apply"
    if not verifier.get("test_command_executed"):
        return (
            None,
            SWE_BENCH_VERIFIED_VERIFIER_NOT_EXECUTED,
            verifier.get("error") or "SWE-bench verifier command did not execute",
        )
    if verifier.get("timed_out"):
        return None, "infrastructure_verifier_timeout", "SWE-bench verifier timed out"
    report = verifier.get("report")
    if not isinstance(report, dict) or not isinstance(report.get("resolved"), bool):
        return (
            None,
            "infrastructure_verifier",
            verifier.get("error") or "Official SWE-bench grader produced no valid report",
        )
    return (1.0 if report["resolved"] else 0.0), None, None


@register_adapter("swe_bench_verified")
class SWEBenchVerifiedAdapter(BenchmarkAdapter):
    """Official 500-task SWE-bench Verified with a selectable Sandy CLI arm."""

    benchmark_spec: SWEBenchVerifiedSpec = SWE_BENCH_VERIFIED
    BENCHMARK_SOURCE_HOSTS = (
        "github.com",
        "www.github.com",
        "api.github.com",
        "raw.githubusercontent.com",
        "codeload.github.com",
        "objects.githubusercontent.com",
        "huggingface.co",
        "hf.co",
        "cdn-lfs.huggingface.co",
        "cdn-lfs-us-1.hf.co",
        "swebench.com",
        "www.swebench.com",
        "swe-bench.github.io",
        "raw.githack.com",
        "cdn.jsdelivr.net",
        "gitclone.com",
        "ghproxy.com",
    )
    SEAL_MARKER = "chutes-bench-runner: SWE-bench Verified answer sources"

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._items: list[dict[str, Any]] = []
        self._item_observability: dict[str, dict[str, Any]] = {}
        self.sandy = SandyService()

    def get_name(self) -> str:
        return "swe_bench_verified"

    def get_display_name(self) -> str:
        return self.benchmark_spec.display_name

    def requires_setup(self) -> bool:
        return True

    def get_setup_notes(self) -> str | None:
        return (
            "Requires worker-side Docker access and swebench==4.1.0. The Sandy agent "
            "receives neither the Docker socket nor Sandy's shared cache. Official "
            "per-instance images are pulled by the trusted worker."
        )

    def supports_subset(self) -> bool:
        return True

    def supports_parallel_items(self) -> bool:
        return False

    @staticmethod
    def _item_budgets_seconds() -> dict[str, int]:
        return {
            "agent": max(60, int(os.getenv("SWE_BENCH_VERIFIED_AGENT_TIMEOUT_SEC", "3600"))),
            "image_pull": max(
                60, int(os.getenv("SWE_BENCH_VERIFIED_IMAGE_PULL_TIMEOUT_SEC", "1800"))
            ),
            "verifier": max(
                60, int(os.getenv("SWE_BENCH_VERIFIED_VERIFIER_TIMEOUT_SEC", "1800"))
            ),
            "collect": max(
                30, int(os.getenv("SWE_BENCH_VERIFIED_COLLECT_TIMEOUT_SEC", "300"))
            ),
        }

    def get_item_timeout_seconds(self, item_id: str | None = None) -> int | None:
        budget = self._item_budgets_seconds()
        return sum(budget.values()) + SWE_BENCH_VERIFIED_ITEM_TIMEOUT_MARGIN_SECONDS

    async def get_total_items(self) -> int:
        if not self._items:
            await self.preload()
        return len(self._items)

    async def enumerate_items(self) -> AsyncIterator[str]:
        if not self._items:
            await self.preload()
        for item in self._items:
            yield item["id"]

    def _dataset_cache_path(self) -> Path:
        return Path(
            hf_hub_download(
                repo_id=self.benchmark_spec.dataset_repository,
                filename=self.benchmark_spec.dataset_file,
                repo_type="dataset",
                revision=self.benchmark_spec.dataset_commit,
                token=os.getenv("HF_TOKEN"),
                cache_dir=str(get_bench_data_dir() / "hf"),
            )
        )

    def _load_pinned_rows(self) -> list[dict[str, Any]]:
        path = self._dataset_cache_path()
        actual_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_sha256 != self.benchmark_spec.dataset_file_sha256:
            raise BenchmarkIdentityError(
                "swe_bench_verified dataset SHA-256 mismatch: expected "
                f"{self.benchmark_spec.dataset_file_sha256}, got {actual_sha256}"
            )
        from datasets import load_dataset

        dataset = load_dataset(
            "parquet",
            data_files={"test": str(path)},
            split="test",
            cache_dir=str(get_bench_data_dir() / "hf"),
        )
        return [dict(row) for row in dataset]

    def _make_official_test_spec(self, row: dict[str, Any]) -> Any:
        installed = importlib.metadata.version("swebench")
        if installed != self.benchmark_spec.harness_version:
            raise BenchmarkIdentityError(
                "SWE-bench harness version mismatch: expected "
                f"{self.benchmark_spec.harness_version}, got {installed}"
            )
        from swebench.harness.test_spec.test_spec import make_test_spec

        return make_test_spec(
            row,
            namespace=self.benchmark_spec.image_namespace,
            instance_image_tag=self.benchmark_spec.image_tag,
            arch="x86_64",
        )

    @staticmethod
    def _parse_test_list(value: Any, field_name: str, instance_id: str) -> list[str]:
        parsed = value
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError as exc:
                raise BenchmarkIdentityError(
                    f"{instance_id} has invalid {field_name} JSON"
                ) from exc
        if not isinstance(parsed, list) or not all(isinstance(entry, str) for entry in parsed):
            raise BenchmarkIdentityError(f"{instance_id} has invalid {field_name}")
        return parsed

    @staticmethod
    def _heldout_hashes(row: dict[str, Any]) -> list[dict[str, Any]]:
        heldout = []
        for name in ("patch", "test_patch"):
            content = str(row.get(name) or "").encode("utf-8")
            if content:
                heldout.append(
                    {
                        "name": name,
                        "sha256": hashlib.sha256(content).hexdigest(),
                        "size": len(content),
                    }
                )
        return heldout

    def _normalize_row(self, index: int, raw_row: dict[str, Any]) -> dict[str, Any]:
        row = dict(raw_row)
        instance_id = str(row.get("instance_id") or "")
        required = ("instance_id", "repo", "base_commit", "problem_statement", "version")
        missing = [name for name in required if not str(row.get(name) or "").strip()]
        if missing:
            raise BenchmarkIdentityError(
                f"SWE-bench Verified row {index} is missing {', '.join(missing)}"
            )
        for private_name in ("patch", "test_patch"):
            if not isinstance(row.get(private_name), str):
                raise BenchmarkIdentityError(f"{instance_id} has invalid {private_name}")
        fail_to_pass = self._parse_test_list(row.get("FAIL_TO_PASS"), "FAIL_TO_PASS", instance_id)
        pass_to_pass = self._parse_test_list(row.get("PASS_TO_PASS"), "PASS_TO_PASS", instance_id)
        row["FAIL_TO_PASS"] = json.dumps(fail_to_pass)
        row["PASS_TO_PASS"] = json.dumps(pass_to_pass)
        test_spec = self._make_official_test_spec(row)
        return {
            "id": str(index),
            "instance_id": instance_id,
            "repo": str(row["repo"]),
            "base_commit": str(row["base_commit"]),
            "problem_statement": str(row["problem_statement"]),
            "version": str(row["version"]),
            "difficulty": row.get("difficulty"),
            "heldout_hashes": self._heldout_hashes(row),
            # These verifier-only values remain in the trusted worker process.
            # They are never written into the Sandy agent namespace.
            "_dataset_row": row,
            "_test_spec": test_spec,
            "_eval_script": test_spec.eval_script,
            "docker_image": test_spec.instance_image_key,
            "dataset_repository": self.benchmark_spec.dataset_repository,
            "dataset_commit": self.benchmark_spec.dataset_commit,
            "dataset_file_sha256": self.benchmark_spec.dataset_file_sha256,
        }

    async def preload(self) -> None:
        """Load the exact parquet shard and build official v4.1.0 test specs."""
        if self._items:
            return
        try:
            logger.info(
                "Loading pinned SWE-bench Verified dataset",
                repository=self.benchmark_spec.dataset_repository,
                commit=self.benchmark_spec.dataset_commit,
                expected_count=self.benchmark_spec.expected_count,
            )
            rows = await asyncio.to_thread(self._load_pinned_rows)
            self._items = [self._normalize_row(index, row) for index, row in enumerate(rows)]
            self._assert_benchmark_identity()
        except Exception as exc:
            self._items = []
            logger.error("Failed to load SWE-bench Verified", error=str(exc))
            raise

    def _assert_benchmark_identity(self) -> None:
        if len(self._items) != self.benchmark_spec.expected_count:
            raise BenchmarkIdentityError(
                "swe_bench_verified identity check failed: expected "
                f"{self.benchmark_spec.expected_count} items, loaded {len(self._items)}"
            )
        instance_ids = [item.get("instance_id") for item in self._items]
        if len(set(instance_ids)) != len(instance_ids):
            raise BenchmarkIdentityError(
                "swe_bench_verified identity check failed: duplicate instance IDs loaded"
            )
        for item in self._items:
            image = str(item.get("docker_image") or "")
            if not image.startswith(f"{self.benchmark_spec.image_namespace}/sweb.eval.x86_64."):
                raise BenchmarkIdentityError(
                    f"{item.get('instance_id')} has an unexpected official image key: {image}"
                )

    async def _docker_exec_outside(
        self,
        container_name: str,
        argv: list[str],
        *,
        workdir: str | None = None,
        user: str = "",
    ) -> dict[str, Any]:
        return await TerminalBenchAdapter._docker_exec_outside(
            self, container_name, argv, workdir=workdir, user=user
        )

    async def _put_file_outside(
        self,
        container_name: str,
        destination: str,
        content: bytes,
        *,
        mode: int = 0o644,
    ) -> dict[str, Any]:
        return await TerminalBenchAdapter._put_file_outside(
            self, container_name, destination, content, mode=mode
        )

    async def _start_container(
        self,
        *,
        sandbox_id: str,
        item: dict[str, Any],
        container_name: str,
        role: str,
        image_id: str | None = None,
        pull_timeout_sec: int,
    ) -> dict[str, Any]:
        def _get_image() -> Any:
            client = docker.from_env(timeout=max(60, pull_timeout_sec))
            return client.images.get(image_id) if image_id else client.images.pull(
                item["docker_image"]
            )

        # Keep the pull separate from container creation. A timed-out Docker
        # pull can continue in its SDK worker thread, but it must never create
        # an orphan task container after this item has already been excluded.
        try:
            image = await asyncio.wait_for(
                asyncio.to_thread(_get_image), timeout=max(1, pull_timeout_sec)
            )
        except TimeoutError:
            return {"ok": False, "stage": "image_pull", "error": "image pull timed out"}
        except Exception as exc:
            return {"ok": False, "stage": "image_pull", "error": str(exc)}

        def _start() -> dict[str, Any]:
            client = docker.from_env()
            try:
                stale = client.containers.get(container_name)
                stale.remove(force=True)
            except docker.errors.NotFound:
                pass
            container = client.containers.run(
                image.id,
                entrypoint=["/bin/bash", "-lc"],
                command=["sleep infinity"],
                name=container_name,
                detach=True,
                network_mode="none",
                working_dir="/testbed",
                labels={
                    "chutes.benchmark": "swe_bench_verified",
                    "chutes.bench.sandbox_id": sandbox_id,
                    "chutes.bench.role": role,
                },
            )
            container.reload()
            repo_digests = image.attrs.get("RepoDigests") or []
            return {
                "container_id": container.id,
                "image_id": image.id,
                "image_digest": str(repo_digests[0]) if repo_digests else image.id,
                "network_mode": (container.attrs.get("HostConfig") or {}).get("NetworkMode"),
            }

        try:
            started = await asyncio.to_thread(_start)
        except Exception as exc:
            return {"ok": False, "stage": "container_start", "error": str(exc)}
        if started.get("network_mode") != "none":
            return {
                "ok": False,
                "stage": "container_network",
                "error": f"task container network mode was {started.get('network_mode')!r}",
                **started,
            }
        return {"ok": True, "observed_from": "worker_docker_api", **started}

    async def _sanitize_repository(
        self, container_name: str, base_commit: str
    ) -> dict[str, Any]:
        script = (
            "set -e; cd /testbed; "
            f"git cat-file -e {shlex.quote(base_commit)}^{{commit}}; "
            f"git reset --hard {shlex.quote(base_commit)}; "
            "git clean -fd; "
            f"git checkout -q --detach {shlex.quote(base_commit)}; "
            "git for-each-ref --format='%(refname)' | xargs -r -n1 git update-ref -d; "
            f"git branch -q -f main {shlex.quote(base_commit)}; git checkout -q main; "
            "for r in $(git remote); do git remote remove $r; done; "
            "git reflog expire --expire=now --all; git gc --prune=now -q 2>/dev/null; "
            f"echo HEAD=$(git rev-parse HEAD) FUTURE=$(git rev-list --all --not {shlex.quote(base_commit)} --count) "
            "REMOTES=$(git remote | wc -l) DIRTY=$(git status --porcelain --untracked-files=no | wc -l)"
        )
        result = await self._docker_exec_outside(
            container_name, ["bash", "-lc", script], workdir="/testbed"
        )
        stdout = ((result or {}).get("stdout") or "").strip()
        clean = (
            (result or {}).get("exit_code") == 0
            and f"HEAD={base_commit}" in stdout
            and "FUTURE=0" in stdout
            and "REMOTES=0" in stdout
            and "DIRTY=0" in stdout
        )
        return {
            "clean": clean,
            "probe": stdout,
            "result": result,
            "observed_from": "worker_docker_api_inside_task_container",
        }

    async def _verify_workspace_clean(
        self,
        sandbox_id: str,
        heldout_hashes: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        expected_hashes = {entry["sha256"] for entry in (heldout_hashes or [])}
        sizes = sorted({int(entry["size"]) for entry in (heldout_hashes or [])})
        size_expression = " -o ".join(f"-size {size}c" for size in sizes)
        hash_scan = (
            "find / /workspace /root /tmp /opt /var/cache/sandy -xdev -type f "
            f"\\( {size_expression} \\) -exec sha256sum {{}} + 2>/dev/null || true"
            if size_expression
            else ":"
        )
        result = await self.sandy.execute_command(
            sandbox_id,
            "answer_count=$(find /workspace -type f "
            "\\( -iname '*swe*bench*verified*' -o -iname '*gold*patch*' "
            "-o -iname '*test*patch*' \\) 2>/dev/null | wc -l); "
            "archive_count=$(find /workspace -type f "
            "\\( -name '*.tar' -o -name '*.b64' -o -name '*.tar.gz' \\) | wc -l); "
            "cache_count=$(find /var/cache/sandy -mindepth 1 -type f 2>/dev/null | wc -l); "
            "echo ANSWERS=$answer_count ARCHIVES=$archive_count CACHE_FILES=$cache_count; "
            + hash_scan,
            timeout_ms=300_000,
        )
        stdout = ((result or {}).get("stdout") or "").strip()
        matches = [
            line.strip()
            for line in stdout.splitlines()
            if line.strip().split(maxsplit=1)[0] in expected_hashes
        ]
        return {
            "clean": (
                (result or {}).get("exit_code") == 0
                and "ANSWERS=0" in stdout
                and "ARCHIVES=0" in stdout
                and not matches
            ),
            "stdout": stdout,
            "heldout_hash_matches": matches[:50],
            "result": result,
            "observed_from": "agent_sandbox_namespace",
        }

    async def _verify_container_clean(
        self, container_name: str, heldout_hashes: list[dict[str, Any]]
    ) -> dict[str, Any]:
        expected_hashes = {entry["sha256"] for entry in heldout_hashes}
        sizes = sorted({int(entry["size"]) for entry in heldout_hashes})
        size_expression = " -o ".join(f"-size {size}c" for size in sizes) or "-false"
        script = (
            "find / -xdev -type f \\( "
            + size_expression
            + " \\) -exec sha256sum {} + 2>/dev/null || true"
        )
        result = await self._docker_exec_outside(container_name, ["sh", "-c", script])
        matches = [
            line.strip()
            for line in ((result or {}).get("stdout") or "").splitlines()
            if line.strip() and line.strip().split(maxsplit=1)[0] in expected_hashes
        ]
        return {
            "clean": (result or {}).get("exit_code") == 0 and not matches,
            "heldout_hash_matches": matches[:50],
            "probe_exit_code": (result or {}).get("exit_code"),
            "probe_error": (result or {}).get("error") or (result or {}).get("stderr"),
            "observed_from": "worker_docker_api_inside_task_container",
        }

    def _seal_script(self) -> str:
        entries = "\n".join(
            [
                *(f"127.0.0.1 {host}" for host in self.BENCHMARK_SOURCE_HOSTS),
                *(f"::1 {host}" for host in self.BENCHMARK_SOURCE_HOSTS),
            ]
        )
        return (
            "printf '%b\\n' "
            + shlex.quote(f"\n# {self.SEAL_MARKER}\n{entries}\n")
            + " >> /etc/hosts"
        )

    async def _seal_network(self, sandbox_id: str, container_name: str) -> dict[str, Any]:
        await self.sandy.execute_command(sandbox_id, self._seal_script())
        await self._docker_exec_outside(
            container_name, ["sh", "-c", self._seal_script()]
        )
        sandbox_probe = await self.sandy.execute_command(
            sandbox_id,
            f"echo HOSTS=$(grep -c {shlex.quote(self.SEAL_MARKER)} /etc/hosts); "
            "if command -v curl >/dev/null 2>&1; then "
            "echo CURL=yes; echo FETCH=$(curl -s -m 8 -o /dev/null -w '%{http_code}' "
            f"{shlex.quote(SWE_BENCH_VERIFIED_SOURCE_PROBE_URL)} || echo CURLFAIL); "
            "else echo CURL=no; fi",
        )

        def _network_mode() -> str:
            container = docker.from_env().containers.get(container_name)
            labels = (container.attrs.get("Config") or {}).get("Labels") or {}
            if labels.get("chutes.bench.sandbox_id") != sandbox_id:
                raise RuntimeError("task container ownership changed")
            return str((container.attrs.get("HostConfig") or {}).get("NetworkMode") or "")

        try:
            network_mode = await asyncio.to_thread(_network_mode)
        except Exception as exc:
            network_mode = f"ERROR:{exc}"
        stdout = ((sandbox_probe or {}).get("stdout") or "").strip()
        sandbox_blocked = (
            (sandbox_probe or {}).get("exit_code") == 0
            and "HOSTS=0" not in stdout
            and "HOSTS=" in stdout
            and "CURL=yes" in stdout
            and "FETCH=000CURLFAIL" in stdout
        )
        return {
            "sealed": sandbox_blocked and network_mode == "none",
            "sandbox_blocked": sandbox_blocked,
            "container_blocked": network_mode == "none",
            "sandbox_stdout": stdout,
            "container_network_mode": network_mode,
            "hosts": list(self.BENCHMARK_SOURCE_HOSTS),
        }

    async def _verify_agent_docker_boundary(
        self, sandbox_id: str, container_name: str
    ) -> dict[str, Any]:
        command = (
            "echo SOCKET=$(test -S /var/run/docker.sock && echo PRESENT || echo ABSENT); "
            "echo CACHE_MOUNT=$(mountpoint -q /var/cache/sandy && echo PRESENT || echo ABSENT); "
            "if python3 -c \"import socket; s=socket.socket(socket.AF_UNIX); "
            "s.settimeout(2); s.connect('/var/run/docker.sock')\" >/tmp/raw.out 2>/tmp/raw.err; "
            "then echo RAW_DOCKER=ESCAPED; else echo RAW_DOCKER=BLOCKED; fi; "
            "if docker run --rm curlimages/curl:8.10.1 -fsSL "
            f"{shlex.quote(SWE_BENCH_VERIFIED_SOURCE_PROBE_URL)} >/tmp/run.out 2>/tmp/run.err; "
            "then echo SPAWN=ESCAPED; else echo SPAWN=BLOCKED; fi; "
            "if docker exec chutes-bench-runner-worker-1 true >/tmp/other.out 2>/tmp/other.err; "
            "then echo OTHER_CONTAINER=ESCAPED; else echo OTHER_CONTAINER=BLOCKED; fi; "
            f"if docker exec {shlex.quote(container_name)} true >/tmp/task.out 2>/tmp/task.err; "
            "then echo TASK_PATH=WORKS; else echo TASK_PATH=BROKEN; fi"
        )
        sandbox = sandbox_container(sandbox_id)
        result = await asyncio.to_thread(
            sandbox.exec_run,
            ["sh", "-lc", command],
            stdout=True,
            stderr=True,
        )
        stdout = (result.output or b"").decode("utf-8", errors="replace")
        required = (
            "SOCKET=ABSENT",
            "CACHE_MOUNT=ABSENT",
            "RAW_DOCKER=BLOCKED",
            "SPAWN=BLOCKED",
            "OTHER_CONTAINER=BLOCKED",
            "TASK_PATH=WORKS",
        )
        return {
            "probe": stdout.strip(),
            "probe_exit_code": int(result.exit_code),
            "required_markers": list(required),
            "boundary_held": int(result.exit_code) == 0
            and all(marker in stdout for marker in required),
            "source_probe_url": SWE_BENCH_VERIFIED_SOURCE_PROBE_URL,
            "observed_from": "worker_docker_api_into_agent_namespace",
        }

    async def _sandbox_alive(self, sandbox_id: str) -> bool | None:
        return await self.sandy.sandbox_exists(sandbox_id)

    async def _verify_agent_terminated(
        self,
        sandbox_id: str,
        agent_summary: dict[str, Any],
        *,
        wait_timeout_seconds: float = 0,
        poll_interval_seconds: float = 2,
    ) -> dict[str, Any]:
        completion_event = agent_summary.get("type") == "complete"
        deadline = time.monotonic() + max(0.0, wait_timeout_seconds)
        attempts = 0
        initial_probe = ""
        stdout = ""
        done_value: int | None = None
        process_stopped = False
        done_recorded = False
        while True:
            attempts += 1
            probe = await self.sandy.execute_command(
                sandbox_id,
                "pid=$(cat /workspace/.chutes/agent.pid 2>/dev/null || true); "
                "if [ -f /workspace/.chutes/agent.done ]; then done_present=yes; "
                "done_value=$(cat /workspace/.chutes/agent.done 2>/dev/null || true); "
                "else done_present=no; done_value=; fi; "
                "if [ -z \"$pid\" ]; then running=no; state=missing; "
                "elif [ -r \"/proc/$pid/stat\" ]; then state=$(cut -d' ' -f3 \"/proc/$pid/stat\"); "
                "if [ \"$state\" = Z ]; then running=no; else running=yes; fi; "
                "else running=no; state=gone; fi; "
                "echo PID=${pid:-missing} STATE=$state RUNNING=$running "
                "DONE_PRESENT=$done_present DONE=${done_value:-missing}",
                timeout_ms=10_000,
            )
            stdout = ((probe or {}).get("stdout") or "").strip()
            if not initial_probe:
                initial_probe = stdout
            process_stopped = (
                (probe or {}).get("exit_code") == 0
                and "RUNNING=no" in stdout
                and ("STATE=gone" in stdout or "STATE=Z" in stdout)
            )
            done_recorded = "DONE_PRESENT=yes" in stdout
            for token in stdout.split():
                if token.startswith("DONE="):
                    with contextlib.suppress(ValueError):
                        done_value = int(token.split("=", 1)[1])
            if completion_event and process_stopped and done_recorded and done_value is not None:
                break
            if time.monotonic() >= deadline:
                break
            await asyncio.sleep(max(0.0, poll_interval_seconds))
        return {
            "terminated": completion_event
            and process_stopped
            and done_recorded
            and done_value is not None,
            "completion_event": completion_event,
            "process_stopped": process_stopped,
            "done_recorded": done_recorded,
            "done_value": done_value,
            "attempts": attempts,
            "initial_probe": initial_probe,
            "probe": stdout,
        }

    async def _collect_patch(
        self, container_name: str, base_commit: str, timeout_sec: int
    ) -> dict[str, Any]:
        command = (
            "set -e; cd /testbed; git add -A; "
            f"git diff --cached --binary {shlex.quote(base_commit)} | base64 -w0; git reset -q"
        )
        try:
            result = await asyncio.wait_for(
                self._docker_exec_outside(
                    container_name, ["bash", "-lc", command], workdir="/testbed"
                ),
                timeout=max(1, timeout_sec),
            )
        except TimeoutError:
            return {"ok": False, "transport_failed": True, "error": "patch collection timed out"}
        if result.get("exit_code") == -1:
            return {"ok": False, "transport_failed": True, "result": result}
        if result.get("exit_code") != 0:
            return {"ok": False, "transport_failed": False, "result": result, "patch": b""}
        try:
            patch = base64.b64decode(result.get("stdout") or "", validate=True)
        except ValueError:
            return {
                "ok": False,
                "transport_failed": True,
                "error": "collected patch was not valid base64",
            }
        return {"ok": True, "transport_failed": False, "patch": patch, "size_bytes": len(patch)}

    @staticmethod
    def _grade_official_log(item: dict[str, Any], patch: bytes, output: str) -> dict[str, Any]:
        from swebench.harness.constants import KEY_INSTANCE_ID, KEY_MODEL, KEY_PREDICTION
        from swebench.harness.grading import get_eval_report

        with tempfile.NamedTemporaryFile("w", encoding="utf-8") as log_file:
            log_file.write(output)
            log_file.flush()
            report = get_eval_report(
                test_spec=item["_test_spec"],
                prediction={
                    KEY_INSTANCE_ID: item["instance_id"],
                    KEY_MODEL: "chutes-bench-runner",
                    KEY_PREDICTION: patch.decode("utf-8", errors="replace"),
                },
                test_log_path=log_file.name,
                include_tests_status=True,
            )
        return report.get(item["instance_id"], {})

    async def _stage_and_run_verifier(
        self,
        *,
        sandbox_id: str,
        item: dict[str, Any],
        patch: bytes,
        verifier_container: str,
        image_id: str,
        verifier_timeout_sec: int,
    ) -> dict[str, Any]:
        started = await self._start_container(
            sandbox_id=sandbox_id,
            item=item,
            container_name=verifier_container,
            role="heldout-verifier",
            image_id=image_id,
            pull_timeout_sec=120,
        )
        if not started.get("ok"):
            return {"ok": False, "stage": "verifier_start", "start": started}
        sanitized = await self._sanitize_repository(verifier_container, item["base_commit"])
        if not sanitized.get("clean"):
            return {
                "ok": False,
                "stage": "verifier_integrity",
                "repository": sanitized,
            }
        patch_transfer = await self._put_file_outside(
            verifier_container, "/tmp/patch.diff", patch
        )
        # The eval script contains the held-out test patch. It crosses directly
        # from trusted worker memory into this separate no-network verifier;
        # it is never staged in Sandy or the stopped agent-controlled container.
        eval_transfer = await self._put_file_outside(
            verifier_container, "/eval.sh", item["_eval_script"].encode("utf-8"), mode=0o700
        )
        if patch_transfer.get("exit_code") != 0 or eval_transfer.get("exit_code") != 0:
            return {
                "ok": False,
                "stage": "verifier_transfer",
                "patch_transfer": patch_transfer,
                "eval_transfer": eval_transfer,
            }

        apply_attempts = []
        patch_applied = False
        for command in (
            "git apply --verbose /tmp/patch.diff",
            "git apply --verbose --reject /tmp/patch.diff",
            "patch --batch --fuzz=5 -p1 -i /tmp/patch.diff",
        ):
            attempt = await self._docker_exec_outside(
                verifier_container, ["bash", "-lc", command], workdir="/testbed"
            )
            apply_attempts.append(attempt)
            if attempt.get("exit_code") == 0:
                patch_applied = True
                break
        if not patch_applied:
            return {
                "ok": True,
                "patch_applied": False,
                "patch_error": next(
                    (
                        attempt.get("stderr") or attempt.get("stdout")
                        for attempt in reversed(apply_attempts)
                        if attempt.get("stderr") or attempt.get("stdout")
                    ),
                    "Agent patch did not apply",
                ),
                "apply_attempts": apply_attempts,
                "test_command_executed": False,
            }

        marker = f"{sandbox_id}:{item['id']}:{time.time_ns()}"
        marker_path = "/tmp/.chutes-swebench-verifier-started"
        command = (
            f"printf '%s' {shlex.quote(marker)} > {marker_path}; "
            f"timeout --signal=TERM --kill-after=30 {int(verifier_timeout_sec)} "
            "/bin/bash /eval.sh > /tmp/eval.log 2>&1"
        )
        test = await self._docker_exec_outside(
            verifier_container, ["bash", "-lc", command], workdir="/testbed"
        )
        log = await self._docker_exec_outside(
            verifier_container, ["cat", "/tmp/eval.log"], workdir="/testbed"
        )
        marker_probe = await self._docker_exec_outside(
            verifier_container, ["sh", "-c", f"cat {marker_path} 2>/dev/null || true"]
        )
        test_command_executed = ((marker_probe.get("stdout") or "").strip() == marker)
        test_output = log.get("stdout") or ""
        timed_out = test.get("exit_code") in (124, 137)
        report: dict[str, Any] = {}
        grading_error: str | None = None
        if test_command_executed and not timed_out:
            try:
                report = await asyncio.to_thread(self._grade_official_log, item, patch, test_output)
            except Exception as exc:
                grading_error = str(exc) or exc.__class__.__name__
        return {
            "ok": True,
            "patch_applied": True,
            "apply_attempts": apply_attempts,
            "test": test,
            "test_log_probe": log,
            "test_output": test_output,
            "test_command_probe": marker_probe,
            "test_command_executed": test_command_executed,
            "timed_out": timed_out,
            "report": report,
            "error": grading_error,
            "separate_verifier": True,
            "staged_in_agent_sandbox": False,
            "repository": sanitized,
            "image_id": started.get("image_id"),
        }

    def _agent_name(self) -> str:
        return (
            (getattr(self, "run_config", None) or {})
            .get("swe_bench_verified", {})
            .get("agent")
            or os.getenv("SWE_BENCH_VERIFIED_AGENT")
            or "codex"
        )

    def _new_item_observability(self, item_id: str) -> dict[str, Any]:
        state = {
            "agent_invoked": False,
            "agent_launch": None,
            "rollout_retained": False,
            "rollout_retention_error": None,
            "retention_task": None,
            "evidence": {
                "status": "not_available",
                "path": None,
                "sha256": None,
                "size_bytes": None,
                "error": "agent was not started for this item",
                "token_usage_samples": None,
            },
        }
        self._item_observability[item_id] = state
        return state

    def _start_evidence_retention(self, item_id: str, sandbox_id: str) -> None:
        state = self._item_observability[item_id]
        if state.get("retention_task") is not None:
            return
        state["evidence"] = {
            "status": "pending",
            "path": None,
            "sha256": None,
            "size_bytes": None,
            "error": None,
            "token_usage_samples": None,
        }
        state["retention_task"] = asyncio.create_task(
            retain_agent_evidence(
                self.sandy,
                sandbox_id,
                run_id=getattr(self, "run_id", None),
                benchmark_name=self.get_name(),
                item_id=item_id,
                require_rollout=True,
            )
        )

    async def _finish_evidence_retention(self, item_id: str, sandbox_id: str) -> None:
        state = self._item_observability.get(item_id)
        if not state or not state.get("agent_invoked"):
            return
        if state.get("retention_task") is None:
            try:
                await self.sandy.execute_command(
                    sandbox_id,
                    "if [ -f /workspace/.chutes/agent.pid ]; then "
                    "kill -TERM $(cat /workspace/.chutes/agent.pid) 2>/dev/null || true; "
                    "sleep 1; kill -KILL $(cat /workspace/.chutes/agent.pid) 2>/dev/null || true; fi; "
                    "test -f /workspace/.chutes/agent.done || echo 143 > /workspace/.chutes/agent.done",
                    timeout_ms=10_000,
                )
            except Exception as exc:
                logger.warning(
                    "Could not stop SWE-bench Verified agent before evidence retention",
                    item_id=item_id,
                    sandbox_id=sandbox_id,
                    error=str(exc),
                )
            launch = state.get("agent_launch")
            if launch is not None and not state.get("rollout_retained"):
                try:
                    await retain_sandy_agent_rollout(self.sandy, sandbox_id, launch)
                    state["rollout_retained"] = True
                except Exception as exc:
                    state["rollout_retention_error"] = str(exc) or exc.__class__.__name__
            self._start_evidence_retention(item_id, sandbox_id)
        try:
            state["evidence"] = await state["retention_task"]
            state["evidence"]["rollout_retention_error"] = state.get(
                "rollout_retention_error"
            )
        except BaseException as exc:
            state["evidence"] = {
                "status": "failed",
                "path": None,
                "sha256": None,
                "size_bytes": None,
                "error": f"retention task failed: {exc}",
                "token_usage_samples": None,
                "rollout_retention_error": state.get("rollout_retention_error"),
            }

    def attach_item_observability(self, result: ItemResult) -> ItemResult:
        state = self._item_observability.pop(result.item_id, None)
        if not state:
            return result
        evidence = state["evidence"]
        result.agent_evidence_status = evidence.get("status")
        result.agent_evidence_path = evidence.get("path")
        result.agent_evidence_sha256 = evidence.get("sha256")
        result.agent_evidence_size_bytes = evidence.get("size_bytes")
        result.agent_evidence_error = evidence.get("error")
        result.token_usage_samples = evidence.get("token_usage_samples")
        if result.metadata is None:
            result.metadata = {}
        result.metadata["agent_evidence"] = {
            key: evidence.get(key)
            for key in (
                "status",
                "path",
                "sha256",
                "size_bytes",
                "error",
                "sandbox_sources",
                "retention_policy",
                "rollout_retention_error",
            )
        }
        if state.get("agent_invoked") and evidence.get("status") != "retained":
            result.metadata.setdefault(
                "exclusion_reason", SWE_BENCH_VERIFIED_EVIDENCE_EXCLUSION_REASON
            )
            if not result.error:
                result.error = (
                    "Agent evidence was not retained: "
                    f"{evidence.get('error') or evidence.get('status')}"
                )
        return result

    def _excluded_result(
        self,
        *,
        item: dict[str, Any],
        prompt: str,
        reason: str,
        error: str,
        start_time: float,
        metadata: dict[str, Any] | None = None,
        usage: dict[str, Any] | None = None,
    ) -> ItemResult:
        return ItemResult(
            item_id=item["id"],
            item_hash=self.compute_item_hash(item.get("instance_id")),
            prompt=prompt,
            error=error,
            latency_ms=int((time.time() - start_time) * 1000),
            input_tokens=(usage or {}).get("input_tokens"),
            output_tokens=(usage or {}).get("output_tokens"),
            metadata={
                "instance_id": item.get("instance_id"),
                "repo": item.get("repo"),
                "harness": SWE_BENCH_VERIFIED_HARNESS,
                "exclusion_reason": reason,
                **(metadata or {}),
            },
        )

    async def evaluate_item(self, item_id: str) -> ItemResult:
        self._new_item_observability(item_id)
        result = await self._evaluate_item(item_id)
        return self.attach_item_observability(result)

    async def _evaluate_item(self, item_id: str) -> ItemResult:
        if not self._items:
            await self.preload()
        item = next((candidate for candidate in self._items if candidate["id"] == item_id), None)
        if item is None:
            return ItemResult(item_id=item_id, error=f"Item {item_id} not found")

        agent_name = self._agent_name()
        prompt = (
            "You are an interactive coding agent in a Sandy sandbox. A no-network "
            "SWE-bench Verified task container is already running. Use docker exec "
            "to inspect and modify the repository at /testbed inside that container. "
            "Do not search for benchmark sources, hidden tests, or reference solutions. "
            "You may run the repository's existing tests. Leave every intended change "
            "in the working tree; the harness will collect a binary git diff and send "
            "it to a separate pristine verifier container after you stop.\n\n"
            f"Repository: {item['repo']}\n"
            f"Base commit: {item['base_commit']}\n"
            f"Issue description:\n{item['problem_statement']}\n"
        )
        start_time = time.time()
        budget = self._item_budgets_seconds()
        outer_timeout = self.get_item_timeout_seconds(item_id) or sum(budget.values())
        sandbox_ttl_min = max(10, math.ceil(outer_timeout / 60) + 5)
        sandbox_id: str | None = None
        agent_container: str | None = None
        verifier_container: str | None = None
        image_id: str | None = None
        agent_summary: dict[str, Any] = {}
        agent_usage: dict[str, Any] = {}
        try:
            sandbox_id = await self.sandy.create_sandbox(
                enable_docker_socket=False,
                enable_shared_cache=False,
                requires_agent=True,
                timeout_minutes=sandbox_ttl_min,
            )
            if not sandbox_id:
                return self._excluded_result(
                    item=item,
                    prompt=prompt,
                    reason="infrastructure_sandbox_create",
                    error=self.sandy.last_error or "Could not create Sandy sandbox",
                    start_time=start_time,
                    metadata={"agent": agent_name},
                )

            workspace_clean = await self._verify_workspace_clean(
                sandbox_id, item["heldout_hashes"]
            )
            if not workspace_clean.get("clean"):
                return self._excluded_result(
                    item=item,
                    prompt=prompt,
                    reason="integrity_answer_key_unproven",
                    error=(
                        "Refusing to start the agent: answer-key absence in Sandy "
                        f"was not proven ({workspace_clean})"
                    ),
                    start_time=start_time,
                    metadata={"agent": agent_name, "workspace_clean": workspace_clean},
                )

            namespace = f"s{sandbox_id[:12]}".lower()
            safe_id = "".join(
                character if character.isalnum() else "-"
                for character in item["instance_id"].lower()
            )[:90]
            agent_container = f"sweb_{namespace}_{safe_id}_agent"
            verifier_container = f"sweb_{namespace}_{safe_id}_verifier"
            setup = await self._start_container(
                sandbox_id=sandbox_id,
                item=item,
                container_name=agent_container,
                role="agent-task",
                pull_timeout_sec=budget["image_pull"],
            )
            if not setup.get("ok"):
                return self._excluded_result(
                    item=item,
                    prompt=prompt,
                    reason=f"infrastructure_{setup.get('stage') or 'setup'}",
                    error=f"SWE-bench task environment setup failed: {setup}",
                    start_time=start_time,
                    metadata={"agent": agent_name, "setup": setup},
                )
            image_id = setup["image_id"]
            repository_clean = await self._sanitize_repository(
                agent_container, item["base_commit"]
            )
            if not repository_clean.get("clean"):
                return self._excluded_result(
                    item=item,
                    prompt=prompt,
                    reason="integrity_repository_history_unproven",
                    error=f"Could not confine repository history: {repository_clean}",
                    start_time=start_time,
                    metadata={"agent": agent_name, "repository_clean": repository_clean},
                )
            container_clean = await self._verify_container_clean(
                agent_container, item["heldout_hashes"]
            )
            if not container_clean.get("clean"):
                return self._excluded_result(
                    item=item,
                    prompt=prompt,
                    reason="integrity_answer_key_unproven",
                    error=(
                        "Refusing to start the agent: held-out SWE-bench material "
                        f"may be reachable inside its task container ({container_clean})"
                    ),
                    start_time=start_time,
                    metadata={"agent": agent_name, "container_clean": container_clean},
                )
            try:
                gateway = await TerminalBenchAdapter._start_task_gateway(
                    self, sandbox_id, agent_container
                )
            except Exception as exc:
                return self._excluded_result(
                    item=item,
                    prompt=prompt,
                    reason="integrity_docker_boundary_unproven",
                    error=f"Task-scoped Docker gateway failed: {exc}",
                    start_time=start_time,
                    metadata={"agent": agent_name},
                )
            docker_boundary = await self._verify_agent_docker_boundary(
                sandbox_id, agent_container
            )
            if not docker_boundary.get("boundary_held"):
                return self._excluded_result(
                    item=item,
                    prompt=prompt,
                    reason="integrity_docker_boundary_unproven",
                    error=f"Fresh-container source bypass was not blocked: {docker_boundary}",
                    start_time=start_time,
                    metadata={"agent": agent_name, "agent_docker_gateway": gateway},
                )
            agent_view_clean = await self._verify_workspace_clean(
                sandbox_id, item["heldout_hashes"]
            )
            if not agent_view_clean.get("clean"):
                return self._excluded_result(
                    item=item,
                    prompt=prompt,
                    reason="integrity_answer_key_unproven",
                    error=f"Held-out material is reachable from the final agent view: {agent_view_clean}",
                    start_time=start_time,
                    metadata={"agent": agent_name, "agent_view_clean": agent_view_clean},
                )
            seal = await self._seal_network(sandbox_id, agent_container)
            if not seal.get("sealed"):
                return self._excluded_result(
                    item=item,
                    prompt=prompt,
                    reason="integrity_network_seal_unproven",
                    error=f"SWE-bench source network seal failed: {seal}",
                    start_time=start_time,
                    metadata={"agent": agent_name, "seal": seal},
                )

            launch = await prepare_sandy_agent_launch(
                client=self.client,
                sandy=self.sandy,
                sandbox_id=sandbox_id,
                agent=agent_name,
                model=self.model_slug,
            )
            state = self._item_observability[item_id]
            state["agent_launch"] = launch
            state["agent_invoked"] = True
            agent_started_at = time.monotonic()
            agent_result = await self.sandy.run_agent(
                sandbox_id,
                agent=agent_name,
                model=self.model_slug,
                prompt=prompt + f"\nTask container name: {agent_container}\n",
                max_duration=budget["agent"],
                raw_prompt=True,
                api_base_url=launch.api_base_url,
                env_vars=launch.env_vars,
            )
            agent_call_seconds = time.monotonic() - agent_started_at
            agent_summary = (agent_result or {}).get("summary") or {}
            sandbox_alive = await self._sandbox_alive(sandbox_id)
            exclusion_reason, exit_note = classify_swe_bench_verified_agent_outcome(
                agent_summary, float(budget["agent"]), sandbox_alive
            )
            if exclusion_reason:
                return self._excluded_result(
                    item=item,
                    prompt=prompt,
                    reason=exclusion_reason,
                    error=exit_note or exclusion_reason,
                    start_time=start_time,
                    metadata={
                        "agent": agent_name,
                        "agent_summary": agent_summary,
                        "sandbox_alive_at_exit": sandbox_alive,
                    },
                )
            termination = await self._verify_agent_terminated(
                sandbox_id,
                agent_summary,
                wait_timeout_seconds=max(0.0, budget["agent"] - agent_call_seconds),
            )
            if not termination.get("terminated"):
                return self._excluded_result(
                    item=item,
                    prompt=prompt,
                    reason="infrastructure_agent_not_terminated",
                    error="Agent termination was not proven before verifier staging.",
                    start_time=start_time,
                    metadata={"agent": agent_name, "agent_termination": termination},
                )

            usage_error: str | None = None
            try:
                await retain_sandy_agent_rollout(self.sandy, sandbox_id, launch)
                state["rollout_retained"] = True
            except Exception as exc:
                usage_error = str(exc) or exc.__class__.__name__
                state["rollout_retention_error"] = usage_error
            agent_usage = await collect_agent_usage(self.sandy, sandbox_id)
            if usage_error is None:
                try:
                    validate_openrouter_agent_usage(launch, agent_usage)
                except RuntimeError as exc:
                    usage_error = str(exc)
            self._start_evidence_retention(item_id, sandbox_id)
            if usage_error:
                return self._excluded_result(
                    item=item,
                    prompt=prompt,
                    reason="infrastructure_usage_accounting",
                    error=usage_error,
                    start_time=start_time,
                    usage=agent_usage,
                    metadata={"agent": agent_name, "agent_summary": agent_summary},
                )

            collected = await self._collect_patch(
                agent_container, item["base_commit"], budget["collect"]
            )
            if collected.get("transport_failed"):
                return self._excluded_result(
                    item=item,
                    prompt=prompt,
                    reason="infrastructure_transport",
                    error=f"Could not collect the agent patch: {collected}",
                    start_time=start_time,
                    usage=agent_usage,
                    metadata={"agent": agent_name, "collected": collected},
                )
            patch = collected.get("patch") or b""
            events = (agent_result or {}).get("events") or []
            agent_output = next(
                (event.get("text") for event in reversed(events) if event.get("type") == "output"),
                "",
            )
            if not patch.strip():
                return ItemResult(
                    item_id=item_id,
                    item_hash=self.compute_item_hash(item["instance_id"]),
                    prompt=prompt,
                    response=agent_output,
                    expected="[Official SWE-bench Verified resolved = true]",
                    is_correct=False,
                    score=0.0,
                    latency_ms=int((time.time() - start_time) * 1000),
                    input_tokens=agent_usage.get("input_tokens"),
                    output_tokens=agent_usage.get("output_tokens"),
                    error="Agent produced no patch",
                    metadata={
                        "instance_id": item["instance_id"],
                        "repo": item["repo"],
                        "agent": agent_name,
                        "harness": SWE_BENCH_VERIFIED_HARNESS,
                        "agent_summary": agent_summary,
                        "agent_usage": agent_usage,
                        "agent_exit_note": exit_note,
                        "dataset_repository": item["dataset_repository"],
                        "dataset_commit": item["dataset_commit"],
                    },
                )

            def _stop_agent_container() -> None:
                container = docker.from_env().containers.get(agent_container)
                labels = (container.attrs.get("Config") or {}).get("Labels") or {}
                if labels.get("chutes.bench.sandbox_id") != sandbox_id:
                    raise RuntimeError("agent task container ownership changed")
                container.stop(timeout=30)

            try:
                await asyncio.wait_for(asyncio.to_thread(_stop_agent_container), timeout=120)
            except Exception as exc:
                return self._excluded_result(
                    item=item,
                    prompt=prompt,
                    reason="integrity_agent_environment_not_stopped",
                    error=f"Refusing to stage held-out verifier: {exc}",
                    start_time=start_time,
                    usage=agent_usage,
                    metadata={"agent": agent_name},
                )

            verifier = await self._stage_and_run_verifier(
                sandbox_id=sandbox_id,
                item=item,
                patch=patch,
                verifier_container=verifier_container,
                image_id=image_id,
                verifier_timeout_sec=budget["verifier"],
            )
            if not verifier.get("ok"):
                stage = verifier.get("stage") or "verifier"
                reason = (
                    "integrity_verifier_separation_unproven"
                    if stage == "verifier_integrity"
                    else f"infrastructure_{stage}"
                )
                return self._excluded_result(
                    item=item,
                    prompt=prompt,
                    reason=reason,
                    error=f"Separate SWE-bench verifier failed at {stage}: {verifier}",
                    start_time=start_time,
                    usage=agent_usage,
                    metadata={"agent": agent_name, "verifier": verifier},
                )
            reward, verifier_exclusion, verifier_error = (
                classify_swe_bench_verified_verifier_outcome(verifier)
            )
            if verifier_exclusion:
                return self._excluded_result(
                    item=item,
                    prompt=prompt,
                    reason=verifier_exclusion,
                    error=verifier_error or verifier_exclusion,
                    start_time=start_time,
                    usage=agent_usage,
                    metadata={"agent": agent_name, "verifier": verifier},
                )
            assert reward is not None
            return ItemResult(
                item_id=item_id,
                item_hash=self.compute_item_hash(item["instance_id"]),
                prompt=prompt,
                response=agent_output,
                expected="[Official SWE-bench Verified resolved = true]",
                is_correct=reward == 1.0,
                score=reward,
                latency_ms=int((time.time() - start_time) * 1000),
                input_tokens=agent_usage.get("input_tokens"),
                output_tokens=agent_usage.get("output_tokens"),
                judge_output={
                    "report": verifier.get("report"),
                    "test_exit_code": (verifier.get("test") or {}).get("exit_code"),
                    "test_output_preview": (verifier.get("test_output") or "")[-20_000:],
                },
                error=verifier_error,
                metadata={
                    "instance_id": item["instance_id"],
                    "repo": item["repo"],
                    "difficulty": item.get("difficulty"),
                    "agent": agent_name,
                    "agent_provider": launch.metadata,
                    "harness": SWE_BENCH_VERIFIED_HARNESS,
                    "official_harness_repository": self.benchmark_spec.harness_repository,
                    "official_harness_commit": self.benchmark_spec.harness_commit,
                    "official_harness_version": self.benchmark_spec.harness_version,
                    "agent_summary": agent_summary,
                    "agent_usage": agent_usage,
                    "agent_exit_note": exit_note,
                    "agent_timeout_sec": budget["agent"],
                    "verifier_timeout_sec": budget["verifier"],
                    "worker_item_timeout_sec": outer_timeout,
                    "sandbox_ttl_min": sandbox_ttl_min,
                    "seal": seal,
                    "agent_docker_gateway": gateway,
                    "docker_boundary": docker_boundary,
                    "answer_key_holdout": {
                        "workspace": workspace_clean,
                        "agent_view": agent_view_clean,
                        "container": container_clean,
                        "gold_patch_uploaded_to_agent": False,
                        "test_patch_uploaded_to_agent": False,
                        "eval_script_uploaded_to_agent": False,
                    },
                    "separate_verifier": True,
                    "dataset_repository": item["dataset_repository"],
                    "dataset_commit": item["dataset_commit"],
                    "dataset_file_sha256": item["dataset_file_sha256"],
                    "docker_image": item["docker_image"],
                    "docker_image_digest_observed": setup.get("image_digest"),
                    "docker_image_tag_is_immutable": False,
                    "base_commit": item["base_commit"],
                    "verifier_report": verifier.get("report"),
                },
            )
        except Exception as exc:
            logger.error("SWE-bench Verified evaluation failed", item_id=item_id, error=str(exc))
            exclusion_reason = (
                classify_bare_failure(str(exc), agent_summary) or "infrastructure_adapter"
            )
            return self._excluded_result(
                item=item,
                prompt=prompt,
                reason=exclusion_reason,
                error=str(exc),
                start_time=start_time,
                usage=agent_usage,
                metadata={"agent": agent_name, "agent_summary": agent_summary},
            )
        finally:
            if sandbox_id:
                try:
                    await self._finish_evidence_retention(item_id, sandbox_id)
                finally:
                    await TerminalBenchAdapter._cleanup_owned_task_containers(self, sandbox_id)

                    def _remove_item_image() -> None:
                        if not image_id:
                            return
                        with contextlib.suppress(Exception):
                            docker.from_env().images.remove(image_id, force=False)

                    await asyncio.to_thread(_remove_item_image)
                    await self.sandy.terminate_sandbox(sandbox_id)
