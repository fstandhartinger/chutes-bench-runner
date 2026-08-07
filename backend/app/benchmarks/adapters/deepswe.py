"""Pinned DeepSWE v1.1 driven by Sandy CLI agents.

The official leaderboard uses Pier + mini-SWE-agent on Modal. This adapter
preserves DeepSWE's task and separate-verifier protocol while deliberately
swapping the agent scaffold, so its scores must be labelled "Sandy CLI" and
not presented as official-leaderboard-equivalent numbers.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import copy
import hashlib
import io
import json
import math
import os
import shlex
import tarfile
import tempfile
import time
import tomllib
from collections.abc import AsyncIterator
from pathlib import Path, PurePosixPath
from typing import Any

import docker
import httpx

from app.benchmarks.adapters.deepswe_identity import DEEPSWE_V1_1, DeepSWESpec
from app.benchmarks.adapters.terminal_bench import (
    BenchmarkIdentityError,
    TerminalBenchAdapter,
    classify_agent_exit,
    classify_bare_failure,
)
from app.benchmarks.agent_evidence import retain_agent_evidence
from app.benchmarks.agent_provider_config import (
    AgentProviderLaunch,
    prepare_sandy_agent_launch,
    retain_sandy_agent_rollout,
    validate_openrouter_agent_usage,
)
from app.benchmarks.agent_usage import collect_agent_usage
from app.benchmarks.base import BenchmarkAdapter, ItemResult
from app.benchmarks.registry import register_adapter
from app.core.logging import get_logger
from app.services.provenance_service import sandbox_container
from app.services.sandy_service import SandyService

logger = get_logger(__name__)

DEEPSWE_ITEM_TIMEOUT_MARGIN_SECONDS = 15 * 60
DEEPSWE_HARNESS = "sandy-cli-separate-verifier"
DEEPSWE_AGENT_NOT_TERMINATED_EXCLUSION_REASON = "infrastructure_agent_not_terminated"
DEEPSWE_USAGE_ACCOUNTING_EXCLUSION_REASON = "infrastructure_usage_accounting"
DEEPSWE_VERIFIER_NOT_EXECUTED_EXCLUSION_REASON = "infrastructure_verifier_not_executed"
DEEPSWE_SOURCE_PROBE_URL = (
    "https://raw.githubusercontent.com/datacurve-ai/deep-swe/"
    "435ee89ec2f2e2289f33b0da4f992f0b7b7266b9/README.md"
)


def classify_deepswe_agent_outcome(
    agent_summary: dict | None,
    agent_timeout_sec: float,
    sandbox_alive: bool | None,
) -> tuple[str | None, str | None]:
    """Classify agent termination without hiding a live-sandbox CLI crash.

    Sandy always emits a completion summary for a completed agent process. An
    empty summary means the runner lost the stream and cannot attribute an
    outcome to the harness, so it is an infrastructure exclusion. Non-zero
    exits retain Terminal-Bench's stricter distinction: only a dead sandbox is
    excluded; a CLI crash in a live sandbox is scored.
    """
    if not agent_summary:
        return (
            "infrastructure_transport",
            "Agent stream ended without a Sandy completion summary.",
        )
    return classify_agent_exit(agent_summary, agent_timeout_sec, sandbox_alive)


def classify_deepswe_exception(
    error_text: str,
    agent_summary: dict | None,
) -> str | None:
    """Classify transport exceptions that never produced an agent summary."""
    return classify_bare_failure(error_text, agent_summary)


def classify_deepswe_verifier_outcome(
    test_result: dict | None,
    reward_result: dict | None,
    *,
    test_command_executed: bool,
) -> tuple[float | None, dict[str, Any], str | None, str | None]:
    """Accept a scored failure while excluding an unproven verifier run.

    DeepSWE's verifier normally exits zero even when tests fail and records the
    binary outcome in reward.json.  The reward is the authority, not the test
    shell's exit code: a valid zero must remain a scored zero.  Conversely, a
    reward file is accepted only after an in-container sentinel proves that the
    current verifier command started; this prevents a transport failure or a
    stale artifact from becoming a plausible score.
    """
    if not test_command_executed:
        detail = (
            (test_result or {}).get("stderr")
            or (test_result or {}).get("error")
            or "DeepSWE verifier test command did not execute"
        )
        return None, {}, DEEPSWE_VERIFIER_NOT_EXECUTED_EXCLUSION_REASON, detail

    reward, metrics = DeepSWEAdapter._parse_reward((reward_result or {}).get("stdout") or "")
    if (reward_result or {}).get("exit_code") != 0 or reward not in (0.0, 1.0):
        detail = (
            (reward_result or {}).get("stderr")
            or (reward_result or {}).get("error")
            or (test_result or {}).get("stderr")
            or (test_result or {}).get("error")
            or "DeepSWE verifier did not produce a valid binary reward"
        )
        return None, metrics, "infrastructure_verifier", detail
    return reward, metrics, None, None


@register_adapter("deepswe")
class DeepSWEAdapter(BenchmarkAdapter):
    """DeepSWE v1.1 with a selectable Sandy CLI agent scaffold."""

    benchmark_spec: DeepSWESpec = DEEPSWE_V1_1

    # DeepSWE task definitions, reference solutions, and held-out verifier
    # sources are public. The task container itself is no-network; the Sandy
    # agent process still needs the model endpoint, so its namespace blocks
    # the benchmark sources by hostname just as the Terminal-Bench adapter
    # does. This is explicitly verified before the CLI starts.
    BENCHMARK_SOURCE_HOSTS = (
        "github.com",
        "www.github.com",
        "api.github.com",
        "raw.githubusercontent.com",
        "codeload.github.com",
        "objects.githubusercontent.com",
        "gist.github.com",
        "gist.githubusercontent.com",
        "huggingface.co",
        "hf.co",
        "cdn-lfs.huggingface.co",
        "cdn-lfs-us-1.hf.co",
        "deepswe.datacurve.ai",
        "datacurve.ai",
        "www.datacurve.ai",
        "harborframework.com",
        "www.harborframework.com",
        "hub.harborframework.com",
        "raw.githack.com",
        "cdn.jsdelivr.net",
        "gitclone.com",
        "hub.fastgit.org",
        "ghproxy.com",
    )
    SEAL_MARKER = "chutes-bench-runner: DeepSWE answer sources"

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._items: list[dict[str, Any]] = []
        self._item_observability: dict[str, dict[str, Any]] = {}
        self.sandy = SandyService()

    def get_name(self) -> str:
        return "deepswe"

    def get_display_name(self) -> str:
        return self.benchmark_spec.display_name

    def requires_setup(self) -> bool:
        return True

    def get_setup_notes(self) -> str | None:
        return (
            "Requires worker-side Docker access; the Sandy agent sandbox receives "
            "neither the Docker socket nor Sandy's shared cache. Pulls one unique "
            "DeepSWE v1.1 image per item and removes task-specific images after scoring."
        )

    def supports_subset(self) -> bool:
        return True

    def supports_parallel_items(self) -> bool:
        # Each item can consume the declared 8 GiB RAM and up to 20 GiB disk.
        return False

    @staticmethod
    def _item_budgets_seconds(item: dict[str, Any]) -> dict[str, int | float]:
        multiplier = float(os.getenv("DEEPSWE_AGENT_TIMEOUT_MULTIPLIER") or "1.0")
        agent_base = float(item.get("max_agent_timeout_sec") or 5400)
        return {
            "agent_base": agent_base,
            "agent_multiplier": multiplier,
            "agent": max(60, math.ceil(agent_base * multiplier)),
            "verifier": max(1, math.ceil(float(item.get("max_test_timeout_sec") or 1800))),
            "environment_build": max(
                1, math.ceil(float(item.get("environment_build_timeout_sec") or 1800))
            ),
            "verifier_build": max(
                1, math.ceil(float(item.get("verifier_build_timeout_sec") or 1800))
            ),
            "collect": max(1, math.ceil(float(item.get("collect_timeout_sec") or 300))),
        }

    def get_item_timeout_seconds(self, item_id: str | None = None) -> int | None:
        """Cover every task-declared phase, especially the 90-minute agent."""
        items = self._items
        if item_id is not None:
            item = next((candidate for candidate in items if candidate["id"] == item_id), None)
            if item is None:
                return None
            items = [item]
        if not items:
            return None

        totals = []
        for item in items:
            budget = self._item_budgets_seconds(item)
            totals.append(
                int(budget["agent"])
                + int(budget["verifier"])
                + int(budget["environment_build"])
                + int(budget["verifier_build"])
                + int(budget["collect"])
                + DEEPSWE_ITEM_TIMEOUT_MARGIN_SECONDS
            )
        return max(totals)

    async def get_total_items(self) -> int:
        if not self._items:
            await self.preload()
        return len(self._items)

    async def enumerate_items(self) -> AsyncIterator[str]:
        if not self._items:
            await self.preload()
        for item in self._items:
            yield item["id"]

    async def preload(self) -> None:
        """Load the exact public v1.1 source revision and assert membership."""
        if self._items:
            return
        source_archive: bytes | None = None
        try:
            logger.info(
                "Loading pinned DeepSWE dataset",
                repository=self.benchmark_spec.repository,
                commit=self.benchmark_spec.commit,
                expected_count=self.benchmark_spec.expected_count,
            )
            source_archive = await self._load_source_archive()
            self._items = await asyncio.to_thread(self._items_from_source_archive, source_archive)
            self._assert_benchmark_identity()
        except Exception as exc:
            self._items = []
            logger.error("Failed to load DeepSWE", error=str(exc))
            raise
        finally:
            # The codeload tar contains reference solutions. It is useful only
            # while producing sanitized agent and tests-only verifier bundles,
            # and must not remain on the runner filesystem afterward.
            if source_archive is not None:
                await asyncio.to_thread(self._cache_path().unlink, missing_ok=True)

    def _cache_path(self) -> Path:
        configured = os.getenv("DEEPSWE_DATASET_CACHE")
        cache_root = (
            Path(configured) if configured else Path(tempfile.gettempdir()) / "chutes-bench-runner"
        )
        return cache_root / f"deep-swe-{self.benchmark_spec.commit}.tar.gz"

    async def _load_source_archive(self) -> bytes:
        cache_path = self._cache_path()
        if cache_path.is_file():
            cached = await asyncio.to_thread(cache_path.read_bytes)
            if hashlib.sha256(cached).hexdigest() == self.benchmark_spec.archive_sha256:
                return cached
            logger.warning("Discarding DeepSWE cache with wrong SHA-256", path=str(cache_path))
            await asyncio.to_thread(cache_path.unlink)

        async with httpx.AsyncClient(follow_redirects=True, timeout=180.0) as client:
            response = await client.get(self.benchmark_spec.archive_url)
            response.raise_for_status()
            archive = response.content
        actual = hashlib.sha256(archive).hexdigest()
        if actual != self.benchmark_spec.archive_sha256:
            raise BenchmarkIdentityError(
                "deepswe source archive SHA-256 mismatch: "
                f"expected {self.benchmark_spec.archive_sha256}, got {actual}"
            )

        await asyncio.to_thread(cache_path.parent.mkdir, parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(prefix=cache_path.name, dir=cache_path.parent)
        os.close(fd)
        temp_path = Path(temp_name)
        try:
            await asyncio.to_thread(temp_path.write_bytes, archive)
            await asyncio.to_thread(os.replace, temp_path, cache_path)
        finally:
            if temp_path.exists():
                await asyncio.to_thread(temp_path.unlink)
        return archive

    @staticmethod
    def _task_archive(
        source: tarfile.TarFile,
        members: list[tuple[tarfile.TarInfo, str]],
        *,
        include: str,
    ) -> bytes:
        """Build either the sanitized agent bundle or held-out verifier bundle."""
        output = io.BytesIO()
        with tarfile.open(fileobj=output, mode="w") as target:
            for member, relative_name in members:
                path = PurePosixPath(relative_name)
                if include == "agent":
                    if path.parts[0] in {"tests", "solution"}:
                        continue
                    target_name = relative_name
                else:
                    if path.parts[0] != "tests" or len(path.parts) < 2:
                        continue
                    target_name = str(PurePosixPath(*path.parts[1:]))

                copied = copy.copy(member)
                copied.name = target_name
                extracted = source.extractfile(member) if member.isfile() else None
                if extracted is not None:
                    target.addfile(copied, io.BytesIO(extracted.read()))
                elif target_name not in {".", ""}:
                    target.addfile(copied)
        return output.getvalue()

    def _items_from_source_archive(self, archive: bytes) -> list[dict[str, Any]]:
        members_by_task: dict[str, list[tuple[tarfile.TarInfo, str]]] = {
            task_id: [] for task_id in self.benchmark_spec.task_ids
        }
        files_by_task: dict[str, dict[str, bytes]] = {
            task_id: {} for task_id in self.benchmark_spec.task_ids
        }
        markers: set[str] = set()

        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as source:
            for member in source.getmembers():
                parts = PurePosixPath(member.name).parts
                if len(parts) < 4 or parts[1] != "tasks":
                    continue
                task_id = parts[2]
                if task_id not in members_by_task:
                    continue
                relative = PurePosixPath(*parts[3:])
                if relative.is_absolute() or ".." in relative.parts:
                    raise BenchmarkIdentityError(
                        f"Unsafe path in DeepSWE source archive: {member.name}"
                    )
                if member.issym() or member.islnk():
                    link = PurePosixPath(member.linkname)
                    if link.is_absolute() or ".." in link.parts:
                        raise BenchmarkIdentityError(
                            f"Unsafe link in DeepSWE source archive: {member.name}"
                        )
                relative_name = str(relative)
                members_by_task[task_id].append((member, relative_name))
                if member.isfile():
                    extracted = source.extractfile(member)
                    if extracted is not None:
                        files_by_task[task_id][relative_name] = extracted.read()
                if relative_name == "task.toml":
                    markers.add(task_id)

            missing = set(self.benchmark_spec.task_ids) - markers
            if missing:
                raise BenchmarkIdentityError(
                    f"deepswe source is missing {len(missing)} canonical tasks: "
                    f"{', '.join(sorted(missing))}"
                )

            items = []
            for index, task_id in enumerate(self.benchmark_spec.task_ids):
                files = files_by_task[task_id]
                manifest_text = files["task.toml"].decode("utf-8")
                manifest = tomllib.loads(manifest_text)
                verifier = manifest.get("verifier") or {}
                environment = manifest.get("environment") or {}
                metadata = manifest.get("metadata") or {}
                collect = verifier.get("collect") or []

                heldout_hashes = []
                for name, content in files.items():
                    if name.startswith("tests/") or name.startswith("solution/"):
                        heldout_hashes.append(
                            {
                                "name": PurePosixPath(name).name,
                                "sha256": hashlib.sha256(content).hexdigest(),
                                "size": len(content),
                            }
                        )

                items.append(
                    {
                        "id": str(index),
                        "task_id": task_id,
                        "instruction": files["instruction.md"].decode("utf-8"),
                        "task_toml": manifest_text,
                        "parsed_toml": manifest,
                        # The agent bundle is sanitized while still in the
                        # bench-runner process. A full archive containing the
                        # solution is never uploaded to Sandy.
                        "agent_archive": self._task_archive(
                            source, members_by_task[task_id], include="agent"
                        ),
                        "verifier_archive": self._task_archive(
                            source, members_by_task[task_id], include="verifier"
                        ),
                        "heldout_hashes": heldout_hashes,
                        "docker_image": environment.get("docker_image"),
                        "cpus": environment.get("cpus"),
                        "memory_mb": environment.get("memory_mb"),
                        "storage_mb": environment.get("storage_mb"),
                        "base_commit_hash": metadata.get("base_commit_hash"),
                        "language": metadata.get("language"),
                        "max_agent_timeout_sec": (manifest.get("agent") or {}).get("timeout_sec"),
                        "max_test_timeout_sec": verifier.get("timeout_sec"),
                        "environment_build_timeout_sec": environment.get("build_timeout_sec"),
                        "verifier_build_timeout_sec": (verifier.get("environment") or {}).get(
                            "build_timeout_sec"
                        ),
                        "collect_command": (collect[0] if collect else {}).get("command"),
                        "collect_timeout_sec": (collect[0] if collect else {}).get("timeout_sec"),
                        "agent_network_mode": (manifest.get("agent") or {}).get("network_mode"),
                        "verifier_network_mode": verifier.get("network_mode"),
                        "verifier_environment_mode": verifier.get("environment_mode"),
                        "dataset_repository": self.benchmark_spec.repository,
                        "dataset_commit": self.benchmark_spec.commit,
                    }
                )
        return items

    def _assert_benchmark_identity(self) -> None:
        expected = list(self.benchmark_spec.task_ids)
        loaded = [item.get("task_id") for item in self._items]
        if len(self._items) != self.benchmark_spec.expected_count:
            raise BenchmarkIdentityError(
                "deepswe identity check failed: expected "
                f"{self.benchmark_spec.expected_count} items, loaded {len(self._items)}"
            )
        if len(set(loaded)) != len(loaded):
            raise BenchmarkIdentityError("deepswe identity check failed: duplicate task IDs loaded")
        if loaded != expected:
            missing = sorted(set(expected) - set(loaded))
            unexpected = sorted(set(loaded) - set(expected))
            raise BenchmarkIdentityError(
                f"deepswe task manifest mismatch; missing={missing}, unexpected={unexpected}"
            )

        for item in self._items:
            task_id = item["task_id"]
            violations = []
            if not item.get("docker_image"):
                violations.append("missing docker_image")
            if not item.get("base_commit_hash"):
                violations.append("missing base_commit_hash")
            if item.get("agent_network_mode") != "no-network":
                violations.append("agent is not no-network")
            if item.get("verifier_network_mode") != "no-network":
                violations.append("verifier is not no-network")
            if item.get("verifier_environment_mode") != "separate":
                violations.append("verifier is not separate")
            if not item.get("collect_command"):
                violations.append("missing verifier collect hook")
            if violations:
                raise BenchmarkIdentityError(
                    f"deepswe task {task_id} violates the v1.1 protocol: " + ", ".join(violations)
                )

    async def _upload_archive(
        self,
        sandbox_id: str,
        archive: bytes,
        *,
        basename: str,
        destination: str,
    ) -> dict[str, Any]:
        encoded_name = f"/workspace/{basename}.b64"
        tar_name = f"/workspace/{basename}.tar"
        encoded = base64.b64encode(archive).decode("ascii")
        if not await self.sandy.write_file(sandbox_id, encoded_name, encoded):
            return {"ok": False, "error": self.sandy.last_error or "archive upload failed"}
        command = (
            f"mkdir -p {shlex.quote(destination)} && "
            f"base64 -d {shlex.quote(encoded_name)} > {shlex.quote(tar_name)} && "
            f"tar -xf {shlex.quote(tar_name)} -C {shlex.quote(destination)}; "
            "archive_status=$?; "
            f"rm -f {shlex.quote(encoded_name)} {shlex.quote(tar_name)}; "
            "exit $archive_status"
        )
        result = await self.sandy.execute_command(sandbox_id, command)
        return {"ok": (result or {}).get("exit_code") == 0, "result": result}

    async def _verify_workspace_clean(
        self,
        sandbox_id: str,
        heldout_hashes: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        expected_hashes = {entry["sha256"] for entry in (heldout_hashes or [])}
        heldout_sizes = sorted({int(entry["size"]) for entry in (heldout_hashes or [])})
        size_expr = " -o ".join(f"-size {size}c" for size in heldout_sizes)
        hash_scan = (
            f"find / -xdev -type f \\( {size_expr} \\) -exec sha256sum {{}} + 2>/dev/null || true"
            if size_expr
            else ":"
        )
        result = await self.sandy.execute_command(
            sandbox_id,
            "answer_count=$(find /workspace/task/tests /workspace/task/solution "
            "-type f 2>/dev/null | wc -l); "
            "archive_count=$(find /workspace -type f "
            "\\( -name '*.tar' -o -name '*.b64' -o -name '*.tar.gz' \\) | wc -l); "
            "cache_count=$(find /var/cache/sandy -mindepth 1 -type f 2>/dev/null | wc -l); "
            "echo ANSWERS=$answer_count ARCHIVES=$archive_count CACHE_FILES=$cache_count; "
            + hash_scan,
            timeout_ms=300000,
        )
        stdout = ((result or {}).get("stdout") or "").strip()
        found_hashes = []
        for line in stdout.splitlines():
            digest = line.strip().split(maxsplit=1)[0] if line.strip() else ""
            if digest in expected_hashes:
                found_hashes.append(line.strip())
        return {
            "clean": (
                (result or {}).get("exit_code") == 0
                and "ANSWERS=0" in stdout
                and "ARCHIVES=0" in stdout
                and not found_hashes
            ),
            "stdout": stdout,
            "heldout_hash_matches": found_hashes[:50],
            "result": result,
            "observed_from": "agent_sandbox_namespace",
        }

    async def _verify_container_clean(
        self,
        sandbox_id: str,
        container_name: str,
        heldout_hashes: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Prove from inside the runnable image that held-out bytes are absent."""
        expected_hashes = {entry["sha256"] for entry in heldout_hashes}
        heldout_sizes = sorted({int(entry["size"]) for entry in heldout_hashes})
        size_expr = " -o ".join(f"-size {size}c" for size in heldout_sizes)
        hash_scan = (
            f"find / -xdev -type f \\( {size_expr} \\) -exec sha256sum {{}} + 2>/dev/null || true"
            if size_expr
            else ":"
        )
        script = f"set -e; find /tests /solution -type f 2>/dev/null || true; {hash_scan}"
        result = await TerminalBenchAdapter._docker_exec_outside(
            self,
            container_name,
            ["sh", "-c", script],
        )
        found = []
        for line in ((result or {}).get("stdout") or "").splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            digest = stripped.split(maxsplit=1)[0]
            if (
                stripped.startswith("/tests/")
                or stripped.startswith("/solution/")
                or digest in expected_hashes
            ):
                found.append(stripped)
        return {
            "clean": (result or {}).get("exit_code") == 0 and not found,
            "found_in_container": found[:50],
            "probe_exit_code": (result or {}).get("exit_code"),
            "probe_error": (result or {}).get("error") or (result or {}).get("stderr"),
            "observed_from": "worker_docker_api_inside_task_container",
        }

    def _seal_script(self) -> str:
        entries = "\\n".join(
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
        await TerminalBenchAdapter._docker_exec_outside(
            self,
            container_name,
            ["sh", "-c", self._seal_script()],
        )
        sandbox_probe = await self.sandy.execute_command(
            sandbox_id,
            f"echo HOSTS=$(grep -c {shlex.quote(self.SEAL_MARKER)} /etc/hosts); "
            "if command -v curl >/dev/null 2>&1; then "
            "echo CURL=yes; echo FETCH=$(curl -s -m 8 -o /dev/null -w '%{http_code}' "
            f"{shlex.quote(DEEPSWE_SOURCE_PROBE_URL)} "
            "|| echo CURLFAIL); else echo CURL=no; fi",
        )
        container_hosts = await TerminalBenchAdapter._docker_exec_outside(
            self,
            container_name,
            ["sh", "-c", f"grep -c {shlex.quote(self.SEAL_MARKER)} /etc/hosts"],
        )

        def _network_mode() -> str:
            container = docker.from_env().containers.get(container_name)
            labels = (container.attrs.get("Config") or {}).get("Labels") or {}
            if not labels.get("chutes.bench.sandbox_id"):
                raise RuntimeError("task container is missing its sandbox ownership label")
            return str((container.attrs.get("HostConfig") or {}).get("NetworkMode") or "")

        try:
            network_mode = await asyncio.to_thread(_network_mode)
        except Exception as exc:
            network_mode = f"ERROR:{exc}"
        container_stdout = (
            f"MODE={network_mode}\nHOSTS={((container_hosts or {}).get('stdout') or '').strip()}"
        )
        sandbox_stdout = ((sandbox_probe or {}).get("stdout") or "").strip()
        sandbox_blocked = (
            (sandbox_probe or {}).get("exit_code") == 0
            and "HOSTS=0" not in sandbox_stdout
            and "HOSTS=" in sandbox_stdout
            and "CURL=yes" in sandbox_stdout
            and "FETCH=200" not in sandbox_stdout
        )
        container_blocked = (
            (container_hosts or {}).get("exit_code") == 0
            and "MODE=none" in container_stdout
            and "HOSTS=0" not in container_stdout
            and "HOSTS=" in container_stdout
        )
        return {
            "sealed": sandbox_blocked and container_blocked,
            "sandbox_blocked": sandbox_blocked,
            "container_blocked": container_blocked,
            "sandbox_stdout": sandbox_stdout,
            "container_stdout": container_stdout,
            "hosts": list(self.BENCHMARK_SOURCE_HOSTS),
        }

    async def _verify_agent_docker_boundary(
        self,
        sandbox_id: str,
        container_name: str,
    ) -> dict[str, Any]:
        """Attempt the fresh-container source-fetch bypass from the agent namespace."""
        command = (
            "echo SOCKET=$(test -S /var/run/docker.sock && echo PRESENT || echo ABSENT); "
            "echo CACHE_MOUNT=$(mountpoint -q /var/cache/sandy && echo PRESENT || echo ABSENT); "
            "echo CACHE_FILES=$(find /var/cache/sandy -mindepth 1 -print -quit "
            "2>/dev/null | wc -l); "
            'if python3 -c "import socket; s=socket.socket(socket.AF_UNIX); '
            "s.settimeout(2); s.connect('/var/run/docker.sock')\" "
            ">/tmp/raw.out 2>/tmp/raw.err; then echo RAW_DOCKER=ESCAPED; "
            "else echo RAW_DOCKER=BLOCKED; fi; "
            "if docker run --rm "
            f"curlimages/curl:8.10.1 -fsSL {shlex.quote(DEEPSWE_SOURCE_PROBE_URL)} "
            ">/tmp/run.out 2>/tmp/run.err; "
            "then echo SPAWN=ESCAPED; else echo SPAWN=BLOCKED; fi; "
            "if docker exec chutes-bench-runner-worker-1 true >/tmp/other.out 2>/tmp/other.err; "
            "then echo OTHER_CONTAINER=ESCAPED; else echo OTHER_CONTAINER=BLOCKED; fi; "
            f"if docker exec {shlex.quote(container_name)} true >/tmp/task.out 2>/tmp/task.err; "
            "then echo TASK_PATH=WORKS; else echo TASK_PATH=BROKEN; fi; "
            "echo RAW_SHA256=$(sha256sum /tmp/raw.out 2>/dev/null | cut -d' ' -f1); "
            "echo SPAWN_SHA256=$(sha256sum /tmp/run.out 2>/dev/null | cut -d' ' -f1); "
            "echo SPAWN_ERROR=$(tr '\\n' ' ' </tmp/run.err | head -c 240)"
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
            "source_probe_url": DEEPSWE_SOURCE_PROBE_URL,
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
        """Prove the CLI is gone before held-out verifier files are uploaded.

        Returning from Sandy's SSE request proves only that the stream ended.
        If the stream breaks while the CLI remains alive, that process still
        has the sandbox filesystem and Docker socket.  Uploading verifier files
        at that point would expose the answer key to the arm under test.

        Sandy can also emit a synthetic ``complete`` after a transient failure
        in its own process-status probe.  In that case the wrapper is still
        running and ``agent.done`` has not been written.  Keep the source seal
        in place and wait only for the unused portion of this item's declared
        agent budget.  Never turn an unproven exit into a scored observation.
        """
        completion_event = agent_summary.get("type") == "complete"
        deadline = time.monotonic() + max(0.0, wait_timeout_seconds)
        attempts = 0
        initial_probe = ""
        stdout = ""
        probe: dict[str, Any] = {}
        process_stopped = False
        done_recorded = False

        while True:
            attempts += 1
            probe = await self.sandy.execute_command(
                sandbox_id,
                "pid=$(cat /workspace/.chutes/agent.pid 2>/dev/null || true); "
                "if [ -f /workspace/.chutes/agent.done ]; then "
                "done_present=yes; "
                "done_value=$(cat /workspace/.chutes/agent.done 2>/dev/null || true); "
                "else done_present=no; done_value=; fi; "
                'if [ -z "$pid" ]; then running=no; state=missing; '
                'elif [ -r "/proc/$pid/stat" ]; then '
                "state=$(cut -d' ' -f3 \"/proc/$pid/stat\" 2>/dev/null || echo unknown); "
                'case "$state" in Z*) running=no;; *) running=yes;; esac; '
                "else running=no; state=gone; fi; "
                "echo PID=${pid:-missing} STATE=$state RUNNING=$running "
                "DONE_PRESENT=$done_present DONE=${done_value:-missing}",
            )
            stdout = ((probe or {}).get("stdout") or "").strip()
            if attempts == 1:
                initial_probe = stdout
            process_stopped = "RUNNING=no" in stdout
            done_recorded = "DONE_PRESENT=yes" in stdout
            if completion_event and process_stopped and done_recorded:
                break
            remaining = deadline - time.monotonic()
            if not completion_event or remaining <= 0:
                break
            await asyncio.sleep(min(max(0.0, poll_interval_seconds), remaining))

        done_value: int | None = None
        for field in stdout.split():
            if field.startswith("DONE="):
                candidate = field.removeprefix("DONE=")
                if candidate.lstrip("-").isdigit():
                    done_value = int(candidate)
                break
        return {
            "completion_event": completion_event,
            "process_stopped": process_stopped,
            "done_recorded": done_recorded,
            "done_value": done_value,
            "terminated": completion_event and process_stopped and done_recorded,
            "attempts": attempts,
            "waited_seconds": max(
                0.0,
                wait_timeout_seconds - max(0.0, deadline - time.monotonic()),
            ),
            "initial_probe": initial_probe,
            "probe": stdout,
            "probe_exit": (probe or {}).get("exit_code"),
        }

    async def _docker_exec_outside(
        self,
        container_name: str,
        argv: list[str],
        *,
        workdir: str | None = None,
        user: str = "",
    ) -> dict[str, Any]:
        return await TerminalBenchAdapter._docker_exec_outside(
            self,
            container_name,
            argv,
            workdir=workdir,
            user=user,
        )

    async def _put_archive_outside(
        self,
        container_name: str,
        destination: str,
        archive_bytes: bytes,
    ) -> dict[str, Any]:
        return await TerminalBenchAdapter._put_archive_outside(
            self,
            container_name,
            destination,
            archive_bytes,
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
            self,
            container_name,
            destination,
            content,
            mode=mode,
        )

    @staticmethod
    def _resource_kwargs(item: dict[str, Any]) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        if item.get("cpus"):
            kwargs["nano_cpus"] = int(float(item["cpus"]) * 1_000_000_000)
        if item.get("memory_mb"):
            kwargs["mem_limit"] = f"{int(item['memory_mb'])}m"
        return kwargs

    async def _start_agent_container(
        self,
        sandbox_id: str,
        item: dict[str, Any],
        container_name: str,
        pull_timeout_sec: int,
    ) -> dict[str, Any]:
        image = item["docker_image"]

        def _pull_and_start() -> dict[str, Any]:
            client = docker.from_env()
            pulled = client.images.pull(image)
            try:
                stale = client.containers.get(container_name)
                stale.remove(force=True)
            except docker.errors.NotFound:
                pass
            container = client.containers.run(
                pulled.id,
                command=["sleep", "infinity"],
                name=container_name,
                detach=True,
                network_mode="none",
                working_dir="/app",
                labels={
                    "chutes.benchmark": "deepswe",
                    "chutes.bench.sandbox_id": sandbox_id,
                    "chutes.bench.role": "agent-task",
                },
                **self._resource_kwargs(item),
            )
            container.reload()
            repo_digests = pulled.attrs.get("RepoDigests") or []
            return {
                "container_id": container.id,
                "image_id": pulled.id,
                "image_digest": str(repo_digests[0]) if repo_digests else pulled.id,
                "network_mode": (container.attrs.get("HostConfig") or {}).get("NetworkMode"),
            }

        try:
            started = await asyncio.wait_for(
                asyncio.to_thread(_pull_and_start), timeout=max(1, pull_timeout_sec)
            )
        except TimeoutError:
            return {"ok": False, "stage": "image_pull", "error": "image pull timed out"}
        except Exception as exc:
            return {"ok": False, "stage": "container_start", "error": str(exc)}
        prepared = await TerminalBenchAdapter._docker_exec_outside(
            self,
            container_name,
            ["mkdir", "-p", "/logs/agent"],
        )
        if prepared.get("exit_code") != 0:
            return {"ok": False, "stage": "container_start", "result": prepared}
        return {
            "ok": True,
            **started,
            "observed_from": "worker_docker_api",
        }

    async def _collect_patch(
        self,
        sandbox_id: str,
        item: dict[str, Any],
        container_name: str,
        timeout_sec: int,
    ) -> dict[str, Any]:
        command = item["collect_command"]
        try:
            collect = await asyncio.wait_for(
                TerminalBenchAdapter._docker_exec_outside(
                    self,
                    container_name,
                    ["bash", "-lc", command],
                    workdir="/app",
                ),
                timeout=max(1, timeout_sec),
            )
        except TimeoutError:
            collect = {"exit_code": -1, "error": "patch collection timed out"}
        copy_result: dict[str, Any] = {}
        fallback_result: dict[str, Any] = {}
        patch_bytes = b""
        if (collect or {}).get("exit_code") == 0:
            copy_result = await TerminalBenchAdapter._docker_exec_outside(
                self,
                container_name,
                ["base64", "-w0", "/logs/artifacts/model.patch"],
            )
            if copy_result.get("exit_code") == 0:
                try:
                    patch_bytes = base64.b64decode(copy_result.get("stdout") or "", validate=True)
                except ValueError:
                    copy_result = {
                        "exit_code": -1,
                        "error": "model.patch transfer was not valid base64",
                    }
        if (collect or {}).get("exit_code") != 0 or copy_result.get("exit_code") != 0:
            # A CLI that deletes .git, does not commit, or otherwise breaks its
            # own submission path receives a normal zero. Only transport loss
            # is excluded by the caller. Materialize the official no-patch
            # input so the pristine verifier can produce that zero.
            fallback_result = {"exit_code": 0, "used_empty_patch": True}
            patch_bytes = b""
        return {
            "collect": collect,
            "copy": copy_result,
            "fallback": fallback_result,
            "_patch_bytes": patch_bytes,
        }

    async def _stage_and_run_verifier(
        self,
        sandbox_id: str,
        item: dict[str, Any],
        patch_bytes: bytes,
        verifier_container: str,
        verifier_image: str,
        build_timeout_sec: int,
        verifier_timeout_sec: int,
    ) -> dict[str, Any]:
        def _build_and_start() -> dict[str, Any]:
            client = docker.from_env()
            built, _logs = client.images.build(
                fileobj=io.BytesIO(item["verifier_archive"]),
                custom_context=True,
                tag=verifier_image,
                network_mode="none",
                rm=True,
            )
            try:
                stale = client.containers.get(verifier_container)
                stale.remove(force=True)
            except docker.errors.NotFound:
                pass
            container = client.containers.run(
                built.id,
                command=["sleep", "infinity"],
                name=verifier_container,
                detach=True,
                network_mode="none",
                working_dir="/app",
                labels={
                    "chutes.benchmark": "deepswe",
                    "chutes.bench.sandbox_id": sandbox_id,
                    "chutes.bench.role": "heldout-verifier",
                },
                **self._resource_kwargs(item),
            )
            container.reload()
            return {
                "image_id": built.id,
                "container_id": container.id,
                "network_mode": (container.attrs.get("HostConfig") or {}).get("NetworkMode"),
            }

        try:
            build = await asyncio.wait_for(
                asyncio.to_thread(_build_and_start), timeout=max(1, build_timeout_sec)
            )
        except TimeoutError:
            return {"ok": False, "stage": "verifier_build", "error": "build timed out"}
        except Exception as exc:
            return {"ok": False, "stage": "verifier_build", "error": str(exc)}

        verifier_integrity_script = (
            "test -f /tests/test.sh && test ! -e /solution && "
            f'test "$(git -C /app rev-parse HEAD)" = "{item["base_commit_hash"]}"'
        )
        separate_probe = await TerminalBenchAdapter._docker_exec_outside(
            self,
            verifier_container,
            ["sh", "-c", verifier_integrity_script],
        )
        if (separate_probe or {}).get("exit_code") != 0 or build.get("network_mode") != "none":
            return {
                "ok": False,
                "stage": "verifier_integrity",
                "probe": separate_probe,
                "network_mode": build.get("network_mode"),
            }

        prepare_dirs = await TerminalBenchAdapter._docker_exec_outside(
            self,
            verifier_container,
            ["mkdir", "-p", "/logs/artifacts", "/logs/verifier"],
        )
        prepare = await TerminalBenchAdapter._put_file_outside(
            self,
            verifier_container,
            "/logs/artifacts/model.patch",
            patch_bytes,
        )
        if prepare_dirs.get("exit_code") != 0:
            prepare = prepare_dirs
        if (prepare or {}).get("exit_code") != 0:
            return {"ok": False, "stage": "verifier_transfer", "prepare": prepare}

        test_command_marker = f"{sandbox_id}:{item['id']}:{time.time_ns()}"
        marker_path = "/logs/verifier/.chutes-test-command-started"
        verifier_command = (
            "rm -f /logs/verifier/reward.json /logs/verifier/reward.txt "
            f"{marker_path} && "
            f"printf '%s' {shlex.quote(test_command_marker)} > {marker_path} && "
            "exec bash /tests/test.sh"
        )
        try:
            test = await asyncio.wait_for(
                TerminalBenchAdapter._docker_exec_outside(
                    self,
                    verifier_container,
                    ["bash", "-c", verifier_command],
                    workdir="/app",
                ),
                timeout=max(1, verifier_timeout_sec),
            )
        except TimeoutError:
            test = {"exit_code": -1, "error": "verifier timed out"}
        test_command_probe = await TerminalBenchAdapter._docker_exec_outside(
            self,
            verifier_container,
            ["sh", "-c", f"cat {marker_path} 2>/dev/null || true"],
        )
        test_command_executed = (
            (test_command_probe or {}).get("stdout") or ""
        ).strip() == test_command_marker
        reward = await TerminalBenchAdapter._docker_exec_outside(
            self,
            verifier_container,
            [
                "sh",
                "-c",
                "if [ -f /logs/verifier/reward.json ]; then "
                "cat /logs/verifier/reward.json; "
                "elif [ -f /logs/verifier/reward.txt ]; then "
                "cat /logs/verifier/reward.txt; else exit 1; fi",
            ],
        )
        return {
            "ok": True,
            "staged_in_agent_sandbox": False,
            "build": build,
            "separate_probe": separate_probe,
            "prepare": prepare,
            "test": test,
            "test_command_probe": test_command_probe,
            "test_command_executed": test_command_executed,
            "reward": reward,
        }

    @staticmethod
    def _parse_reward(raw: str) -> tuple[float | None, dict[str, Any]]:
        value = (raw or "").strip()
        if not value:
            return None, {}
        try:
            parsed = json.loads(value)
        except (ValueError, TypeError):
            try:
                return float(value), {"reward": float(value)}
            except ValueError:
                return None, {}
        if isinstance(parsed, (int, float)):
            return float(parsed), {"reward": float(parsed)}
        if isinstance(parsed, dict) and isinstance(parsed.get("reward"), (int, float)):
            return float(parsed["reward"]), parsed
        return None, parsed if isinstance(parsed, dict) else {}

    def _agent_name(self) -> str:
        return (
            (getattr(self, "run_config", None) or {}).get("deepswe", {}).get("agent")
            or os.getenv("DEEPSWE_AGENT")
            or "codex"
        )

    def _context_limit_tokens(self) -> int | None:
        value = (
            (getattr(self, "run_config", None) or {}).get("deepswe", {}).get("context_limit_tokens")
        )
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError("config.deepswe.context_limit_tokens must be a positive integer")
        return value

    async def _prepare_agent_launch(
        self,
        sandbox_id: str,
        agent_name: str,
        context_limit_tokens: int | None,
    ) -> AgentProviderLaunch:
        return await prepare_sandy_agent_launch(
            client=self.client,
            sandy=self.sandy,
            sandbox_id=sandbox_id,
            agent=agent_name,
            model=self.model_slug,
            context_limit_tokens=context_limit_tokens,
        )

    def _new_item_observability(self, item_id: str) -> dict[str, Any]:
        state = {
            "agent": None,
            "context_limit_tokens": None,
            "configured_context_window": None,
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
                "rollout_metrics": None,
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
            "rollout_metrics": None,
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
        """Archive stable rollout evidence before teardown without affecting scoring."""
        state = self._item_observability.get(item_id)
        if not state or not state.get("agent_invoked"):
            return
        if state.get("retention_task") is None:
            # The agent stream can fail or be canceled while its process is
            # still alive. Stop it before copying/archiving a stable final
            # prefix, then mirror any alternate OpenRouter config home into
            # the evidence paths consumed by retain_agent_evidence.
            try:
                await self.sandy.execute_command(
                    sandbox_id,
                    "if [ -f /workspace/.chutes/agent.pid ]; then "
                    "kill -TERM $(cat /workspace/.chutes/agent.pid) 2>/dev/null || true; "
                    "sleep 1; kill -KILL $(cat /workspace/.chutes/agent.pid) "
                    "2>/dev/null || true; fi; "
                    "test -f /workspace/.chutes/agent.done || "
                    "echo 143 > /workspace/.chutes/agent.done",
                    timeout_ms=10_000,
                )
            except Exception as exc:
                logger.warning(
                    "Could not stop DeepSWE agent before evidence retention",
                    item_id=item_id,
                    sandbox_id=sandbox_id,
                    error=str(exc),
                )
            agent_launch = state.get("agent_launch")
            if agent_launch is not None and not state.get("rollout_retained"):
                try:
                    await retain_sandy_agent_rollout(self.sandy, sandbox_id, agent_launch)
                    state["rollout_retained"] = True
                except Exception as exc:
                    state["rollout_retention_error"] = str(exc) or exc.__class__.__name__
                    logger.warning(
                        "Could not retain DeepSWE rollout before evidence archive",
                        item_id=item_id,
                        sandbox_id=sandbox_id,
                        error=state["rollout_retention_error"],
                    )
            self._start_evidence_retention(item_id, sandbox_id)
        try:
            state["evidence"] = await state["retention_task"]
            state["evidence"]["rollout_retention_error"] = state.get("rollout_retention_error")
        except BaseException as exc:
            # This hook runs from the sandbox finalizer, including worker
            # cancellation. Evidence failure must not replace a valid score.
            state["evidence"] = {
                "status": "failed",
                "path": None,
                "sha256": None,
                "size_bytes": None,
                "error": f"retention task failed: {exc}",
                "token_usage_samples": None,
                "rollout_metrics": None,
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
        rollout_metrics = evidence.get("rollout_metrics") or {}
        result.metadata["compaction_experiment"] = {
            "schema_version": 1,
            "arm": state.get("agent") or result.metadata.get("agent"),
            "context_limit_tokens": state.get("context_limit_tokens"),
            "configured_context_window": state.get("configured_context_window"),
            "compaction_events": rollout_metrics.get("compaction_events"),
            "compaction_events_by_type": rollout_metrics.get("compaction_events_by_type"),
            "rollout_line_count": rollout_metrics.get("rollout_line_count"),
            "tool_calls_by_name": rollout_metrics.get("tool_calls_by_name"),
            "rollout_metrics_complete": rollout_metrics.get("complete"),
            "score": result.score,
        }
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
        details = {
            "task_id": item.get("task_id"),
            "harness": DEEPSWE_HARNESS,
            "exclusion_reason": reason,
            **(metadata or {}),
        }
        return ItemResult(
            item_id=item["id"],
            item_hash=self.compute_item_hash(item.get("task_id")),
            prompt=prompt,
            error=error,
            latency_ms=int((time.time() - start_time) * 1000),
            input_tokens=(usage or {}).get("input_tokens"),
            output_tokens=(usage or {}).get("output_tokens"),
            metadata=details,
        )

    async def evaluate_item(self, item_id: str) -> ItemResult:
        """Evaluate and attach evidence for scored and excluded outcomes."""
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
        state = self._item_observability[item_id]
        state["agent"] = agent_name
        instruction = item["instruction"]
        prompt = (
            "You are an interactive coding agent in a Sandy sandbox. A no-network "
            "DeepSWE task container is already running. Use docker exec to inspect "
            "and modify the repository at /app inside that container. Do not search "
            "for benchmark sources, hidden tests, or reference solutions. You may run "
            "the repository's existing tests. When finished, commit every intended "
            "change to git; the harness submits the binary diff from the pinned base "
            "commit through HEAD to a separate pristine verifier container.\n\n"
            f"Task instruction:\n{instruction}\n"
        )
        start_time = time.time()
        sandbox_id: str | None = None
        agent_container: str | None = None
        verifier_container: str | None = None
        verifier_image: str | None = None
        image = item["docker_image"]
        agent_summary: dict[str, Any] = {}
        agent_usage: dict[str, Any] = {}

        try:
            context_limit_tokens = self._context_limit_tokens()
            state["context_limit_tokens"] = context_limit_tokens
            budget = self._item_budgets_seconds(item)
            outer_timeout = self.get_item_timeout_seconds(item_id)
            sandbox_ttl_min = math.ceil((outer_timeout or 0) / 60)
            sandbox_id = await self.sandy.create_sandbox(
                # Docker image/container setup is driven by this worker through
                # the outside helpers. The future agent namespace never receives
                # the host Docker socket or Sandy's cross-job cache.
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

            staged = await self._upload_archive(
                sandbox_id,
                item["agent_archive"],
                basename="deepswe-agent",
                destination="/workspace/task",
            )
            if not staged.get("ok"):
                return self._excluded_result(
                    item=item,
                    prompt=prompt,
                    reason="infrastructure_setup",
                    error="Could not stage the sanitized DeepSWE task metadata",
                    start_time=start_time,
                    metadata={"agent": agent_name, "staged": staged},
                )

            # This check happens before even pulling the task image. Only the
            # sanitized archive was uploaded, and that archive is now deleted.
            workspace_clean = await self._verify_workspace_clean(sandbox_id)
            if not workspace_clean.get("clean"):
                return self._excluded_result(
                    item=item,
                    prompt=prompt,
                    reason="integrity_answer_key_unproven",
                    error=(
                        "Refusing to score: answer-key absence before image setup "
                        f"could not be proven ({workspace_clean.get('stdout')!r})"
                    ),
                    start_time=start_time,
                    metadata={"agent": agent_name, "workspace_clean": workspace_clean},
                )

            namespace = f"s{sandbox_id[:12]}".lower()
            safe_task = "".join(ch if ch.isalnum() else "-" for ch in item["task_id"].lower())
            agent_container = f"deepswe_{namespace}_{safe_task}_agent"
            verifier_container = f"deepswe_{namespace}_{safe_task}_verifier"
            verifier_image = f"deepswe_{namespace}_{safe_task}_verifier"

            setup = await self._start_agent_container(
                sandbox_id,
                item,
                agent_container,
                int(budget["environment_build"]),
            )
            if not setup.get("ok"):
                return self._excluded_result(
                    item=item,
                    prompt=prompt,
                    reason=f"infrastructure_{setup.get('stage') or 'setup'}",
                    error=f"DeepSWE task environment setup failed: {setup}",
                    start_time=start_time,
                    metadata={"agent": agent_name, "setup": setup},
                )

            container_clean = await self._verify_container_clean(
                sandbox_id, agent_container, item["heldout_hashes"]
            )
            if not container_clean.get("clean"):
                return self._excluded_result(
                    item=item,
                    prompt=prompt,
                    reason="integrity_answer_key_unproven",
                    error=(
                        "Refusing to score: held-out DeepSWE material may be reachable "
                        f"inside the agent container: {container_clean}"
                    ),
                    start_time=start_time,
                    metadata={
                        "agent": agent_name,
                        "workspace_clean": workspace_clean,
                        "container_clean": container_clean,
                    },
                )

            try:
                gateway = await TerminalBenchAdapter._start_task_gateway(
                    self,
                    sandbox_id,
                    agent_container,
                )
            except Exception as exc:
                return self._excluded_result(
                    item=item,
                    prompt=prompt,
                    reason="integrity_docker_boundary_unproven",
                    error=f"Refusing to score: task-scoped Docker gateway failed: {exc}",
                    start_time=start_time,
                    metadata={"agent": agent_name},
                )

            docker_boundary = await self._verify_agent_docker_boundary(
                sandbox_id,
                agent_container,
            )
            if not docker_boundary.get("boundary_held"):
                return self._excluded_result(
                    item=item,
                    prompt=prompt,
                    reason="integrity_docker_boundary_unproven",
                    error=(
                        "Refusing to score: the fresh-container benchmark-source "
                        f"bypass was not blocked: {docker_boundary}"
                    ),
                    start_time=start_time,
                    metadata={
                        "agent": agent_name,
                        "agent_docker_gateway": gateway,
                        "docker_boundary": docker_boundary,
                    },
                )

            agent_view_clean = await self._verify_workspace_clean(
                sandbox_id,
                item["heldout_hashes"],
            )
            if not agent_view_clean.get("clean"):
                return self._excluded_result(
                    item=item,
                    prompt=prompt,
                    reason="integrity_answer_key_unproven",
                    error=(
                        "Refusing to score: held-out DeepSWE material is reachable "
                        f"from the final agent namespace: {agent_view_clean}"
                    ),
                    start_time=start_time,
                    metadata={
                        "agent": agent_name,
                        "agent_docker_gateway": gateway,
                        "docker_boundary": docker_boundary,
                        "agent_view_clean": agent_view_clean,
                    },
                )

            seal = await self._seal_network(sandbox_id, agent_container)
            if not seal.get("sealed"):
                return self._excluded_result(
                    item=item,
                    prompt=prompt,
                    reason="integrity_network_seal_unproven",
                    error=f"Refusing to score: DeepSWE source network seal failed: {seal}",
                    start_time=start_time,
                    metadata={"agent": agent_name, "seal": seal},
                )

            agent_launch = await self._prepare_agent_launch(
                sandbox_id,
                agent_name,
                context_limit_tokens,
            )
            state["agent_launch"] = agent_launch
            if agent_launch.setup is not None:
                state["configured_context_window"] = agent_launch.setup.context_window
            agent_started_at = time.monotonic()
            state["agent_invoked"] = True
            agent_result = await self.sandy.run_agent(
                sandbox_id,
                agent=agent_name,
                model=self.model_slug,
                prompt=prompt + f"\nTask container name: {agent_container}\n",
                max_duration=int(budget["agent"]),
                raw_prompt=True,
                api_base_url=agent_launch.api_base_url,
                env_vars=agent_launch.env_vars,
            )
            agent_call_seconds = time.monotonic() - agent_started_at
            agent_summary = (agent_result or {}).get("summary") or {}
            sandbox_alive = await self._sandbox_alive(sandbox_id)
            exclusion_reason, exit_note = classify_deepswe_agent_outcome(
                agent_summary, float(budget["agent"]), sandbox_alive
            )
            if exclusion_reason:
                return self._excluded_result(
                    item=item,
                    prompt=prompt,
                    reason=exclusion_reason,
                    error=exit_note or exclusion_reason,
                    start_time=start_time,
                    usage=agent_usage,
                    metadata={
                        "agent": agent_name,
                        "agent_summary": agent_summary,
                        "sandbox_alive_at_exit": sandbox_alive,
                        "seal": seal,
                    },
                )

            agent_termination = await self._verify_agent_terminated(
                sandbox_id,
                agent_summary,
                wait_timeout_seconds=max(0.0, float(budget["agent"]) - agent_call_seconds),
            )
            if not agent_termination.get("terminated"):
                return self._excluded_result(
                    item=item,
                    prompt=prompt,
                    reason=DEEPSWE_AGENT_NOT_TERMINATED_EXCLUSION_REASON,
                    error=(
                        "Agent completion could not be verified before held-out "
                        "DeepSWE verifier files would be uploaded "
                        f"(completion_event={agent_termination.get('completion_event')}, "
                        f"probe={agent_termination.get('probe')!r})."
                    ),
                    start_time=start_time,
                    metadata={
                        "agent": agent_name,
                        "agent_summary": agent_summary,
                        "agent_termination": agent_termination,
                        "seal": seal,
                    },
                )

            usage_error: str | None = None
            try:
                await retain_sandy_agent_rollout(self.sandy, sandbox_id, agent_launch)
                state["rollout_retained"] = True
            except Exception as exc:
                usage_error = str(exc) or exc.__class__.__name__
                state["rollout_retention_error"] = usage_error
            agent_usage = await collect_agent_usage(self.sandy, sandbox_id)
            if usage_error is None:
                try:
                    validate_openrouter_agent_usage(agent_launch, agent_usage)
                except RuntimeError as exc:
                    usage_error = str(exc)
            # The CLI is proven stopped and alternate-home rollouts have been
            # mirrored. Transfer the verified archive while patch collection
            # and the separate verifier continue; the finalizer always awaits
            # it before destroying the sandbox.
            self._start_evidence_retention(item_id, sandbox_id)
            if usage_error:
                return self._excluded_result(
                    item=item,
                    prompt=prompt,
                    reason=DEEPSWE_USAGE_ACCOUNTING_EXCLUSION_REASON,
                    error=usage_error,
                    start_time=start_time,
                    usage=agent_usage,
                    metadata={
                        "agent": agent_name,
                        "agent_summary": agent_summary,
                        "agent_termination": agent_termination,
                        "agent_usage": agent_usage,
                        "seal": seal,
                    },
                )

            events = (agent_result or {}).get("events") or []
            agent_output = next(
                (event.get("text") for event in reversed(events) if event.get("type") == "output"),
                "",
            )

            collected = await self._collect_patch(
                sandbox_id,
                item,
                agent_container,
                int(budget["collect"]),
            )
            patch_bytes = collected.pop("_patch_bytes", b"")
            collection_transport_failed = any(
                result.get("exit_code") == -1
                for result in (
                    collected.get("collect") or {},
                    collected.get("copy") or {},
                    collected.get("fallback") or {},
                )
            )
            if collection_transport_failed:
                alive_after_collect = await self._sandbox_alive(sandbox_id)
                reason = (
                    "infrastructure_sandbox_gone"
                    if alive_after_collect is False
                    else "infrastructure_transport"
                )
                return self._excluded_result(
                    item=item,
                    prompt=prompt,
                    reason=reason,
                    error=f"Could not collect the agent patch: {collected}",
                    start_time=start_time,
                    usage=agent_usage,
                    metadata={"agent": agent_name, "collected": collected},
                )

            # Stop the environment the agent controlled before verifier files
            # are uploaded or the pristine verifier container is built.
            def _stop_agent_container() -> None:
                container = docker.from_env().containers.get(agent_container)
                labels = (container.attrs.get("Config") or {}).get("Labels") or {}
                if labels.get("chutes.bench.sandbox_id") != sandbox_id:
                    raise RuntimeError("agent task container ownership changed")
                container.stop(timeout=30)

            try:
                await asyncio.wait_for(
                    asyncio.to_thread(_stop_agent_container),
                    timeout=120,
                )
            except Exception as exc:
                return self._excluded_result(
                    item=item,
                    prompt=prompt,
                    reason="integrity_agent_environment_not_stopped",
                    error=f"Refusing to stage held-out verifier: {exc}",
                    start_time=start_time,
                    usage=agent_usage,
                    metadata={"agent": agent_name, "docker_boundary": docker_boundary},
                )

            verifier = await self._stage_and_run_verifier(
                sandbox_id,
                item,
                patch_bytes,
                verifier_container,
                verifier_image,
                int(budget["verifier_build"]),
                int(budget["verifier"]),
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
                    error=f"DeepSWE separate verifier failed at {stage}: {verifier}",
                    start_time=start_time,
                    usage=agent_usage,
                    metadata={"agent": agent_name, "verifier": verifier},
                )

            test_result = verifier.get("test") or {}
            reward_result = verifier.get("reward") or {}
            reward, reward_metrics, verifier_exclusion, verifier_error = (
                classify_deepswe_verifier_outcome(
                    test_result,
                    reward_result,
                    test_command_executed=bool(verifier.get("test_command_executed")),
                )
            )
            if verifier_exclusion:
                return self._excluded_result(
                    item=item,
                    prompt=prompt,
                    reason=verifier_exclusion,
                    error=(f"{verifier_error}: test={test_result}, reward={reward_result}"),
                    start_time=start_time,
                    usage=agent_usage,
                    metadata={
                        "agent": agent_name,
                        "agent_summary": agent_summary,
                        "agent_termination": agent_termination,
                        "test_command_probe": verifier.get("test_command_probe"),
                        "test_command_executed": verifier.get("test_command_executed"),
                        "reward_metrics": reward_metrics,
                    },
                )

            assert reward is not None
            latency_ms = int((time.time() - start_time) * 1000)
            return ItemResult(
                item_id=item_id,
                item_hash=self.compute_item_hash(item["task_id"]),
                prompt=prompt,
                response=agent_output,
                expected="[DeepSWE verifier binary reward = 1]",
                is_correct=reward >= 1.0,
                score=reward,
                latency_ms=latency_ms,
                input_tokens=agent_usage.get("input_tokens"),
                output_tokens=agent_usage.get("output_tokens"),
                judge_output={
                    "test": test_result,
                    "reward": reward_result,
                    "reward_metrics": reward_metrics,
                    "collect": collected,
                },
                error=None
                if reward >= 1.0
                else (test_result.get("stderr") or test_result.get("error")),
                metadata={
                    "task_id": item["task_id"],
                    "language": item.get("language"),
                    "agent": agent_name,
                    "agent_provider": agent_launch.metadata,
                    "harness": DEEPSWE_HARNESS,
                    "official_leaderboard_harness": "Pier + mini-SWE-agent on Modal",
                    "agent_summary": agent_summary,
                    "agent_termination": agent_termination,
                    "agent_usage": agent_usage,
                    "agent_exit_note": exit_note,
                    "agent_timeout_sec": budget["agent"],
                    "agent_timeout_base_sec": budget["agent_base"],
                    "agent_timeout_multiplier": budget["agent_multiplier"],
                    "verifier_timeout_sec": budget["verifier"],
                    "worker_item_timeout_sec": outer_timeout,
                    "sandbox_ttl_min": sandbox_ttl_min,
                    "cpus": item.get("cpus"),
                    "memory_mb": item.get("memory_mb"),
                    "storage_mb": item.get("storage_mb"),
                    "seal": seal,
                    "agent_docker_gateway": gateway,
                    "docker_boundary": docker_boundary,
                    "answer_key_holdout": {
                        "workspace": workspace_clean,
                        "agent_view": agent_view_clean,
                        "container": container_clean,
                        "full_source_archive_uploaded": False,
                        "solution_uploaded": False,
                        "verifier_uploaded_to_agent_sandbox": False,
                    },
                    "separate_verifier": True,
                    "dataset_repository": item["dataset_repository"],
                    "dataset_commit": item["dataset_commit"],
                    "docker_image": image,
                    "docker_image_digest": setup.get("image_digest"),
                    "base_commit_hash": item.get("base_commit_hash"),
                    "reward_metrics": reward_metrics,
                    "test_command_executed": verifier.get("test_command_executed"),
                },
            )
        except Exception as exc:
            logger.error("DeepSWE evaluation failed", item_id=item_id, error=str(exc))
            exclusion_reason = (
                classify_deepswe_exception(str(exc), agent_summary) or "infrastructure_adapter"
            )
            return self._excluded_result(
                item=item,
                prompt=prompt,
                reason=exclusion_reason,
                error=str(exc),
                start_time=start_time,
                usage=agent_usage,
                metadata={
                    "agent": agent_name,
                    "agent_summary": agent_summary,
                },
            )
        finally:
            if sandbox_id:
                try:
                    await self._finish_evidence_retention(item_id, sandbox_id)
                finally:
                    await TerminalBenchAdapter._cleanup_owned_task_containers(
                        self,
                        sandbox_id,
                    )

                    def _remove_owned_images() -> None:
                        client = docker.from_env()
                        if verifier_image:
                            with contextlib.suppress(Exception):
                                client.images.remove(verifier_image, force=False)
                        if image:
                            # A shared source image is removed only when Docker proves
                            # no concurrent task container still references it.
                            with contextlib.suppress(Exception):
                                client.images.remove(image, force=False)

                    await asyncio.to_thread(_remove_owned_images)
                    await self.sandy.terminate_sandbox(sandbox_id)
