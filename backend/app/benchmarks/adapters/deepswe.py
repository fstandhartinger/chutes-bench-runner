"""Pinned DeepSWE v1.1 driven by Sandy CLI agents.

The official leaderboard uses Pier + mini-SWE-agent on Modal. This adapter
preserves DeepSWE's task and separate-verifier protocol while deliberately
swapping the agent scaffold, so its scores must be labelled "Sandy CLI" and
not presented as official-leaderboard-equivalent numbers.
"""

from __future__ import annotations

import asyncio
import base64
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

import httpx

from app.benchmarks.adapters.deepswe_identity import DEEPSWE_V1_1, DeepSWESpec
from app.benchmarks.adapters.terminal_bench import (
    BenchmarkIdentityError,
    classify_agent_exit,
    classify_bare_failure,
)
from app.benchmarks.agent_usage import collect_agent_usage
from app.benchmarks.base import BenchmarkAdapter, ItemResult
from app.benchmarks.registry import register_adapter
from app.core.config import get_settings
from app.core.logging import get_logger
from app.services.sandy_service import SandyService

logger = get_logger(__name__)

DEEPSWE_ITEM_TIMEOUT_MARGIN_SECONDS = 15 * 60
DEEPSWE_HARNESS = "sandy-cli-separate-verifier"


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
        self.sandy = SandyService()

    def get_name(self) -> str:
        return "deepswe"

    def get_display_name(self) -> str:
        return self.benchmark_spec.display_name

    def requires_setup(self) -> bool:
        return True

    def get_setup_notes(self) -> str | None:
        return (
            "Requires Sandy with Docker socket access; pulls one unique DeepSWE "
            "v1.1 image per item and removes task-specific images after scoring."
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

    async def _verify_workspace_clean(self, sandbox_id: str) -> dict[str, Any]:
        result = await self.sandy.execute_command(
            sandbox_id,
            "answer_count=$(find /workspace/task/tests /workspace/task/solution "
            "-type f 2>/dev/null | wc -l); "
            "archive_count=$(find /workspace -maxdepth 1 -type f "
            "\\( -name '*.tar' -o -name '*.b64' -o -name '*.tar.gz' \\) | wc -l); "
            "echo ANSWERS=$answer_count ARCHIVES=$archive_count",
        )
        stdout = ((result or {}).get("stdout") or "").strip()
        return {
            "clean": (
                (result or {}).get("exit_code") == 0
                and "ANSWERS=0" in stdout
                and "ARCHIVES=0" in stdout
            ),
            "stdout": stdout,
            "result": result,
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
        result = await self.sandy.execute_command(
            sandbox_id,
            f"docker exec {shlex.quote(container_name)} sh -c {shlex.quote(script)}",
            timeout_ms=300000,
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
        await self.sandy.execute_command(
            sandbox_id,
            f"docker exec {shlex.quote(container_name)} sh -c {shlex.quote(self._seal_script())}",
        )
        sandbox_probe = await self.sandy.execute_command(
            sandbox_id,
            f"echo HOSTS=$(grep -c {shlex.quote(self.SEAL_MARKER)} /etc/hosts); "
            "if command -v curl >/dev/null 2>&1; then "
            "echo CURL=yes; echo FETCH=$(curl -s -m 8 -o /dev/null -w '%{http_code}' "
            "https://raw.githubusercontent.com/datacurve-ai/deep-swe/main/README.md "
            "|| echo CURLFAIL); else echo CURL=no; fi",
        )
        container_probe = await self.sandy.execute_command(
            sandbox_id,
            f"echo MODE=$(docker inspect -f '{{{{.HostConfig.NetworkMode}}}}' "
            f"{shlex.quote(container_name)}); "
            f"echo HOSTS=$(docker exec {shlex.quote(container_name)} sh -c "
            f"{shlex.quote(f'grep -c {shlex.quote(self.SEAL_MARKER)} /etc/hosts')})",
        )
        sandbox_stdout = ((sandbox_probe or {}).get("stdout") or "").strip()
        container_stdout = ((container_probe or {}).get("stdout") or "").strip()
        sandbox_blocked = (
            (sandbox_probe or {}).get("exit_code") == 0
            and "HOSTS=0" not in sandbox_stdout
            and "HOSTS=" in sandbox_stdout
            and "CURL=yes" in sandbox_stdout
            and "FETCH=200" not in sandbox_stdout
        )
        container_blocked = (
            (container_probe or {}).get("exit_code") == 0
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

    async def _sandbox_alive(self, sandbox_id: str) -> bool | None:
        return await self.sandy.sandbox_exists(sandbox_id)

    async def _reap_orphans(self, sandbox_id: str) -> None:
        """Remove only DeepSWE resources whose owning Sandy sandbox is gone."""
        await self.sandy.execute_command(
            sandbox_id,
            "live=$(docker ps --format '{{.Names}}' | grep '^sandy_' | sed 's/^sandy_/s/'); "
            "for container in $(docker ps -a --format '{{.Names}}' | grep '^deepswe_s'); do "
            '  namespace=$(echo "$container" | sed -E '
            "'s/^deepswe_(s[0-9a-f]+)_.*/\\1/'); "
            '  if ! echo "$live" | grep -qx "$namespace"; then '
            "    image=$(docker inspect -f '{{.Config.Image}}' \"$container\" 2>/dev/null || true); "
            '    docker rm -f "$container" >/dev/null 2>&1 || true; '
            '    case "$image" in public.ecr.aws/d3j8x8q7/swe-bench-202605:*-v1.1|deepswe_s*) '
            '      docker rmi "$image" >/dev/null 2>&1 || true ;; esac; '
            "  fi; "
            "done; true",
            timeout_ms=300000,
        )

    @staticmethod
    def _resource_flags(item: dict[str, Any]) -> list[str]:
        flags = []
        if item.get("cpus"):
            flags.extend(["--cpus", shlex.quote(str(item["cpus"]))])
        if item.get("memory_mb"):
            flags.extend(["--memory", f"{int(item['memory_mb'])}m"])
        return flags

    async def _start_agent_container(
        self,
        sandbox_id: str,
        item: dict[str, Any],
        container_name: str,
        pull_timeout_sec: int,
    ) -> dict[str, Any]:
        image = item["docker_image"]
        pull = await self.sandy.execute_command(
            sandbox_id,
            f"docker pull {shlex.quote(image)}",
            timeout_ms=pull_timeout_sec * 1000,
        )
        if (pull or {}).get("exit_code") != 0:
            return {"ok": False, "stage": "image_pull", "result": pull}

        digest = await self.sandy.execute_command(
            sandbox_id,
            f"docker image inspect -f '{{{{index .RepoDigests 0}}}}' {shlex.quote(image)}",
        )
        command = " ".join(
            [
                "docker run -d",
                "--name",
                shlex.quote(container_name),
                "--label",
                "chutes.benchmark=deepswe",
                "--network",
                "none",
                "--workdir",
                "/app",
                *self._resource_flags(item),
                shlex.quote(image),
                "sleep infinity",
            ]
        )
        run = await self.sandy.execute_command(sandbox_id, command, timeout_ms=300000)
        if (run or {}).get("exit_code") != 0:
            return {"ok": False, "stage": "container_start", "result": run}
        await self.sandy.execute_command(
            sandbox_id,
            f"docker exec {shlex.quote(container_name)} mkdir -p /logs/agent",
        )
        return {
            "ok": True,
            "pull": pull,
            "run": run,
            "image_digest": ((digest or {}).get("stdout") or "").strip(),
        }

    async def _collect_patch(
        self,
        sandbox_id: str,
        item: dict[str, Any],
        container_name: str,
        timeout_sec: int,
    ) -> dict[str, Any]:
        command = item["collect_command"]
        collect = await self.sandy.execute_command(
            sandbox_id,
            f"docker exec {shlex.quote(container_name)} bash -lc {shlex.quote(command)}",
            timeout_ms=timeout_sec * 1000,
        )
        copy_result: dict[str, Any] = {}
        fallback_result: dict[str, Any] = {}
        if (collect or {}).get("exit_code") == 0:
            copy_result = await self.sandy.execute_command(
                sandbox_id,
                f"docker cp {shlex.quote(container_name)}:/logs/artifacts/model.patch "
                "/workspace/model.patch",
            )
        if (collect or {}).get("exit_code") != 0 or copy_result.get("exit_code") != 0:
            # A CLI that deletes .git, does not commit, or otherwise breaks its
            # own submission path receives a normal zero. Only transport loss
            # is excluded by the caller. Materialize the official no-patch
            # input so the pristine verifier can produce that zero.
            fallback_result = await self.sandy.execute_command(
                sandbox_id, ": > /workspace/model.patch"
            )
        return {"collect": collect, "copy": copy_result, "fallback": fallback_result}

    async def _stage_and_run_verifier(
        self,
        sandbox_id: str,
        item: dict[str, Any],
        verifier_container: str,
        verifier_image: str,
        build_timeout_sec: int,
        verifier_timeout_sec: int,
    ) -> dict[str, Any]:
        staged = await self._upload_archive(
            sandbox_id,
            item["verifier_archive"],
            basename="deepswe-verifier",
            destination="/workspace/verifier",
        )
        if not staged.get("ok"):
            return {"ok": False, "stage": "verifier_upload", "staged": staged}

        verifier_probe = await self.sandy.execute_command(
            sandbox_id,
            "test -f /workspace/verifier/Dockerfile && "
            "test -f /workspace/verifier/test.sh && "
            "test ! -e /workspace/verifier/solution && "
            "test $(find /workspace -maxdepth 1 -type f "
            "\\( -name '*.tar' -o -name '*.b64' -o -name '*.tar.gz' \\) | wc -l) -eq 0",
        )
        if (verifier_probe or {}).get("exit_code") != 0:
            return {
                "ok": False,
                "stage": "verifier_integrity",
                "probe": verifier_probe,
            }

        build = await self.sandy.execute_command(
            sandbox_id,
            f"docker build --network none -t {shlex.quote(verifier_image)} /workspace/verifier",
            timeout_ms=build_timeout_sec * 1000,
        )
        if (build or {}).get("exit_code") != 0:
            return {"ok": False, "stage": "verifier_build", "build": build}

        run_command = " ".join(
            [
                "docker run -d",
                "--name",
                shlex.quote(verifier_container),
                "--label",
                "chutes.benchmark=deepswe",
                "--network",
                "none",
                "--workdir",
                "/app",
                *self._resource_flags(item),
                shlex.quote(verifier_image),
                "sleep infinity",
            ]
        )
        run = await self.sandy.execute_command(sandbox_id, run_command, timeout_ms=300000)
        if (run or {}).get("exit_code") != 0:
            return {"ok": False, "stage": "verifier_start", "run": run}

        verifier_integrity_script = (
            "test -f /tests/test.sh && test ! -e /solution && "
            f'test "$(git -C /app rev-parse HEAD)" = "{item["base_commit_hash"]}"'
        )
        separate_probe = await self.sandy.execute_command(
            sandbox_id,
            f"docker exec {shlex.quote(verifier_container)} sh -c "
            f"{shlex.quote(verifier_integrity_script)} "
            f"&& test \"$(docker inspect -f '{{{{.HostConfig.NetworkMode}}}}' "
            f'{shlex.quote(verifier_container)})" = none',
        )
        if (separate_probe or {}).get("exit_code") != 0:
            return {
                "ok": False,
                "stage": "verifier_integrity",
                "probe": separate_probe,
            }

        prepare = await self.sandy.execute_command(
            sandbox_id,
            f"docker exec {shlex.quote(verifier_container)} mkdir -p "
            "/logs/artifacts /logs/verifier && "
            f"docker cp /workspace/model.patch {shlex.quote(verifier_container)}:"
            "/logs/artifacts/model.patch",
        )
        if (prepare or {}).get("exit_code") != 0:
            return {"ok": False, "stage": "verifier_transfer", "prepare": prepare}

        test = await self.sandy.execute_command(
            sandbox_id,
            f"docker exec -w /app {shlex.quote(verifier_container)} bash /tests/test.sh",
            timeout_ms=verifier_timeout_sec * 1000,
        )
        reward = await self.sandy.execute_command(
            sandbox_id,
            f"docker exec {shlex.quote(verifier_container)} sh -c "
            + shlex.quote(
                "if [ -f /logs/verifier/reward.json ]; then "
                "cat /logs/verifier/reward.json; "
                "elif [ -f /logs/verifier/reward.txt ]; then "
                "cat /logs/verifier/reward.txt; else exit 1; fi"
            ),
        )
        return {
            "ok": True,
            "staged": staged,
            "build": build,
            "run": run,
            "separate_probe": separate_probe,
            "prepare": prepare,
            "test": test,
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
        if not self._items:
            await self.preload()
        item = next((candidate for candidate in self._items if candidate["id"] == item_id), None)
        if item is None:
            return ItemResult(item_id=item_id, error=f"Item {item_id} not found")

        agent_name = self._agent_name()
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
            budget = self._item_budgets_seconds(item)
            outer_timeout = self.get_item_timeout_seconds(item_id)
            sandbox_ttl_min = math.ceil((outer_timeout or 0) / 60)
            sandbox_id = await self.sandy.create_sandbox(
                enable_docker_socket=True,
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

            await self._reap_orphans(sandbox_id)
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

            settings = get_settings()
            api_key = self.client.get_api_key() or settings.chutes_api_key
            agent_result = await self.sandy.run_agent(
                sandbox_id,
                agent=agent_name,
                model=self.model_slug,
                prompt=prompt + f"\nTask container name: {agent_container}\n",
                max_duration=int(budget["agent"]),
                raw_prompt=True,
                env_vars={"CHUTES_API_KEY": api_key},
            )
            agent_usage = await collect_agent_usage(self.sandy, sandbox_id)
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
            await self.sandy.execute_command(
                sandbox_id,
                f"docker stop {shlex.quote(agent_container)} >/dev/null 2>&1 || true",
                timeout_ms=120000,
            )

            verifier = await self._stage_and_run_verifier(
                sandbox_id,
                item,
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
            reward, reward_metrics = self._parse_reward(reward_result.get("stdout") or "")
            if (
                test_result.get("exit_code") not in (0, None)
                or reward_result.get("exit_code") != 0
                or reward not in (0.0, 1.0)
            ):
                return self._excluded_result(
                    item=item,
                    prompt=prompt,
                    reason="infrastructure_verifier",
                    error=(
                        "DeepSWE verifier did not produce a valid binary reward: "
                        f"test={test_result}, reward={reward_result}"
                    ),
                    start_time=start_time,
                    usage=agent_usage,
                    metadata={
                        "agent": agent_name,
                        "agent_summary": agent_summary,
                        "reward_metrics": reward_metrics,
                    },
                )

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
                    "harness": DEEPSWE_HARNESS,
                    "official_leaderboard_harness": "Pier + mini-SWE-agent on Modal",
                    "agent_summary": agent_summary,
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
                    "answer_key_holdout": {
                        "workspace": workspace_clean,
                        "container": container_clean,
                        "full_source_archive_uploaded": False,
                        "solution_uploaded": False,
                    },
                    "separate_verifier": True,
                    "dataset_repository": item["dataset_repository"],
                    "dataset_commit": item["dataset_commit"],
                    "docker_image": image,
                    "docker_image_digest": setup.get("image_digest"),
                    "base_commit_hash": item.get("base_commit_hash"),
                    "reward_metrics": reward_metrics,
                },
            )
        except Exception as exc:
            logger.error("DeepSWE evaluation failed", item_id=item_id, error=str(exc))
            exclusion_reason = classify_deepswe_exception(str(exc), agent_summary)
            if exclusion_reason:
                return self._excluded_result(
                    item=item,
                    prompt=prompt,
                    reason=exclusion_reason,
                    error=str(exc),
                    start_time=start_time,
                    usage=agent_usage,
                    metadata={"agent": agent_name, "agent_summary": agent_summary},
                )
            return ItemResult(
                item_id=item_id,
                item_hash=self.compute_item_hash(item["task_id"]),
                prompt=prompt,
                error=str(exc),
                input_tokens=agent_usage.get("input_tokens"),
                output_tokens=agent_usage.get("output_tokens"),
                metadata={
                    "task_id": item["task_id"],
                    "agent": agent_name,
                    "harness": DEEPSWE_HARNESS,
                    "agent_summary": agent_summary,
                    "agent_usage": agent_usage,
                },
            )
        finally:
            if sandbox_id:
                names = [name for name in (verifier_container, agent_container) if name]
                if names:
                    await self.sandy.execute_command(
                        sandbox_id,
                        "docker rm -f "
                        + " ".join(shlex.quote(name) for name in names)
                        + " >/dev/null 2>&1 || true",
                    )
                if verifier_image:
                    await self.sandy.execute_command(
                        sandbox_id,
                        f"docker rmi {shlex.quote(verifier_image)} >/dev/null 2>&1 || true",
                    )
                if image:
                    # Do not force-remove a shared tag: another paired arm may
                    # be using it. Docker removes it once no live/stopped task
                    # container references it.
                    await self.sandy.execute_command(
                        sandbox_id,
                        f"docker rmi {shlex.quote(image)} >/dev/null 2>&1 || true",
                    )
                await self.sandy.terminate_sandbox(sandbox_id)
