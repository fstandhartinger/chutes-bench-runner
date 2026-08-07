"""Version-pinned Terminal-Bench benchmark adapters."""
from __future__ import annotations

import asyncio
import base64
import copy
import hashlib
import io
import os
import shlex
import tarfile
import tempfile
import time
import tomllib
from collections.abc import AsyncIterator
from pathlib import Path, PurePosixPath
from typing import Any, Optional

import httpx
import yaml

from app.benchmarks.adapters.terminal_bench_identity import (
    TERMINAL_BENCH_1,
    TERMINAL_BENCH_2_0,
    TERMINAL_BENCH_2_1,
    TERMINAL_BENCH_HARD,
    TerminalBenchSpec,
)
from app.benchmarks.adapters.terminal_bench_scoring import (
    FUNCTIONAL,
    NON_FUNCTIONAL_SCORING_CLASSES,
    PERFORMANCE_GATED,
    RESOURCE_GATED,
    TERMINAL_BENCH_2_1_SCORING_AUDIT_COMMIT,
    TERMINAL_BENCH_2_1_SCORING_AUDIT_TASK_COUNT,
    terminal_bench_2_1_scoring_classification,
)
from app.benchmarks.agent_provider_config import (
    prepare_sandy_agent_launch,
    retain_sandy_agent_rollout,
    validate_openrouter_agent_usage,
)
from app.benchmarks.agent_evidence import retain_agent_evidence
from app.benchmarks.agent_usage import collect_agent_usage
from app.benchmarks.base import BenchmarkAdapter, ItemResult
from app.benchmarks.registry import register_adapter
from app.core.logging import get_logger
from app.services.sandy_service import SandyService

logger = get_logger(__name__)

TERMINAL_BENCH_ITEM_TIMEOUT_MARGIN_SECONDS = 15 * 60
TERMINAL_BENCH_CAPABILITY_FILTER_OPTION = "exclude_performance_and_resource_gated_tasks"



def classify_agent_exit(
    agent_summary: dict,
    agent_timeout_sec: float,
    sandbox_alive: Optional[bool],
) -> tuple[Optional[str], Optional[str]]:
    """Separate an infrastructure kill from a harness crash. They are not the same.

    An agent that exits non-zero well before its budget did not run out of
    time -- something ended it. But "something" splits two ways, and only one
    of them is ours:

      * the sandbox was reaped underneath it (Sandy's TTL). Infrastructure,
        nothing to do with the agent, and scoring it 0 would be recording our
        own bug as a capability failure. Excluded.

      * the agent process died while the sandbox was still alive -- kernel
        boot failure, a bad Python cell taking the turn down, the RLM sidecar
        dying. That is a **robustness property of the harness under test**.
        Excluding it would systematically flatter whichever arm crashes more,
        and the RLM arm has strictly more machinery to crash. Scored, not
        excluded.

    The distinction is checkable rather than a judgement call: the sandbox is
    queryable at the moment of the exit, so we ask it.

    Returns (exclusion_reason, note). exclusion_reason is None when the item
    should still be scored; note carries the classification either way.
    """
    if not agent_summary:
        return None, None
    if agent_summary.get("exitCode") in (0, None):
        return None, None
    duration = agent_summary.get("duration")
    if duration is None or not agent_timeout_sec:
        return None, None
    if duration >= 0.9 * agent_timeout_sec:
        return None, "agent_exhausted_budget"

    detail = (
        f"exit {agent_summary.get('exitCode')} after {duration:.0f}s against a "
        f"{agent_timeout_sec:.0f}s budget ({duration / agent_timeout_sec:.0%})"
    )
    if sandbox_alive is False:
        return (
            "infrastructure_sandbox_gone",
            f"Sandbox was gone at exit -- {detail}. Excluded: this is our "
            f"infrastructure, not the agent.",
        )
    # Sandbox alive (or we could not tell -- in which case do NOT exclude,
    # because the failure mode of a wrong guess here is silently dropping
    # harness crashes from the arm that produces them).
    return (
        None,
        f"Agent process died while the sandbox was still alive -- {detail}. "
        f"Scored: this is a robustness property of the harness, not infrastructure.",
    )


# Transport-level failures between bench-runner and Sandy. These never reach
# classify_agent_exit, because that function guards the case where the agent
# ran and exited -- here the *request* broke, so there is no agent_summary, no
# duration and no exit code to classify. The exclusion logic had a hole exactly
# where the most common failure now lives, and these landed as score 0 with no
# exclusion_reason: infrastructure recorded as capability, again.
TRANSPORT_FAILURE_MARKERS = (
    "peer closed connection",
    "incomplete chunked read",
    "server disconnected",
    "connection reset",
    "remote protocol error",
    "readtimeout",
    "read timeout",
    "connecttimeout",
    "all connection attempts failed",
    "sandbox not found",
)

# These are verifier-side connectivity failures, not failures of the agent's
# solution. Keep this deliberately narrower than generic strings such as
# "failed to download": a missing artifact or a bad URL can be a real task
# failure, while these signatures state that the network path itself failed.
VERIFIER_NETWORK_FAILURE_MARKERS = (
    "curl: (5) could not resolve proxy",
    "curl: (6) could not resolve host",
    "curl: (7) failed to connect",
    "curl: (28) connection timed out",
    "temporary failure in name resolution",
    "network is unreachable",
    "no route to host",
)

VERIFIER_NETWORK_EXCLUSION_REASON = "infrastructure_verifier_network"
VERIFIER_NOT_EXECUTED_EXCLUSION_REASON = "infrastructure_verifier_not_executed"
AGENT_NOT_TERMINATED_EXCLUSION_REASON = "infrastructure_agent_not_terminated"
AGENT_LAUNCH_FAILED_EXCLUSION_REASON = "infrastructure_agent_launch_failed"

# httpx includes one of these prefixes in HTTPStatusError messages. Restrict
# the match to Sandy's agent endpoint so an unrelated 4xx/5xx later in an item
# does not get mislabeled as a launch failure.
AGENT_RUN_HTTP_FAILURE_MARKERS = (
    "redirect response",
    "client error",
    "server error",
)
UNKNOWN_AGENT_FAILURE_MARKERS = (
    "unknown agent",
    "unregistered agent",
    "unsupported agent",
    "invalid agent",
)


def classify_agent_launch_failure(
    error_text: str,
    agent_summary: Optional[dict],
    agent_usage: Optional[dict],
    *,
    agent_invoked: bool,
) -> Optional[str]:
    """Exclude requests where there is no evidence that an agent ran.

    A non-2xx from Sandy's agent-run endpoint is a launch/configuration
    failure, not a capability result. The same is true when the request
    returns without a completion summary, token counts, or a rollout. Requiring
    all three pieces of evidence to be absent keeps real agent failures scored.
    """
    if agent_summary:
        return None

    text = (error_text or "").lower()
    if (
        "/agent/run" in text
        and any(marker in text for marker in AGENT_RUN_HTTP_FAILURE_MARKERS)
    ) or any(marker in text for marker in UNKNOWN_AGENT_FAILURE_MARKERS):
        return AGENT_LAUNCH_FAILED_EXCLUSION_REASON

    usage = agent_usage or {}
    has_tokens = any(
        isinstance(usage.get(field), int)
        for field in ("input_tokens", "output_tokens")
    )
    has_rollout = any(
        usage.get(field) not in (None, "")
        for field in ("rollout", "session", "usage_source")
    )
    if agent_invoked and not has_tokens and not has_rollout:
        return AGENT_LAUNCH_FAILED_EXCLUSION_REASON
    return None


def classify_verifier_network_failure(test_result: Optional[dict]) -> Optional[str]:
    """Identify proof that verifier tooling could not reach the network."""
    result = test_result or {}
    text = "\n".join(
        str(result.get(field) or "") for field in ("stdout", "stderr", "error")
    ).lower()
    if any(marker in text for marker in VERIFIER_NETWORK_FAILURE_MARKERS):
        return VERIFIER_NETWORK_EXCLUSION_REASON
    return None


def classify_bare_failure(
    error_text: str,
    agent_summary: Optional[dict],
    agent_usage: Optional[dict] = None,
    *,
    agent_invoked: bool = False,
) -> Optional[str]:
    """Exclusion reason for a failure that produced no agent summary.

    If the agent never reported at all, we cannot say the harness failed --
    we can only say we never heard back. Scoring that 0 penalises whichever
    arm happened to be running when the transport broke.
    """
    if agent_summary:
        return None
    text = (error_text or "").lower()
    launch_failure = classify_agent_launch_failure(
        error_text,
        agent_summary,
        agent_usage,
        agent_invoked=False,
    )
    if launch_failure:
        return launch_failure
    if any(marker in text for marker in TRANSPORT_FAILURE_MARKERS):
        return "infrastructure_transport"
    return classify_agent_launch_failure(
        error_text,
        agent_summary,
        agent_usage,
        agent_invoked=agent_invoked,
    )


def settings_allow_unsealed() -> bool:
    """Escape hatch for deliberately running Terminal-Bench with open egress.

    Off by default on purpose: the failure mode of an unsealed run is a
    plausible-looking score, not an error, so it has to be opted into.
    """
    return (os.getenv("TERMINAL_BENCH_ALLOW_UNSEALED") or "").strip().lower() in {
        "1",
        "true",
        "yes",
    }


class BenchmarkIdentityError(RuntimeError):
    """The requested benchmark identity cannot be reproduced exactly."""


class TerminalBenchBaseAdapter(BenchmarkAdapter):
    """
    Shared evaluator for pinned Terminal-Bench releases.

    Legacy Terminal-Bench 1.0 tasks and Harbor-format 2.x tasks use different
    execution protocols. The containment, answer-key holdout, usage recording,
    and infrastructure-exclusion safeguards below are shared by both.
    """

    benchmark_name = ""
    benchmark_spec: TerminalBenchSpec

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._items: list[dict[str, Any]] = []
        self._item_observability: dict[str, dict[str, Any]] = {}
        self._active_sandbox_ids: set[str] = set()
        self._scoring_policy_report: Optional[dict[str, Any]] = None
        self.sandy = SandyService()

    def get_name(self) -> str:
        return self.benchmark_name

    def get_display_name(self) -> str:
        return self.benchmark_spec.display_name

    def requires_setup(self) -> bool:
        return True

    def get_setup_notes(self) -> Optional[str]:
        return "Requires Sandy with Docker socket access."

    def supports_subset(self) -> bool:
        return True

    @staticmethod
    def _item_budgets_ms(item: dict[str, Any]) -> tuple[float, float, int, int]:
        """Return the exact agent and verifier budgets used by evaluation."""
        timeout_multiplier = float(
            os.getenv("TERMINAL_BENCH_AGENT_TIMEOUT_MULTIPLIER") or "1.0"
        )
        base_agent_timeout_sec = item.get("max_agent_timeout_sec") or 180
        agent_timeout_ms = int(base_agent_timeout_sec * timeout_multiplier * 1000)
        test_timeout_ms = int((item.get("max_test_timeout_sec") or 300) * 1000)
        return (
            base_agent_timeout_sec,
            timeout_multiplier,
            agent_timeout_ms,
            test_timeout_ms,
        )

    def get_item_timeout_seconds(self, item_id: Optional[str] = None) -> Optional[int]:
        """Cover the declared agent and verifier budgets plus harness overhead.

        The worker owns this outer deadline. It must not expire before the
        inner Terminal-Bench budgets that it is supervising.
        """
        items = self._items
        if item_id is not None:
            item = next((candidate for candidate in items if candidate["id"] == item_id), None)
            if item is None:
                return None
            items = [item]
        if not items:
            return None

        item_timeouts = []
        for item in items:
            _, _, agent_timeout_ms, test_timeout_ms = self._item_budgets_ms(item)
            # run_agent enforces a 60-second minimum even if a malformed task
            # declares less, so the outer deadline must cover that same floor.
            agent_timeout_seconds = max(60, (agent_timeout_ms + 999) // 1000)
            test_timeout_seconds = (test_timeout_ms + 999) // 1000
            item_timeouts.append(
                agent_timeout_seconds
                + test_timeout_seconds
                + TERMINAL_BENCH_ITEM_TIMEOUT_MARGIN_SECONDS
            )
        return max(item_timeouts)

    async def get_total_items(self) -> int:
        if not self._items:
            await self.preload()
        return len(self._items)

    def _terminal_bench_config(self) -> dict[str, Any]:
        """Merge family and concrete run config without losing versioned keys."""
        config = getattr(self, "run_config", None) or {}
        merged: dict[str, Any] = {}
        keys = (
            "terminal_bench",
            self.get_name().rsplit("_", 1)[0],
            self.get_name(),
        )
        for key in dict.fromkeys(keys):
            candidate = config.get(key) or {}
            if isinstance(candidate, dict):
                merged.update(candidate)
        return merged

    def _has_scoring_classification_audit(self) -> bool:
        return self.benchmark_spec.commit == TERMINAL_BENCH_2_1_SCORING_AUDIT_COMMIT

    def _classification_for_task(self, task_id: str) -> Optional[dict[str, Any]]:
        if not self._has_scoring_classification_audit():
            return None
        return terminal_bench_2_1_scoring_classification(task_id)

    def _build_scoring_policy_report(
        self,
        selected_before_exclusion: list[str],
        *,
        capability_filter_enabled: bool,
    ) -> tuple[dict[str, Any], list[str]]:
        items_by_id = {item["id"]: item for item in self._items}
        class_counts = {
            FUNCTIONAL: 0,
            PERFORMANCE_GATED: 0,
            RESOURCE_GATED: 0,
        }
        for item in self._items:
            scoring_class = item.get("scoring_class")
            if scoring_class in class_counts:
                class_counts[scoring_class] += 1

        selected_gated = [
            item_id
            for item_id in selected_before_exclusion
            if items_by_id[item_id].get("scoring_class") in NON_FUNCTIONAL_SCORING_CLASSES
        ]
        excluded_ids = selected_gated if capability_filter_enabled else []
        excluded_set = set(excluded_ids)
        selected_after_exclusion = [
            item_id for item_id in selected_before_exclusion if item_id not in excluded_set
        ]

        excluded_by_class = {
            PERFORMANCE_GATED: sum(
                items_by_id[item_id].get("scoring_class") == PERFORMANCE_GATED
                for item_id in excluded_ids
            ),
            RESOURCE_GATED: sum(
                items_by_id[item_id].get("scoring_class") == RESOURCE_GATED
                for item_id in excluded_ids
            ),
        }
        excluded_tasks = [
            {
                "item_id": item_id,
                "task_id": items_by_id[item_id].get("task_id"),
                "scoring_class": items_by_id[item_id].get("scoring_class"),
                "reason": items_by_id[item_id].get("scoring_reason"),
            }
            for item_id in excluded_ids
        ]
        classified_gated_count = class_counts[PERFORMANCE_GATED] + class_counts[RESOURCE_GATED]
        if capability_filter_enabled:
            summary = (
                "NON-STANDARD CAPABILITY-ONLY SCORE: excluded "
                f"{len(excluded_ids)} of {len(selected_before_exclusion)} selected "
                "tasks because their upstream verifiers gate on performance or "
                "resources; verifier thresholds were not changed."
            )
        else:
            summary = (
                "STANDARD TASK SELECTION: excluded 0 tasks by scoring policy; "
                f"the full {len(self._items)}-task benchmark contains "
                f"{classified_gated_count} "
                "performance/resource-gated tasks."
            )

        report = {
            "audit_commit": TERMINAL_BENCH_2_1_SCORING_AUDIT_COMMIT,
            "audit_applies": self._has_scoring_classification_audit(),
            "mode": ("capability_only" if capability_filter_enabled else "standard"),
            "standard_terminal_bench_score": not capability_filter_enabled,
            "thresholds_relaxed": False,
            "option": TERMINAL_BENCH_CAPABILITY_FILTER_OPTION,
            "option_enabled": capability_filter_enabled,
            "benchmark_task_count": len(self._items),
            "scoring_class_counts": class_counts,
            "classified_gated_task_count": classified_gated_count,
            "selected_before_policy_exclusion": len(selected_before_exclusion),
            "selected_after_policy_exclusion": len(selected_after_exclusion),
            "selected_gated_task_count": len(selected_gated),
            "excluded_task_count": len(excluded_ids),
            "excluded_by_class": excluded_by_class,
            "excluded_tasks": excluded_tasks,
            "summary": summary,
        }
        return report, selected_after_exclusion

    async def get_items_for_evaluation(
        self,
        subset_pct: int,
        seed: str,
        subset_count: Optional[int] = None,
    ) -> tuple[int, list[str]]:
        total_items, selected = await super().get_items_for_evaluation(
            subset_pct,
            seed,
            subset_count,
        )
        option_value = self._terminal_bench_config().get(
            TERMINAL_BENCH_CAPABILITY_FILTER_OPTION,
            False,
        )
        if not isinstance(option_value, bool):
            raise ValueError(f"{TERMINAL_BENCH_CAPABILITY_FILTER_OPTION} must be a JSON boolean")
        if option_value and not self._has_scoring_classification_audit():
            raise ValueError(
                f"{TERMINAL_BENCH_CAPABILITY_FILTER_OPTION} is only available for "
                "the audited Terminal-Bench 2.1 release"
            )

        if self._has_scoring_classification_audit():
            report, filtered = self._build_scoring_policy_report(
                selected,
                capability_filter_enabled=option_value,
            )
            self._scoring_policy_report = report
            if option_value:
                logger.warning(
                    "Terminal-Bench capability-only task filter enabled",
                    benchmark=self.get_name(),
                    excluded_tasks=report["excluded_task_count"],
                    selected_before=report["selected_before_policy_exclusion"],
                    selected_after=report["selected_after_policy_exclusion"],
                    standard_terminal_bench_score=False,
                )
                if not filtered:
                    task_ids = [entry["task_id"] for entry in report["excluded_tasks"]]
                    raise ValueError(
                        "Capability-only policy excluded every selected task "
                        f"({', '.join(task_ids)}); no score can be computed"
                    )
            return total_items, filtered

        self._scoring_policy_report = None
        return total_items, selected

    async def postprocess(self, results: list[ItemResult]) -> dict[str, Any]:
        """Put the selection policy beside the score in run-level metrics."""
        report = getattr(self, "_scoring_policy_report", None)
        if report is None:
            return {}
        return {"terminal_bench_scoring_policy": copy.deepcopy(report)}

    async def preload(self) -> None:
        """Load an exact, pinned upstream release and assert its identity."""
        if self._items:
            return

        try:
            logger.info(
                "Loading pinned Terminal-Bench dataset",
                benchmark=self.get_name(),
                repository=self.benchmark_spec.repository,
                commit=self.benchmark_spec.commit,
                expected_count=self.benchmark_spec.expected_count,
            )
            source_archive = await self._load_source_archive()
            self._items = await asyncio.to_thread(
                self._items_from_source_archive, source_archive
            )
            self._assert_benchmark_identity()
            logger.info(
                "Loaded pinned Terminal-Bench dataset",
                benchmark=self.get_name(),
                item_count=len(self._items),
                commit=self.benchmark_spec.commit,
            )
        except Exception as e:
            logger.error(
                "Failed to load Terminal-Bench",
                benchmark=self.get_name(),
                error=str(e),
            )
            self._items = []
            raise

    def _cache_path(self) -> Path:
        configured = os.getenv("TERMINAL_BENCH_DATASET_CACHE")
        cache_root = Path(configured) if configured else Path(tempfile.gettempdir()) / "chutes-bench-runner"
        repo_name = self.benchmark_spec.repository.rsplit("/", 1)[-1]
        return cache_root / f"{repo_name}-{self.benchmark_spec.commit}.tar.gz"

    async def _load_source_archive(self) -> bytes:
        """Download and cache the immutable upstream Git archive."""
        cache_path = self._cache_path()
        if cache_path.is_file():
            cached = await asyncio.to_thread(cache_path.read_bytes)
            if hashlib.sha256(cached).hexdigest() == self.benchmark_spec.archive_sha256:
                return cached
            logger.warning(
                "Discarding Terminal-Bench cache with wrong SHA-256",
                path=str(cache_path),
            )
            await asyncio.to_thread(cache_path.unlink)

        async with httpx.AsyncClient(follow_redirects=True, timeout=180.0) as client:
            response = await client.get(self.benchmark_spec.archive_url)
            response.raise_for_status()
            archive = response.content

        actual_sha256 = hashlib.sha256(archive).hexdigest()
        if actual_sha256 != self.benchmark_spec.archive_sha256:
            raise BenchmarkIdentityError(
                f"{self.get_name()} source archive SHA-256 mismatch: "
                f"expected {self.benchmark_spec.archive_sha256}, got {actual_sha256}"
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

    def _items_from_source_archive(self, archive: bytes) -> list[dict[str, Any]]:
        """Package each canonical upstream task as the evaluator's task archive."""
        members_by_task: dict[str, list[tuple[tarfile.TarInfo, str]]] = {
            task_id: [] for task_id in self.benchmark_spec.task_ids
        }
        metadata_markers: set[str] = set()
        root_parts = PurePosixPath(self.benchmark_spec.task_root).parts

        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as source:
            for member in source.getmembers():
                parts = PurePosixPath(member.name).parts
                if len(parts) < 2:
                    continue
                relative_parts = parts[1:]  # GitHub's generated top-level directory.
                if root_parts:
                    if relative_parts[: len(root_parts)] != root_parts:
                        continue
                    relative_parts = relative_parts[len(root_parts) :]
                if len(relative_parts) < 2:
                    continue

                task_id = relative_parts[0]
                task_relative = PurePosixPath(*relative_parts[1:])
                if task_relative.is_absolute() or ".." in task_relative.parts:
                    raise BenchmarkIdentityError(
                        f"Unsafe path in {self.get_name()} source archive: {member.name}"
                    )
                if task_id not in members_by_task:
                    continue
                if member.issym() or member.islnk():
                    link = PurePosixPath(member.linkname)
                    if link.is_absolute() or ".." in link.parts:
                        raise BenchmarkIdentityError(
                            f"Unsafe link in {self.get_name()} source archive: {member.name}"
                        )
                relative_name = str(task_relative)
                members_by_task[task_id].append((member, relative_name))
                expected_marker = (
                    "task.yaml" if self.benchmark_spec.task_format == "legacy" else "task.toml"
                )
                if relative_name == expected_marker:
                    metadata_markers.add(task_id)

            missing = set(self.benchmark_spec.task_ids) - metadata_markers
            if missing:
                raise BenchmarkIdentityError(
                    f"{self.get_name()} source is missing {len(missing)} canonical tasks: "
                    f"{', '.join(sorted(missing))}"
                )

            items: list[dict[str, Any]] = []
            for index, task_id in enumerate(self.benchmark_spec.task_ids):
                task_archive = io.BytesIO()
                file_contents: dict[str, bytes] = {}
                with tarfile.open(fileobj=task_archive, mode="w") as task_tar:
                    for member, relative_name in members_by_task[task_id]:
                        copied = copy.copy(member)
                        copied.name = relative_name
                        extracted = source.extractfile(member) if member.isfile() else None
                        if extracted is not None:
                            content = extracted.read()
                            file_contents[relative_name] = content
                            task_tar.addfile(copied, io.BytesIO(content))
                        else:
                            task_tar.addfile(copied)
                items.append(
                    self._make_item(index, task_id, task_archive.getvalue(), file_contents)
                )
        return items

    def _make_item(
        self,
        index: int,
        task_id: str,
        archive: bytes,
        files: dict[str, bytes],
    ) -> dict[str, Any]:
        if self.benchmark_spec.task_format == "legacy":
            task_yaml = files["task.yaml"].decode("utf-8")
            parsed = yaml.safe_load(task_yaml) or {}
            instruction = parsed.get("instruction", "")
            environment: dict[str, Any] = {}
            task_manifest = task_yaml
            agent_timeout = parsed.get("max_agent_timeout_sec")
            test_timeout = parsed.get("max_test_timeout_sec")
        else:
            task_manifest = files["task.toml"].decode("utf-8")
            parsed = tomllib.loads(task_manifest)
            instruction = files["instruction.md"].decode("utf-8")
            environment = parsed.get("environment") or {}
            agent_timeout = (parsed.get("agent") or {}).get("timeout_sec")
            test_timeout = (parsed.get("verifier") or {}).get("timeout_sec")

        item = {
            "id": str(index),
            "task_id": task_id,
            "task_yaml": task_manifest,
            "instruction": instruction,
            "archive": archive,
            "difficulty": (parsed.get("metadata") or parsed).get("difficulty", ""),
            "parsed_yaml": parsed,
            "max_agent_timeout_sec": agent_timeout,
            "max_test_timeout_sec": test_timeout,
            "task_format": self.benchmark_spec.task_format,
            "docker_image": environment.get("docker_image"),
            "cpus": environment.get("cpus"),
            "memory_mb": environment.get("memory_mb"),
            "dataset_repository": self.benchmark_spec.repository,
            "dataset_commit": self.benchmark_spec.commit,
            "manifest_repository": self.benchmark_spec.manifest_repository,
            "manifest_commit": self.benchmark_spec.manifest_commit,
        }
        classification = self._classification_for_task(task_id)
        if classification is not None:
            item.update(
                {
                    "scoring_class": classification["scoring_class"],
                    "scoring_reason": classification["reason"],
                    "scoring_evidence": classification["evidence"],
                }
            )
        return item

    def _assert_benchmark_identity(self) -> None:
        """Fail startup if count, ordering, uniqueness, or membership drifted."""
        expected = list(self.benchmark_spec.task_ids)
        loaded = [item.get("task_id") for item in self._items]
        if len(self._items) != self.benchmark_spec.expected_count:
            raise BenchmarkIdentityError(
                f"{self.get_name()} identity check failed: expected "
                f"{self.benchmark_spec.expected_count} items, loaded {len(self._items)}"
            )
        if len(set(loaded)) != len(loaded):
            raise BenchmarkIdentityError(
                f"{self.get_name()} identity check failed: duplicate task IDs loaded"
            )
        if loaded != expected:
            missing = sorted(set(expected) - set(loaded))
            unexpected = sorted(set(loaded) - set(expected))
            raise BenchmarkIdentityError(
                f"{self.get_name()} task manifest mismatch; "
                f"missing={missing}, unexpected={unexpected}"
            )
        if self._has_scoring_classification_audit():
            if len(self._items) != TERMINAL_BENCH_2_1_SCORING_AUDIT_TASK_COUNT:
                raise BenchmarkIdentityError(
                    "Terminal-Bench 2.1 scoring audit is tied to "
                    f"{TERMINAL_BENCH_2_1_SCORING_AUDIT_TASK_COUNT} tasks, "
                    f"loaded {len(self._items)}"
                )
            invalid = [
                item.get("task_id")
                for item in self._items
                if item.get("scoring_class") not in {FUNCTIONAL, PERFORMANCE_GATED, RESOURCE_GATED}
            ]
            if invalid:
                raise BenchmarkIdentityError(
                    "Terminal-Bench 2.1 scoring classification is missing or "
                    f"invalid for: {', '.join(invalid)}"
                )

    async def enumerate_items(self) -> AsyncIterator[str]:
        if not self._items:
            await self.preload()
        for item in self._items:
            yield item["id"]

    async def _extract_archive(self, sandbox_id: str, archive_bytes: bytes) -> bool:
        encoded = base64.b64encode(archive_bytes).decode("ascii")
        if not await self.sandy.write_file(sandbox_id, "archive.b64", encoded):
            return False
        result = await self.sandy.execute_command(
            sandbox_id,
            "base64 -d archive.b64 > archive.tar",
        )
        if result.get("exit_code") != 0:
            return False
        result = await self.sandy.execute_command(
            sandbox_id,
            "mkdir -p task && tar -xf archive.tar -C task",
        )
        return result.get("exit_code") == 0

    def _add_compose_build_context(self, content: str) -> str:
        lines = content.splitlines()
        updated: list[str] = []
        i = 0
        while i < len(lines):
            line = lines[i]
            updated.append(line)
            if line.strip() == "build:":
                indent = len(line) - len(line.lstrip())
                j = i + 1
                has_context = False
                while j < len(lines):
                    candidate = lines[j]
                    if candidate.strip() == "":
                        j += 1
                        continue
                    candidate_indent = len(candidate) - len(candidate.lstrip())
                    if candidate_indent <= indent:
                        break
                    if candidate.strip().startswith("context:"):
                        has_context = True
                        break
                    j += 1
                if not has_context:
                    updated.append(" " * (indent + 2) + "context: .")
            i += 1
        return "\n".join(updated) + ("\n" if content.endswith("\n") else "")

    async def _ensure_compose_context(
        self, sandbox_id: str, compose_path: str, cwd: Optional[str] = None
    ) -> None:
        read_result = await self.sandy.execute_command(
            sandbox_id, f"cat {compose_path}", cwd=cwd
        )
        if read_result.get("exit_code") != 0:
            return
        content = read_result.get("stdout") or ""
        if not content:
            return
        patched = self._add_compose_build_context(content)
        if patched != content:
            target_path = compose_path
            if cwd:
                target_path = f"{cwd.rstrip('/')}/{compose_path}"
            await self.sandy.write_file(sandbox_id, target_path, patched)

    async def _reap_orphans(self, sandbox_id: str) -> None:
        """Remove task containers/images whose sandbox is already gone.

        Cleanup normally runs `compose down` *inside* the sandbox, so if the
        sandbox dies first -- timeout, eviction, a failed item -- its task
        containers and images are stranded on the shared host daemon with
        nothing left to remove them. Observed five orphans accumulating across
        a handful of runs on a host that sits at ~92% disk.

        Safe because names are namespaced: anything whose `sN` prefix has no
        live `sandy_N` container cannot belong to a running item.
        """
        await self.sandy.execute_command(
            sandbox_id,
            "live=$(docker ps --format '{{.Names}}' | grep '^sandy_' | sed 's/^sandy_/s/'); "
            "for c in $(docker ps -a --format '{{.Names}}' | grep '^tbench_s'); do "
            "  ns=$(echo \"$c\" | sed -E 's/^tbench_(s[0-9a-f]+)_.*/\\1/'); "
            "  echo \"$live\" | grep -qx \"$ns\" || docker rm -f \"$c\" >/dev/null 2>&1; "
            "done; "
            "for i in $(docker images --format '{{.Repository}}:{{.Tag}}' | grep -E '^(client_s|tbench_s)'); do "
            "  docker rmi \"$i\" >/dev/null 2>&1 || true; "
            "done; true",
        )

    async def _sandbox_identity(self, sandbox_id: str) -> Optional[tuple[str, str]]:
        """Read the exact Sandy id/owner labels from this sandbox container."""
        sandbox_name = f"sandy_{sandbox_id}"
        result = await self.sandy.execute_command(
            sandbox_id,
            "docker inspect -f "
            "'{{ index .Config.Labels \"sandy.id\" }}|"
            "{{ index .Config.Labels \"sandy.owner\" }}' "
            f"{shlex.quote(sandbox_name)}",
        )
        if result.get("exit_code") != 0:
            return None
        raw = (result.get("stdout") or "").strip()
        actual_id, separator, owner = raw.partition("|")
        if not separator or actual_id != sandbox_id or not owner:
            return None
        return actual_id, owner

    async def _task_container_label_flags(self, sandbox_id: str) -> str:
        identity = await self._sandbox_identity(sandbox_id)
        if not identity:
            logger.warning(
                "Could not verify Sandy ownership labels for task container",
                sandbox_id=sandbox_id,
            )
            return ""
        _, owner = identity
        labels = (
            f"sandy.owner={owner}",
            f"chutes.bench.sandbox_id={sandbox_id}",
        )
        return " ".join(f"--label {shlex.quote(label)}" for label in labels)

    async def _cleanup_owned_task_containers(self, sandbox_id: str) -> bool:
        """Remove only task containers proven to belong to ``sandbox_id``.

        Every new task container copies the Sandy owner's label and carries the
        full sandbox id. Re-reading those labels before deletion prevents a
        cancellation from touching another live run on the shared Docker host.
        """
        identity = await self._sandbox_identity(sandbox_id)
        if not identity:
            logger.warning(
                "Skipping task-container sweep because sandbox ownership is unverified",
                sandbox_id=sandbox_id,
            )
            return False
        _, owner = identity
        owner_filter = shlex.quote(f"label=sandy.owner={owner}")
        sandbox_filter = shlex.quote(f"label=chutes.bench.sandbox_id={sandbox_id}")
        quoted_owner = shlex.quote(owner)
        quoted_sandbox_id = shlex.quote(sandbox_id)
        namespace = f"s{sandbox_id[:12]}".lower()
        command = (
            f"for c in $(docker ps -aq --filter {owner_filter} --filter {sandbox_filter}); do "
            "  test \"$(docker inspect -f '{{ index .Config.Labels \"sandy.owner\" }}' \"$c\")\" "
            f"    = {quoted_owner} || continue; "
            "  test \"$(docker inspect -f '{{ index .Config.Labels \"chutes.bench.sandbox_id\" }}' \"$c\")\" "
            f"    = {quoted_sandbox_id} || continue; "
            "  docker rm -f \"$c\" >/dev/null 2>&1 || true; "
            "done; "
            f"for image in $(docker images --format '{{{{.Repository}}}}:{{{{.Tag}}}}'); do "
            f"  case \"$image\" in tbench_{namespace}_*|client_{namespace}:*) "
            "    docker rmi -f \"$image\" >/dev/null 2>&1 || true ;; "
            "  esac; "
            "done; true"
        )
        result = await self.sandy.execute_command(sandbox_id, command, timeout_ms=300_000)
        return result.get("exit_code") == 0

    async def cleanup(self) -> None:
        """Terminate any item sandbox still active when a run is canceled."""
        active_sandbox_ids = list(getattr(self, "_active_sandbox_ids", set()))
        for sandbox_id in active_sandbox_ids:
            try:
                await self._cleanup_owned_task_containers(sandbox_id)
            finally:
                await self.sandy.terminate_sandbox(sandbox_id)
                self._active_sandbox_ids.discard(sandbox_id)

    async def _run_harbor_task(
        self, sandbox_id: str, item: dict[str, Any]
    ) -> dict[str, Any]:
        """Start a Harbor-format 2.x task from its release-pinned image."""
        task_id = item.get("task_id") or "task"
        source_image = item.get("docker_image")
        if not source_image:
            return {"error": f"Harbor task {task_id} has no docker_image"}

        ns = f"s{sandbox_id[:12]}".lower()
        container_name = f"tbench_{ns}_{task_id}_client".lower()
        pull_result = await self.sandy.execute_command(
            sandbox_id,
            f"docker pull {shlex.quote(source_image)}",
            timeout_ms=900000,
        )
        if pull_result.get("exit_code") != 0:
            return {
                "error": pull_result.get("stderr")
                or pull_result.get("stdout")
                or pull_result.get("error")
                or f"Could not pull {source_image}",
                "pull": pull_result,
            }

        resource_flags: list[str] = []
        if item.get("cpus"):
            resource_flags.extend(["--cpus", shlex.quote(str(item["cpus"]))])
        if item.get("memory_mb"):
            resource_flags.extend(["--memory", f"{int(item['memory_mb'])}m"])
        label_flags = await self._task_container_label_flags(sandbox_id)
        run_command = " ".join(
            [
                "docker run -d",
                "--name",
                shlex.quote(container_name),
                label_flags,
                *resource_flags,
                shlex.quote(source_image),
                "sleep infinity",
            ]
        )
        run_result = await self.sandy.execute_command(
            sandbox_id,
            run_command,
            timeout_ms=300000,
        )
        if run_result.get("exit_code") != 0:
            return {
                "error": run_result.get("stderr")
                or run_result.get("stdout")
                or run_result.get("error"),
                "run": run_result,
            }
        await self.sandy.execute_command(
            sandbox_id,
            f"docker exec {shlex.quote(container_name)} mkdir -p /logs/agent /logs/verifier",
        )
        return {
            "container_name": container_name,
            "cleanup_cmd": f"docker rm -f {shlex.quote(container_name)}",
            "cleanup_cwd": None,
            # Official release images are shared across tasks/runs and should
            # remain cached. Only legacy per-sandbox build images are reaped.
            "image_name": None,
            "namespace": ns,
            "source_image": source_image,
            "cleanup_images": False,
        }

    async def _run_terminal_bench(
        self, sandbox_id: str, item: dict[str, Any]
    ) -> dict[str, Any]:
        if item.get("task_format") == "harbor":
            return await self._run_harbor_task(sandbox_id, item)

        task_id = item.get("task_id") or "task"
        task_dir = "/workspace/task"
        compose_path = "docker-compose.yaml"
        compose_check = await self.sandy.execute_command(
            sandbox_id,
            f"test -f {compose_path}",
            cwd=task_dir,
        )
        if compose_check.get("exit_code") != 0:
            compose_path = "docker-compose.yml"
            compose_check = await self.sandy.execute_command(
                sandbox_id,
                f"test -f {compose_path}",
                cwd=task_dir,
            )
        has_compose = compose_check.get("exit_code") == 0
        # Every sandbox shares the HOST docker daemon (Sandy mounts the socket),
        # so anything Terminal-Bench names has to be namespaced per sandbox or
        # concurrent items -- and concurrent *runs*, e.g. arm B and arm C of a
        # paired comparison -- collide on the same daemon:
        #
        #   Conflict. The container name "/tbench_stable-parallel-kmeans_client"
        #   is already in use by container "0546b477..."
        #   removal of container aa1114b2... is already in progress
        #
        # Worse than the crash: upstream's default image name is the literal
        # string "client", so two items building at once silently overwrite each
        # other's image and one can end up running the other's container. That
        # fails quietly rather than loudly, which is the dangerous kind.
        ns = f"s{sandbox_id[:12]}".lower()
        image_name = f"tbench_{ns}_{task_id}".lower()
        container_name = f"{image_name}_container"
        cleanup_cmd = None
        cleanup_cwd = None

        if has_compose:
            compose_cmd = "docker compose"
            compose_check = await self.sandy.execute_command(sandbox_id, "docker compose version")
            if compose_check.get("exit_code") != 0:
                install = await self.sandy.execute_command(
                    sandbox_id,
                    "apt-get update && apt-get install -y docker-compose-plugin",
                    timeout_ms=600000,
                )
                compose_check = await self.sandy.execute_command(
                    sandbox_id,
                    "docker compose version",
                )
                if install.get("exit_code") != 0 or compose_check.get("exit_code") != 0:
                    compose_cmd = "/usr/local/bin/docker-compose-v2"
                    download = await self.sandy.execute_command(
                        sandbox_id,
                        "curl -fsSL https://github.com/docker/compose/releases/download/v2.29.7/docker-compose-linux-x86_64 "
                        "-o /usr/local/bin/docker-compose-v2 && chmod +x /usr/local/bin/docker-compose-v2",
                        timeout_ms=300000,
                    )
                    version_check = await self.sandy.execute_command(
                        sandbox_id,
                        f"{compose_cmd} version",
                    )
                    if download.get("exit_code") != 0 or version_check.get("exit_code") != 0:
                        compose_cmd = "docker-compose"
            await self._ensure_compose_context(sandbox_id, compose_path, cwd=task_dir)
            logs_dir = f"{task_dir}/logs"
            await self.sandy.execute_command(sandbox_id, f"mkdir -p {logs_dir}")
            env = {
                "T_BENCH_TASK_DOCKER_CLIENT_IMAGE_NAME": f"client_{ns}",
                "T_BENCH_TASK_DOCKER_NAME_PREFIX": f"tbench_{ns}_{task_id}",
                "T_BENCH_TASK_DOCKER_CLIENT_CONTAINER_NAME": f"tbench_{ns}_{task_id}_client",
                "T_BENCH_TASK_LOGS_PATH": logs_dir,
                "T_BENCH_CONTAINER_LOGS_PATH": "/var/log/tbench",
                "T_BENCH_TASK_AGENT_LOGS_PATH": logs_dir,
                "T_BENCH_CONTAINER_AGENT_LOGS_PATH": "/var/log/tbench/agent",
                "T_BENCH_TEST_DIR": "/tests",
                "DOCKER_HOST": "unix:///var/run/docker.sock",
            }
            identity = await self._sandbox_identity(sandbox_id)
            label_override_path = None
            if identity:
                _, owner = identity
                services_result = await self.sandy.execute_command(
                    sandbox_id,
                    f"{compose_cmd} -f {compose_path} config --services",
                    cwd=task_dir,
                )
                services = [
                    service.strip()
                    for service in (services_result.get("stdout") or "").splitlines()
                    if service.strip()
                    and all(char.isalnum() or char in "_.-" for char in service.strip())
                ]
                if services:
                    label_override_path = ".chutes-bench-labels.yaml"
                    override = {
                        "services": {
                            service: {
                                "labels": {
                                    "sandy.owner": owner,
                                    "chutes.bench.sandbox_id": sandbox_id,
                                }
                            }
                            for service in services
                        }
                    }
                    await self.sandy.write_file(
                        sandbox_id,
                        f"{task_dir}/{label_override_path}",
                        yaml.safe_dump(override, sort_keys=True),
                    )
            env_lines = [f"{key}={value}" for key, value in env.items()]
            await self.sandy.write_file(
                sandbox_id,
                f"{task_dir}/.env",
                "\n".join(env_lines) + "\n",
            )
            # -p: the project name otherwise defaults to the working directory,
            # which is "task" in every sandbox, so compose treats concurrent
            # items as the same project and recreates/removes each other's
            # containers.
            compose_project = f"tb{ns}"
            compose_files = f"-f {compose_path}"
            if label_override_path:
                compose_files += f" -f {label_override_path}"
            up_cmd = (
                f"{compose_cmd} -p {compose_project} --env-file .env "
                f"{compose_files} up --build -d"
            )
            up_result = await self.sandy.execute_command(
                sandbox_id,
                up_cmd,
                env=env,
                cwd=task_dir,
                timeout_ms=900000,
            )
            if up_result.get("exit_code") != 0:
                error_detail = up_result.get("stderr") or up_result.get("stdout") or up_result.get("error")
                return {
                    "error": error_detail,
                    "exit_code": up_result.get("exit_code"),
                    "stdout": up_result.get("stdout"),
                    "stderr": up_result.get("stderr"),
                }
            container_name = env["T_BENCH_TASK_DOCKER_CLIENT_CONTAINER_NAME"]
            cleanup_cmd = f"{compose_cmd} -p {compose_project} {compose_files} down"
            cleanup_cwd = task_dir
        else:
            build_result = await self.sandy.execute_command(
                sandbox_id,
                f"docker build -t {image_name} {task_dir}",
                timeout_ms=900000,
            )
            if build_result.get("exit_code") != 0:
                error_detail = build_result.get("stderr") or build_result.get("stdout") or build_result.get("error")
                return {
                    "error": error_detail,
                    "exit_code": build_result.get("exit_code"),
                    "stdout": build_result.get("stdout"),
                    "stderr": build_result.get("stderr"),
                }
            label_flags = await self._task_container_label_flags(sandbox_id)
            run_result = await self.sandy.execute_command(
                sandbox_id,
                f"docker run -d --name {container_name} {label_flags} "
                f"{image_name} sleep infinity",
            )
            if run_result.get("exit_code") != 0:
                error_detail = run_result.get("stderr") or run_result.get("stdout") or run_result.get("error")
                return {
                    "error": error_detail,
                    "exit_code": run_result.get("exit_code"),
                    "stdout": run_result.get("stdout"),
                    "stderr": run_result.get("stderr"),
                }
            cleanup_cmd = f"docker rm -f {container_name}"

        return {
            "container_name": container_name,
            "cleanup_cmd": cleanup_cmd,
            "cleanup_cwd": cleanup_cwd,
            # Needed by evaluate_item's teardown so the per-sandbox images do
            # not accumulate.
            "image_name": image_name,
            "namespace": ns,
        }

    # Hosts that serve Terminal-Bench's own task definitions, held-out tests and
    # reference solutions. Left reachable, an agent finds them: on the very
    # first scored task we ran, the agent walked the GitHub tree API to locate
    # the current org, fetched `tests/test_outputs.py` and `solution.sh`, and
    # announced in its trajectory "I have the tests. Let me peek at the
    # reference solution for output conventions, then write my own."
    #
    # Blocking by name rather than cutting all egress is deliberate. Several
    # tasks legitimately install dependencies during the agent phase (one of
    # ours pip-installed RDKit), so a full seal would fail them for a reason
    # unrelated to the harness, and that failure would be indistinguishable
    # from a real one. pypi/apt/npm stay reachable; the answer key does not.
    #
    # This is a floor, not a guarantee. It stops name-based retrieval, not a
    # determined agent that hardcodes an IP or finds an unlisted mirror. Read
    # trajectories before believing a high score.
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
        "hub.harborframework.com",
        "harborframework.com",
        "www.harborframework.com",
        "tbench.ai",
        "www.tbench.ai",
        "raw.githack.com",
        "cdn.jsdelivr.net",
        "gitclone.com",
        "hub.fastgit.org",
        "ghproxy.com",
    )

    def _seal_script(self) -> str:
        """Shell that blackholes the benchmark's own sources via /etc/hosts."""
        lines = "\\n".join(
            f"127.0.0.1 {host}" for host in self.BENCHMARK_SOURCE_HOSTS
        )
        # ::1 too, or a v6-preferring resolver walks straight around the v4 entry.
        lines6 = "\\n".join(f"::1 {host}" for host in self.BENCHMARK_SOURCE_HOSTS)
        return (
            "printf '%b\\n' '\\n# chutes-bench-runner: benchmark answer sources\\n"
            f"{lines}\\n{lines6}\\n' >> /etc/hosts"
        )

    # Upstream Terminal-Bench task directories ship the reference solution and
    # the held-out tests alongside the task, and `_extract_archive` unpacks the
    # whole thing into /workspace/task -- the agent's own working directory. So
    # every run had `solution.sh` (a complete working implementation, canary
    # string and all) and `tests/test_outputs.py` sitting in front of the agent,
    # and the prompt then told it to "write a solution script to
    # /workspace/task/solution.sh" -- the exact path the answer already
    # occupied.
    #
    # Two failures, not one:
    #   * contamination -- the agent can read the reference solution and the
    #     tests. Sealing the network (BENCHMARK_SOURCE_HOSTS) was beside the
    #     point while the answer key was a local file.
    #   * false credit -- if the agent never overwrites solution.sh, the harness
    #     copies the REFERENCE solution into the container and runs it, scoring
    #     a pass the agent did not earn.
    #
    # So the answer key is moved out of the workspace before the agent starts
    # and the tests are put back only for scoring. The reference solution is
    # never restored: we score what the agent wrote.
    HOLDOUT_ROOT = "/opt/tb-holdout"

    # Globs, not three literal filenames. The first version listed
    # solution.sh / tests / run-tests.sh and reported withheld: true while
    # `evaluation_tests_hidden/` -- which it had never heard of -- sat in the
    # task dir and the agent read it.
    ANSWER_KEY_GLOBS = ("*solution*", "*test*", "*hidden*", "*answer*")

    async def _withhold_answer_key(self, sandbox_id: str, ns: str) -> dict:
        """Remove the answer key from everywhere the agent can reach.

        Must run BEFORE the task image is built. `_run_terminal_bench` uses the
        task dir as the docker build context, so anything still present at
        build time is baked into the image and reachable via `docker exec` --
        which is exactly what the prompt tells the agent to use. Withholding
        only from the sandbox filesystem left three live copies: inside the
        container, in archive.tar, and in archive.b64.
        """
        holdout = f"{self.HOLDOUT_ROOT}/{ns}"
        globs = " ".join(f"'{g}'" for g in self.ANSWER_KEY_GLOBS)
        await self.sandy.execute_command(
            sandbox_id,
            f"mkdir -p {holdout} && cd /workspace/task && "
            f"for g in {globs}; do "
            f"  for p in $g; do [ -e \"$p\" ] && mv \"$p\" {holdout}/ 2>/dev/null; done; "
            f"done; "
            # One `tar -x` would restore everything the move just removed.
            f"rm -f /workspace/archive.tar /workspace/archive.b64; true",
        )
        probe = await self.sandy.execute_command(
            sandbox_id,
            "cd /workspace/task && "
            "echo LEFT=$(ls -A . 2>/dev/null | tr '\n' ',') "
            "ARCHIVES=$(ls /workspace/archive.* 2>/dev/null | wc -l) "
            f"HELD=$(ls -A {holdout} 2>/dev/null | tr '\n' ',')",
        )
        out = ((probe or {}).get("stdout") or "").strip()
        left = ""
        for token in out.split():
            if token.startswith("LEFT="):
                left = token[len("LEFT="):]
        leaked = [
            n for n in left.split(",")
            if n and any(k in n.lower() for k in ("solution", "test", "hidden", "answer"))
        ]
        return {
            "holdout_dir": holdout,
            "probe": out,
            "leaked_in_workspace": leaked,
            "archives_removed": "ARCHIVES=0" in out,
            "withheld": not leaked and "ARCHIVES=0" in out,
        }

    async def _verify_container_clean(self, sandbox_id: str, container_name: str) -> dict:
        """Assert the property from OUTSIDE: is the answer key in the container?

        The previous check asserted what our own code did (`SOL=absent`), which
        was a true statement about the sandbox filesystem and a false statement
        about the world -- the agent was reading the tests out of the container
        the whole time. This looks for the files where the agent can actually
        get at them.
        """
        result = await self.sandy.execute_command(
            sandbox_id,
            f"docker exec {container_name} sh -c "
            f"\"{{ find /tests /solution -type f 2>/dev/null; "
            f"find / -xdev \\( -name 'solution.sh' -o -name 'test_outputs.py' "
            f"-o -name 'evaluation_tests_hidden' -o -name 'run-tests.sh' \\) "
            f"-not -path '*/proc/*' 2>/dev/null; }} | sort -u | head -20\"",
        )
        found = [
            line.strip()
            for line in ((result or {}).get("stdout") or "").splitlines()
            if line.strip()
        ]
        return {"found_in_container": found, "clean": not found}

    async def _restore_tests(self, sandbox_id: str, ns: str) -> dict:
        """Put the tests back for scoring. Never the reference solution."""
        holdout = f"{self.HOLDOUT_ROOT}/{ns}"
        result = await self.sandy.execute_command(
            sandbox_id,
            f"for p in tests run-tests.sh; do "
            f"  [ -e {holdout}/$p ] && cp -r {holdout}/$p task/ || true; "
            f"done; "
            f"echo RESTORED=$(test -e task/tests && echo yes || echo no) "
            f"REF_SOLUTION_STILL_WITHHELD=$(test -e task/solution.sh && echo no || echo yes)",
        )
        return {"probe": ((result or {}).get("stdout") or "").strip()}

    async def _seal_network(self, sandbox_id: str, container_name: str) -> dict:
        """Blackhole the answer sources, then prove it took effect.

        Returns a verdict dict recorded on the item. If sealing cannot be
        verified the caller fails the item outright rather than scoring it --
        an unsealed run produces a number that looks fine and means nothing,
        which is the worst of the available outcomes.
        """
        verdict: dict[str, Any] = {"hosts": list(self.BENCHMARK_SOURCE_HOSTS)}

        await self.sandy.execute_command(sandbox_id, self._seal_script())
        # The task container is a separate netns with its own /etc/hosts, and
        # the agent reaches it through `docker exec`.
        await self.sandy.execute_command(
            sandbox_id,
            f"docker exec {container_name} sh -c {shlex.quote(self._seal_script())}",
        )

        # Two checks, because either alone can pass for the wrong reason.
        #
        #  HOSTS= counts the blackhole entries actually present in /etc/hosts.
        #         This is the real check and it works everywhere.
        #  FETCH= curls a known answer-key URL. Stronger evidence when it runs,
        #         but the task containers are minimal and many have no curl --
        #         and "curl: not found" is not a 200, so a naive
        #         "anything but 200 means blocked" would read a missing curl as
        #         a successful seal. Hence NOCURL, treated as no evidence
        #         rather than as proof.
        probe = (
            "echo HOSTS=$(grep -c 'chutes-bench-runner: benchmark answer sources' /etc/hosts); "
            "if command -v curl >/dev/null 2>&1; then "
            "echo FETCH=$(curl -s -m 8 -o /dev/null -w '%{http_code}' "
            "https://raw.githubusercontent.com/harbor-framework/terminal-bench/main/README.md "
            "|| echo CURLFAIL); "
            "else echo FETCH=NOCURL; fi"
        )
        sandbox_probe = await self.sandy.execute_command(sandbox_id, probe)
        container_probe = await self.sandy.execute_command(
            sandbox_id, f"docker exec {container_name} sh -c {shlex.quote(probe)}"
        )

        def _judge(result: dict) -> tuple[bool, str]:
            out = ((result or {}).get("stdout") or "").strip()
            hosts_ok = "HOSTS=0" not in out and "HOSTS=" in out
            fetched = "FETCH=200" in out
            # Sealed iff the entries are there AND nothing actually fetched.
            return (hosts_ok and not fetched), out

        verdict["sandbox_blocked"], verdict["sandbox_stdout"] = _judge(sandbox_probe)
        verdict["container_blocked"], verdict["container_stdout"] = _judge(container_probe)
        verdict["sealed"] = verdict["sandbox_blocked"] and verdict["container_blocked"]
        return verdict

    async def _unseal_network(self, sandbox_id: str, container_name: str) -> dict:
        """Lift the seal for the scoring phase only.

        The seal must not still be up when the tests run. Terminal-Bench's own
        test scaffolding installs its tooling from the network -- several tasks
        ship `setup-uv-pytest.sh`, and the `uv` installer resolves through
        GitHub releases. With github blackholed, that scaffolding fails and the
        item scores 0 for a reason that has nothing to do with the agent:

            /tests/setup-uv-pytest.sh: line 18: uv: command not found

        We hit exactly that and briefly read the resulting 0/2 as evidence that
        the previous 2/2 had been contamination. It was not; it was this.

        Contamination only matters while the agent (and the solution it wrote)
        is running, and both are finished by the time this is called, so
        restoring egress here costs nothing and keeps the scorer working.
        """
        # /etc/hosts is a bind mount in both Docker environments. `sed -i`
        # writes a temporary file and renames it over the original; rename(2)
        # cannot replace a mount point, so GNU sed exits 4 and leaves the seal
        # intact. Stream the prefix back into the mounted inode instead.
        restore = (
            "tmp=/tmp/chutes-bench-hosts.$$; "
            "awk '/^# chutes-bench-runner: benchmark answer sources$/{exit} "
            "{print}' /etc/hosts > \"$tmp\" && cat \"$tmp\" > /etc/hosts; "
            "status=$?; rm -f \"$tmp\"; exit $status"
        )
        result = await self.sandy.execute_command(sandbox_id, restore)
        container = await self.sandy.execute_command(
            sandbox_id, f"docker exec {container_name} sh -c {shlex.quote(restore)}"
        )

        # Do not infer anything from either restore exit code. The old exit 4
        # appeared on passing and failing items alike. Assert the actual
        # postcondition from inside both network namespaces: the blackhole
        # entries are gone and the exact answer-source endpoint is reachable.
        probe = (
            "echo HOSTS=$(grep -c "
            "'chutes-bench-runner: benchmark answer sources' /etc/hosts); "
            "if command -v curl >/dev/null 2>&1; then "
            "code=$(curl -sS --connect-timeout 8 --max-time 12 -o /dev/null "
            "-w '%{http_code}' "
            "https://raw.githubusercontent.com/harbor-framework/terminal-bench/main/README.md "
            "2>/dev/null || true); "
            "case \"$code\" in 2*|3*) echo NETWORK=OPEN;; "
            "*) echo NETWORK=CLOSED:$code;; esac; "
            "elif command -v wget >/dev/null 2>&1; then "
            "if wget -q --spider -T 12 "
            "https://raw.githubusercontent.com/harbor-framework/terminal-bench/main/README.md; "
            "then echo NETWORK=OPEN; else echo NETWORK=CLOSED; fi; "
            "elif command -v bash >/dev/null 2>&1; then "
            "if bash -c 'exec 3<>/dev/tcp/raw.githubusercontent.com/443' "
            "2>/dev/null; then echo NETWORK=OPEN; else echo NETWORK=CLOSED; fi; "
            "else echo NETWORK=UNVERIFIABLE; fi"
        )
        sandbox_probe = await self.sandy.execute_command(
            sandbox_id, probe, timeout_ms=20000
        )
        container_probe = await self.sandy.execute_command(
            sandbox_id,
            f"docker exec {container_name} sh -c {shlex.quote(probe)}",
            timeout_ms=20000,
        )

        def _judge(probe_result: Optional[dict]) -> tuple[bool, bool, str]:
            out = ((probe_result or {}).get("stdout") or "").strip()
            hosts_removed = "HOSTS=0" in out
            network_open = "NETWORK=OPEN" in out
            return hosts_removed, network_open, out

        sandbox_hosts_removed, sandbox_connected, sandbox_stdout = _judge(
            sandbox_probe
        )
        container_hosts_removed, container_connected, container_stdout = _judge(
            container_probe
        )
        verdict = {
            "sandbox_exit": (result or {}).get("exit_code"),
            "container_exit": (container or {}).get("exit_code"),
            "sandbox_hosts_removed": sandbox_hosts_removed,
            "sandbox_connected": sandbox_connected,
            "sandbox_stdout": sandbox_stdout,
            "container_hosts_removed": container_hosts_removed,
            "container_connected": container_connected,
            "container_stdout": container_stdout,
        }
        verdict["restored"] = all(
            (
                sandbox_hosts_removed,
                sandbox_connected,
                container_hosts_removed,
                container_connected,
            )
        )
        return verdict

    async def _verify_agent_terminated(
        self, sandbox_id: str, agent_summary: dict
    ) -> dict:
        """Prove Sandy's tracked agent process is gone before unsealing.

        `run_agent` is an SSE request. Returning from it only proves that the
        stream ended; requiring Sandy's terminal event plus an independent
        /proc check closes the unsafe case where a broken stream leaves the
        agent alive while scoring restores access to the answer sources.
        """
        probe = await self.sandy.execute_command(
            sandbox_id,
            "pid=$(cat /workspace/.chutes/agent.pid 2>/dev/null || true); "
            "done_value=$(cat /workspace/.chutes/agent.done 2>/dev/null || true); "
            "if [ -z \"$pid\" ]; then running=no; state=missing; "
            "elif [ -r \"/proc/$pid/stat\" ]; then "
            "state=$(cut -d' ' -f3 \"/proc/$pid/stat\" 2>/dev/null || echo unknown); "
            "case \"$state\" in Z*) running=no;; *) running=yes;; esac; "
            "else running=no; state=gone; fi; "
            "echo PID=${pid:-missing} STATE=$state RUNNING=$running "
            "DONE=${done_value:-missing}",
        )
        out = ((probe or {}).get("stdout") or "").strip()
        completion_event = agent_summary.get("type") == "complete"
        process_stopped = "RUNNING=no" in out
        return {
            "completion_event": completion_event,
            "process_stopped": process_stopped,
            "terminated": completion_event and process_stopped,
            "probe": out,
            "probe_exit": (probe or {}).get("exit_code"),
        }

    async def _sandbox_alive(self, sandbox_id: str) -> Optional[bool]:
        """Is the sandbox still there? None if we could not tell."""
        try:
            result = await self.sandy.execute_command(sandbox_id, "echo alive")
            if result is None:
                return False
            if (result.get("stdout") or "").strip() == "alive":
                return True
            return False
        except Exception:
            return None

    async def _collect_agent_usage(self, sandbox_id: str) -> dict:
        return await collect_agent_usage(self.sandy, sandbox_id)

    def _new_item_observability(self, item_id: str) -> dict[str, Any]:
        state = {
            "agent_invoked": False,
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
            )
        )

    async def _finish_evidence_retention(self, item_id: str, sandbox_id: str) -> None:
        """Finish evidence collection before teardown, isolated from scoring."""
        state = self._item_observability.get(item_id)
        if not state or not state.get("agent_invoked"):
            return
        if state.get("retention_task") is None:
            # run_agent can fail or be canceled after Sandy launched the
            # process but before it returned its event list. Stop that process
            # before archiving so the retained files are a stable final prefix.
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
                    "Could not stop agent before evidence retention",
                    item_id=item_id,
                    sandbox_id=sandbox_id,
                    error=str(exc),
                )
            self._start_evidence_retention(item_id, sandbox_id)

        task = state.get("retention_task")
        try:
            state["evidence"] = await task
        except BaseException as exc:
            # This hook runs from a sandbox finalizer, including after the
            # worker cancels an item. Evidence may fail; the score path may not.
            state["evidence"] = {
                "status": "failed",
                "path": None,
                "sha256": None,
                "size_bytes": None,
                "error": f"retention task failed: {exc}",
                "token_usage_samples": None,
            }

    def attach_item_observability(self, result: ItemResult) -> ItemResult:
        # Classification is adapter-owned observability too.  Attach it before
        # looking up evidence state so worker-created timeout/error results are
        # classified even when the adapter did not return normally.
        item = next(
            (
                candidate
                for candidate in getattr(self, "_items", [])
                if candidate.get("id") == result.item_id
            ),
            None,
        )
        if item is not None and item.get("scoring_class"):
            if result.metadata is None:
                result.metadata = {}
            result.metadata["scoring_class"] = item["scoring_class"]
            result.metadata["scoring_classification"] = {
                "audit_commit": TERMINAL_BENCH_2_1_SCORING_AUDIT_COMMIT,
                "reason": item.get("scoring_reason"),
                "evidence": copy.deepcopy(item.get("scoring_evidence") or []),
            }

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
            )
        }
        return result

    @staticmethod
    def _parse_harbor_reward(raw_reward: str) -> Optional[float]:
        """Parse the scalar reward emitted by a Harbor verifier."""
        value = (raw_reward or "").strip()
        if not value:
            return None
        try:
            return float(value)
        except ValueError:
            pass
        try:
            import json

            parsed = json.loads(value)
        except (ValueError, TypeError):
            return None
        if isinstance(parsed, (int, float)):
            return float(parsed)
        if isinstance(parsed, dict):
            if isinstance(parsed.get("reward"), (int, float)):
                return float(parsed["reward"])
            numeric = [float(v) for v in parsed.values() if isinstance(v, (int, float))]
            if len(numeric) == 1:
                return numeric[0]
        return None

    @classmethod
    def _harbor_verifier_outcome(
        cls,
        test_result: Optional[dict],
        reward_result: Optional[dict],
        *,
        test_command_executed: bool,
    ) -> dict[str, Any]:
        """Validate verifier execution before accepting Harbor's scalar."""
        network_exclusion = classify_verifier_network_failure(test_result)
        if network_exclusion:
            detail = "\n".join(
                str((test_result or {}).get(field) or "")
                for field in ("stderr", "stdout", "error")
            ).strip()
            return {
                "reward": None,
                "is_correct": None,
                "error": detail or "Verifier could not reach the network",
                "exclusion_reason": network_exclusion,
            }
        if not test_command_executed:
            detail = (
                (test_result or {}).get("stderr")
                or (test_result or {}).get("error")
                or "Harbor verifier test command did not execute"
            )
            return {
                "reward": None,
                "is_correct": None,
                "error": detail,
                "exclusion_reason": VERIFIER_NOT_EXECUTED_EXCLUSION_REASON,
            }

        reward = cls._parse_harbor_reward(
            (reward_result or {}).get("stdout") or ""
        )
        is_correct = reward is not None and reward >= 1.0
        error = None
        if reward is None:
            error = (
                (test_result or {}).get("stderr")
                or (test_result or {}).get("error")
                or "Harbor verifier did not emit a scalar reward"
            )
        return {
            "reward": reward,
            "is_correct": is_correct,
            "error": error,
            "exclusion_reason": None,
        }

    async def evaluate_item(self, item_id: str) -> ItemResult:
        """Evaluate and attach evidence even when the scored result is a failure."""
        self._new_item_observability(item_id)
        result = await self._evaluate_item(item_id)
        return self.attach_item_observability(result)

    async def _evaluate_item(self, item_id: str) -> ItemResult:
        """Evaluate a single Terminal-Bench item."""
        if not self._items:
            await self.preload()

        item = next((i for i in self._items if i["id"] == item_id), None)
        if not item:
            return ItemResult(item_id=item_id, error=f"Item {item_id} not found")

        instruction = item.get("instruction") or ""
        task_yaml = item.get("task_yaml") or ""
        if item.get("task_format") == "harbor":
            # Harbor agents change the running task environment directly; the
            # verifier later inspects that same container. Requiring a legacy
            # solution.sh here would silently turn TB 2.x into another harness.
            prompt = (
                "You are an interactive terminal agent working inside a sandbox "
                "that can execute docker commands.\n"
                "A task container will be running. Use docker exec to inspect it "
                "and make all required changes directly inside that container.\n"
                "The task container's working directory is /app. Do not look for "
                "or run verifier tests; the harness will verify the final container "
                "state after you finish.\n\n"
                f"Task instruction:\n{instruction}\n"
            )
        else:
            prompt = (
                "You are an interactive terminal agent working inside a sandbox that can execute docker commands.\n"
                "A task container will be running. Use docker exec to inspect and solve the task.\n"
                "Write a solution script to /workspace/task/solution.sh that completes the task.\n"
                "Do NOT run the task tests yourself; the harness will run them after you finish.\n\n"
                f"Task instruction:\n{instruction}\n\n"
                f"Task YAML:\n{task_yaml}\n"
            )

        try:
            start_time = time.time()
            archive = item.get("archive")
            if not isinstance(archive, (bytes, bytearray)):
                return ItemResult(item_id=item_id, error="Missing task archive bytes")

            # Budget has to be known before the sandbox exists, because the
            # sandbox TTL must outlive it (see below).
            (
                base_agent_timeout_sec,
                timeout_multiplier,
                agent_timeout,
                test_timeout,
            ) = self._item_budgets_ms(item)

            # Sandy's default TTL is 10 minutes and bench-runner never calls
            # /refresh, so the sandbox was being reaped mid-item no matter what
            # the agent budget said. Give it the agent budget + the test budget
            # + 15 minutes of slack for image build and teardown.
            sandbox_ttl_min = int(
                (agent_timeout + test_timeout) / 60000
            ) + TERMINAL_BENCH_ITEM_TIMEOUT_MARGIN_SECONDS // 60

            # Terminal-Bench requires Docker socket access for running docker-compose
            sandbox_id = await self.sandy.create_sandbox(
                enable_docker_socket=True, timeout_minutes=sandbox_ttl_min
            )
            if not sandbox_id:
                sandbox_error = self.sandy.last_error or "Could not create sandbox"
                return ItemResult(item_id=item_id, error=sandbox_error)
            self._active_sandbox_ids.add(sandbox_id)

            try:
                await self._reap_orphans(sandbox_id)
                extracted = await self._extract_archive(sandbox_id, archive)
                if not extracted:
                    return ItemResult(item_id=item_id, error="Failed to extract task archive")

                # BEFORE the build: the task dir is the docker build context,
                # so anything left here is baked into the image the agent then
                # reaches with `docker exec`.
                ns = f"s{sandbox_id[:12]}".lower()
                holdout = await self._withhold_answer_key(sandbox_id, ns)
                if not holdout.get("withheld"):
                    return ItemResult(
                        item_id=item_id,
                        error=(
                            "Refusing to score: answer key still present before build "
                            f"(leaked={holdout.get('leaked_in_workspace')}, "
                            f"probe={holdout.get('probe')!r})"
                        ),
                        judge_output={"holdout": holdout},
                        metadata={"task_id": item.get("task_id"), "holdout": holdout},
                    )

                setup_result = await self._run_terminal_bench(sandbox_id, item)
                container_name = setup_result.get("container_name")
                cleanup_cmd = setup_result.get("cleanup_cmd")
                cleanup_cwd = setup_result.get("cleanup_cwd")
                if not container_name:
                    return ItemResult(
                        item_id=item_id,
                        error=setup_result.get("error") or "Container setup failed",
                        judge_output={"setup": setup_result},
                        metadata={"task_id": item.get("task_id")},
                    )

                try:
                    # Assert the property from OUTSIDE: look for the answer
                    # key where the agent can actually reach it. The task image
                    # was built from the sanitised dir, but verify rather than
                    # assume -- the last three integrity checks were all true
                    # about our code and false about the world.
                    container_clean = await self._verify_container_clean(
                        sandbox_id, container_name
                    )
                    holdout["container"] = container_clean
                    if not container_clean.get("clean"):
                        return ItemResult(
                            item_id=item_id,
                            error=(
                                "Refusing to score: answer key reachable inside the "
                                f"task container: {container_clean.get('found_in_container')}"
                            ),
                            judge_output={"setup": setup_result, "holdout": holdout},
                            metadata={"task_id": item.get("task_id"), "holdout": holdout},
                        )

                    # Seal before the agent starts, and refuse to score an
                    # item we could not seal. See BENCHMARK_SOURCE_HOSTS.
                    seal = await self._seal_network(sandbox_id, container_name)
                    if not seal.get("sealed") and not settings_allow_unsealed():
                        return ItemResult(
                            item_id=item_id,
                            error=(
                                "Refusing to score: could not verify the sandbox is "
                                "sealed from the benchmark's own sources "
                                f"(sandbox={seal.get('sandbox_stdout')!r}, "
                                f"container={seal.get('container_stdout')!r}). "
                                "Set TERMINAL_BENCH_ALLOW_UNSEALED=1 to override, "
                                "and label any resulting score as contaminated."
                            ),
                            judge_output={"setup": setup_result, "seal": seal},
                            metadata={"task_id": item.get("task_id"), "seal": seal},
                        )


                    # Which Sandy CLI agent drives the task. Selectable so the
                    # same benchmark can be run as a paired A/B of harnesses on
                    # one model -- "codex" (upstream) vs "chutescoder" (the
                    # Chutes fork with the RLM harness). Set per run with
                    #   "config": {"terminal_bench": {"agent": "chutescoder"}}
                    # which the worker already attaches as adapter.run_config.
                    # Read under the adapter's real name. This used the literal
                    # "terminal_bench" while base.get_items_for_evaluation reads
                    # self.get_name() == "terminal_bench_hard", so a config that
                    # set both `agent` and `item_ids` had the agent honoured and
                    # the item_ids silently ignored -- launched with ["72"], ran
                    # item 24. Both keys accepted; the real name wins.
                    _cfg = getattr(self, "run_config", None) or {}
                    _tb = {**_cfg.get("terminal_bench", {}), **_cfg.get(self.get_name(), {})}
                    agent_name = (
                        _tb.get("agent")
                        or os.getenv("TERMINAL_BENCH_AGENT")
                        or "codex"
                    )
                    try:
                        agent_launch = await prepare_sandy_agent_launch(
                            client=self.client,
                            sandy=self.sandy,
                            sandbox_id=sandbox_id,
                            agent=agent_name,
                            model=self.model_slug,
                        )
                    except (RuntimeError, ValueError) as exc:
                        return ItemResult(
                            item_id=item_id,
                            error=str(exc),
                            metadata={
                                "task_id": item.get("task_id"),
                                "agent": agent_name,
                                "provider": getattr(self.client, "provider", "chutes"),
                            },
                        )
                    agent_env_vars = dict(agent_launch.env_vars)
                    agent_provider_metadata = agent_launch.metadata
                    if (
                        agent_name == "prime-agent"
                        and agent_launch.provider != "openrouter"
                    ):
                        prime_provider = str(
                            _tb.get("provider")
                            or os.getenv("TERMINAL_BENCH_PRIME_PROVIDER")
                            or "chutes"
                        ).strip().lower()
                        agent_env_vars["PRIME_AGENT_PROVIDER"] = prime_provider
                        if prime_provider == "openrouter":
                            openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
                            if not openrouter_api_key:
                                return ItemResult(
                                    item_id=item_id,
                                    error=(
                                        "OPENROUTER_API_KEY is required for the "
                                        "Prime Agent OpenRouter arm"
                                    ),
                                    metadata={
                                        "task_id": item.get("task_id"),
                                        "agent": agent_name,
                                        "provider": prime_provider,
                                    },
                                )
                            agent_env_vars = {
                                "OPENROUTER_API_KEY": openrouter_api_key,
                                "PRIME_AGENT_PROVIDER": prime_provider,
                            }
                        elif prime_provider != "chutes":
                            return ItemResult(
                                item_id=item_id,
                                error=(
                                    f"Unsupported Prime Agent provider: {prime_provider}; "
                                    "expected chutes or openrouter"
                                ),
                                metadata={
                                    "task_id": item.get("task_id"),
                                    "agent": agent_name,
                                    "provider": prime_provider,
                                },
                            )
                    self._item_observability[item_id]["agent_invoked"] = True
                    agent_result = await self.sandy.run_agent(
                        sandbox_id,
                        agent=agent_name,
                        model=self.model_slug,
                        prompt=prompt + f"\nContainer name: {container_name}\n",
                        max_duration=max(60, int(agent_timeout / 1000)),
                        raw_prompt=True,
                        api_base_url=agent_launch.api_base_url,
                        env_vars=agent_env_vars,
                    )
                    usage_error: Optional[str] = None
                    try:
                        await retain_sandy_agent_rollout(
                            self.sandy, sandbox_id, agent_launch
                        )
                    except RuntimeError as exc:
                        usage_error = str(exc)
                    agent_usage = await self._collect_agent_usage(sandbox_id)
                    agent_summary = agent_result.get("summary") or {}
                    if usage_error is None:
                        try:
                            validate_openrouter_agent_usage(agent_launch, agent_usage)
                        except RuntimeError as exc:
                            usage_error = str(exc)

                    launch_exclusion_reason = classify_agent_launch_failure(
                        "",
                        agent_summary,
                        agent_usage,
                        agent_invoked=True,
                    )
                    if launch_exclusion_reason:
                        agent_events = agent_result.get("events") or []
                        launch_error = next(
                            (
                                event.get("error") or event.get("message")
                                for event in reversed(agent_events)
                                if isinstance(event, dict)
                                and event.get("type") == "error"
                            ),
                            None,
                        )
                        return ItemResult(
                            item_id=item_id,
                            item_hash=self.compute_item_hash(item.get("task_id")),
                            prompt=prompt,
                            error=(
                                str(launch_error)
                                if launch_error
                                else (
                                    "Agent launch produced no completion summary, "
                                    "token counts, or rollout."
                                )
                            ),
                            latency_ms=int((time.time() - start_time) * 1000),
                            metadata={
                                "task_id": item.get("task_id"),
                                "agent": agent_name,
                                "agent_summary": agent_summary,
                                "agent_usage": agent_usage,
                                "agent_provider": agent_provider_metadata,
                                "holdout": holdout,
                                "seal": seal,
                                "exclusion_reason": launch_exclusion_reason,
                            },
                        )

                    sandbox_alive = await self._sandbox_alive(sandbox_id)
                    exclusion_reason, exit_note = classify_agent_exit(
                        agent_summary, agent_timeout / 1000, sandbox_alive
                    )
                    if exclusion_reason:
                        return ItemResult(
                            item_id=item_id,
                            item_hash=self.compute_item_hash(item.get("task_id")),
                            prompt=prompt,
                            error=exit_note,
                            latency_ms=int((time.time() - start_time) * 1000),
                            input_tokens=agent_usage.get("input_tokens"),
                            output_tokens=agent_usage.get("output_tokens"),
                            metadata={
                                "task_id": item.get("task_id"),
                                "agent": agent_name,
                                "agent_summary": agent_summary,
                                "agent_usage": agent_usage,
                                "holdout": holdout,
                                "seal": seal,
                                "agent_timeout_sec": agent_timeout / 1000,
                                "agent_timeout_multiplier": timeout_multiplier,
                                "sandbox_ttl_min": sandbox_ttl_min,
                                "sandbox_alive_at_exit": sandbox_alive,
                                "exclusion_reason": exclusion_reason,
                                "exit_note": exit_note,
                            },
                        )

                    agent_termination = await self._verify_agent_terminated(
                        sandbox_id, agent_summary
                    )
                    if not agent_termination.get("terminated"):
                        return ItemResult(
                            item_id=item_id,
                            item_hash=self.compute_item_hash(item.get("task_id")),
                            prompt=prompt,
                            error=(
                                "Agent completion could not be verified before the "
                                "scoring network would be restored "
                                f"(completion_event="
                                f"{agent_termination.get('completion_event')}, "
                                f"probe={agent_termination.get('probe')!r})."
                            ),
                            latency_ms=int((time.time() - start_time) * 1000),
                            input_tokens=agent_usage.get("input_tokens"),
                            output_tokens=agent_usage.get("output_tokens"),
                            metadata={
                                "task_id": item.get("task_id"),
                                "agent": agent_name,
                                "agent_summary": agent_summary,
                                "agent_usage": agent_usage,
                                "agent_termination": agent_termination,
                                "holdout": holdout,
                                "seal": seal,
                                "agent_timeout_sec": agent_timeout / 1000,
                                "agent_timeout_multiplier": timeout_multiplier,
                                "exclusion_reason": (
                                    AGENT_NOT_TERMINATED_EXCLUSION_REASON
                                ),
                            },
                        )

                    # The agent is now proven stopped, so its rollout and
                    # combined stdout/stderr are stable. Prepare/compress/
                    # transfer while the verifier runs. The finalizer waits
                    # before sandbox deletion; failure never changes scoring.
                    self._start_evidence_retention(item_id, sandbox_id)

                    if usage_error:
                        return ItemResult(
                            item_id=item_id,
                            item_hash=self.compute_item_hash(item.get("task_id")),
                            prompt=prompt,
                            error=usage_error,
                            latency_ms=int((time.time() - start_time) * 1000),
                            input_tokens=agent_usage.get("input_tokens"),
                            output_tokens=agent_usage.get("output_tokens"),
                            metadata={
                                "task_id": item.get("task_id"),
                                "agent": agent_name,
                                "agent_summary": agent_summary,
                                "agent_usage": agent_usage,
                                "agent_provider": agent_provider_metadata,
                                "agent_termination": agent_termination,
                                "holdout": holdout,
                                "seal": seal,
                            },
                        )

                    agent_events = agent_result.get("events") or []
                    agent_output = next(
                        (event.get("text") for event in reversed(agent_events) if event.get("type") == "output"),
                        "",
                    )
                    latency_ms = int((time.time() - start_time) * 1000)

                    if item.get("task_format") == "legacy":
                        solution_check = await self.sandy.execute_command(
                            sandbox_id,
                            "test -f task/solution.sh",
                        )
                        if solution_check.get("exit_code") != 0:
                            return ItemResult(
                                item_id=item_id,
                                item_hash=self.compute_item_hash(item.get("task_id")),
                                prompt=prompt,
                                response=agent_output,
                                error=(
                                    "Agent did not write task/solution.sh. "
                                    "(Meaningful only since the reference solution "
                                    "is withheld -- before that, this file always "
                                    "existed and the harness would have scored the "
                                    "reference implementation.)"
                                ),
                                latency_ms=latency_ms,
                                # Failed items cost real money. Without this the
                                # arm that fails looks free: the first paired run
                                # reported arm B at 0 tokens / $0.00 across three
                                # items, which reads as "never ran" when in fact it
                                # ran and spent -- a 1-item rerun of the same task
                                # billed $0.18. Cost is part of the result, so an
                                # unsuccessful item has to carry its usage too.
                                input_tokens=agent_usage.get("input_tokens"),
                                output_tokens=agent_usage.get("output_tokens"),
                                metadata={
                                    "task_id": item.get("task_id"),
                                    "agent": agent_name,
                                    "agent_summary": agent_summary,
                                    "agent_usage": agent_usage,
                                    "holdout": holdout,
                                    "seal": seal,
                                    "agent_timeout_sec": agent_timeout / 1000,
                                    "agent_timeout_multiplier": timeout_multiplier,
                                },
                            )

                        await self.sandy.execute_command(
                            sandbox_id,
                            "chmod +x task/solution.sh",
                        )
                        await self.sandy.execute_command(
                            sandbox_id,
                            f"docker cp task/solution.sh {container_name}:/solution.sh",
                        )
                        agent_exec = await self.sandy.execute_command(
                            sandbox_id,
                            f"docker exec {container_name} bash -c 'bash /solution.sh'",
                            timeout_ms=agent_timeout,
                        )
                    else:
                        # Harbor's protocol is direct manipulation of the task
                        # container. Requiring or executing solution/solve.sh
                        # would run the held-out oracle rather than the agent.
                        agent_exec = {"mode": "direct-container", "exit_code": 0}

                    # Agent and its solution are both done -- release the seal
                    # so the benchmark's own test scaffolding can install its
                    # tooling. See _unseal_network.
                    seal["released_for_tests"] = await self._unseal_network(
                        sandbox_id, container_name
                    )
                    if not seal["released_for_tests"].get("restored"):
                        release = seal["released_for_tests"]
                        return ItemResult(
                            item_id=item_id,
                            item_hash=self.compute_item_hash(item.get("task_id")),
                            prompt=prompt,
                            response=agent_output,
                            error=(
                                "Verifier network connectivity was not restored "
                                "inside both the sandbox and task container "
                                f"(sandbox={release.get('sandbox_stdout')!r}, "
                                f"container={release.get('container_stdout')!r})."
                            ),
                            latency_ms=latency_ms,
                            judge_output={
                                "setup": setup_result,
                                "agent_summary": agent_summary,
                                "agent_exec": agent_exec,
                            },
                            input_tokens=agent_usage.get("input_tokens"),
                            output_tokens=agent_usage.get("output_tokens"),
                            metadata={
                                "task_id": item.get("task_id"),
                                "agent": agent_name,
                                "agent_summary": agent_summary,
                                "agent_usage": agent_usage,
                                "agent_termination": agent_termination,
                                "holdout": holdout,
                                "seal": seal,
                                "agent_timeout_sec": agent_timeout / 1000,
                                "agent_timeout_multiplier": timeout_multiplier,
                                "exclusion_reason": (
                                    VERIFIER_NETWORK_EXCLUSION_REASON
                                ),
                            },
                        )
                    # Put the tests back for scoring. The reference solution
                    # stays withheld -- we score what the agent wrote.
                    holdout["restored_for_tests"] = await self._restore_tests(
                        sandbox_id, ns
                    )

                    # Test phase: copy tests and run
                    await self.sandy.execute_command(
                        sandbox_id,
                        f"docker exec {container_name} sh -c 'rm -rf /tests && mkdir -p /tests'",
                    )
                    await self.sandy.execute_command(
                        sandbox_id,
                        f"docker cp task/tests/. {container_name}:/tests/",
                    )
                    reward_result: Optional[dict[str, Any]] = None
                    harbor_reward: Optional[float] = None
                    verifier_exclusion_reason: Optional[str] = None
                    test_command_probe: Optional[dict[str, Any]] = None
                    test_command_executed = True
                    if item.get("task_format") == "harbor":
                        # A unique in-container sentinel is stronger than an
                        # execute API response: transport failures return
                        # exit_code=-1 too, and a stale reward file can still be
                        # parseable. Only the shell that execs test.sh can write
                        # this exact value, and it removes old rewards first.
                        test_command_marker = (
                            f"{sandbox_id}:{item_id}:{time.time_ns()}"
                        )
                        marker_path = "/logs/verifier/.chutes-test-command-started"
                        verifier_command = (
                            "rm -f /logs/verifier/reward.json "
                            "/logs/verifier/reward.txt && "
                            f"printf '%s' {shlex.quote(test_command_marker)} > "
                            f"{marker_path} && exec bash /tests/test.sh"
                        )
                        test_result = await self.sandy.execute_command(
                            sandbox_id,
                            f"docker exec -w /app {container_name} bash -c "
                            f"{shlex.quote(verifier_command)}",
                            timeout_ms=test_timeout,
                        )
                        test_command_probe = await self.sandy.execute_command(
                            sandbox_id,
                            f"docker exec {container_name} sh -c "
                            f"{shlex.quote(f'cat {marker_path} 2>/dev/null || true')}",
                        )
                        test_command_executed = (
                            ((test_command_probe or {}).get("stdout") or "").strip()
                            == test_command_marker
                        )
                        reward_result = await self.sandy.execute_command(
                            sandbox_id,
                            f"docker exec {container_name} sh -c "
                            "'if [ -f /logs/verifier/reward.json ]; then "
                            "cat /logs/verifier/reward.json; "
                            "elif [ -f /logs/verifier/reward.txt ]; then "
                            "cat /logs/verifier/reward.txt; else exit 1; fi'",
                        )
                        verifier_outcome = self._harbor_verifier_outcome(
                            test_result,
                            reward_result,
                            test_command_executed=test_command_executed,
                        )
                        harbor_reward = verifier_outcome["reward"]
                        is_correct = verifier_outcome["is_correct"]
                        error = verifier_outcome["error"]
                        verifier_exclusion_reason = verifier_outcome[
                            "exclusion_reason"
                        ]
                    else:
                        test_script_check = await self.sandy.execute_command(
                            sandbox_id,
                            "test -f task/run-tests.sh",
                        )
                        if test_script_check.get("exit_code") == 0:
                            await self.sandy.execute_command(
                                sandbox_id,
                                f"docker cp task/run-tests.sh {container_name}:/run-tests.sh",
                            )
                        else:
                            default_script = (
                                "#!/bin/bash\n"
                                "set -e\n"
                                "cd /tests\n"
                                "python -m pip install pytest\n"
                                "python -m pytest test_outputs.py -v\n"
                            )
                            await self.sandy.write_file(
                                sandbox_id, "default-run-tests.sh", default_script
                            )
                            await self.sandy.execute_command(
                                sandbox_id,
                                f"docker cp default-run-tests.sh {container_name}:/run-tests.sh",
                            )

                        test_result = await self.sandy.execute_command(
                            sandbox_id,
                            f"docker exec {container_name} bash /run-tests.sh",
                            timeout_ms=test_timeout,
                        )
                        is_correct = test_result.get("exit_code") == 0
                        error = None if is_correct else (
                            test_result.get("stderr") or test_result.get("error")
                        )
                        verifier_exclusion_reason = (
                            classify_verifier_network_failure(test_result)
                        )

                    if verifier_exclusion_reason:
                        if not error:
                            error = "\n".join(
                                str((test_result or {}).get(field) or "")
                                for field in ("stderr", "stdout", "error")
                            ).strip() or "Verifier could not reach the network"
                        return ItemResult(
                            item_id=item_id,
                            item_hash=self.compute_item_hash(item.get("task_id")),
                            prompt=prompt,
                            response=agent_output,
                            expected=(
                                "[Harbor verifier reward >= 1]"
                                if item.get("task_format") == "harbor"
                                else "[Tests passed]"
                            ),
                            error=error,
                            latency_ms=latency_ms,
                            judge_output={
                                "setup": setup_result,
                                "agent_summary": agent_summary,
                                "agent_exec": agent_exec,
                                "test_result": test_result,
                                "test_command_probe": test_command_probe,
                                "test_command_executed": test_command_executed,
                                "reward_result": reward_result,
                            },
                            input_tokens=agent_usage.get("input_tokens"),
                            output_tokens=agent_usage.get("output_tokens"),
                            metadata={
                                "task_id": item.get("task_id"),
                                "difficulty": item.get("difficulty"),
                                "agent": agent_name,
                                "agent_usage": agent_usage,
                                "agent_summary": agent_summary,
                                "agent_termination": agent_termination,
                                "seal": seal,
                                "holdout": holdout,
                                "task_format": item.get("task_format"),
                                "test_command_executed": test_command_executed,
                                "exclusion_reason": verifier_exclusion_reason,
                            },
                        )

                    return ItemResult(
                        item_id=item_id,
                        item_hash=self.compute_item_hash(item.get("task_id")),
                        prompt=prompt,
                        response=agent_output,
                        expected=(
                            "[Harbor verifier reward >= 1]"
                            if item.get("task_format") == "harbor"
                            else "[Tests passed]"
                        ),
                        is_correct=is_correct,
                        score=(
                            harbor_reward
                            if item.get("task_format") == "harbor" and harbor_reward is not None
                            else (1.0 if is_correct else 0.0)
                        ),
                        latency_ms=latency_ms,
                        judge_output={
                            "setup": setup_result,
                            "agent_summary": agent_summary,
                            "agent_exec": agent_exec,
                            "test_result": test_result,
                            "test_command_probe": test_command_probe,
                            "test_command_executed": test_command_executed,
                            "reward_result": reward_result,
                        },
                        error=error,
                        input_tokens=agent_usage.get("input_tokens"),
                        output_tokens=agent_usage.get("output_tokens"),
                        metadata={
                            "task_id": item.get("task_id"),
                            "difficulty": item.get("difficulty"),
                            "agent": agent_name,
                            "agent_provider": agent_provider_metadata,
                            "agent_usage": agent_usage,
                            "agent_termination": agent_termination,
                            # Recorded per item so a reviewer can confirm the
                            # run was sealed without taking it on trust.
                            "seal": seal,
                            "holdout": holdout,
                            "agent_timeout_sec": agent_timeout / 1000,
                            "agent_timeout_base_sec": base_agent_timeout_sec,
                            "agent_timeout_multiplier": timeout_multiplier,
                            "agent_summary": agent_summary,
                            "agent_output_excerpt": agent_output[:2000] if agent_output else "",
                            "task_yaml": task_yaml,
                            "task_format": item.get("task_format"),
                            "test_command_executed": test_command_executed,
                            "dataset_repository": item.get("dataset_repository"),
                            "dataset_commit": item.get("dataset_commit"),
                            "manifest_repository": item.get("manifest_repository"),
                            "manifest_commit": item.get("manifest_commit"),
                        },
                    )
                finally:
                    if cleanup_cmd:
                        await self.sandy.execute_command(
                            sandbox_id, cleanup_cmd, cwd=cleanup_cwd
                        )
                    # Namespacing the task image per sandbox (see _run_terminal_bench)
                    # fixed items overwriting each other, but it also means every
                    # item now leaves its own multi-GB image behind instead of
                    # reusing one. On a host that runs at ~95% disk that fills the
                    # filesystem within a few dozen items, so the image goes when
                    # the item does. `docker compose down` removes containers, not
                    # images.
                    stale = ""
                    if setup_result.get("cleanup_images", True):
                        stale = " ".join(
                            x for x in (
                                setup_result.get("image_name"),
                                f"client_{setup_result.get('namespace')}"
                                if setup_result.get("namespace") else None,
                            ) if x
                        )
                    if stale:
                        await self.sandy.execute_command(
                            sandbox_id, f"docker rmi -f {stale} 2>/dev/null || true"
                        )
            finally:
                try:
                    await self._finish_evidence_retention(item_id, sandbox_id)
                finally:
                    try:
                        await self._cleanup_owned_task_containers(sandbox_id)
                    finally:
                        await self.sandy.terminate_sandbox(sandbox_id)
                        self._active_sandbox_ids.discard(sandbox_id)

        except Exception as e:
            logger.error("Terminal-Bench evaluation failed", item_id=item_id, error=str(e))
            summary = locals().get("agent_summary")
            observability = self._item_observability.get(item_id) or {}
            exclusion_reason = classify_bare_failure(
                str(e),
                summary,
                locals().get("agent_usage"),
                agent_invoked=bool(observability.get("agent_invoked")),
            )
            return ItemResult(
                item_id=item_id,
                prompt=prompt,
                response=locals().get("agent_output", "") or "",
                error=str(e),
                metadata={
                    "task_id": item.get("task_id"),
                    "agent": locals().get("agent_name"),
                    "agent_summary": summary,
                    "agent_usage": locals().get("agent_usage"),
                    "exclusion_reason": exclusion_reason,
                },
            )


@register_adapter("terminal_bench")
class TerminalBenchAdapter(TerminalBenchBaseAdapter):
    """The current stable Terminal-Bench release (2.1)."""

    benchmark_name = "terminal_bench"
    benchmark_spec = TERMINAL_BENCH_2_1


@register_adapter("terminal_bench_1")
class TerminalBench1Adapter(TerminalBenchBaseAdapter):
    """The official legacy Terminal-Bench 1.0 Core v0.1.1 task set."""

    benchmark_name = "terminal_bench_1"
    benchmark_spec = TERMINAL_BENCH_1


@register_adapter("terminal_bench_2")
class TerminalBench2Adapter(TerminalBenchBaseAdapter):
    """The current verified 2.x release, Terminal-Bench 2.1."""

    benchmark_name = "terminal_bench_2"
    benchmark_spec = TERMINAL_BENCH_2_1


@register_adapter("terminal_bench_2_0")
class TerminalBench20Adapter(TerminalBenchBaseAdapter):
    """The original Terminal-Bench 2.0 release."""

    benchmark_name = "terminal_bench_2_0"
    benchmark_spec = TERMINAL_BENCH_2_0


@register_adapter("terminal_bench_2_1")
class TerminalBench21Adapter(TerminalBenchBaseAdapter):
    """The verified Terminal-Bench 2.1 release."""

    benchmark_name = "terminal_bench_2_1"
    benchmark_spec = TERMINAL_BENCH_2_1


@register_adapter("terminal_bench_hard")
class TerminalBenchHardAdapter(TerminalBenchBaseAdapter):
    """The reproducible 47-task Terminal-Bench Hard leaderboard subset."""

    benchmark_name = "terminal_bench_hard"
    benchmark_spec = TERMINAL_BENCH_HARD
