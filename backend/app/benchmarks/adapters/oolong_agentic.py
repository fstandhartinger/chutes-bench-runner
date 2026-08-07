"""OOLONG driven by a CLI agent in a sandbox, instead of one API call.

Why this exists
---------------
`oolong` sends the whole corpus in a single completion request. That measures
the *model*, and nothing about the harness wrapped around it: every CLI agent
would produce an identical number, because none of them is involved.

Terminal-Bench Hard was the only adapter in this repo that drove a Sandy CLI
agent, which meant the chutescoder-vs-codex experiment had exactly one
benchmark, and it was an agentic-coding one. Long-context-across-compaction is
the RLM design's central claim and the only place it has a mechanism the
baseline structurally lacks, so it is the thing most worth measuring.

Here the corpus is written into the sandbox as a file and the agent is asked to
answer from it. Scoring is unchanged -- it reuses `OolongAdapter`'s own
`_extract_answer` / numeric / exact-match logic -- so the only variable is the
harness.

What this is NOT
----------------
**These numbers are not comparable to `oolong`, and not comparable to published
OOLONG figures.** Handing the agent a *file* changes the task: instead of
holding 130k tokens of transcript in its context, the agent can read, chunk,
grep and summarise it. That is exactly the capability the RLM harness claims to
add, so it is a fair comparison *between arms* -- but it is a different task
from single-shot OOLONG, and reporting it against those numbers would be wrong.

It also means a task whose answer can be found with `grep` measures tool use
rather than long-context reasoning. OOLONG's questions are aggregate/reasoning
questions over dialogue, not needle retrieval, which is why this variant is
built on OOLONG rather than on S-NIAH -- S-NIAH in a sandbox is a `grep`
benchmark and would measure nothing.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import os
import re
import shlex
import time
from typing import Any
from urllib.parse import urlparse

from app.benchmarks.adapters.oolong import (
    OOLONG_SYNTH_REPO,
    OOLONG_SYNTH_REVISION,
    OOLONG_SYNTH_SPLIT,
    OolongAdapter,
    _extract_answer,
    _normalize_prediction,
    score_answer,
)
from app.benchmarks.adapters.terminal_bench import (
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
from app.benchmarks.base import ItemResult
from app.benchmarks.registry import register_adapter
from app.core.logging import get_logger
from app.services.sandy_service import SandyService

logger = get_logger(__name__)

CORPUS_PATH = "/workspace/corpus.txt"
ANSWER_PATH = "/workspace/answer.txt"
OOLONG_AGENTIC_ITEM_TIMEOUT_MARGIN_SECONDS = 5 * 60

SANDBOX_CREATION_EXCLUSION_REASON = "infrastructure_sandbox_creation"
CORPUS_UPLOAD_EXCLUSION_REASON = "infrastructure_corpus_upload"
AGENT_SETUP_EXCLUSION_REASON = "infrastructure_agent_setup"
TOKEN_ACCOUNTING_EXCLUSION_REASON = "infrastructure_token_accounting"
EVIDENCE_RETENTION_EXCLUSION_REASON = "infrastructure_evidence_retention"
LOCAL_ANSWER_LEAK_EXCLUSION_REASON = "integrity_local_answer_leak"
NETWORK_SEAL_EXCLUSION_REASON = "integrity_network_seal_unproven"
NETWORK_SEAL_BROKEN_EXCLUSION_REASON = "integrity_network_seal_broken"
NETWORK_PROBE_EXCLUSION_REASON = "infrastructure_network_probe"


@register_adapter("oolong_agentic")
class OolongAgenticAdapter(OolongAdapter):
    """OOLONG, answered by a CLI agent working over a file in a sandbox."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.sandy = SandyService()
        self._item_observability: dict[str, dict[str, Any]] = {}

    def get_name(self) -> str:
        return "oolong_agentic"

    def get_display_name(self) -> str:
        return "OOLONG (agentic)"

    def supports_parallel_items(self) -> bool:
        # Each item holds a sandbox for minutes; the sandbox host is the
        # bottleneck, not the model endpoint.
        return False

    def get_item_timeout_seconds(self, item_id: str | None = None) -> int:
        """Keep the worker's outer cap above the agent plus evidence transfer."""
        return (
            int(os.getenv("OOLONG_AGENTIC_MAX_SECONDS", "900"))
            + OOLONG_AGENTIC_ITEM_TIMEOUT_MARGIN_SECONDS
        )

    def _agent_name(self) -> str:
        return (
            (getattr(self, "run_config", None) or {})
            .get("oolong_agentic", {})
            .get("agent")
            or os.getenv("OOLONG_AGENTIC_AGENT")
            or "codex"
        )

    async def _write_corpus(self, sandbox_id: str, text: str) -> dict[str, Any]:
        """Put the corpus in the sandbox, and prove it arrived intact.

        Written with `sandy.write_file` rather than shell `printf`/heredoc:
        an OOLONG context runs to hundreds of KB, and pushing that through a
        command line silently truncates. The first attempt did exactly that --
        48,514 bytes of an expected 198,514 -- which the length check below
        caught. Without that check it would have looked like the agent giving a
        wrong answer to a question whose evidence was missing.
        """
        raw = text.encode("utf-8")
        expected_bytes = len(raw)
        expected_sha256 = hashlib.sha256(raw).hexdigest()
        encoded = base64.b64encode(raw).decode("ascii")
        if not await self.sandy.write_file(sandbox_id, "corpus.b64", encoded):
            logger.error("Corpus upload failed", chars=len(encoded))
            return {
                "uploaded": False,
                "expected_bytes": expected_bytes,
                "expected_sha256": expected_sha256,
                "error": self.sandy.last_error or "sandy.write_file returned false",
            }
        result = await self.sandy.execute_command(
            sandbox_id, f"base64 -d corpus.b64 > {CORPUS_PATH} && rm -f corpus.b64"
        )
        if (result or {}).get("exit_code") != 0:
            logger.error("Corpus decode failed", result=str(result)[:300])
            return {
                "uploaded": False,
                "expected_bytes": expected_bytes,
                "expected_sha256": expected_sha256,
                "error": (result or {}).get("error")
                or (result or {}).get("stderr")
                or "corpus base64 decode failed",
            }

        check = await self.sandy.execute_command(
            sandbox_id,
            f"wc -c < {CORPUS_PATH}; sha256sum {CORPUS_PATH} | awk '{{print $1}}'",
        )
        lines = ((check or {}).get("stdout") or "").splitlines()
        try:
            written = int(lines[0].strip())
            written_sha256 = lines[1].strip()
        except (IndexError, ValueError):
            return {
                "uploaded": False,
                "expected_bytes": expected_bytes,
                "expected_sha256": expected_sha256,
                "probe_stdout": (check or {}).get("stdout"),
                "error": "corpus verification probe did not return size and SHA-256",
            }
        uploaded = written == expected_bytes and written_sha256 == expected_sha256
        if not uploaded:
            logger.error(
                "Corpus write mismatch",
                written=written,
                expected=expected_bytes,
                written_sha256=written_sha256,
                expected_sha256=expected_sha256,
            )
        return {
            "uploaded": uploaded,
            "expected_bytes": expected_bytes,
            "written_bytes": written,
            "expected_sha256": expected_sha256,
            "written_sha256": written_sha256,
            "path": CORPUS_PATH,
            "error": None if uploaded else "corpus size or SHA-256 mismatch",
        }

    def _build_prompt(self, item: dict) -> str:
        return (
            f"The file {CORPUS_PATH} contains a long transcript.\n\n"
            f"Question: {item['question']}\n\n"
            "Work out the answer from the file. You may read it however you "
            "like -- in full, in chunks, or with shell tools.\n\n"
            "Do not use the network; all task data is in the file.\n\n"
            f"When you are done, write ONLY the final answer to {ANSWER_PATH}, "
            "with no explanation, no units, no punctuation and no surrounding "
            "text. Then stop."
        )

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
        "datasets-server.huggingface.co",
    )
    SEAL_MARKER = "chutes-bench-runner: OOLONG answer sources"

    def _provider_host(self) -> str:
        host = urlparse(self.client.get_api_base_url()).hostname
        if not host:
            raise ValueError("The inference provider URL has no hostname")
        return host

    def _seal_script(self) -> str:
        provider_host = self._provider_host()
        entries = "\n".join(
            [
                *(f"127.0.0.1 {host}" for host in self.BENCHMARK_SOURCE_HOSTS),
                *(f"::1 {host}" for host in self.BENCHMARK_SOURCE_HOSTS),
            ]
        )
        return (
            "provider_host="
            + shlex.quote(provider_host)
            + '; provider_addresses=$(getent ahosts "$provider_host" 2>/dev/null '
            "| awk '{print $1}' | sort -u); "
            'test -n "$provider_addresses" || exit 41; '
            "for address in $provider_addresses; do "
            'printf \'%s %s\\n\' "$address" "$provider_host"; done >> /etc/hosts; '
            "printf '%b\\n' "
            + shlex.quote(f"\n# {self.SEAL_MARKER}\n{entries}\n")
            + " >> /etc/hosts; "
            # Model calls still resolve through the provider entry captured
            # above. Everything else loses DNS, which closes aliases, dataset
            # APIs, mirrors, and search engines that a finite denylist misses.
            "printf 'nameserver 127.0.0.1\\noptions attempts:1 timeout:1\\n' "
            "> /etc/resolv.conf"
        )

    async def _probe_network_seal(self, sandbox_id: str) -> dict[str, Any]:
        probe_url = (
            "https://huggingface.co/datasets/"
            f"{OOLONG_SYNTH_REPO}/resolve/{OOLONG_SYNTH_REVISION}/README.md"
        )
        provider_url = self.client.get_api_base_url().rstrip("/") + "/models"
        result = await self.sandy.execute_command(
            sandbox_id,
            f"echo HOSTS=$(grep -c {shlex.quote(self.SEAL_MARKER)} /etc/hosts); "
            "if command -v curl >/dev/null 2>&1; then "
            "echo CURL=yes; "
            f"source_code=$(curl -sS --connect-timeout 4 --max-time 8 -o /dev/null "
            f"-w '%{{http_code}}' {shlex.quote(probe_url)} 2>/dev/null || true); "
            "echo SOURCE_FETCH=${source_code:-ERR}; "
            "public_code=$(curl -sS --connect-timeout 4 --max-time 8 -o /dev/null "
            "-w '%{http_code}' https://example.com/ 2>/dev/null || true); "
            "echo PUBLIC_FETCH=${public_code:-ERR}; "
            f"provider_code=$(curl -sS --connect-timeout 4 --max-time 8 -o /dev/null "
            f"-w '%{{http_code}}' {shlex.quote(provider_url)} 2>/dev/null || true); "
            "echo PROVIDER_FETCH=${provider_code:-ERR}; "
            "else echo CURL=no; echo SOURCE_FETCH=NOCURL; "
            "echo PUBLIC_FETCH=NOCURL; echo PROVIDER_FETCH=NOCURL; fi",
        )
        stdout = ((result or {}).get("stdout") or "").strip()
        source_fetch = re.search(r"(?:^|\s)SOURCE_FETCH=(\d{3})", stdout)
        public_fetch = re.search(r"(?:^|\s)PUBLIC_FETCH=(\d{3})", stdout)
        provider_fetch = re.search(r"(?:^|\s)PROVIDER_FETCH=(\d{3})", stdout)
        source_reachable = bool(
            source_fetch and source_fetch.group(1).startswith(("2", "3"))
        )
        public_reachable = bool(
            public_fetch and public_fetch.group(1).startswith(("2", "3"))
        )
        provider_reachable = bool(
            provider_fetch and provider_fetch.group(1) not in {"000"}
        )
        sealed = (
            (result or {}).get("exit_code") == 0
            and "HOSTS=0" not in stdout
            and "HOSTS=" in stdout
            and "CURL=yes" in stdout
            and not source_reachable
            and not public_reachable
            and provider_reachable
        )
        return {
            "sealed": sealed,
            "stdout": stdout,
            "probe_exit_code": (result or {}).get("exit_code"),
            "probe_error": (result or {}).get("error"),
            "probe_complete": all(
                marker in stdout
                for marker in (
                    "HOSTS=",
                    "CURL=",
                    "SOURCE_FETCH=",
                    "PUBLIC_FETCH=",
                    "PROVIDER_FETCH=",
                )
            ),
            "probe_url": probe_url,
            "provider_probe_url": provider_url,
            "provider_host": self._provider_host(),
            "hosts": list(self.BENCHMARK_SOURCE_HOSTS),
        }

    async def _seal_network(self, sandbox_id: str) -> dict[str, Any]:
        installed = await self.sandy.execute_command(sandbox_id, self._seal_script())
        verdict = await self._probe_network_seal(sandbox_id)
        verdict["install_exit_code"] = (installed or {}).get("exit_code")
        return verdict

    @staticmethod
    def _network_probe_exclusion_reason(
        verdict: dict[str, Any], integrity_reason: str
    ) -> str:
        """Separate an unreachable verifier from a reachable leaked network.

        A resource-exhausted sandbox can return ``sh: Cannot fork`` before the
        probe prints any of its markers. That proves neither that the seal held
        nor that it broke, so it is infrastructure and must never be reported
        as an integrity violation (or scored as a model failure).
        """
        if verdict.get("probe_exit_code") != 0 or not verdict.get("probe_complete"):
            return NETWORK_PROBE_EXCLUSION_REASON
        return integrity_reason

    async def _probe_local_answer_sources(self, sandbox_id: str) -> dict[str, Any]:
        """Prove the sandbox did not inherit a local OOLONG dataset cache."""
        result = await self.sandy.execute_command(
            sandbox_id,
            f"echo ANSWER=$(test -e {ANSWER_PATH} && echo present || echo absent); "
            "echo MATCHES_BEGIN; "
            "for root in /workspace /root/.cache; do "
            '[ ! -e "$root" ] || find "$root" -xdev '
            "\\( -iname '*oolong*' -o -path '*datasets--oolongbench--oolong-synth*' \\) "
            "2>/dev/null; done | head -20; echo MATCHES_END",
        )
        stdout = ((result or {}).get("stdout") or "").strip()
        between = stdout.partition("MATCHES_BEGIN")[2].partition("MATCHES_END")[0]
        matches = [line.strip() for line in between.splitlines() if line.strip()]
        return {
            "clean": (result or {}).get("exit_code") == 0
            and "ANSWER=absent" in stdout
            and not matches,
            "answer_path_absent": "ANSWER=absent" in stdout,
            "matches": matches,
            "stdout": stdout,
        }

    async def _probe_agent_runtime(
        self, sandbox_id: str, agent_name: str
    ) -> dict[str, Any]:
        """Fail fast when an arm's executable runtime is incomplete.

        Merely finding the ``chutescoder`` binary is insufficient for the RLM
        arm: its persistent kernel is a Python sidecar and the CLI can otherwise
        spend the whole item budget retrying a missing IPython dependency.  The
        baseline deliberately does not require that sidecar.
        """
        commands = {
            "prime-agent": "command -v prime-agent && prime-agent --version",
            "codex": "command -v codex && codex --version",
            "chutescoder-baseline": ("command -v chutescoder && chutescoder --version"),
            "chutescoder": (
                "command -v chutescoder && chutescoder --version && "
                "PYTHONPATH=/opt/chutescoder/python python3 -c "
                '"import IPython,dill,chutescoder_rlm; '
                "print(IPython.__version__,dill.__version__,"
                'chutescoder_rlm.__version__)"'
            ),
        }
        command = commands.get(agent_name)
        if command is None:
            return {
                "ready": False,
                "agent": agent_name,
                "error": f"Unsupported OOLONG agent arm: {agent_name}",
            }
        result = await self.sandy.execute_command(
            sandbox_id, command, timeout_ms=30_000
        )
        return {
            "ready": (result or {}).get("exit_code") == 0,
            "agent": agent_name,
            "exit_code": (result or {}).get("exit_code"),
            "stdout": ((result or {}).get("stdout") or "").strip(),
            "stderr": ((result or {}).get("stderr") or "").strip(),
            "error": (result or {}).get("error"),
        }

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
                    "sleep 1; kill -KILL $(cat /workspace/.chutes/agent.pid) "
                    "2>/dev/null || true; fi; "
                    "test -f /workspace/.chutes/agent.done || "
                    "echo 143 > /workspace/.chutes/agent.done",
                    timeout_ms=10_000,
                )
            except Exception as exc:
                logger.warning(
                    "Could not stop OOLONG agent before evidence retention",
                    item_id=item_id,
                    sandbox_id=sandbox_id,
                    error=str(exc),
                )
            agent_launch = state.get("agent_launch")
            if agent_launch is not None and not state.get("rollout_retained"):
                try:
                    await retain_sandy_agent_rollout(
                        self.sandy, sandbox_id, agent_launch
                    )
                    state["rollout_retained"] = True
                except Exception as exc:
                    state["rollout_retention_error"] = (
                        str(exc) or exc.__class__.__name__
                    )
                    logger.warning(
                        "Could not retain OOLONG rollout before evidence archive",
                        item_id=item_id,
                        sandbox_id=sandbox_id,
                        error=state["rollout_retention_error"],
                    )
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
                "exclusion_reason", EVIDENCE_RETENTION_EXCLUSION_REASON
            )
            if not result.error:
                result.error = (
                    "Agent evidence was not retained: "
                    f"{evidence.get('error') or evidence.get('status')}"
                )
        return result

    def _item_metadata(self, item: dict[str, Any], agent_name: str) -> dict[str, Any]:
        return {
            "agent": agent_name,
            "task": item["task"],
            "task_group": item["task_group"],
            "answer_type": item["answer_type"],
            "context_len": item["context_len"],
            "dataset": item["dataset"],
            "dataset_repo": item.get("dataset_repo", OOLONG_SYNTH_REPO),
            "dataset_revision": item.get("dataset_revision", OOLONG_SYNTH_REVISION),
            "dataset_split": item.get("dataset_split", OOLONG_SYNTH_SPLIT),
            "dataset_transport": item.get("dataset_transport"),
            "dataset_shard": item.get("dataset_shard"),
            "dataset_shard_row": item.get("dataset_shard_row"),
            "delivery": "file_in_sandbox",
            "comparable_to_single_shot_oolong": False,
        }

    def _excluded_result(
        self,
        *,
        item_id: str,
        item: dict[str, Any],
        prompt: str,
        agent_name: str,
        reason: str,
        error: str,
        start_time: float,
        usage: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ItemResult:
        return ItemResult(
            item_id=item_id,
            item_hash=self.compute_item_hash(item["question"]),
            prompt=prompt,
            expected=item["answer"],
            error=error,
            latency_ms=int((time.time() - start_time) * 1000),
            input_tokens=(usage or {}).get("input_tokens"),
            output_tokens=(usage or {}).get("output_tokens"),
            metadata={
                **self._item_metadata(item, agent_name),
                **(metadata or {}),
                "exclusion_reason": reason,
            },
        )

    async def evaluate_item(self, item_id: str) -> ItemResult:
        """Evaluate and attach evidence, including after worker cancellation."""
        self._new_item_observability(item_id)
        result = await self._evaluate_item(item_id)
        return self.attach_item_observability(result)

    async def _evaluate_item(self, item_id: str) -> ItemResult:
        item = await self._get_item(item_id)
        if not item:
            return ItemResult(
                item_id=item_id,
                error=f"Item {item_id} not found in pinned OOLONG dataset",
                metadata={"exclusion_reason": "infrastructure_dataset_load"},
            )

        agent_name = self._agent_name()
        prompt = self._build_prompt(item)
        start_time = time.time()
        sandbox_id: str | None = None
        agent_max_seconds = int(os.getenv("OOLONG_AGENTIC_MAX_SECONDS", "900"))

        try:
            # Sandy otherwise applies its 10-minute default TTL even though the
            # agent is allowed 15 minutes. The item deadline also includes five
            # minutes for upload, probes, evidence transfer, and teardown.
            sandbox_ttl_minutes = (
                self.get_item_timeout_seconds(item_id) + 59
            ) // 60 + 1
            sandbox_id = await self.sandy.create_sandbox(
                timeout_minutes=sandbox_ttl_minutes,
                requires_agent=True,
            )
            if not sandbox_id:
                return self._excluded_result(
                    item_id=item_id,
                    item=item,
                    prompt=prompt,
                    agent_name=agent_name,
                    reason=SANDBOX_CREATION_EXCLUSION_REASON,
                    error=self.sandy.last_error or "Could not create Sandy sandbox",
                    start_time=start_time,
                )
            try:
                corpus = await self._write_corpus(
                    sandbox_id, item["context_window_text"]
                )
                if not corpus.get("uploaded"):
                    return self._excluded_result(
                        item_id=item_id,
                        item=item,
                        prompt=prompt,
                        agent_name=agent_name,
                        reason=CORPUS_UPLOAD_EXCLUSION_REASON,
                        error=f"Failed to prove corpus upload: {corpus.get('error')}",
                        start_time=start_time,
                        metadata={"corpus": corpus},
                    )

                local_sources = await self._probe_local_answer_sources(sandbox_id)
                if not local_sources.get("clean"):
                    return self._excluded_result(
                        item_id=item_id,
                        item=item,
                        prompt=prompt,
                        agent_name=agent_name,
                        reason=LOCAL_ANSWER_LEAK_EXCLUSION_REASON,
                        error=(
                            "Refusing to score: a local OOLONG answer source or a "
                            f"pre-existing answer file was visible: {local_sources}"
                        ),
                        start_time=start_time,
                        metadata={
                            "corpus": corpus,
                            "local_answer_sources": local_sources,
                        },
                    )

                network_seal_before = await self._seal_network(sandbox_id)
                if not network_seal_before.get("sealed"):
                    return self._excluded_result(
                        item_id=item_id,
                        item=item,
                        prompt=prompt,
                        agent_name=agent_name,
                        reason=self._network_probe_exclusion_reason(
                            network_seal_before, NETWORK_SEAL_EXCLUSION_REASON
                        ),
                        error=(
                            "Refusing to score: OOLONG source network seal could not "
                            f"be proved: {network_seal_before}"
                        ),
                        start_time=start_time,
                        metadata={
                            "corpus": corpus,
                            "local_answer_sources": local_sources,
                            "network_seal_before": network_seal_before,
                        },
                    )

                runtime_preflight = await self._probe_agent_runtime(
                    sandbox_id, agent_name
                )
                if not runtime_preflight.get("ready"):
                    return self._excluded_result(
                        item_id=item_id,
                        item=item,
                        prompt=prompt,
                        agent_name=agent_name,
                        reason=AGENT_SETUP_EXCLUSION_REASON,
                        error=(f"Agent runtime preflight failed: {runtime_preflight}"),
                        start_time=start_time,
                        metadata={
                            "corpus": corpus,
                            "local_answer_sources": local_sources,
                            "network_seal_before": network_seal_before,
                            "agent_runtime_preflight": runtime_preflight,
                        },
                    )

                try:
                    agent_launch = await prepare_sandy_agent_launch(
                        client=self.client,
                        sandy=self.sandy,
                        sandbox_id=sandbox_id,
                        agent=agent_name,
                        model=self.model_slug,
                    )
                    self._item_observability[item_id]["agent_launch"] = agent_launch
                except Exception as exc:
                    return self._excluded_result(
                        item_id=item_id,
                        item=item,
                        prompt=prompt,
                        agent_name=agent_name,
                        reason=AGENT_SETUP_EXCLUSION_REASON,
                        error=f"Could not prepare Sandy agent: {exc}",
                        start_time=start_time,
                        metadata={
                            "corpus": corpus,
                            "local_answer_sources": local_sources,
                            "network_seal_before": network_seal_before,
                            "agent_runtime_preflight": runtime_preflight,
                        },
                    )

                self._item_observability[item_id]["agent_invoked"] = True
                agent_result = await self.sandy.run_agent(
                    sandbox_id,
                    agent=agent_name,
                    model=self.model_slug,
                    prompt=prompt,
                    max_duration=agent_max_seconds,
                    raw_prompt=True,
                    api_base_url=agent_launch.api_base_url,
                    env_vars=agent_launch.env_vars,
                )
                agent_summary = (agent_result or {}).get("summary") or {}

                rollout_error: str | None = None
                try:
                    await retain_sandy_agent_rollout(
                        self.sandy, sandbox_id, agent_launch
                    )
                    self._item_observability[item_id]["rollout_retained"] = True
                except Exception as exc:
                    rollout_error = str(exc) or exc.__class__.__name__
                    self._item_observability[item_id]["rollout_retention_error"] = (
                        rollout_error
                    )
                agent_usage = await collect_agent_usage(self.sandy, sandbox_id)
                usage_error = rollout_error
                if usage_error is None:
                    try:
                        validate_openrouter_agent_usage(agent_launch, agent_usage)
                    except RuntimeError as exc:
                        usage_error = str(exc)

                sandbox_alive = await self.sandy.sandbox_exists(sandbox_id)
                common_metadata = {
                    "agent_provider": agent_launch.metadata,
                    "agent_summary": agent_summary,
                    "agent_usage": agent_usage,
                    "agent_max_seconds": agent_max_seconds,
                    "sandbox_alive_at_exit": sandbox_alive,
                    "corpus": corpus,
                    "local_answer_sources": local_sources,
                    "network_seal_before": network_seal_before,
                    "agent_runtime_preflight": runtime_preflight,
                }

                if not agent_summary:
                    launch_exclusion = classify_bare_failure(
                        "Agent stream ended without a Sandy completion summary",
                        agent_summary,
                        agent_usage,
                        agent_invoked=True,
                    )
                    return self._excluded_result(
                        item_id=item_id,
                        item=item,
                        prompt=prompt,
                        agent_name=agent_name,
                        reason=launch_exclusion or "infrastructure_transport",
                        error="Agent stream ended without a Sandy completion summary",
                        start_time=start_time,
                        usage=agent_usage,
                        metadata=common_metadata,
                    )

                exclusion_reason, exit_note = classify_agent_exit(
                    agent_summary, agent_max_seconds, sandbox_alive
                )
                common_metadata["agent_exit_note"] = exit_note
                if exclusion_reason:
                    return self._excluded_result(
                        item_id=item_id,
                        item=item,
                        prompt=prompt,
                        agent_name=agent_name,
                        reason=exclusion_reason,
                        error=exit_note or exclusion_reason,
                        start_time=start_time,
                        usage=agent_usage,
                        metadata=common_metadata,
                    )
                if sandbox_alive is False:
                    return self._excluded_result(
                        item_id=item_id,
                        item=item,
                        prompt=prompt,
                        agent_name=agent_name,
                        reason="infrastructure_sandbox_gone",
                        error="Sandbox disappeared before the answer could be read",
                        start_time=start_time,
                        usage=agent_usage,
                        metadata=common_metadata,
                    )

                # The completion summary proves the agent process stopped. Its
                # rollout is now stable and can be transferred while we finish
                # the deterministic integrity and scoring checks.
                self._start_evidence_retention(item_id, sandbox_id)

                network_seal_after = await self._probe_network_seal(sandbox_id)
                common_metadata["network_seal_after"] = network_seal_after
                if not network_seal_after.get("sealed"):
                    return self._excluded_result(
                        item_id=item_id,
                        item=item,
                        prompt=prompt,
                        agent_name=agent_name,
                        reason=self._network_probe_exclusion_reason(
                            network_seal_after, NETWORK_SEAL_BROKEN_EXCLUSION_REASON
                        ),
                        error=(
                            "Refusing to score: the OOLONG source seal did not "
                            f"survive the agent run: {network_seal_after}"
                        ),
                        start_time=start_time,
                        usage=agent_usage,
                        metadata=common_metadata,
                    )

                # Do not turn a live-sandbox CLI crash into an infrastructure
                # exclusion merely because that crash also prevented accounting.
                # It is a harness robustness failure and stays scored. Successful
                # runs without exact OpenRouter usage are unreportable, however.
                agent_exit_code = agent_summary.get("exitCode")
                if usage_error and agent_exit_code in (0, None):
                    return self._excluded_result(
                        item_id=item_id,
                        item=item,
                        prompt=prompt,
                        agent_name=agent_name,
                        reason=TOKEN_ACCOUNTING_EXCLUSION_REASON,
                        error=usage_error,
                        start_time=start_time,
                        usage=agent_usage,
                        metadata=common_metadata,
                    )
                if usage_error:
                    common_metadata["token_accounting_error"] = usage_error

                read = await self.sandy.execute_command(
                    sandbox_id, f"cat {ANSWER_PATH} 2>/dev/null"
                )
                raw_answer = ((read or {}).get("stdout") or "").strip()
            finally:
                try:
                    await self._finish_evidence_retention(item_id, sandbox_id)
                finally:
                    terminated = await self.sandy.terminate_sandbox(sandbox_id)
                    if not terminated:
                        logger.error(
                            "Could not terminate OOLONG sandbox",
                            item_id=item_id,
                            sandbox_id=sandbox_id,
                            error=self.sandy.last_error,
                        )

            latency_ms = int((time.time() - start_time) * 1000)

            if not raw_answer:
                return ItemResult(
                    item_id=item_id,
                    item_hash=self.compute_item_hash(item["question"]),
                    prompt=prompt,
                    response="",
                    expected=item["answer"],
                    is_correct=False,
                    score=0.0,
                    error=f"Agent did not write {ANSWER_PATH}",
                    latency_ms=latency_ms,
                    input_tokens=agent_usage.get("input_tokens"),
                    output_tokens=agent_usage.get("output_tokens"),
                    metadata={
                        **self._item_metadata(item, agent_name),
                        **common_metadata,
                    },
                )

            # Identical scoring to single-shot OOLONG, so the arms differ only
            # in the harness -- including the raw/normalised pair, so the two
            # adapters cannot drift apart on the metric.
            extracted = _extract_answer(raw_answer)
            score_raw, correct_raw = score_answer(
                item["answer"], extracted, item["answer_type"]
            )
            normalized = _normalize_prediction(extracted)
            score, is_correct = score_answer(
                item["answer"], normalized, item["answer_type"]
            )

            return ItemResult(
                item_id=item_id,
                item_hash=self.compute_item_hash(item["question"]),
                prompt=prompt,
                response=raw_answer,
                expected=item["answer"],
                is_correct=is_correct,
                score=score,
                latency_ms=latency_ms,
                input_tokens=agent_usage.get("input_tokens"),
                output_tokens=agent_usage.get("output_tokens"),
                metadata={
                    **self._item_metadata(item, agent_name),
                    **common_metadata,
                    "extracted_answer": extracted,
                    "normalized_answer": normalized,
                    "score_raw": score_raw,
                    "is_correct_raw": correct_raw,
                    "score_normalized": score,
                    "normalization": "formatting_only",
                },
            )
        except Exception as exc:
            logger.error("OOLONG agentic item failed", item_id=item_id, error=str(exc))
            summary = locals().get("agent_summary")
            usage = locals().get("agent_usage")
            exclusion_reason = classify_bare_failure(
                str(exc),
                summary,
                usage,
                agent_invoked=bool(
                    self._item_observability.get(item_id, {}).get("agent_invoked")
                ),
            )
            if exclusion_reason is None and summary:
                # A transport failure after a completed agent is still
                # infrastructure (for example while fetching its answer).
                exclusion_reason = classify_bare_failure(str(exc), None)
            metadata = {
                **self._item_metadata(item, agent_name),
                "agent_summary": summary,
                "agent_usage": locals().get("agent_usage"),
                "corpus": locals().get("corpus"),
                "network_seal_before": locals().get("network_seal_before"),
                "exclusion_reason": exclusion_reason,
            }
            return ItemResult(
                item_id=item_id,
                item_hash=self.compute_item_hash(item["question"]),
                prompt=prompt,
                expected=item["answer"],
                error=str(exc),
                latency_ms=int((time.time() - start_time) * 1000),
                input_tokens=(locals().get("agent_usage") or {}).get("input_tokens"),
                output_tokens=(locals().get("agent_usage") or {}).get("output_tokens"),
                metadata=metadata,
            )
