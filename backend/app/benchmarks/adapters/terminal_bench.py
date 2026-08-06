"""Terminal-Bench Hard benchmark adapter."""
from __future__ import annotations

import base64
import json
import os
import shlex
import time
from typing import Any, AsyncIterator, Optional

import yaml

from app.benchmarks.base import BenchmarkAdapter, ItemResult
from app.benchmarks.registry import register_adapter
from app.benchmarks.utils import load_dataset_with_retry
from app.core.config import get_settings
from app.core.logging import get_logger
from app.services.sandy_service import SandyService

logger = get_logger(__name__)



# Cumulative token usage for a Sandy-driven agent has to be read back out of
# the sandbox: the tokens are spent by the CLI agent over a connection
# bench-runner never sees, so the run row otherwise reports 0 in / 0 out / $0.
# That makes token efficiency unmeasurable, and token efficiency is half of
# what a harness comparison is for -- an arm that wins by spending 6x the
# tokens has not obviously won.
#
# Both codex and chutescoder persist a rollout JSONL under
# <config_home>/sessions/YYYY/MM/DD/ and emit `token_count` events carrying
# TokenUsageInfo. The last one holds the session total, split into
# input / cached_input / output / reasoning_output.
AGENT_USAGE_PROBE = r"""
import glob, json, os, sys
PATTERNS = [
    "/root/.chutescoder/sessions/*/*/*/rollout-*.jsonl",
    "/root/.codex/sessions/*/*/*/rollout-*.jsonl",
]
files = []
for pattern in PATTERNS:
    files.extend(glob.glob(pattern))
if not files:
    print(json.dumps({"error": "no rollout found"}))
    raise SystemExit
path = max(files, key=os.path.getmtime)
last, seen = None, 0
for line in open(path, errors="replace"):
    try:
        obj = json.loads(line)
    except Exception:
        continue
    payload = obj.get("payload") or {}
    if payload.get("type") == "token_count":
        seen += 1
        if payload.get("info"):
            last = payload["info"]
if not last:
    print(json.dumps({"error": "no token_count events", "events_seen": seen}))
    raise SystemExit
total = last.get("total_token_usage") or {}
print(json.dumps({
    "rollout": os.path.basename(path),
    "token_count_events": seen,
    "input_tokens": total.get("input_tokens"),
    "cached_input_tokens": total.get("cached_input_tokens"),
    "output_tokens": total.get("output_tokens"),
    "reasoning_output_tokens": total.get("reasoning_output_tokens"),
    "total_tokens": total.get("total_tokens"),
}))
"""

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


@register_adapter("terminal_bench_hard")
class TerminalBenchHardAdapter(BenchmarkAdapter):
    """
    Terminal-Bench Hard adapter.

    Uses the official task archive and docker-based evaluation harness.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._items: list[dict[str, Any]] = []
        self.sandy = SandyService()

    def get_name(self) -> str:
        return "terminal_bench_hard"

    def get_display_name(self) -> str:
        return "Terminal-Bench Hard"

    def requires_setup(self) -> bool:
        return False

    def get_setup_notes(self) -> Optional[str]:
        return None

    def supports_subset(self) -> bool:
        return True

    async def get_total_items(self) -> int:
        if not self._items:
            await self.preload()
        return len(self._items)

    async def preload(self) -> None:
        """Load Terminal-Bench dataset."""
        if self._items:
            return

        try:
            logger.info("Loading Terminal-Bench dataset")
            dataset = await load_dataset_with_retry(
                "ia03/terminal-bench",
                split="test",
                token=os.environ.get("HF_TOKEN"),
            )
            self._items = []
            for i, item in enumerate(dataset):
                task_yaml = item.get("task_yaml") or ""
                instruction = ""
                try:
                    parsed_yaml = yaml.safe_load(task_yaml) if task_yaml else {}
                    instruction = parsed_yaml.get("instruction", "")
                except Exception:
                    parsed_yaml = {}
                self._items.append(
                    {
                        "id": str(i),
                        "task_id": item.get("task_id"),
                        "task_yaml": task_yaml,
                        "instruction": instruction,
                        "archive": item.get("archive"),
                        "difficulty": item.get("difficulty", ""),
                        "parsed_yaml": parsed_yaml,
                        "max_agent_timeout_sec": item.get("max_agent_timeout_sec"),
                        "max_test_timeout_sec": item.get("max_test_timeout_sec"),
                    }
                )
            logger.info("Loaded %s Terminal-Bench items", len(self._items))
        except Exception as e:
            logger.error("Failed to load Terminal-Bench", error=str(e))
            self._items = []
            raise

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

    async def _run_terminal_bench(self, sandbox_id: str, task_id: str) -> dict[str, Any]:
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
        image_name = f"tbench_{task_id}".lower()
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
                "T_BENCH_TASK_DOCKER_CLIENT_IMAGE_NAME": "client",
                "T_BENCH_TASK_DOCKER_NAME_PREFIX": f"tbench_{task_id}",
                "T_BENCH_TASK_DOCKER_CLIENT_CONTAINER_NAME": f"tbench_{task_id}_client",
                "T_BENCH_TASK_LOGS_PATH": logs_dir,
                "T_BENCH_CONTAINER_LOGS_PATH": "/var/log/tbench",
                "T_BENCH_TASK_AGENT_LOGS_PATH": logs_dir,
                "T_BENCH_CONTAINER_AGENT_LOGS_PATH": "/var/log/tbench/agent",
                "T_BENCH_TEST_DIR": "/tests",
                "DOCKER_HOST": "unix:///var/run/docker.sock",
            }
            env_lines = [f"{key}={value}" for key, value in env.items()]
            await self.sandy.write_file(
                sandbox_id,
                f"{task_dir}/.env",
                "\n".join(env_lines) + "\n",
            )
            up_cmd = f"{compose_cmd} --env-file .env -f {compose_path} up --build -d"
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
            cleanup_cmd = f"{compose_cmd} -f {compose_path} down"
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
            run_result = await self.sandy.execute_command(
                sandbox_id,
                f"docker run -d --name {container_name} {image_name} sleep infinity",
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

        probe = (
            "curl -s -m 8 -o /dev/null -w '%{http_code}' "
            "https://raw.githubusercontent.com/harbor-framework/terminal-bench/main/README.md "
            "|| echo BLOCKED"
        )
        sandbox_probe = await self.sandy.execute_command(sandbox_id, probe)
        container_probe = await self.sandy.execute_command(
            sandbox_id, f"docker exec {container_name} sh -c {shlex.quote(probe)}"
        )

        def _blocked(result: dict) -> bool:
            out = ((result or {}).get("stdout") or "").strip()
            # A blackholed host gives a connection failure, not a 200.
            return out != "200"

        verdict["sandbox_stdout"] = ((sandbox_probe or {}).get("stdout") or "").strip()
        verdict["container_stdout"] = ((container_probe or {}).get("stdout") or "").strip()
        verdict["sandbox_blocked"] = _blocked(sandbox_probe)
        verdict["container_blocked"] = _blocked(container_probe)
        verdict["sealed"] = verdict["sandbox_blocked"] and verdict["container_blocked"]
        return verdict

    async def _collect_agent_usage(self, sandbox_id: str) -> dict:
        """Read the agent's own cumulative token usage out of the sandbox.

        Must run before the sandbox is terminated. Returns `{"error": ...}`
        rather than an estimate when the rollout is missing or carries no
        usage events -- an invented number here would be worse than a gap,
        because it would silently become a token-efficiency claim.
        """
        try:
            encoded = base64.b64encode(AGENT_USAGE_PROBE.encode()).decode("ascii")
            result = await self.sandy.execute_command(
                sandbox_id,
                f"echo {encoded} | base64 -d > /tmp/_usage_probe.py "
                "&& python3 /tmp/_usage_probe.py",
            )
            raw = ((result or {}).get("stdout") or "").strip()
            if not raw:
                return {"error": "usage probe produced no output"}
            return json.loads(raw)
        except Exception as exc:
            return {"error": f"usage probe failed: {exc}"}

    async def evaluate_item(self, item_id: str) -> ItemResult:
        """Evaluate a single Terminal-Bench item."""
        if not self._items:
            await self.preload()

        item = next((i for i in self._items if i["id"] == item_id), None)
        if not item:
            return ItemResult(item_id=item_id, error=f"Item {item_id} not found")

        instruction = item.get("instruction") or ""
        task_yaml = item.get("task_yaml") or ""
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

            # Terminal-Bench requires Docker socket access for running docker-compose
            sandbox_id = await self.sandy.create_sandbox(enable_docker_socket=True)
            if not sandbox_id:
                sandbox_error = self.sandy.last_error or "Could not create sandbox"
                return ItemResult(item_id=item_id, error=sandbox_error)

            try:
                extracted = await self._extract_archive(sandbox_id, archive)
                if not extracted:
                    return ItemResult(item_id=item_id, error="Failed to extract task archive")

                setup_result = await self._run_terminal_bench(sandbox_id, item.get("task_id") or "task")
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

                    agent_timeout = int((item.get("max_agent_timeout_sec") or 180) * 1000)
                    test_timeout = int((item.get("max_test_timeout_sec") or 300) * 1000)

                    settings = get_settings()
                    agent_api_key = self.client.get_api_key() or settings.chutes_api_key
                    agent_env_vars = {
                        "CHUTES_API_KEY": agent_api_key,
                    }

                    # Which Sandy CLI agent drives the task. Selectable so the
                    # same benchmark can be run as a paired A/B of harnesses on
                    # one model -- "codex" (upstream) vs "chutescoder" (the
                    # Chutes fork with the RLM harness). Set per run with
                    #   "config": {"terminal_bench": {"agent": "chutescoder"}}
                    # which the worker already attaches as adapter.run_config.
                    agent_name = (
                        (getattr(self, "run_config", None) or {})
                        .get("terminal_bench", {})
                        .get("agent")
                        or os.getenv("TERMINAL_BENCH_AGENT")
                        or "codex"
                    )
                    agent_result = await self.sandy.run_agent(
                        sandbox_id,
                        agent=agent_name,
                        model=self.model_slug,
                        prompt=prompt + f"\nContainer name: {container_name}\n",
                        max_duration=max(60, int(agent_timeout / 1000)),
                        raw_prompt=True,
                        env_vars=agent_env_vars,
                    )
                    agent_usage = await self._collect_agent_usage(sandbox_id)
                    agent_summary = agent_result.get("summary") or {}
                    agent_events = agent_result.get("events") or []
                    agent_output = next(
                        (event.get("text") for event in reversed(agent_events) if event.get("type") == "output"),
                        "",
                    )
                    latency_ms = int((time.time() - start_time) * 1000)

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
                            error="Agent did not write task/solution.sh",
                            latency_ms=latency_ms,
                            metadata={
                                "task_id": item.get("task_id"),
                                "agent": agent_name,
                                "agent_summary": agent_summary,
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

                    # Test phase: copy tests and run
                    await self.sandy.execute_command(
                        sandbox_id,
                        f"docker cp task/tests {container_name}:/tests",
                    )
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
                        await self.sandy.write_file(sandbox_id, "default-run-tests.sh", default_script)
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
                    error = None if is_correct else (test_result.get("stderr") or test_result.get("error"))

                    return ItemResult(
                        item_id=item_id,
                        item_hash=self.compute_item_hash(item.get("task_id")),
                        prompt=prompt,
                        response=agent_output,
                        expected="[Tests passed]",
                        is_correct=is_correct,
                        score=1.0 if is_correct else 0.0,
                        latency_ms=latency_ms,
                        judge_output={
                            "setup": setup_result,
                            "agent_summary": agent_summary,
                            "agent_exec": agent_exec,
                            "test_result": test_result,
                        },
                        error=error,
                        input_tokens=agent_usage.get("input_tokens"),
                        output_tokens=agent_usage.get("output_tokens"),
                        metadata={
                            "task_id": item.get("task_id"),
                            "difficulty": item.get("difficulty"),
                            "agent": agent_name,
                            "agent_usage": agent_usage,
                            # Recorded per item so a reviewer can confirm the
                            # run was sealed without taking it on trust.
                            "seal": seal,
                            "agent_summary": agent_summary,
                            "agent_output_excerpt": agent_output[:2000] if agent_output else "",
                            "task_yaml": task_yaml,
                        },
                    )
                finally:
                    if cleanup_cmd:
                        await self.sandy.execute_command(
                            sandbox_id, cleanup_cmd, cwd=cleanup_cwd
                        )
            finally:
                await self.sandy.terminate_sandbox(sandbox_id)

        except Exception as e:
            logger.error("Terminal-Bench evaluation failed", item_id=item_id, error=str(e))
            return ItemResult(
                item_id=item_id,
                prompt=prompt,
                response=locals().get("agent_output", "") or "",
                error=str(e),
                metadata={"task_id": item.get("task_id")},
            )
