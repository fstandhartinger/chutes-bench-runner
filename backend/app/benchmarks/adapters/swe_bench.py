"""SWE-Bench Pro benchmark adapter."""
from __future__ import annotations

import asyncio
import ast
import json
import os
import re
import shlex
import time
from typing import Any, AsyncIterator, Optional

from app.benchmarks.base import BenchmarkAdapter, ItemResult
from app.benchmarks.registry import register_adapter
from app.benchmarks.utils import download_http_file, load_dataset_with_retry
from app.core.config import get_settings
from app.core.logging import get_logger
from app.services.sandy_service import SandyService

logger = get_logger(__name__)

SWE_BENCH_REPO = "https://raw.githubusercontent.com/scaleapi/SWE-bench_Pro-os/main"

_DOCKER_RUN_OUTPUT_PREVIEW_CHARS = 4000

# SWE-Bench Pro leaderboard constraints (Scale).
_MAX_AGENT_STEPS = 150
_MAX_AGENT_TOTAL_TOKENS = 1_000_000

# Keep agent/tool I/O bounded so we don't explode context windows.
# PATCH actions can be moderately large; keep this high enough that unified diffs fit
# without getting cut mid-fence, but still bounded to avoid runaway cost.
_MAX_AGENT_RESPONSE_TOKENS = 4096
_MAX_TOOL_OUTPUT_CHARS = 12000
_DEFAULT_FILE_READ_LINES = 200
_MAX_SEARCH_LINES = 200


@register_adapter("swe_bench_pro")
class SWEBenchProAdapter(BenchmarkAdapter):
    """
    SWE-Bench Pro adapter.

    Uses the official evaluation scripts with Docker Hub images.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._items: list[dict[str, Any]] = []
        self.sandy = SandyService()
        self._sandbox_id: Optional[str] = None

    def get_name(self) -> str:
        return "swe_bench_pro"

    def get_display_name(self) -> str:
        return "SWE-Bench Pro"

    def requires_setup(self) -> bool:
        return False

    def get_setup_notes(self) -> Optional[str]:
        return None

    def supports_subset(self) -> bool:
        return True

    def get_item_timeout_seconds(self) -> Optional[int]:
        return 3600

    async def get_total_items(self) -> int:
        if not self._items:
            await self.preload()
        return len(self._items)

    async def preload(self) -> None:
        """Load SWE-Bench Pro dataset."""
        if self._items:
            return

        try:
            logger.info("Loading SWE-Bench Pro dataset")
            dataset = await load_dataset_with_retry(
                "ScaleAI/SWE-bench_Pro",
                split="test",
                token=os.environ.get("HF_TOKEN"),
            )
            self._items = []
            for i, item in enumerate(dataset):
                self._items.append(
                    {
                        "id": str(i),
                        "instance_id": str(item.get("instance_id", "")),
                        "repo": str(item.get("repo", "")),
                        "problem_statement": str(item.get("problem_statement", "")),
                        "before_repo_set_cmd": str(item.get("before_repo_set_cmd", "")),
                        "selected_test_files_to_run": str(item.get("selected_test_files_to_run", "")),
                        "base_commit": str(item.get("base_commit", "")),
                        "dockerhub_tag": str(item.get("dockerhub_tag", "")),
                        "fail_to_pass": str(item.get("fail_to_pass", "")),
                        "pass_to_pass": str(item.get("pass_to_pass", "")),
                    }
                )
            logger.info("Loaded %s SWE-Bench Pro items", len(self._items))
        except Exception as e:
            logger.error("Failed to load SWE-Bench Pro", error=str(e))
            self._items = []
            raise

    async def enumerate_items(self) -> AsyncIterator[str]:
        if not self._items:
            await self.preload()
        for item in self._items:
            yield item["id"]

    def _download_run_script(self, instance_id: str, filename: str) -> str:
        path = download_http_file(
            f"{SWE_BENCH_REPO}/run_scripts/{instance_id}/{filename}",
            cache_subdir=f"swe_bench/{instance_id}",
            filename=filename,
        )
        return path.read_text(encoding="utf-8")

    def _download_dockerfile(self, folder: str, instance_id: str) -> str:
        path = download_http_file(
            f"{SWE_BENCH_REPO}/dockerfiles/{folder}/{instance_id}/Dockerfile",
            cache_subdir=f"swe_bench/{instance_id}",
            filename=f"{folder}.Dockerfile",
        )
        return path.read_text(encoding="utf-8")

    def _get_dockerhub_image_uri(self, uid: str, repo_name: str, dockerhub_username: str) -> str:
        repo_base, repo_name_only = repo_name.lower().split("/")
        hsh = uid.replace("instance_", "")
        if uid == "instance_element-hq__element-web-ec0f940ef0e8e3b61078f145f34dc40d1938e6c5-vnan":
            repo_name_only = "element-web"
        elif "element-hq" in repo_name.lower() and "element-web" in repo_name.lower():
            repo_name_only = "element"
            if hsh.endswith("-vnan"):
                hsh = hsh[:-5]
        elif hsh.endswith("-vnan"):
            hsh = hsh[:-5]
        tag = f"{repo_base}.{repo_name_only}-{hsh}"
        if len(tag) > 128:
            tag = tag[:128]
        return f"{dockerhub_username}/sweap-images:{tag}"

    def _create_entryscript(self, sample: dict[str, Any]) -> str:
        # Align with the official SWE-Bench Pro harness: `before_repo_set_cmd` contains multiple lines,
        # but only the *final* command (typically `git checkout <sha> -- <test files>`) should run
        # after applying the model patch. Running the full block would reset/clean the repo again
        # and wipe the patch we just applied.
        before_repo_set_cmd = sample.get("before_repo_set_cmd", "").strip()
        if before_repo_set_cmd:
            before_repo_set_cmd = before_repo_set_cmd.splitlines()[-1].strip()
        try:
            selected = ast.literal_eval(sample.get("selected_test_files_to_run", "[]"))
        except Exception:
            selected = []
        selected_test_files_to_run = ",".join(selected)
        base_commit = sample.get("base_commit", "")
        base_dockerfile = self._download_dockerfile("base_dockerfile", sample["instance_id"])
        instance_dockerfile = self._download_dockerfile("instance_dockerfile", sample["instance_id"])

        env_cmds: list[str] = []
        for dockerfile_content in (base_dockerfile, instance_dockerfile):
            for line in dockerfile_content.splitlines():
                line = line.strip()
                if line.startswith("ENV"):
                    env_cmds.append(line.replace("ENV", "export", 1))
        env_block = "\n".join(env_cmds)

        entry_script = f"""
{env_block}
# apply patch
cd /app
git reset --hard {base_commit}
git checkout {base_commit}
git apply -v /workspace/patch.diff
{before_repo_set_cmd}
# run test and save stdout and stderr to separate files
bash /workspace/run_script.sh {selected_test_files_to_run} > /workspace/stdout.log 2> /workspace/stderr.log
# run parsing script
python /workspace/parser.py /workspace/stdout.log /workspace/stderr.log /workspace/output.json
""".strip()
        return entry_script + "\n"

    async def _read_file(self, sandbox_id: str, path: str) -> str:
        result = await self.sandy.execute_command(sandbox_id, f"cat {path}")
        return result.get("stdout", "")

    async def _get_env(self, sandbox_id: str, key: str) -> str:
        result = await self.sandy.execute_command(sandbox_id, f"printenv {key}")
        return result.get("stdout", "")

    async def _get_host_volume(self, sandbox_id: str) -> Optional[str]:
        result = await self.sandy.execute_command(sandbox_id, "printenv SANDY_HOST_VOLUME")
        host_volume = (result.get("stdout") or "").strip()
        if host_volume:
            return host_volume
        settings = get_settings()
        fallback_root = (settings.sandy_volume_root or "").rstrip("/")
        if fallback_root:
            return f"{fallback_root}/{sandbox_id}"
        return None

    def _agent_system_prompt(self) -> str:
        return (
            "You are an autonomous software engineering agent for SWE-Bench Pro.\n"
            "You are operating inside a Linux sandbox with a git repository checked out at /workspace/repo.\n"
            "You MUST respond with exactly ONE action per turn in the format below and nothing else.\n"
            "Do NOT repeat the action block. Do NOT add any explanation.\n\n"
            "ACTION: RUN\n"
            "COMMAND: <shell command>\n\n"
            "ACTION: READ\n"
            "PATH: <repo-relative path>\n"
            "START_LINE: <optional integer>\n"
            "END_LINE: <optional integer>\n\n"
            "ACTION: SEARCH\n"
            "PATTERN: <regex>\n"
            "PATH: <optional repo-relative path or '.'>\n\n"
            "ACTION: PATCH\n"
            "```diff\n"
            "<unified diff to apply to the repo>\n"
            "```\n\n"
            "ACTION: DONE\n\n"
            "Rules:\n"
            "- Use repo-relative paths (no leading '/').\n"
            "- Do NOT open interactive tools/editors (vim, nano, less, more).\n"
            "- Keep changes minimal and focused.\n"
            "- Prefer small incremental patches.\n"
            "- For PATCH: the diff MUST be complete and MUST include the closing ``` fence.\n"
            "- If you run tests, run only what is necessary.\n"
        )

    @staticmethod
    def _truncate_for_prompt(value: str, limit: int = _MAX_TOOL_OUTPUT_CHARS) -> str:
        value = value or ""
        if len(value) <= limit:
            return value
        return value[:limit].rstrip() + "…"

    @staticmethod
    def _usage_tokens(metadata: dict[str, Any]) -> tuple[int, int]:
        usage = metadata.get("usage") if isinstance(metadata, dict) else None
        if not isinstance(usage, dict):
            return 0, 0
        input_tokens = (
            usage.get("prompt_tokens")
            or usage.get("input_tokens")
            or usage.get("inputTokens")
            or 0
        )
        output_tokens = (
            usage.get("completion_tokens")
            or usage.get("output_tokens")
            or usage.get("outputTokens")
            or 0
        )
        try:
            return int(input_tokens or 0), int(output_tokens or 0)
        except Exception:
            return 0, 0

    def _parse_agent_action(self, text: str) -> dict[str, Any]:
        # Enforce "exactly one action" to prevent ambiguous multi-block outputs that are
        # easy for models to produce when truncation happens.
        action_headers = re.findall(r"^\\s*ACTION:\\s*\\w+\\s*$", text or "", re.IGNORECASE | re.MULTILINE)
        if len(action_headers) != 1:
            raise ValueError("Response must contain exactly ONE ACTION block")
        match = re.search(r"^\\s*ACTION:\\s*(\\w+)\\s*$", text or "", re.IGNORECASE | re.MULTILINE)
        if not match:
            raise ValueError("Missing ACTION header")
        action = match.group(1).strip().upper()

        if action == "DONE":
            return {"action": "DONE"}

        if action == "RUN":
            cmd_match = re.search(r"^\\s*COMMAND:\\s*(.+)$", text, re.IGNORECASE | re.MULTILINE)
            if not cmd_match:
                raise ValueError("Missing COMMAND for RUN action")
            return {"action": "RUN", "command": cmd_match.group(1).strip()}

        if action == "READ":
            path_match = re.search(r"^\\s*PATH:\\s*(.+)$", text, re.IGNORECASE | re.MULTILINE)
            if not path_match:
                raise ValueError("Missing PATH for READ action")
            path = path_match.group(1).strip().lstrip("/")
            start_match = re.search(r"^\\s*START_LINE:\\s*(\\d+)\\s*$", text, re.IGNORECASE | re.MULTILINE)
            end_match = re.search(r"^\\s*END_LINE:\\s*(\\d+)\\s*$", text, re.IGNORECASE | re.MULTILINE)
            start_line = int(start_match.group(1)) if start_match else 1
            end_line = int(end_match.group(1)) if end_match else start_line + _DEFAULT_FILE_READ_LINES - 1
            start_line = max(1, start_line)
            end_line = max(start_line, end_line)
            # Clamp to a sane window.
            if end_line - start_line + 1 > _DEFAULT_FILE_READ_LINES:
                end_line = start_line + _DEFAULT_FILE_READ_LINES - 1
            return {"action": "READ", "path": path, "start_line": start_line, "end_line": end_line}

        if action == "SEARCH":
            pattern_match = re.search(r"^\\s*PATTERN:\\s*(.+)$", text, re.IGNORECASE | re.MULTILINE)
            if not pattern_match:
                raise ValueError("Missing PATTERN for SEARCH action")
            path_match = re.search(r"^\\s*PATH:\\s*(.+)$", text, re.IGNORECASE | re.MULTILINE)
            scope = (path_match.group(1).strip() if path_match else ".").lstrip("/") or "."
            return {"action": "SEARCH", "pattern": pattern_match.group(1).strip(), "path": scope}

        if action == "PATCH":
            diff_match = re.search(r"```diff\\s*(.*?)```", text, re.IGNORECASE | re.DOTALL)
            if not diff_match:
                raise ValueError("Missing ```diff fenced block for PATCH action")
            diff = diff_match.group(1).strip("\n") + "\n"
            return {"action": "PATCH", "diff": diff}

        raise ValueError(f"Unknown ACTION: {action}")

    async def _exec_repo(self, sandbox_id: str, command: str, *, timeout_ms: int = 300000) -> dict[str, Any]:
        return await self.sandy.execute_command(
            sandbox_id,
            command,
            cwd="/workspace/repo",
            timeout_ms=timeout_ms,
        )

    async def _execute_agent_action(self, sandbox_id: str, action: dict[str, Any]) -> str:
        kind = action.get("action")
        if kind == "RUN":
            res = await self._exec_repo(sandbox_id, str(action.get("command") or ""))
            stdout = self._truncate_for_prompt(res.get("stdout") or "")
            stderr = self._truncate_for_prompt(res.get("stderr") or "")
            return f"OBSERVATION: exit_code={res.get('exit_code')}\\nSTDOUT:\\n{stdout}\\n\\nSTDERR:\\n{stderr}"

        if kind == "READ":
            path = str(action.get("path") or "").lstrip("/")
            start_line = int(action.get("start_line") or 1)
            end_line = int(action.get("end_line") or start_line + _DEFAULT_FILE_READ_LINES - 1)
            cmd = f"sed -n '{start_line},{end_line}p' {shlex.quote(path)}"
            res = await self._exec_repo(sandbox_id, cmd)
            stdout = self._truncate_for_prompt(res.get("stdout") or "")
            stderr = self._truncate_for_prompt(res.get("stderr") or "")
            return (
                f"OBSERVATION: READ {path} lines {start_line}-{end_line} exit_code={res.get('exit_code')}\\n"
                f"{stdout}\\n\\nSTDERR:\\n{stderr}"
            )

        if kind == "SEARCH":
            pattern = str(action.get("pattern") or "")
            scope = str(action.get("path") or ".").lstrip("/") or "."
            cmd = (
                f"rg -n --no-heading --line-number -S {shlex.quote(pattern)} {shlex.quote(scope)} "
                f"| head -n {_MAX_SEARCH_LINES} || true"
            )
            res = await self._exec_repo(sandbox_id, cmd)
            stdout = self._truncate_for_prompt(res.get("stdout") or "")
            stderr = self._truncate_for_prompt(res.get("stderr") or "")
            return (
                f"OBSERVATION: SEARCH pattern={pattern!r} scope={scope!r} exit_code={res.get('exit_code')}\\n"
                f"{stdout}\\n\\nSTDERR:\\n{stderr}"
            )

        if kind == "PATCH":
            diff = str(action.get("diff") or "")
            if len(diff) > 200000:
                return "OBSERVATION: PATCH too large; please send a smaller, focused diff."
            await self.sandy.write_file(sandbox_id, "agent.patch", diff)
            check = await self._exec_repo(sandbox_id, "git apply --check /workspace/agent.patch")
            if check.get("exit_code") != 0:
                stderr = self._truncate_for_prompt(check.get("stderr") or check.get("stdout") or "")
                return f"OBSERVATION: PATCH failed pre-check\\n{stderr}"
            apply_res = await self._exec_repo(sandbox_id, "git apply -v --whitespace=nowarn /workspace/agent.patch")
            stdout = self._truncate_for_prompt(apply_res.get("stdout") or "")
            stderr = self._truncate_for_prompt(apply_res.get("stderr") or "")
            return (
                f"OBSERVATION: PATCH applied exit_code={apply_res.get('exit_code')}\\nSTDOUT:\\n{stdout}\\n\\nSTDERR:\\n{stderr}"
            )

        return f"OBSERVATION: Unknown action kind: {kind!r}"

    async def _run_agent_loop(
        self,
        sandbox_id: str,
        *,
        prompt: str,
        item: dict[str, Any],
    ) -> tuple[str, dict[str, Any], int, int]:
        start = time.monotonic()
        input_tokens = 0
        output_tokens = 0
        trace: list[dict[str, Any]] = []

        status_res = await self._exec_repo(sandbox_id, "git status -sb || true")
        ls_res = await self._exec_repo(sandbox_id, "ls -la || true")
        status_txt = self._truncate_for_prompt(status_res.get("stdout") or "")
        ls_txt = self._truncate_for_prompt(ls_res.get("stdout") or "")

        extra_context = (
            "\n\nDataset fields (for reference):\n"
            f"- selected_test_files_to_run: {item.get('selected_test_files_to_run')}\n"
            f"- fail_to_pass: {item.get('fail_to_pass')}\n"
            f"- pass_to_pass: {item.get('pass_to_pass')}\n"
            f"- before_repo_set_cmd: {item.get('before_repo_set_cmd')}\n"
        )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._agent_system_prompt()},
            {
                "role": "user",
                "content": (
                    prompt
                    + extra_context
                    + "\n\nInitial repo status:\n"
                    + status_txt
                    + "\n\nTop-level listing:\n"
                    + ls_txt
                    + f"\n\nConstraints: max_steps={_MAX_AGENT_STEPS}, token_limit={_MAX_AGENT_TOTAL_TOKENS}."
                    + "\nRespond with an ACTION now."
                ),
            },
        ]

        last_assistant = ""
        for step in range(1, _MAX_AGENT_STEPS + 1):
            if (input_tokens + output_tokens) >= _MAX_AGENT_TOTAL_TOKENS:
                break
            # Conservative time cap for the agent loop so we still have time to run the harness.
            if time.monotonic() - start > 1800:
                break

            assistant_text, meta = await self.client.get_completion_messages(
                self.model_slug,
                messages,
                temperature=0.0,
                max_tokens=_MAX_AGENT_RESPONSE_TOKENS,
            )
            last_assistant = assistant_text
            in_toks, out_toks = self._usage_tokens(meta)
            input_tokens += in_toks
            output_tokens += out_toks

            trace.append(
                {
                    "step": step,
                    "action_raw": self._truncate_for_prompt(assistant_text, 4000),
                    "usage": meta.get("usage") if isinstance(meta, dict) else None,
                }
            )

            messages.append({"role": "assistant", "content": assistant_text})

            try:
                action = self._parse_agent_action(assistant_text)
            except Exception as exc:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "OBSERVATION: Invalid action format. "
                            f"Error: {exc}. "
                            "Please respond again with a single ACTION in the required format."
                        ),
                    }
                )
                continue

            if action.get("action") == "DONE":
                break

            observation = await self._execute_agent_action(sandbox_id, action)
            observation = (
                observation
                + f"\n\nBudget: step={step}/{_MAX_AGENT_STEPS}, "
                f"tokens_used={input_tokens + output_tokens}/{_MAX_AGENT_TOTAL_TOKENS}."
            )
            messages.append({"role": "user", "content": observation})

        meta_out = {
            "agent_steps": len(trace),
            "agent_input_tokens": input_tokens,
            "agent_output_tokens": output_tokens,
            "agent_trace_tail": trace[-25:],
            "agent_wall_time_s": round(time.monotonic() - start, 3),
        }
        return last_assistant, meta_out, input_tokens, output_tokens

    async def _ensure_sandbox(self) -> Optional[str]:
        if self._sandbox_id:
            exists = await self.sandy.sandbox_exists(self._sandbox_id)
            if exists:
                return self._sandbox_id
            self._sandbox_id = None

        self._sandbox_id = await self.sandy.create_sandbox(enable_docker_socket=True)
        return self._sandbox_id

    async def cleanup(self) -> None:
        sandbox_id = self._sandbox_id
        self._sandbox_id = None
        if sandbox_id:
            await self.sandy.terminate_sandbox(sandbox_id)

    async def evaluate_item(self, item_id: str) -> ItemResult:
        """Evaluate a single SWE-Bench Pro item."""
        if not self._items:
            await self.preload()

        item = next((i for i in self._items if i["id"] == item_id), None)
        if not item:
            return ItemResult(item_id=item_id, error=f"Item {item_id} not found")

        repo = item.get("repo", "")
        base_commit = item.get("base_commit", "")
        instance_id = item.get("instance_id") or ""

        prompt = (
            "You are a software engineer fixing a bug in a GitHub repository. "
            "The repository is cloned at /workspace/repo and checked out to the base commit. "
            "Make the required code changes in the repo.\n"
            "Use the PATCH action (a unified diff) to apply your changes.\n"
            "When the fix is complete, respond with ACTION: DONE.\n\n"
            f"Repository: {repo or 'unknown'}\n"
            f"Base Commit: {base_commit}\n"
            f"Issue Description:\n{item.get('problem_statement')}\n"
        )

        if not repo or not base_commit:
            return ItemResult(
                item_id=item_id,
                error="Missing repo or base commit",
                metadata={"instance_id": instance_id, "repo": repo},
            )

        start_time = time.time()
        agent_output = ""
        agent_meta: dict[str, Any] = {}
        agent_input_tokens = 0
        agent_output_tokens = 0
        agent_name = "bench_runner_loop"

        try:
            # SWE-Bench requires Docker socket access for running docker pull/run.
            # Keep a single sandbox for the full benchmark run to reduce upstream churn.
            sandbox_id = await self._ensure_sandbox()
            if not sandbox_id:
                sandbox_error = self.sandy.last_error or "Could not create sandbox"
                return ItemResult(
                    item_id=item_id,
                    error=sandbox_error,
                    metadata={"instance_id": instance_id, "repo": repo},
                )

            await self.sandy.execute_command(
                sandbox_id,
                "rm -rf /workspace/repo && rm -f /workspace/patch.diff /workspace/stdout.log /workspace/stderr.log /workspace/output.json || true",
            )

            # Ensure git is available
            git_check = await self.sandy.execute_command(sandbox_id, "git --version")
            if git_check.get("exit_code") != 0:
                await self.sandy.execute_command(
                    sandbox_id,
                    "apt-get update && apt-get install -y git",
                    timeout_ms=600000,
                )

            clone_cmd = f"rm -rf /workspace/repo && git clone https://github.com/{repo}.git /workspace/repo"
            clone_result: Optional[dict[str, Any]] = None
            for attempt in range(1, 4):
                clone_result = await self.sandy.execute_command(
                    sandbox_id,
                    clone_cmd,
                    timeout_ms=900000,
                )
                if clone_result.get("exit_code") == 0:
                    break
                if attempt < 3:
                    await asyncio.sleep(min(5 * attempt, 15))
            if not clone_result or clone_result.get("exit_code") != 0:
                return ItemResult(
                    item_id=item_id,
                    error=(clone_result or {}).get("stderr") or "Failed to clone repo",
                    metadata={"instance_id": instance_id, "repo": repo},
                )

            checkout_result = await self.sandy.execute_command(
                sandbox_id,
                f"cd /workspace/repo && git checkout {base_commit}",
            )
            if checkout_result.get("exit_code") != 0:
                return ItemResult(
                    item_id=item_id,
                    error=checkout_result.get("stderr") or "Failed to checkout base commit",
                    metadata={"instance_id": instance_id, "repo": repo},
                )

            agent_output, agent_meta, agent_input_tokens, agent_output_tokens = await self._run_agent_loop(
                sandbox_id,
                prompt=prompt,
                item=item,
            )

            # Include untracked/new files (git diff alone will not).
            await self.sandy.execute_command(sandbox_id, "cd /workspace/repo && git add -A")
            patch_result = await self.sandy.execute_command(
                sandbox_id,
                f"cd /workspace/repo && git diff --cached --binary {base_commit} > /workspace/patch.diff && git reset",
            )
            if patch_result.get("exit_code") != 0:
                return ItemResult(
                    item_id=item_id,
                    error=patch_result.get("stderr") or "Failed to generate patch",
                    metadata={"instance_id": instance_id, "repo": repo},
                )

            patch = await self._read_file(sandbox_id, "/workspace/patch.diff")
            item_hash = self.compute_item_hash(instance_id)

            if not patch.strip():
                latency_ms = int((time.time() - start_time) * 1000)
                return ItemResult(
                    item_id=item_id,
                    item_hash=item_hash,
                    prompt=prompt,
                    response=agent_output,
                    is_correct=False,
                    score=0.0,
                    latency_ms=latency_ms,
                    input_tokens=agent_input_tokens,
                    output_tokens=agent_output_tokens,
                    judge_output={
                        "agent": agent_name,
                        "agent_meta": agent_meta,
                        "failure_reason": "Agent did not produce a patch",
                    },
                    metadata={
                        "instance_id": instance_id,
                        "repo": repo,
                        "agent": agent_name,
                        "agent_meta": agent_meta,
                    },
                )

            entryscript = self._create_entryscript(item)
            await self.sandy.write_file(sandbox_id, "patch.diff", patch)
            await self.sandy.write_file(
                sandbox_id,
                "run_script.sh",
                self._download_run_script(item["instance_id"], "run_script.sh"),
            )
            await self.sandy.write_file(
                sandbox_id,
                "parser.py",
                self._download_run_script(item["instance_id"], "parser.py"),
            )
            await self.sandy.write_file(sandbox_id, "entryscript.sh", entryscript)

            dockerhub_username = "jefzda"
            dockerhub_tag = (item.get("dockerhub_tag") or "").strip()
            if dockerhub_tag:
                image_uri = f"{dockerhub_username}/sweap-images:{dockerhub_tag}"
            else:
                image_uri = self._get_dockerhub_image_uri(
                    item["instance_id"], item.get("repo", ""), dockerhub_username
                )
            pull_result: Optional[dict[str, Any]] = None
            for attempt in range(1, 4):
                pull_result = await self.sandy.execute_command(
                    sandbox_id,
                    f"docker pull {image_uri}",
                    timeout_ms=900000,
                )
                if pull_result.get("exit_code") == 0:
                    break
                if attempt < 3:
                    await asyncio.sleep(min(10 * attempt, 30))
            if not pull_result or pull_result.get("exit_code") != 0:
                return ItemResult(
                    item_id=item_id,
                    error=(pull_result or {}).get("stderr") or "Failed to pull Docker image",
                    metadata={"instance_id": instance_id},
                )

            host_volume = await self._get_host_volume(sandbox_id)
            if not host_volume:
                return ItemResult(item_id=item_id, error="Sandy host volume path unavailable")

            run_result = await self.sandy.execute_command(
                sandbox_id,
                f"docker run --rm -v {host_volume}:/workspace --entrypoint /bin/bash {image_uri} -c \"bash /workspace/entryscript.sh\"",
                timeout_ms=900000,
            )

            output_raw = await self._read_file(sandbox_id, "/workspace/output.json")
            try:
                output = json.loads(output_raw) if output_raw else None
            except json.JSONDecodeError:
                output = None

            stdout_log = await self._read_file(sandbox_id, "/workspace/stdout.log")
            stderr_log = await self._read_file(sandbox_id, "/workspace/stderr.log")

            docker_stdout = run_result.get("stdout") or ""
            docker_stderr = run_result.get("stderr") or ""
            docker_combined_lower = (docker_stdout + "\n" + docker_stderr).lower()

            passed_tests = set()
            if output and isinstance(output.get("tests"), list):
                passed_tests = {
                    test.get("name")
                    for test in output.get("tests", [])
                    if test.get("status") == "PASSED"
                }
            try:
                f2p = set(ast.literal_eval(item.get("fail_to_pass", "[]")))
            except Exception:
                f2p = set()
            try:
                p2p = set(ast.literal_eval(item.get("pass_to_pass", "[]")))
            except Exception:
                p2p = set()
            is_correct = (f2p | p2p) <= passed_tests

            error = None
            failure_reason = None
            if not output:
                # If the harness doesn't produce parseable output, we need to decide whether this is
                # infra (docker/sandbox) or a model failure (e.g., patch did not apply cleanly).
                if any(
                    needle in docker_combined_lower
                    for needle in (
                        "patch does not apply",
                        "error: patch failed",
                        "patch failed",
                    )
                ):
                    # Model produced an invalid/unapplicable patch.
                    failure_reason = "Patch failed to apply in harness"
                    error = None
                    is_correct = False
                else:
                    # Treat missing harness output as infra by default so we don't silently hide
                    # issues like docker daemon failures or missing evaluation scripts.
                    failure_reason = "Harness did not produce parseable output"
                    error = run_result.get("stderr") or run_result.get("error") or "Harness execution failed"
                    if isinstance(error, str) and len(error) > 800:
                        error = error[:800].rstrip() + "…"

            latency_ms = int((time.time() - start_time) * 1000)
            return ItemResult(
                item_id=item_id,
                item_hash=item_hash,
                prompt=prompt,
                response=agent_output,
                expected="[SWE-bench Pro tests passed]",
                is_correct=is_correct,
                score=1.0 if is_correct else 0.0,
                latency_ms=latency_ms,
                input_tokens=agent_input_tokens,
                output_tokens=agent_output_tokens,
                judge_output={
                    "output": output,
                    "stdout": stdout_log,
                    "stderr": stderr_log,
                    "exit_code": run_result.get("exit_code"),
                    "docker_stdout_preview": docker_stdout[:_DOCKER_RUN_OUTPUT_PREVIEW_CHARS],
                    "docker_stderr_preview": docker_stderr[:_DOCKER_RUN_OUTPUT_PREVIEW_CHARS],
                    "agent": agent_name,
                    "agent_meta": agent_meta,
                    "failure_reason": failure_reason,
                },
                error=error,
                metadata={
                    "instance_id": instance_id,
                    "repo": repo,
                    "agent": agent_name,
                    "agent_meta": agent_meta,
                },
            )

        except Exception as e:
            logger.error("SWE-Bench Pro evaluation failed", item_id=item_id, error=str(e))
            return ItemResult(
                item_id=item_id,
                prompt=prompt,
                response=agent_output or "",
                error=str(e),
                metadata={"instance_id": instance_id, "repo": repo},
            )
