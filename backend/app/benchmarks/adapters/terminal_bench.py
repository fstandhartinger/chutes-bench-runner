"""Terminal-Bench Hard benchmark adapter."""
from __future__ import annotations

import base64
import os
import time
from typing import Any, AsyncIterator, Optional

import yaml
from datasets import load_dataset

from app.benchmarks.base import BenchmarkAdapter, ItemResult
from app.benchmarks.registry import register_adapter
from app.core.logging import get_logger
from app.services.sandy_service import SandyService

logger = get_logger(__name__)


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
        return "Terminal-Bench auto-downloads task archives and evaluates them in Sandy."

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
            dataset = load_dataset(
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

    async def _run_terminal_bench(self, sandbox_id: str, task_id: str) -> dict[str, Any]:
        task_dir = "/workspace/task"
        compose_check = await self.sandy.execute_command(
            sandbox_id,
            f"test -f {task_dir}/docker-compose.yaml",
        )
        has_compose = compose_check.get("exit_code") == 0
        image_name = f"tbench_{task_id}".lower()
        container_name = f"{image_name}_container"
        cleanup_cmd = None

        if has_compose:
            compose_cmd = "docker-compose"
            compose_check = await self.sandy.execute_command(sandbox_id, "docker-compose version")
            if compose_check.get("exit_code") != 0:
                compose_cmd = "docker compose"
            logs_dir = f"{task_dir}/logs"
            await self.sandy.execute_command(sandbox_id, f"mkdir -p {logs_dir}")
            env = {
                "T_BENCH_TASK_DOCKER_CLIENT_IMAGE_NAME": "client",
                "T_BENCH_TASK_DOCKER_NAME_PREFIX": f"tbench_{task_id}",
                "T_BENCH_TASK_DOCKER_CLIENT_CONTAINER_NAME": f"tbench_{task_id}_client",
                "T_BENCH_TASK_LOGS_PATH": logs_dir,
                "T_BENCH_CONTAINER_LOGS_PATH": "/var/log/tbench",
                "T_BENCH_TEST_DIR": "/tests",
            }
            up_cmd = f"{compose_cmd} -f {task_dir}/docker-compose.yaml up --build -d"
            up_result = await self.sandy.execute_command(
                sandbox_id,
                up_cmd,
                env=env,
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
            cleanup_cmd = f"{compose_cmd} -f {task_dir}/docker-compose.yaml down"
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

        return {"container_name": container_name, "cleanup_cmd": cleanup_cmd}

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
            "You are solving a terminal task. Provide a bash script that completes the task.\n"
            "The script will be executed inside the task container.\n\n"
            f"Task:\n{instruction}\n\n"
            "Script:\n```bash\n"
        )
        system_prompt = "Output ONLY the bash script within a markdown code block. No explanations."

        try:
            start_time = time.time()
            response_text, metadata = await self.client.get_completion_text(
                self.model_slug,
                prompt,
                system_prompt=system_prompt,
                max_tokens=4096,
                temperature=0.0,
            )
            latency_ms = int((time.time() - start_time) * 1000)

            if not response_text:
                return ItemResult(
                    item_id=item_id,
                    item_hash=self.compute_item_hash(item.get("task_id")),
                    prompt=prompt,
                    response="",
                    error=self.format_empty_response_error(metadata),
                    latency_ms=latency_ms,
                    metadata={"task_id": item.get("task_id"), "system_prompt": system_prompt},
                )

            script = self.extract_python_code(response_text)
            if not script.strip().startswith("#!"):
                script = "#!/usr/bin/env bash\nset -e\n" + script

            archive = item.get("archive")
            if not isinstance(archive, (bytes, bytearray)):
                return ItemResult(item_id=item_id, error="Missing task archive bytes")

            sandbox_id = await self.sandy.create_sandbox()
            if not sandbox_id:
                return ItemResult(item_id=item_id, error="Could not create sandbox")

            try:
                extracted = await self._extract_archive(sandbox_id, archive)
                if not extracted:
                    return ItemResult(item_id=item_id, error="Failed to extract task archive")

                await self.sandy.write_file(sandbox_id, "task/solution.sh", script)
                await self.sandy.execute_command(
                    sandbox_id,
                    "chmod +x task/solution.sh",
                )

                setup_result = await self._run_terminal_bench(sandbox_id, item.get("task_id") or "task")
                container_name = setup_result.get("container_name")
                cleanup_cmd = setup_result.get("cleanup_cmd")
                if not container_name:
                    return ItemResult(
                        item_id=item_id,
                        error=setup_result.get("error") or "Container setup failed",
                        judge_output={"setup": setup_result},
                        metadata={"task_id": item.get("task_id"), "system_prompt": system_prompt},
                    )

                try:
                    agent_timeout = int((item.get("max_agent_timeout_sec") or 180) * 1000)
                    test_timeout = int((item.get("max_test_timeout_sec") or 300) * 1000)

                    # Agent phase: execute solution.sh
                    await self.sandy.execute_command(
                        sandbox_id,
                        f"docker cp task/solution.sh {container_name}:/solution.sh",
                    )
                    agent_result = await self.sandy.execute_command(
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
                        response=response_text.strip(),
                        expected="[Tests passed]",
                        is_correct=is_correct,
                        score=1.0 if is_correct else 0.0,
                        latency_ms=latency_ms,
                        input_tokens=metadata.get("usage", {}).get("prompt_tokens"),
                        output_tokens=metadata.get("usage", {}).get("completion_tokens"),
                        judge_output={
                            "agent_exit_code": agent_result.get("exit_code"),
                            "agent_stderr": agent_result.get("stderr"),
                            "stdout": test_result.get("stdout"),
                            "stderr": test_result.get("stderr"),
                            "exit_code": test_result.get("exit_code"),
                        },
                        error=error,
                        metadata={
                            "task_id": item.get("task_id"),
                            "difficulty": item.get("difficulty"),
                            "system_prompt": system_prompt,
                            "task_yaml": task_yaml,
                        },
                    )
                finally:
                    if cleanup_cmd:
                        await self.sandy.execute_command(sandbox_id, cleanup_cmd)
            finally:
                await self.sandy.terminate_sandbox(sandbox_id)

        except Exception as e:
            logger.error("Terminal-Bench evaluation failed", item_id=item_id, error=str(e))
            return ItemResult(
                item_id=item_id,
                prompt=prompt,
                response=locals().get("response_text", "") or "",
                error=str(e),
                metadata={"task_id": item.get("task_id")},
            )
