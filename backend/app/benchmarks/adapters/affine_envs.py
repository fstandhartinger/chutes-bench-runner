"""Affine environment benchmark adapters."""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import random
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Optional
from uuid import uuid4

from datasets import load_dataset_builder

from app.benchmarks.base import BenchmarkAdapter, ItemResult
from app.benchmarks.registry import register_adapter
from app.core.config import get_settings
from app.services.sandy_service import SandyService


@dataclass(frozen=True)
class AffineEnvSpec:
    name: str
    display_name: str
    description: str
    env_name: str
    image: str
    total_items: int
    eval_params: dict[str, Any] = field(default_factory=dict)
    env_vars: dict[str, str] = field(default_factory=dict)
    mem_limit: str = "10g"
    cpu_limit: Optional[str] = None
    item_timeout_seconds: int = 1200
    proxy_timeout: Optional[int] = None
    task_space_size: Optional[int] = None
    dataset_name: Optional[str] = None
    dataset_subset: Optional[str] = None
    dataset_split: str = "train"
    volumes: Optional[dict[str, dict[str, str]]] = None
    max_replicas: int = 4


AFFINE_ENV_SPECS: dict[str, AffineEnvSpec] = {
    "affine_print": AffineEnvSpec(
        name="affine_print",
        display_name="Affine PRINT",
        description="Affine environment: PRINT (evaluation environment for AI model testing).",
        env_name="print",
        image="affinefoundation/cde:print",
        total_items=11000,
        env_vars={"UVICORN_WORKERS": "15"},
        eval_params={"temperature": 0.0, "timeout": 600},
        mem_limit="10g",
        item_timeout_seconds=900,
        proxy_timeout=600,
    ),
    "affine_lgc_v2": AffineEnvSpec(
        name="affine_lgc_v2",
        display_name="Affine LGC-V2",
        description="Affine environment: LGC-V2 logic evaluation.",
        env_name="lgc-v2",
        image="affinefoundation/lgc:pi-v2",
        total_items=10600,
        env_vars={"UVICORN_WORKERS": "15"},
        eval_params={"temperature": 0.0, "timeout": 1200},
        mem_limit="20g",
        item_timeout_seconds=1500,
        task_space_size=100_000_000 * 4,
        proxy_timeout=1300,
    ),
    "affine_game": AffineEnvSpec(
        name="affine_game",
        display_name="Affine GAME",
        description="Affine environment: GAME (OpenSpiel evaluation).",
        env_name="game",
        image="affinefoundation/game:openspiel",
        total_items=7300,
        env_vars={"UVICORN_WORKERS": "50"},
        eval_params={"temperature": 0.0, "timeout": 7200},
        mem_limit="8g",
        cpu_limit="2000m",
        item_timeout_seconds=7500,
        task_space_size=10**11,
        proxy_timeout=7400,
    ),
    "affine_ded": AffineEnvSpec(
        name="affine_ded",
        display_name="Affine DED",
        description="Affine environment: DED (program deduction).",
        env_name="affine:ded-v2",
        image="affinefoundation/affine-env:v4",
        total_items=23300,
        env_vars={"UVICORN_WORKERS": "10"},
        eval_params={"task_type": "ded", "temperature": 0.0, "timeout": 600},
        mem_limit="10g",
        item_timeout_seconds=900,
        dataset_name="AffineFoundation/rl-python",
        proxy_timeout=600,
    ),
    "affine_cde": AffineEnvSpec(
        name="affine_cde",
        display_name="Affine CDE",
        description="Affine environment: CDE code evaluation.",
        env_name="cde",
        image="affinefoundation/cde:pi",
        total_items=8600,
        env_vars={"UVICORN_WORKERS": "4"},
        eval_params={"temperature": 0.0, "timeout": 600},
        mem_limit="25g",
        item_timeout_seconds=900,
        dataset_name="PrimeIntellect/INTELLECT-3-RL",
        dataset_subset="code",
        proxy_timeout=600,
    ),
    "affine_lgc": AffineEnvSpec(
        name="affine_lgc",
        display_name="Affine LGC",
        description="Affine environment: LGC logic evaluation.",
        env_name="lgc",
        image="affinefoundation/lgc:pi",
        total_items=1_100_000,
        env_vars={"UVICORN_WORKERS": "15"},
        eval_params={"temperature": 0.0, "timeout": 1200},
        mem_limit="20g",
        item_timeout_seconds=1500,
        dataset_name="AffineFoundation/affine-lgc-xlarge",
        proxy_timeout=1300,
    ),
    "affine_abd": AffineEnvSpec(
        name="affine_abd",
        display_name="Affine ABD",
        description="Affine environment: ABD (program abduction).",
        env_name="affine:abd-v2",
        image="affinefoundation/affine-env:v4",
        total_items=23300,
        env_vars={"UVICORN_WORKERS": "10"},
        eval_params={"task_type": "abd", "temperature": 0.0, "timeout": 600},
        mem_limit="10g",
        item_timeout_seconds=900,
        dataset_name="AffineFoundation/rl-python",
        proxy_timeout=600,
    ),
    "affine_swe_pro": AffineEnvSpec(
        name="affine_swe_pro",
        display_name="Affine SWE-PRO",
        description="Affine environment: SWE-Pro (software engineering).",
        env_name="swe-pro",
        image="affinefoundation/swebench:pro",
        total_items=731,
        env_vars={"UVICORN_WORKERS": "10"},
        eval_params={"temperature": 0.0, "timeout": 1800, "max_iterations": 200},
        mem_limit="10g",
        item_timeout_seconds=2100,
        max_replicas=1,
        volumes={
            "/var/run/docker.sock": {"bind": "/var/run/docker.sock", "mode": "rw"},
        },
        proxy_timeout=2000,
    ),
}


def _extract_json_from_output(output: str) -> dict[str, Any]:
    stdout = output.strip()
    if not stdout:
        raise ValueError("Empty response from Affine env")
    try:
        return json.loads(stdout)
    except json.JSONDecodeError:
        lines = [line.strip() for line in stdout.splitlines() if line.strip()]
        for line in reversed(lines):
            if not (line.startswith("{") and line.endswith("}")):
                continue
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
        last_brace = stdout.rfind("{")
        if last_brace != -1:
            candidate = stdout[last_brace:]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass
    raise ValueError(f"Invalid JSON response from Affine env: {stdout[:500]}")


AFFINE_RUNNER_SCRIPT = """#!/usr/bin/env python3
import argparse
import asyncio
import base64
import json
import os
import sys

import affinetes as af_env
from affinetes.backends import local as local_backend

_STARTUP_TIMEOUT_SECONDS = int(os.environ.get("AFFINETES_STARTUP_TIMEOUT", "300"))
os.environ.setdefault("AFFINETES_LOG_LEVEL", "ERROR")


_orig_wait_for_http_ready = local_backend.LocalBackend._wait_for_http_ready


async def _patched_wait_for_http_ready(self, timeout: int = 60) -> bool:
    return await _orig_wait_for_http_ready(self, timeout=max(timeout, _STARTUP_TIMEOUT_SECONDS))


def _force_runtime_environment(self) -> str:
    # Force IP-based access to avoid Docker DNS resolution issues inside sandboxes.
    return "dind"


local_backend.LocalBackend._wait_for_http_ready = _patched_wait_for_http_ready
local_backend.LocalBackend._detect_runtime_environment = _force_runtime_environment


def _decode_payload(value: str) -> dict:
    raw = base64.b64decode(value.encode("utf-8")).decode("utf-8")
    return json.loads(raw)


async def _handle_init(payload: dict) -> None:
    env = af_env.load_env(
        image=payload["image"],
        mode="docker",
        replicas=payload.get("replicas", 1),
        load_balance=payload.get("load_balance", "round_robin"),
        container_name=payload.get("container_prefix"),
        env_vars=payload.get("env_vars") or {},
        mem_limit=payload.get("mem_limit"),
        cpu_limit=payload.get("cpu_limit"),
        cleanup=False,
        force_recreate=payload.get("force_recreate", False),
        pull=True,
        volumes=payload.get("volumes"),
    )
    stats = None
    try:
        stats = env.get_stats()
    except Exception:
        stats = None
    print(json.dumps({"status": "ok", "stats": stats}))


async def _handle_eval(payload: dict) -> None:
    env = af_env.load_env(
        image=payload["image"],
        mode="docker",
        replicas=payload.get("replicas", 1),
        load_balance=payload.get("load_balance", "round_robin"),
        container_name=payload.get("container_prefix"),
        env_vars=payload.get("env_vars") or {},
        mem_limit=payload.get("mem_limit"),
        cpu_limit=payload.get("cpu_limit"),
        cleanup=False,
        force_recreate=False,
        pull=False,
        volumes=payload.get("volumes"),
    )
    eval_params = payload.get("eval_params") or {}
    proxy_timeout = payload.get("proxy_timeout")
    eval_params.update(
        {
            "model": payload["model"],
            "base_url": payload["base_url"],
            "task_id": payload["task_id"],
        }
    )
    api_key = payload.get("api_key")
    if api_key:
        eval_params["api_key"] = api_key
    seed = payload.get("seed")
    if seed is not None:
        eval_params["seed"] = seed
    if proxy_timeout is not None:
        result = await env.evaluate(_timeout=proxy_timeout, **eval_params)
    else:
        result = await env.evaluate(**eval_params)
    print(json.dumps(result))


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["init", "eval"])
    parser.add_argument("--payload", required=True)
    args = parser.parse_args()

    payload = _decode_payload(args.payload)
    if args.mode == "init":
        await _handle_init(payload)
    else:
        await _handle_eval(payload)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as exc:
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}))
        sys.exit(1)
"""


class AffineEnvAdapter(BenchmarkAdapter):
    """Adapter for Affine Affinetes environments executed inside Sandy sandboxes."""

    spec: AffineEnvSpec

    def __init__(self, *args: Any, spec: AffineEnvSpec, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.spec = spec
        self.sandy = SandyService()
        self._sandbox_id: Optional[str] = None
        self._sandbox_error: Optional[str] = None
        self._container_prefix = f"{spec.env_name.replace(':', '-')}-{uuid4().hex[:8]}"
        self._prepared = False
        self._total_items: Optional[int] = None
        self._task_space_size: Optional[int] = spec.task_space_size
        self._setup_lock = asyncio.Lock()
        self._replicas = 1

    def get_name(self) -> str:
        return self.spec.name

    def get_display_name(self) -> str:
        return self.spec.display_name

    def supports_parallel_items(self) -> bool:
        return True

    def get_item_concurrency(self) -> Optional[int]:
        return self._replicas

    def get_item_timeout_seconds(self) -> Optional[int]:
        return self.spec.item_timeout_seconds

    async def get_total_items(self) -> int:
        if self._total_items is not None:
            return self._total_items
        total = self.spec.total_items
        if self.spec.dataset_name:
            dataset_total = await self._get_dataset_size(
                self.spec.dataset_name,
                self.spec.dataset_subset,
                self.spec.dataset_split,
            )
            if dataset_total:
                total = min(total, dataset_total) if total else dataset_total
                if self._task_space_size is None:
                    self._task_space_size = dataset_total
        if self._task_space_size is None:
            self._task_space_size = total
        self._total_items = total
        return total

    async def preload(self) -> None:
        async with self._setup_lock:
            if self._prepared:
                return
            settings = get_settings()
            self._replicas = max(1, min(self.spec.max_replicas, settings.worker_item_concurrency))
            sandbox_id = await self._ensure_sandbox()
            await self._install_affinetes(sandbox_id)
            await self._write_runner(sandbox_id)
            await self._init_environment(sandbox_id)
            self._prepared = True

    async def cleanup(self) -> None:
        if not self._sandbox_id:
            return None
        await self._cleanup_containers(self._sandbox_id)
        await self.sandy.terminate_sandbox(self._sandbox_id)
        self._sandbox_id = None
        self._prepared = False
        return None

    async def enumerate_items(self) -> AsyncIterator[str]:
        total = await self.get_total_items()
        for idx in range(total):
            yield str(idx)

    async def get_items_for_evaluation(self, subset_pct: int, seed: str) -> tuple[int, list[str]]:
        total = await self.get_total_items()
        if total <= 0:
            return 0, []
        if subset_pct >= 100:
            subset_count = total
        else:
            subset_count = max(1, int(total * subset_pct / 100))
        seed_int = int(hashlib.sha256(seed.encode()).hexdigest()[:16], 16)
        rng = random.Random(seed_int)
        task_space = self._task_space_size or total
        if subset_count >= total and task_space == total:
            items = list(range(total))
        else:
            items = rng.sample(range(task_space), subset_count)
        return total, [str(item) for item in items]

    async def evaluate_item(self, item_id: str) -> ItemResult:
        await self.preload()
        if not self._sandbox_id:
            sandbox_error = self._sandbox_error or "Sandbox initialization failed"
            return ItemResult(item_id=item_id, error=sandbox_error)

        task_id = int(item_id)
        payload = self._build_payload(task_id)
        encoded = base64.b64encode(json.dumps(payload).encode("utf-8")).decode("utf-8")
        command = f"python3 /workspace/affine_runner.py eval --payload '{encoded}'"
        result = await self.sandy.execute_command(
            self._sandbox_id,
            command,
            timeout_ms=self.spec.item_timeout_seconds * 1000,
        )
        if not result.get("success") or result.get("exit_code") not in (0, None):
            error = result.get("stderr") or result.get("error")
            if not error:
                stdout = (result.get("stdout") or "").strip()
                if stdout:
                    try:
                        data = json.loads(stdout)
                        error = data.get("error") or data.get("error_type") or stdout
                    except json.JSONDecodeError:
                        error = stdout
            return ItemResult(
                item_id=item_id,
                error=error or "Affine eval failed",
            )

        try:
            data = _extract_json_from_output(result.get("stdout", ""))
        except ValueError as exc:
            return ItemResult(
                item_id=item_id,
                error=str(exc),
            )

        error_message = data.get("error") or data.get("error_type")
        score = data.get("score")
        success = data.get("success")
        latency_ms = None
        if data.get("time_taken") is not None:
            latency_ms = int(float(data.get("time_taken")) * 1000)

        conversation = (data.get("extra") or {}).get("conversation") or []
        prompt = None
        response = None
        if len(conversation) >= 2:
            prompt = conversation[0].get("content")
            response = conversation[1].get("content")

        usage = (data.get("extra") or {}).get("usage") or {}
        input_tokens = usage.get("prompt_tokens") or usage.get("input_tokens")
        output_tokens = usage.get("completion_tokens") or usage.get("output_tokens")

        return ItemResult(
            item_id=item_id,
            item_hash=self.compute_item_hash({"env": self.spec.env_name, "task_id": task_id}),
            prompt=prompt,
            response=response,
            expected=None,
            is_correct=bool(success) if success is not None else None,
            score=float(score) if score is not None else None,
            judge_output=data.get("extra") if data.get("extra") else None,
            latency_ms=latency_ms,
            input_tokens=int(input_tokens) if input_tokens is not None else None,
            output_tokens=int(output_tokens) if output_tokens is not None else None,
            error=error_message,
            metadata={
                "env": self.spec.env_name,
                "task_id": task_id,
                "seed": payload.get("seed"),
                "task_space": self._task_space_size,
            },
        )

    async def _ensure_sandbox(self) -> Optional[str]:
        if self._sandbox_id:
            return self._sandbox_id
        sandbox_id = await self.sandy.create_sandbox(enable_docker_socket=True)
        if not sandbox_id:
            self._sandbox_error = self.sandy.last_error or "Could not create sandbox"
        self._sandbox_id = sandbox_id
        return sandbox_id

    async def _install_affinetes(self, sandbox_id: str) -> None:
        commands = [
            "python3 -m pip --version",
            "git --version",
            "docker --version",
        ]
        for command in commands:
            check = await self.sandy.execute_command(sandbox_id, command)
            if check.get("exit_code") == 0:
                continue
            if command.startswith("python3"):
                await self.sandy.execute_command(
                    sandbox_id,
                    "apt-get update && apt-get install -y python3 python3-pip",
                    timeout_ms=300000,
                )
            elif command.startswith("git"):
                await self.sandy.execute_command(
                    sandbox_id,
                    "apt-get update && apt-get install -y git",
                    timeout_ms=300000,
                )
            elif command.startswith("docker"):
                await self.sandy.execute_command(
                    sandbox_id,
                    "apt-get update && apt-get install -y docker.io",
                    timeout_ms=300000,
                )
        pip_commands = [
            ("python3 -m pip install --break-system-packages --upgrade pip", 300000),
            (
                "python3 -m pip install --break-system-packages "
                "'affinetes @ git+https://github.com/AffineFoundation/affinetes.git'",
                900000,
            ),
        ]
        for command, timeout_ms in pip_commands:
            result = await self.sandy.execute_command(
                sandbox_id,
                command,
                timeout_ms=timeout_ms,
            )
            if result.get("exit_code") not in (0, None):
                error = (result.get("stderr") or result.get("stdout") or result.get("error") or "").strip()
                raise RuntimeError(f"Failed to install affinetes: {error[:500] or 'unknown error'}")

    async def _write_runner(self, sandbox_id: str) -> None:
        await self.sandy.write_file(sandbox_id, "/workspace/affine_runner.py", AFFINE_RUNNER_SCRIPT)

    async def _init_environment(self, sandbox_id: str) -> None:
        payload = self._build_payload(task_id=0, include_task=False)
        encoded = base64.b64encode(json.dumps(payload).encode("utf-8")).decode("utf-8")
        command = f"python3 /workspace/affine_runner.py init --payload '{encoded}'"
        result = await self.sandy.execute_command(
            sandbox_id,
            command,
            timeout_ms=900000,
        )
        if not result.get("success") or result.get("exit_code") not in (0, None):
            error = result.get("stderr") or result.get("error")
            if not error:
                stdout = result.get("stdout") or ""
                if stdout.strip():
                    try:
                        data = _extract_json_from_output(stdout)
                        error = data.get("error") or data.get("error_type") or stdout.strip()
                    except ValueError:
                        error = stdout.strip()
            raise RuntimeError(error or "Affine init failed")

    async def _cleanup_containers(self, sandbox_id: str) -> None:
        for idx in range(self._replicas):
            name = f"{self._container_prefix}-{idx}"
            await self.sandy.execute_command(sandbox_id, f"docker rm -f {name}")

    def _build_payload(self, task_id: int, include_task: bool = True) -> dict[str, Any]:
        settings = get_settings()
        api_key = self.client.user_access_token or self.client.api_key
        env_vars = {
            "CHUTES_API_KEY": api_key,
            "HF_TOKEN": os.getenv("HF_TOKEN", ""),
            "HUGGINGFACE_HUB_TOKEN": os.getenv("HF_TOKEN", ""),
        }
        task_type = self.spec.eval_params.get("task_type")
        if task_type:
            env_vars["ENV_NAME"] = task_type
        env_vars.update(self.spec.env_vars)
        payload = {
            "image": self.spec.image,
            "container_prefix": self._container_prefix,
            "replicas": self._replicas,
            "load_balance": "round_robin",
            "env_vars": env_vars,
            "mem_limit": self.spec.mem_limit,
            "cpu_limit": self.spec.cpu_limit,
            "volumes": self.spec.volumes,
            "eval_params": self.spec.eval_params,
            "model": self.model_slug,
            "base_url": settings.chutes_api_base_url,
            "api_key": api_key,
            "proxy_timeout": self.spec.proxy_timeout,
        }
        if include_task:
            payload["task_id"] = task_id
            payload["seed"] = self._generate_seed(task_id)
        return payload

    def _generate_seed(self, task_id: int) -> int:
        seed_string = f"{self.spec.env_name}:{task_id}"
        hash_bytes = hashlib.sha256(seed_string.encode()).digest()[:8]
        return int.from_bytes(hash_bytes, byteorder="big") % (2**32)

    async def _get_dataset_size(
        self,
        dataset_name: str,
        dataset_subset: Optional[str],
        dataset_split: str,
    ) -> Optional[int]:
        loop = asyncio.get_running_loop()
        token = os.getenv("HF_TOKEN") or None

        def _load() -> Optional[int]:
            try:
                builder = load_dataset_builder(dataset_name, dataset_subset, token=token)
                split = builder.info.splits.get(dataset_split)
                if split and split.num_examples is not None:
                    return int(split.num_examples)
            except Exception:
                return None
            return None

        return await loop.run_in_executor(None, _load)


def _register_affine_adapter(spec: AffineEnvSpec) -> None:
    @register_adapter(spec.name)
    class _Adapter(AffineEnvAdapter):
        def __init__(self, *args: Any, **kwargs: Any):
            super().__init__(*args, spec=spec, **kwargs)


for _spec in AFFINE_ENV_SPECS.values():
    _register_affine_adapter(_spec)
