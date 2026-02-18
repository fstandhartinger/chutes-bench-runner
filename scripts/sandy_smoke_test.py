#!/usr/bin/env python3
"""
Sandy smoke test for chutes-bench-runner.

Verifies the core Sandy control-plane flows the benchmark worker relies on:
- authenticate
- create sandbox
- exec a command
- write a file + read it back
- terminate sandbox and confirm cleanup

Usage:
  SANDY_BASE_URL=... SANDY_API_KEY=... ./scripts/sandy_smoke_test.py

Notes:
- This script intentionally avoids printing secrets.
- It sets minimal required backend settings env vars when missing to satisfy pydantic Settings.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path


def _ensure_env(key: str, value: str) -> None:
    if not os.getenv(key):
        os.environ[key] = value


async def _wait_for_sandbox_gone(sandy, sandbox_id: str, timeout_seconds: int = 20) -> bool:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        exists = await sandy.sandbox_exists(sandbox_id)
        if exists is False:
            return True
        await asyncio.sleep(1)
    return False


async def main() -> int:
    sandy_base_url = (os.getenv("SANDY_BASE_URL") or "").strip()
    sandy_api_key = (os.getenv("SANDY_API_KEY") or "").strip()
    if not sandy_base_url or not sandy_api_key:
        print("ERROR: SANDY_BASE_URL and SANDY_API_KEY must be set.")
        return 2

    # SandyService reads backend Settings, which require these fields even if unused here.
    _ensure_env("DATABASE_URL", "postgresql://test:test@localhost/test")
    _ensure_env("CHUTES_API_KEY", os.getenv("CHUTES_API_KEY") or "test-key")
    _ensure_env("SKIP_MODEL_SYNC", "true")

    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root / "backend"))

    from app.services.sandy_service import SandyService  # noqa: E402

    sandy = SandyService()

    resources = await sandy.get_resources()
    if resources is None:
        print(f"ERROR: Sandy resources request failed: {sandy.last_error}")
        return 1
    print("OK: authenticated + fetched /api/resources")

    enable_docker = (os.getenv("SANDY_SMOKE_ENABLE_DOCKER_SOCKET") or "").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    sandbox_id = await sandy.create_sandbox(enable_docker_socket=enable_docker)
    if not sandbox_id:
        print(f"ERROR: sandbox create failed: {sandy.last_error}")
        return 1
    print(f"OK: created sandbox {sandbox_id}")

    try:
        exec_result = await sandy.execute_command(sandbox_id, "echo sandy-smoke-ok")
        if not exec_result.get("success"):
            print(f"ERROR: exec failed: {exec_result}")
            return 1
        print("OK: exec")

        if not await sandy.write_file(sandbox_id, "sandy-smoke.txt", "sandy smoke test\n"):
            print(f"ERROR: write_file failed: {sandy.last_error}")
            return 1
        read_result = await sandy.execute_command(sandbox_id, "cat sandy-smoke.txt")
        if not read_result.get("success"):
            print(f"ERROR: readback exec failed: {read_result}")
            return 1
        if "sandy smoke test" not in (read_result.get("stdout") or ""):
            print("ERROR: readback content mismatch")
            return 1
        print("OK: write_file + readback")
    finally:
        terminated = await sandy.terminate_sandbox(sandbox_id)
        if terminated:
            print("OK: terminate requested")
        else:
            print(f"ERROR: terminate failed: {sandy.last_error}")
            return 1

    if not await _wait_for_sandbox_gone(sandy, sandbox_id):
        print("ERROR: sandbox still exists after termination window")
        return 1
    print("OK: sandbox cleanup confirmed (404 on GET /api/sandboxes/{id})")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

