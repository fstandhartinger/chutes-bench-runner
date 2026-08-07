from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from app.benchmarks.agent_usage import AGENT_USAGE_PROBE


def test_prime_agent_probe_includes_child_usage_and_normalizes_cache(
    tmp_path: Path,
) -> None:
    session_dir = (
        tmp_path
        / "workspace"
        / ".chutes"
        / "prime-agent-sessions"
        / "sandbox-id"
    )
    session_dir.mkdir(parents=True)
    session_path = session_dir / "root-session.jsonl"
    entries = [
        {
            "type": "session",
            "version": 3,
            "id": "root",
            "timestamp": "2026-08-07T00:00:00Z",
            "cwd": "/workspace",
            "rlmDepth": 0,
        },
        {
            "type": "message",
            "id": "assistant-1",
            "message": {
                "role": "assistant",
                "usage": {
                    "input": 100,
                    "output": 20,
                    "cacheRead": 10,
                    "cacheWrite": 5,
                    "totalTokens": 135,
                    "cost": {"total": 0.1},
                },
            },
        },
        {
            "type": "child_usage_attributed",
            "targetId": "assistant-1",
            "childUsage": {
                "input": 50,
                "output": 6,
                "cacheRead": 2,
                "cacheWrite": 0,
                "totalTokens": 58,
                "cost": {"total": 0.05},
            },
            "aggregateUsage": {
                "input": 150,
                "output": 26,
                "cacheRead": 12,
                "cacheWrite": 5,
                "totalTokens": 135,
                "cost": {"total": 0.15},
            },
        },
    ]
    session_path.write_text(
        "".join(json.dumps(entry) + "\n" for entry in entries),
        encoding="utf-8",
    )
    env = {
        **os.environ,
        "CHUTES_BENCH_AGENT_USAGE_ROOT": str(tmp_path),
    }

    completed = subprocess.run(
        [sys.executable, "-c", AGENT_USAGE_PROBE],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    usage = json.loads(completed.stdout)

    assert usage["usage_source"] == "prime-agent-session"
    assert usage["input_tokens"] == 167
    assert usage["non_cached_input_tokens"] == 150
    assert usage["cached_input_tokens"] == 12
    assert usage["cache_write_input_tokens"] == 5
    assert usage["output_tokens"] == 26
    assert usage["total_tokens"] == 193
    assert usage["child_usage_attribution_events"] == 1
