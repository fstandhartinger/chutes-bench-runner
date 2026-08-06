"""Reading a Sandy-driven CLI agent's own token usage back out of the sandbox.

Benchmarks that drive a CLI agent (Terminal-Bench, the agentic OOLONG variant)
cannot see the agent's token usage: it is spent inside the sandbox over a
connection bench-runner never opens, and Sandy's `complete` event carries only
`exitCode`, `duration` and `hasFileChanges`. Left alone, every such run reports
0 in / 0 out / $0.00.

That matters more here than it looks. These adapters exist to compare harnesses,
and a harness that wins by spending 6x the tokens has not obviously won -- token
efficiency is half the claim under test.

Both `codex` and `chutescoder` persist a rollout JSONL under
<config_home>/sessions/YYYY/MM/DD/ and emit `token_count` events carrying
TokenUsageInfo. The last one holds the session total.
"""
from __future__ import annotations

import base64
import json
from typing import Any

AGENT_USAGE_PROBE = r"""
import glob, json, os
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


async def collect_agent_usage(sandy: Any, sandbox_id: str) -> dict:
    """Cumulative usage for the last agent session in `sandbox_id`.

    Must run before the sandbox is terminated. Returns `{"error": ...}` rather
    than an estimate when the rollout is missing or carries no usage events --
    a fabricated number here would quietly become a token-efficiency claim.
    """
    try:
        encoded = base64.b64encode(AGENT_USAGE_PROBE.encode()).decode("ascii")
        result = await sandy.execute_command(
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
