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
TokenUsageInfo. The last one holds the session total. Upstream Prime Agent
persists exact usage on every assistant message in its session JSONL and adds
`child_usage_attributed` entries when recursive children spend tokens.
"""
from __future__ import annotations

import base64
import json
from typing import Any

AGENT_USAGE_PROBE = r"""
import glob, json, os
ROOT = os.environ.get("CHUTES_BENCH_AGENT_USAGE_ROOT", "")
def rooted(path):
    return os.path.join(ROOT, path.lstrip("/")) if ROOT else path
PATTERNS = [
    rooted("/root/.chutescoder/sessions/*/*/*/rollout-*.jsonl"),
    rooted("/root/.codex/sessions/*/*/*/rollout-*.jsonl"),
]
files = []
for pattern in PATTERNS:
    files.extend(("codex", path) for path in glob.glob(pattern))

prime_files = []
for path in glob.glob(rooted("/workspace/.chutes/prime-agent-sessions/**/*.jsonl"), recursive=True):
    try:
        with open(path, errors="replace") as session:
            header = json.loads(session.readline())
    except Exception:
        continue
    if header.get("type") != "session":
        continue
    if header.get("parentSession") or (header.get("rlmDepth") not in (None, 0)):
        continue
    prime_files.append(("prime-agent", path))
files.extend(prime_files)
if not files:
    print(json.dumps({"error": "no agent usage file found"}))
    raise SystemExit
kind, path = max(files, key=lambda candidate: os.path.getmtime(candidate[1]))

if kind == "prime-agent":
    assistant_order = []
    assistant_usage = {}
    attributions = 0
    for line in open(path, errors="replace"):
        try:
            entry = json.loads(line)
        except Exception:
            continue
        if entry.get("type") == "message":
            message = entry.get("message") or {}
            usage = message.get("usage") or {}
            if message.get("role") == "assistant" and isinstance(usage, dict):
                entry_id = entry.get("id")
                if entry_id:
                    assistant_order.append(entry_id)
                    assistant_usage[entry_id] = usage
        elif entry.get("type") == "child_usage_attributed":
            target_id = entry.get("targetId")
            aggregate = entry.get("aggregateUsage")
            if target_id in assistant_usage and isinstance(aggregate, dict):
                assistant_usage[target_id] = aggregate
                attributions += 1

    if not assistant_order:
        print(json.dumps({"error": "no Prime Agent assistant usage found"}))
        raise SystemExit

    totals = {key: 0 for key in ("input", "output", "cacheRead", "cacheWrite")}
    cost_total = 0.0
    for entry_id in assistant_order:
        usage = assistant_usage[entry_id]
        for key in totals:
            totals[key] += int(usage.get(key) or 0)
        cost = usage.get("cost") or {}
        cost_total += float(cost.get("total") or 0)
    # Codex/OpenAI input_tokens includes cached prompt tokens, with the cached
    # subset also reported separately. Prime's internal `input` excludes cache
    # reads/writes, so normalize it to the same public accounting convention.
    input_tokens = totals["input"] + totals["cacheRead"] + totals["cacheWrite"]
    total_tokens = input_tokens + totals["output"]
    print(json.dumps({
        "usage_source": "prime-agent-session",
        "session": os.path.basename(path),
        "assistant_messages": len(assistant_order),
        "child_usage_attribution_events": attributions,
        "input_tokens": input_tokens,
        "non_cached_input_tokens": totals["input"],
        "cached_input_tokens": totals["cacheRead"],
        "cache_write_input_tokens": totals["cacheWrite"],
        "output_tokens": totals["output"],
        "reasoning_output_tokens": None,
        "total_tokens": total_tokens,
        "cost_total": cost_total,
    }))
    raise SystemExit

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
    "usage_source": "codex-token-count",
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
