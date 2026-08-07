import base64
import hashlib
import io
import json
import os
import re
import tarfile
import time
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.benchmarks.adapters.terminal_bench import TerminalBench21Adapter
from app.benchmarks.agent_evidence import (
    _reserve_local_file,
    read_token_usage_samples,
    retain_agent_evidence,
)
from app.benchmarks.base import ItemResult
from app.services.run_service import save_item_result


def _bundle_bytes() -> bytes:
    first = {
        "timestamp": "2026-08-07T10:00:02Z",
        "payload": {
            "type": "token_count",
            "info": {
                "last_token_usage": {"input_tokens": 20, "output_tokens": 2},
                "total_token_usage": {"input_tokens": 30, "output_tokens": 3},
                "model_context_window": 100_000,
            },
        },
    }
    second = {
        "timestamp": "2026-08-07T10:00:01Z",
        "payload": {
            "type": "token_count",
            "info": {
                "last_token_usage": {"input_tokens": 10, "output_tokens": 1},
                "total_token_usage": {"input_tokens": 10, "output_tokens": 1},
                "model_context_window": 100_000,
            },
        },
    }
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as bundle:
        for name, events in (
            (
                "rollouts/chutescoder/sessions/2026/08/07/rollout-parent.jsonl",
                [first],
            ),
            ("rollouts/chutescoder/sessions/2026/08/07/rollout-child.jsonl", [second]),
        ):
            payload = b"".join(
                (json.dumps(event) + "\n").encode() for event in events
            )
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            bundle.addfile(info, io.BytesIO(payload))
        output = b"agent stdout\nagent stderr\n"
        info = tarfile.TarInfo("agent/combined-stdout-stderr.log")
        info.size = len(output)
        bundle.addfile(info, io.BytesIO(output))
    return buffer.getvalue()


class _FakeSandy:
    def __init__(self, bundle: bytes, *, expected_sha256: str | None = None):
        self.bundle = bundle
        self.expected_sha256 = expected_sha256 or hashlib.sha256(bundle).hexdigest()
        self.chunk_calls = 0
        self.truncate_offset: int | None = None

    async def execute_command(self, sandbox_id, command, timeout_ms=None):
        if "_prepare_agent_evidence.py" in command:
            return {
                "exit_code": 0,
                "stdout": json.dumps(
                    {
                        "path": "/tmp/chutes-bench-agent-evidence.tar.gz",
                        "size_bytes": len(self.bundle),
                        "sha256": self.expected_sha256,
                        "sources": [
                            "rollouts/chutescoder/sessions",
                            "agent/combined-stdout-stderr.log",
                        ],
                    }
                ),
            }
        match = re.search(r"skip=(\d+) count=(\d+)", command)
        assert match is not None
        offset, count = (int(value) for value in match.groups())
        chunk = self.bundle[offset : offset + count]
        if self.truncate_offset == offset:
            chunk = chunk[:-1]
        self.chunk_calls += 1
        return {
            "exit_code": 0,
            "stdout": base64.b64encode(chunk).decode(),
        }


def _settings(root: Path, *, chunk_bytes: int = 97) -> SimpleNamespace:
    return SimpleNamespace(
        agent_evidence_dir=str(root),
        agent_evidence_max_item_bytes=64 * 1024 * 1024,
        agent_evidence_max_total_bytes=5 * 1024 * 1024 * 1024,
        agent_evidence_max_age_days=14,
        agent_evidence_min_free_bytes=0,
        agent_evidence_transfer_timeout_seconds=10,
        agent_evidence_chunk_bytes=chunk_bytes,
        agent_evidence_chunk_concurrency=3,
    )


@pytest.mark.asyncio
async def test_retention_transfers_chunks_and_verifies_in_sandbox_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _bundle_bytes()
    sandy = _FakeSandy(bundle)
    monkeypatch.setattr(
        "app.benchmarks.agent_evidence.get_settings", lambda: _settings(tmp_path)
    )

    result = await retain_agent_evidence(
        sandy,
        "sandbox-1",
        run_id="run-1",
        benchmark_name="terminal_bench_2_1",
        item_id="item/with/slashes",
    )

    assert result["status"] == "retained"
    assert result["sha256"] == hashlib.sha256(bundle).hexdigest()
    assert Path(result["path"]).read_bytes() == bundle
    assert sandy.chunk_calls > 1
    series = result["token_usage_samples"]
    assert series["complete"] is True
    assert series["events_seen"] == 2
    assert [sample["last_token_usage"]["input_tokens"] for sample in series["samples"]] == [
        10,
        20,
    ]
    assert [sample["sequence"] for sample in series["samples"]] == [1, 2]


@pytest.mark.asyncio
async def test_hash_mismatch_is_quarantined_and_marked_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sandy = _FakeSandy(_bundle_bytes(), expected_sha256="0" * 64)
    monkeypatch.setattr(
        "app.benchmarks.agent_evidence.get_settings", lambda: _settings(tmp_path)
    )

    result = await retain_agent_evidence(
        sandy,
        "sandbox-2",
        run_id="run-2",
        benchmark_name="terminal_bench_2_1",
        item_id="2",
    )

    assert result["status"] == "failed"
    assert result["path"] is None
    assert "SHA-256 mismatch" in result["error"]
    assert not list(tmp_path.rglob("*.tar.gz"))
    assert len(list(tmp_path.rglob("*.CORRUPT"))) == 1


@pytest.mark.asyncio
async def test_truncated_chunk_is_detected_before_hash_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sandy = _FakeSandy(_bundle_bytes())
    sandy.truncate_offset = 97
    monkeypatch.setattr(
        "app.benchmarks.agent_evidence.get_settings", lambda: _settings(tmp_path)
    )

    result = await retain_agent_evidence(
        sandy,
        "sandbox-3",
        run_id="run-3",
        benchmark_name="terminal_bench_2_1",
        item_id="3",
    )

    assert result["status"] == "failed"
    assert "truncated evidence chunk" in result["error"]
    assert not list(tmp_path.rglob("*.tar.gz"))
    assert not list(tmp_path.rglob("*.partial"))


def test_age_pruning_happens_before_disk_reservation(tmp_path: Path) -> None:
    old = tmp_path / "old" / "bench" / "item" / "old.tar.gz"
    old.parent.mkdir(parents=True)
    old.write_bytes(b"old evidence")
    old_time = time.time() - 15 * 86400
    os.utime(old, (old_time, old_time))
    final = tmp_path / "new" / "bench" / "item" / "new.tar.gz"

    partial, handle = _reserve_local_file(
        final,
        128,
        max_total_bytes=1024,
        max_age_days=14,
        min_free_bytes=0,
    )
    handle.close()

    assert not old.exists()
    assert partial.stat().st_size == 128
    assert '"reason": "age_limit"' in (tmp_path / "pruned.jsonl").read_text()


def test_token_series_reports_malformed_rollout_lines(tmp_path: Path) -> None:
    path = tmp_path / "evidence.tar.gz"
    with tarfile.open(path, mode="w:gz") as bundle:
        payload = b"not-json\n"
        info = tarfile.TarInfo(
            "rollouts/codex/sessions/2026/08/07/rollout-broken.jsonl"
        )
        info.size = len(payload)
        bundle.addfile(info, io.BytesIO(payload))

    result = read_token_usage_samples(path)

    assert result["complete"] is False
    assert result["malformed_lines"] == 1


def test_retention_failure_metadata_does_not_change_score() -> None:
    adapter = TerminalBench21Adapter.__new__(TerminalBench21Adapter)
    adapter._item_observability = {
        "50": {
            "evidence": {
                "status": "failed",
                "path": None,
                "sha256": None,
                "size_bytes": None,
                "error": "transfer failed",
                "token_usage_samples": None,
            }
        }
    }
    scored = ItemResult(item_id="50", is_correct=True, score=1.0)

    result = adapter.attach_item_observability(scored)

    assert result.score == 1.0
    assert result.is_correct is True
    assert result.error is None
    assert result.agent_evidence_status == "failed"
    assert result.agent_evidence_error == "transfer failed"


@pytest.mark.asyncio
async def test_evidence_provenance_and_token_series_are_stored_on_item_row(
    test_session,
) -> None:
    series = {
        "schema_version": 1,
        "complete": True,
        "events_seen": 1,
        "samples": [{"sequence": 1, "last_token_usage": {"input_tokens": 10}}],
    }

    row = await save_item_result(
        test_session,
        str(uuid4()),
        item_id="50",
        score=0.0,
        agent_evidence_status="retained",
        agent_evidence_path="/var/lib/sandy/cache/chutes-bench-evidence/evidence.tar.gz",
        agent_evidence_sha256="a" * 64,
        agent_evidence_size_bytes=1234,
        token_usage_samples=series,
    )

    assert row.agent_evidence_status == "retained"
    assert row.agent_evidence_path.endswith("evidence.tar.gz")
    assert row.agent_evidence_sha256 == "a" * 64
    assert row.agent_evidence_size_bytes == 1234
    assert row.token_usage_samples == series
