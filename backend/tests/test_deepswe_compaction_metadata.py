"""Regression coverage for parsed DeepSWE compaction metadata persistence."""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import pytest

from app.benchmarks.adapters.deepswe import DeepSWEAdapter
from app.benchmarks.agent_evidence import read_rollout_metrics
from app.benchmarks.base import ItemResult
from app.services.run_service import save_item_result


@pytest.mark.asyncio
async def test_observed_rollout_compactions_land_in_top_level_item_metadata(
    tmp_path: Path,
    test_session,
) -> None:
    evidence_path = tmp_path / "observed-compaction.tar.gz"
    events = [
        {"type": "event_msg", "payload": {"type": "context_compacted"}},
        {"type": "event_msg", "payload": {"type": "context_compacted"}},
    ]
    with tarfile.open(evidence_path, mode="w:gz") as bundle:
        payload = b"".join((json.dumps(event) + "\n").encode() for event in events)
        member = tarfile.TarInfo(
            "rollouts/chutescoder/sessions/2026/08/07/rollout-compaction.jsonl"
        )
        member.size = len(payload)
        bundle.addfile(member, io.BytesIO(payload))

    parsed_metrics = read_rollout_metrics(evidence_path)
    assert parsed_metrics["compaction_events"] > 0

    adapter = DeepSWEAdapter.__new__(DeepSWEAdapter)
    adapter._item_observability = {
        "task": {
            "agent": "chutescoder",
            "context_limit_tokens": 48_000,
            "configured_context_window": 48_000,
            "evidence": {
                "status": "retained",
                "path": str(evidence_path),
                "sha256": "a" * 64,
                "size_bytes": evidence_path.stat().st_size,
                "error": None,
                "token_usage_samples": None,
                "rollout_metrics": parsed_metrics,
            },
        }
    }

    result = adapter.attach_item_observability(ItemResult(item_id="task", score=0.0))
    row = await save_item_result(
        test_session,
        "00000000-0000-0000-0000-000000000000",
        item_id=result.item_id,
        score=result.score,
        item_metadata=result.metadata,
    )

    assert row is not None
    assert row.item_metadata["compaction_events"] == parsed_metrics["compaction_events"]
    assert row.item_metadata["compaction_events_by_type"] == parsed_metrics[
        "compaction_events_by_type"
    ]
    assert row.item_metadata["compaction_experiment"]["compaction_events"] == parsed_metrics[
        "compaction_events"
    ]
