"""OOLONG driven by a CLI agent in a sandbox, instead of one API call.

Why this exists
---------------
`oolong` sends the whole corpus in a single completion request. That measures
the *model*, and nothing about the harness wrapped around it: every CLI agent
would produce an identical number, because none of them is involved.

Terminal-Bench Hard was the only adapter in this repo that drove a Sandy CLI
agent, which meant the chutescoder-vs-codex experiment had exactly one
benchmark, and it was an agentic-coding one. Long-context-across-compaction is
the RLM design's central claim and the only place it has a mechanism the
baseline structurally lacks, so it is the thing most worth measuring.

Here the corpus is written into the sandbox as a file and the agent is asked to
answer from it. Scoring is unchanged -- it reuses `OolongAdapter`'s own
`_extract_answer` / numeric / exact-match logic -- so the only variable is the
harness.

What this is NOT
----------------
**These numbers are not comparable to `oolong`, and not comparable to published
OOLONG figures.** Handing the agent a *file* changes the task: instead of
holding 130k tokens of transcript in its context, the agent can read, chunk,
grep and summarise it. That is exactly the capability the RLM harness claims to
add, so it is a fair comparison *between arms* -- but it is a different task
from single-shot OOLONG, and reporting it against those numbers would be wrong.

It also means a task whose answer can be found with `grep` measures tool use
rather than long-context reasoning. OOLONG's questions are aggregate/reasoning
questions over dialogue, not needle retrieval, which is why this variant is
built on OOLONG rather than on S-NIAH -- S-NIAH in a sandbox is a `grep`
benchmark and would measure nothing.
"""
from __future__ import annotations

import base64
import os
import time
from typing import Any, Optional

from app.benchmarks.agent_usage import collect_agent_usage
from app.benchmarks.base import ItemResult
from app.benchmarks.registry import register_adapter
from app.core.logging import get_logger
from app.services.sandy_service import SandyService

from app.benchmarks.adapters.oolong import (
    OolongAdapter,
    _extract_answer,
    _normalize_prediction,
    score_answer,
)

logger = get_logger(__name__)

CORPUS_PATH = "/workspace/corpus.txt"
ANSWER_PATH = "/workspace/answer.txt"



@register_adapter("oolong_agentic")
class OolongAgenticAdapter(OolongAdapter):
    """OOLONG, answered by a CLI agent working over a file in a sandbox."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.sandy = SandyService()

    def get_name(self) -> str:
        return "oolong_agentic"

    def get_display_name(self) -> str:
        return "OOLONG (agentic)"

    def supports_parallel_items(self) -> bool:
        # Each item holds a sandbox for minutes; the sandbox host is the
        # bottleneck, not the model endpoint.
        return False

    def _agent_name(self) -> str:
        return (
            (getattr(self, "run_config", None) or {})
            .get("oolong_agentic", {})
            .get("agent")
            or os.getenv("OOLONG_AGENTIC_AGENT")
            or "codex"
        )

    async def _write_corpus(self, sandbox_id: str, text: str) -> bool:
        """Put the corpus in the sandbox, and prove it arrived intact.

        Written with `sandy.write_file` rather than shell `printf`/heredoc:
        an OOLONG context runs to hundreds of KB, and pushing that through a
        command line silently truncates. The first attempt did exactly that --
        48,514 bytes of an expected 198,514 -- which the length check below
        caught. Without that check it would have looked like the agent giving a
        wrong answer to a question whose evidence was missing.
        """
        encoded = base64.b64encode(text.encode("utf-8")).decode("ascii")
        if not await self.sandy.write_file(sandbox_id, "corpus.b64", encoded):
            logger.error("Corpus upload failed", chars=len(encoded))
            return False
        result = await self.sandy.execute_command(
            sandbox_id, f"base64 -d corpus.b64 > {CORPUS_PATH} && rm -f corpus.b64"
        )
        if (result or {}).get("exit_code") != 0:
            logger.error("Corpus decode failed", result=str(result)[:300])
            return False

        check = await self.sandy.execute_command(sandbox_id, f"wc -c < {CORPUS_PATH}")
        try:
            written = int(((check or {}).get("stdout") or "0").strip())
        except ValueError:
            return False
        expected = len(text.encode("utf-8"))
        if written != expected:
            logger.error("Corpus write mismatch", written=written, expected=expected)
            return False
        return True

    def _build_prompt(self, item: dict) -> str:
        return (
            f"The file {CORPUS_PATH} contains a long transcript.\n\n"
            f"Question: {item['question']}\n\n"
            "Work out the answer from the file. You may read it however you "
            "like -- in full, in chunks, or with shell tools.\n\n"
            f"When you are done, write ONLY the final answer to {ANSWER_PATH}, "
            "with no explanation, no units, no punctuation and no surrounding "
            "text. Then stop."
        )

    async def evaluate_item(self, item_id: str) -> ItemResult:
        item = await self._get_item(item_id)
        if not item:
            return ItemResult(item_id=item_id, error=f"Item {item_id} not found")

        agent_name = self._agent_name()
        prompt = self._build_prompt(item)
        start_time = time.time()
        sandbox_id: Optional[str] = None

        try:
            sandbox_id = await self.sandy.create_sandbox()
            if not sandbox_id:
                return ItemResult(
                    item_id=item_id,
                    error=self.sandy.last_error or "Could not create sandbox",
                )
            try:
                if not await self._write_corpus(
                    sandbox_id, item["context_window_text"]
                ):
                    return ItemResult(
                        item_id=item_id,
                        error="Failed to write the corpus into the sandbox",
                        metadata={"agent": agent_name},
                    )

                agent_result = await self.sandy.run_agent(
                    sandbox_id,
                    agent=agent_name,
                    model=self.model_slug,
                    prompt=prompt,
                    max_duration=int(os.getenv("OOLONG_AGENTIC_MAX_SECONDS", "900")),
                    raw_prompt=True,
                    env_vars={"CHUTES_API_KEY": self.client.get_api_key()},
                )
                agent_usage = await collect_agent_usage(self.sandy, sandbox_id)
                agent_summary = (agent_result or {}).get("summary") or {}

                read = await self.sandy.execute_command(
                    sandbox_id, f"cat {ANSWER_PATH} 2>/dev/null"
                )
                raw_answer = ((read or {}).get("stdout") or "").strip()
            finally:
                await self.sandy.terminate_sandbox(sandbox_id)

            latency_ms = int((time.time() - start_time) * 1000)

            if not raw_answer:
                return ItemResult(
                    item_id=item_id,
                    item_hash=self.compute_item_hash(item["question"]),
                    prompt=prompt,
                    response="",
                    expected=item["answer"],
                    is_correct=False,
                    score=0.0,
                    error=f"Agent did not write {ANSWER_PATH}",
                    latency_ms=latency_ms,
                    input_tokens=agent_usage.get("input_tokens"),
                    output_tokens=agent_usage.get("output_tokens"),
                    metadata={
                        "agent": agent_name,
                        "agent_summary": agent_summary,
                        "agent_usage": agent_usage,
                        "task": item["task"],
                        "context_len": item["context_len"],
                    },
                )

            # Identical scoring to single-shot OOLONG, so the arms differ only
            # in the harness -- including the raw/normalised pair, so the two
            # adapters cannot drift apart on the metric.
            extracted = _extract_answer(raw_answer)
            score_raw, correct_raw = score_answer(
                item["answer"], extracted, item["answer_type"]
            )
            normalized = _normalize_prediction(extracted)
            score, is_correct = score_answer(
                item["answer"], normalized, item["answer_type"]
            )

            return ItemResult(
                item_id=item_id,
                item_hash=self.compute_item_hash(item["question"]),
                prompt=prompt,
                response=raw_answer,
                expected=item["answer"],
                is_correct=is_correct,
                score=score,
                latency_ms=latency_ms,
                input_tokens=agent_usage.get("input_tokens"),
                output_tokens=agent_usage.get("output_tokens"),
                metadata={
                    "agent": agent_name,
                    "agent_summary": agent_summary,
                    "agent_usage": agent_usage,
                    "extracted_answer": extracted,
                    "normalized_answer": normalized,
                    "score_raw": score_raw,
                    "is_correct_raw": correct_raw,
                    "score_normalized": score,
                    "normalization": "formatting_only",
                    "task": item["task"],
                    "task_group": item["task_group"],
                    "answer_type": item["answer_type"],
                    "context_len": item["context_len"],
                    # Loud, because these must never be quoted against `oolong`
                    # or against published OOLONG numbers. See module docstring.
                    "delivery": "file_in_sandbox",
                    "comparable_to_single_shot_oolong": False,
                },
            )
        except Exception as exc:
            logger.error("OOLONG agentic item failed", item_id=item_id, error=str(exc))
            return ItemResult(
                item_id=item_id,
                error=str(exc),
                latency_ms=int((time.time() - start_time) * 1000),
                metadata={"agent": agent_name},
            )
