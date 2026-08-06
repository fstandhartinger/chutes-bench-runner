"""Kimi Vendor Verifier (K2VV) benchmark adapter."""
from __future__ import annotations

import asyncio
import json
import tarfile
import time
from pathlib import Path
from typing import Any, AsyncIterator, Optional

from jsonschema import ValidationError, validate

from app.benchmarks.base import BenchmarkAdapter, ItemResult
from app.benchmarks.registry import register_adapter
from app.benchmarks.utils import download_http_file_async, get_bench_data_dir
from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

DATASET_URL_DEFAULT = "https://statics.moonshot.cn/k2vv/tool-calls.tar.gz"
DATASET_ARCHIVE_NAME = "tool-calls.tar.gz"
DATASET_FOLDER = "tool-calls"
CACHE_SUBDIR = "k2vv"
REFERENCE_MODEL_DEFAULT = "kimi-k2-0905-preview"
DEFAULT_ITEM_CONCURRENCY = 5


def _safe_extract(archive: Path, destination: Path) -> None:
    destination = destination.resolve()
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            member_path = (destination / member.name).resolve()
            if not str(member_path).startswith(str(destination)):
                raise RuntimeError(f"Unsafe path in archive: {member.name}")
        tar.extractall(path=destination)


def _parse_json_override(value: Optional[str]) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        logger.warning("Failed to parse K2VV overrides JSON", error=str(exc))
        return {}
    if not isinstance(parsed, dict):
        logger.warning("K2VV overrides JSON must be an object")
        return {}
    return parsed


def _extract_prompt(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if isinstance(message, dict):
            if message.get("role") == "user" and isinstance(message.get("content"), str):
                return message["content"]
    for message in messages:
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            return message["content"]
    return ""


def compute_tool_call_metrics(results: list[ItemResult]) -> dict[str, Any]:
    tp = fp = fn = tn = 0
    finish_tool_calls = 0
    successful_tool_calls = 0
    schema_errors = 0
    compared = 0

    for result in results:
        metadata = result.metadata or {}
        candidate_finish = metadata.get("candidate_finish_reason")
        reference_finish = metadata.get("reference_finish_reason")
        if reference_finish is None:
            continue
        compared += 1
        candidate_tool = candidate_finish == "tool_calls"
        reference_tool = reference_finish == "tool_calls"

        if candidate_tool and reference_tool:
            tp += 1
        elif candidate_tool and not reference_tool:
            fp += 1
        elif not candidate_tool and reference_tool:
            fn += 1
        else:
            tn += 1

        if candidate_tool:
            finish_tool_calls += 1
            if metadata.get("tool_calls_valid") is True:
                successful_tool_calls += 1
            elif metadata.get("tool_calls_valid") is False:
                schema_errors += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    schema_accuracy = (
        successful_tool_calls / finish_tool_calls if finish_tool_calls > 0 else 0.0
    )

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "compared_items": compared,
        "tool_call_precision": precision,
        "tool_call_recall": recall,
        "tool_call_f1": f1,
        "count_finish_reason_tool_calls": finish_tool_calls,
        "count_successful_tool_call": successful_tool_calls,
        "schema_validation_error_count": schema_errors,
        "schema_accuracy": schema_accuracy,
    }


@register_adapter("kimi_vendor_verifier")
class KimiVendorVerifierAdapter(BenchmarkAdapter):
    """Kimi Vendor Verifier (K2VV) benchmark adapter."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        settings = get_settings()
        self.dataset_url = settings.k2vv_dataset_url or DATASET_URL_DEFAULT
        self.reference_model = settings.k2vv_reference_model or REFERENCE_MODEL_DEFAULT
        self.reference_results_path = settings.k2vv_reference_results_path
        self.request_overrides = _parse_json_override(settings.k2vv_request_overrides_json)
        self.timeout_seconds = settings.k2vv_timeout_seconds

        self._items: list[dict[str, Any]] = []
        self._items_by_id: dict[str, dict[str, Any]] = {}
        self._reference_finish_reasons: dict[str, str] = {}
        self._dataset_dir: Optional[Path] = None
        self._preload_lock = asyncio.Lock()
        self._reference_results_file: Optional[Path] = None

    def get_name(self) -> str:
        return "kimi_vendor_verifier"

    def get_display_name(self) -> str:
        return "Kimi Vendor Verifier"

    def supports_parallel_items(self) -> bool:
        return True

    def get_item_concurrency(self) -> Optional[int]:
        return DEFAULT_ITEM_CONCURRENCY

    def get_item_timeout_seconds(self, item_id: Optional[str] = None) -> Optional[int]:
        return self.timeout_seconds

    def requires_setup(self) -> bool:
        return False

    def get_setup_notes(self) -> Optional[str]:
        return "Downloads K2 Vendor Verifier sample dataset from Moonshot AI."

    async def get_total_items(self) -> int:
        if not self._items:
            await self.preload()
        return len(self._items)

    async def preload(self) -> None:
        if self._items:
            return
        async with self._preload_lock:
            if self._items:
                return
            dataset_dir = await self._ensure_dataset()
            self._dataset_dir = dataset_dir

            samples_path = dataset_dir / "samples.jsonl"
            if not samples_path.exists():
                raise FileNotFoundError(f"K2VV samples not found at {samples_path}")

            self._items = []
            self._items_by_id = {}
            with samples_path.open("r", encoding="utf-8") as handle:
                for index, line in enumerate(handle, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        raw = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    item_id = str(index)
                    item = {"id": item_id, "request": raw}
                    self._items.append(item)
                    self._items_by_id[item_id] = item

            self._reference_finish_reasons = self._load_reference_finish_reasons(dataset_dir)
            logger.info(
                "Loaded K2VV dataset",
                items=len(self._items),
                reference_model=self.reference_model,
                reference_results=str(self._reference_results_file) if self._reference_results_file else None,
            )

    async def enumerate_items(self) -> AsyncIterator[str]:
        if not self._items:
            await self.preload()
        for item in self._items:
            yield item["id"]

    async def evaluate_item(self, item_id: str) -> ItemResult:
        if not self._items:
            await self.preload()
        item = self._items_by_id.get(item_id)
        if not item:
            return ItemResult(item_id=item_id, error=f"Item {item_id} not found")

        raw_request = item.get("request") or {}
        request = dict(raw_request)
        messages = request.pop("messages", [])
        temperature = request.pop("temperature", 0.0)
        max_tokens = request.pop("max_tokens", None)
        request.pop("model", None)
        request.pop("stream", None)
        request.pop("stream_options", None)

        overrides = dict(self.request_overrides)
        if "temperature" in overrides:
            temperature = overrides.pop("temperature")
        if "max_tokens" in overrides:
            max_tokens = overrides.pop("max_tokens")
        request.update(overrides)

        if max_tokens is None:
            max_tokens = 4096

        tools = request.get("tools") or []
        prompt = _extract_prompt(messages)
        start_time = time.time()

        try:
            response = await self.client.run_inference(
                self.model_slug,
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=self.timeout_seconds,
                **request,
            )
        except Exception as exc:
            logger.error("K2VV inference failed", item_id=item_id, error=str(exc))
            return ItemResult(item_id=item_id, prompt=prompt, error=str(exc))

        latency_ms = int((time.time() - start_time) * 1000)
        choice = (response.get("choices") or [{}])[0]
        finish_reason = choice.get("finish_reason")
        message = choice.get("message") or {}
        tool_calls = message.get("tool_calls") if isinstance(message, dict) else None

        tool_calls_valid: Optional[bool] = None
        if finish_reason == "tool_calls":
            tool_calls_valid = self._validate_tool_calls(tool_calls, tools)

        reference_finish = self._reference_finish_reasons.get(item_id)
        candidate_tool = finish_reason == "tool_calls"
        reference_tool = reference_finish == "tool_calls" if reference_finish is not None else None
        is_correct = None
        score = None
        if reference_tool is not None:
            is_correct = candidate_tool == reference_tool
            score = 1.0 if is_correct else 0.0

        usage = response.get("usage") or {}
        return ItemResult(
            item_id=item_id,
            item_hash=self.compute_item_hash(raw_request),
            prompt=prompt,
            response=finish_reason,
            expected=reference_finish,
            is_correct=is_correct,
            score=score,
            judge_output={
                "finish_reason": finish_reason,
                "tool_calls_valid": tool_calls_valid,
            },
            latency_ms=latency_ms,
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
            metadata={
                "candidate_finish_reason": finish_reason,
                "reference_finish_reason": reference_finish,
                "tool_calls_valid": tool_calls_valid,
                "tool_calls_count": len(tool_calls) if isinstance(tool_calls, list) else 0,
                "request_temperature": temperature,
                "request_max_tokens": max_tokens,
                "reference_results_file": str(self._reference_results_file)
                if self._reference_results_file
                else None,
            },
        )

    async def postprocess(self, results: list[ItemResult]) -> dict[str, Any]:
        metrics = compute_tool_call_metrics(results)
        metrics.update(
            {
                "reference_model": self.reference_model,
                "reference_results_file": str(self._reference_results_file)
                if self._reference_results_file
                else None,
                "dataset_url": self.dataset_url,
                "score_override": metrics.get("tool_call_f1", 0.0),
            }
        )
        return metrics

    async def _ensure_dataset(self) -> Path:
        cache_root = get_bench_data_dir() / CACHE_SUBDIR
        cache_root.mkdir(parents=True, exist_ok=True)
        dataset_dir = cache_root / DATASET_FOLDER
        samples_path = dataset_dir / "samples.jsonl"
        if samples_path.exists():
            return dataset_dir

        archive_path = await download_http_file_async(
            self.dataset_url,
            cache_subdir=CACHE_SUBDIR,
            filename=DATASET_ARCHIVE_NAME,
            timeout_seconds=max(60, int(self.timeout_seconds or 0)),
        )
        _safe_extract(archive_path, cache_root)
        return dataset_dir

    def _resolve_reference_results_path(self, dataset_dir: Path) -> Optional[Path]:
        if self.reference_results_path:
            candidate = Path(self.reference_results_path)
            if candidate.exists():
                return candidate
            logger.warning("K2VV reference results path missing", path=str(candidate))

        candidate = dataset_dir / f"{self.reference_model}_results.jsonl"
        if candidate.exists():
            return candidate

        fallbacks = [
            dataset_dir / "kimi-k2-0905-preview_results.jsonl",
            dataset_dir / "kimi-k2-thinking_results.jsonl",
        ]
        for fallback in fallbacks:
            if fallback.exists():
                return fallback
        return None

    def _load_reference_finish_reasons(self, dataset_dir: Path) -> dict[str, str]:
        reference_path = self._resolve_reference_results_path(dataset_dir)
        if not reference_path:
            raise FileNotFoundError("K2VV reference results not found")
        self._reference_results_file = reference_path

        results: dict[str, str] = {}
        with reference_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                data_index = payload.get("data_index")
                if data_index is None:
                    continue
                finish_reason = payload.get("finish_reason")
                if finish_reason is None:
                    response = payload.get("response") or {}
                    choice = (response.get("choices") or [{}])[0]
                    finish_reason = choice.get("finish_reason")
                if finish_reason is None:
                    continue
                results[str(data_index)] = finish_reason
        if not results:
            logger.warning("K2VV reference results parsed empty", path=str(reference_path))
        return results

    def _validate_tool_calls(self, tool_calls: Any, tools: list[dict[str, Any]]) -> bool:
        if not isinstance(tool_calls, list) or not tool_calls:
            return False
        if not isinstance(tools, list) or not tools:
            return False
        return all(self._validate_tool_call(tc, tools) for tc in tool_calls)

    def _validate_tool_call(self, tool_call: dict[str, Any], tools: list[dict[str, Any]]) -> bool:
        if not isinstance(tool_call, dict):
            return False
        function = tool_call.get("function")
        if not isinstance(function, dict):
            return False
        tool_name = function.get("name")
        if not tool_name:
            return False
        schema = None
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            tool_func = tool.get("function") or {}
            if tool_func.get("name") == tool_name:
                schema = tool_func.get("parameters")
                break
        if not schema:
            return False

        args = function.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                return False
        if args is None:
            return False
        try:
            validate(instance=args, schema=schema)
        except ValidationError:
            return False
        return True
