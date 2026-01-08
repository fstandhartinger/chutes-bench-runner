"""Signed JSON export service for benchmark runs."""
from __future__ import annotations

import base64
import hashlib
import io
import json
import zipfile
from datetime import datetime
from typing import Any, Optional

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.run import BenchmarkItemResult, BenchmarkRun, BenchmarkRunBenchmark

logger = get_logger(__name__)
settings = get_settings()

RESULTS_SCHEMA_VERSION = "1.0"
SIGNATURE_ALGORITHM = "ed25519"


class SigningKeyError(RuntimeError):
    """Raised when signing keys are missing or invalid."""


def _decode_key_bytes(key_value: str) -> bytes:
    return base64.b64decode(key_value.encode("utf-8"))


def _load_private_key() -> Ed25519PrivateKey:
    raw_key = settings.bench_signing_private_key
    if not raw_key:
        raise SigningKeyError("BENCH_SIGNING_PRIVATE_KEY is not configured")
    try:
        if raw_key.startswith("-----BEGIN"):
            return serialization.load_pem_private_key(raw_key.encode("utf-8"), password=None)
        return Ed25519PrivateKey.from_private_bytes(_decode_key_bytes(raw_key))
    except Exception as exc:
        raise SigningKeyError("Invalid BENCH_SIGNING_PRIVATE_KEY") from exc


def _load_public_key() -> Ed25519PublicKey:
    raw_key = settings.bench_signing_public_key
    if raw_key:
        try:
            if raw_key.startswith("-----BEGIN"):
                return serialization.load_pem_public_key(raw_key.encode("utf-8"))
            return Ed25519PublicKey.from_public_bytes(_decode_key_bytes(raw_key))
        except Exception as exc:
            raise SigningKeyError("Invalid BENCH_SIGNING_PUBLIC_KEY") from exc

    # Fall back to deriving from private key if only private key is provided
    private_key = _load_private_key()
    return private_key.public_key()


def _public_key_raw(public_key: Ed25519PublicKey) -> bytes:
    return public_key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )


def _public_key_fingerprint(public_key: Ed25519PublicKey) -> str:
    return hashlib.sha256(_public_key_raw(public_key)).hexdigest()


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _isoformat(value: Optional[datetime]) -> Optional[str]:
    if not value:
        return None
    return value.replace(microsecond=0).isoformat() + "Z"


async def _build_results_payload(db: AsyncSession, run: BenchmarkRun) -> dict[str, Any]:
    result = await db.execute(
        select(BenchmarkRunBenchmark).where(BenchmarkRunBenchmark.run_id == run.id)
    )
    run_benchmarks = result.scalars().all()

    benchmarks_payload: list[dict[str, Any]] = []
    for rb in run_benchmarks:
        items_result = await db.execute(
            select(BenchmarkItemResult)
            .where(BenchmarkItemResult.run_benchmark_id == rb.id)
            .order_by(BenchmarkItemResult.created_at)
        )
        items = items_result.scalars().all()

        benchmarks_payload.append(
            {
                "benchmark_name": rb.benchmark_name,
                "status": rb.status,
                "score": rb.score,
                "metrics": rb.metrics,
                "total_items": rb.total_items,
                "sampled_items": rb.sampled_items,
                "completed_items": rb.completed_items,
                "sampled_item_ids": rb.sampled_item_ids,
                "items": [
                    {
                        "item_id": item.item_id,
                        "item_hash": item.item_hash,
                        "prompt": item.prompt,
                        "response": item.response,
                        "expected": item.expected,
                        "is_correct": item.is_correct,
                        "score": item.score,
                        "latency_ms": item.latency_ms,
                        "input_tokens": item.input_tokens,
                        "output_tokens": item.output_tokens,
                        "error": item.error,
                        "test_code": item.test_code,
                        "item_metadata": item.item_metadata,
                        "judge_output": item.judge_output,
                        "created_at": _isoformat(item.created_at),
                    }
                    for item in items
                ],
            }
        )

    return {
        "schema_version": RESULTS_SCHEMA_VERSION,
        "exported_at": _isoformat(datetime.utcnow()),
        "run": {
            "id": run.id,
            "model_id": run.model_id,
            "model_slug": run.model_slug,
            "subset_pct": run.subset_pct,
            "subset_count": run.subset_count,
            "subset_seed": run.subset_seed,
            "status": run.status,
            "selected_benchmarks": run.selected_benchmarks,
            "overall_score": run.overall_score,
            "error_message": run.error_message,
            "started_at": _isoformat(run.started_at),
            "completed_at": _isoformat(run.completed_at),
            "created_at": _isoformat(run.created_at),
            "code_version": run.code_version,
            "git_sha": run.git_sha,
        },
        "benchmarks": benchmarks_payload,
    }


async def generate_signed_zip_export(
    db: AsyncSession,
    run: BenchmarkRun,
) -> tuple[str, bytes]:
    """Generate a signed zip containing JSON results and manifest."""
    public_key = _load_public_key()
    private_key = _load_private_key()
    fingerprint = _public_key_fingerprint(public_key)

    results_payload = await _build_results_payload(db, run)
    results_bytes = _json_bytes(results_payload)
    results_hash = hashlib.sha256(results_bytes).hexdigest()

    manifest_payload = {
        "schema_version": RESULTS_SCHEMA_VERSION,
        "run_id": run.id,
        "generated_at": _isoformat(datetime.utcnow()),
        "results_sha256": results_hash,
        "signature_algorithm": SIGNATURE_ALGORITHM,
        "public_key_fingerprint": fingerprint,
    }
    manifest_bytes = _json_bytes(manifest_payload)
    signature = private_key.sign(manifest_bytes)
    signature_b64 = base64.b64encode(signature).decode("utf-8")

    public_key_b64 = base64.b64encode(_public_key_raw(public_key)).decode("utf-8")

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("results.json", results_bytes)
        zf.writestr("manifest.json", manifest_bytes)
        zf.writestr("signature.txt", signature_b64)
        zf.writestr("public_key.txt", public_key_b64)

    filename = f"benchmark_results_{run.id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.zip"
    return filename, buffer.getvalue()


def get_public_key_info() -> dict[str, str]:
    """Return public key information for external verification."""
    public_key = _load_public_key()
    return {
        "algorithm": SIGNATURE_ALGORITHM,
        "public_key": base64.b64encode(_public_key_raw(public_key)).decode("utf-8"),
        "public_key_fingerprint": _public_key_fingerprint(public_key),
    }


def verify_signed_zip_export(content: bytes) -> dict[str, Any]:
    """Verify a signed zip export."""
    errors: list[str] = []
    manifest_payload: dict[str, Any] | None = None
    results_payload: dict[str, Any] | None = None

    try:
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            try:
                results_bytes = zf.read("results.json")
                manifest_bytes = zf.read("manifest.json")
                signature_raw = zf.read("signature.txt")
            except KeyError as exc:
                errors.append(f"Missing required file: {exc}")
                return {
                    "valid": False,
                    "signature_valid": False,
                    "hash_match": False,
                    "errors": errors,
                }
    except zipfile.BadZipFile:
        return {
            "valid": False,
            "signature_valid": False,
            "hash_match": False,
            "errors": ["Invalid zip archive"],
        }

    signature_valid = False
    hash_match = False

    try:
        public_key = _load_public_key()
    except SigningKeyError as exc:
        return {
            "valid": False,
            "signature_valid": False,
            "hash_match": False,
            "errors": [str(exc)],
        }

    try:
        signature_bytes = base64.b64decode(signature_raw.strip())
    except Exception:
        errors.append("Signature is not valid base64")
        signature_bytes = b""

    try:
        public_key.verify(signature_bytes, manifest_bytes)
        signature_valid = True
    except Exception as exc:
        errors.append(f"Signature verification failed: {exc}")

    try:
        manifest_payload = json.loads(manifest_bytes.decode("utf-8"))
    except Exception:
        errors.append("Manifest JSON is invalid")

    if manifest_payload:
        expected_fingerprint = _public_key_fingerprint(public_key)
        if manifest_payload.get("public_key_fingerprint") != expected_fingerprint:
            errors.append("Public key fingerprint mismatch")
            signature_valid = False

    results_hash = hashlib.sha256(results_bytes).hexdigest()
    if manifest_payload and manifest_payload.get("results_sha256") == results_hash:
        hash_match = True
    else:
        errors.append("Results hash does not match manifest")

    if signature_valid:
        try:
            results_payload = json.loads(results_bytes.decode("utf-8"))
        except Exception:
            errors.append("Results JSON is invalid")

    summary = {
        "run_id": results_payload.get("run", {}).get("id") if results_payload else None,
        "model_slug": results_payload.get("run", {}).get("model_slug") if results_payload else None,
        "subset_pct": results_payload.get("run", {}).get("subset_pct") if results_payload else None,
        "subset_count": results_payload.get("run", {}).get("subset_count") if results_payload else None,
        "subset_seed": results_payload.get("run", {}).get("subset_seed") if results_payload else None,
        "overall_score": results_payload.get("run", {}).get("overall_score") if results_payload else None,
        "exported_at": results_payload.get("exported_at") if results_payload else None,
        "benchmark_count": len(results_payload.get("benchmarks", [])) if results_payload else None,
        "public_key_fingerprint": manifest_payload.get("public_key_fingerprint") if manifest_payload else None,
    }

    return {
        "valid": signature_valid and hash_match,
        "signature_valid": signature_valid,
        "hash_match": hash_match,
        "errors": errors,
        **summary,
    }
