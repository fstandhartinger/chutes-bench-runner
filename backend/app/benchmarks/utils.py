"""Shared helpers for benchmark adapters."""
from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import Any, Iterable, Optional

import httpx
from huggingface_hub import hf_hub_download, snapshot_download

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def get_bench_data_dir() -> Path:
    """Return (and create) the benchmark data cache directory."""
    settings = get_settings()
    root = Path(settings.bench_data_dir)
    root.mkdir(parents=True, exist_ok=True)
    return root


def _resolve_cache_dir(cache_subdir: Optional[str]) -> Path:
    root = get_bench_data_dir()
    if not cache_subdir:
        return root
    target = root / cache_subdir
    target.mkdir(parents=True, exist_ok=True)
    return target


def download_hf_file(
    repo_id: str,
    filename: str,
    *,
    repo_type: str = "dataset",
    token: Optional[str] = None,
    cache_subdir: Optional[str] = None,
) -> Path:
    """Download a single file from Hugging Face into the benchmark cache."""
    cache_dir = _resolve_cache_dir(cache_subdir)
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type=repo_type,
        token=token,
        cache_dir=str(cache_dir),
    )
    return Path(path)


async def download_hf_file_async(
    repo_id: str,
    filename: str,
    *,
    repo_type: str = "dataset",
    token: Optional[str] = None,
    cache_subdir: Optional[str] = None,
) -> Path:
    """Async wrapper for download_hf_file to avoid blocking the event loop."""
    return await asyncio.to_thread(
        download_hf_file,
        repo_id,
        filename,
        repo_type=repo_type,
        token=token,
        cache_subdir=cache_subdir,
    )


def download_hf_snapshot(
    repo_id: str,
    *,
    repo_type: str = "dataset",
    token: Optional[str] = None,
    cache_subdir: Optional[str] = None,
    allow_patterns: Optional[Iterable[str]] = None,
) -> Path:
    """Download a full snapshot (or filtered subset) into the benchmark cache."""
    cache_dir = _resolve_cache_dir(cache_subdir)
    path = snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        token=token,
        cache_dir=str(cache_dir),
        allow_patterns=list(allow_patterns) if allow_patterns else None,
    )
    return Path(path)


def download_http_file(
    url: str,
    *,
    cache_subdir: Optional[str] = None,
    filename: Optional[str] = None,
    headers: Optional[dict[str, str]] = None,
    timeout_seconds: int = 120,
) -> Path:
    """Download a file over HTTP into the benchmark cache (with caching)."""
    cache_dir = _resolve_cache_dir(cache_subdir)
    target_name = filename or url.split("/")[-1]
    target_path = cache_dir / target_name
    if target_path.exists():
        return target_path

    tmp_path = target_path.with_suffix(f".{uuid.uuid4().hex}.tmp")
    try:
        with httpx.Client(timeout=httpx.Timeout(timeout_seconds, connect=10.0)) as client:
            with client.stream(
                "GET",
                url,
                follow_redirects=True,
                headers=headers,
            ) as response:
                response.raise_for_status()
                with tmp_path.open("wb") as handle:
                    for chunk in response.iter_bytes():
                        if chunk:
                            handle.write(chunk)
        tmp_path.replace(target_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
    return target_path


async def download_http_file_async(
    url: str,
    *,
    cache_subdir: Optional[str] = None,
    filename: Optional[str] = None,
    headers: Optional[dict[str, str]] = None,
    timeout_seconds: int = 120,
) -> Path:
    """Async wrapper for download_http_file to avoid blocking the event loop."""
    return await asyncio.to_thread(
        download_http_file,
        url,
        cache_subdir=cache_subdir,
        filename=filename,
        headers=headers,
        timeout_seconds=timeout_seconds,
    )


async def load_dataset_with_retry(
    *args: Any,
    max_attempts: int = 3,
    initial_delay_seconds: float = 2.0,
    timeout_seconds: float = 600.0,
    **kwargs: Any,
) -> Any:
    """Load a Hugging Face dataset with retries."""
    from datasets import load_dataset

    cache_root = get_bench_data_dir() / "hf"
    cache_root.mkdir(parents=True, exist_ok=True)
    kwargs.setdefault("cache_dir", str(cache_root))

    attempt = 0
    delay_seconds = initial_delay_seconds
    last_error: Optional[Exception] = None
    while attempt < max_attempts:
        attempt += 1
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(load_dataset, *args, **kwargs),
                timeout=timeout_seconds,
            )
        except Exception as exc:
            last_error = exc
            dataset_name = args[0] if args else "unknown"
            logger.warning(
                "Failed to load dataset",
                dataset=dataset_name,
                attempt=attempt,
                error=str(exc) or exc.__class__.__name__,
            )
            if attempt >= max_attempts:
                raise
            await asyncio.sleep(delay_seconds)
            delay_seconds = min(delay_seconds * 2, 30)
    if last_error:
        raise last_error
