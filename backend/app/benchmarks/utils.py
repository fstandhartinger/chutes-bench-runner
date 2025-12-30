"""Shared helpers for benchmark adapters."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import httpx
from huggingface_hub import hf_hub_download, snapshot_download

from app.core.config import get_settings


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
    timeout_seconds: int = 120,
) -> Path:
    """Download a file over HTTP into the benchmark cache (with caching)."""
    cache_dir = _resolve_cache_dir(cache_subdir)
    target_name = filename or url.split("/")[-1]
    target_path = cache_dir / target_name
    if target_path.exists():
        return target_path

    response = httpx.get(url, timeout=timeout_seconds, follow_redirects=True)
    response.raise_for_status()
    tmp_path = target_path.with_suffix(".tmp")
    tmp_path.write_bytes(response.content)
    tmp_path.replace(target_path)
    return target_path
