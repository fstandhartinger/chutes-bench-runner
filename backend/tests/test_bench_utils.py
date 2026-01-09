"""Tests for benchmark utils cache behavior."""
from pathlib import Path

from app.benchmarks.utils import get_bench_data_dir
from app.core.config import get_settings


def test_bench_data_dir_prefers_sandy_cache(tmp_path, monkeypatch):
    monkeypatch.delenv("BENCH_DATA_DIR", raising=False)
    monkeypatch.setenv("SANDY_CACHE_DIR", str(tmp_path / "cache"))
    get_settings.cache_clear()

    cache_dir = get_bench_data_dir()
    expected = Path(tmp_path / "cache" / "chutes-bench-data")
    assert cache_dir == expected
    assert cache_dir.exists()
