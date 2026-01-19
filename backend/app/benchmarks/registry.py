"""Benchmark adapter registry."""
from typing import Optional, Type

from app.benchmarks.base import BenchmarkAdapter
from app.services.inference_client import InferenceClient

# Registry of benchmark adapters
_adapters: dict[str, Type[BenchmarkAdapter]] = {}
_adapters_loaded: bool = False


def register_adapter(name: str) -> callable:
    """Decorator to register a benchmark adapter."""

    def decorator(cls: Type[BenchmarkAdapter]) -> Type[BenchmarkAdapter]:
        _adapters[name] = cls
        return cls

    return decorator


def _ensure_adapters_loaded() -> None:
    """Lazily load all adapters on first use."""
    global _adapters_loaded
    if _adapters_loaded:
        return
    _adapters_loaded = True
    try:
        from app.benchmarks.adapters import *  # noqa: F401, F403
    except Exception:
        pass  # Ignore errors during loading


def get_adapter(
    name: str,
    client: InferenceClient,
    model_slug: str,
    judge_client: Optional[InferenceClient] = None,
) -> Optional[BenchmarkAdapter]:
    """Get an adapter instance by name."""
    _ensure_adapters_loaded()
    if name not in _adapters:
        return None
    return _adapters[name](client, model_slug, judge_client=judge_client)


def get_all_adapters() -> dict[str, Type[BenchmarkAdapter]]:
    """Get all registered adapters."""
    _ensure_adapters_loaded()
    return _adapters.copy()




























