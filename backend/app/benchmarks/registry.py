"""Benchmark adapter registry."""
from typing import Optional, Type

from app.benchmarks.base import BenchmarkAdapter
from app.services.chutes_client import ChutesClient

# Registry of benchmark adapters
_adapters: dict[str, Type[BenchmarkAdapter]] = {}


def register_adapter(name: str) -> callable:
    """Decorator to register a benchmark adapter."""

    def decorator(cls: Type[BenchmarkAdapter]) -> Type[BenchmarkAdapter]:
        _adapters[name] = cls
        return cls

    return decorator


def get_adapter(name: str, client: ChutesClient, model_slug: str) -> Optional[BenchmarkAdapter]:
    """Get an adapter instance by name."""
    if name not in _adapters:
        return None
    return _adapters[name](client, model_slug)


def get_all_adapters() -> dict[str, Type[BenchmarkAdapter]]:
    """Get all registered adapters."""
    return _adapters.copy()


# Import adapters to trigger registration
from app.benchmarks.adapters import *  # noqa: F401, F403, E402








