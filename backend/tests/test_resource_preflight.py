"""Dataset resource preflight tests."""

from collections import namedtuple
from unittest.mock import MagicMock

from app.benchmarks.registry import get_all_adapters
from app.benchmarks.resource_preflight import (
    ADAPTER_DATASET_FOOTPRINTS,
    GIB,
    RESOURCE_PREFLIGHT_INSUFFICIENT_DISK,
    RESOURCE_PREFLIGHT_UNKNOWN_FOOTPRINT,
    DatasetCacheLocation,
    DatasetFootprint,
    check_dataset_disk_capacity,
    unknown_footprint,
)

DiskUsage = namedtuple("DiskUsage", "total used free")


def test_every_registered_adapter_declares_a_dataset_footprint() -> None:
    adapters = get_all_adapters()
    identity_adapters = {
        "deepswe",
        "swe_bench_verified",
        "terminal_bench",
        "terminal_bench_1",
        "terminal_bench_2",
        "terminal_bench_2_0",
        "terminal_bench_2_1",
        "terminal_bench_hard",
    }

    assert set(adapters) == set(ADAPTER_DATASET_FOOTPRINTS) | identity_adapters
    for name, adapter_class in adapters.items():
        adapter = adapter_class(MagicMock(), "resource-contract-test")
        footprint = adapter.get_dataset_footprint()
        assert footprint.basis, name
        assert footprint.bytes is None or footprint.bytes >= 0, name


def test_run_refuses_when_declared_footprint_does_not_fit(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HF_DATASETS_CACHE", str(tmp_path))
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))

    result = check_dataset_disk_capacity(
        {
            "oolong_agentic": DatasetFootprint(
                35 * GIB,
                DatasetCacheLocation.HF,
                "incident measurement",
            )
        },
        run_config=None,
        safety_margin_bytes=10 * GIB,
        disk_usage=lambda path: DiskUsage(100 * GIB, 60 * GIB, 40 * GIB),
    )

    assert result.allowed is False
    assert result.exclusion_reason == RESOURCE_PREFLIGHT_INSUFFICIENT_DISK
    assert "35.00 GiB" in result.message
    assert "10.00 GiB safety margin" in result.message
    assert "45.00 GiB total" in result.message


def test_run_proceeds_when_declared_footprint_and_margin_fit(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HF_DATASETS_CACHE", str(tmp_path))
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))

    result = check_dataset_disk_capacity(
        {
            "oolong_agentic": DatasetFootprint(
                35 * GIB,
                DatasetCacheLocation.HF,
                "incident measurement",
            )
        },
        run_config=None,
        safety_margin_bytes=10 * GIB,
        disk_usage=lambda path: DiskUsage(100 * GIB, 50 * GIB, 50 * GIB),
    )

    assert result.allowed is True
    assert result.exclusion_reason is None
    assert result.checks == (
        {
            "cache_paths": [str(tmp_path)],
            "probe_path": str(tmp_path),
            "adapters": ["oolong_agentic"],
            "declared_footprint_bytes": 35 * GIB,
            "safety_margin_bytes": 10 * GIB,
            "required_free_bytes": 45 * GIB,
            "free_bytes": 50 * GIB,
        },
    )


def test_unknown_footprint_refuses_without_explicit_override(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HF_DATASETS_CACHE", str(tmp_path))
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    declarations = {
        "unmeasured": unknown_footprint(
            DatasetCacheLocation.HF,
            "not measured",
        )
    }

    refused = check_dataset_disk_capacity(
        declarations,
        run_config=None,
        safety_margin_bytes=10 * GIB,
    )
    allowed = check_dataset_disk_capacity(
        declarations,
        run_config={"resource_preflight": {"allow_unknown_dataset_footprint": True}},
        safety_margin_bytes=10 * GIB,
        disk_usage=lambda path: DiskUsage(100 * GIB, 50 * GIB, 50 * GIB),
    )

    assert refused.allowed is False
    assert refused.exclusion_reason == RESOURCE_PREFLIGHT_UNKNOWN_FOOTPRINT
    assert "Unknown is never treated as zero" in refused.message
    assert "config.resource_preflight.allow_unknown_dataset_footprint=true" in refused.message
    assert allowed.allowed is True
    assert "explicit unknown-footprint override for: unmeasured" in allowed.message


def test_hf_preflight_probes_the_actual_hf_cache_path(tmp_path, monkeypatch) -> None:
    datasets_cache = tmp_path / "datasets"
    hub_cache = tmp_path / "hub"
    datasets_cache.mkdir()
    hub_cache.mkdir()
    monkeypatch.setenv("HF_DATASETS_CACHE", str(datasets_cache))
    monkeypatch.setenv("HF_HUB_CACHE", str(hub_cache))
    probed = []

    result = check_dataset_disk_capacity(
        {
            "adapter": DatasetFootprint(
                1 * GIB,
                DatasetCacheLocation.HF,
                "measurement",
            )
        },
        run_config=None,
        safety_margin_bytes=1 * GIB,
        disk_usage=lambda path: (probed.append(path) or DiskUsage(10 * GIB, 1 * GIB, 9 * GIB)),
    )

    assert result.allowed is True
    assert probed == [datasets_cache]
    assert result.checks[0]["cache_paths"] == [str(datasets_cache), str(hub_cache)]
