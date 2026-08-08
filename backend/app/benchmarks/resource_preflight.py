"""Fail-closed dataset disk-capacity checks for benchmark launches."""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

KIB = 1024
MIB = 1024 * KIB
GIB = 1024 * MIB

RESOURCE_PREFLIGHT_INSUFFICIENT_DISK = "resource_preflight_insufficient_disk"
RESOURCE_PREFLIGHT_UNKNOWN_FOOTPRINT = "resource_preflight_unknown_dataset_footprint"
UNKNOWN_FOOTPRINT_OVERRIDE = "allow_unknown_dataset_footprint"


class DatasetCacheLocation(str, Enum):
    """Cache family an adapter writes while obtaining its dataset."""

    HF = "huggingface"
    BENCH_DATA = "bench_data"
    TEMP = "temp"


@dataclass(frozen=True)
class DatasetFootprint:
    """Maximum expected on-disk dataset growth and how it was established."""

    bytes: int | None
    cache_location: DatasetCacheLocation
    basis: str

    def __post_init__(self) -> None:
        if self.bytes is not None and self.bytes < 0:
            raise ValueError("dataset footprint cannot be negative")
        if not self.basis.strip():
            raise ValueError("dataset footprint basis cannot be empty")


@dataclass(frozen=True)
class ResourcePreflightResult:
    """A launch decision plus evidence suitable for a durable run event."""

    allowed: bool
    exclusion_reason: str | None
    message: str
    checks: tuple[dict[str, Any], ...]


@dataclass
class _DeviceRequirement:
    probe_path: Path
    requested_paths: set[str]
    adapters: list[str]
    declared_bytes: int = 0


def unknown_footprint(
    cache_location: DatasetCacheLocation,
    basis: str = "not measured",
) -> DatasetFootprint:
    """Declare an unknown footprint explicitly; unknown never means zero."""

    return DatasetFootprint(bytes=None, cache_location=cache_location, basis=basis)


# Declarations for adapters whose benchmark identity is defined in their
# implementation module. Pinned corpus identities (Terminal-Bench, DeepSWE,
# and SWE-bench Verified) carry the same field beside expected_count instead.
ADAPTER_DATASET_FOOTPRINTS: dict[str, DatasetFootprint] = {
    "aa_omniscience": DatasetFootprint(
        1 * MIB,
        DatasetCacheLocation.BENCH_DATA,
        "155,928-byte realized cache measured on own_postgres on 2026-08-08",
    ),
    "aa_lcr": DatasetFootprint(
        8 * MIB,
        DatasetCacheLocation.BENCH_DATA,
        "4,174,622-byte realized dataset plus document cache measured on own_postgres on 2026-08-08",
    ),
    "aime_2025": unknown_footprint(
        DatasetCacheLocation.BENCH_DATA,
        "pinned AIME 2025 cache has not been measured",
    ),
    "critpt": DatasetFootprint(
        1 * MIB,
        DatasetCacheLocation.BENCH_DATA,
        "310,656-byte realized cache measured on own_postgres on 2026-08-08",
    ),
    "deepresearch_bench": DatasetFootprint(
        8 * MIB,
        DatasetCacheLocation.BENCH_DATA,
        "6,168,382-byte realized query/reference cache measured on own_postgres on 2026-08-08",
    ),
    "gdpval_aa": unknown_footprint(
        DatasetCacheLocation.BENCH_DATA,
        "reference-file downloads are item-dependent and have not been measured",
    ),
    "gpqa_diamond": DatasetFootprint(
        2 * MIB,
        DatasetCacheLocation.BENCH_DATA,
        "1,433,610-byte realized cache measured on own_postgres on 2026-08-08",
    ),
    "hle": unknown_footprint(
        DatasetCacheLocation.BENCH_DATA,
        "Humanity's Last Exam cache has not been measured",
    ),
    "ifbench": unknown_footprint(
        DatasetCacheLocation.BENCH_DATA,
        "IFBench cache has not been measured",
    ),
    "kimi_vendor_verifier": DatasetFootprint(
        64 * MIB,
        DatasetCacheLocation.BENCH_DATA,
        "63,197,805-byte realized archive and extracted cache measured on own_postgres on 2026-08-08",
    ),
    "livecodebench": unknown_footprint(
        DatasetCacheLocation.BENCH_DATA,
        "LiveCodeBench cache has not been measured",
    ),
    "mmlu_pro": DatasetFootprint(
        16 * MIB,
        DatasetCacheLocation.BENCH_DATA,
        "12,988,018 bytes of realized test/validation caches measured on own_postgres on 2026-08-08",
    ),
    "oolong": DatasetFootprint(
        35 * GIB,
        DatasetCacheLocation.HF,
        "own_postgres incident: 12 GB oolong-synth hub data plus 23 GB derived Arrow data",
    ),
    "oolong_agentic": DatasetFootprint(
        35 * GIB,
        DatasetCacheLocation.HF,
        "own_postgres incident: 12 GB oolong-synth hub data plus 23 GB derived Arrow data",
    ),
    "oolong_pairs": DatasetFootprint(
        18 * GIB,
        DatasetCacheLocation.HF,
        "own_postgres incident: 18 GB oolong-real Hugging Face cache",
    ),
    "s_niah": unknown_footprint(
        DatasetCacheLocation.BENCH_DATA,
        "in-memory generator has not been measured for temporary spill or cache growth",
    ),
    "scicode": DatasetFootprint(
        1 * GIB,
        DatasetCacheLocation.BENCH_DATA,
        "1,051,710,225-byte realized dataset and HDF5 cache measured on own_postgres on 2026-08-08",
    ),
    "swe_bench_pro": DatasetFootprint(
        32 * MIB,
        DatasetCacheLocation.BENCH_DATA,
        "31,492,057 bytes of realized Hugging Face cache measured on own_postgres on 2026-08-08",
    ),
    "tau_bench_telecom": DatasetFootprint(
        1 * GIB,
        DatasetCacheLocation.BENCH_DATA,
        "905,741,206-byte realized repository cache measured on own_postgres on 2026-08-08",
    ),
}

for _affine_name, _cache_location in {
    "affine_print": DatasetCacheLocation.BENCH_DATA,
    "affine_lgc_v2": DatasetCacheLocation.BENCH_DATA,
    "affine_game": DatasetCacheLocation.BENCH_DATA,
    "affine_ded": DatasetCacheLocation.HF,
    "affine_cde": DatasetCacheLocation.HF,
    "affine_lgc": DatasetCacheLocation.HF,
    "affine_abd": DatasetCacheLocation.HF,
    "affine_swe_pro": DatasetCacheLocation.BENCH_DATA,
}.items():
    ADAPTER_DATASET_FOOTPRINTS[_affine_name] = unknown_footprint(
        _cache_location,
        f"{_affine_name} resource footprint has not been measured",
    )


def _cache_paths(
    location: DatasetCacheLocation,
    bench_data_dir: str | Path | None,
) -> tuple[Path, ...]:
    if location is DatasetCacheLocation.HF:
        hf_home = Path(
            os.getenv("HF_HOME") or Path.home() / ".cache" / "huggingface"
        )
        return (
            Path(os.getenv("HF_DATASETS_CACHE") or hf_home / "datasets"),
            Path(os.getenv("HF_HUB_CACHE") or hf_home / "hub"),
        )
    if location is DatasetCacheLocation.BENCH_DATA:
        return (
            Path(
                bench_data_dir
                or os.getenv("BENCH_DATA_DIR")
                or "/tmp/chutes-bench-data"
            ),
        )
    if location is DatasetCacheLocation.TEMP:
        return (Path(tempfile.gettempdir()),)
    raise AssertionError(f"unhandled cache location: {location}")


def _existing_probe_path(requested_path: Path) -> Path:
    probe = requested_path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return probe


def _human_bytes(value: int) -> str:
    for unit, divisor in (("TiB", 1024**4), ("GiB", GIB), ("MiB", MIB), ("KiB", KIB)):
        if value >= divisor:
            return f"{value / divisor:.2f} {unit}"
    return f"{value} bytes"


def _unknown_override_enabled(run_config: dict[str, Any] | None) -> bool:
    if not isinstance(run_config, dict):
        return False
    preflight_config = run_config.get("resource_preflight")
    return (
        isinstance(preflight_config, dict)
        and preflight_config.get(UNKNOWN_FOOTPRINT_OVERRIDE) is True
    )


def check_dataset_disk_capacity(
    declarations: Mapping[str, DatasetFootprint],
    *,
    run_config: dict[str, Any] | None,
    safety_margin_bytes: int,
    bench_data_dir: str | Path | None = None,
    disk_usage: Callable[[Path], Any] = shutil.disk_usage,
    stat: Callable[[Path], Any] = os.stat,
) -> ResourcePreflightResult:
    """Refuse a run before preload if its declared data cannot fit safely."""

    if safety_margin_bytes < 0:
        raise ValueError("dataset disk safety margin cannot be negative")

    unknown = sorted(name for name, footprint in declarations.items() if footprint.bytes is None)
    if unknown and not _unknown_override_enabled(run_config):
        names = ", ".join(unknown)
        message = (
            f"Dataset resource preflight refused: {names} has an unknown dataset footprint. "
            "Unknown is never treated as zero. To accept that unmeasured risk for this run, set "
            f"config.resource_preflight.{UNKNOWN_FOOTPRINT_OVERRIDE}=true."
        )
        return ResourcePreflightResult(
            allowed=False,
            exclusion_reason=RESOURCE_PREFLIGHT_UNKNOWN_FOOTPRINT,
            message=message,
            checks=(),
        )

    requirements: dict[int, _DeviceRequirement] = {}
    for adapter_name, footprint in declarations.items():
        adapter_devices: set[int] = set()
        for requested_path in _cache_paths(footprint.cache_location, bench_data_dir):
            probe_path = _existing_probe_path(requested_path)
            device = int(stat(probe_path).st_dev)
            requirement = requirements.setdefault(
                device,
                _DeviceRequirement(
                    probe_path=probe_path,
                    requested_paths=set(),
                    adapters=[],
                ),
            )
            requirement.requested_paths.add(str(requested_path))
            if device not in adapter_devices:
                requirement.adapters.append(adapter_name)
                requirement.declared_bytes += footprint.bytes or 0
                adapter_devices.add(device)

    checks: list[dict[str, Any]] = []
    for requirement in requirements.values():
        usage = disk_usage(requirement.probe_path)
        required_bytes = requirement.declared_bytes + safety_margin_bytes
        check = {
            "cache_paths": sorted(requirement.requested_paths),
            "probe_path": str(requirement.probe_path),
            "adapters": sorted(requirement.adapters),
            "declared_footprint_bytes": requirement.declared_bytes,
            "safety_margin_bytes": safety_margin_bytes,
            "required_free_bytes": required_bytes,
            "free_bytes": int(usage.free),
        }
        checks.append(check)
        if usage.free < required_bytes:
            paths = ", ".join(check["cache_paths"])
            message = (
                f"Dataset resource preflight refused: {paths} has {_human_bytes(usage.free)} free; "
                f"the selected adapters require {_human_bytes(requirement.declared_bytes)} of "
                f"declared dataset footprint plus a {_human_bytes(safety_margin_bytes)} safety "
                f"margin ({_human_bytes(required_bytes)} total)."
            )
            return ResourcePreflightResult(
                allowed=False,
                exclusion_reason=RESOURCE_PREFLIGHT_INSUFFICIENT_DISK,
                message=message,
                checks=tuple(checks),
            )

    message = "Dataset resource preflight passed"
    if unknown:
        message += f" with explicit unknown-footprint override for: {', '.join(unknown)}"
    return ResourcePreflightResult(
        allowed=True,
        exclusion_reason=None,
        message=message,
        checks=tuple(checks),
    )
