"""Immutable identity for the official 500-task SWE-bench Verified corpus."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SWEBenchVerifiedSpec:
    """Pinned dataset and official evaluation-harness identities."""

    display_name: str
    expected_count: int
    dataset_repository: str
    dataset_commit: str
    dataset_file: str
    dataset_file_sha256: str
    harness_repository: str
    harness_commit: str
    harness_version: str
    image_namespace: str
    image_tag: str

    def __post_init__(self) -> None:
        if self.expected_count <= 0:
            raise ValueError("SWE-bench Verified expected_count must be positive")
        for field_name in ("dataset_commit", "dataset_file_sha256", "harness_commit"):
            value = getattr(self, field_name)
            if len(value) != 40 and field_name != "dataset_file_sha256":
                raise ValueError(f"{field_name} must be a 40-character git commit")
            if field_name == "dataset_file_sha256" and len(value) != 64:
                raise ValueError("dataset_file_sha256 must be a 64-character SHA-256")


# Dataset identity was established from the official Hugging Face repository at
# the pinned commit. Its repository metadata declares one 500-row test split;
# the LFS object id is the SHA-256 of the exact parquet shard.
#
# Harness v4.1.0 is the official SWE-bench PyPI release whose tag resolves to
# the commit below. Official prebuilt evaluation images are selected by the
# harness and their pulled content digest is retained per item at evaluation
# time; the public image tag itself is not immutable.
SWE_BENCH_VERIFIED = SWEBenchVerifiedSpec(
    display_name="SWE-bench Verified (Sandy CLI scaffold)",
    expected_count=500,
    dataset_repository="princeton-nlp/SWE-bench_Verified",
    dataset_commit="c104f840cc67f8b6eec6f759ebc8b2693d585d4a",
    dataset_file="data/test-00000-of-00001.parquet",
    dataset_file_sha256="a45b1fe4e2f0c8390b2b2938ac83e92ed5979000856808f3679c07812e9e6dcd",
    harness_repository="SWE-bench/SWE-bench",
    harness_commit="726c5461e2ef52d83cf1ea2107870a8bb3328d57",
    harness_version="4.1.0",
    image_namespace="swebench",
    image_tag="latest",
)
