"""Pinned identities for the Terminal-Bench benchmark family.

The task IDs in this file are benchmark manifests, not convenience filters.
Changing one of them changes the benchmark and must be accompanied by updated
provenance in docs/BENCHMARK_IDENTITY.md and a new expected-count assertion.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TerminalBenchSpec:
    """An immutable, obtainable Terminal-Bench release."""

    display_name: str
    expected_count: int
    repository: str
    commit: str
    archive_sha256: str
    task_root: str
    task_format: str
    task_ids: tuple[str, ...]
    manifest_repository: str
    manifest_commit: str

    def __post_init__(self) -> None:
        if len(self.task_ids) != self.expected_count:
            raise ValueError(
                f"{self.display_name} manifest has {len(self.task_ids)} task IDs; "
                f"the named release requires {self.expected_count}"
            )

    @property
    def archive_url(self) -> str:
        return f"https://codeload.github.com/{self.repository}/tar.gz/{self.commit}"


# Source: harbor-framework/terminal-bench registry.json, dataset
# terminal-bench-core==0.1.1. The registry pins task contents to commit
# 91e10457b5410f16c44364da1a34cb6de8c488a5 and lists these 80 task IDs.
TERMINAL_BENCH_1_TASK_IDS = (
    "blind-maze-explorer-5x5",
    "blind-maze-explorer-algorithm",
    "blind-maze-explorer-algorithm.easy",
    "blind-maze-explorer-algorithm.hard",
    "build-initramfs-qemu",
    "build-linux-kernel-qemu",
    "build-tcc-qemu",
    "cartpole-rl-training",
    "chess-best-move",
    "conda-env-conflict-resolution",
    "configure-git-webserver",
    "count-dataset-tokens",
    "crack-7z-hash",
    "crack-7z-hash.easy",
    "crack-7z-hash.hard",
    "create-bucket",
    "cron-broken-network",
    "csv-to-parquet",
    "decommissioning-service-with-sensitive-data",
    "download-youtube",
    "eval-mteb",
    "eval-mteb.hard",
    "extract-moves-from-video",
    "extract-safely",
    "fibonacci-server",
    "fix-git",
    "fix-pandas-version",
    "fix-permissions",
    "get-bitcoin-nodes",
    "git-multibranch",
    "git-workflow-hack",
    "gpt2-codegolf",
    "grid-pattern-transform",
    "hello-world",
    "heterogeneous-dates",
    "hf-model-inference",
    "incompatible-python-fasttext",
    "incompatible-python-fasttext.base_with_hint",
    "intrusion-detection",
    "jupyter-notebook-server",
    "modernize-fortran-build",
    "new-encrypt-command",
    "nginx-request-logging",
    "oom",
    "openssl-selfsigned-cert",
    "organization-json-generator",
    "password-recovery",
    "path-tracing-reverse",
    "path-tracing",
    "play-zork",
    "polyglot-c-py",
    "polyglot-rust-c",
    "processing-pipeline",
    "prove-plus-comm",
    "pytorch-model-cli",
    "pytorch-model-cli.easy",
    "pytorch-model-cli.hard",
    "qemu-alpine-ssh",
    "qemu-startup",
    "raman-fitting",
    "raman-fitting.easy",
    "reshard-c4-data",
    "run-pdp11-code",
    "sanitize-git-repo",
    "sanitize-git-repo.hard",
    "security-vulhub-minio",
    "simple-sheets-put",
    "simple-web-scraper",
    "solana-data",
    "sqlite-db-truncate",
    "sqlite-with-gcov",
    "super-benchmark-upet",
    "swe-bench-astropy-1",
    "swe-bench-astropy-2",
    "swe-bench-fsspec",
    "swe-bench-langcodes",
    "tmux-advanced-workflow",
    "train-fasttext",
    "vim-terminal-task",
    "write-compressor",
)


# Source: the 89 task directories in the official Terminal-Bench 2.0 and 2.1
# repositories. Both releases intentionally have the same task membership;
# 2.1 changes task contents and metadata, not the task set.
TERMINAL_BENCH_2_TASK_IDS = (
    "adaptive-rejection-sampler",
    "bn-fit-modify",
    "break-filter-js-from-html",
    "build-cython-ext",
    "build-pmars",
    "build-pov-ray",
    "caffe-cifar-10",
    "cancel-async-tasks",
    "chess-best-move",
    "circuit-fibsqrt",
    "cobol-modernization",
    "code-from-image",
    "compile-compcert",
    "configure-git-webserver",
    "constraints-scheduling",
    "count-dataset-tokens",
    "crack-7z-hash",
    "custom-memory-heap-crash",
    "db-wal-recovery",
    "distribution-search",
    "dna-assembly",
    "dna-insert",
    "extract-elf",
    "extract-moves-from-video",
    "feal-differential-cryptanalysis",
    "feal-linear-cryptanalysis",
    "filter-js-from-html",
    "financial-document-processor",
    "fix-code-vulnerability",
    "fix-git",
    "fix-ocaml-gc",
    "gcode-to-text",
    "git-leak-recovery",
    "git-multibranch",
    "gpt2-codegolf",
    "headless-terminal",
    "hf-model-inference",
    "install-windows-3.11",
    "kv-store-grpc",
    "large-scale-text-editing",
    "largest-eigenval",
    "llm-inference-batching-scheduler",
    "log-summary-date-ranges",
    "mailman",
    "make-doom-for-mips",
    "make-mips-interpreter",
    "mcmc-sampling-stan",
    "merge-diff-arc-agi-task",
    "model-extraction-relu-logits",
    "modernize-scientific-stack",
    "mteb-leaderboard",
    "mteb-retrieve",
    "multi-source-data-merger",
    "nginx-request-logging",
    "openssl-selfsigned-cert",
    "overfull-hbox",
    "password-recovery",
    "path-tracing",
    "path-tracing-reverse",
    "polyglot-c-py",
    "polyglot-rust-c",
    "portfolio-optimization",
    "protein-assembly",
    "prove-plus-comm",
    "pypi-server",
    "pytorch-model-cli",
    "pytorch-model-recovery",
    "qemu-alpine-ssh",
    "qemu-startup",
    "query-optimize",
    "raman-fitting",
    "regex-chess",
    "regex-log",
    "reshard-c4-data",
    "rstan-to-pystan",
    "sam-cell-seg",
    "sanitize-git-repo",
    "schemelike-metacircular-eval",
    "sparql-university",
    "sqlite-db-truncate",
    "sqlite-with-gcov",
    "torch-pipeline-parallelism",
    "torch-tensor-parallelism",
    "train-fasttext",
    "tune-mjcf",
    "video-processing",
    "vulnerable-secret",
    "winning-avg-corewars",
    "write-compressor",
)


# Source: NVIDIA-NeMo/Evaluator commit bd952253260e7077973aadf5fc656e425d2758e1,
# `_TB_HARD_TASKS`. NVIDIA identifies this as the curated 47-task leaderboard
# subset. Every ID is a `difficulty: hard` task at upstream Terminal-Bench
# commit 74221fb; the only other hard-tagged task there, super-benchmark-upet,
# is not part of this manifest.
TERMINAL_BENCH_HARD_TASK_IDS = (
    "aimo-airline-departures",
    "blind-maze-explorer-5x5",
    "cartpole-rl-training",
    "causal-inference-r",
    "chem-property-targeting",
    "chem-rf",
    "circuit-fibsqrt",
    "cobol-modernization",
    "configure-git-webserver",
    "cross-entropy-method",
    "extract-moves-from-video",
    "feal-differential-cryptanalysis",
    "feal-linear-cryptanalysis",
    "form-filling",
    "git-multibranch",
    "gpt2-codegolf",
    "install-windows-3.11",
    "install-windows-xp",
    "lean4-proof",
    "make-doom-for-mips",
    "make-mips-interpreter",
    "mcmc-sampling-stan",
    "model-extraction-relu-logits",
    "movie-helper",
    "neuron-to-jaxley-conversion",
    "oom",
    "organization-json-generator",
    "parallel-particle-simulator",
    "parallelize-graph",
    "password-recovery",
    "path-tracing",
    "path-tracing-reverse",
    "play-zork",
    "play-zork-easy",
    "polyglot-rust-c",
    "prove-plus-comm",
    "pytorch-model-cli",
    "rare-mineral-allocation",
    "recover-obfuscated-files",
    "reverse-engineering",
    "run-pdp11-code",
    "stable-parallel-kmeans",
    "swe-bench-astropy-1",
    "swe-bench-astropy-2",
    "train-fasttext",
    "word2vec-from-scratch",
    "write-compressor",
)


TERMINAL_BENCH_1 = TerminalBenchSpec(
    display_name="Terminal-Bench 1.0 Core",
    expected_count=80,
    repository="harbor-framework/terminal-bench",
    commit="91e10457b5410f16c44364da1a34cb6de8c488a5",
    archive_sha256="c0dd697e58672818836607a3af007c16b67368237703c4f6fb4b713eaf9ef0f4",
    task_root="tasks",
    task_format="legacy",
    task_ids=TERMINAL_BENCH_1_TASK_IDS,
    manifest_repository="harbor-framework/terminal-bench",
    manifest_commit="d28711d0da2675d0bb1d56de45ae5df6082438a3",
)

TERMINAL_BENCH_2_0 = TerminalBenchSpec(
    display_name="Terminal-Bench 2.0",
    expected_count=89,
    repository="harbor-framework/terminal-bench-2",
    commit="2fd12b88aafdd04a52c298e3940bcb189f9766d6",
    archive_sha256="6718ca1bd5c3536c9099cb1b2cf22a78c7c7159a85a26053a2d85d20dc0b6f4d",
    task_root="",
    task_format="harbor",
    task_ids=TERMINAL_BENCH_2_TASK_IDS,
    manifest_repository="harbor-framework/terminal-bench-2",
    manifest_commit="2fd12b88aafdd04a52c298e3940bcb189f9766d6",
)

TERMINAL_BENCH_2_1 = TerminalBenchSpec(
    display_name="Terminal-Bench 2.1",
    expected_count=89,
    repository="harbor-framework/terminal-bench-2-1",
    commit="5c8eadf1f393183288fa08b8f73ca9a469cc5e00",
    archive_sha256="f9298006a7462a0b933c880aed8494c8c7b68ea4f97d792460f1bdffff2e6620",
    task_root="tasks",
    task_format="harbor",
    task_ids=TERMINAL_BENCH_2_TASK_IDS,
    manifest_repository="harbor-framework/terminal-bench-2-1",
    manifest_commit="5c8eadf1f393183288fa08b8f73ca9a469cc5e00",
)

TERMINAL_BENCH_HARD = TerminalBenchSpec(
    display_name="Terminal-Bench Hard (47-task leaderboard subset)",
    expected_count=47,
    repository="harbor-framework/terminal-bench",
    commit="74221fb0b6b5a7f88e53bed5726edaaf236348c9",
    archive_sha256="38298cd844fa3798a720f4e9de8872c69216d7b7760148368a8b20381e8500ae",
    task_root="tasks",
    task_format="legacy",
    task_ids=TERMINAL_BENCH_HARD_TASK_IDS,
    manifest_repository="NVIDIA-NeMo/Evaluator",
    manifest_commit="bd952253260e7077973aadf5fc656e425d2758e1",
)
