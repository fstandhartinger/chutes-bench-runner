# Terminal-Bench benchmark identity

Last verified: 2026-08-06.

This harness does not infer a benchmark from a Hugging Face dataset name or a
`difficulty` column. Every runnable Terminal-Bench adapter is tied to an exact
upstream commit, an explicit task-ID manifest, an expected item count, and the
SHA-256 of the downloaded source archive. Preload fails before enumeration if
any of those identity checks disagree.

## Adapters in this repository

| Adapter | Benchmark actually run | Expected items | Task format and execution |
| --- | --- | ---: | --- |
| `terminal_bench` | Current stable Terminal-Bench, pinned to 2.1 | 89 | Harbor task format; the agent modifies the task container directly and `tests/test.sh` emits the reward |
| `terminal_bench_1` | Terminal-Bench 1.0 leaderboard set, `terminal-bench-core==0.1.1` | 80 | Legacy Terminal-Bench format and legacy `solution.sh` protocol |
| `terminal_bench_2` | Current verified 2.x set, pinned to 2.1 | 89 | Same exact release as `terminal_bench_2_1` |
| `terminal_bench_2_0` | Terminal-Bench 2.0 | 89 | Harbor task format |
| `terminal_bench_2_1` | Terminal-Bench 2.1 | 89 | Harbor task format |
| `terminal_bench_hard` | Historical Terminal-Bench Hard leaderboard subset | 47 | Legacy Terminal-Bench format, pinned to the reported source revision |

The complete task-ID manifests used by the runnable adapters are checked into
[`backend/app/benchmarks/adapters/terminal_bench_identity.py`](../backend/app/benchmarks/adapters/terminal_bench_identity.py).
Numeric item IDs remain `0..N-1` in manifest order, so explicit `item_ids`
selection continues to work. Each result also records both the task-content
repository/commit and the manifest repository/commit.

## Exact provenance

### Terminal-Bench 1.0 Core

The official legacy repository calls the leaderboard dataset
`terminal-bench-core==0.1.1`. Its registry provides an explicit 80-item
`task_id_subset` and pins task content to commit
`91e10457b5410f16c44364da1a34cb6de8c488a5`:

- Manifest: [`registry.json` at repository commit `d28711d`](https://github.com/harbor-framework/terminal-bench/blob/d28711d0da2675d0bb1d56de45ae5df6082438a3/registry.json)
- Task content: [`terminal-bench` commit `91e10457`](https://github.com/harbor-framework/terminal-bench/commit/91e10457b5410f16c44364da1a34cb6de8c488a5)
- Official 1.0 leaderboard instruction: [`terminal-bench-core==0.1.1`](https://www.tbench.ai/leaderboard/terminal-bench/1.0)
- Expected count: **80**

The repository also records `terminal-bench-core==0.1.0` with 71 task IDs and
a mutable `terminal-bench-core==head`. Neither is exposed by an adapter because
the requested named releases are 1.0 Core v0.1.1 and current 2.x.

### Terminal-Bench 2.0

Terminal-Bench 2.0 is a separate benchmark, not the legacy repository's
current `tasks/` directory. The canonical list is the 89 task directories at
the root of the official 2.0 repository:

- Task list and content: [`terminal-bench-2` commit `2fd12b88`](https://github.com/harbor-framework/terminal-bench-2/tree/2fd12b88aafdd04a52c298e3940bcb189f9766d6)
- Official Harbor dataset: [`terminal-bench/terminal-bench-2`](https://hub.harborframework.com/datasets/terminal-bench/terminal-bench-2)
- Official Hugging Face mirror: [`harborframework/terminal-bench-2.0` at `f2e8c75e`](https://huggingface.co/datasets/harborframework/terminal-bench-2.0/tree/f2e8c75e23add71613117eecc9498f53bcd7e04e)
- Expected count: **89**

The adapter downloads the pinned GitHub source archive because it contains the
same Harbor task directories and gives the harness an immutable commit. All 89
tasks declare a prebuilt Docker image, Linux environment, and scalar Harbor
verifier.

### Terminal-Bench 2.1

Terminal-Bench 2.1 retains the same 89 task IDs and revises task contents and
metadata. The release announcement describes it as a verified iteration of
2.0, and the official repository is the canonical source used here:

- Task list and content: [`terminal-bench-2-1` commit `5c8eadf1`](https://github.com/harbor-framework/terminal-bench-2-1/tree/5c8eadf1f393183288fa08b8f73ca9a469cc5e00/tasks)
- Official Harbor dataset: [`terminal-bench/terminal-bench-2-1`](https://hub.harborframework.com/datasets/terminal-bench/terminal-bench-2-1/latest)
- Release announcement: [Terminal-Bench 2.1](https://www.tbench.ai/news/terminal-bench-2-1)
- Expected count: **89**

No official 2.1 Hugging Face repository was found. The official GitHub and
Harbor Hub datasets are public and obtainable.

### Terminal-Bench 3.0 status

An official public Harbor dataset named
[`terminal-bench/terminal-bench-3`](https://hub.harborframework.com/datasets/terminal-bench)
currently reports 75 tasks. However, the official
[`terminal-bench-3` GitHub repository](https://github.com/harbor-framework/terminal-bench-3)
still describes the suite as an ongoing construction effort and does not expose
the Harbor dataset's 75-task release manifest in its public `tasks/` tree.
There is no `terminal_bench_3` adapter here: adding one without a pinned,
reviewable public manifest would reintroduce the identity problem this change
is intended to prevent.

## Terminal-Bench Hard: 47-task and 44-task manifests

Artificial Analysis's own benchmark page links the general Terminal-Bench
repository and registry, but not a task-ID manifest. The historical benchmark
is nevertheless reproducible now because NVIDIA's public NeMo Evaluator checks
in the missing membership list:

- [`_TB_HARD_TASKS` at NeMo Evaluator commit `bd952253`](https://github.com/NVIDIA-NeMo/Evaluator/blob/bd952253260e7077973aadf5fc656e425d2758e1/src/nemo_evaluator/benchmarks/terminal_bench_hard.py)
  is explicitly identified as the **47-task curated leaderboard subset**.
- The [Falcon-H1R evaluation report](https://arxiv.org/abs/2601.02346) identifies
  Terminal-Bench commit `74221fb`, 47 hard tasks, and the Terminus 2 agent as
  its evaluation configuration.
- All 47 manifest IDs exist at
  [`terminal-bench` commit `74221fb`](https://github.com/harbor-framework/terminal-bench/tree/74221fb0b6b5a7f88e53bed5726edaaf236348c9/tasks)
  and all 47 have `difficulty: hard`. That revision contains one additional
  hard-tagged task, `super-benchmark-upet`; NVIDIA's manifest establishes that
  it is not in the historical 47-task leaderboard subset.
- Expected count: **47**

`terminal_bench_hard` therefore uses the NVIDIA membership manifest and task
content from the pinned upstream commit. It does not derive membership by
filtering a mutable difficulty column.

There is also a later **44-task** variant. The same NeMo source checks it in as
`_TB_HARD_AA_SPLIT_TASKS` and identifies it as the split used by the AA scoring
pipeline. It removes `causal-inference-r`, `install-windows-3.11`, `lean4-proof`,
and `mcmc-sampling-stan` from the historical 47, and adds
`super-benchmark-upet`. This agrees with the live Artificial Analysis page's
shipped `taskCount: 44` on 2026-08-06. No adapter is named for that evolving
split: `terminal_bench_hard` retains the 47-task identity specified in this
task, while the 44-task list and its provenance are recorded here for future
versioning. The live page is [Terminal-Bench Hard](https://artificialanalysis.ai/evaluations/terminalbench-hard).

## The former `ia03/terminal-bench` source

[`ia03/terminal-bench`](https://huggingface.co/datasets/ia03/terminal-bench)
is a useful community archive of 112 legacy-format tasks, generated at Hugging
Face commit `2e9f4a8195789dea113eba0fc4a02aa5eaf35b15`. Its `test` split is not an
official versioned leaderboard manifest. It contains mixed difficulty labels
and tasks outside `terminal-bench-core==0.1.1`.

No adapter now loads this dataset. Previously, the adapter registered as
`terminal_bench_hard` loaded all 112 rows without filtering; that result was
neither Terminal-Bench Hard nor Terminal-Bench 2.x.

## Pinned source archive checksums

The loader downloads GitHub's codeload archive for the task-content commit and
verifies it before parsing:

| Task content | SHA-256 |
| --- | --- |
| Terminal-Bench 1.0 Core, `91e10457` | `c0dd697e58672818836607a3af007c16b67368237703c4f6fb4b713eaf9ef0f4` |
| Terminal-Bench Hard, `74221fb` | `38298cd844fa3798a720f4e9de8872c69216d7b7760148368a8b20381e8500ae` |
| Terminal-Bench 2.0, `2fd12b88` | `6718ca1bd5c3536c9099cb1b2cf22a78c7c7159a85a26053a2d85d20dc0b6f4d` |
| Terminal-Bench 2.1, `5c8eadf1` | `f9298006a7462a0b933c880aed8494c8c7b68ea4f97d792460f1bdffff2e6620` |

## Startup checks and preserved integrity controls

For every runnable adapter, preload verifies all of the following before a run
can enumerate items:

1. The downloaded archive SHA-256 matches the pinned release.
2. Every canonical task contains its format's manifest (`task.yaml` or
   `task.toml`).
3. The loaded count equals the named benchmark's expected count.
4. Task IDs are unique and exactly match the checked-in manifest in order.

The existing incident-driven controls remain in both execution paths:

- the local answer key is removed by `_withhold_answer_key` before the agent
  can access it;
- `_verify_container_clean` checks the actual task container from outside;
- benchmark source hosts, including Harbor Hub and tbench.ai, are sealed during
  the agent phase and restored only for verification;
- agent token usage and the effective agent/test budgets are recorded even on
  failed items;
- `classify_agent_exit` and bare transport exclusions retain the distinction
  between infrastructure loss and scored harness robustness;
- explicit `item_ids` selection is still honored under the concrete adapter
  name, with immediate family defaults as a fallback (`terminal_bench` for
  `_1`, `_2`, and `_hard`; `terminal_bench_2` for `_2_0` and `_2_1`).
