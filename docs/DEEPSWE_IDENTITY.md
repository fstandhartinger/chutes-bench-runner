# DeepSWE v1.1 benchmark identity

Last verified: 2026-08-07.

The `deepswe` adapter runs the public DeepSWE v1.1 corpus through a selectable
Sandy CLI (`codex`, `chutescoder`, or `chutescoder-baseline`). This intentionally
differs from the official leaderboard's Pier + mini-SWE-agent + Modal scaffold,
so exported results identify the harness as
`sandy-cli-direct-workspace-separate-verifier`.
Select an arm with run config
`{"deepswe": {"agent": "chutescoder"}}` (also accepts
`chutescoder-baseline` and `codex`) or the `DEEPSWE_AGENT` environment variable.

For an OpenRouter run, `config.deepswe.context_limit_tokens` caps the generated
Codex-family agent config identically for all three arms. The cap is written to
`model_context_window` and to both context-window fields in the generated model
catalog; setting this knob with another provider fails explicitly because that
path does not install a runner-controlled config. Each retained DeepSWE item
stores a queryable `item_metadata.compaction_experiment` object containing the
arm, requested/configured context, structured compaction-event count, rollout
line count, tool-call counts by name, and item score.
Retained rollout parsing also populates `item_metadata.repository_access` with
RLM native-helper counts and paths plus Python-cell Docker/subprocess counts.

## Pinned corpus

- Official source: [`datacurve-ai/deep-swe`](https://github.com/datacurve-ai/deep-swe)
- Pinned task revision: [`435ee89ec2f2e2289f33b0da4f992f0b7b7266b9`](https://github.com/datacurve-ai/deep-swe/tree/435ee89ec2f2e2289f33b0da4f992f0b7b7266b9/tasks)
- GitHub codeload archive SHA-256: `34c6fabd3dad1770d753829378a81c3d8bb658ff255de9f01f3606e213cd2b46`
- Manifest: 113 task IDs checked into
  [`deepswe_identity.py`](../backend/app/benchmarks/adapters/deepswe_identity.py)
- Expected count: **113**

The official repository has a `v1.0.0` tag but no `v1.1` tag. The adapter
therefore pins the exact public commit above. At that revision all 113 task
images are tagged `v1.1`, all task manifests use schema 1.3, and every task
declares `environment_mode = "separate"` for its verifier.

Preload verifies archive SHA-256, exact task membership and order, uniqueness,
the expected count, a prebuilt image, a base commit, no-network agent and
verifier phases, a separate verifier environment, and a collect hook for every
task.

## Scoring and execution contract

DeepSWE v1.1's grader writes `reward.json`. Its ranking reward is binary: 1 only
when at least one fail-to-pass test exists, every fail-to-pass test passes, and
no pass-to-pass test fails. The adapter uses that `reward` as the item score and
records the accompanying `f2p`, `p2p`, and `partial` diagnostics. A run score is
the mean binary reward over non-excluded tasks. Agent timeouts and live-sandbox
CLI crashes are scored; missing sandboxes, broken transport, outer runner
timeouts, verifier failures, and unproven integrity seals are excluded.

Each pinned task declares the same limits:

- agent: 5,400 seconds (90 minutes);
- verifier: 1,800 seconds (30 minutes);
- agent-image pull/build allowance: 1,800 seconds;
- verifier-image build allowance: 1,800 seconds;
- patch collection: 300 seconds;
- CPU/RAM/storage: 2 CPUs, 8,192 MiB RAM, 20,480 MiB storage.

The worker's item timeout is derived from all of those item fields plus a
15-minute harness margin: 12,000 seconds (3 hours 20 minutes) at the default
1.0 agent-timeout multiplier. The Sandy TTL is derived from the same outer cap.

## Holdout, network, and verifier isolation

The full source archive contains both `tests/` and `solution/`, so it is never
uploaded to Sandy. The loader creates a sanitized agent archive in the worker
process and a tests-only verifier archive; reference-solution bytes are
discarded, and the downloaded source archive is deleted from the runner cache.
The uploaded sanitized archive is deleted immediately after extraction and its
absence is proved before the task image is pulled. The task image's `/app`
checkout is then copied into the sandbox-private `/workspace/repo` and mounted
back into the no-network task runtime at `/app`. A two-way sentinel proves both
paths expose the same writable bytes. The CLI enters `/workspace/repo`, so its
native repository helpers operate on the task files; task-image-only commands
remain available through the task-scoped Docker gateway. The adapter checks
from both namespaces that `/tests`, `/solution`, and exact held-out file hashes
are absent, including if those bytes were renamed, before the CLI starts.

During the agent phase the task container uses Docker `--network none`. The
Sandy CLI namespace also blackholes GitHub, Hugging Face, Datacurve, and Harbor
source hosts while retaining access to the model endpoint. Both the source-host
block and the container's Docker network mode are checked before the CLI runs.

Only after the agent has finished, its official collect hook has produced
`model.patch`, and the agent container has stopped does the adapter upload the
tests-only archive. It builds the task's `tests/Dockerfile`, starts a distinct
no-network verifier container at the pinned base commit, transfers only
`model.patch`, and runs `/tests/test.sh` there. The reference solution is never
uploaded or used for grading.

The shared checkout deliberately joins only the two components already in the
agent trust domain: the Sandy CLI and its editable no-network task runtime. The
raw Docker socket and shared Sandy cache remain absent, fresh-container and
cross-container gateway probes must still fail, and the verifier has neither
the shared mount nor any other path into the agent sandbox.
The temporary no-network copy container receives only the empty repository
directory, not the rest of the Sandy workspace.

## Storage behavior

DeepSWE has 113 unique task-image tags. The adapter streams one item at a time,
removes its agent/verifier containers and task-specific verifier image, and
attempts a non-forced removal of the source image so concurrent paired arms are
not disrupted. It also reaps only orphaned `deepswe_s…` resources; it never
runs a broad Docker prune.

Registry manifest inspection on 2026-08-07 resolved 97 of the 113 public ECR
images before rate limiting. Their compressed layer totals ranged from 0.704
GiB to 2.373 GiB (median 0.794 GiB, mean 0.922 GiB). The verifier image reuses
the source layers and adds the small task-specific tests layer rather than a
second full copy. Runtime writable-layer demand can still approach the task's
declared 20 GiB storage ceiling. Direct workspace access also materializes one
copy of `/app` in the sandbox workspace for the duration of an item, so a full
run must remain sequential on the current disk-constrained host.
