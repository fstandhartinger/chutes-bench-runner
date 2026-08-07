# OOLONG agentic four-arm audit and pilot — 2026-08-07

## Adapter correctness findings

- Dataset: `oolongbench/oolong-synth`, test split, pinned revision
  `f0d59eaf0febf130664cfceb710436c8e3216b2b`. Items 803 and 804 are shard 6
  rows 41 and 42. Explicit item IDs are honored; they are not replaced by
  hash sampling.
- Both items use context window ID 40033 from the `negation` dataset. The
  corpus is 6,125,881 UTF-8 bytes and 44,272 lines, with SHA-256
  `0add526f205e2a936b4507ee5e516a51baf8d46c0f44251cc2265521e2336109`.
  Its declared OOLONG `context_len` is 2,097,152. The corpus is uploaded to
  `/workspace/corpus.txt`; the prompt contains only its path and the question.
- Every upload is verified by byte count and SHA-256 inside the sandbox. A
  local-answer/cache probe must be clean. DNS is then disabled except for a
  pinned OpenRouter address, and GitHub/Hugging Face source hosts are mapped
  to loopback. Pre/post probes observed source `000`, public internet `000`,
  and OpenRouter `200`.
- Model identity was read from the `models` table, not guessed: UUID
  `7a24af8a-df2f-4d2e-86cf-55cdf6e28eef`, exact slug
  `deepseek/deepseek-v4-flash-0731`, provider `openrouter`, active.
- Scoring now follows the released OOLONG scorer: live enum names are handled,
  numeric predictions use integer parsing and `0.75 ** abs(gold-prediction)`,
  dates are parsed, comparison relations are canonicalized, and other answers
  use stripped case-sensitive exact match. The two gold answers are 9543
  (`False`) and 34721 (`True`).
- The file-delivery task is valid for comparison among these four arms but is
  not comparable to published single-shot OOLONG scores.
- Audit defects found and fixed before/while piloting: broken answer-type/date
  scoring, ignored explicit IDs, mutable/full-dataset loading, unproved corpus
  upload, public answer-source access, local cache leakage, insufficient PID
  budget, missing IPython in the RLM runtime, missing runtime preflight, and
  missing timeout-path rollout mirroring. The last defect means the completed
  pilot's timed-out chutescoder rows have evidence but no retained token
  events; commit `73321d5` fixes future runs.
- Sandy's nominal `maxDuration=900` stream is not a hard process kill. The
  worker's independent 1,200-second item cap is the effective boundary and
  correctly records `infrastructure_item_timeout` exclusions.

Focused post-fix tests: 56 passed. No Sandy restart was performed. One idle
benchmark worker was rolled to the runtime-preflight adapter after announcing
it; the other live worker was not interrupted.

## Final two-item sample

Experiment: `oolong-agentic-2m-paired-20260807-final-v6`. Each cell is one run,
so the table contains raw observations only. No token ratio, percentage,
mean, or efficiency claim is computed.

| Item | Arm | Score | Input tokens | Output tokens | Wall clock | Exclusion | Evidence |
|---|---|---:|---:|---:|---:|---|---|
| 803 (`False`) | prime-agent | excluded | 191,974* | 18,286* | 1,200 s cap | `infrastructure_item_timeout` | P803 |
| 803 (`False`) | chutescoder | excluded | unavailable | unavailable | 1,200 s cap | `infrastructure_item_timeout` | R803 |
| 803 (`False`) | chutescoder-baseline | excluded | unavailable | unavailable | 1,200 s cap | `infrastructure_item_timeout` | B803 |
| 803 (`False`) | codex | 2.9538409086585202e-195 | 3,514,263 | 124,931 | 1,055.782 s | — | C803 |
| 804 (`True`) | prime-agent | excluded | 342,175* | 14,792* | 1,200 s cap | `infrastructure_item_timeout` | P804 |
| 804 (`True`) | chutescoder | excluded | unavailable | unavailable | 1,200 s cap | `infrastructure_item_timeout` | R804 |
| 804 (`True`) | chutescoder-baseline | excluded | unavailable | unavailable | 1,200 s cap | `infrastructure_item_timeout` | B804 |
| 804 (`True`) | codex | excluded | unavailable | unavailable | 1,200 s cap | `infrastructure_item_timeout` | C804 |

`*` Prime token counts are the last retained cumulative usage event before
timeout, not a claim that no additional unreported tokens occurred. The other
excluded rows had zero retained usage events. Codex item 803 answered `11100`
against gold `9543`; `0.75 ** 1557` is the reported score. Aggregate exclusion
counts are Prime 2, chutescoder 2, baseline 2, Codex 1. Exclusions did not enter
scores; the three all-excluded arms have null aggregate score.

Run IDs:

- Prime: `ea9192af-9ede-46ad-8081-5fa7e6793583`
- Chutescoder: `d9d5f6c8-8066-47cd-9d62-4013d8ec68e4`
- Baseline: `c4751f2c-ae5b-4f04-886e-a2fd09413c81`
- Codex: `e579f508-fde8-4037-af54-d71785ab00cc`

The first RLM attempt (`992cd708-6532-4d81-bddc-7132b47a636a`) was canceled as
invalid preflight after its kernel repeatedly failed to import IPython. It has
no item rows and is not part of the table.

## Evidence manifest

All paths are on `own_postgres` (`94.130.222.43`). File sizes and hashes were
recomputed after the runs and matched the database.

| Ref | Size | SHA-256 | Path |
|---|---:|---|---|
| P803 | 10,124,664 | `b91d580405c1a19a8e324a09f5cb2a4cba045db5e7957bd0161a7322766016eb` | `/var/lib/sandy/cache/chutes-bench-evidence/ea9192af-9ede-46ad-8081-5fa7e6793583-7fcbec009b/oolong_agentic-6463308b27/803-9f006addc8/12332fb93853-89386a4b0d.tar.gz` |
| P804 | 6,285,496 | `ef6dbc4b0fb1ab9f7be19f4f476c01c05030b0a246d0dac8dd595b33ec245367` | `/var/lib/sandy/cache/chutes-bench-evidence/ea9192af-9ede-46ad-8081-5fa7e6793583-7fcbec009b/oolong_agentic-6463308b27/804-dccb3c52e7/f52cc348133d-c2371c274b.tar.gz` |
| R803 | 2,727 | `b702b9a49f73973ce639ba0e693f11c9d51949f8fbeb8a2b3f30121fcad24c52` | `/var/lib/sandy/cache/chutes-bench-evidence/d9d5f6c8-8066-47cd-9d62-4013d8ec68e4-af65700e99/oolong_agentic-6463308b27/803-9f006addc8/db7f7ca41d59-5b04bc4154.tar.gz` |
| R804 | 2,563 | `70d1a8ece6fc9420de47f27446c8533a87f135e2b09bfd974760532528469a88` | `/var/lib/sandy/cache/chutes-bench-evidence/d9d5f6c8-8066-47cd-9d62-4013d8ec68e4-af65700e99/oolong_agentic-6463308b27/804-dccb3c52e7/8dd1a7eb22b9-1a28915cd7.tar.gz` |
| B803 | 11,952 | `05b3b8ea34eaf209d8efe7e68eb2c0e57d5dd6669eb7b695615b9f520b9249f7` | `/var/lib/sandy/cache/chutes-bench-evidence/c4751f2c-ae5b-4f04-886e-a2fd09413c81-b73a28edac/oolong_agentic-6463308b27/803-9f006addc8/73b7f5939d58-da7e94f87a.tar.gz` |
| B804 | 12,484 | `6f3cad1e300a4076d7e2b4cb8c93e31e942b08654d7159a10afaacbb84b68364` | `/var/lib/sandy/cache/chutes-bench-evidence/c4751f2c-ae5b-4f04-886e-a2fd09413c81-b73a28edac/oolong_agentic-6463308b27/804-dccb3c52e7/e2827de9d9ae-a48787f419.tar.gz` |
| C803 | 314,190 | `20558b27f43e83eb150f59b5903514c74b33a139a1ff1ae5a0a511d2071ec8a6` | `/var/lib/sandy/cache/chutes-bench-evidence/e579f508-fde8-4037-af54-d71785ab00cc-003cc313cd/oolong_agentic-6463308b27/803-9f006addc8/bc72de7a65b6-98bf1059ba.tar.gz` |
| C804 | 10,100 | `a9cfdc025e7b6cefa5e5b0b430e2b1074071e3b56f2eeeb13fb95c5e5cff57d5` | `/var/lib/sandy/cache/chutes-bench-evidence/e579f508-fde8-4037-af54-d71785ab00cc-003cc313cd/oolong_agentic-6463308b27/804-dccb3c52e7/6844723af857-713f490bc2.tar.gz` |

## Verdict

**No — this sample does not show that our harness beats Codex.** It is not a
tie: only Codex item 803 produced a score, while the other seven cells were
excluded. There is no item with a non-excluded result from both Codex and
chutescoder, so the benchmark is inconclusive about the underlying score or
efficiency difference. Confidence is high that no win was demonstrated and
low about the true arm ordering. A larger sample was not launched because the
small sample showed neither an ahead/behind signal nor a valid tie; repeats or
more items at this point would be hunting through timeout exclusions.

