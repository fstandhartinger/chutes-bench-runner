# Terminal-Bench 2.1 capability versus resource audit

Audit date: 2026-08-07.

This audit separates functional capability from verifier gates on performance,
cost, or resource use for the exact benchmark run by `terminal_bench`,
`terminal_bench_2`, and `terminal_bench_2_1`:

- repository: `harbor-framework/terminal-bench-2-1`
- commit: `5c8eadf1f393183288fa08b8f73ca9a469cc5e00`
- source archive SHA-256:
  `f9298006a7462a0b933c880aed8494c8c7b68ea4f97d792460f1bdffff2e6620`
- audited population: all 89 canonical task directories
- verifier inventory: all 89 top-level `tests/` directories, containing 280
  files recursively at the pinned commit

The result is **14 affected tasks out of 89 (15.7%)**: 4 are
`performance_gated`, 10 are `resource_gated`, and the remaining 75 are
`functional`.

## Classification rule

`performance_gated` means an upstream assertion compares measured host elapsed
time or a speed factor. `resource_gated` means an upstream assertion imposes a
deterministic compute-cost, modeled-latency, artifact-size, artifact-count, or
similar resource ceiling. If a task has both, host timing takes precedence and
the task is `performance_gated`.

Ordinary verifier liveness timeouts are not classified as performance scoring:
they bound a stuck test process but do not award credit for beating a measured
runtime. Domain-validity constraints are also functional. For example,
`cancel-async-tasks` asserts a runtime lower bound to prove that
`max_concurrent` was obeyed; slower execution is not penalized, so it remains
functional. This distinction avoids classifying nearly every executable test
as a speed benchmark merely because the verifier cannot wait forever.

## Definitive affected-task list

Paths and line numbers below are relative to each pinned task directory.
Assertion text is recorded in executable form in
`backend/app/benchmarks/adapters/terminal_bench_scoring.py`.

| Task | Class | Verifier evidence that gates the score |
| --- | --- | --- |
| `circuit-fibsqrt` | resource | `tests/test_outputs.py:23` — `assert line_count < 32000` |
| `gpt2-codegolf` | resource | `tests/test_outputs.py:15` — `assert gpt2_path.stat().st_size < 5000` |
| `large-scale-text-editing` | resource | `tests/test_outputs.py:215` — `assert total < 200`, capping the total submitted Vim macro keystrokes |
| `largest-eigenval` | performance | `tests/test_outputs.py:111` — `assert dt < ref_dt`; `dt` and `ref_dt` are medians of `time.perf_counter()` samples |
| `llm-inference-batching-scheduler` | resource | `tests/test_outputs.py:322,324,326,328` cap bucket 1 cost, padding ratio, modeled p95 latency, and sequential time cost; lines `331,333,335,337` impose the corresponding bucket 2 caps; line `366` asserts `len(unique_shapes_combined) <= 8` |
| `path-tracing` | resource | `tests/test_outputs.py:37` — `assert len(zlib.compress(open("/app/image.c").read().encode())) < 2100` |
| `path-tracing-reverse` | resource | `tests/test_outputs.py:39` — `assert len(zlib.compress(open("/app/mystery.c").read().encode())) < 2100` |
| `portfolio-optimization` | performance | `tests/test_outputs.py:129` — `assert speedup >= required_speedup`, where `required_speedup = 1.2` and both runtimes use `time.perf_counter()` |
| `query-optimize` | performance | `tests/test_outputs.py:142` — solution median elapsed time must be at most `1.05 *` the golden median; line `244` also asserts the SQL is at most 2,000 characters |
| `regex-chess` | resource | `tests/test_outputs.py:148` caps `re.json` below 10 MB; line `149` caps the decoded list below 100,000 pairs |
| `reshard-c4-data` | resource | `tests/test_outputs.py:128` caps each output file at 15 MiB; line `135` caps each directory at 30 entries |
| `train-fasttext` | resource | `tests/test_outputs.py:41` — `assert model_size < MAX_MODEL_SIZE`, where `MAX_MODEL_SIZE = 150 * 1024 * 1024` |
| `tune-mjcf` | performance | `tests/test_outputs.py:111` — `assert act_time_pctg <= pctg`, where `pctg = 0.6` and the values come from `time.perf_counter()` |
| `write-compressor` | resource | `tests/test_outputs.py:70` — `assert file_size <= max_size_bytes`, where `max_size_bytes = 2500` |

No pinned verifier test imposes a peak-memory/RSS ceiling. The
`custom-memory-heap-crash` task uses Valgrind to detect leaks, which is a
functional memory-correctness property rather than a quantitative memory
budget. No pinned verifier test imposes an LLM token budget, API-call budget,
or request quota.

## Run policy

The standard behavior is unchanged: all selected tasks run and all upstream
assertions remain intact. Every Terminal-Bench 2.1 item result now records
`item_metadata.scoring_class` plus the audit commit, reason, and exact verifier
evidence under `item_metadata.scoring_classification`.

Capability-only selection is an explicit run-config opt-in:

```json
{
  "terminal_bench_2_1": {
    "exclude_performance_and_resource_gated_tasks": true
  }
}
```

The family-level `terminal_bench` key and the concrete adapter key are both
accepted; the concrete key wins. The filter is applied after the ordinary
explicit-item or deterministic-subset selection, so the score report can state
exactly which selected tasks were removed. If every selected task is gated,
the run fails with the task names instead of producing an empty or misleading
score.

Run metrics include `terminal_bench_scoring_policy`, with the complete
classification counts, selected counts before and after filtering, exclusions
by class, every excluded item/task/reason, and a summary beginning
`NON-STANDARD CAPABILITY-ONLY SCORE`. This filtered score is not comparable to
published 89-task Terminal-Bench results. The option is rejected for other
Terminal-Bench releases because their verifier contents were not part of this
audit.

## Threshold relaxation decision

Threshold relaxation was investigated but is not implemented. The four
host-timed tests use different relative comparisons (strictly faster, 1.2x,
within 5%, and at most 60%), while the resource tasks use unrelated units and
objectives. There is no single defensible relaxation factor, and making any of
them effectively infinite would turn a published benchmark verifier into an
unversioned local fork.

Excluding whole audited tasks leaves every executed upstream verifier byte-for-
byte unchanged. This preserves a clean distinction:

- default mode produces the standard task selection and thresholds;
- capability-only mode produces an explicitly non-standard filtered score;
- no mode claims that a locally weakened verifier is a published
  Terminal-Bench score.
