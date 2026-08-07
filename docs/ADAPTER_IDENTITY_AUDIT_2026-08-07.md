# Adapter identity audit — 2026-08-07

This audit used registry-created adapters and their real `preload()` methods
inside `chutes-bench-runner-worker:deepswe-evidence-384f763`, with the current
backend source mounted at `/src`. A live row count from a mutable source is
reported as an observation, not as proof of immutable benchmark membership.
No evaluation item, Sandy environment, or benchmark run was launched.

| Adapter | Claimed identity | What the real loader produced | Identity status |
| --- | --- | --- | --- |
| `terminal_bench` | Current stable Terminal-Bench = 2.1 | 89 unique IDs, exact ordered pinned manifest, empty diff | Exact; pinned archive, SHA-256, commit, and manifest |
| `terminal_bench_1` | Terminal-Bench 1.0 Core v0.1.1 | 80 unique IDs, exact ordered pinned manifest, empty diff | Exact; pinned archive, SHA-256, commit, and manifest |
| `terminal_bench_2` | Current verified 2.x = 2.1 | 89 unique IDs, exact ordered pinned manifest, empty diff | Exact; pinned archive, SHA-256, commit, and manifest |
| `terminal_bench_2_0` | Terminal-Bench 2.0 | 89 unique IDs, exact ordered pinned manifest, empty diff | Exact; pinned archive, SHA-256, commit, and manifest |
| `terminal_bench_2_1` | Terminal-Bench 2.1 | 89 unique IDs, exact ordered pinned manifest, empty diff | Exact; pinned archive, SHA-256, commit, and manifest |
| `terminal_bench_hard` | NVIDIA's 47-task curated leaderboard subset | 47 unique IDs, exact ordered JSON manifest, empty diff | Exact; registry + real preload from an empty cache |
| `aime_2025` | AIME 2025 I and II | 30 unique IDs (`2025-I-01..15`, `2025-II-01..15`), exact ordered manifest | Exact after this change; pinned HF revision and runtime 15+15 assertions |
| `deepswe` | Public DeepSWE v1.1 corpus | 113 unique task IDs, exact pinned manifest | Exact; pinned archive, SHA-256, commit, and manifest |
| `oolong` | OOLONG-synth test split | Declares 5,200; real range loader materialized IDs 803 and 5199 from revision `f0d59eaf…` | Partial; revision is pinned, full split was not materialized |
| `oolong_agentic` | Same OOLONG set, CLI-agent harness | Declares 5,200; real range loader materialized IDs 803 and 5199 from revision `f0d59eaf…` | Partial for the same reason as `oolong` |
| `mmlu_pro` | TIGER-Lab MMLU-Pro test | 12,032 rows | Observed real preload; source revision and membership are not pinned |
| `ifbench` | AllenAI IFBench test corpus | 300 rows | Observed real preload; source revision and membership are not pinned |
| `aa_lcr` | Artificial Analysis AA-LCR test | 100 rows | Observed real preload; source revision and membership are not pinned |
| `aa_omniscience` | AA-Omniscience Public | 600 unique `question_id` values | Observed real preload; source revision and membership are not pinned |
| `gdpval_aa` | GDPval-AA over OpenAI GDPval | 220 unique task UUIDs | Observed real preload; source revision and membership are not pinned |
| `scicode` | SciCode test | 65 unique problem IDs | Observed real preload; source revision and membership are not pinned |
| `critpt` | CritPt | 70 unique problem IDs | Observed real preload; source revision and membership are not pinned |
| `deepresearch_bench` | English DeepResearch-Bench/RACE subset, IDs 51–100 | 50 unique English query IDs | Observed real preload; URLs track GitHub `main` and are not pinned |
| `swe_bench_pro` | Scale SWE-Bench Pro test | 731 unique instance IDs | Observed real preload; source revision and membership are not pinned |
| `tau_bench_telecom` | τ²-Bench Telecom `base` task split | 114 unique task IDs | Observed real preload; loader downloads GitHub `main.zip`, so identity is mutable |
| `kimi_vendor_verifier` | Moonshot K2 Vendor Verifier tool-call set | 2,000 rows | Observed real preload; archive URL has no revision or checksum pin |
| `gpqa_diamond` | GPQA Diamond train configuration | Did not load: this HF account lacks dataset gate access | Not verified |
| `hle` | Humanity's Last Exam test | Did not load: this HF account lacks dataset gate access | Not verified |
| `livecodebench` | LiveCodeBench code-generation test | Two real preloads hung at `load_dataset()` for eight minutes; no row materialized | Not verified; the code's `400` fallback was not treated as proof |
| `oolong_pairs` | Custom pairwise extension over OOLONG-real D&D | Loader was not fully materialized; code declares 6,072 | Not verified; source is mutable |
| `s_niah` | Locally generated single-needle benchmark inspired by RULER | 60 deterministic synthetic items | Verified as local generation, not as a published fixed RULER task manifest |
| `affine_abd` | Affine ABD environment | Count path returned 23,300; no tasks materialized | Dynamic environment, not a pinned published task manifest |
| `affine_cde` | Affine CDE environment | Count path returned 8,579 from dataset metadata | Dynamic environment, not a pinned published task manifest |
| `affine_ded` | Affine DED environment | Count path returned 23,300; no tasks materialized | Dynamic environment, not a pinned published task manifest |
| `affine_game` | Affine GAME environment | Count path returned 7,300; no tasks materialized | Dynamic environment, not a pinned published task manifest |
| `affine_lgc` | Affine LGC environment | Count path returned 1,081,566 from dataset metadata | Dynamic environment, not a pinned published task manifest |
| `affine_lgc_v2` | Affine LGC-V2 environment | Count path returned 10,600; no tasks materialized | Dynamic environment, not a pinned published task manifest |
| `affine_print` | Affine PRINT environment | Count path returned 11,000; no tasks materialized | Dynamic environment, not a pinned published task manifest |
| `affine_swe_pro` | Affine SWE-PRO environment | Count path returned 731; no tasks materialized | Dynamic environment, not a pinned published task manifest |

Before the fix in this change, `aime_2025` performed a real preload of 90
rows from `AI-MO/aimo-validation-aime`: 30 each from 2022, 2023, and 2024,
and zero from 2025. The replacement source and proof are documented in
[`AIME_2025_IDENTITY.md`](AIME_2025_IDENTITY.md).
