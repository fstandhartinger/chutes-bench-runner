# Chutes Bench Runner

> **PRODUCTION SERVICE**: This is a revenue-generating product. ALL bench-runner components are consolidated on a **dedicated server** at `88.99.58.39` (bench\_runner\_sandy): Sandy sandbox service (port 7331, 256GB RAM, 12 cores, max 200 sandboxes), 36 autoscaled workers (`SANDY_BASE_URL=http://host.docker.internal:7331` inside worker containers), the autoscaler systemd service, and the queue health monitor (Telegram alerts). Do NOT deploy general Sandy changes to this server. new\_sandy and old\_sandy no longer run any bench-runner workers, autoscaler, or queue monitor.

Chutes Bench Runner is a web app + API for running reproducible benchmark suites against
models hosted on Chutes or OpenRouter. It provides a modern UI, API-triggered runs, detailed per-item
results, and verifiable signed exports for sharing results.

## What this project includes

- **Frontend**: Next.js UI for selecting models, choosing subsets, tracking progress,
  and reviewing detailed results.
- **Backend API**: FastAPI service for run orchestration, exports, model sync, and auth.
- **Worker**: Background worker that executes benchmarks and streams progress events.
- **Sandy sandbox**: Isolated execution environment for code/CLI benchmarks.
- **Postgres database**: Stores runs, benchmarks, item results, and exports.
  Production now uses self-hosted Postgres 16 on Hetzner instead of Neon.

## Core features

- **One-click runs** in the UI with live status + SSE updates
- **Deterministic subset sampling** (1/5/10/25/50/100%) for reproducibility
- **API-triggered runs** using bearer API keys
- **Detailed per-item results** with prompts, responses, latency, and judge output
- **Token usage + cost breakdown** using Chutes pricing metadata
- **Provider pre-flight** with exact usage validation before any benchmark item runs
- **Signed exports** (CSV, PDF, and signed ZIP with JSON + signature)
- **Verification endpoint + UI** for signed ZIP files
- **Queue + ETA** estimation for running and queued jobs
- **Maintenance mode** to prevent new runs during deploys

## Benchmarks and evaluation methods

| Benchmark | Dataset / Source | Evaluation method |
|----------|------------------|-------------------|
| MMLU-Pro | TIGER-Lab/MMLU-Pro | Multiple-choice letter parsing |
| GPQA Diamond | Idavidrein/gpqa (gated) | Multiple-choice letter parsing |
| Humanity's Last Exam (HLE) | cais/hle (gated) | LLM judge (official HLE judge prompt, multimodal) |
| AIME 2025 | AI-MO/aimo-validation-aime | Numeric answer extraction |
| IFBench | allenai/IFBench_test | Official AllenAI IFBench checks, scored as 5-repeat loose prompt accuracy |
| AA-LCR | ArtificialAnalysis/AA-LCR | LLM-judge consistency check |
| AA-Omniscience | ArtificialAnalysis/AA-Omniscience-Public | LLM judge using official rubric |
| GDPval-AA | openai/gdpval | LLM judge against reference docs |
| LiveCodeBench | livecodebench/code_generation | Run public+private IO tests |
| SciCode | SciCode1/SciCode + HDF5 tests | Official multi-step prompts + numeric tests |
| Terminal-Bench Hard | ia03/terminal-bench | Official docker harness |
| SWE-Bench Pro | ScaleAI/SWE-bench_Pro + SWE-bench_Pro-os | Official scripts + docker images |
| tau2-Bench Telecom | tau2-bench repo | Official tau2 simulation framework |
| CritPt | CritPt-Benchmark/CritPt | External CritPt evaluation server |
| S-NIAH | RULER-style synthetic | Single needle-in-haystack retrieval (exact match) |
| OOLONG | oolongbench/oolong-synth | Semantic aggregation (numeric: 0.75^diff, else exact match) |
| OOLONG-Pairs | oolongbench/oolong-real (dnd) | Pairwise aggregation (F1 scoring) |

### Long-context benchmarks (RLM paper)

The S-NIAH, OOLONG, and OOLONG-Pairs benchmarks are based on the [RLM paper](https://arxiv.org/html/2512.24601v1)
and evaluate long-context reasoning capabilities:

- **S-NIAH** (Needle-in-a-Haystack): Tests retrieval of specific information from long distractor text.
  Processing costs scale roughly constant with input length. Uses configurable context sizes from 8K to 256K tokens.

- **OOLONG**: Tests semantic classification and aggregation across dataset entries.
  Processing costs scale linearly with input length. Uses the [oolongbench/oolong-synth](https://huggingface.co/datasets/oolongbench/oolong-synth) dataset.

- **OOLONG-Pairs**: Extension requiring pairwise aggregation across entries.
  Processing costs scale quadratically with input length. Uses D&D transcripts from [oolongbench/oolong-real](https://huggingface.co/datasets/oolongbench/oolong-real).

Notes:
- **Gated datasets** (HLE, GPQA) require HF access for the provided HF token.
- Some benchmarks require Sandy for sandboxed execution (LiveCodeBench, SciCode,
  Terminal-Bench, SWE-Bench Pro). Terminal-Bench and SWE-Bench Pro run through
  Sandy's agent API to support agentic execution.

## API usage

### Start a run (frontend / IDP)
Users sign in with Chutes IDP in the UI. The worker uses the user's token if it has
`chutes:invoke` scope; otherwise it falls back to the system API key.

### Start a run (API key)
```bash
curl -X POST "https://chutes-bench-runner-api-v2.onrender.com/api/runs/api" \
  -H "Authorization: Bearer <CHUTES_API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "<bench-runner UUID or chute_id or model slug>",
    "subset_pct": 5,
    "selected_benchmarks": ["mmlu_pro"]
  }'
```

To offload an exploratory run to OpenRouter, configure `OPENROUTER_API_KEY`
in both the API and worker environments and select the OpenRouter model/provider:

```bash
curl -X POST "https://chutes-bench-runner-api-v2.onrender.com/api/runs/api" \
  -H "Authorization: Bearer <CHUTES_API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "deepseek/deepseek-v4-flash-0731",
    "provider": "openrouter",
    "subset_count": 1,
    "selected_benchmarks": ["terminal_bench_2_1"],
    "config": {"terminal_bench_2_1": {"agent": "codex"}}
  }'
```

The bearer key still authenticates the bench-runner request; inference for
this arm uses only the server-side `OPENROUTER_API_KEY`.

### Export results
```bash
# CSV
curl -o results.csv "https://chutes-bench-runner-api-v2.onrender.com/api/runs/<run-id>/export?format=csv"

# Signed ZIP
curl -o results.zip "https://chutes-bench-runner-api-v2.onrender.com/api/runs/<run-id>/export?format=zip"
```

### Verify a signed ZIP
```bash
curl -X POST "https://chutes-bench-runner-api-v2.onrender.com/api/exports/verify" \
  -F "file=@results.zip"
```

## Architecture overview

```
frontend (Next.js)
  -> backend API (FastAPI)
       -> Postgres
       -> worker (benchmark execution)
            -> Chutes LLM API
            -> Sandy sandbox (code/CLI benchmarks)
```

Production note: all bench-runner components (workers, autoscaler, queue monitor, Sandy sandbox service) are consolidated on the dedicated bench-runner-sandy server (88.99.58.39). Worker containers must use `SANDY_BASE_URL=http://host.docker.internal:7331` plus a Docker host-gateway mapping; `localhost` points at the container itself and breaks sandboxed benchmarks. The Render worker service stays disabled. new_sandy and own_postgres run Sandy for other apps only.

## Local development

### Prerequisites
- Python 3.11+
- Node.js 20+
- Postgres
- Chutes API key

### Setup
```bash
cp .env.example .env
# Fill in DATABASE_URL, CHUTES_API_KEY, and optional HF_TOKEN / SANDY settings

cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
alembic upgrade head

cd ../frontend
npm install
```

### Run services
```bash
# Terminal 1
cd backend
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2
cd backend
python -m app.worker.runner

# Terminal 3
cd frontend
npm run dev
```

### Docker compose (UI + API + worker + Postgres)
```bash
CHUTES_API_KEY=your_key docker-compose up
```

### Sandy sandbox (for code/CLI benchmarks)
Run Sandy separately and set:
```
SANDY_BASE_URL=https://<sandy-host>
SANDY_API_KEY=<sandy-key>
SANDY_DOCKER_UPSTREAM=docker-primary
```
Production uses a dedicated Sandy host (internal).

If you invoke Sandy’s `/agent/run` for agentic benchmarks, pass `apiBaseUrl` pointing at the Janus model router, keep `model=janus-router`, and set `rawPrompt: true` so Claude Code runs in research mode. Upload the agent pack and set `JANUS_SYSTEM_PROMPT_PATH` (or `systemPromptPath`) to `/workspace/agent-pack/prompts/system.md`.
If benchmarks emit files (logs, images), write them to `/workspace/artifacts` and cache sandbox artifact URLs server‑side before the sandbox exits.

## Configuration reference

Required:
- `DATABASE_URL`
- `CHUTES_API_KEY`

Production DSN example:

```bash
DATABASE_URL=postgresql://<user>:<password>@94.130.222.43:5432/chutes_bench_runner?sslmode=require
```

Optional but recommended:
- `OPENROUTER_API_KEY` (enables the OpenRouter model/provider arm)
- `HF_TOKEN` (for gated datasets)
- `SANDY_BASE_URL`, `SANDY_API_KEY` (sandboxed benchmarks)
- `SANDY_DOCKER_UPSTREAM` (route Docker-socket benchmarks to a Docker-backed Sandy upstream)
- `BENCH_DATA_DIR`, `HF_HOME`, `HF_DATASETS_CACHE`, `HF_HUB_CACHE` (shared cache paths)
- `DATASET_DISK_SAFETY_MARGIN_BYTES` (free space retained after declared dataset growth; defaults to 10 GiB)
- `BENCH_SIGNING_PRIVATE_KEY`, `BENCH_SIGNING_PUBLIC_KEY` (signed exports)
- `ADMIN_SECRET` (admin endpoints)
- `CRITPT_EVAL_URL`, `CRITPT_API_KEY` (CritPt evaluation service)
- `AA_OMNISCIENCE_JUDGE_MODEL`, `GDPVAL_JUDGE_MODEL` (LLM judges for AA-Omniscience/GDPval)

Before loading any dataset, the worker checks the filesystem backing each effective cache path.
Adapters with an unmeasured footprint are refused unless that individual run sets
`config.resource_preflight.allow_unknown_dataset_footprint=true`.

## Testing

```bash
cd backend
pytest -v

cd frontend
npm test
npm run test:watch
npm run test:e2e

# Sandy smoke test (requires a live Sandy host)
cd ..
SANDY_BASE_URL="https://<sandy-host>" SANDY_API_KEY="<sandy-key>" ./scripts/sandy_smoke_test.py
```

## Deployment

The `render.yaml` blueprint deploys:
- Backend API service
- Worker service
- Frontend service
- Postgres database

Operational note: production app traffic now targets the shared Hetzner Postgres host on `94.130.222.43:5432`. This service uses `asyncpg`, so it connects directly instead of going through PgBouncer transaction pooling. Neon remains available only as a temporary rollback path.

## Maintenance mode (deploy safety)

Before making any changes or deployments, **protect active runs**:

1. **Check for running benchmarks**  
   `GET /api/runs?status=running` must be empty.
2. **Enable maintenance mode**  
   Set `MAINTENANCE_MODE=true` (Render env var). This blocks new runs
   via the API and UI and shows a maintenance banner.
3. **Apply changes + deploy**  
   Make updates, commit, push, wait for Render deploys to finish.
4. **Disable maintenance mode**  
   Only after confirming no runs are active, set `MAINTENANCE_MODE=false`.

This prevents worker restarts during deploys from stalling in-flight runs.

See `INTERNAL.md` for deployment details, operational notes, and debugging tips.
