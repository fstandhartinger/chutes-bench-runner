# Chutes Bench Runner

Chutes Bench Runner is a web app + API for running reproducible benchmark suites against
models hosted on Chutes. It provides a modern UI, API-triggered runs, detailed per-item
results, and verifiable signed exports for sharing results.

## What this project includes

- **Frontend**: Next.js UI for selecting models, choosing subsets, tracking progress,
  and reviewing detailed results.
- **Backend API**: FastAPI service for run orchestration, exports, model sync, and auth.
- **Worker**: Background worker that executes benchmarks and streams progress events.
- **Sandy sandbox**: Isolated execution environment for code/CLI benchmarks.
- **Postgres database**: Stores runs, benchmarks, item results, and exports.

## Core features

- **One-click runs** in the UI with live status + SSE updates
- **Deterministic subset sampling** (1/5/10/25/50/100%) for reproducibility
- **API-triggered runs** using bearer API keys
- **Detailed per-item results** with prompts, responses, latency, and judge output
- **Token usage + cost breakdown** using Chutes pricing metadata
- **Signed exports** (CSV, PDF, and signed ZIP with JSON + signature)
- **Verification endpoint + UI** for signed ZIP files
- **Queue + ETA** estimation for running and queued jobs
- **Maintenance mode** to prevent new runs during deploys

## Benchmarks and evaluation methods

| Benchmark | Dataset / Source | Evaluation method |
|----------|------------------|-------------------|
| MMLU-Pro | TIGER-Lab/MMLU-Pro | Multiple-choice letter parsing |
| GPQA Diamond | Idavidrein/gpqa (gated) | Multiple-choice letter parsing |
| Humanity's Last Exam (HLE) | cais/hle (gated) | Normalized exact match |
| AIME 2025 | AI-MO/aimo-validation-aime | Numeric answer extraction |
| IFBench | google/IFEval | Official IFEval instruction checks |
| AA-LCR | ArtificialAnalysis/AA-LCR | LLM-judge consistency check |
| LiveCodeBench | livecodebench/code_generation | Run public+private IO tests |
| SciCode | SciCode1/SciCode + HDF5 tests | Official multi-step prompts + numeric tests |
| Terminal-Bench Hard | ia03/terminal-bench | Official docker harness |
| SWE-Bench Pro | ScaleAI/SWE-bench_Pro + SWE-bench_Pro-os | Official scripts + docker images |
| tau2-Bench Telecom | tau2-bench repo | Official tau2 simulation framework |

Notes:
- **Gated datasets** (HLE, GPQA) require HF access for the provided HF token.
- Some benchmarks require Sandy for sandboxed execution (LiveCodeBench, SciCode,
  Terminal-Bench, SWE-Bench Pro).

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

Production note: benchmark workers run on the dedicated Sandy server (`94.130.222.43`) for stability and cost; the Render worker service stays disabled.

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
```
Production uses the dedicated Sandy host `https://sandy.94.130.222.43.nip.io`.

## Configuration reference

Required:
- `DATABASE_URL`
- `CHUTES_API_KEY`

Optional but recommended:
- `HF_TOKEN` (for gated datasets)
- `SANDY_BASE_URL`, `SANDY_API_KEY` (sandboxed benchmarks)
- `BENCH_SIGNING_PRIVATE_KEY`, `BENCH_SIGNING_PUBLIC_KEY` (signed exports)
- `ADMIN_SECRET` (admin endpoints)

## Testing

```bash
cd backend
pytest -v

cd frontend
npm test
```

## Deployment

The `render.yaml` blueprint deploys:
- Backend API service
- Worker service
- Frontend service
- Postgres database

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
