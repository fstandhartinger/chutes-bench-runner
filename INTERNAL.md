# Internal Documentation - Chutes Bench Runner

> **For AI Agents**: This file contains implementation details, deployment specifics, and lessons learned. Also see [README.md](./README.md) for a high-level project overview.

## Deployment Architecture

### Hosting (Render.com)

| Service | Type | Plan | Purpose |
|---------|------|------|---------|
| `chutes-bench-runner-api-v2` | Web Service | Starter | FastAPI backend + API |
| `chutes-bench-runner-ui` | Web Service | Starter | Next.js frontend |

**Note**: Benchmark execution is now on the dedicated Sandy server (Hetzner). The Render worker has been removed; do not reintroduce it to avoid duplicate workers.

### Database (Hetzner self-hosted Postgres)

- **Project**: `chutes-bench-runner` (primary runtime database)
- **Database**: `chutes_bench_runner`
- **Host**: `94.130.222.43:5432` (`own_postgres`)
- **Important**: The backend uses `asyncpg`, so production connects directly to Postgres on `5432` instead of PgBouncer transaction pooling on `6432`. The connection string still uses `sslmode=require`, and `config.py` converts this automatically for asyncpg.

### MCP Tools Available

- **Render MCP**: `mcp_Render_MCP_*` - List services, deploys, logs, metrics
- **Hetzner Postgres MCP**: `mcp__hetzner_postgres__*` - Query the primary production database
- **Neon MCP**: `mcp_neon_*` - Rollback-only fallback while Neon remains online
- **Browser MCP**: `mcp_cursor-ide-browser_*` - Test frontend, take screenshots

## Environment Variables

### Backend (API & Worker)

```
DATABASE_URL=postgresql://bench_runner:<password>@94.130.222.43:5432/chutes_bench_runner?sslmode=require
CHUTES_API_KEY=<system API key>
OPENROUTER_API_KEY=<system OpenRouter API key; optional alternate provider>
CHUTES_CLIENT_ID=<IDP client ID>
CHUTES_CLIENT_SECRET=<IDP client secret>
CHUTES_IDP_URL=https://auth.chutes.ai
FRONTEND_URL=https://chutes-bench-runner-ui.onrender.com
ADMIN_SECRET=<secret for admin endpoints>
SANDY_BASE_URL=http://88.99.58.39:7331
SANDY_API_KEY=<sandy-api-key>
SANDY_DOCKER_UPSTREAM=docker-primary
BENCH_SIGNING_PRIVATE_KEY=<base64 or PEM Ed25519 private key>
BENCH_SIGNING_PUBLIC_KEY=<optional public key>
SKIP_MODEL_SYNC=true
CRITPT_EVAL_URL=https://artificialanalysis.ai/api/v2/critpt/evaluate
CRITPT_API_KEY=<optional CritPt API key>
AA_OMNISCIENCE_JUDGE_MODEL=<optional override>
GDPVAL_JUDGE_MODEL=<optional override>
```

**Full Render env vars**: Stored locally in `.env` (gitignored). This file contains the actual values for all Render environment variables including secrets.

**API Key Location**: System-wide `$CHUTES_API_KEY` environment variable. Use `echo $CHUTES_API_KEY` to access.

**Sandy host**: As of 2026-05-10, bench-runner workers run on `own_postgres` (94.130.222.43), co-located with the production Sandy stack on port 7331. From inside worker containers use `SANDY_BASE_URL=http://host.docker.internal:7331` (with the docker host-gateway mapping in `docker-compose.worker.yml`). The previous dedicated `bench_runner_sandy` server (88.99.58.39) is decommissioned and being cancelled at Hetzner.

If `/api/ops/sandy/resources` returns 502, the Sandy controller/worker stack likely crashed on the Sandy host:
```
cd /opt/sandy/deploy
docker compose --env-file .env up -d sandy-controller sandy-worker
```

### Shared Cache (Bench Data + HF)

Workers use the Sandy host cache to avoid repeated dataset downloads:

- `BENCH_DATA_DIR=/var/lib/sandy/cache/chutes-bench-data`
- `HF_HOME=/var/lib/sandy/cache/hf`
- `HF_DATASETS_CACHE=/var/lib/sandy/cache/hf/datasets`
- `HF_HUB_CACHE=/var/lib/sandy/cache/hf/hub`

The Sandy server enforces a 200GB cache budget (`SANDY_CACHE_MAX_BYTES`) for
explicit cache fetches. HF/Datasets caches share the same disk; monitor usage.

### Frontend

```
NEXT_PUBLIC_BACKEND_URL=https://chutes-bench-runner-api-v2.onrender.com
```

## Key Implementation Details

### Chutes IDP Authentication Flow

1. Frontend redirects to `/api/auth/login` → Backend generates PKCE + state
2. User redirected to `auth.chutes.ai` for login
3. Callback to backend → Token exchange → Session stored in DB
4. Frontend cookie `chutes_bench_runner_session` stores session ID
5. Frontend uses proxy routes (`/api/auth/status`, `/api/auth/logout`) to handle cross-origin cookies

**Important**: When user has `chutes:invoke` scope, benchmark inference uses their token via IDP's inference proxy endpoint, not the system API key.

### API-Triggered Runs (Bearer Key)

- `POST /api/runs/api` accepts `Authorization: Bearer <CHUTES_API_KEY>` to run benchmarks with the caller’s key.
- `BenchmarkRun.auth_mode` and `auth_api_key` determine which credentials the worker uses.

### Signed Result Exports

- `GET /api/runs/{id}/export?format=zip` returns a signed ZIP that includes `results.json`, `manifest.json`, `signature.txt`, and `public_key.txt`.
- `POST /api/exports/verify` validates the ZIP signature/hash; frontend has `/verify` for uploads.

### Model Sync from Chutes API

- Public endpoint: `https://api.chutes.ai/chutes/?include_public=true`
- **Must NOT send Authorization header** - causes 401 error
- Sync uses upsert (`ON CONFLICT DO UPDATE`) to handle duplicates
- Auto-syncs on backend startup

### Worker Architecture

- Polls database every 5 seconds for queued runs
- Uses `SELECT ... FOR UPDATE SKIP LOCKED` to claim runs (prevents race conditions)
- **Critical**: Cannot use eager loading (`joined`) with `FOR UPDATE` - must manually refresh relationships after claiming
- Health check server on port 10000 for Render's health checks
- Supports resume-on-restart: the worker skips completed items for in-progress benchmarks and requeues stale runs

### Database Schema Notes

- `BenchmarkItemResult.item_metadata` - renamed from `metadata` (reserved SQLAlchemy attribute)
- Models use `slug` as unique identifier, not Chutes ID
- Sessions store OAuth tokens with expiry for auto-refresh

## Lessons Learned / Common Pitfalls

### 1. asyncpg SSL Parameter
**Problem**: Postgres URLs often use `sslmode=require`, but asyncpg only accepts `ssl=require`
**Fix**: `config.py` replaces `sslmode=` with `ssl=` in the async database URL

### 2. CORS with Credentials
**Problem**: `allow_origins=["*"]` doesn't work with `allow_credentials=True`
**Fix**: Explicitly list allowed origins in `CORSMiddleware`

### 3. FOR UPDATE with JOINs
**Problem**: PostgreSQL can't use `FOR UPDATE` on nullable outer joins
**Fix**: Select base model first, then manually `db.refresh(run, attribute_names=["model"])`

### 4. Worker Out of Memory
**Problem**: Starter plan (512MB) insufficient for loading benchmark datasets
**Fix**: Use Standard plan (2GB) for worker service

### 12. Hetzner Worker Pool (Production)
Benchmark workers run on a dedicated Sandy host (internal) to avoid Render OOMs and reduce cost. Render worker should stay disabled.

**Why**: Hetzner has plenty of CPU/RAM and is cheaper than multiple Render instances.

**Requirements**:
- `DATABASE_URL` (Hetzner Postgres connection string with `sslmode=require`)
- `CHUTES_API_KEY`
- `OPENROUTER_API_KEY` (required on API and worker for OpenRouter runs)
- `CHUTES_CLIENT_ID`
- `CHUTES_CLIENT_SECRET` (needed to refresh IDP tokens)
- `HF_TOKEN` (for gated datasets like HLE/GPQA)
- `SANDY_BASE_URL` + `SANDY_API_KEY` (for code benchmarks)
- `SANDY_DOCKER_UPSTREAM` (route Docker-socket benchmarks to a Docker-backed Sandy upstream)
- `SANDY_VOLUME_ROOT` (host path for Sandy volumes, default `/var/lib/sandy/volumes`)
- Optional: `WORKER_MAX_CONCURRENT`, `WORKER_ITEM_CONCURRENCY`, `WORKER_STALE_RUN_MINUTES`

OpenRouter runs use `deepseek/deepseek-v4-flash-0731`. The worker executes a
one-token Chat Completions pre-flight before the first item and requires
provider-reported input/output counts. Terminal-Bench Codex-family agents use
a sandbox-local Responses API config; credentials stay in the launch
environment and are never written into retained config or rollout artifacts.

**Setup Steps (Hetzner)**:
1. Create a working directory:
   ```bash
   sudo mkdir -p /opt/chutes-bench-runner
   sudo chown $USER:$USER /opt/chutes-bench-runner
   ```
2. Clone repo and checkout `main`:
   ```bash
   git clone https://github.com/fstandhartinger/chutes-bench-runner /opt/chutes-bench-runner
   cd /opt/chutes-bench-runner
   git checkout main
   git pull
   ```
3. Create `/opt/chutes-bench-runner/.env.worker` with the required env vars (do not commit).
4. Create `/opt/chutes-bench-runner/docker-compose.worker.yml`:
   ```yaml
   services:
     worker:
       build:
         context: ./backend
         dockerfile: Dockerfile
       env_file:
         - .env.worker
       command: python -m app.worker.runner
       restart: unless-stopped
   ```
5. Build and start N workers:
   ```bash
   cd /opt/chutes-bench-runner
   docker-compose -f docker-compose.worker.yml up -d --build --scale worker=4
   ```
6. Updating workers after code changes:
   ```bash
   cd /opt/chutes-bench-runner
   git pull
   docker-compose -f docker-compose.worker.yml up -d --build --scale worker=4
   ```

**Notes**:
- Use `app.worker.runner` (no health server) to avoid port conflicts.
- On the dedicated bench-runner-sandy server (256 GB RAM, 12 cores), the autoscaler manages up to 36 workers. Monitor memory + load with `docker stats` before scaling beyond that.

### Bench-runner host (own_postgres, since 2026-05-10)

Bench-runner workers, autoscaler, and queue monitor run on `own_postgres` (94.130.222.43) co-located with the production Sandy stack and the shared Postgres host:
- **IP**: 94.130.222.43
- **SSH**: `ssh root@94.130.222.43` (via Unified Remote MCP `own_postgres`)
- **Specs**: Intel i7-6700, 64 GB RAM, 2x 250 GB SATA SSD, 4C/8T, Ubuntu 24.04 LTS
- **Sandy port**: 7331 (shared with the rest of the Sandy stack — workers point at `host.docker.internal:7331`)
- **MCP server_id**: `own_postgres`
- **Worker resource limits**: `WORKER_CONTAINER_MEM_LIMIT=8g`, `WORKER_CONTAINER_CPU_LIMIT=1.0` (right-sized for the 64 GB host shared with Postgres + Sandy + 12 production DBs)
- **Autoscaler limits**: `MIN_WORKERS=1`, `MAX_WORKERS=3`, `BASE_MAX_WORKERS=3` (`/etc/systemd/system/chutes-bench-runner-autoscaler.service`). The previous dropin at `…/override.conf` from the pre-incident era is moved to `…/override.conf.bak-20260508` — do not restore without re-sizing for the host.

The previous dedicated server `bench_runner_sandy` (88.99.58.39, Xeon E5-1650v3, 256 GB RAM, 2x4 TB HDD, ~€69/month) ran the workers from 2026-03-14 to 2026-05-10. It is now stopped (services disabled, Sandy still up as a rollback option) and being cancelled at Hetzner.

**Important — incident memory (2026-03-13)**: A previous co-location of bench-runner workers on own_postgres caused throughput degradation when bursty bench-runs competed with production Sandy workloads (OpenClaw user VMs, etc.). The current configuration mitigates that with hard worker caps (max 3) and per-worker mem_limit=8g. If cisterciansis triggers a large burst, watch `Resource guard` in autoscaler logs; the memory watermark guards should freeze scale-up before OOM, but other Sandy tenants may slow down.

### Autoscaler (own_postgres)

All bench-runner workers and the autoscaler run on `own_postgres`. Worker containers must use `SANDY_BASE_URL=http://host.docker.internal:7331` with a Docker host-gateway mapping; `localhost` targets the container and breaks Sandy access. The dedicated `bench_runner_sandy` and `new_sandy` profiles are no longer used.

The autoscaler scales from **running + queued** backlog (not queued-only), so it avoids
killing active workers while long runs are in flight.

**Install:**
```bash
sudo cp /opt/chutes-bench-runner/scripts/chutes-bench-runner-autoscaler.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now chutes-bench-runner-autoscaler
```

**Logs:**
```bash
tail -n 200 /var/log/chutes-bench-runner-autoscaler.log
```

**Config (env vars):**
- `BACKEND_URL` (default `https://chutes-bench-runner-api-v2.onrender.com`)
- `MIN_WORKERS` / `MAX_WORKERS` / `BASE_MAX_WORKERS` / `EXTRA_MAX_WORKERS`
- `WORKER_MAX_CONCURRENT` (must match `.env.worker`)
- `SCALE_INTERVAL_SECONDS`
- `MEMORY_HIGH_WATERMARK` (default `85`) – freeze scale-up above this %.
- `MEMORY_EMERGENCY_WATERMARK` (default `92`) – scale down when above this %.
- `MEMORY_SCALE_DOWN_STEP` (default `2`) – workers to drop per emergency tick.
- `DISK_CHECK_PATH` (default `/`) – filesystem used for disk pressure checks.
- `DISK_HIGH_WATERMARK` / `DISK_EMERGENCY_WATERMARK` – freeze scale-up / force scale-down on disk pressure.
- `CPU_HIGH_WATERMARK` – freeze scale-up when host 1-minute load is too high.
- `LOG_PATH`

All bench-runner components (workers, autoscaler, queue monitor) run on own_postgres (94.130.222.43) since 2026-05-10. The bench-runner-sandy profile is decommissioned; new_sandy is unused for bench-runner.

### Priority workers (internal API-key runs)

To fast-lane internal runs that were started with `CHUTES_API_KEY`, launch a
separate worker pool with filtering enabled:

1. Create `/opt/chutes-bench-runner/.env.worker.priority` (do not commit):
   ```bash
   # Same base env as .env.worker
   CHUTES_API_KEY=...
   DATABASE_URL=...
   SANDY_API_KEY=...
   SANDY_BASE_URL=...

   # Filter only internal runs
   WORKER_ONLY_AUTH_MODE=api_key
   WORKER_ONLY_API_KEY=<same CHUTES_API_KEY>
   ```
2. Start up to 4 priority workers:
   ```bash
   docker-compose -p chutes-bench-runner-priority -f docker-compose.worker.priority.yml \
     --env-file .env.worker.priority up -d --build --scale worker=4
   ```
3. Stop when no internal backlog remains:
   ```bash
   docker-compose -p chutes-bench-runner-priority -f docker-compose.worker.priority.yml \
     --env-file .env.worker.priority down --remove-orphans
   ```

### 5. Frontend lib/ Directory Ignored
**Problem**: Root `.gitignore` had `lib/` which ignored `frontend/lib/`
**Fix**: Added `!frontend/lib/` to `.gitignore` to explicitly include it

### 6. Model Response Parsing (MMLU-Pro)
**Problem**: Chain-of-thought models return verbose responses with `<think>` tokens, not just answer letters
**Fix**: Parse response to extract answer letter from patterns like "Answer: A" or "The answer is A"

### 7. f-string Backslash Syntax
**Problem**: Python < 3.12 doesn't allow backslashes in f-string expressions
**Fix**: Use string concatenation instead: `f"prefix" + f"{'\\n'.join(items)}"`

### 8. Render Auto-Deploy Timing
**Problem**: Old instances continue running during deploy, causing confusion with logs
**Fix**: Wait for deploy status `live` before testing; check instance IDs in logs

### 9. SSE Named Events vs onmessage
**Problem**: Backend sent SSE events with `event: type` field, but frontend used `onmessage` which only catches unnamed events
**Fix**: Remove `event:` field from SSE output - send only `id:` and `data:`, include `event_type` in the JSON payload. Use `addEventListener` only for special events like `done`.

### 10. Robust Payload Extraction (Reasoning Models)
**Problem**: Chain-of-thought models (like DeepSeek R1 or Qwen3) often ignore "Output ONLY X" instructions and provide a `<think>` block, even if told not to.
**Fix**: All adapters now use `extract_python_code` (which handles case-insensitive `<think>` removal and markdown block extraction) and have improved regex patterns to find the final answer/command/action even if embedded in prose.

### 11. Benchmark Specific Fixes
- **AIME 2025**: Changed to integer comparison (e.g., "24" == "024") and increased `max_tokens` to 8192.
- **Terminal-Bench**: Added few-shot examples and command-line heuristics to find the command if no markdown is used.
- **Tau-Bench**: Added few-shot examples and regex keyword matching for action names.
- **GPQA**: Added case-insensitive think tag handling and standalone letter matching (e.g., "C." or "C").
- **IFBench**: Uses AllenAI's official IFBench instruction registry and evaluation logic against `allenai/IFBench_test`.
- **Transparency**: Added `test_code` to `ItemResult` so users can see exactly what validation code was run in the sandbox.
- **Error Resilience**: Fixed `NoneType` attribute errors by ensuring `ChutesClient` always returns a string and adding null checks in adapters.

## Testing

### Backend Tests
```bash
cd backend
pytest -v                    # All tests
pytest tests/test_api.py     # API unit tests
pytest tests/test_integration.py  # Integration tests (requires deployed backend)
```

### Frontend Tests
```bash
cd frontend
npm test                     # Vitest unit tests
npx playwright test          # E2E tests
```

### Manual API Testing
```bash
# Check models
curl https://chutes-bench-runner-api-v2.onrender.com/api/models

# Check benchmarks
curl https://chutes-bench-runner-api-v2.onrender.com/api/benchmarks

# Create run (POST)
curl -X POST https://chutes-bench-runner-api-v2.onrender.com/api/runs \
  -H "Content-Type: application/json" \
  -d '{"model_id":"<uuid>","subset_pct":1,"benchmark_names":["mmlu_pro"]}'

# Create run (Bearer API key)
curl -X POST https://chutes-bench-runner-api-v2.onrender.com/api/runs/api \
  -H "Authorization: Bearer <CHUTES_API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{"model_id":"<uuid>","subset_pct":1,"benchmark_names":["mmlu_pro"]}'

# Export signed ZIP
curl -O "https://chutes-bench-runner-api-v2.onrender.com/api/runs/<run-id>/export?format=zip"

# Verify signed ZIP
curl -X POST https://chutes-bench-runner-api-v2.onrender.com/api/exports/verify \
  -F "file=@benchmark_results.zip"
```

## Benchmark Adapter Status

Each benchmark adapter in `backend/app/benchmarks/adapters/` uses official datasets or evaluation harnesses. Gated datasets require an `HF_TOKEN` with access.

| Benchmark | Dataset/Source | Notes |
|-----------|----------------|-------|
| `mmlu_pro` | TIGER-Lab/MMLU-Pro | Public dataset |
| `gpqa_diamond` | Idavidrein/gpqa (GATED) | Requires HF access |
| `aime_2025` | AI-MO/aimo-validation-aime (fallbacks: lighteval/MATH, hendrycks/competition_math) | Public fallbacks |
| `ifbench` | allenai/IFBench_test | Official AllenAI IFBench scoring (5-repeat loose prompt accuracy) |
| `hle` | cais/hle (GATED) | Requires HF access |
| `livecodebench` | livecodebench/code_generation | Runs public + private tests in Sandy |
| `scicode` | SciCode1/SciCode + Srimadh/Scicode-test-data-h5 | Official stepwise prompts + HDF5 tests |
| `aa_lcr` | ArtificialAnalysis/AA-LCR | Uses official document bundle + LLM judge |
| `swe_bench_pro` | ScaleAI/SWE-bench_Pro + SWE-bench_Pro-os scripts | Docker Hub images + official parsers |
| `tau_bench_telecom` | sierra-research/tau2-bench | Official tau2 simulation framework |
| `terminal_bench_hard` | ia03/terminal-bench | Docker-based harness per README |
| `aa_omniscience` | ArtificialAnalysis/AA-Omniscience-Public | LLM judge with official rubric |
| `gdpval_aa` | openai/gdpval | LLM judge vs reference docs |
| `critpt` | CritPt-Benchmark/CritPt | External evaluation server |

**Note**: To enable gated datasets, set `HF_TOKEN` in the worker environment with access to the datasets.

### Benchmark Scoring Notes

- **Code benchmarks** (livecodebench, scicode, aa_lcr, swe_bench_pro, terminal_bench_hard): Use the **Sandy Sandbox** on the Hetzner Server for execution.
- **IFBench**: Uses the official AllenAI IFBench checker and reports loose prompt accuracy.
- **Terminal-Bench**: Uses Sandy agent execution to drive the task container, then runs the official tests.
- **SWE-Bench Pro**: Runs a Sandy agent against a sandboxed repo checkout, derives a patch, then executes the official harness.

### Sandy Sandbox Security: Docker Socket Access

**Docker socket access is disabled by default** in Sandy sandboxes for security reasons. However, some benchmarks require Docker access:

| Benchmark | Requires Docker | Reason |
|-----------|-----------------|--------|
| `livecodebench` | No | Python code execution only |
| `scicode` | No | Python code execution only |
| `aa_lcr` | No | LLM judge evaluation |
| `swe_bench_pro` | **Yes** | Runs `docker pull` and `docker run` to execute test harness |
| `terminal_bench_hard` | **Yes** | Uses `docker-compose` and `docker build/run` for task environments |

**Implementation**: The `sandy_service.create_sandbox()` method accepts an `enable_docker_socket` parameter. Terminal-Bench and SWE-Bench adapters pass `enable_docker_socket=True` when creating sandboxes. All other benchmarks use the default (Docker socket disabled).

**Security note**: Docker socket access allows sandbox code to escape isolation. The Hetzner server should not store sensitive credentials in locations accessible via Docker volume mounts.

## File Reference

| File | Purpose |
|------|---------|
| `backend/app/core/config.py` | All settings, env var loading, SSL fix |
| `backend/app/services/chutes_client.py` | Chutes API + IDP inference |
| `backend/app/services/auth_service.py` | OAuth/PKCE flow, session management |
| `backend/app/services/sandy_service.py` | Sandy Sandbox integration |
| `backend/app/services/signed_export_service.py` | Signed ZIP export + verification |
| `backend/app/worker/runner.py` | Benchmark execution loop |
| `backend/app/benchmarks/adapters/` | Individual benchmark implementations |
| `frontend/contexts/auth-context.tsx` | Frontend auth state |
| `frontend/app/api/auth/*/route.ts` | Auth proxy routes for cookies |
| `frontend/app/api-docs/page.tsx` | API usage guide page |
| `frontend/app/verify/page.tsx` | Signed ZIP verification UI |
| `render.yaml` | IaC deployment definition |

## Debugging Checklist

1. **Models not loading**: Check if startup sync succeeded in backend logs
2. **Auth not working**: Verify cookie is set, check frontend proxy routes
3. **Runs staying queued**: Check worker logs, verify API key is correct
4. **0% benchmark scores**: Check model response format, review parsing logic
5. **Frontend not updating**: Verify deploy is `live`, not `update_in_progress`
6. **CORS errors**: Check allowed origins in backend `main.py`
