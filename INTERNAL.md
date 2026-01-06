# Internal Documentation - Chutes Bench Runner

> **For AI Agents**: This file contains implementation details, deployment specifics, and lessons learned. Also see [README.md](./README.md) for a high-level project overview.

## Deployment Architecture

### Hosting (Render.com)

| Service | Type | Plan | Purpose |
|---------|------|------|---------|
| `chutes-bench-runner-api-v2` | Web Service | Starter | FastAPI backend + API |
| `chutes-bench-runner-worker-v2` | Web Service | Standard (2GB RAM) | Benchmark execution worker |
| `chutes-bench-runner-ui` | Web Service | Starter | Next.js frontend |

**Note**: The worker uses "Standard" plan (2GB RAM) because benchmark execution (especially with `datasets` library) requires more memory than the Starter plan's 512MB.

### Database (Neon.tech)

- **Project**: `chutes-bench-runner` (`tiny-credit-33821742`)
- **Database**: `neondb`
- **Important**: Connection string uses `sslmode=require`, but asyncpg requires `ssl=require`. The backend's `config.py` converts this automatically.

### MCP Tools Available

- **Render MCP**: `mcp_Render_MCP_*` - List services, deploys, logs, metrics
- **Neon MCP**: `mcp_neon_*` - Query database, run migrations, manage projects
- **Browser MCP**: `mcp_cursor-ide-browser_*` - Test frontend, take screenshots

## Environment Variables

### Backend (API & Worker)

```
DATABASE_URL=postgresql://neondb_owner:<password>@ep-sweet-pond-aepk3yhf-pooler.c-2.us-east-2.aws.neon.tech/neondb?sslmode=require
CHUTES_API_KEY=<system API key>
CHUTES_CLIENT_ID=<IDP client ID>
CHUTES_CLIENT_SECRET=<IDP client secret>
CHUTES_IDP_URL=https://auth.chutes.ai
    FRONTEND_URL=https://chutes-bench-runner-ui.onrender.com
    ADMIN_SECRET=<secret for admin endpoints>
    SANDY_BASE_URL=https://sandy.94.130.222.43.nip.io
    SANDY_API_KEY=<sandy-api-key>
    BENCH_SIGNING_PRIVATE_KEY=<base64 or PEM Ed25519 private key>
    BENCH_SIGNING_PUBLIC_KEY=<optional public key>
    SKIP_MODEL_SYNC=true
    ```

**API Key Location**: System-wide `$CHUTES_API_KEY` environment variable. Use `echo $CHUTES_API_KEY` to access.

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
**Problem**: Neon URLs use `sslmode=require`, but asyncpg only accepts `ssl=require`
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

### 12. Hetzner Worker Pool (Preferred for Cost)
We can run additional benchmark workers on the Hetzner server (94.130.222.43) to avoid scaling Render. Use Docker to avoid Python 3.11 install conflicts.

**Why**: Hetzner has plenty of CPU/RAM and is cheaper than multiple Render instances.

**Requirements**:
- `DATABASE_URL` (Neon connection string with sslmode=require)
- `CHUTES_API_KEY`
- `CHUTES_CLIENT_ID`
- `CHUTES_CLIENT_SECRET` (needed to refresh IDP tokens)
- `HF_TOKEN` (for gated datasets like HLE/GPQA)
- `SANDY_BASE_URL` + `SANDY_API_KEY` (for code benchmarks)
- Optional: `WORKER_MAX_CONCURRENT`, `WORKER_ITEM_CONCURRENCY`, `WORKER_STALE_RUN_MINUTES`

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
   docker compose -f docker-compose.worker.yml up -d --build --scale worker=4
   ```
6. Updating workers after code changes:
   ```bash
   cd /opt/chutes-bench-runner
   git pull
   docker compose -f docker-compose.worker.yml up -d --build --scale worker=4
   ```

**Notes**:
- Use `app.worker.runner` (no health server) to avoid port conflicts.
- Safe to run 4–8 workers; monitor memory/CPU with `htop` or `docker stats`.
- Keep existing services intact (Sandy, TAO trader, dashboards, nginx).

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
- **IFEval**: Implemented custom internal checker for 10+ instruction types (punctuation, case, length, JSON, quotes, keywords).
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
| `ifbench` | google/IFEval | Official IFEval scoring |
| `hle` | cais/hle (GATED) | Requires HF access |
| `livecodebench` | livecodebench/code_generation | Runs public + private tests in Sandy |
| `scicode` | SciCode1/SciCode + Srimadh/Scicode-test-data-h5 | Official stepwise prompts + HDF5 tests |
| `aa_lcr` | ArtificialAnalysis/AA-LCR | Uses official document bundle + LLM judge |
| `swe_bench_pro` | ScaleAI/SWE-bench_Pro + SWE-bench_Pro-os scripts | Docker Hub images + official parsers |
| `tau_bench_telecom` | sierra-research/tau2-bench | Official tau2 simulation framework |
| `terminal_bench_hard` | ia03/terminal-bench | Docker-based harness per README |

**Note**: To enable gated datasets, set `HF_TOKEN` in the worker environment with access to the datasets.

### Benchmark Scoring Notes

- **Code benchmarks** (livecodebench, scicode, aa_lcr, swe_bench_pro, terminal_bench_hard): Use the **Sandy Sandbox** on the Hetzner Server for execution.
- **IFBench**: Uses the official IFEval checker.
- **Terminal-Bench**: Follows the official Docker harness (agent + test phases).

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
