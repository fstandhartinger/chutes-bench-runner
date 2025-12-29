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
    SANDY_BASE_URL=https://sandy.65.109.49.103.nip.io
    SANDY_API_KEY=4e253b301b6934ec824c24565270713007dde805c937599449dc35ee078ca9d0
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
```

## Benchmark Adapter Status

Each benchmark adapter in `backend/app/benchmarks/adapters/` tries to load data from HuggingFace datasets. If unavailable, they fall back to placeholder data.

| Benchmark | Dataset Source | Fallback |
|-----------|---------------|----------|
| `mmlu_pro` | TIGER-Lab/MMLU-Pro | None needed (public) |
| `gpqa_diamond` | Idavidrein/gpqa (GATED) | 3 placeholder science questions |
| `aime_2025` | AI-MO/aimo-validation-aime, lighteval/MATH, hendrycks/competition_math | Multiple fallbacks |
| `ifbench` | google/IFEval | None (public dataset) |
| `hle` | cais/hle | 3 placeholder expert questions |
| `livecodebench` | livecodebench/code_generation_lite, codeparrot/apps, openai_humaneval | 3 placeholder coding problems |
| `scicode` | SciCode-Bench/SciCode, bigcode/humanevalpack | 3 placeholder sci-computing problems |
| `aa_lcr` | deepmind/code_contests, codeparrot/apps | 3 placeholder code reasoning |
| `swe_bench_pro` | princeton-nlp/SWE-bench_Lite, princeton-nlp/SWE-bench | 2 placeholder issues |
| `tau_bench_telecom` | None (hardcoded) | 3 telecom scenarios |
| `terminal_bench_hard` | None (hardcoded) | 5 shell command tasks |

**Note**: To enable gated datasets, set `HF_TOKEN` environment variable in the worker with a HuggingFace access token that has been granted access to the relevant datasets.

### Benchmark Scoring Notes

- **Code benchmarks** (livecodebench, scicode, aa_lcr, swe_bench_pro): Use the **Sandy Sandbox** on the Hetzner Server for code execution.
- **IFBench**: Uses basic heuristic (response length > 20 chars) - full IFEval checker needed for accurate scoring.
- **Terminal-Bench**: Executes commands in the Sandy Sandbox and checks for successful exit codes.

## File Reference

| File | Purpose |
|------|---------|
| `backend/app/core/config.py` | All settings, env var loading, SSL fix |
| `backend/app/services/chutes_client.py` | Chutes API + IDP inference |
| `backend/app/services/auth_service.py` | OAuth/PKCE flow, session management |
| `backend/app/services/sandy_service.py` | Sandy Sandbox integration |
| `backend/app/worker/runner.py` | Benchmark execution loop |
| `backend/app/benchmarks/adapters/` | Individual benchmark implementations |
| `frontend/contexts/auth-context.tsx` | Frontend auth state |
| `frontend/app/api/auth/*/route.ts` | Auth proxy routes for cookies |
| `render.yaml` | IaC deployment definition |

## Debugging Checklist

1. **Models not loading**: Check if startup sync succeeded in backend logs
2. **Auth not working**: Verify cookie is set, check frontend proxy routes
3. **Runs staying queued**: Check worker logs, verify API key is correct
4. **0% benchmark scores**: Check model response format, review parsing logic
5. **Frontend not updating**: Verify deploy is `live`, not `update_in_progress`
6. **CORS errors**: Check allowed origins in backend `main.py`

