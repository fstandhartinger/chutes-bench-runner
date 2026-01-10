# Gremium Benchmark Integration Concept

## Summary
Enable Chutes Bench Runner to execute benchmarks against the Gremium consensus endpoint (OpenAI/Anthropic compatible), alongside the existing direct-Chutes model runs. The goal is to treat Gremium as a first-class inference provider, capture reproducible metadata about its consensus configuration, and surface results in exports and the UI.

## Goals
- Run any benchmark suite against Gremium (OpenAI-compat or Anthropic-compat endpoints).
- Preserve reproducibility by storing Gremium version + participant lineup in run metadata.
- Keep Chutes-based runs unchanged and default.
- Provide a clean UI toggle: **Chutes** vs **Gremium**.
- Make signed exports include provider + Gremium metadata.

## Non-goals
- No changes to the benchmark evaluation logic itself.
- No refactor of the scoring or judge pipelines beyond routing inference.
- No immediate support for multi-provider mixing within a single run (one provider per run).

## Current State (Relevant)
- All inference goes through `app/services/chutes_client.py` → `run_inference()` → `settings.chutes_api_base_url` (`https://llm.chutes.ai/v1`).
- Model catalog is synced from Chutes and stored in `Model`.
- Runs reference a `model_slug`/`model_id` that must be a Chutes model.
- Judges and tool benchmarks call Chutes directly.

## Proposed Design

### 1) Provider Abstraction
Introduce a minimal inference interface used by the benchmark adapters and worker:

- `InferenceClient.run_inference(model_id, messages, **kwargs)`
- `InferenceClient.list_models()` (optional, for UI)
- `InferenceClient.get_pricing_map()` (optional, for cost)

Implement:
- **ChutesClient**: current implementation (no change in behavior).
- **GremiumClient**: new implementation that targets:
  - OpenAI: `https://chutes-model-gremium.onrender.com/openai/v1/chat/completions`
  - Anthropic: `https://chutes-model-gremium.onrender.com/anthropic/v1/messages`

Routing rule:
- Provider = `chutes` → existing `ChutesClient`
- Provider = `gremium-openai` → `GremiumClient` (OpenAI endpoint)
- Provider = `gremium-anthropic` → `GremiumClient` (Anthropic endpoint)

### 2) Configuration
Add settings to `app/core/config.py`:

- `gremium_api_base_url` (default: `https://chutes-model-gremium.onrender.com`)
- `gremium_provider_default` (default: `gremium-openai`)
- `gremium_api_key` (optional; default to `CHUTES_API_KEY`)
- `gremium_timeout_seconds` (optional override)

### 3) Data Model Extensions
Add a provider field to `Run` and `Model`:

- `Run.provider`: enum (`chutes`, `gremium-openai`, `gremium-anthropic`)
- `Model.provider`: enum to label catalog entries

Add a synthetic model entry for Gremium in the local DB:
- `slug`: `gremium-consensus` (OpenAI)
- `slug`: `gremium-consensus-anthropic` (Anthropic)
- `name`: `Gremium (consensus)`
- `provider`: `gremium-*`

This keeps the UI and API consistent with existing model selection flows.

### 4) Run Creation API
Extend `/api/runs/api` payload:

- `provider` (optional): `chutes | gremium-openai | gremium-anthropic`

If `provider` is set to gremium, the backend:
- Validates `model_id` == the gremium synthetic slug.
- Stores provider on the run record.

### 5) Worker Execution
In `app/worker/runner.py`:
- Create client based on `run.provider`.
- For gremium runs, `model_slug` is one of:
  - `gremium-consensus` (OpenAI)
  - `gremium-consensus-anthropic`

### 6) Metadata and Reproducibility
When starting a gremium run, capture and store:
- `gremium_version` (from `/health`)
- `gremium_participants` (from `/health` or a new `/metadata` endpoint)
- `gremium_base_url`

Persist this in `Run.metadata` and include in exports.

### 7) Cost + Token Usage
Gremium should return `usage` compatible with OpenAI/Anthropic:
- If available, use directly.
- If not available, compute approximate usage from response length.

Cost calculation options:
- **Option A**: Keep cost as `null` for gremium runs (avoid misleading numbers).
- **Option B**: Add Gremium-specific pricing config (flat cost per request).
- **Option C**: Extend Gremium to return aggregated per-participant usage so cost can be computed accurately.

### 8) UI Changes
- Add a provider toggle above model selection.
- When provider = gremium, show a single model option `Gremium (consensus)`.
- Show a badge in run history and detail view (e.g., `Provider: gremium-openai`).
- Expose Gremium metadata (participants + version) in run details.

### 9) Judge Models
Keep judge models (AA-Omniscience, GDPval, etc.) running against Chutes by default to avoid circular evaluation.
Optionally allow a second provider toggle for judge models later.

## Testing Plan
- Unit tests for provider selection (API + worker).
- Integration test using a mocked Gremium endpoint (responses for OpenAI + Anthropic).
- Regression tests to ensure Chutes provider remains unchanged.

## Rollout Plan
1. Add provider fields + config + synthetic model record.
2. Implement GremiumClient and provider routing.
3. Add UI toggle + metadata display.
4. Add export metadata.
5. Deploy behind feature flag `ENABLE_GREMIUM_PROVIDER`.

## Open Questions
- Should Gremium expose a richer `/metadata` endpoint with participant IDs + thinking flags?
- Do we want a dedicated gremium cost model or leave cost null?
- Should Anthropic and OpenAI gremium runs be separate providers or a single provider with a mode flag?
