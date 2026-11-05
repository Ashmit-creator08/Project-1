# Implementation Plan — AI-Powered Restaurant Recommendation System

## Purpose

Phase-wise plan to deliver the system described in [`problemStatement.md`](./problemStatement.md) and designed in [`architecture.md`](./architecture.md).

**Delivery model:** filter-then-LLM — hard filters on structured Zomato data, then LLM ranking and explanations only on a short candidate list.

**Default stack:** Python 3.11+, FastAPI, pandas + Hugging Face `datasets`, OpenAI-compatible LLM (Groq from Phase 4), Streamlit UI.

---

## Traceability

| Problem-statement workflow | Architecture layer | Primary phase |
|---|---|---|
| Data Ingestion | Data Layer | Phase 1 |
| User Input | Presentation | Phase 3 |
| Integration Layer | API / Orchestration + Filter | Phase 1–2 |
| Recommendation Engine | LLM Ranker | Phase 2 |
| Output Display | Presentation | Phase 3 |
| Hardening / extensions | Cross-cutting | Phase 4 |

---

## Phase Overview

| Phase | Name | Outcome | Depends on |
|---|---|---|---|
| **0** | Project bootstrap | Runnable repo skeleton + config | — |
| **1** | Data foundation | Clean dataset + working filters | Phase 0 |
| **2** | Recommendation path | End-to-end API with LLM + fallback | Phase 1 |
| **3** | UI & polish | Demo-ready app + docs | Phase 2 |
| **4** | Hardening + Groq | Groq LLM, cache, telemetry, rate limit, fuzzy matching, eval harness, prompt A/B | Phase 3 |

Phases 0–3 are required for v1. Phase 4 is optional.

---

## Phase 0 — Project Bootstrap

### Goal

Create a reproducible project skeleton so later phases plug into a known layout.

### Tasks

1. Create package layout from architecture §7:
   - `src/restaurant_recommender/` with `config`, `models`, `data`, `llm`, `services`, `api`
   - `ui/`, `data/processed/`, `tests/`, `docs/`
2. Add `requirements.txt` (or `pyproject.toml`) with baseline deps:
   - `fastapi`, `uvicorn`, `pydantic`, `pydantic-settings`
   - `pandas`, `datasets`, `pyarrow`
   - `httpx` / OpenAI SDK, `python-dotenv`
   - `streamlit` (UI phase), `pytest`
3. Add `.env.example` (`LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL`, `DATA_PATH`, `MAX_CANDIDATES`, `TOP_N_DEFAULT`)
4. Implement `config.py` with pydantic-settings loading env vars
5. Update `README.md` with project purpose and “how to run (coming)” stub
6. Confirm `python -c "import restaurant_recommender"` (or equivalent) works

### Deliverables

- [ ] Repo structure matching architecture
- [ ] Dependency file + `.env.example`
- [ ] Working settings module

### Exit criteria

- Fresh clone → create venv → install deps → import package succeeds
- No secrets committed

---

## Phase 1 — Data Foundation

### Goal

Load and preprocess the Hugging Face Zomato dataset; expose deterministic preference filters. Maps to **Data Ingestion** and the structured half of **Integration Layer**.

### Tasks

#### 1.1 Ingestion

1. Load [`ManikaSaini/zomato-restaurant-recommendation`](https://huggingface.co/datasets/ManikaSaini/zomato-restaurant-recommendation)
2. Inspect raw columns; map to canonical schema:

   | Canonical field | Purpose |
   |---|---|
   | `id` | Stable unique key |
   | `name` | Display + LLM context |
   | `location` / `city` / `locality` | Location filter |
   | `cuisine` | Cuisine filter |
   | `rating` | Min-rating filter + sort |
   | `cost_for_two` | Budget mapping |
   | `budget_band` | `low` \| `medium` \| `high` |
   | optional notes/highlights | Soft LLM context |

3. Clean nulls; normalize cuisine strings; coerce rating/cost types
4. Derive `budget_band` from cost (fixed thresholds or dataset percentiles — document choice in code comments)
5. Write cleaned Parquet to `data/processed/restaurants.parquet`
6. Add CLI or `python -m` entrypoint: `ingest` (download → clean → save)

#### 1.2 Store

1. Implement `store.py`: load Parquet into memory (DataFrame) at process start
2. Provide `get_all()` / lookup-by-id helpers for later fact merging

#### 1.3 Filter service

1. Implement filters (architecture §5.3), applied in order:
   - Location (case-insensitive contains / city match)
   - Cuisine (substring / token match)
   - Budget (`budget_band` or cost range)
   - Min rating (`rating >= min_rating`)
2. Cap candidates (`MAX_CANDIDATES`, default ≤ 30); when truncating, prefer higher ratings
3. Treat `additional_preferences` as **soft** (do not hard-filter in v1; pass through later to LLM)
4. Return structured candidate list + filter metadata (which filters applied, hit counts)

#### 1.4 Schemas (shared)

1. Define Pydantic models early for reuse in Phase 2–3:
   - `PreferenceRequest` (location, budget, cuisine, min_rating, additional_preferences, top_n)
   - `Restaurant` (canonical fields)
   - Placeholder `RecommendationItem` / `RecommendationResponse` stubs OK if finalized in Phase 2

#### 1.5 Tests

1. `test_filter.py`: each filter alone, combined filters, empty results, candidate cap
2. Ingest smoke test: schema columns present, non-zero row count after clean
3. Budget derivation unit tests for edge costs

### Deliverables

- [ ] `ingest.py`, `store.py`, `filter.py`
- [ ] Processed Parquet artifact (gitignored if large; document how to regenerate)
- [ ] Unit tests for filters + budget mapping
- [ ] Short note in README: how to run ingest

### Exit criteria

- Given sample prefs (e.g. Bangalore + Italian + medium + rating ≥ 3.5), filter returns a non-empty, correctly constrained list
- Zero-match prefs return empty list without errors
- Filter path completes in &lt; 100 ms on cached DataFrame (local check)

### Risks & mitigations

| Risk | Mitigation |
|---|---|
| Dataset columns differ from expectations | Inspect HF schema first; keep a single mapping table in `ingest.py` |
| Ambiguous location strings | Start with contains-match; refine in Phase 4 |
| Large download / OneDrive path issues | Cache Parquet locally; document offline re-run |

---

## Phase 2 — Recommendation Path

### Goal

Wire preferences → filter → LLM rank/explain → validated API response. Maps to **Integration Layer**, **Recommendation Engine**, and API half of the architecture.

### Tasks

#### 2.1 Finalize API contracts

1. Lock request/response schemas (architecture §5.2):
   - Request: location, budget, cuisine, min_rating, additional_preferences, top_n
   - Response: `summary`, `recommendations[]` (rank, name, cuisine, rating, estimated_cost, location, explanation, source_id), `meta`
2. Validate bounds: `top_n` max, string lengths, rating range

#### 2.2 LLM client

1. Implement provider-agnostic client (`LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`)
2. Prefer JSON mode / schema-constrained output where supported
3. Timeouts and clear error types for orchestration fallback
4. Low temperature (≈ 0.2–0.4)

#### 2.3 Prompt builder

1. System prompt rules:
   - Only use restaurants from the provided candidate list
   - Do not invent ratings, costs, or restaurants
   - Return JSON matching the schema
2. User prompt: preferences block + compact candidate table (`id`, name, cuisine, rating, cost, location, optional notes)
3. Ask for: ordered top-N, per-item explanation grounded in prefs, optional summary
4. Keep prompt size bounded via candidate cap from Phase 1

#### 2.4 Parser + fact merger

1. Parse LLM JSON; validate required fields
2. Drop hallucinated `source_id`s not in the candidate set
3. **Fact merger:** take name, cuisine, rating, cost, location from dataset; take rank + explanation (+ summary) from LLM
4. One repair retry on invalid JSON; then fail to fallback

#### 2.5 Orchestration service

Implement `services/recommend.py` flow:

1. Validate preferences  
2. Map budget → cost bounds / band  
3. Filter → candidates  
4. If empty → return empty response (skip LLM)  
5. Build prompt → call LLM → parse → merge  
6. On LLM failure → **rating-sorted fallback** with template explanations  
7. Attach `meta` (candidates_considered, filters_applied, fallback_used)

#### 2.6 FastAPI

1. `POST /api/v1/recommendations`
2. Health endpoint `GET /health` (optional but useful)
3. Load dataset store on startup
4. Consistent HTTP errors for validation failures (4xx)

#### 2.7 Tests

1. `test_prompt.py`: required fields present; candidate serialization
2. `test_parser.py`: valid JSON, missing fields, hallucinated IDs
3. `test_recommend_fallback.py`: mock LLM failure → rating fallback; empty filter → no LLM call
4. API tests: valid payload 200; invalid payload 422

### Deliverables

- [ ] `llm/client.py`, `prompts.py`, `parser.py`
- [ ] `services/recommend.py`
- [ ] FastAPI app with recommendations route
- [ ] Unit/integration tests with mocked LLM

### Exit criteria

- Curl/httpie against local API returns top-N with dataset-backed facts + AI explanations
- Empty filter path skips LLM and returns clear empty payload
- LLM outage still returns usable rating-based recommendations
- No restaurant appears that was not in the filtered candidate set

### Risks & mitigations

| Risk | Mitigation |
|---|---|
| LLM invents restaurants | Fact merger + ID allowlist |
| High latency / cost | Cap candidates; small top_n; optional cheaper model |
| Prompt fragility across providers | Keep schema strict; parser + one repair retry |

---

## Phase 3 — UI & Polish

### Goal

Ship a demo-ready experience for **User Input** and **Output Display**: preferences in, clear recommendation cards out.

### Tasks

#### 3.1 Streamlit UI (`ui/app.py`)

1. Preference form:
   - Location (text)
   - Budget (low / medium / high)
   - Cuisine (text or select from known values if available)
   - Minimum rating (slider/number)
   - Additional preferences (free text)
   - Optional top_n
2. Submit → call `POST /api/v1/recommendations` (or call orchestration service in-process if single-process demo is preferred — document choice)
3. Loading state while waiting on LLM
4. Results:
   - Optional summary at top
   - Cards/rows: name, cuisine, rating, estimated cost, AI explanation
5. Empty state when no matches
6. Error state for API/LLM failures (still show fallback results if API returns them)

#### 3.2 UX polish

1. Disable submit while request in flight
2. Show meta lightly (e.g. “18 candidates considered”) for demos — optional
3. Keep first screen simple: one form, one results section

#### 3.3 Documentation

1. README: setup, `.env`, ingest, run API, run UI, example request
2. Link docs: problem statement, architecture, this plan
3. Sample preference recipes for demos (e.g. Bangalore Italian medium; Delhi Chinese low)

#### 3.4 Smoke verification

1. Manual E2E: form → ranked list with explanations
2. Manual empty-result path
3. Manual bad API key → fallback or clear error (per implemented behavior)

### Deliverables

- [ ] Streamlit (or chosen) UI meeting problem-statement output fields
- [ ] README sufficient for a new developer to run the demo
- [ ] Demo script / example prefs

### Exit criteria

- Non-technical demo works end-to-end without touching code
- Output always includes: restaurant name, cuisine, rating, estimated cost, AI explanation
- Problem-statement objectives for v1 are demonstrably met

### Risks & mitigations

| Risk | Mitigation |
|---|---|
| CORS / two-process friction | Document ports; or in-process Streamlit→service call for demos |
| Slow first HF download | Pre-run ingest; commit instructions, not necessarily data |

---

## Phase 4 — Hardening + Groq LLM

### Goal

Improve reliability, speed, and evaluation without changing the core filter-then-LLM design, and make **Groq** the production LLM provider.

### 4.0 LLM provider: Groq

Groq serves an OpenAI-compatible API, so the Phase 2 client stays; Groq becomes the default provider preset.

| Item | Choice |
|---|---|
| Endpoint | `https://api.groq.com/openai/v1` (preset for `LLM_PROVIDER=groq`) |
| Ranker model | `openai/gpt-oss-120b` (Groq production model): strict `json_schema` output with candidate ids as an enum, `reasoning_effort=low` |
| Judge model (eval only) | `openai/gpt-oss-20b` (production), strict `json_schema` output, temperature 0 |
| Faster / cheaper ranker | `openai/gpt-oss-20b` |
| Why not Llama / Qwen | Groq's docs list `llama-3.3-70b-versatile` and `llama-3.1-8b-instant` as production, but the project key gets **404 (no access)** for both, so they are only `LLM_MODEL` overrides where enabled. `qwen/qwen3.8-27b` (a different family, which would be preferable as a judge) is preview-only and failed Groq's JSON validation in testing. Judge self-preference risk (same gpt-oss family) is mitigated by the heuristic judge cross-check. |
| Configuration | `LLM_PROVIDER=groq`, `GROQ_API_KEY` (or `LLM_API_KEY`); optional `LLM_MODEL`, `JUDGE_MODEL`, `LLM_BASE_URL` overrides |
| Output format | `LLM_RESPONSE_FORMAT=auto`: strict `json_schema` (candidate ids as an `enum`) on models that support it, otherwise `json_object`; degrade automatically if the provider rejects a format |

**Groq-specific handling**

- **Rate limits** (base developer plan ≈ 30 RPM, 1K RPD and 8K TPM on gpt-oss; check *Settings → Limits*). One v1 prompt over a 30-candidate shortlist is ≈ 4.8K prompt tokens, so the levers are the response cache, the compact v2 prompt (≈ 2.5K tokens), the `MAX_CANDIDATES` cap and a bounded `max_tokens`. The eval harness paces itself on `x-ratelimit-remaining-tokens`.
- **429:** honour `retry-after` once when it is ≤ `LLM_MAX_RETRY_WAIT_SECONDS` (default 2 s); otherwise fall back to rating order immediately instead of blocking the request.
- **`json_validate_failed` (400):** treated as unusable output; one repair turn using the provider's `failed_generation`, then fallback.
- **Telemetry:** token usage per request and `x-ratelimit-remaining-tokens/requests` headroom.
- **Secrets:** key only in `.env` / environment, held as `SecretStr`, never logged or written to metrics or scorecards.
- Only production models are used; preview models can be withdrawn at short notice.

### Tasks

| # | Task | Implementation | Edge cases |
|---|---|---|---|
| 1 | **Response caching** | `services/cache.py`: in-memory TTL + LRU keyed by SHA-256 of normalized prefs + data version + prompt variant + model. Fallback responses are not cached. `CACHE_TTL_SECONDS` (600, 0 = off), `CACHE_MAX_ENTRIES` (256), `meta.cache_hit`. | CF-06, CF-07 |
| 2 | **Locality matching** | `data/locations.py`: normalized keys (case, punctuation, spacing, `Rd`→road, `St`→street, drop "Layout", Bengaluru alias); substring match on locality / Zomato zone / city; otherwise fuzzy match (difflib ≥ 0.82, best match only) including block-less names ("Koramangla" → Koramangala). Cuisine gets synonyms (pizza → Italian) and fuzzy correction (Itallian → Italian). Corrections are reported in `meta.query_notes`. | FL-02, FL-04 |
| 3 | **Telemetry** | `services/telemetry.py`: request / empty / fallback / cache counters, fallback reasons by kind, token usage, Groq rate-limit headroom, p50/p95 latency for filter, LLM and total. `GET /metrics`. One structured log line per request (no free text, no secrets). | CF-04, CF-05 |
| 4 | **Eval set + harness** | `tests/eval/cases.json` gains Phase 4 cases (misspellings, synonyms). `python -m restaurant_recommender.eval.run` writes `evals/scorecard-*.md` per [`eval.md`](./eval.md), with a mock or live Groq LLM, `--inject llm_failures`, and a latency sample. Exits non-zero when a hard metric fails. | INV-01..05 |
| 5 | **Rate limiting** | `api/ratelimit.py`: per-client token bucket on `POST /api/v1/recommendations`, `RATE_LIMIT_PER_MINUTE` (30, 0 = off), 429 + `Retry-After`. It also protects the Groq quota. | CF-08 |
| 6 | **UI upgrade** | **Deferred.** Streamlit stays: there is no Node toolchain on the dev machine and Streamlit meets demo needs. Streamlit gains Phase 4 cues: correction notes, a cached badge, and provider status. | — |
| 7 | **Prompt A/B** | `llm/prompts/` variants `v1_baseline` and `v2_stricter_json` (compact pipe table with fewer tokens, explicit item count, per-explanation grounding rules), selected by `PROMPT_VARIANT`. `eval.run --prompt all` scores both with an LLM-as-judge (Groq `openai/gpt-oss-20b`; heuristic judge offline) and records the winner in the scorecard. | LM-15, LM-16 |

### Deliverables

- [x] Groq provider preset, retry/429 handling, structured outputs, usage + rate-limit capture
- [x] `services/cache.py`, `services/telemetry.py`, `api/ratelimit.py`, `GET /metrics`
- [x] `data/locations.py` + cuisine synonyms / fuzzy correction
- [x] `llm/prompts/` variants + `eval/` harness (checks, judge, scorecard CLI)
- [x] Tests for each item; live Groq eval scorecard + A/B winner recorded

### Results (live Groq, 2026-11-05)

Ranker `openai/gpt-oss-120b`, judge `openai/gpt-oss-20b`, eval cases tagged `happy_path,judge,phase4` (8 live cases per variant, to stay within the key's daily token quota; the full 17-case set passes offline with the mocked LLM).

| Metric | v1_baseline | v2_stricter_json |
|---|---|---|
| Hard metrics (constraint adherence, ID validity, fact fidelity, completeness, cardinality, duplicates) | 100% | 100% |
| Failure-injection suite | 7/7 | 7/7 |
| Explanation Acceptable% (LLM judge on EV-01, EV-02, EV-07) | 100% | 100% |
| Mean judge score (0–6) | 5.07 | **5.60** |
| Preference coverage / summary consistency | 100% / 100% | 100% / 100% |
| Ranker prompt tokens (8 cases + latency sample) | 46.8K | **24.7K (−47%)** |
| End-to-end latency (single warm sample) | 2.2 s | 2.5 s |

**Winner: `v2_stricter_json`**, now the default `PROMPT_VARIANT`. It scores higher on explanation quality (it acknowledges unconfirmed asks such as "quiet" instead of ignoring them) and halves token use, which is the binding constraint under Groq's 8K TPM. All exit criteria below are met.

### Exit criteria

- All Phase 1–3 tests and exit criteria still pass
- Hard metrics are 100% on the eval set with both the mocked LLM and live Groq
- A repeated identical query is served from cache with no LLM call; data, prompt or model changes produce a new cache key
- Groq 429, timeout or invalid JSON still returns rating-ranked results within the timeout; no secrets in logs or metrics
- A/B scorecard recorded with a winning prompt variant

### Risks & mitigations

| Risk | Mitigation |
|---|---|
| Groq TPM exhausted during eval runs | `--delay` between live calls, compact prompt, cache disabled only inside eval |
| Model deprecation | Models are configuration; production models only |
| Judge bias | Ranker and judge share the gpt-oss family on this account, so the heuristic judge is the cross-check (it agreed: 100% Acceptable for both variants), plus a human spot check of demo cases |
| Fuzzy matching over-matches | Exact match always wins; high cutoff; single best match; note shown to the user; Delhi/Atlantis eval cases must stay empty |

### Explicitly deferred (architecture §13)

- Embeddings / semantic “vibe” search
- User accounts / history
- Geospatial radius search
- Streaming tokens to UI
- Fine-tuned custom model
- React/Next.js UI (task 6)

---

## Cross-Phase Engineering Rules

Apply in every phase:

1. **Facts from dataset only** — LLM may rank and explain; it must not overwrite rating/cost/name
2. **Never send the full dataset to the LLM** — filtered shortlist only
3. **Secrets only via env** — never commit `.env`
4. **Tests before “done”** — filters/parser/orchestration covered; LLM mocked in CI
5. **Prefer small PRs / checkpoints** — finish a phase’s exit criteria before starting the next

---

## Suggested Checkpoint Demos

| After | Show |
|---|---|
| Phase 1 | Notebook or script: prefs in → filtered DataFrame out |
| Phase 2 | API response JSON with ranks + explanations |
| Phase 3 | Full UI walkthrough for stakeholders |
| Phase 4 | Faster repeat queries / eval scorecard |

---

## Definition of Done (v1)

v1 is complete when Phases **0–3** meet their exit criteria and the app can:

1. Ingest/preprocess the Hugging Face Zomato dataset  
2. Accept location, budget, cuisine, min rating, and additional preferences  
3. Filter candidates deterministically  
4. Use an LLM to rank and explain (with safe fallback)  
5. Display top recommendations with name, cuisine, rating, cost, and AI explanation  

---

## Release 1.0 — Deployment

Phases 0–4 are complete. The release packaging:

| Item | Implementation |
|---|---|
| Container image | `Dockerfile`: `python:3.12-slim`, pinned deps from `requirements.lock`, non-root user, dataset baked in at build (or ingested on first start with `INGEST_AT_BUILD=0`) |
| Roles | `docker/entrypoint.sh`: `api` (uvicorn, `--proxy-headers`, `PORT` / `WEB_CONCURRENCY` / `FORWARDED_ALLOW_IPS`), `ui` (Streamlit), `ingest`, `eval` |
| Local / single-host deploy | `docker-compose.yml`: API (holds the Groq key from `.env`) + UI (calls the API over the internal network), both with healthchecks and restart policies |
| Production settings | `LOG_FORMAT=json`, `LOG_LEVEL`, opt-in `CORS_ALLOW_ORIGINS`, `METRICS_TOKEN` for `/metrics`, `X-Request-ID` echo, `.streamlit/config.toml` (headless, no usage stats, minimal toolbar) |
| CI | `.github/workflows/ci.yml`: pinned install + `pytest`, then a Docker build |
| Scaling note | Cache, rate limiter and metrics are per process. Run one API replica, or move them to Redis / a gateway before scaling out. |

## Next Steps

- Deploy the image to the target platform (any container host that sets `PORT`) and set `GROQ_API_KEY` there as a secret.
- Optional follow-ups: Redis-backed cache and rate limiter for multiple replicas; React/Next.js UI (task 6); a model-level A/B once more Groq model families are enabled on the account.
