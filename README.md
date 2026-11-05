# AI-Powered Restaurant Recommendation System

Zomato-inspired service that filters a real restaurant dataset by user preferences, then uses an LLM (**Groq**) to rank candidates and explain recommendations.

**Design:** filter first, then reason. Hard filters (location, budget, cuisine, rating) run on structured data; the LLM only ranks and explains a short candidate list. Every fact shown to the user (name, cuisine, rating, cost, location) comes from the dataset, never from the LLM.

## Docs

- [Problem statement](Docs/problemStatement.md)
- [Architecture](Docs/architecture.md)
- [Implementation plan](Docs/implementation-plan.md) (Phases 0–4)
- [Edge cases](Docs/edge-case.md)
- [Eval plan](Docs/eval.md)

## Status

| Phase | What | Where |
|---|---|---|
| 1 | Ingest Hugging Face dataset → Parquet, store, filters | `src/restaurant_recommender/data/` |
| 2 | LLM client, prompts, parser, orchestration with rating fallback, REST API | `llm/`, `services/recommend.py`, `api/main.py` |
| 3 | Streamlit UI | `ui/app.py` |
| 4 | Groq provider, response cache, telemetry (`/metrics`), rate limiting, fuzzy locality/cuisine matching, eval harness + scorecard, prompt A/B (winner: `v2_stricter_json`) | `llm/client.py`, `services/cache.py`, `services/telemetry.py`, `api/ratelimit.py`, `data/locations.py`, `eval/`, `llm/prompts/` |
| 1.0 | Deployment: Docker image, compose, CI, production settings | `Dockerfile`, `docker/`, `docker-compose.yml`, `.github/workflows/ci.yml` |

## Quick start

```bash
# 1. Virtualenv (Python 3.11+) and dependencies
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # Windows PowerShell (macOS/Linux: source .venv/bin/activate)
pip install -e ".[dev]"

# 2. Configuration (never commit .env)
copy .env.example .env                # macOS/Linux: cp .env.example .env
# Set GROQ_API_KEY in .env (https://console.groq.com/keys).
# Without a key the app still works in rating-fallback mode.

# 3. Build the cleaned dataset (one-off, ~2-3 minutes)
python -m restaurant_recommender.data.ingest

# 4a. Demo UI (single process: calls the service in-process)
streamlit run ui/app.py               # http://localhost:8501

# 4b. REST API
uvicorn restaurant_recommender.api.main:app --reload --port 8000   # docs at http://localhost:8000/docs
```

To point the UI at the API instead of running the service in-process, set `RECOMMENDER_API_URL=http://localhost:8000` before `streamlit run`.

## Deployment

### Docker Compose (single host)

```bash
copy .env.example .env        # set GROQ_API_KEY
docker compose up --build -d
```

- API at <http://localhost:8000> (`/docs`, `/health`, `/metrics`); UI at <http://localhost:8501>.
- Only the API container receives `.env` (the Groq key). The UI calls the API at `http://api:8000`.
- The image bakes in the cleaned dataset at build time (about 2–3 minutes, reading from Hugging Face). If `data/processed/restaurants.parquet` already exists locally it is copied instead.
- Both containers have healthchecks and `restart: unless-stopped`; the UI starts once the API is healthy.

### Any container platform

Build once and run the same image in two roles:

```bash
docker build -t restaurant-recommender:1.0.0 .
docker run -d -p 8000:8000 --env-file .env restaurant-recommender:1.0.0 api
docker run -d -p 8501:8501 -e RECOMMENDER_API_URL=https://<your-api-host> restaurant-recommender:1.0.0 ui
```

| Variable | Use |
|---|---|
| `PORT` | Port to listen on (platforms such as Render, Railway, Cloud Run and Azure Container Apps set this) |
| `GROQ_API_KEY` | Set as a platform **secret** on the API service only |
| `WEB_CONCURRENCY` | Uvicorn workers (default 1; cache, rate limiter and metrics are per process) |
| `FORWARDED_ALLOW_IPS` | Proxy IPs to trust for `X-Forwarded-For`, so rate limits key on the real client (`*` behind a managed load balancer) |
| `LOG_FORMAT` / `LOG_LEVEL` | `json` for log aggregators; `INFO` default |
| `CORS_ALLOW_ORIGINS` | Comma-separated origins if a browser app calls the API directly |
| `METRICS_TOKEN` | Protects `GET /metrics` with `Authorization: Bearer <token>` |
| `INGEST_AT_BUILD` | Build arg; `0` skips baking the dataset (it is ingested on first start instead) |

Other roles: `docker run ... restaurant-recommender:1.0.0 ingest` rebuilds the dataset, and `... eval --llm mock` writes a scorecard.

### Production checklist

- [ ] `GROQ_API_KEY` stored as a secret, never in the image (`.env` is excluded by `.dockerignore`)
- [ ] `/health` wired to the platform's health check
- [ ] `METRICS_TOKEN` set if the API is public
- [ ] `FORWARDED_ALLOW_IPS` set to your proxy so per-client rate limits work
- [ ] One API replica, or Redis / gateway rate limiting before scaling out
- [ ] Groq plan limits checked (base tier: 8K tokens/min on gpt-oss). The cache and the v2 prompt keep usage low.

### Example request

```bash
curl -X POST http://localhost:8000/api/v1/recommendations \
  -H "Content-Type: application/json" \
  -d '{"location": "Indiranagar", "budget": "medium", "cuisine": "Italian", "min_rating": 3.5, "additional_preferences": "family-friendly", "top_n": 3}'
```

PowerShell:

```powershell
$body = @{ location = "Indiranagar"; budget = "medium"; cuisine = "Italian"; min_rating = 3.5; additional_preferences = "family-friendly"; top_n = 3 } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://localhost:8000/api/v1/recommendations -Body $body -ContentType "application/json"
```

Response (abridged):

```json
{
  "summary": "Bologna, Glen's Bakehouse, and Onesta are top Italian spots in Indiranagar within the medium budget.",
  "recommendations": [
    {"rank": 1, "name": "Bologna", "cuisine": "Italian", "rating": 4.5, "estimated_cost": "₹1,000 for two",
     "location": "Indiranagar, Bangalore", "explanation": "Highest rating (4.5) and fits Italian, Indiranagar, ₹1000 budget; family-friendly not confirmed by the data.",
     "explanation_source": "llm", "source_id": "rest_9b57406f69ff"}
  ],
  "meta": {"candidates_considered": 28, "total_matches": 28, "filters_applied": ["location", "cuisine", "budget", "min_rating"],
           "llm_used": true, "fallback_used": false, "model": "openai/gpt-oss-120b", "prompt_variant": "v2_stricter_json",
           "cache_hit": false, "query_notes": [], "usage": {"prompt_tokens": 2529, "completion_tokens": 376},
           "timings_ms": {"filter": 50.1, "llm": 1505.6, "total": 1658.7}}
}
```

### Endpoints

| Endpoint | Purpose |
|---|---|
| `POST /api/v1/recommendations` | Recommendations. Rate limited per client (429 + `Retry-After`). |
| `GET /api/v1/options` | Known locations, cuisines and budget bands (for pickers). |
| `GET /health` | 503 until the dataset is loaded. Reports data version, provider, model, prompt variant, cache and rate-limit settings. |
| `GET /metrics` | Requests, cache hit rate, fallback rate and reasons, token usage, Groq rate-limit headroom, p50/p95 latency for filter, LLM and total. No secrets or user text. |

## Configuration (`.env`)

| Variable | Default | Notes |
|---|---|---|
| `LLM_PROVIDER` | `groq` | `groq`, `openai` or `custom` (any OpenAI-compatible endpoint, e.g. Ollama) |
| `GROQ_API_KEY` / `LLM_API_KEY` | empty | Empty = fallback-only mode. Not needed for a local `custom` endpoint. |
| `LLM_MODEL` | `openai/gpt-oss-120b` | Ranker. `openai/gpt-oss-20b` is faster and cheaper. |
| `JUDGE_MODEL` | `openai/gpt-oss-20b` | Eval-only LLM-as-judge |
| `LLM_BASE_URL` | provider preset | `https://api.groq.com/openai/v1` for Groq |
| `LLM_RESPONSE_FORMAT` | `auto` | Strict `json_schema` where supported (gpt-oss), else `json_object`; degrades automatically |
| `LLM_REASONING_EFFORT` | `low` | gpt-oss only; keeps hidden reasoning tokens low |
| `LLM_TIMEOUT_SECONDS` / `LLM_MAX_TOKENS` | `20` / `1200` | A timeout leads to rating fallback |
| `LLM_MAX_RETRY_WAIT_SECONDS` | `2` | A 429 is retried once only when its `retry-after` is this short; otherwise immediate fallback |
| `PROMPT_VARIANT` | `v2_stricter_json` | Winner of the live Groq A/B (judge 5.60 vs 5.07, −47% prompt tokens); `v1_baseline` is available |
| `LOG_LEVEL` / `LOG_FORMAT` | `INFO` / `text` | `json` for log aggregators |
| `CORS_ALLOW_ORIGINS` | empty | Comma-separated origins; empty = CORS off |
| `METRICS_TOKEN` | empty | When set, `/metrics` requires `Authorization: Bearer <token>` |
| `DATA_PATH` | `data/processed/restaurants.parquet` | Relative paths resolve from the repo root |
| `MAX_CANDIDATES` / `TOP_N_DEFAULT` | `30` / `5` | `top_n` is capped at 20 |
| `CACHE_TTL_SECONDS` / `CACHE_MAX_ENTRIES` | `600` / `256` | Response cache; `0` disables it |
| `RATE_LIMIT_PER_MINUTE` | `30` | Per client; `0` disables it |
| `RECOMMENDER_API_URL` | empty | UI only: call the API instead of the in-process service |

**Groq limits:** on the base developer plan, gpt-oss allows about 8K tokens/minute and 1K requests/day. A v1 prompt over 30 candidates is about 4.8K tokens and v2 about 2.5K. The cache, the compact v2 prompt and a lower `MAX_CANDIDATES` all stretch the quota.

## How it works

1. **Ingest:** reads the needed columns of [`ManikaSaini/zomato-restaurant-recommendation`](https://huggingface.co/datasets/ManikaSaini/zomato-restaurant-recommendation) from Hugging Face's Parquet export, parses ratings and costs, dedupes the 51,717 listings into **12,494 unique restaurants**, and writes Parquet atomically.
2. **Filter** (all optional except location):
   - *Location*: normalized matching on locality, Zomato zone and city. `HSR Layout` = `HSR`, `Indira Nagar` = `Indiranagar`, `M.G. Rd` = `MG Road`, `Bengaluru` = `Bangalore`. Misspellings fall back to the closest locality (`Koramangla` → Koramangala), and the user is told.
   - *Cuisine*: whole-word match (`Indian` matches `North Indian`). Dish words expand (`pizza` → Pizza or Italian) and misspellings are corrected (`Itallian` → Italian).
   - *Budget*: low ≤ ₹400 < medium ≤ ₹1,000 < high. *Min rating* is inclusive.
   - The shortlist is sorted by rating, votes, name and capped at `MAX_CANDIDATES`.
3. **Cache:** identical (normalized) preferences within the TTL are served from memory. The key includes the data version, prompt variant and model. Fallback responses are never cached.
4. **Rank and explain (Groq):** the shortlist goes to the LLM with a strict JSON schema whose `id` is an enum of shortlisted ids, so the model cannot return an unknown restaurant. `additional_preferences` is a soft signal only.
5. **Merge and fall back:** unknown or duplicate ids are dropped and gaps backfilled in rating order; facts always come from the dataset. Invalid JSON (including Groq `json_validate_failed`) gets one repair turn. Timeouts, 429s and 5xx errors fall back to rating order with template explanations.

**Dataset scope:** Bangalore only (93 localities). Other cities return an empty result with a hint; fuzzy matching never maps them onto Bangalore localities.

### Demo recipes

| Preferences | Shows |
|---|---|
| Indiranagar · Italian · medium · ≥ 3.5 · "family-friendly" | Happy path |
| Koramangla · North Indian · ≥ 4.0 | Locality spelling correction |
| HSR Layout · pizza · ≥ 3.5 | Layout suffix + dish synonym |
| Indiranagar · Mexican · medium · ≥ 4.0, 5 results | Only 2 matches, no padding |
| Same query twice | Second answer is served from cache (`⚡ cached`) |
| Atlantis (or Delhi) · Italian | Empty state, no LLM call |

## Tests and eval

```bash
pytest -q                                                         # unit + contract + offline eval
python -m restaurant_recommender.eval.run --llm mock              # offline scorecard (hard metrics)
python -m restaurant_recommender.eval.run                         # live Groq + LLM judge
python -m restaurant_recommender.eval.run --prompt all            # prompt A/B, winner recorded
python -m restaurant_recommender.eval.run --inject llm_failures   # failure-injection suite only
```

The harness writes `evals/scorecard-latest.md` (gitignored) in the [eval.md](Docs/eval.md) template. It exits 1 if any hard metric or invariant fails. Live runs pace themselves on Groq's `x-ratelimit-remaining-tokens`; use `--tags happy_path,judge,phase4` to spend fewer tokens.

## Project layout

```
src/restaurant_recommender/
  config.py                 # env-backed settings + provider presets (Groq default)
  models/schemas.py         # PreferenceRequest, Restaurant, RecommendationResponse
  data/ingest.py            # HF download + clean + atomic Parquet write
  data/store.py             # in-memory store + location index
  data/locations.py         # location keys + fuzzy resolution
  data/filter.py            # hard filters, cuisine synonyms/corrections
  data/budget.py            # budget bands
  llm/client.py             # OpenAI-compatible client: Groq preset, structured output, retries, usage
  llm/prompts/              # v1_baseline, v2_stricter_json, strict ranking schema
  llm/parser.py             # JSON extraction + id allowlist
  services/recommend.py     # orchestration + cache + fact merge + fallback
  services/cache.py         # TTL/LRU response cache
  services/telemetry.py     # counters, rates, latency percentiles
  services/explain.py       # template explanations, summaries, formatting
  api/main.py               # FastAPI app
  api/ratelimit.py          # per-client token bucket
  eval/                     # checks, judges, mock LLMs, scorecard CLI
  logging_config.py         # text / JSON logging
ui/app.py                   # Streamlit UI
.streamlit/config.toml      # Streamlit production config
tests/                      # pytest suite + eval/cases.json
Dockerfile, docker/entrypoint.sh, docker-compose.yml, .dockerignore
requirements.lock           # exact, tested dependency versions
.github/workflows/ci.yml    # tests + Docker build
Docs/
```

## Known limitations

- Bangalore only, because that is the dataset's coverage.
- Cache and rate limiter are in-process (one API instance); use Redis or a gateway for several replicas.
- The LLM judge uses the same model family as the ranker (Groq gives this account no other production family); the heuristic judge is the cross-check.
- React/Next.js UI is deferred (Phase 4 task 6); Streamlit remains the UI.
