# Architecture — AI-Powered Restaurant Recommendation System

## 1. Overview

This document defines the system architecture for an AI-powered restaurant recommendation service inspired by Zomato. The application combines **deterministic filtering** on structured restaurant data with an **LLM reasoning layer** that ranks candidates and produces human-like explanations.

**Design principle:** Filter first, then reason. Structured preferences (location, budget, cuisine, rating) are applied as hard filters against the dataset. Only a shortlist of candidates is passed to the LLM for ranking, explanation, and optional summarization. This keeps latency, cost, and hallucination risk under control.

```
┌─────────────┐     ┌──────────────────┐     ┌────────────────────┐     ┌──────────────┐
│  User UI    │────▶│  API / Orchestr. │────▶│  Filter Engine     │────▶│  LLM Ranker  │
│  (prefs)    │     │  Layer           │     │  (structured data) │     │  + Explain   │
└─────────────┘     └──────────────────┘     └────────────────────┘     └──────┬───────┘
                                                                               │
                                                                               ▼
                                                                        ┌──────────────┐
                                                                        │  Result UI   │
                                                                        │  (top N)     │
                                                                        └──────────────┘
```

---

## 2. Goals & Non-Goals

### Goals

| Goal | How architecture supports it |
|---|---|
| Preference-based recommendations | Typed preference schema → filter pipeline → LLM rank |
| Real-world data | Hugging Face Zomato dataset ingested and normalized once |
| Personalized, human-like output | Prompted LLM with structured candidate context |
| Clear, useful results | Fixed response contract: name, cuisine, rating, cost, explanation |

### Non-Goals (v1)

- User accounts, history, or collaborative filtering
- Real-time booking / availability
- Geospatial distance / map search beyond city/locality text match
- Fine-tuning a custom model (prompt + filter is sufficient for v1)
- Multi-tenant production scale-out

---

## 3. High-Level Architecture

The system is organized into five layers that map 1:1 to the problem statement workflow.

| Layer | Responsibility | Problem-statement mapping |
|---|---|---|
| **Presentation** | Collect preferences; render recommendations | User Input, Output Display |
| **API / Orchestration** | Validate input; coordinate filter → LLM → response | Integration Layer |
| **Data** | Ingest, clean, store, query restaurant records | Data Ingestion |
| **Recommendation** | Rank + explain candidates via LLM | Recommendation Engine |
| **Infrastructure** | Config, secrets, logging, caching | Cross-cutting |

```
                    ┌─────────────────────────────────────────┐
                    │           Presentation Layer            │
                    │  Web UI (forms + recommendation cards)  │
                    └────────────────────┬────────────────────┘
                                         │ HTTP / JSON
                    ┌────────────────────▼────────────────────┐
                    │        API / Orchestration Layer        │
                    │  validate → filter → prompt → parse     │
                    └──────┬─────────────────────┬────────────┘
                           │                     │
              ┌────────────▼──────────┐  ┌───────▼────────────┐
              │      Data Layer       │  │ Recommendation     │
              │  Dataset · Store ·    │  │ Engine             │
              │  Filter Service       │  │ Prompt · LLM ·     │
              │                       │  │ Response Parser    │
              └───────────────────────┘  └────────────────────┘
```

---

## 4. Recommended Tech Stack

Chosen for fast delivery of an LLM + tabular-data MVP, with a clean path to a production API.

| Concern | Choice | Rationale |
|---|---|---|
| Language | Python 3.11+ | Best ecosystem for Hugging Face datasets + LLM SDKs |
| API | FastAPI | Typed request/response models; async-friendly for LLM calls |
| UI (v1) | Streamlit *or* simple React/Next.js SPA | Streamlit for fastest demo; React if polished product UI is required |
| Data load | `datasets` (Hugging Face) + pandas | Direct load from [ManikaSaini/zomato-restaurant-recommendation](https://huggingface.co/datasets/ManikaSaini/zomato-restaurant-recommendation) |
| Persistence (v1) | In-memory DataFrame / Parquet cache on disk | Avoid DB until query patterns demand it |
| Persistence (optional later) | SQLite / PostgreSQL | If filtering becomes heavy or multi-user write needed |
| LLM | OpenAI-compatible API (OpenAI, Groq, Azure, or local Ollama) | Swappable via env config; structured JSON output |
| Config | `.env` + pydantic-settings | Secrets and model names outside code |
| Packaging | `pyproject.toml` / `requirements.txt` | Reproducible installs |

**Default path for this repo:** FastAPI backend + Streamlit frontend (two processes), shared Python package for domain logic.

---

## 5. Component Design

### 5.1 Presentation Layer

**Responsibilities**

- Collect: location, budget (`low` | `medium` | `high`), cuisine, minimum rating, free-text additional preferences
- Call `POST /api/v1/recommendations`
- Render top-N cards: restaurant name, cuisine, rating, estimated cost, AI explanation
- Show optional overall summary from the LLM
- Surface empty-state and error messages (no matches, LLM timeout)

**UX contract**

- Hard filters are explicit (user understands why results are limited)
- Explanations are short (1–3 sentences) and preference-grounded
- Never invent restaurants not present in the filtered candidate set

### 5.2 API / Orchestration Layer

**Endpoint (v1)**

```
POST /api/v1/recommendations
```

**Request schema**

```json
{
  "location": "Bangalore",
  "budget": "medium",
  "cuisine": "Italian",
  "min_rating": 3.5,
  "additional_preferences": "family-friendly, outdoor seating",
  "top_n": 5
}
```

**Response schema**

```json
{
  "summary": "Short overall suggestion narrative…",
  "recommendations": [
    {
      "rank": 1,
      "name": "Trattoria XYZ",
      "cuisine": "Italian",
      "rating": 4.3,
      "estimated_cost": "₹1,200 for two",
      "location": "Indiranagar, Bangalore",
      "explanation": "Matches your Italian + medium budget preference…",
      "source_id": "rest_123"
    }
  ],
  "meta": {
    "candidates_considered": 18,
    "filters_applied": ["location", "budget", "cuisine", "min_rating"]
  }
}
```

**Orchestration steps (single request)**

1. Validate and normalize preferences
2. Map budget band → cost range using dataset statistics or fixed thresholds
3. Run filter service → candidate list (cap, e.g. 20–50)
4. If zero candidates → return empty result (skip LLM)
5. Build prompt with preferences + compact candidate table
6. Call LLM; parse structured JSON
7. Merge LLM output with canonical restaurant fields (trust data layer for facts)
8. Return ranked recommendations

### 5.3 Data Layer

#### Ingestion

Source: Hugging Face dataset `ManikaSaini/zomato-restaurant-recommendation`.

Pipeline (run once at startup or via a CLI ingest job):

1. Download / load dataset split
2. Select and rename columns to a canonical schema
3. Clean nulls, normalize cuisine strings, parse cost/rating types
4. Derive `budget_band` from cost if not present
5. Persist cleaned snapshot (Parquet) for fast reload

**Canonical restaurant schema**

| Field | Type | Notes |
|---|---|---|
| `id` | string | Stable unique key |
| `name` | string | Restaurant name |
| `location` / `city` / `locality` | string | Match against user location |
| `cuisine` | string \| list | Support multi-cuisine rows |
| `rating` | float | Used in min-rating filter |
| `cost_for_two` | number | Basis for budget mapping |
| `budget_band` | enum | `low` \| `medium` \| `high` (derived or mapped) |
| `votes` / `highlights` | string (optional) | Extra context for LLM |
| raw extras | optional | Kept only if useful for explanations |

#### Filter Service

Deterministic rules, applied in order:

1. **Location** — case-insensitive contains / exact city match
2. **Cuisine** — substring or token match against cuisine field
3. **Budget** — `budget_band` equality or cost within band thresholds
4. **Min rating** — `rating >= min_rating`
5. **Additional preferences** — *soft* signal only: passed to LLM as guidance; optional keyword boost if fields exist (e.g. “family”, “delivery”)

**Guardrails**

- Always cap candidates before LLM (e.g. `max_candidates = 30`)
- Prefer higher-rated rows when truncating
- Log filter hit counts for debugging empty results

### 5.4 Recommendation Engine (LLM)

**Role:** Rank filtered candidates and write explanations — **not** invent new restaurants or change factual fields.

**Subcomponents**

| Subcomponent | Role |
|---|---|
| Prompt builder | Inject system rules + user prefs + candidate JSON/table |
| LLM client | Provider-agnostic chat completion with JSON mode / schema |
| Response parser | Validate ranks, IDs, and required explanation fields |
| Fact merger | Overlay LLM rank/explanation onto dataset facts (name, rating, cost) |

**Prompt design principles**

- System: “You are a restaurant recommendation assistant. Only use restaurants in the provided list. Return JSON matching the schema. Do not invent ratings or costs.”
- User: preferences block + numbered candidate list with `id`, name, cuisine, rating, cost, location, optional notes
- Ask for: ordered top-N, per-item explanation tied to preferences, optional summary
- Temperature: low (e.g. 0.2–0.4) for consistent ranking

**Failure handling**

| Failure | Behavior |
|---|---|
| LLM timeout / 5xx | Fallback: return top candidates by rating with template explanations |
| Invalid JSON | One retry with repair prompt; then rating fallback |
| Hallucinated ID | Drop unknown IDs; fill remaining slots from rating-sorted candidates |

### 5.5 Infrastructure / Cross-Cutting

- **Config:** `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL`, `DATA_PATH`, `MAX_CANDIDATES`, `TOP_N_DEFAULT`
- **Logging:** request id, filter counts, LLM latency, fallback used
- **Caching (optional):** cache identical preference hashes → recommendations for short TTL
- **Observability:** simple timing metrics around filter vs LLM phases

---

## 6. End-to-End Data Flow

```
User submits preferences
        │
        ▼
API validates PreferenceDTO
        │
        ▼
BudgetMapper: low/medium/high → cost bounds
        │
        ▼
FilterEngine.query(prefs) → List[Restaurant] (≤ max_candidates)
        │
        ├── empty? ──▶ return { recommendations: [], summary: "No matches…" }
        │
        ▼
PromptBuilder.build(prefs, candidates)
        │
        ▼
LLMClient.complete(prompt) → raw JSON
        │
        ▼
ResponseParser + FactMerger → RecommendationResponse
        │
        ▼
UI renders ranked cards + summary
```

---

## 7. Repository Layout (suggested)

```
Apps/
├── docs/
│   ├── problemStatement.md
│   └── architecture.md
├── data/
│   └── processed/                 # Parquet / cached cleaned dataset
├── src/
│   └── restaurant_recommender/
│       ├── __init__.py
│       ├── config.py              # settings / env
│       ├── models/                # pydantic schemas (prefs, restaurant, response)
│       ├── data/
│       │   ├── ingest.py          # HF download + clean
│       │   ├── store.py           # load Parquet / DataFrame access
│       │   └── filter.py          # preference filters
│       ├── llm/
│       │   ├── client.py          # provider wrapper
│       │   ├── prompts.py         # system + user templates
│       │   └── parser.py          # JSON validate / repair
│       ├── services/
│       │   └── recommend.py       # orchestration
│       └── api/
│           └── main.py            # FastAPI app + routes
├── ui/
│   └── app.py                     # Streamlit (or frontend/)
├── tests/
│   ├── test_filter.py
│   ├── test_prompt.py
│   └── test_recommend_fallback.py
├── .env.example
├── requirements.txt
└── README.md
```

---

## 8. Key Design Decisions

| Decision | Choice | Why |
|---|---|---|
| Hybrid filter + LLM | Yes | Accurate hard constraints; LLM for soft ranking & language |
| Facts from dataset only | Yes | Prevents hallucinated ratings/costs |
| Cap candidates before LLM | Yes | Cost, latency, prompt size |
| Budget as enum mapped to cost | Yes | Matches problem statement UX; maps cleanly to numeric data |
| Additional prefs as soft constraints | Yes | Free text is ambiguous; LLM handles nuance better than brittle rules |
| Rating-based fallback | Yes | App remains usable if LLM is down |
| No vector DB in v1 | Deferred | Structured filters cover primary axes; embeddings optional later for “vibe” search |

---

## 9. Security & Configuration

- Store API keys only in environment / secret manager — never in source
- Validate and bound all inputs (`top_n` max, string lengths, rating range)
- Do not send the full dataset to the LLM — only filtered candidates
- Rate-limit the recommendations endpoint if exposed publicly
- Log prompts carefully (may contain user free-text); avoid logging API keys

---

## 10. Performance Targets (v1)

| Metric | Target |
|---|---|
| Filter latency | &lt; 100 ms on cached DataFrame |
| LLM latency | 1–5 s typical (provider-dependent) |
| End-to-end p95 | &lt; 8 s for demo workloads |
| Candidates to LLM | ≤ 30 |
| Returned recommendations | 3–5 default |

Optimizations if needed: Parquet cache, preference-hash response cache, smaller/faster model for ranking, parallel-safe DataFrame reads.

---

## 11. Testing Strategy

| Layer | What to test |
|---|---|
| Ingest | Column mapping, null handling, budget derivation |
| Filter | Each preference alone and combined; empty-result cases |
| Prompt | Candidate serialization size; required fields present |
| Parser | Valid JSON, missing fields, hallucinated IDs |
| Orchestration | Happy path; LLM failure → rating fallback |
| API | Schema validation; 4xx on bad input |
| UI (smoke) | Form submit → cards render |

Prefer deterministic unit tests for filter/parser; mock the LLM client in orchestration tests.

---

## 12. Implementation Phases

### Phase 1 — Data foundation

- Ingest Hugging Face dataset
- Canonical schema + Parquet cache
- Filter service with unit tests

### Phase 2 — Recommendation path

- Preference + response schemas
- Prompt builder + LLM client
- Orchestration with fallback
- FastAPI `POST /api/v1/recommendations`

### Phase 3 — UI & polish

- Preference form + results display
- Empty states, loading, errors
- Optional summary block
- README with run instructions

### Phase 4 — Hardening (optional)

- Response caching
- Better locality matching
- Telemetry / simple eval set (prefs → expected cuisine/budget adherence)
- Swap Streamlit for React if product UI is required

---

## 13. Open Extension Points

These are intentionally out of v1 but supported by the layering:

- **Embeddings / semantic search** for additional preferences (“romantic quiet place”)
- **User session history** for re-ranking
- **Multi-city geospatial** filters
- **A/B prompts** and evaluation harness
- **Streaming** LLM tokens to the UI for perceived speed

---

## 14. Summary

Build a **filter-then-LLM** pipeline: normalize the Zomato Hugging Face dataset, apply hard preference filters, ask an LLM only to **rank and explain** a small candidate set, and always display **dataset-backed facts** with AI-generated explanations. FastAPI orchestrates the flow; Streamlit (or a SPA) collects input and shows top recommendations.
