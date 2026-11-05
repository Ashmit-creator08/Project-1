# Edge Cases & Corner Scenarios

Catalog of edge cases for the AI-powered restaurant recommendation system, derived from [`architecture.md`](./architecture.md) and [`implementation-plan.md`](./implementation-plan.md).

**How to use:** Treat each row as a test or handling requirement. `Expected behavior` is the v1 contract unless marked Phase 4.

**Severity legend**

| Level | Meaning |
|---|---|
| **P0** | Breaks correctness or trust (wrong facts, crashes, invented restaurants) |
| **P1** | Poor UX or silent wrong filtering; must handle in v1 |
| **P2** | Degraded quality / rare; acceptable with documented behavior |
| **P3** | Nice-to-have / Phase 4 |

---

## 1. Input Validation (API / UI)

| ID | Scenario | Expected behavior | Sev |
|---|---|---|---|
| IN-01 | Missing required fields (e.g. empty location) | 422 / form validation error; no LLM call | P0 |
| IN-02 | `budget` not in `{low, medium, high}` | Reject with clear validation message | P0 |
| IN-03 | `min_rating` &lt; 0 or &gt; 5 (or dataset max) | Reject or clamp with documented rule; prefer reject | P0 |
| IN-04 | `min_rating` non-numeric / NaN | Reject | P0 |
| IN-05 | `top_n` = 0 or negative | Reject | P0 |
| IN-06 | `top_n` above configured max (e.g. 50) | Reject or clamp to max; document choice | P1 |
| IN-07 | Extremely long strings (location / cuisine / additional prefs) | Reject over max length; prevent prompt blow-up | P0 |
| IN-08 | Whitespace-only location or cuisine (`"   "`) | Treat as empty → validation error | P1 |
| IN-09 | Leading/trailing spaces (`" Bangalore "`) | Trim before filter | P1 |
| IN-10 | Mixed case prefs (`"bAnGaLoRe"`, `"italian"`) | Case-insensitive matching | P1 |
| IN-11 | Special characters / emoji in free text | Accept if within length; sanitize for logs; do not crash prompt builder | P1 |
| IN-12 | SQL/prompt injection-like text in additional prefs | Still only a soft LLM signal; never execute; facts stay from dataset | P0 |
| IN-13 | Null vs omitted optional `additional_preferences` | Treat as empty string; proceed | P1 |
| IN-14 | Duplicate submit / double-click in UI | Disable button while in flight; single request | P1 |
| IN-15 | Unsupported content-type / malformed JSON body | 422/400; no partial processing | P0 |

---

## 2. Data Ingestion & Store

| ID | Scenario | Expected behavior | Sev |
|---|---|---|---|
| DI-01 | Hugging Face download failure / offline | Fail ingest with clear error; do not write partial corrupt Parquet | P0 |
| DI-02 | Dataset schema drift (renamed/missing columns) | Fail fast with mapping error naming missing columns | P0 |
| DI-03 | Duplicate restaurant rows | Dedupe by stable `id` or name+location rule; document | P1 |
| DI-04 | Missing `id` in source | Generate deterministic id (e.g. hash of name+location+cuisine) | P1 |
| DI-05 | Null / empty restaurant name | Drop row or mark unusable; never send nameless candidates to LLM | P0 |
| DI-06 | Null rating | Exclude from min-rating filter path or treat as non-match; do not coerce to 0 silently without docs | P1 |
| DI-07 | Null / non-numeric cost | Exclude from budget filter or leave `budget_band` unset and exclude; log count | P1 |
| DI-08 | Rating as string (`"NEW"`, `"-"`, `"4.5/5"`) | Parse where possible; else null-handle per DI-06 | P1 |
| DI-09 | Cost with currency symbols / commas (`"₹1,200"`) | Normalize to number during ingest | P1 |
| DI-10 | Multi-cuisine cell (`"Italian, Chinese"`) | Store parseable form; cuisine filter matches any token | P1 |
| DI-11 | Extreme outliers (cost = 0 or huge) | Still assign band; thresholds documented | P2 |
| DI-12 | Empty dataset after cleaning | Abort startup/ingest; API health unhealthy | P0 |
| DI-13 | Parquet missing at API startup | Clear error / health fail; instruct to run ingest | P0 |
| DI-14 | Corrupt Parquet file | Fail load with error; do not serve empty silently | P0 |
| DI-15 | Re-ingest overwrites cache | Atomic replace (write temp → rename) to avoid half-read | P1 |
| DI-16 | OneDrive / path lock on Windows while reading Parquet | Retry or clear error; document known issue | P2 |

---

## 3. Filtering & Budget Mapping

| ID | Scenario | Expected behavior | Sev |
|---|---|---|---|
| FL-01 | No restaurants match all hard filters | Empty `recommendations`; skip LLM; friendly summary | P0 |
| FL-02 | Location typo (`"Bengaluru"` vs `"Bangalore"`) | May return empty in v1 (exact/contains only); Phase 4 fuzzy optional | P2 |
| FL-03 | Location matches locality but user typed city only | Contains-match should still hit if city substring exists in field | P1 |
| FL-04 | Cuisine synonym (`"Pizza"` vs `"Italian"`) | May miss in v1 substring match; soft prefs won’t hard-fix | P2 |
| FL-05 | Cuisine partial overlap (`"North Indian"` query vs `"Indian"`) | Document substring/token rules; avoid false negatives where possible | P1 |
| FL-06 | Budget band boundary (cost exactly on threshold) | Inclusive rule documented (e.g. low ≤ X inclusive) | P1 |
| FL-07 | `min_rating` equal to restaurant rating | Include (`>=`) | P1 |
| FL-08 | Filters leave 1 candidate | Still call LLM (or short-circuit rank=1); return 1 item | P1 |
| FL-09 | Filters leave more than `MAX_CANDIDATES` | Truncate to cap; prefer higher rating; log truncated count | P0 |
| FL-10 | Candidates &lt; requested `top_n` | Return only available count; do not pad with invented rows | P0 |
| FL-11 | All remaining candidates share same rating when truncating | Stable secondary sort (name or id) for determinism | P2 |
| FL-12 | Additional prefs contradict hard filters (“cheap” + budget=high) | Hard filters win; LLM may note tension in explanation | P1 |
| FL-13 | Additional prefs ask for unavailable attribute (parking) with no data field | LLM must not invent; may say data doesn’t confirm | P1 |
| FL-14 | Only soft prefs provided conceptually but hard fields empty | Blocked by validation (hard fields required in v1) | P1 |
| FL-15 | Very broad prefs (no cuisine, popular city) → huge match set | Cap before LLM; still &lt; 100 ms filter target | P0 |
| FL-16 | Unicode / non-Latin cuisine or location strings | Do not crash; match if data contains same script | P2 |

---

## 4. LLM, Prompt, Parser & Fact Merger

| ID | Scenario | Expected behavior | Sev |
|---|---|---|---|
| LM-01 | LLM timeout | Rating-sorted fallback + template explanations; `meta.fallback_used=true` | P0 |
| LM-02 | LLM 401 / invalid API key | Fallback or clear 503-style error; never expose key | P0 |
| LM-03 | LLM 429 rate limit | Retry once with backoff optional; then fallback | P1 |
| LM-04 | LLM 5xx | Fallback | P0 |
| LM-05 | Empty LLM content | Repair retry once; then fallback | P0 |
| LM-06 | Invalid JSON (markdown fences, trailing commas) | Strip/repair once; then fallback | P0 |
| LM-07 | Valid JSON but wrong schema (missing `explanation`) | Reject item or retry; do not return incomplete cards silently | P1 |
| LM-08 | Hallucinated `source_id` not in candidates | Drop ID; fill remaining slots from rating-sorted candidates | P0 |
| LM-09 | LLM invents restaurant name/rating/cost | Fact merger overwrites with dataset values | P0 |
| LM-10 | LLM returns fewer than `top_n` | Fill from rating-sorted remainder or return fewer; never invent | P1 |
| LM-11 | LLM returns more than `top_n` | Truncate to `top_n` | P1 |
| LM-12 | Duplicate IDs in LLM output | Dedupe preserving first rank order | P1 |
| LM-13 | Rank numbers missing / non-sequential | Reassign ranks by list order after merge | P1 |
| LM-14 | Explanation empty or whitespace | Use template explanation or exclude item per policy | P1 |
| LM-15 | Explanation contradicts hard prefs (“great for sushi” when cuisine=Italian) | Prefer dataset-grounded prompt; accept residual LLM quality risk (P2) or light post-check | P2 |
| LM-16 | Summary references restaurants not returned | Strip or regenerate summary from final list | P2 |
| LM-17 | Prompt would exceed model context | Should not happen if cap ≤ 30; if it does, reduce candidates further | P1 |
| LM-18 | Provider returns non-JSON despite JSON mode | Same as LM-06 | P1 |
| LM-19 | Unsafe / toxic content in model output | Basic strip or reject explanation; keep factual fields | P2 |
| LM-20 | Very slow LLM (&gt; target e2e) | Timeout → fallback; UI shows loading then results | P1 |

---

## 5. Orchestration & API

| ID | Scenario | Expected behavior | Sev |
|---|---|---|---|
| OR-01 | Happy path | Ranked list with dataset facts + LLM explanations | P0 |
| OR-02 | Empty filter → must not call LLM | Assert no client call; empty response | P0 |
| OR-03 | Fallback path still returns dataset-backed fields | Name/rating/cost from store only | P0 |
| OR-04 | Concurrent requests | Thread/async-safe read of DataFrame; no cross-request prompt mix | P1 |
| OR-05 | Store not loaded | 503 + health unhealthy | P0 |
| OR-06 | Partial failure mid-merge | Return consistent response or fallback; never half-corrupt ranks | P1 |
| OR-07 | `meta.candidates_considered` accuracy | Equals pre-LLM shortlist size (after cap) | P2 |
| OR-08 | `filters_applied` lists only filters that ran | Reflect actual request fields used | P2 |
| OR-09 | Health check while ingest file missing | Unhealthy | P1 |
| OR-10 | Oversized request body | Reject early | P1 |

---

## 6. UI / Presentation

| ID | Scenario | Expected behavior | Sev |
|---|---|---|---|
| UI-01 | No matches | Empty state message; no blank cards | P0 |
| UI-02 | API unreachable | Clear error; no spinner forever | P0 |
| UI-03 | Long explanation text | Wrap / clamp without breaking layout | P2 |
| UI-04 | Missing optional summary | Hide summary block; still show cards | P1 |
| UI-05 | Fallback results | Show results; optional subtle “AI unavailable, ranked by rating” note | P2 |
| UI-06 | Rapid preference changes before response | Ignore stale response or cancel; show latest only | P1 |
| UI-07 | Zero-cost / null cost display | Show “N/A” rather than crash or `None` | P1 |
| UI-08 | Rating display precision | Consistent formatting (e.g. one decimal) | P3 |
| UI-09 | Streamlit rerun mid-request | No duplicate side effects; idempotent UX | P2 |

---

## 7. Configuration, Security & Infrastructure

| ID | Scenario | Expected behavior | Sev |
|---|---|---|---|
| CF-01 | Missing `LLM_API_KEY` | Fail startup or disable LLM with fallback-only mode (document) | P0 |
| CF-02 | Wrong `LLM_BASE_URL` / model name | Fallback after error; log without secrets | P1 |
| CF-03 | `.env` committed accidentally | Prevent via `.gitignore`; rotate key if leaked | P0 |
| CF-04 | Logging includes API key or full secrets | Never log secrets; redact | P0 |
| CF-05 | Logging full user free-text prompts in prod | Prefer truncated / sampled logs | P2 |
| CF-06 | Cache (Phase 4) returns stale results after re-ingest | Bust cache on ingest version / file mtime | P2 |
| CF-07 | Identical prefs hashed differently due to case/space | Normalize before cache key | P2 |
| CF-08 | Public exposure without rate limit | Phase 4 rate limit; v1 local-only assumption | P3 |

---

## 8. Cross-Cutting Correctness Invariants

These are non-negotiable checks for any recommendation response:

| ID | Invariant | Sev |
|---|---|---|---|
| INV-01 | Every `source_id` exists in the filtered candidate set for that request | P0 |
| INV-02 | `name`, `rating`, `estimated_cost`, `cuisine`, `location` match the data store for that `source_id` | P0 |
| INV-03 | Result count ≤ `min(top_n, candidates_considered)` | P0 |
| INV-04 | No LLM call when candidates_considered = 0 | P0 |
| INV-05 | Hard filters are never overridden by additional preferences or LLM | P0 |

---

## 9. Suggested Test Matrix (by phase)

| Phase | Priority edge cases to automate |
|---|---|
| **1 — Data** | DI-05–10, DI-12, FL-01, FL-06–11, FL-15 |
| **2 — Recommend API** | IN-01–07, LM-01, LM-06–12, OR-02–03, INV-01–05 |
| **3 — UI** | UI-01–02, UI-04, UI-07, IN-14 |
| **4 — Hardening** | FL-02, FL-04, CF-06–07, LM-16, rate limit |

---

## 10. Example Fixture Cases (for eval / demos)

| Case | Prefs | Likely edge |
|---|---|---|
| E1 | Location `Atlantis` + any cuisine | FL-01 empty |
| E2 | Cuisine `Italian`, `min_rating` = 4.9 in sparse city | FL-01 or FL-10 few results |
| E3 | Budget `low` + fine-dining locality only | FL-01 or very few |
| E4 | `top_n` = 5 but only 2 candidates | FL-10 |
| E5 | Additional: `romantic rooftop` with no such fields | FL-13 soft-only |
| E6 | Valid prefs + mocked LLM invalid JSON | LM-06 → fallback |
| E7 | Valid prefs + mocked hallucinated IDs | LM-08 + INV-01 |
| E8 | `min_rating` = 4.0 where ratings stored as strings pre-fix | DI-08 ingest must normalize |

---

## Summary

Most P0 risk sits in three places: **bad/missing data after ingest**, **empty or over-broad filters**, and **LLM hallucination / failure**. v1 handles these with validation, candidate caps, skip-LLM-on-empty, fact merging, ID allowlisting, and rating-based fallback. Fuzzy location/cuisine and cache correctness are deferred to Phase 4 unless they block the demo.
