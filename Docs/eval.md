# Evaluation Plan — AI-Powered Restaurant Recommendation System

## Purpose

Define how we measure whether the system meets [`implementation-plan.md`](./implementation-plan.md) exit criteria and stays aligned with the filter-then-LLM design in [`architecture.md`](./architecture.md).

Evaluation answers four questions:

1. Do hard preferences actually constrain results?
2. Are returned facts faithful to the dataset (no hallucinations)?
3. Are LLM rankings/explanations useful when the model is available?
4. Does the system degrade safely when the LLM fails?

---

## Evaluation Layers

| Layer | What it measures | When | Automation |
|---|---|---|---|
| **L1 — Unit / contract** | Filters, parser, fallback, API validation | Phase 1–2 (CI) | Fully automated |
| **L2 — Constraint adherence** | Cuisine / budget / rating / location hold on live outputs | Phase 2+ | Automated against API or service |
| **L3 — Fidelity invariants** | IDs and facts match store; no invented restaurants | Phase 2+ | Automated |
| **L4 — Explanation quality** | Preference-grounded, non-contradictory prose | Phase 3–4 | Rubric (human or LLM-as-judge) |
| **L5 — Ops / UX smoke** | Latency, empty states, demo readiness | Phase 3–4 | Semi-automated + manual |

L1–L3 are required for v1. L4–L5 deepen confidence; Phase 4 formalizes the eval set and prompt A/B.

---

## Success Metrics

### Hard metrics (must pass)

| Metric | Definition | v1 target |
|---|---|---|
| **Constraint adherence** | Share of returned items that satisfy location, cuisine, budget band, and `min_rating` | **100%** on eval set |
| **ID validity** | Share of `source_id`s present in that request’s filtered candidate set | **100%** |
| **Fact fidelity** | Share of items where name/cuisine/rating/cost/location match the store for `source_id` | **100%** |
| **Empty-path correctness** | Impossible prefs → empty list and **no** LLM call | **100%** of empty fixtures |
| **Fallback availability** | Mocked LLM failure still returns ≤ `top_n` rating-sorted candidates with template explanations | **Pass** |
| **Result cardinality** | `len(recommendations) ≤ min(top_n, candidates_considered)` | **100%** |
| **Schema completeness** | Each item has name, cuisine, rating, estimated cost, explanation | **100%** on non-empty responses |

### Soft metrics (track; improve in Phase 4)

| Metric | Definition | Guidance |
|---|---|---|
| **Explanation groundedness** | Explanation references real prefs and/or candidate attributes without inventing amenities | ≥ 80% “Acceptable+” on rubric |
| **Preference coverage** | Explanation mentions at least one hard preference dimension | ≥ 80% |
| **Summary consistency** | Summary only refers to restaurants in the final list | ≥ 90% |
| **Duplicate-free ranking** | No duplicate `source_id` in one response | **100%** (promote to hard if easy) |
| **Latency** | End-to-end recommendation time | p95 &lt; 8 s (architecture target) |
| **Fallback rate** | Share of live requests using fallback | Monitor; spike = provider issue |
| **Filter hit rate** | Share of requests with `candidates_considered > 0` on demo prefs | Demo set should be mostly non-empty |

---

## Invariants (auto-check every eval run)

Copied as pass/fail gates from architecture + edge-case INV-* :

1. Every `source_id` ∈ filtered candidates for that request  
2. Display facts come from the data store for that id (LLM may only supply rank/explanation/summary)  
3. Result count ≤ `min(top_n, candidates_considered)`  
4. If `candidates_considered = 0`, LLM is not called  
5. Additional preferences never override hard filters  

Any invariant failure = **eval run failed**, regardless of soft scores.

---

## Eval Set

Keep a versioned fixture file (suggested path: `tests/eval/cases.json` or `evals/cases.yaml`). Each case includes preferences, expectations, and tags.

### Case schema

```json
{
  "id": "EV-01",
  "description": "Bangalore Italian medium budget",
  "prefs": {
    "location": "Bangalore",
    "budget": "medium",
    "cuisine": "Italian",
    "min_rating": 3.5,
    "additional_preferences": "family-friendly",
    "top_n": 5
  },
  "expect": {
    "non_empty": true,
    "all_match_hard_filters": true,
    "max_items": 5,
    "llm_called": true
  },
  "tags": ["happy_path", "phase2", "phase3_demo"]
}
```

### Seed cases (v1)

| ID | Prefs (summary) | Expect | Tags |
|---|---|---|---|
| EV-01 | Bangalore + Italian + medium + rating ≥ 3.5 | Non-empty; all hard filters hold | happy_path, demo |
| EV-02 | Delhi + Chinese + low + rating ≥ 3.0 | Non-empty if data allows; filters hold | happy_path, demo |
| EV-03 | Location `Atlantis` + Italian + medium | Empty; LLM not called | empty, phase2 |
| EV-04 | Valid city + cuisine with `min_rating` = 4.9 | Empty or few; cardinality rule holds; no invented fill | sparse |
| EV-05 | Broad city prefs that yield &gt; `MAX_CANDIDATES` matches | `candidates_considered` ≤ cap; filters hold | cap |
| EV-06 | `top_n` = 5 but only 2 filter matches | Exactly 2 items; no padding | cardinality |
| EV-07 | Additional prefs only soft (`romantic rooftop`) + valid hard prefs | Hard filters hold; no invented amenities required in facts | soft_prefs |
| EV-08 | Mixed case / padded strings (`" bangalore "`, `"italian"`) | Same as normalized happy path | normalization |
| EV-09 | Invalid budget / rating / `top_n` | API 422; no recommendations body | validation |
| EV-10 | Valid prefs + **mocked** LLM timeout/invalid JSON | Fallback; facts valid; explanations present (template OK) | fallback |
| EV-11 | Valid prefs + **mocked** hallucinated IDs | Only allowlisted IDs returned; slots backfilled from rating sort | hallucination |
| EV-12 | Contradictory soft vs hard (“very cheap” + budget=high) | Hard budget wins on all items | conflict |

Refresh EV-01/EV-02 after first ingest if city/cuisine coverage differs from assumptions — update fixtures to cities/cuisines that exist in the cleaned Parquet.

---

## Scoring Procedures

### A. Automated constraint & fidelity check

For each non-empty case (live or mocked LLM):

1. Re-run the filter engine on `prefs` → `candidates`  
2. Assert every result `source_id ∈ candidates`  
3. Assert each hard filter predicate on every result  
4. Assert store lookup equality for factual fields  
5. Assert cardinality and schema completeness  

**Score:** pass rate over cases (target 100% on hard metrics).

### B. Fallback / failure injection

With LLM client mocked:

| Injection | Assert |
|---|---|
| Timeout / 500 | Fallback list, `fallback_used`, facts valid |
| Invalid JSON | Repair or fallback; never crash |
| Hallucinated IDs | Dropped; backfill; INV-01 holds |
| Empty candidates | No mock call recorded |

### C. Explanation quality rubric (Phase 3–4)

Score each explanation 0–2 on:

| Dimension | 0 | 1 | 2 |
|---|---|---|---|
| **Groundedness** | Invents facts/amenities | Generic but safe | Ties to prefs and/or real attributes |
| **Preference fit** | Ignores prefs | Mentions one pref loosely | Clearly reflects hard prefs |
| **Clarity** | Incoherent | Readable | Concise (1–3 sentences) |

**Acceptable** = total ≥ 4 / 6. Track % Acceptable across EV-01, EV-02, EV-07.

Modes:

- **Human** — quick stakeholder pass on demo cases  
- **LLM-as-judge** — separate judge prompt with candidates + prefs + explanation; output JSON scores (Phase 4)

### D. Latency sample

On a warm API (Parquet loaded), run EV-01 N times (e.g. 10):

- Record filter time vs LLM time if instrumented  
- Report p50 / p95 end-to-end  
- Fail soft gate if p95 &gt; 8 s on the chosen provider (investigate before demo)

---

## Phase Mapping

| Implementation phase | Eval focus | Gate to exit phase |
|---|---|---|
| **Phase 1** | Filter unit tests; EV-03/05/06 style filter fixtures | Filters deterministic; empty & cap cases pass |
| **Phase 2** | L1–L3 + EV-01–12 (API); mocked LLM suite | Hard metrics 100%; invariants green |
| **Phase 3** | Manual smoke on EV-01/02/03; schema fields visible in UI | Demo checklist pass |
| **Phase 4** | Frozen eval set; scorecard; prompt A/B on L4 | Soft metrics baseline recorded; no hard-metric regression |

---

## Prompt A/B (Phase 4)

Keep prompt variants under `src/restaurant_recommender/llm/prompts/` (e.g. `v1_baseline`, `v2_stricter_json`).

| Step | Action |
|---|---|
| 1 | Run same eval cases with variant A and B |
| 2 | Hard metrics must remain 100% for both |
| 3 | Compare L4 Acceptable% and summary consistency |
| 4 | Prefer variant with higher groundedness without latency blow-up |
| 5 | Record winner + date in scorecard below |

Never ship a prompt that regresses ID validity or fact fidelity.

---

## Scorecard Template

Copy per release or prompt version:

```text
Eval run:     <date>
Git SHA:      <sha>
Dataset:      data/processed/restaurants.parquet (<row count>)
LLM model:    <model>
Prompt:       <variant>

Hard metrics
- Constraint adherence:     ___%  (target 100)
- ID validity:              ___%  (target 100)
- Fact fidelity:            ___%  (target 100)
- Empty-path correctness:   ___%  (target 100)
- Fallback suite:           PASS / FAIL
- Invariants:               PASS / FAIL

Soft metrics
- Explanation Acceptable%:  ___%
- Preference coverage:      ___%
- Summary consistency:      ___%
- e2e p95 latency:          ___ s

Cases failed: <ids + reason>
Decision:     Ship / Fix / Re-run
```

---

## How to Run (target workflow)

Once implemented:

```bash
# Unit + contract (CI)
pytest tests/ -q

# Eval harness against local API or in-process service
python -m restaurant_recommender.eval.run --cases tests/eval/cases.json --out evals/scorecard-latest.md

# Failure injection only
python -m restaurant_recommender.eval.run --cases tests/eval/cases.json --inject llm_failures
```

Until the harness exists, Phase 2 exit = pytest + manual curl of EV-01 and EV-03; Phase 3 = UI smoke on demo tags.

---

## What We Explicitly Do Not Eval in v1

- Personalization quality vs user history (no accounts)  
- Geospatial “nearest to me” accuracy  
- Booking availability / live Zomato parity  
- Absolute culinary taste judgments beyond preference adherence  
- Multi-tenant load testing  

---

## Definition of “Eval Passed” for a Release

A release may be called **eval-passed** when:

1. All hard metrics hit targets on the frozen eval set  
2. All invariants pass  
3. Fallback injection suite passes  
4. Soft metrics are recorded (even if below stretch goals)  
5. No open P0 edge cases from [`edge-case.md`](./edge-case.md) unmarked as handled  

v1 minimum: items 1–3 on Phases 1–2 scope; items 4–5 completed as part of Phase 3 demo readiness / Phase 4 hardening.
