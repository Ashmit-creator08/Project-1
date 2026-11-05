"""Orchestration: LLM path, fallbacks and the INV-* invariants with a mocked LLM."""

from __future__ import annotations

import json

import pytest

from restaurant_recommender.data.filter import filter_restaurants
from restaurant_recommender.llm.errors import LLMRequestError, LLMTimeoutError
from restaurant_recommender.models.schemas import PreferenceRequest
from restaurant_recommender.services.recommend import RecommendationService
from tests.helpers import FakeLLM, hallucinating_ranker, prompt_ids


def make_service(store, settings, llm):
    return RecommendationService(store, llm, settings)


def assert_invariants(store, settings, prefs, response):
    """INV-01..INV-03 and schema completeness."""
    shortlist = filter_restaurants(store, prefs, settings.max_candidates).candidates
    allowed = {c.id for c in shortlist}
    recs = response.recommendations
    assert len(recs) <= min(prefs.top_n, len(shortlist))  # INV-03
    assert [r.rank for r in recs] == list(range(1, len(recs) + 1))
    assert len({r.source_id for r in recs}) == len(recs)
    for item in recs:
        assert item.source_id in allowed  # INV-01
        fact = store.get(item.source_id)  # INV-02
        assert (item.name, item.rating, item.cost_for_two, item.location, item.cuisine) == (
            fact.name, fact.rating, fact.cost_for_two, fact.location_display, fact.cuisine_display
        )
        assert item.explanation.strip()


def test_happy_path_uses_llm_ranking_and_dataset_facts(store, settings):  # OR-01
    llm = FakeLLM()
    prefs = PreferenceRequest(location="Bangalore", cuisine="Italian", top_n=3)
    response = make_service(store, settings, llm).recommend(prefs)

    assert len(llm.calls) == 1
    shortlist_ids = prompt_ids(llm.calls[0])
    assert [r.source_id for r in response.recommendations] == list(reversed(shortlist_ids))[:3]
    assert all(r.explanation_source == "llm" for r in response.recommendations)
    assert all(r.name != "INVENTED NAME" and r.rating != 5.0 for r in response.recommendations)  # LM-09
    assert response.summary == "A strong set of picks for your evening."
    meta = response.meta
    assert meta.llm_used and not meta.fallback_used and meta.model == "fake-llm"
    assert meta.candidates_considered == 5 and meta.total_matches == 5  # OR-07
    assert meta.filters_applied == ["location", "cuisine"]
    assert_invariants(store, settings, prefs, response)


def test_empty_filter_never_calls_llm(store, settings):  # OR-02 / INV-04 / FL-01
    llm = FakeLLM()
    response = make_service(store, settings, llm).recommend(PreferenceRequest(location="Atlantis"))
    assert llm.calls == []
    assert response.recommendations == []
    assert "Atlantis" in response.summary and "Bangalore" in response.summary
    assert response.meta.candidates_considered == 0 and not response.meta.llm_used


@pytest.mark.parametrize(
    "error",
    [LLMTimeoutError("LLM did not respond within 20s"), LLMRequestError("LLM provider returned HTTP 500")],
)
def test_llm_failure_falls_back_to_rating_order(store, settings, error):  # LM-01 / LM-04 / OR-03
    prefs = PreferenceRequest(location="Bangalore", top_n=4, additional_preferences="quiet")
    response = make_service(store, settings, FakeLLM(error)).recommend(prefs)
    meta = response.meta
    assert meta.fallback_used and not meta.llm_used and meta.fallback_reason == str(error)
    assert [r.name for r in response.recommendations] == ["Udupi Grand", "Toscano", "Steak House", "Pasta Street"]
    assert all(r.explanation_source == "template" for r in response.recommendations)
    assert "unavailable" in response.summary and "not applied" in response.summary
    assert_invariants(store, settings, prefs, response)


def test_no_llm_configured_is_fallback_only_mode(store, settings):  # CF-01
    response = make_service(store, settings, None).recommend(PreferenceRequest(location="Indiranagar"))
    assert response.meta.fallback_used
    assert "GROQ_API_KEY" in response.meta.fallback_reason
    assert "no LLM configured" in response.summary
    assert response.recommendations[0].name == "Toscano"


def test_invalid_json_is_repaired_once(store, settings):  # LM-06
    llm = FakeLLM("Sorry, here are my picks: Toscano and Pasta Street")
    response = make_service(store, settings, llm).recommend(PreferenceRequest(location="Indiranagar", top_n=2))
    assert len(llm.calls) == 2
    assert llm.calls[1][-1]["role"] == "user" and "could not be used" in llm.calls[1][-1]["content"]
    assert response.meta.llm_used and not response.meta.fallback_used


def test_invalid_json_twice_falls_back(store, settings):  # LM-06 / E6
    llm = FakeLLM("nope", "still nope")
    prefs = PreferenceRequest(location="Indiranagar", top_n=2)
    response = make_service(store, settings, llm).recommend(prefs)
    assert len(llm.calls) == 2
    assert response.meta.fallback_used and "JSON" in response.meta.fallback_reason
    assert [r.name for r in response.recommendations] == ["Toscano", "Steak House"]
    assert_invariants(store, settings, prefs, response)


def test_hallucinated_ids_are_dropped_and_backfilled(store, settings):  # LM-08 / E7
    prefs = PreferenceRequest(location="Bangalore", top_n=3)
    response = make_service(store, settings, FakeLLM(hallucinating_ranker)).recommend(prefs)
    recs = response.recommendations
    assert "rest_doesnotexist" not in {r.source_id for r in recs}
    assert recs[0].name == "Toscano" and recs[0].explanation == "Real pick."  # 2nd candidate, LLM-picked
    assert [r.explanation_source for r in recs] == ["llm", "template", "template"]
    assert [r.name for r in recs[1:]] == ["Udupi Grand", "Steak House"]  # rating-order backfill
    assert response.meta.backfilled == 2 and not response.meta.fallback_used
    assert_invariants(store, settings, prefs, response)


def test_fewer_candidates_than_top_n_is_not_padded(store, settings):  # FL-10 / E4
    prefs = PreferenceRequest(location="Indiranagar", cuisine="Italian", top_n=5)
    response = make_service(store, settings, FakeLLM()).recommend(prefs)
    assert len(response.recommendations) == 2
    assert_invariants(store, settings, prefs, response)


def test_single_candidate(store, settings):  # FL-08
    prefs = PreferenceRequest(location="Whitefield", cuisine="Italian")
    response = make_service(store, settings, FakeLLM()).recommend(prefs)
    assert [r.name for r in response.recommendations] == ["Olive Bistro"]


def test_soft_preferences_never_override_hard_filters(store, settings):  # FL-12 / INV-05
    prefs = PreferenceRequest(location="Bangalore", budget="high", additional_preferences="very cheap")
    response = make_service(store, settings, FakeLLM()).recommend(prefs)
    assert {r.budget_band.value for r in response.recommendations} == {"high"}


def test_missing_explanation_gets_template(store, settings):  # LM-14
    def ranker(messages):
        ids = prompt_ids(messages)
        return json.dumps({"recommendations": [{"id": ids[0]}, {"id": ids[1], "explanation": "Lovely."}]})

    response = make_service(store, settings, FakeLLM(ranker)).recommend(PreferenceRequest(location="Indiranagar", top_n=2))
    first, second = response.recommendations
    assert first.explanation_source == "template" and first.explanation.startswith("Rated")
    assert second.explanation == "Lovely."


def test_summary_naming_an_unreturned_restaurant_is_replaced(store, settings):  # LM-16
    def ranker(messages):
        ids = prompt_ids(messages)
        return json.dumps({"summary": "Toscano and Pasta Street are both great.", "recommendations": [{"id": ids[1], "explanation": "x"}]})

    prefs = PreferenceRequest(location="Indiranagar", cuisine="Italian", top_n=1)
    response = make_service(store, settings, FakeLLM(ranker)).recommend(prefs)
    assert [r.name for r in response.recommendations] == ["Pasta Street"]
    assert "Toscano" not in response.summary


def test_template_explanations_mention_real_facts(store, settings):
    prefs = PreferenceRequest(location="Indiranagar", cuisine="Italian", budget="medium")
    item = make_service(store, settings, None).recommend(prefs).recommendations[0]
    assert item.name == "Pasta Street"
    assert "4.1/5" in item.explanation and "Italian" in item.explanation
    assert "₹800 for two" in item.explanation and "medium budget" in item.explanation
    assert item.estimated_cost == "₹800 for two"


def test_unknown_cost_displays_na(store, settings):  # UI-07
    item = make_service(store, settings, None).recommend(PreferenceRequest(location="Indiranagar", cuisine="Steak")).recommendations[0]
    assert item.estimated_cost == "N/A" and item.cost_for_two is None
