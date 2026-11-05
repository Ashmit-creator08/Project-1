"""Phase 4: prompt variants (A/B), strict output schema, and explanation judges."""

from __future__ import annotations

import json

import pytest

from restaurant_recommender.data.filter import filter_restaurants
from restaurant_recommender.eval.judge import HeuristicJudge, LLMJudge, mentions_preference
from restaurant_recommender.llm.prompts import VARIANTS, build_messages, get_variant, ranking_schema
from restaurant_recommender.models.schemas import PreferenceRequest
from restaurant_recommender.services.recommend import RecommendationService
from tests.helpers import FakeLLM, prompt_ids


def _shortlist(store, **prefs):
    prefs = PreferenceRequest(**prefs)
    return prefs, filter_restaurants(store, prefs, 30).candidates


def test_registry():
    assert set(VARIANTS) == {"v1_baseline", "v2_stricter_json"}
    assert get_variant(None).name == "v1_baseline"
    with pytest.raises(ValueError):
        get_variant("nope")


def test_v2_is_compact_and_complete(store):
    prefs, shortlist = _shortlist(store, location="Bangalore", additional_preferences="family-friendly")
    v1, _ = build_messages(prefs, shortlist, 5, variant="v1_baseline")
    v2, included = build_messages(prefs, shortlist, 5, variant="v2_stricter_json")
    user = v2[1]["content"]
    assert prompt_ids(v2) == [c.id for c in included] == [c.id for c in shortlist]
    assert "id | name | cuisines" in user and "Return exactly 5 recommendations" in user
    assert "<user_notes>family-friendly</user_notes>" in user
    assert "Output JSON only" in v2[0]["content"]
    assert sum(len(m["content"]) for m in v2) < sum(len(m["content"]) for m in v1)


def test_v2_cells_cannot_break_the_table(store, restaurants_df):
    from restaurant_recommender.data.store import RestaurantStore

    df = restaurants_df.copy()
    df.loc[0, "name"] = "Pipe | Bar"
    prefs, shortlist = _shortlist(RestaurantStore(df), location="Indiranagar", cuisine="Italian")
    user = build_messages(prefs, shortlist, 2, variant="v2_stricter_json")[0][1]["content"]
    assert "Pipe / Bar" in user


def test_ranking_schema_is_strict_with_id_enum():
    schema = ranking_schema(["rest_a", "rest_b"])
    item = schema["properties"]["recommendations"]["items"]
    assert item["properties"]["id"]["enum"] == ["rest_a", "rest_b"]
    assert schema["additionalProperties"] is False and item["additionalProperties"] is False
    assert set(item["required"]) == {"id", "explanation"} and set(schema["required"]) == {"summary", "recommendations"}


def test_service_sends_schema_for_included_candidates(store, settings):
    llm = FakeLLM()
    RecommendationService(store, llm, settings).recommend(PreferenceRequest(location="Indiranagar", top_n=2))
    enum = llm.schemas[0]["properties"]["recommendations"]["items"]["properties"]["id"]["enum"]
    assert enum == prompt_ids(llm.calls[0])


@pytest.mark.parametrize("variant", list(VARIANTS))
def test_each_variant_runs_end_to_end(store, settings, variant):
    response = RecommendationService(store, FakeLLM(), settings, prompt_variant=variant).recommend(
        PreferenceRequest(location="Bangalore", cuisine="Italian", top_n=3)
    )
    assert len(response.recommendations) == 3 and response.meta.llm_used
    assert response.meta.prompt_variant == variant


# --------------------------------------------------------------------------- judges


@pytest.fixture
def pasta(store):
    return next(r for r in store.get_many(store.get_all()["id"]) if r.name == "Pasta Street")


PREFS = PreferenceRequest(location="Indiranagar", cuisine="Italian", budget="medium", min_rating=3.5)


def test_heuristic_rewards_grounded_specific_explanations(pasta):
    text = "Pasta Street serves Italian in Indiranagar, rated 4.1, and at ₹800 for two it fits your medium budget."
    score = HeuristicJudge().score(PREFS, pasta, text)
    assert (score.groundedness, score.preference_fit, score.clarity) == (2, 2, 2) and score.acceptable


def test_heuristic_penalizes_invented_amenities(pasta):
    score = HeuristicJudge().score(PREFS, pasta, "A lovely rooftop with free parking, perfect for you.")
    assert score.groundedness == 0 and not score.acceptable


def test_heuristic_accepts_hedged_amenities_the_user_asked_for(pasta):
    prefs = PREFS.model_copy(update={"additional_preferences": "rooftop"})
    text = "Italian in Indiranagar rated 4.1; a rooftop is not confirmed by the data."
    assert HeuristicJudge().score(prefs, pasta, text).groundedness == 2


def test_heuristic_clarity_drops_for_rambling_text(pasta):
    text = " ".join(["Italian food is good."] * 8)
    assert HeuristicJudge().score(PREFS, pasta, text).clarity < 2


def test_mentions_preference(pasta):
    assert mentions_preference("Great Italian spot.", PREFS, pasta)
    assert not mentions_preference("A nice place.", PREFS, pasta)


def test_llm_judge_parses_scores_and_handles_bad_output(pasta):
    good = FakeLLM(json.dumps({"groundedness": 2, "preference_fit": 1, "clarity": 2, "rationale": "ok"}))
    score = LLMJudge(good).score(PREFS, pasta, "Italian in Indiranagar.")
    assert score.total == 5 and score.acceptable and score.judge == "llm:fake-llm"
    assert good.schemas[0]["required"] == ["groundedness", "preference_fit", "clarity", "rationale"]
    assert LLMJudge(FakeLLM("not json")).score(PREFS, pasta, "x") is None
