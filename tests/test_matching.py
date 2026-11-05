"""Phase 4: location normalization / fuzzy matching and cuisine synonyms / corrections."""

from __future__ import annotations

import pytest

from restaurant_recommender.data.filter import filter_restaurants, resolve_cuisines
from restaurant_recommender.data.locations import LocationIndex, location_key
from restaurant_recommender.models.schemas import PreferenceRequest


@pytest.mark.parametrize(
    ("text", "key"),
    [
        ("HSR Layout", "hsr"),
        ("Indira Nagar", "indiranagar"),
        ("M.G. Rd", "mgroad"),
        ("MG Road", "mgroad"),
        ("Church St", "churchstreet"),
        ("St. Marks Road", "streetmarksroad"),
        ("Bengaluru", "bangalore"),
        ("Koramangala Bangalore", "koramangala"),
        ("Koramangala 5th Block", "koramangala5thblock"),
        ("   ", ""),
    ],
)
def test_location_key(text, key):
    assert location_key(text) == key


@pytest.fixture
def index() -> LocationIndex:
    return LocationIndex(
        [
            ["Indiranagar", "Bangalore"],
            ["Koramangala 5th Block", "Bangalore"],
            ["Koramangala 7th Block", "Bangalore"],
            ["HSR", "Bangalore"],
            ["MG Road", "Church Street", "Bangalore"],
            ["St. Marks Road", "Bangalore"],
        ]
    )


@pytest.mark.parametrize(
    ("query", "names", "fuzzy", "label"),
    [
        ("indiranagar", ["Indiranagar"], False, "indiranagar"),
        ("Indira Nagar", ["Indiranagar"], False, "Indira Nagar"),
        ("Indranagar", ["Indiranagar"], True, "Indiranagar"),
        ("Koramangla", ["Koramangala 5th Block", "Koramangala 7th Block"], True, "Koramangala"),
        ("Koramangala", ["Koramangala 5th Block", "Koramangala 7th Block"], False, "Koramangala"),
        ("HSR Layout", ["HSR"], False, "HSR Layout"),
        ("M.G. Rd", ["MG Road"], False, "M.G. Rd"),
        ("Church St", ["Church Street"], False, "Church St"),
        ("St Marks Rd", ["St. Marks Road"], False, "St Marks Rd"),
        ("Bengaluru", ["Bangalore"], False, "Bengaluru"),
        ("Delhi", [], False, "Delhi"),
        ("Atlantis", [], False, "Atlantis"),
    ],
)
def test_resolve(index, query, names, fuzzy, label):
    match = index.resolve(query)
    assert (match.names, match.fuzzy, match.label) == (names, fuzzy, label)


def test_mask(index):
    assert index.mask(["Indiranagar", "HSR"]).tolist() == [True, False, False, True, False, False]
    assert index.mask(["Bangalore"]).all()
    assert not index.mask(["Nowhere"]).any()


def test_fuzzy_location_in_filter_reports_correction(store):
    result = filter_restaurants(store, PreferenceRequest(location="Indranagar"), 30)
    assert {r.name for r in result.candidates} == {"Toscano", "Pasta Street", "Punjabi Tadka", "Steak House"}
    assert result.notes == ["Showing results for Indiranagar (closest match to “Indranagar”)."]
    assert result.location_label == "Indiranagar"


def test_cuisine_spelling_correction(store):
    result = filter_restaurants(store, PreferenceRequest(location="Bangalore", cuisine="Itallian"), 30)
    assert {r.name for r in result.candidates} == {"Toscano", "Pasta Street", "Pizza Bakery", "New Cafe", "Olive Bistro"}
    assert result.notes == ["Showing Italian (closest match to “itallian”)."]
    assert result.cuisine_terms == ["italian"]


def test_cuisine_synonym_expansion(store):
    result = filter_restaurants(store, PreferenceRequest(location="Bangalore", cuisine="pizza"), 30)
    assert "Pizza Bakery" in {r.name for r in result.candidates}
    assert {"Toscano", "Olive Bistro"} <= {r.name for r in result.candidates}
    assert result.notes == ["“pizza” also matches Italian restaurants."]


def test_unknown_cuisine_stays_empty(store):
    result = filter_restaurants(store, PreferenceRequest(location="Bangalore", cuisine="Klingon"), 30)
    assert result.candidates == [] and result.eliminated_by == "cuisine" and result.notes == []


def test_resolve_cuisines_unit():
    vocab = ["North Indian", "South Indian", "Italian", "Pizza", "Chinese"]
    assert resolve_cuisines(["indian"], vocab) == (["indian"], [])
    terms, notes = resolve_cuisines(["chineese", "dosa"], vocab)
    assert terms == ["chinese", "south indian"]
    assert notes[0].startswith("Showing Chinese") and "South Indian" in notes[1]


def test_service_shows_llm_the_corrected_location(store, settings):
    from restaurant_recommender.services.recommend import RecommendationService
    from tests.helpers import FakeLLM

    llm = FakeLLM()
    response = RecommendationService(store, llm, settings).recommend(PreferenceRequest(location="Indranagar", top_n=2))
    prompt = llm.calls[0][1]["content"]
    assert "location=Indiranagar" in prompt and "Indranagar" not in prompt
    assert response.meta.query_notes and "closest match" in response.meta.query_notes[0]
