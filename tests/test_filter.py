"""Hard filters (architecture §5.3, edge cases FL-* / IN-09 / IN-10)."""

from __future__ import annotations

import pytest

from restaurant_recommender.data.filter import filter_restaurants
from restaurant_recommender.models.schemas import PreferenceRequest


def names(store, max_candidates=30, **prefs) -> list[str]:
    result = filter_restaurants(store, PreferenceRequest(**prefs), max_candidates)
    return [r.name for r in result.candidates]


def test_location_only(store):
    assert set(names(store, location="Indiranagar")) == {"Toscano", "Pasta Street", "Punjabi Tadka", "Steak House"}


@pytest.mark.parametrize("location", ["  indiRANagar ", "INDIRANAGAR", "indiranagar"])
def test_location_is_trimmed_and_case_insensitive(store, location):  # IN-09, IN-10
    assert len(names(store, location=location)) == 4


def test_location_city_alias_and_city_match(store):  # FL-02 alias, FL-03 city substring
    assert len(names(store, location="Bengaluru")) == 12
    assert len(names(store, location="Bangalore")) == 12


def test_location_partial_and_multi_part(store):
    assert set(names(store, location="Koramangala")) == {"Pizza Bakery", "New Cafe", "Thai Garden"}
    assert len(names(store, location="Indiranagar, Bangalore")) == 4  # all parts must match
    assert names(store, location="Indiranagar, Whitefield") == []


def test_location_ignores_street_address(restaurants_df):
    from restaurant_recommender.data.store import RestaurantStore

    df = restaurants_df.copy()
    df.loc[0, "address"] = "Near Delhi Public School, Indiranagar, Bangalore"
    assert names(RestaurantStore(df), location="Delhi") == []


def test_cuisine_whole_word_match(store):  # FL-05
    assert set(names(store, location="Bangalore", cuisine="Italian")) == {
        "Toscano", "Pasta Street", "Pizza Bakery", "New Cafe", "Olive Bistro"
    }
    assert set(names(store, location="Bangalore", cuisine="indian")) == {"Punjabi Tadka", "Udupi Grand", "Dhaba 24"}
    assert set(names(store, location="Bangalore", cuisine="North Indian")) == {"Punjabi Tadka", "Dhaba 24"}
    assert names(store, location="Bangalore", cuisine="Tea") == ["Chai Point"]  # not "Steak"


def test_cuisine_any_of_several(store):
    assert set(names(store, location="Bangalore", cuisine="Thai, Steak")) == {"Thai Garden", "Steak House"}


def test_budget_bands_exclude_unknown_cost(store):  # DI-07
    medium = set(names(store, location="Bangalore", budget="medium"))
    assert medium == {"Pasta Street", "Pizza Bakery", "Punjabi Tadka", "New Cafe", "Thai Garden", "Olive Bistro"}
    assert set(names(store, location="Bangalore", budget="low")) == {"Wok Express", "Udupi Grand", "Chai Point", "Dhaba 24"}
    assert names(store, location="Bangalore", budget="high") == ["Toscano"]
    all_banded = medium | set(names(store, location="Bangalore", budget="low")) | {"Toscano"}
    assert "Steak House" not in all_banded


def test_min_rating_is_inclusive_and_excludes_unrated(store):  # FL-07, DI-06
    result = set(names(store, location="Bangalore", min_rating=3.5))
    assert {"Thai Garden", "Olive Bistro"} <= result
    assert "New Cafe" not in result and "Dhaba 24" not in result


def test_zero_min_rating_is_not_a_filter(store):
    result = filter_restaurants(store, PreferenceRequest(location="Bangalore", min_rating=0), 30)
    assert "min_rating" not in result.filters_applied
    assert result.total_matches == 12


def test_combined_filters_and_counts(store):
    result = filter_restaurants(
        store,
        PreferenceRequest(location="Indiranagar", cuisine="Italian", budget="medium", min_rating=3.5),
        30,
    )
    assert [r.name for r in result.candidates] == ["Pasta Street"]
    assert result.filters_applied == ["location", "cuisine", "budget", "min_rating"]  # OR-08
    assert result.filter_counts == {"all": 12, "location": 4, "cuisine": 2, "budget": 1, "min_rating": 1}


def test_empty_results_report_eliminating_filter(store):  # FL-01
    atlantis = filter_restaurants(store, PreferenceRequest(location="Atlantis", cuisine="Italian"), 30)
    assert atlantis.candidates == [] and atlantis.eliminated_by == "location"
    btm = filter_restaurants(store, PreferenceRequest(location="BTM", cuisine="Italian"), 30)
    assert btm.candidates == [] and btm.eliminated_by == "cuisine"


def test_cap_keeps_highest_rated(store):  # FL-09, FL-15
    result = filter_restaurants(store, PreferenceRequest(location="Bangalore"), max_candidates=3)
    assert [r.name for r in result.candidates] == ["Udupi Grand", "Toscano", "Steak House"]
    assert result.total_matches == 12 and result.truncated


def test_order_is_deterministic_with_ties_and_unrated_last(store):  # FL-11
    first = names(store, location="Bangalore")
    assert first == names(store, location="Bangalore")
    assert first.index("Thai Garden") < first.index("Olive Bistro")  # same rating, more votes first
    assert first[-1] == "New Cafe"  # unrated last


def test_additional_preferences_never_change_hard_filters(store):  # INV-05 / FL-12
    base = names(store, location="Bangalore", budget="high")
    assert names(store, location="Bangalore", budget="high", additional_preferences="very cheap please") == base
