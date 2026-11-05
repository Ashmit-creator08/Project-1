"""Hard-metric checks for one recommendation response (eval.md §A, invariants INV-01..05)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from restaurant_recommender.data.filter import FilterResult, cuisine_pattern, filter_restaurants
from restaurant_recommender.data.locations import split_location
from restaurant_recommender.data.store import RestaurantStore
from restaurant_recommender.models.schemas import PreferenceRequest, RecommendationResponse, Restaurant


@dataclass
class CaseCheck:
    items: int = 0
    id_valid: int = 0
    constraint_ok: int = 0
    fact_ok: int = 0
    complete: int = 0
    cardinality_ok: bool = True
    duplicate_free: bool = True
    violations: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.violations


def satisfies_hard_filters(
    store: RestaurantStore, restaurant: Restaurant, prefs: PreferenceRequest, interpreted: FilterResult
) -> bool:
    """Re-check every hard filter on dataset facts, independently of the filter code path.

    Location and cuisine use the system's interpretation of the query (after
    spelling correction / synonyms), which the user is told about in query_notes.
    """
    names = {n for n in (restaurant.locality, *restaurant.areas, restaurant.city) if n}
    for part in split_location(prefs.location):
        if not names & set(store.locations.resolve(part).names):
            return False
    if interpreted.cuisine_terms:
        if not re.search(cuisine_pattern(interpreted.cuisine_terms), ", ".join(restaurant.cuisines).lower()):
            return False
    if prefs.budget is not None and restaurant.budget_band != prefs.budget:
        return False
    if prefs.min_rating and (restaurant.rating is None or restaurant.rating < prefs.min_rating - 1e-9):
        return False
    return True


def check_response(
    store: RestaurantStore,
    prefs: PreferenceRequest,
    response: RecommendationResponse,
    max_candidates: int,
) -> CaseCheck:
    interpreted = filter_restaurants(store, prefs, max_candidates)
    allowed = {c.id for c in interpreted.candidates}
    recs = response.recommendations
    check = CaseCheck(items=len(recs))

    expected = min(prefs.top_n, len(interpreted.candidates))
    if len(recs) != expected:  # INV-03 + no padding / no silent truncation
        check.cardinality_ok = False
        check.violations.append(f"returned {len(recs)} items, expected {expected}")
    if response.meta.candidates_considered != len(interpreted.candidates):
        check.violations.append("meta.candidates_considered does not match the shortlist")
    if len({r.source_id for r in recs}) != len(recs):
        check.duplicate_free = False
        check.violations.append("duplicate source_id in response")
    if [r.rank for r in recs] != list(range(1, len(recs) + 1)):
        check.violations.append("ranks are not 1..n")

    for item in recs:
        if item.source_id not in allowed:  # INV-01
            check.violations.append(f"{item.source_id} not in the filtered shortlist")
            continue
        check.id_valid += 1
        fact = store.get(item.source_id)
        if satisfies_hard_filters(store, fact, prefs, interpreted):  # INV-05
            check.constraint_ok += 1
        else:
            check.violations.append(f"{fact.name} violates a hard filter")
        if (item.name, item.cuisine, item.rating, item.cost_for_two, item.location) == (
            fact.name, fact.cuisine_display, fact.rating, fact.cost_for_two, fact.location_display
        ):  # INV-02
            check.fact_ok += 1
        else:
            check.violations.append(f"{item.source_id} facts differ from the store")
        if item.name and item.cuisine and item.estimated_cost and item.explanation.strip():
            check.complete += 1
        else:
            check.violations.append(f"{item.source_id} is missing a display field")
    return check
