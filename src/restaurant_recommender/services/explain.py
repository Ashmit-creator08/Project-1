"""Display formatting and template (non-LLM) explanations and summaries."""

from __future__ import annotations

import re

from restaurant_recommender.data.budget import band_range_label
from restaurant_recommender.data.filter import FilterResult, cuisine_pattern, cuisine_terms
from restaurant_recommender.models.schemas import PreferenceRequest, Restaurant

EXAMPLE_LOCALITIES = "Koramangala, Indiranagar, HSR or Jayanagar"


def format_cost(cost: int | None) -> str:
    """'₹1,200 for two', or 'N/A' when unknown (edge case UI-07)."""
    return f"₹{cost:,} for two" if cost else "N/A"


def format_rating(rating: float | None) -> str:
    return f"{rating:.1f}" if rating is not None else "Not rated"


def matching_cuisines(
    restaurant: Restaurant, prefs: PreferenceRequest, terms: list[str] | None = None
) -> list[str]:
    terms = terms if terms is not None else cuisine_terms(prefs.cuisine)
    if not terms:
        return []
    pattern = re.compile(cuisine_pattern(terms))
    return [c for c in restaurant.cuisines if pattern.search(c.lower())]


def template_explanation(
    restaurant: Restaurant, prefs: PreferenceRequest, cuisine_terms_used: list[str] | None = None
) -> str:
    """Fact-only explanation used when the LLM is unavailable or skipped an item."""
    if restaurant.rating is not None:
        first = f"Rated {restaurant.rating:.1f}/5 by {restaurant.votes:,} diners"
    else:
        first = "Not yet rated on Zomato"

    matched = matching_cuisines(restaurant, prefs, cuisine_terms_used)
    serves = ", ".join(matched or restaurant.cuisines[:3])
    where = f" in {restaurant.locality}" if restaurant.locality else ""
    first += f", serving {serves}{where}." if serves else f"{where}."

    second = []
    if restaurant.rest_type:
        second.append(restaurant.rest_type)
    if restaurant.cost_for_two:
        cost = f"₹{restaurant.cost_for_two:,} for two"
        if prefs.budget is not None:
            cost += f", within your {prefs.budget.value} budget"
        second.append(cost)
    extras = []
    if restaurant.book_table:
        extras.append("table booking")
    if restaurant.online_order:
        extras.append("online ordering")
    sentence = " · ".join(second)
    if extras:
        sentence = (sentence + "; " if sentence else "") + " and ".join(extras) + " available"
    return first + (f" {sentence}." if sentence else "")


def fallback_summary(count: int, prefs: PreferenceRequest, *, configured: bool) -> str:
    cause = "AI ranking is unavailable right now" if configured else "AI ranking is off (no LLM configured)"
    picks = "this is the highest-rated match" if count == 1 else f"these are the {count} highest-rated matches"
    text = f"{cause}, so {picks} for your filters."
    if prefs.additional_preferences:
        text += " Your additional preferences were not applied."
    return text


def ranked_summary(count: int, prefs: PreferenceRequest) -> str:
    focus = " ".join(p for p in (prefs.cuisine, "restaurant" if count == 1 else "restaurants") if p)
    return f"Top {count} {focus} in {prefs.location}, ranked for your preferences."


def empty_summary(prefs: PreferenceRequest, result: FilterResult) -> str:
    """Explain which hard filter emptied the result (edge case FL-01)."""
    stage = result.eliminated_by
    if stage == "location":
        return (
            f"No restaurants found for “{prefs.location}”. This dataset covers Bangalore: try "
            f"“Bangalore” or a locality such as {EXAMPLE_LOCALITIES}."
        )
    if stage == "cuisine":
        return (
            f"No {prefs.cuisine} restaurants found in {prefs.location}. Try a broader cuisine "
            "or a nearby locality."
        )
    if stage == "budget" and prefs.budget is not None:
        return (
            f"No matches at a {prefs.budget.value} budget ({band_range_label(prefs.budget)}). "
            "Try another budget."
        )
    if stage == "min_rating":
        return f"No matches rated {prefs.min_rating:g} or higher. Try lowering the minimum rating."
    return "No restaurants matched your preferences."
