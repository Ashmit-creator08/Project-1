"""Deterministic preference filters (architecture §5.3, Phase 4 matching upgrades).

Hard filters, applied in this order; each runs only when the request sets it:

1. **location**   matched against locality, Zomato listing zones and city (not
                  the street address, whose landmarks cause false hits) using
                  normalized keys ("HSR Layout" = "HSR", "Indira Nagar" =
                  "Indiranagar", "Bengaluru" = "Bangalore"). A term with no
                  match falls back to the closest known locality (fuzzy, e.g.
                  "Koramangla" -> Koramangala). Comma-separated parts must all
                  match ("Indiranagar, Bangalore"). See `data/locations.py`.
2. **cuisine**    whole-word, case-insensitive match against the cuisine list.
                  "Indian" matches "North Indian"; "Tea" does not match "Steak".
                  Several cuisines separated by commas match if *any* matches.
                  A few dish words expand to cuisines ("pizza" -> Pizza or
                  Italian) and misspellings are corrected ("Itallian").
3. **budget**     exact `budget_band` match. Restaurants without a known cost
                  never match (edge case DI-07).
4. **min_rating** `rating >= min_rating` (inclusive, FL-07). Unrated restaurants
                  never match. 0 or None means "no rating filter".

Every correction or expansion is reported in `FilterResult.notes`.
`additional_preferences` is never used here; it is a soft signal for the LLM.

Matches are ordered by rating (desc, unrated last), votes (desc), name, id so
truncation to `max_candidates` keeps the best-rated rows deterministically
(edge cases FL-09, FL-11).
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field

import numpy as np

from restaurant_recommender.data.locations import split_location
from restaurant_recommender.data.store import RestaurantStore
from restaurant_recommender.models.schemas import PreferenceRequest, Restaurant

# Dish words that imply cuisines (edge case FL-04). Only terms that exist in
# the dataset's cuisine vocabulary are used.
CUISINE_SYNONYMS: dict[str, list[str]] = {
    "pizza": ["pizza", "italian"],
    "pasta": ["italian"],
    "sushi": ["sushi", "japanese"],
    "ramen": ["japanese"],
    "noodles": ["chinese", "asian"],
    "dim sum": ["chinese"],
    "dimsum": ["chinese"],
    "momo": ["momos"],
    "dosa": ["south indian"],
    "idli": ["south indian"],
    "taco": ["mexican"],
    "tacos": ["mexican"],
    "burrito": ["mexican"],
    "burgers": ["burger"],
    "shawarma": ["lebanese", "arabian", "middle eastern"],
    "kebab": ["kebab", "mughlai"],
    "kebabs": ["kebab", "mughlai"],
    "coffee": ["cafe", "beverages"],
    "dessert": ["desserts"],
}
CUISINE_FUZZY_CUTOFF = 0.8

_SPLIT_CUISINES = re.compile(r"[,/|;&]")
_RATING_EPSILON = 1e-9
SORT_COLUMNS = ["rating", "votes", "name", "id"]
SORT_ASCENDING = [False, False, True, True]


@dataclass
class FilterResult:
    candidates: list[Restaurant]
    total_matches: int
    filters_applied: list[str]
    filter_counts: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    location_label: str | None = None  # location as interpreted (after any correction)
    cuisine_terms: list[str] = field(default_factory=list)  # cuisine terms actually matched on

    @property
    def truncated(self) -> bool:
        return self.total_matches > len(self.candidates)

    @property
    def eliminated_by(self) -> str | None:
        """First filter after which nothing was left (for empty-state messages)."""
        for name in self.filters_applied:
            if self.filter_counts.get(name, 0) == 0:
                return name
        return None


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def cuisine_terms(cuisine: str | None) -> list[str]:
    if not cuisine:
        return []
    return [t for t in (normalize(p) for p in _SPLIT_CUISINES.split(cuisine)) if t]


def cuisine_pattern(terms: list[str]) -> str:
    return r"(?<!\w)(?:" + "|".join(re.escape(t) for t in terms) + r")(?!\w)"


def resolve_cuisines(terms: list[str], vocabulary: list[str]) -> tuple[list[str], list[str]]:
    """Expand dish synonyms and correct misspellings against known cuisines.

    Returns (terms to match on, user-facing notes).
    """
    display = {c.lower(): c for c in vocabulary}

    def known(term: str) -> bool:
        pattern = re.compile(cuisine_pattern([term]))
        return any(pattern.search(v) for v in display)

    resolved: list[str] = []
    notes: list[str] = []
    for term in terms:
        if term in CUISINE_SYNONYMS:
            expanded = [t for t in dict.fromkeys([term, *CUISINE_SYNONYMS[term]]) if known(t)]
            if expanded:
                resolved += expanded
                extra = [display.get(t, t.title()) for t in expanded if t != term]
                if extra:
                    notes.append(f"“{term}” also matches {' and '.join(extra)} restaurants.")
                continue
        if known(term):
            resolved.append(term)
            continue
        close = difflib.get_close_matches(term, list(display), n=1, cutoff=CUISINE_FUZZY_CUTOFF)
        if close:
            resolved.append(close[0])
            notes.append(f"Showing {display[close[0]]} (closest match to “{term}”).")
        else:
            resolved.append(term)  # matches nothing -> empty result with a clear message
    return list(dict.fromkeys(resolved)), notes


def filter_restaurants(
    store: RestaurantStore,
    prefs: PreferenceRequest,
    max_candidates: int,
) -> FilterResult:
    df = store.frame
    counts: dict[str, int] = {"all": len(df)}
    applied: list[str] = []
    notes: list[str] = []
    mask: np.ndarray | None = None

    def apply(name: str, condition: np.ndarray) -> None:
        nonlocal mask
        mask = condition if mask is None else (mask & condition)
        applied.append(name)
        counts[name] = int(mask.sum())

    location_label = None
    parts = split_location(prefs.location)
    if parts:
        condition = np.ones(len(df), dtype=bool)
        labels = []
        for part in parts:
            match = store.locations.resolve(part)
            condition &= store.locations.mask(match.names)
            labels.append(match.label)
            if match.fuzzy:
                notes.append(f"Showing results for {match.label} (closest match to “{match.query}”).")
        location_label = ", ".join(labels)
        apply("location", condition)

    resolved_cuisines: list[str] = []
    raw_cuisines = cuisine_terms(prefs.cuisine)
    if raw_cuisines:
        resolved_cuisines, cuisine_notes = resolve_cuisines(raw_cuisines, store.cuisines())
        notes += cuisine_notes
        hits = df["_cuisine_search"].str.contains(cuisine_pattern(resolved_cuisines), regex=True)
        apply("cuisine", hits.to_numpy(dtype=bool))

    if prefs.budget is not None:
        apply("budget", (df["budget_band"] == prefs.budget.value).fillna(False).to_numpy(dtype=bool))

    if prefs.min_rating:
        apply("min_rating", (df["rating"] >= prefs.min_rating - _RATING_EPSILON).fillna(False).to_numpy(dtype=bool))

    matched = df if mask is None else df[mask]
    total = len(matched)
    shortlist = matched.sort_values(SORT_COLUMNS, ascending=SORT_ASCENDING, na_position="last")
    shortlist = shortlist.head(max_candidates)

    return FilterResult(
        candidates=store.to_restaurants(shortlist),
        total_matches=total,
        filters_applied=applied,
        filter_counts=counts,
        notes=notes,
        location_label=location_label,
        cuisine_terms=resolved_cuisines,
    )
