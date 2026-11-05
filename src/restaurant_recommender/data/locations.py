"""Location normalization and fuzzy matching (Phase 4 task 2, edge cases FL-02 / FL-03).

Matching works on *keys*: lower-case, alphanumeric only, with a few token
rewrites applied identically to queries and to dataset names::

    "HSR Layout"     -> "hsr"            ("layout" dropped)
    "Indira Nagar"   -> "indiranagar"    (spaces ignored)
    "M.G. Rd"        -> "mgroad"         ("rd" -> "road")
    "Church St"      -> "churchstreet"   ("st" -> "street")
    "Bengaluru"      -> "bangalore"      (city alias)

A query term first matches every locality / Zomato zone / city whose key
*contains* the query key. Only when nothing matches is a fuzzy match tried
(difflib ratio >= FUZZY_CUTOFF), against both full names and block-less base
names, so "Koramangla" resolves to all Koramangala blocks. Only the single best
fuzzy match is used, and the caller reports the correction to the user.
"""

from __future__ import annotations

import difflib
import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

CITY = "bangalore"
CITY_ALIASES = {
    "bengaluru": CITY,
    "bangaluru": CITY,
    "banglore": CITY,
    "bengalore": CITY,
    "blr": CITY,
}
_TOKEN_REWRITES = {"rd": "road", "st": "street", "layout": ""}
_NON_ALNUM = re.compile(r"[^0-9a-z]+")
_BLOCK_SUFFIX_KEY = re.compile(r"(?:\d+(?:st|nd|rd|th)?(?:block|stage|phase|sector)|(?:block|stage|phase|sector)\d+)$")
_BLOCK_SUFFIX_NAME = re.compile(r"\s+(?:\d+(?:st|nd|rd|th)?\s+(?:block|stage|phase|sector)|(?:block|stage|phase|sector)\s+\d+)$", re.I)
FUZZY_CUTOFF = 0.82


def tokens(text: str) -> list[str]:
    words = _NON_ALNUM.split(text.lower().replace("&", " and "))
    out = []
    for word in words:
        if not word:
            continue
        word = CITY_ALIASES.get(word, word)
        word = _TOKEN_REWRITES.get(word, word)
        if word:
            out.append(word)
    return out


def location_key(text: str) -> str:
    parts = tokens(text)
    # "Koramangala Bangalore" -> the city is implied; keep the specific part.
    if len(parts) > 1 and CITY in parts:
        parts = [p for p in parts if p != CITY]
    return "".join(parts)


def split_location(location: str) -> list[str]:
    """Comma-separated parts; each must match (\"Indiranagar, Bangalore\")."""
    return [p.strip() for p in location.split(",") if p.strip()]


@dataclass
class LocationMatch:
    query: str
    names: list[str]  # canonical dataset names that match
    fuzzy: bool = False
    label: str = ""  # what we searched for, e.g. "Koramangala" after a correction


class LocationIndex:
    """Maps location names to row positions and resolves free-text queries."""

    def __init__(self, row_names: Sequence[Sequence[str]]) -> None:
        positions: dict[str, list[int]] = defaultdict(list)
        for row, names in enumerate(row_names):
            for name in {n for n in names if n}:
                positions[name].append(row)
        self._rows = len(row_names)
        self._positions = {name: np.asarray(pos, dtype=np.int64) for name, pos in positions.items()}
        self._keys = {name: location_key(name) for name in self._positions}

        # Fuzzy vocabulary: full keys plus block-less base keys -> display label.
        self._vocab: dict[str, str] = {}
        for name, key in self._keys.items():
            if key:
                self._vocab.setdefault(key, name)
                base = _BLOCK_SUFFIX_KEY.sub("", key)
                if base and base != key:
                    self._vocab.setdefault(base, _BLOCK_SUFFIX_NAME.sub("", name))

    @property
    def names(self) -> list[str]:
        return sorted(self._positions)

    def resolve(self, term: str) -> LocationMatch:
        key = location_key(term)
        if not key:
            return LocationMatch(term, [], label=term)
        exact = sorted(n for n, k in self._keys.items() if key in k)
        if exact:
            return LocationMatch(term, exact, fuzzy=False, label=term)
        close = difflib.get_close_matches(key, list(self._vocab), n=1, cutoff=FUZZY_CUTOFF)
        if not close:
            return LocationMatch(term, [], label=term)
        best = close[0]
        names = sorted(n for n, k in self._keys.items() if best in k)
        return LocationMatch(term, names, fuzzy=True, label=self._vocab[best])

    def mask(self, names: Sequence[str]) -> np.ndarray:
        result = np.zeros(self._rows, dtype=bool)
        for name in names:
            positions = self._positions.get(name)
            if positions is not None:
                result[positions] = True
        return result
