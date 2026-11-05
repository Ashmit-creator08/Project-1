"""Short-lived response cache (Phase 4 task 1).

Identical preferences within ``CACHE_TTL_SECONDS`` reuse the previous ranking
instead of spending another LLM call (important under Groq's per-minute limits).

The key covers everything that can change the answer (edge cases CF-06, CF-07):
normalized preferences, the dataset version, the prompt variant and the model.
Only successful LLM rankings and empty results are cached; fallback responses
are not, so the app recovers as soon as the provider does.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from typing import Generic, TypeVar

from restaurant_recommender.data.filter import cuisine_terms
from restaurant_recommender.data.locations import location_key, split_location
from restaurant_recommender.models.schemas import PreferenceRequest, collapse_ws

V = TypeVar("V")


class TTLCache(Generic[V]):
    """Thread-safe LRU cache whose entries expire after `ttl_seconds`."""

    def __init__(self, max_entries: int, ttl_seconds: float, clock: Callable[[], float] = time.monotonic) -> None:
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self._clock = clock
        self._data: OrderedDict[str, tuple[float, V]] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> V | None:
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                self.misses += 1
                return None
            expires_at, value = entry
            if self._clock() >= expires_at:
                del self._data[key]
                self.misses += 1
                return None
            self._data.move_to_end(key)
            self.hits += 1
            return value

    def set(self, key: str, value: V) -> None:
        with self._lock:
            self._data[key] = (self._clock() + self.ttl_seconds, value)
            self._data.move_to_end(key)
            while len(self._data) > self.max_entries:
                self._data.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)


def cache_key(prefs: PreferenceRequest, *, data_version: str, prompt_variant: str, model: str | None) -> str:
    """Stable key: case/spacing/order differences in the request map to the same key (CF-07)."""
    normalized = {
        "location": sorted(location_key(part) for part in split_location(prefs.location)),
        "cuisine": sorted(set(cuisine_terms(prefs.cuisine))),
        "budget": prefs.budget.value if prefs.budget else None,
        "min_rating": round(prefs.min_rating or 0.0, 1),
        "notes": collapse_ws(prefs.additional_preferences).lower(),
        "top_n": prefs.top_n,
        "data": data_version,
        "prompt": prompt_variant,
        "model": model,
    }
    blob = json.dumps(normalized, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()
