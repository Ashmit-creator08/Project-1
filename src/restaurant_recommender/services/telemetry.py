"""In-process telemetry (Phase 4 task 3): counters, rates and latency percentiles.

Exposed at ``GET /metrics``. Holds no user free text and no secrets (CF-04, CF-05).
"""

from __future__ import annotations

import threading
import time
from collections import Counter, deque
from typing import Any

_PHASES = ("filter", "llm", "total")


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = (len(ordered) - 1) * pct / 100
    low, high = int(rank), min(int(rank) + 1, len(ordered) - 1)
    return round(ordered[low] + (ordered[high] - ordered[low]) * (rank - low), 1)


class Telemetry:
    def __init__(self, window: int = 1000) -> None:
        self._lock = threading.Lock()
        self._latency = {phase: deque(maxlen=window) for phase in _PHASES}
        self._counts: Counter[str] = Counter()
        self._fallback_reasons: Counter[str] = Counter()
        self._tokens: Counter[str] = Counter()
        self._rate_limit: dict[str, str] = {}
        self._started = time.time()

    def record_request(
        self,
        *,
        timings_ms: dict[str, float],
        empty: bool,
        cache_hit: bool,
        llm_used: bool,
        fallback_kind: str | None,
        llm_calls: int = 0,
        repaired: bool = False,
        dropped_ids: int = 0,
        backfilled: int = 0,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        rate_limit: dict[str, str] | None = None,
    ) -> None:
        with self._lock:
            self._counts["requests"] += 1
            self._counts["empty_results"] += int(empty)
            self._counts["cache_hits"] += int(cache_hit)
            self._counts["llm_rankings"] += int(llm_used)
            self._counts["llm_calls"] += llm_calls
            self._counts["repairs"] += int(repaired)
            self._counts["dropped_ids"] += dropped_ids
            self._counts["backfilled_items"] += backfilled
            if fallback_kind:
                self._counts["fallbacks"] += 1
                self._fallback_reasons[fallback_kind] += 1
            self._tokens["prompt"] += prompt_tokens
            self._tokens["completion"] += completion_tokens
            if rate_limit:
                self._rate_limit = dict(rate_limit)
            if not cache_hit:  # cached responses would skew phase latencies
                for phase in _PHASES:
                    if phase in timings_ms:
                        self._latency[phase].append(float(timings_ms[phase]))

    def record_rate_limited(self) -> None:
        with self._lock:
            self._counts["rate_limited"] += 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            counts = dict(self._counts)
            requests = counts.get("requests", 0)
            non_empty = requests - counts.get("empty_results", 0) - counts.get("cache_hits", 0)

            def rate(numerator: int, denominator: int) -> float | None:
                return round(numerator / denominator, 3) if denominator > 0 else None

            return {
                "uptime_seconds": round(time.time() - self._started, 1),
                "requests": requests,
                "rate_limited": counts.get("rate_limited", 0),
                "empty_results": counts.get("empty_results", 0),
                "cache": {"hits": counts.get("cache_hits", 0), "hit_rate": rate(counts.get("cache_hits", 0), requests)},
                "fallback": {
                    "count": counts.get("fallbacks", 0),
                    "rate": rate(counts.get("fallbacks", 0), non_empty),
                    "reasons": dict(self._fallback_reasons),
                },
                "llm": {
                    "rankings": counts.get("llm_rankings", 0),
                    "calls": counts.get("llm_calls", 0),
                    "repairs": counts.get("repairs", 0),
                    "dropped_ids": counts.get("dropped_ids", 0),
                    "backfilled_items": counts.get("backfilled_items", 0),
                    "prompt_tokens": self._tokens.get("prompt", 0),
                    "completion_tokens": self._tokens.get("completion", 0),
                    "rate_limit_remaining": dict(self._rate_limit),
                },
                "latency_ms": {
                    phase: {
                        "p50": percentile(list(values), 50),
                        "p95": percentile(list(values), 95),
                        "samples": len(values),
                    }
                    for phase, values in self._latency.items()
                },
            }
