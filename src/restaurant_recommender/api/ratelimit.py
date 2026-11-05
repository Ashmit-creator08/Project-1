"""Per-client token-bucket rate limiter (Phase 4 task 5, edge case CF-08).

In-memory, so it limits one API process; that is enough for a single-instance
deployment and also protects the Groq quota. Use a gateway/Redis limiter when
running several replicas.
"""

from __future__ import annotations

import math
import threading
import time
from collections import OrderedDict
from collections.abc import Callable


class TokenBucketLimiter:
    def __init__(
        self,
        per_minute: int,
        burst: int | None = None,
        clock: Callable[[], float] = time.monotonic,
        max_clients: int = 10_000,
    ) -> None:
        self.per_minute = per_minute
        self.capacity = float(burst or per_minute)
        self.refill_per_second = per_minute / 60.0
        self._clock = clock
        self._max_clients = max_clients
        self._buckets: OrderedDict[str, tuple[float, float]] = OrderedDict()  # key -> (tokens, last)
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self.per_minute > 0

    def check(self, key: str) -> tuple[bool, int]:
        """Consume one token for `key`. Returns (allowed, retry_after_seconds)."""
        if not self.enabled:
            return True, 0
        now = self._clock()
        with self._lock:
            tokens, last = self._buckets.get(key, (self.capacity, now))
            tokens = min(self.capacity, tokens + (now - last) * self.refill_per_second)
            if tokens >= 1.0:
                self._buckets[key] = (tokens - 1.0, now)
                allowed, retry_after = True, 0
            else:
                self._buckets[key] = (tokens, now)
                allowed = False
                retry_after = max(1, math.ceil((1.0 - tokens) / self.refill_per_second))
            self._buckets.move_to_end(key)
            while len(self._buckets) > self._max_clients:
                self._buckets.popitem(last=False)
            return allowed, retry_after
