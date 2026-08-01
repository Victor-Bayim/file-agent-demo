"""Small monotonic-clock sliding-window limits for one Web process."""

from __future__ import annotations

import time
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass
from math import ceil


@dataclass(frozen=True)
class RateLimitDecision:
    allowed: bool
    retry_after_seconds: int = 0


class SlidingWindowRateLimiter:
    """Deterministic in-memory limiter with an injectable monotonic clock."""

    def __init__(
        self,
        *,
        limit: int,
        window_seconds: float = 3600,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if limit <= 0 or window_seconds <= 0:
            raise ValueError("rate limits must be positive")
        self.limit = limit
        self.window_seconds = window_seconds
        self._clock = clock
        self._entries: dict[str, deque[float]] = defaultdict(deque)

    def check(self, key: str, *, consume: bool = False) -> RateLimitDecision:
        now = self._clock()
        entries = self._entries[key]
        cutoff = now - self.window_seconds
        while entries and entries[0] <= cutoff:
            entries.popleft()
        if len(entries) >= self.limit:
            retry_after = max(1, ceil(entries[0] + self.window_seconds - now))
            return RateLimitDecision(False, retry_after)
        if consume:
            entries.append(now)
        return RateLimitDecision(True)

    def discard(self, key: str) -> None:
        self._entries.pop(key, None)
