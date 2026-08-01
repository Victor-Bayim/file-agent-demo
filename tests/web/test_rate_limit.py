from __future__ import annotations

from app.web.rate_limit import SlidingWindowRateLimiter


def test_sliding_window_uses_injected_monotonic_clock() -> None:
    now = [10.0]
    limiter = SlidingWindowRateLimiter(limit=2, window_seconds=60, clock=lambda: now[0])

    assert limiter.check("client", consume=True).allowed
    assert limiter.check("client", consume=True).allowed
    denied = limiter.check("client")
    assert denied.allowed is False
    assert denied.retry_after_seconds == 60

    now[0] = 70.1
    assert limiter.check("client", consume=True).allowed


def test_rate_limit_keys_are_independent() -> None:
    limiter = SlidingWindowRateLimiter(limit=1, clock=lambda: 1.0)

    assert limiter.check("one", consume=True).allowed
    assert not limiter.check("one").allowed
    assert limiter.check("two").allowed
