"""Protection for a public demo backed by a paid API.

* ``RateLimiter``: at most N requests per client per rolling minute.
* ``DailyBudget``: at most N backend (LLM) calls per UTC day. Cache hits do
  not count, so once the budget is spent, cached answers keep working.
In-memory and per process: enough for a single-instance demo, not a fleet.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from datetime import datetime, timezone
from typing import Callable


class RateLimiter:
    def __init__(self, per_minute: int, clock: Callable[[], float] = time.monotonic, max_clients: int = 10_000):
        if per_minute <= 0:
            raise ValueError("per_minute must be positive")
        self.per_minute = per_minute
        self._clock = clock
        self._max_clients = max_clients
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def check(self, client: str) -> float | None:
        """Record a request. Returns None if allowed, else seconds until a slot frees up."""
        now = self._clock()
        with self._lock:
            window = self._hits.setdefault(client, deque())
            while window and now - window[0] >= 60:
                window.popleft()
            if len(window) >= self.per_minute:
                return max(0.0, 60 - (now - window[0]))
            window.append(now)
            if len(self._hits) > self._max_clients:  # forget idle clients
                for key in [k for k, w in self._hits.items() if not w or now - w[-1] >= 60]:
                    del self._hits[key]
            return None


class BudgetExceeded(RuntimeError):
    pass


class DailyBudget:
    def __init__(self, limit: int, today: Callable[[], str] | None = None):
        if limit <= 0:
            raise ValueError("limit must be positive")
        self.limit = limit
        self._today = today or (lambda: datetime.now(timezone.utc).date().isoformat())
        self._day = self._today()
        self._used = 0
        self._lock = threading.Lock()

    def _roll(self) -> None:
        day = self._today()
        if day != self._day:
            self._day, self._used = day, 0

    def consume(self) -> None:
        with self._lock:
            self._roll()
            if self._used >= self.limit:
                raise BudgetExceeded(
                    f"the demo's daily limit of {self.limit} AI calls is used up; cached answers still work"
                )
            self._used += 1

    def refund(self) -> None:
        """Give a call back when the backend failed before doing any work."""
        with self._lock:
            self._used = max(0, self._used - 1)

    def status(self) -> dict:
        with self._lock:
            self._roll()
            return {"day": self._day, "limit": self.limit, "used": self._used, "remaining": self.limit - self._used}
