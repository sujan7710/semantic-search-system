"""A stand-in LLM for load tests: fixed delay, fixed token usage, no network.

Lets a load test measure CacheGuard itself (embedding, lookup, verifier,
locking) without spending API credit or tripping provider rate limits.
"""

from __future__ import annotations

import time
from typing import Sequence

import numpy as np

from cacheguard.backends import BackendResponse


class MockLLMBackend:
    name = "mock"

    def __init__(self, delay_ms: float = 600.0, prompt_tokens: int = 100, completion_tokens: int = 300,
                 price_input_per_m: float = 0.075, price_output_per_m: float = 0.30):
        self.delay_ms = delay_ms
        self.usage = {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                      "total_tokens": prompt_tokens + completion_tokens}
        self.cost = (prompt_tokens * price_input_per_m + completion_tokens * price_output_per_m) / 1e6

    def cache_settings(self) -> dict:
        return {"model": "mock", "delay_ms": self.delay_ms}

    def respond(self, query: str, query_embedding: np.ndarray | None = None, history: Sequence[dict] = ()) -> BackendResponse:
        time.sleep(self.delay_ms / 1000)  # sync endpoints run in a thread pool, like a blocking HTTP call
        return BackendResponse(
            answer=f"(mock answer) {query}",
            meta={"model": "mock", "usage": dict(self.usage), "cost_usd": self.cost, "latency_ms": self.delay_ms},
        )
