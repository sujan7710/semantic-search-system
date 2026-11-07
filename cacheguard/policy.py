"""Threshold policies: how similar is similar enough to reuse an answer?

Phase 2 ships only the global threshold (same behavior as the baseline).
Phase 3 adds calibrated per-cluster and per-entry policies behind this
interface, so the store does not change.
"""

from __future__ import annotations

from typing import Protocol


class ThresholdPolicy(Protocol):
    def threshold(self, partition: str, cluster: int | None) -> float:
        """Minimum cosine similarity for the candidate entry to count as a hit."""
        ...


class GlobalThreshold:
    def __init__(self, value: float = 0.85):
        if not 0.0 < value <= 1.0:
            raise ValueError("threshold must be in (0, 1]")
        self.value = value

    def threshold(self, partition: str, cluster: int | None) -> float:
        return self.value

    def describe(self) -> dict:
        return {"type": "global", "threshold": self.value}
