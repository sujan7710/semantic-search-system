"""Baseline semantic cache: linear scan with one global similarity threshold.

This is intentionally the same algorithm as the original project (fixed 0.85
threshold, nearest cached query wins). Phase 1 benchmarks it as the baseline;
later phases replace the lookup and threshold policy.
"""

from __future__ import annotations

import itertools
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from cacheguard.embeddings import normalize_rows


@dataclass(frozen=True)
class CacheEntry:
    entry_id: int
    query: str
    embedding: np.ndarray
    response: Any
    cluster: int | None
    created_at: float


@dataclass(frozen=True)
class CacheLookup:
    hit: bool
    entry: CacheEntry | None
    similarity: float | None  # similarity to the nearest cached query; None if cache empty


class SemanticCache:
    def __init__(self, threshold: float = 0.85):
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be in (0, 1]")
        self.threshold = threshold
        self._lock = threading.Lock()
        self._ids = itertools.count(1)
        self._entries: list[CacheEntry] = []
        self._matrix: np.ndarray | None = None  # stacked embeddings, rebuilt lazily
        self._hits = 0
        self._misses = 0

    def __len__(self) -> int:
        return len(self._entries)

    def lookup(self, embedding: np.ndarray) -> CacheLookup:
        query = normalize_rows(embedding.reshape(1, -1))[0]
        with self._lock:
            if not self._entries:
                self._misses += 1
                return CacheLookup(hit=False, entry=None, similarity=None)
            if self._matrix is None:
                self._matrix = np.stack([e.embedding for e in self._entries])
            similarities = self._matrix @ query
            best = int(np.argmax(similarities))
            similarity = float(similarities[best])
            if similarity >= self.threshold:
                self._hits += 1
                return CacheLookup(hit=True, entry=self._entries[best], similarity=similarity)
            self._misses += 1
            return CacheLookup(hit=False, entry=None, similarity=similarity)

    def add(self, query: str, embedding: np.ndarray, response: Any, cluster: int | None = None) -> CacheEntry:
        entry = CacheEntry(
            entry_id=next(self._ids),
            query=query,
            embedding=normalize_rows(embedding.reshape(1, -1))[0],
            response=response,
            cluster=cluster,
            created_at=time.time(),
        )
        with self._lock:
            self._entries.append(entry)
            self._matrix = None
        return entry

    def stats(self) -> dict[str, float | int]:
        with self._lock:
            lookups = self._hits + self._misses
            return {
                "total_entries": len(self._entries),
                "hit_count": self._hits,
                "miss_count": self._misses,
                "hit_rate": self._hits / lookups if lookups else 0.0,
                "threshold": self.threshold,
            }

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._matrix = None
            self._hits = 0
            self._misses = 0
