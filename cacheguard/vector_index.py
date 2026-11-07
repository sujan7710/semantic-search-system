"""Exact nearest-neighbour search over unit vectors with FAISS.

Embeddings are unit-length, so inner product equals cosine similarity. Using
IndexFlatIP (instead of the original IndexFlatL2) gives the same ranking but
returns scores in [-1, 1] that are directly comparable to cache thresholds.
"""

from __future__ import annotations

import faiss
import numpy as np


class VectorIndex:
    def __init__(self, dim: int):
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim
        self._index = faiss.IndexFlatIP(dim)

    @classmethod
    def from_embeddings(cls, embeddings: np.ndarray) -> VectorIndex:
        embeddings = np.asarray(embeddings)
        if embeddings.ndim != 2:
            raise ValueError("embeddings must be a 2-D array")
        index = cls(embeddings.shape[1])
        index.add(embeddings)
        return index

    def __len__(self) -> int:
        return int(self._index.ntotal)

    def add(self, embeddings: np.ndarray) -> None:
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        if embeddings.ndim != 2 or embeddings.shape[1] != self.dim:
            raise ValueError(f"expected shape (n, {self.dim}), got {embeddings.shape}")
        self._index.add(embeddings)

    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (scores, ids) of the k most similar vectors, best first."""
        query = np.ascontiguousarray(query, dtype=np.float32).reshape(1, -1)
        if query.shape[1] != self.dim:
            raise ValueError(f"query has dim {query.shape[1]}, index has {self.dim}")
        k = min(k, len(self))
        if k <= 0:
            return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.int64)
        scores, ids = self._index.search(query, k)
        return scores[0], ids[0]
