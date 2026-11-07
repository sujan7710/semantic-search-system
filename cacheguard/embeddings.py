"""Text embedders. Every embedder returns float32, unit-length rows."""

from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np


class Embedder(Protocol):
    model_name: str
    dim: int

    def encode(self, texts: Sequence[str], show_progress: bool = False) -> np.ndarray:
        """Return an (n, dim) float32 array of L2-normalized embeddings."""
        ...


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str, batch_size: int = 64, device: str | None = None):
        from sentence_transformers import SentenceTransformer  # heavy import, load lazily

        self.model_name = model_name
        self.batch_size = batch_size
        self._model = SentenceTransformer(model_name, device=device)
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def encode(self, texts: Sequence[str], show_progress: bool = False) -> np.ndarray:
        embeddings = self._model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
        )
        return np.asarray(embeddings, dtype=np.float32)


def normalize_rows(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize rows; zero rows stay zero instead of becoming NaN."""
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.where(norms == 0, 1.0, norms)
