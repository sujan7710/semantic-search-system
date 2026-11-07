import hashlib
import re

import numpy as np
import pytest

from cacheguard.backends.newsgroups import NewsgroupsSearchBackend
from cacheguard.cache import SemanticCache
from cacheguard.clustering import SoftClusterer
from cacheguard.data.newsgroups import Document
from cacheguard.embeddings import normalize_rows
from cacheguard.service import CacheGuardService
from cacheguard.vector_index import VectorIndex


class FakeEmbedder:
    """Deterministic bag-of-words hashing embedder: no model download in tests."""

    model_name = "fake-hash-embedder"

    def __init__(self, dim: int = 64):
        self.dim = dim

    def encode(self, texts, show_progress=False):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in re.findall(r"\w+", text.lower()):
                bucket = int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dim
                out[row, bucket] += 1.0
        return normalize_rows(out)


CORPUS = [
    ("sci.space", "The shuttle launch was delayed because of weather at the launch pad."),
    ("sci.space", "NASA plans a new orbital mission to study the moon surface."),
    ("rec.autos", "My car engine makes a strange noise when I accelerate on the highway."),
    ("rec.autos", "Looking for advice on changing brake pads on an old sedan."),
    ("sci.med", "Doctors recommend regular exercise to lower blood pressure."),
    ("sci.med", "The new vaccine trial showed promising results in patients."),
]


@pytest.fixture
def embedder():
    return FakeEmbedder()


@pytest.fixture
def documents():
    return [Document(i, group, text, f"mem://{i}") for i, (group, text) in enumerate(CORPUS)]


@pytest.fixture
def service(embedder, documents):
    embeddings = embedder.encode([d.text for d in documents])
    backend = NewsgroupsSearchBackend(documents, VectorIndex.from_embeddings(embeddings), top_k=3)
    clusterer = SoftClusterer(n_clusters=2, covariance_type="diag").fit(embeddings)
    return CacheGuardService(embedder, SemanticCache(threshold=0.85), backend, clusterer)
