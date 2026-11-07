import threading

import numpy as np
import pytest

from cacheguard.artifacts import ArtifactError, build_artifacts, load_artifacts, save_artifacts
from cacheguard.cache import SemanticCache
from cacheguard.clustering import SoftClusterer
from cacheguard.embeddings import normalize_rows
from cacheguard.vector_index import VectorIndex


def unit(*values):
    return normalize_rows(np.array([values], dtype=np.float32))[0]


# --- vector index ---------------------------------------------------------

def test_index_returns_cosine_scores_best_first():
    index = VectorIndex.from_embeddings(np.stack([unit(1, 0), unit(0, 1), unit(1, 1)]))
    scores, ids = index.search(unit(1, 0.1), k=3)
    assert ids.tolist() == [0, 2, 1]
    assert scores[0] == pytest.approx(0.995, abs=1e-3)
    assert np.all(np.diff(scores) <= 0)


def test_index_clamps_k_and_handles_empty():
    index = VectorIndex(2)
    assert index.search(unit(1, 0), k=5)[1].size == 0
    index.add(np.stack([unit(1, 0)]))
    assert index.search(unit(1, 0), k=5)[1].tolist() == [0]


def test_index_rejects_wrong_dimension():
    with pytest.raises(ValueError):
        VectorIndex(3).add(np.zeros((1, 2), dtype=np.float32))


# --- cache ------------------------------------------------------------------

def test_empty_cache_misses_with_no_similarity():
    lookup = SemanticCache().lookup(unit(1, 0))
    assert not lookup.hit and lookup.similarity is None


def test_cache_hits_at_or_above_threshold_only():
    cache = SemanticCache(threshold=0.9)
    cache.add("q", unit(1, 0), response="answer")
    assert cache.lookup(unit(1, 0)).hit
    near = cache.lookup(unit(1, 0.4))  # cos ~= 0.93
    far = cache.lookup(unit(1, 1))  # cos ~= 0.71
    assert near.hit and near.entry.response == "answer"
    assert not far.hit and far.similarity == pytest.approx(0.7071, abs=1e-3)


def test_cache_picks_nearest_entry():
    cache = SemanticCache(threshold=0.5)
    cache.add("x", unit(1, 0), "x-answer")
    cache.add("y", unit(0, 1), "y-answer")
    assert cache.lookup(unit(0.1, 1)).entry.query == "y"


def test_cache_stats_and_clear():
    cache = SemanticCache(threshold=0.9)
    cache.lookup(unit(1, 0))
    cache.add("q", unit(1, 0), "a")
    cache.lookup(unit(1, 0))
    assert cache.stats() == {"total_entries": 1, "hit_count": 1, "miss_count": 1, "hit_rate": 0.5, "threshold": 0.9}
    cache.clear()
    assert cache.stats()["total_entries"] == 0 and cache.stats()["hit_count"] == 0


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.5])
def test_cache_rejects_invalid_threshold(bad):
    with pytest.raises(ValueError):
        SemanticCache(threshold=bad)


def test_cache_is_safe_under_concurrent_use():
    cache = SemanticCache(threshold=0.99)
    rng = np.random.default_rng(0)

    def worker():
        for _ in range(200):
            v = rng.normal(size=8).astype(np.float32)
            if not cache.lookup(v).hit:
                cache.add("q", v, "a")

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    stats = cache.stats()
    assert stats["hit_count"] + stats["miss_count"] == 800


# --- clustering and artifacts ------------------------------------------------

def test_clusterer_save_load_roundtrip_without_pickle(tmp_path):
    rng = np.random.default_rng(0)
    data = np.vstack([rng.normal(0, 1, (50, 4)), rng.normal(6, 1, (50, 4))])
    clusterer = SoftClusterer(n_clusters=2, covariance_type="full").fit(data)
    clusterer.save(tmp_path / "c.npz")
    loaded = SoftClusterer.load(tmp_path / "c.npz")
    np.testing.assert_allclose(loaded.predict_proba(data), clusterer.predict_proba(data))


def test_artifacts_roundtrip_and_model_mismatch(tmp_path, embedder, documents):
    built = build_artifacts(documents, embedder, n_clusters=2, covariance_type="diag", random_seed=0)
    save_artifacts(built, tmp_path)
    loaded = load_artifacts(tmp_path, expected_model=embedder.model_name)
    assert loaded.documents == documents
    np.testing.assert_array_equal(loaded.embeddings, built.embeddings)
    with pytest.raises(ArtifactError):
        load_artifacts(tmp_path, expected_model="some-other-model")


def test_missing_artifacts_give_a_helpful_error(tmp_path):
    with pytest.raises(ArtifactError, match="cacheguard.build"):
        load_artifacts(tmp_path / "nope")
