"""The request path.

    embed -> nearest cached question (same partition)
          -> decide: guard (gate, rules, verifier) or plain threshold
          -> hit: return cached answer | miss: ask backend, store the answer
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from cacheguard.backends import Backend, BackendResponse
from cacheguard.clustering import SoftClusterer
from cacheguard.config import Settings
from cacheguard.embeddings import Embedder
from cacheguard.guard import Guard
from cacheguard.partitions import partition_key
from cacheguard.store import CacheStore


@dataclass(frozen=True)
class DecisionInfo:
    stage: str  # none | gate | rules | similarity | verifier | threshold
    reason: str  # empty_cache | below_gate | negation | number | name | high_similarity | verified | ...
    verifier_score: float | None = None


@dataclass(frozen=True)
class QueryResult:
    query: str
    namespace: str
    cache_hit: bool
    entry_id: int  # the cache entry that answered (hit) or was created (miss)
    matched_query: str | None
    cache_similarity: float | None  # nearest cached query in the partition, hit or not
    threshold: float | None  # similarity gate applied to that nearest entry
    nearest_query: str | None  # nearest cached query, shown even on a miss
    decision: DecisionInfo
    dominant_cluster: int | None
    response: BackendResponse
    latency_ms: float
    decision_ms: float  # time spent deciding (includes the verifier when it runs)


class CacheGuardService:
    def __init__(
        self,
        embedder: Embedder,
        store: CacheStore,
        backend: Backend,
        clusterer: SoftClusterer | None = None,
        guard: Guard | None = None,
    ):
        self.embedder = embedder
        self.store = store
        self.backend = backend
        self.clusterer = clusterer
        self.guard = guard

    def _decide(self, partition: str, query: str, embedding, nearest):
        """Returns (hit, threshold, DecisionInfo). Runs outside the store lock."""
        if nearest is None:
            return False, None, DecisionInfo("none", "empty_cache")
        entry, similarity = nearest
        if self.guard is not None:
            d = self.guard.decide(query, embedding, entry.query, similarity)
            return d.hit, d.gate, DecisionInfo(d.stage, d.reason, d.verifier_score)
        threshold = self.store.policy.threshold(partition, entry.cluster)
        hit = similarity >= threshold
        return hit, threshold, DecisionInfo("threshold", "above_threshold" if hit else "below_threshold")

    def handle(self, query: str, namespace: str = "default") -> QueryResult:
        start = time.perf_counter()
        query = query.strip()
        if not query:
            raise ValueError("query must not be empty")

        partition = partition_key(namespace, self.backend.name)
        embedding = self.embedder.encode([query])[0]
        nearest = self.store.nearest(partition, embedding)
        decide_start = time.perf_counter()
        hit, threshold, info = self._decide(partition, query, embedding, nearest)
        decision_ms = (time.perf_counter() - decide_start) * 1000
        entry, similarity = nearest if nearest else (None, None)
        self.store.record(partition, entry if hit else None, hit=hit, reason=info.reason)

        common = dict(
            query=query,
            namespace=namespace,
            cache_similarity=similarity,
            threshold=threshold,
            nearest_query=entry.query if entry else None,
            decision=info,
            decision_ms=decision_ms,
        )
        if hit:
            return QueryResult(
                cache_hit=True,
                entry_id=entry.entry_id,
                matched_query=entry.query,
                dominant_cluster=entry.cluster,
                response=BackendResponse.from_dict(entry.response),
                latency_ms=(time.perf_counter() - start) * 1000,
                **common,
            )

        response = self.backend.respond(query, embedding)
        cluster = None
        if self.clusterer is not None and self.clusterer.is_fitted:
            cluster = int(self.clusterer.predict(embedding)[0])
        # Tags let one call invalidate every answer built from a given source document,
        # or everything in a namespace.
        tags = [f"namespace:{namespace}", *(f"source:{s.doc_id}" for s in response.sources)]
        created = self.store.add(partition, query, embedding, response.to_dict(), cluster, tags)
        return QueryResult(
            cache_hit=False,
            entry_id=created.entry_id,
            matched_query=None,
            dominant_cluster=cluster,
            response=response,
            latency_ms=(time.perf_counter() - start) * 1000,
            **common,
        )


def build_default_service(settings: Settings) -> CacheGuardService:
    """Wire up the real model, prebuilt index, persistent cache, guard and newsgroups backend."""
    from pathlib import Path

    import faiss

    from cacheguard.artifacts import load_artifacts
    from cacheguard.backends.newsgroups import NewsgroupsSearchBackend
    from cacheguard.embeddings import SentenceTransformerEmbedder
    from cacheguard.policy import GlobalThreshold
    from cacheguard.vector_index import VectorIndex

    faiss.omp_set_num_threads(settings.faiss_threads)
    artifacts = load_artifacts(settings.artifact_dir, expected_model=settings.embedding_model)
    embedder = SentenceTransformerEmbedder(settings.embedding_model)
    backend = NewsgroupsSearchBackend(
        artifacts.documents,
        VectorIndex.from_embeddings(artifacts.embeddings),
        top_k=settings.top_k,
        snippet_chars=settings.snippet_chars,
    )
    store = CacheStore(
        dim=embedder.dim,
        policy=GlobalThreshold(settings.cache_threshold),
        db_path=settings.cache_db_path,
        default_ttl_seconds=settings.cache_ttl_seconds,
        max_entries=settings.cache_max_entries,
    )

    guard = None
    if settings.guard_enabled and (Path(settings.guard_dir) / "guard.json").exists():
        from cacheguard.verify import CrossEncoderVerifier

        guard = Guard.load(settings.guard_dir)
        verifier_model = guard.metadata.get("verifier_model", settings.verifier_model)
        guard.verifier = CrossEncoderVerifier(verifier_model)
    return CacheGuardService(embedder, store, backend, artifacts.clusterer, guard)
