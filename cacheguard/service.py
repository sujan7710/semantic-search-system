"""The request path: embed -> cache lookup -> (hit) return | (miss) backend -> store."""

from __future__ import annotations

import time
from dataclasses import dataclass

from cacheguard.backends import Backend, BackendResponse
from cacheguard.clustering import SoftClusterer
from cacheguard.config import Settings
from cacheguard.embeddings import Embedder
from cacheguard.partitions import partition_key
from cacheguard.store import CacheStore


@dataclass(frozen=True)
class QueryResult:
    query: str
    namespace: str
    cache_hit: bool
    entry_id: int  # the cache entry that answered (hit) or was created (miss)
    matched_query: str | None
    cache_similarity: float | None  # nearest cached query in the partition, hit or not
    threshold: float | None  # threshold applied to that nearest entry
    nearest_query: str | None  # nearest cached query, shown even on a miss
    dominant_cluster: int | None
    response: BackendResponse
    latency_ms: float


class CacheGuardService:
    def __init__(
        self,
        embedder: Embedder,
        store: CacheStore,
        backend: Backend,
        clusterer: SoftClusterer | None = None,
    ):
        self.embedder = embedder
        self.store = store
        self.backend = backend
        self.clusterer = clusterer

    def handle(self, query: str, namespace: str = "default") -> QueryResult:
        start = time.perf_counter()
        query = query.strip()
        if not query:
            raise ValueError("query must not be empty")

        partition = partition_key(namespace, self.backend.name)
        embedding = self.embedder.encode([query])[0]
        lookup = self.store.lookup(partition, embedding)

        if lookup.hit and lookup.entry is not None:
            entry = lookup.entry
            return QueryResult(
                query=query,
                namespace=namespace,
                cache_hit=True,
                entry_id=entry.entry_id,
                matched_query=entry.query,
                cache_similarity=lookup.similarity,
                threshold=lookup.threshold,
                nearest_query=lookup.nearest_query,
                dominant_cluster=entry.cluster,
                response=BackendResponse.from_dict(entry.response),
                latency_ms=(time.perf_counter() - start) * 1000,
            )

        response = self.backend.respond(query, embedding)
        cluster = None
        if self.clusterer is not None and self.clusterer.is_fitted:
            cluster = int(self.clusterer.predict(embedding)[0])
        # Tags let one call invalidate every answer built from a given source document,
        # or everything in a namespace.
        tags = [f"namespace:{namespace}", *(f"source:{s.doc_id}" for s in response.sources)]
        entry = self.store.add(partition, query, embedding, response.to_dict(), cluster, tags)

        return QueryResult(
            query=query,
            namespace=namespace,
            cache_hit=False,
            entry_id=entry.entry_id,
            matched_query=None,
            cache_similarity=lookup.similarity,
            threshold=lookup.threshold,
            nearest_query=lookup.nearest_query,
            dominant_cluster=cluster,
            response=response,
            latency_ms=(time.perf_counter() - start) * 1000,
        )


def build_default_service(settings: Settings) -> CacheGuardService:
    """Wire up the real model, prebuilt index, persistent cache and newsgroups backend."""
    from cacheguard.artifacts import load_artifacts
    from cacheguard.backends.newsgroups import NewsgroupsSearchBackend
    from cacheguard.embeddings import SentenceTransformerEmbedder
    from cacheguard.policy import GlobalThreshold
    from cacheguard.vector_index import VectorIndex

    import faiss

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
    return CacheGuardService(embedder, store, backend, artifacts.clusterer)
