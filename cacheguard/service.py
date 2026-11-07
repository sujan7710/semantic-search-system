"""The request path: embed -> cache lookup -> (hit) return | (miss) backend -> store."""

from __future__ import annotations

import time
from dataclasses import dataclass

from cacheguard.backends import Backend, BackendResponse
from cacheguard.cache import SemanticCache
from cacheguard.clustering import SoftClusterer
from cacheguard.config import Settings
from cacheguard.embeddings import Embedder


@dataclass(frozen=True)
class QueryResult:
    query: str
    cache_hit: bool
    matched_query: str | None
    cache_similarity: float | None  # nearest cached query, hit or not
    dominant_cluster: int | None
    response: BackendResponse
    latency_ms: float


class CacheGuardService:
    def __init__(
        self,
        embedder: Embedder,
        cache: SemanticCache,
        backend: Backend,
        clusterer: SoftClusterer | None = None,
    ):
        self.embedder = embedder
        self.cache = cache
        self.backend = backend
        self.clusterer = clusterer

    def handle(self, query: str) -> QueryResult:
        start = time.perf_counter()
        query = query.strip()
        if not query:
            raise ValueError("query must not be empty")

        embedding = self.embedder.encode([query])[0]
        lookup = self.cache.lookup(embedding)

        if lookup.hit and lookup.entry is not None:
            entry = lookup.entry
            return QueryResult(
                query=query,
                cache_hit=True,
                matched_query=entry.query,
                cache_similarity=lookup.similarity,
                dominant_cluster=entry.cluster,
                response=entry.response,
                latency_ms=(time.perf_counter() - start) * 1000,
            )

        response = self.backend.respond(query, embedding)
        cluster = None
        if self.clusterer is not None and self.clusterer.is_fitted:
            cluster = int(self.clusterer.predict(embedding)[0])
        self.cache.add(query, embedding, response, cluster)

        return QueryResult(
            query=query,
            cache_hit=False,
            matched_query=None,
            cache_similarity=lookup.similarity,
            dominant_cluster=cluster,
            response=response,
            latency_ms=(time.perf_counter() - start) * 1000,
        )


def build_default_service(settings: Settings) -> CacheGuardService:
    """Wire up the real model, prebuilt index and newsgroups backend."""
    from cacheguard.artifacts import load_artifacts
    from cacheguard.backends.newsgroups import NewsgroupsSearchBackend
    from cacheguard.embeddings import SentenceTransformerEmbedder
    from cacheguard.vector_index import VectorIndex

    artifacts = load_artifacts(settings.artifact_dir, expected_model=settings.embedding_model)
    embedder = SentenceTransformerEmbedder(settings.embedding_model)
    backend = NewsgroupsSearchBackend(
        artifacts.documents,
        VectorIndex.from_embeddings(artifacts.embeddings),
        top_k=settings.top_k,
        snippet_chars=settings.snippet_chars,
    )
    return CacheGuardService(embedder, SemanticCache(settings.cache_threshold), backend, artifacts.clusterer)
