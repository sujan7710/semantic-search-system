"""The request path.

    embed -> nearest cached question (same partition)
          -> decide: guard (gate, rules, verifier) or plain threshold
          -> hit: return cached answer | miss: ask backend, store the answer

Partitions are exact: namespace + backend + everything that shapes the
answer (model, system prompt, temperature) + a hash of earlier conversation
turns. Only the *last question* is matched by meaning, so a follow-up like
"and in Python?" is never answered from a different conversation.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass, field
from typing import Sequence

from cacheguard.backends import Backend, BackendResponse
from cacheguard.clustering import SoftClusterer
from cacheguard.config import Settings
from cacheguard.embeddings import Embedder
from cacheguard.guard import Guard
from cacheguard.limits import DailyBudget
from cacheguard.partitions import partition_key
from cacheguard.store import CacheStore


@dataclass(frozen=True)
class DecisionInfo:
    stage: str  # none | gate | rules | similarity | verifier | threshold | bypass
    reason: str  # empty_cache | below_gate | negation | number | name | high_similarity | verified | ...
    verifier_score: float | None = None


@dataclass(frozen=True)
class QueryResult:
    query: str
    namespace: str
    cache_hit: bool
    entry_id: int | None  # the cache entry that answered (hit) or was created (miss); None if bypassed
    matched_query: str | None
    cache_similarity: float | None  # nearest cached query in the partition, hit or not
    threshold: float | None  # similarity gate applied to that nearest entry
    nearest_query: str | None  # nearest cached query, shown even on a miss
    decision: DecisionInfo
    guarded: bool
    dominant_cluster: int | None
    response: BackendResponse
    latency_ms: float
    decision_ms: float  # time spent deciding (includes the verifier when it runs)
    saved: dict | None = None  # on a hit: tokens, cost and backend time this answer avoided


@dataclass
class Savings:
    backend_calls: int = 0
    backend_cost_usd: float = 0.0
    backend_ms: float = 0.0
    cache_hits: int = 0
    tokens_saved: int = 0
    cost_saved_usd: float = 0.0
    backend_ms_saved: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def as_dict(self) -> dict:
        with self.lock:
            return {
                "backend_calls": self.backend_calls,
                "backend_cost_usd": round(self.backend_cost_usd, 6),
                "avg_backend_ms": round(self.backend_ms / self.backend_calls, 1) if self.backend_calls else None,
                "cache_hits": self.cache_hits,
                "tokens_saved": self.tokens_saved,
                "cost_saved_usd": round(self.cost_saved_usd, 6),
                "backend_seconds_saved": round(self.backend_ms_saved / 1000, 2),
            }


def history_digest(history: Sequence[dict]) -> str | None:
    if not history:
        return None
    canonical = json.dumps([{"role": m.get("role"), "content": m.get("content")} for m in history], sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


class CacheGuardService:
    def __init__(
        self,
        embedder: Embedder,
        store: CacheStore,
        backend: Backend,
        clusterer: SoftClusterer | None = None,
        guard: Guard | None = None,
        budget: DailyBudget | None = None,
    ):
        self.embedder = embedder
        self.store = store
        self.backend = backend
        self.clusterer = clusterer
        self.guard = guard
        self.budget = budget
        self.savings = Savings()

    def partition(self, namespace: str, history: Sequence[dict] = ()) -> str:
        settings = dict(self.backend.cache_settings())
        digest = history_digest(history)
        if digest:
            settings["history"] = digest
        return partition_key(namespace, self.backend.name, **settings)

    def _decide(self, partition: str, query: str, embedding, nearest, use_guard: bool):
        """Returns (hit, threshold, DecisionInfo). Runs outside the store lock."""
        if nearest is None:
            return False, None, DecisionInfo("none", "empty_cache")
        entry, similarity = nearest
        if use_guard and self.guard is not None:
            d = self.guard.decide(query, embedding, entry.query, similarity)
            return d.hit, d.gate, DecisionInfo(d.stage, d.reason, d.verifier_score)
        threshold = self.store.policy.threshold(partition, entry.cluster)
        hit = similarity >= threshold
        return hit, threshold, DecisionInfo("threshold", "above_threshold" if hit else "below_threshold")

    def _call_backend(self, query: str, embedding, history: Sequence[dict]) -> BackendResponse:
        if self.budget is not None:
            self.budget.consume()
        try:
            response = self.backend.respond(query, embedding, history)
        except Exception:
            if self.budget is not None:
                self.budget.refund()
            raise
        with self.savings.lock:
            self.savings.backend_calls += 1
            self.savings.backend_cost_usd += float(response.meta.get("cost_usd", 0.0))
            self.savings.backend_ms += float(response.meta.get("latency_ms", 0.0))
        return response

    def explain(self, query: str, namespace: str = "default", history: Sequence[dict] = ()) -> dict:
        """Dry run: what would the plain threshold and the guard decide right now?

        Calls no backend, records nothing. The demo uses it to show the wrong
        answers a plain cache would have served.
        """
        query = query.strip()
        if not query:
            raise ValueError("query must not be empty")
        embedding = self.embedder.encode([query])[0]
        partition = self.partition(namespace, history)
        nearest = self.store.nearest(partition, embedding)
        if nearest is None:
            return {"nearest_query": None, "similarity": None, "plain": None, "guard": None}
        entry, similarity = nearest
        threshold = self.store.policy.threshold(partition, entry.cluster)
        out = {
            "nearest_query": entry.query,
            "similarity": similarity,
            "plain": {"hit": similarity >= threshold, "threshold": threshold},
            "guard": None,
        }
        if self.guard is not None:
            d = self.guard.decide(query, embedding, entry.query, similarity)
            out["guard"] = {"hit": d.hit, "stage": d.stage, "reason": d.reason, "verifier_score": d.verifier_score, "gate": d.gate}
        return out

    def handle(
        self,
        query: str,
        namespace: str = "default",
        history: Sequence[dict] = (),
        use_cache: bool = True,
        use_guard: bool = True,
        bypass_reason: str = "requested",
    ) -> QueryResult:
        start = time.perf_counter()
        query = query.strip()
        if not query:
            raise ValueError("query must not be empty")
        guarded = use_guard and self.guard is not None
        embedding = self.embedder.encode([query])[0]

        if not use_cache:
            response = self._call_backend(query, embedding, history)
            return QueryResult(
                query=query, namespace=namespace, cache_hit=False, entry_id=None, matched_query=None,
                cache_similarity=None, threshold=None, nearest_query=None,
                decision=DecisionInfo("bypass", bypass_reason), guarded=guarded, dominant_cluster=None,
                response=response, latency_ms=(time.perf_counter() - start) * 1000, decision_ms=0.0,
            )

        partition = self.partition(namespace, history)
        nearest = self.store.nearest(partition, embedding)
        decide_start = time.perf_counter()
        hit, threshold, info = self._decide(partition, query, embedding, nearest, use_guard)
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
            guarded=guarded,
            decision_ms=decision_ms,
        )
        if hit:
            response = BackendResponse.from_dict(entry.response)
            usage = response.meta.get("usage", {})
            saved = {
                "tokens": int(usage.get("total_tokens", usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0))),
                "cost_usd": float(response.meta.get("cost_usd", 0.0)),
                "backend_ms": float(response.meta.get("latency_ms", 0.0)),
            }
            with self.savings.lock:
                self.savings.cache_hits += 1
                self.savings.tokens_saved += saved["tokens"]
                self.savings.cost_saved_usd += saved["cost_usd"]
                self.savings.backend_ms_saved += saved["backend_ms"]
            return QueryResult(
                cache_hit=True,
                entry_id=entry.entry_id,
                matched_query=entry.query,
                dominant_cluster=entry.cluster,
                response=response,
                latency_ms=(time.perf_counter() - start) * 1000,
                saved=saved,
                **common,
            )

        response = self._call_backend(query, embedding, history)
        cluster = None
        if self.clusterer is not None and self.clusterer.is_fitted:
            cluster = int(self.clusterer.predict(embedding)[0])
        # Tags let one call invalidate every answer built from a given source
        # document or model, or everything in a namespace.
        tags = [f"namespace:{namespace}", f"backend:{self.backend.name}"]
        tags += [f"source:{s.doc_id}" for s in response.sources]
        if "model" in response.meta:
            tags.append(f"model:{response.meta['model']}")
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


def build_backend(settings: Settings):
    """Returns (backend, newsgroups clusterer or None)."""
    kind = settings.resolved_backend()
    if kind == "groq":
        from cacheguard.backends.groq import GroqBackend

        backend = GroqBackend(
            api_key=settings.groq_api_key,
            model=settings.groq_model,
            system_prompt=settings.system_prompt,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
            reasoning_effort=settings.groq_reasoning_effort,
            base_url=settings.groq_base_url,
            price_input_per_m=settings.price_input_per_m,
            price_output_per_m=settings.price_output_per_m,
        )
        return backend, None

    from cacheguard.artifacts import load_artifacts
    from cacheguard.backends.newsgroups import NewsgroupsSearchBackend
    from cacheguard.vector_index import VectorIndex

    artifacts = load_artifacts(settings.artifact_dir, expected_model=settings.embedding_model)
    backend = NewsgroupsSearchBackend(
        artifacts.documents,
        VectorIndex.from_embeddings(artifacts.embeddings),
        top_k=settings.top_k,
        snippet_chars=settings.snippet_chars,
    )
    return backend, artifacts.clusterer


def build_default_service(settings: Settings) -> CacheGuardService:
    """Wire up the embedder, backend, persistent cache, guard and demo budget."""
    from pathlib import Path

    import faiss

    from cacheguard.embeddings import SentenceTransformerEmbedder
    from cacheguard.policy import GlobalThreshold

    faiss.omp_set_num_threads(settings.faiss_threads)
    backend, clusterer = build_backend(settings)
    embedder = SentenceTransformerEmbedder(settings.embedding_model)
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
        guard.verifier = CrossEncoderVerifier(guard.metadata.get("verifier_model", settings.verifier_model))
    budget = DailyBudget(settings.daily_backend_calls) if settings.daily_backend_calls else None
    return CacheGuardService(embedder, store, backend, clusterer, guard, budget)
