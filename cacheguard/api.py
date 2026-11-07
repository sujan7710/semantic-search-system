"""HTTP API and demo page.

Run with:  uvicorn cacheguard.api:create_default_app --factory

Endpoints
  GET  /                       demo chat page
  POST /query                  ask a question (CacheGuard's own format, explains every decision)
  POST /v1/chat/completions    OpenAI-compatible drop-in (non-streaming)
  GET  /stats                  cache, guard, savings and budget numbers
  GET  /health
  GET  /cache/stats            store counters
  DELETE /cache, DELETE /cache/entries/{id}, POST /cache/invalidate   (admin token if configured)
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field, field_validator, model_validator

from cacheguard import __version__
from cacheguard.backends import BackendError
from cacheguard.config import Settings
from cacheguard.limits import BudgetExceeded, RateLimiter
from cacheguard.metrics import Metrics
from cacheguard.service import CacheGuardService, QueryResult, build_default_service

_STATIC = Path(__file__).parent / "static"
request_log = logging.getLogger("cacheguard.requests")


def client_ip(request: Request, proxy_hops: int, trusted_header: str = "") -> str:
    """The client address to rate-limit on.

    ``trusted_header`` (e.g. Cloudflare's CF-Connecting-IP) wins when set: the
    proxy overwrites it, so visitors cannot forge it. Only safe when the app is
    reachable *only* through that proxy (e.g. bound to 127.0.0.1 behind a tunnel).

    Otherwise each proxy appends the address it received the request from to
    X-Forwarded-For, so with N trusted proxies the real client is the N-th
    entry from the right. Anything further left was sent by the client and
    can be forged, so it is never used.
    """
    peer = request.client.host if request.client else "unknown"
    if trusted_header:
        value = request.headers.get(trusted_header, "").strip()
        if value:
            return value
    if proxy_hops <= 0:
        return peer
    forwarded = [p.strip() for p in request.headers.get("x-forwarded-for", "").split(",") if p.strip()]
    if len(forwarded) < proxy_hops:
        return peer  # fewer entries than proxies: the request did not come through them
    return forwarded[-proxy_hops]


def outcome_of(r: QueryResult) -> str:
    if r.decision.stage == "bypass":
        return "bypass"
    return "hit" if r.cache_hit else "miss"


def log_query(r: QueryResult, endpoint: str) -> None:
    """One JSON line per answered question. The question itself is not logged, only a short hash."""
    request_log.info(json.dumps({
        "event": "query",
        "endpoint": endpoint,
        "namespace": r.namespace,
        "query_sha": hashlib.sha256(r.query.encode("utf-8")).hexdigest()[:12],
        "outcome": outcome_of(r),
        "stage": r.decision.stage,
        "reason": r.decision.reason,
        "similarity": None if r.cache_similarity is None else round(r.cache_similarity, 4),
        "verifier_score": None if r.decision.verifier_score is None else round(r.decision.verifier_score, 4),
        "latency_ms": round(r.latency_ms, 1),
        "decision_ms": round(r.decision_ms, 1),
        "backend_ms": None if r.cache_hit else r.response.meta.get("latency_ms"),
        "tokens_saved": (r.saved or {}).get("tokens"),
    }))
_NAMESPACE_PATTERN = r"^[A-Za-z0-9_.-]{1,64}$"
MAX_QUERY_CHARS = 2000


# ----------------------------------------------------------------------------- models


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=MAX_QUERY_CHARS, examples=["How do I make my laptop battery last longer?"])
    namespace: str = Field("default", pattern=_NAMESPACE_PATTERN, description="Tenant or app; caches never mix across namespaces.")
    cache: bool = Field(True, description="False skips the cache entirely (always asks the backend).")
    guard: bool = Field(True, description="False uses the plain similarity threshold (demo comparison).")

    @field_validator("query")
    @classmethod
    def not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must not be blank")
        return value


class SourceOut(BaseModel):
    doc_id: int
    newsgroup: str
    score: float
    snippet: str


class DecisionOut(BaseModel):
    stage: str = Field(description="Where the decision was made: none, gate, rules, similarity, verifier, threshold or bypass.")
    reason: str = Field(description="e.g. verified, below_gate, negation, number, name, verifier_rejected.")
    verifier_score: float | None = Field(description="Cross-encoder probability that both ask the same thing, if it ran.")
    decision_ms: float
    guarded: bool = Field(description="True if the guard made this decision, False for the plain threshold.")


class QueryResponse(BaseModel):
    query: str
    namespace: str
    cache_hit: bool
    entry_id: int | None = Field(description="Cache entry that answered (hit) or was created (miss).")
    matched_query: str | None
    cache_similarity: float | None = Field(description="Cosine similarity to the nearest cached question (null when none).")
    threshold: float | None = Field(description="Similarity gate applied to that entry.")
    nearest_query: str | None = Field(description="Nearest cached question, also shown on a miss.")
    decision: DecisionOut
    dominant_cluster: int | None
    answer: str
    sources: list[SourceOut]
    backend: dict = Field(description="Backend details for this answer, e.g. model and token usage.")
    saved: dict | None = Field(description="On a hit: tokens, cost (USD) and backend time this answer avoided.")
    latency_ms: float


class InvalidateRequest(BaseModel):
    namespace: str | None = Field(None, pattern=_NAMESPACE_PATTERN)
    tag: str | None = Field(None, min_length=1, max_length=200, examples=["source:1234"])

    @model_validator(mode="after")
    def exactly_one(self) -> InvalidateRequest:
        if (self.namespace is None) == (self.tag is None):
            raise ValueError("pass exactly one of namespace or tag")
        return self


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(max_length=20_000)


class ChatRequest(BaseModel):
    """Subset of the OpenAI chat completions request."""

    model: str | None = None
    messages: list[ChatMessage] = Field(min_length=1, max_length=50)
    temperature: float | None = None
    stream: bool = False
    user: str | None = Field(None, pattern=_NAMESPACE_PATTERN, description="Used as the cache namespace.")


def to_response(r: QueryResult) -> QueryResponse:
    return QueryResponse(
        query=r.query,
        namespace=r.namespace,
        cache_hit=r.cache_hit,
        entry_id=r.entry_id,
        matched_query=r.matched_query,
        cache_similarity=r.cache_similarity,
        threshold=r.threshold,
        nearest_query=r.nearest_query,
        decision=DecisionOut(
            stage=r.decision.stage,
            reason=r.decision.reason,
            verifier_score=r.decision.verifier_score,
            decision_ms=round(r.decision_ms, 2),
            guarded=r.guarded,
        ),
        dominant_cluster=r.dominant_cluster,
        answer=r.response.answer,
        sources=[SourceOut(**vars(s)) for s in r.response.sources],
        backend={"name": "cache" if r.cache_hit else "live", **r.response.meta},
        saved=r.saved,
        latency_ms=round(r.latency_ms, 2),
    )


# ----------------------------------------------------------------------------- app


def create_app(service: CacheGuardService, settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    app = FastAPI(title="CacheGuard", version=__version__)
    app.state.service = service
    metrics = app.state.metrics = Metrics()
    limiter = RateLimiter(settings.rate_limit_per_minute) if settings.rate_limit_per_minute else None

    @app.middleware("http")
    async def http_metrics(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        route = request.scope.get("route")
        path = getattr(route, "path", "unmatched")  # route template, so ids don't explode cardinality
        metrics.http_requests.inc(route=path, method=request.method, status=str(response.status_code))
        metrics.http_latency.observe((time.perf_counter() - start) * 1000, route=path)
        return response

    def observe(r: QueryResult, endpoint: str) -> QueryResult:
        outcome = outcome_of(r)
        metrics.lookups.inc(outcome=outcome, stage=r.decision.stage, reason=r.decision.reason)
        metrics.query_latency.observe(r.latency_ms, outcome=outcome)
        if r.decision.stage not in ("none", "bypass"):
            metrics.decision_latency.observe(r.decision_ms, stage=r.decision.stage)
        if not r.cache_hit and "latency_ms" in r.response.meta:
            metrics.backend_latency.observe(float(r.response.meta["latency_ms"]), backend=service.backend.name)
        log_query(r, endpoint)
        return r

    def client_id(request: Request) -> str:
        return client_ip(request, settings.proxy_hops, settings.client_ip_header)

    def rate_limited(request: Request) -> None:
        if limiter is None:
            return
        wait = limiter.check(client_id(request))
        if wait is not None:
            metrics.rejections.inc(limit="rate")
            raise HTTPException(
                status_code=429,
                detail=f"Too many requests: limit is {settings.rate_limit_per_minute} per minute. Try again in {wait:.0f}s.",
                headers={"Retry-After": str(int(wait) + 1)},
            )

    def require_admin(x_admin_token: str | None = Header(None)) -> None:
        if settings.admin_token and not hmac.compare_digest(x_admin_token or "", settings.admin_token):
            raise HTTPException(status_code=403, detail="admin token required")

    @app.get("/admin/whoami", include_in_schema=False)
    def whoami(request: Request, x_admin_token: str | None = Header(None)) -> dict:
        """Check the proxy setting on a live deployment: which client IP does the server see?"""
        if not settings.admin_token:
            raise HTTPException(status_code=404, detail="set CACHEGUARD_ADMIN_TOKEN to enable this endpoint")
        require_admin(x_admin_token)
        forwarded = [p.strip() for p in request.headers.get("x-forwarded-for", "").split(",") if p.strip()]
        return {
            "client_ip_used_for_rate_limits": client_id(request),
            "proxy_hops": settings.proxy_hops,
            "client_ip_header": settings.client_ip_header or None,
            "x_forwarded_for": forwarded,
            "socket_peer": request.client.host if request.client else None,
        }

    @app.exception_handler(BudgetExceeded)
    def budget_exceeded(_: Request, exc: BudgetExceeded) -> JSONResponse:
        metrics.rejections.inc(limit="daily_budget")
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @app.exception_handler(BackendError)
    def backend_failed(_: Request, exc: BackendError) -> JSONResponse:
        metrics.backend_errors.inc(backend=service.backend.name, retryable=str(exc.retryable).lower())
        request_log.warning(json.dumps({"event": "backend_error", "status": exc.status, "retryable": exc.retryable, "error": str(exc)[:200]}))
        status = 503 if exc.retryable else 502
        return JSONResponse(status_code=status, content={"detail": f"The AI backend failed: {exc}"})

    def run(query: str, namespace: str, history=(), use_cache=True, use_guard=True, bypass_reason="requested",
            endpoint: str = "query") -> QueryResult:
        if not use_guard and not settings.allow_unguarded:
            raise HTTPException(status_code=403, detail="unguarded lookups are disabled on this server")
        return observe(service.handle(query, namespace, history, use_cache, use_guard, bypass_reason), endpoint)

    # --- pages -----------------------------------------------------------------

    @app.get("/", include_in_schema=False)
    def demo_page() -> FileResponse:
        # no-cache: browsers revalidate, so visitors see a new page right after a deploy.
        return FileResponse(_STATIC / "demo.html", media_type="text/html", headers={"Cache-Control": "no-cache"})

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "backend": service.backend.name,
            "cache_entries": len(service.store),
            "guard": service.guard.describe() if service.guard else None,
        }

    @app.get("/metrics", response_class=PlainTextResponse, include_in_schema=False)
    def prometheus_metrics() -> PlainTextResponse:
        savings = service.savings.as_dict()
        gauges = {
            "cacheguard_cache_entries": ("Answers currently cached.", len(service.store)),
            "cacheguard_tokens_saved_total": ("Tokens not sent to the backend thanks to cache hits.", savings["tokens_saved"]),
            "cacheguard_cost_saved_usd_total": ("Estimated backend cost avoided by cache hits, USD.", savings["cost_saved_usd"]),
            "cacheguard_backend_cost_usd_total": ("Estimated backend cost spent, USD.", savings["backend_cost_usd"]),
        }
        if service.budget is not None:
            gauges["cacheguard_daily_budget_remaining"] = ("Backend calls left today.", service.budget.status()["remaining"])
        return PlainTextResponse(metrics.render(gauges), media_type="text/plain; version=0.0.4")

    @app.get("/stats")
    def stats() -> dict:
        store = service.store.stats()
        return {
            "backend": {"name": service.backend.name, **service.backend.cache_settings()},
            "cache": {k: store[k] for k in ("total_entries", "hits", "misses", "hit_rate", "decisions", "max_entries", "default_ttl_seconds")},
            "savings": service.savings.as_dict(),
            "budget": service.budget.status() if service.budget else None,
            "guard": service.guard.describe() if service.guard else None,
            "rate_limit_per_minute": settings.rate_limit_per_minute,
        }

    # --- querying --------------------------------------------------------------

    @app.post("/query", response_model=QueryResponse, dependencies=[Depends(rate_limited)])
    def query(request: QueryRequest) -> QueryResponse:
        return to_response(run(request.query, request.namespace, use_cache=request.cache, use_guard=request.guard))

    @app.post("/explain", dependencies=[Depends(rate_limited)])
    def explain(request: QueryRequest) -> dict:
        """Dry run: what the plain threshold and the guard would decide. No AI call, nothing stored."""
        return service.explain(request.query, request.namespace)

    @app.post("/v1/chat/completions", dependencies=[Depends(rate_limited)])
    def chat_completions(request: ChatRequest) -> dict:
        if request.stream:
            raise HTTPException(status_code=400, detail="stream=true is not supported yet")
        configured_model = service.backend.cache_settings().get("model")
        if request.model and configured_model and request.model != configured_model:
            raise HTTPException(status_code=400, detail=f"this gateway serves model {configured_model!r}")
        last = request.messages[-1]
        if last.role != "user" or not last.content.strip():
            raise HTTPException(status_code=400, detail="the last message must be a non-empty user message")
        if len(last.content) > MAX_QUERY_CHARS:
            raise HTTPException(status_code=400, detail=f"the last message is limited to {MAX_QUERY_CHARS} characters")
        history = [m.model_dump() for m in request.messages[:-1]]

        # Cached answers assume the server's deterministic settings; a different
        # temperature asks for different behavior, so it bypasses the cache.
        server_temperature = service.backend.cache_settings().get("temperature")
        use_cache = request.temperature is None or request.temperature == server_temperature
        r = run(last.content, request.user or "default", history, use_cache=use_cache, bypass_reason="temperature",
                endpoint="chat_completions")

        usage = r.response.meta.get("usage", {})
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": r.response.meta.get("model", configured_model or service.backend.name),
            "choices": [{"index": 0, "message": {"role": "assistant", "content": r.response.answer}, "finish_reason": "stop"}],
            # A cache hit consumed no tokens; what it avoided is under cacheguard.saved.
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0} if r.cache_hit else usage,
            "cacheguard": to_response(r).model_dump(include={"cache_hit", "matched_query", "cache_similarity", "decision", "saved", "latency_ms"}),
        }

    # --- cache administration -----------------------------------------------------

    @app.get("/cache/stats")
    def cache_stats() -> dict:
        return service.store.stats()

    @app.delete("/cache", dependencies=[Depends(require_admin)])
    def clear_cache() -> dict:
        service.store.clear()
        return {"message": "Cache cleared"}

    @app.delete("/cache/entries/{entry_id}", dependencies=[Depends(require_admin)])
    def delete_entry(entry_id: int) -> dict:
        if not service.store.invalidate(entry_id=entry_id):
            raise HTTPException(status_code=404, detail="no such cache entry")
        return {"removed": 1}

    @app.post("/cache/invalidate")
    def invalidate(request: InvalidateRequest, x_admin_token: str | None = Header(None)) -> dict:
        # Demo visitors may reset their own "demo-..." namespace without the admin token.
        own_demo = request.namespace is not None and request.namespace.startswith("demo-")
        if not own_demo:
            require_admin(x_admin_token)
        tag = request.tag if request.tag is not None else f"namespace:{request.namespace}"
        return {"removed": service.store.invalidate(tag=tag)}

    return app


def create_default_app() -> FastAPI:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    settings = Settings.from_env()
    service = build_default_service(settings)
    warm_up(service)
    return create_app(service, settings)


def warm_up(service: CacheGuardService) -> None:
    """Run both models once at startup. Their first call is ~10x slower (lazy init),
    and that cost should not land on the first visitor."""
    start = time.perf_counter()
    service.embedder.encode(["warm up"])
    if service.guard is not None and service.guard.verifier is not None:
        service.guard.verifier.score([("warm up", "warming up")])
    logging.getLogger("cacheguard").info("models warmed up in %.0f ms", (time.perf_counter() - start) * 1000)
