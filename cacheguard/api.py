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

import time
import uuid
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, field_validator, model_validator

from cacheguard import __version__
from cacheguard.backends import BackendError
from cacheguard.config import Settings
from cacheguard.limits import BudgetExceeded, RateLimiter
from cacheguard.service import CacheGuardService, QueryResult, build_default_service

_STATIC = Path(__file__).parent / "static"
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
    limiter = RateLimiter(settings.rate_limit_per_minute) if settings.rate_limit_per_minute else None

    def client_id(request: Request) -> str:
        if settings.trust_proxy:
            forwarded = request.headers.get("x-forwarded-for", "")
            if forwarded:
                return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    def rate_limited(request: Request) -> None:
        if limiter is None:
            return
        wait = limiter.check(client_id(request))
        if wait is not None:
            raise HTTPException(
                status_code=429,
                detail=f"Too many requests: limit is {settings.rate_limit_per_minute} per minute. Try again in {wait:.0f}s.",
                headers={"Retry-After": str(int(wait) + 1)},
            )

    def require_admin(x_admin_token: str | None = Header(None)) -> None:
        if settings.admin_token and x_admin_token != settings.admin_token:
            raise HTTPException(status_code=403, detail="admin token required")

    @app.exception_handler(BudgetExceeded)
    def budget_exceeded(_: Request, exc: BudgetExceeded) -> JSONResponse:
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @app.exception_handler(BackendError)
    def backend_failed(_: Request, exc: BackendError) -> JSONResponse:
        status = 503 if exc.retryable else 502
        return JSONResponse(status_code=status, content={"detail": f"The AI backend failed: {exc}"})

    def run(query: str, namespace: str, history=(), use_cache=True, use_guard=True, bypass_reason="requested") -> QueryResult:
        if not use_guard and not settings.allow_unguarded:
            raise HTTPException(status_code=403, detail="unguarded lookups are disabled on this server")
        return service.handle(query, namespace, history, use_cache, use_guard, bypass_reason)

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
        r = run(last.content, request.user or "default", history, use_cache=use_cache, bypass_reason="temperature")

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
    settings = Settings.from_env()
    return create_app(build_default_service(settings), settings)
