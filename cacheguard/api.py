"""HTTP API.

Run with:  uvicorn cacheguard.api:create_default_app --factory
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator, model_validator

from cacheguard import __version__
from cacheguard.config import Settings
from cacheguard.service import CacheGuardService, build_default_service

_NAMESPACE = Field("default", pattern=r"^[A-Za-z0-9_.-]{1,64}$", description="Tenant or app; caches never mix across namespaces.")


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000, examples=["space shuttle launch delays"])
    namespace: str = _NAMESPACE

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


class QueryResponse(BaseModel):
    query: str
    namespace: str
    cache_hit: bool
    entry_id: int = Field(description="Cache entry that answered (hit) or was created (miss).")
    matched_query: str | None
    cache_similarity: float | None = Field(
        description="Cosine similarity to the nearest cached query in this namespace (null when none)."
    )
    threshold: float | None = Field(description="Similarity required for that entry to count as a hit.")
    nearest_query: str | None = Field(description="Nearest cached query, also shown on a miss.")
    dominant_cluster: int | None
    answer: str
    sources: list[SourceOut]
    latency_ms: float


class InvalidateRequest(BaseModel):
    namespace: str | None = Field(None, pattern=r"^[A-Za-z0-9_.-]{1,64}$")
    tag: str | None = Field(None, min_length=1, max_length=200, examples=["source:1234"])

    @model_validator(mode="after")
    def exactly_one(self) -> InvalidateRequest:
        if (self.namespace is None) == (self.tag is None):
            raise ValueError("pass exactly one of namespace or tag")
        return self


def create_app(service: CacheGuardService) -> FastAPI:
    app = FastAPI(title="CacheGuard", version=__version__)
    app.state.service = service

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "backend": service.backend.name, "cache_entries": len(service.store)}

    @app.post("/query", response_model=QueryResponse)
    def query(request: QueryRequest) -> QueryResponse:
        r = service.handle(request.query, request.namespace)
        return QueryResponse(
            query=r.query,
            namespace=r.namespace,
            cache_hit=r.cache_hit,
            entry_id=r.entry_id,
            matched_query=r.matched_query,
            cache_similarity=r.cache_similarity,
            threshold=r.threshold,
            nearest_query=r.nearest_query,
            dominant_cluster=r.dominant_cluster,
            answer=r.response.answer,
            sources=[SourceOut(**vars(s)) for s in r.response.sources],
            latency_ms=round(r.latency_ms, 2),
        )

    @app.get("/cache/stats")
    def cache_stats() -> dict:
        return service.store.stats()

    @app.delete("/cache")
    def clear_cache() -> dict:
        service.store.clear()
        return {"message": "Cache cleared"}

    @app.delete("/cache/entries/{entry_id}")
    def delete_entry(entry_id: int) -> dict:
        if not service.store.invalidate(entry_id=entry_id):
            raise HTTPException(status_code=404, detail="no such cache entry")
        return {"removed": 1}

    @app.post("/cache/invalidate")
    def invalidate(request: InvalidateRequest) -> dict:
        tag = request.tag if request.tag is not None else f"namespace:{request.namespace}"
        return {"removed": service.store.invalidate(tag=tag)}

    return app


def create_default_app() -> FastAPI:
    return create_app(build_default_service(Settings.from_env()))
