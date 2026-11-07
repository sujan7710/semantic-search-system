"""HTTP API.

Run with:  uvicorn cacheguard.api:create_default_app --factory
"""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator

from cacheguard import __version__
from cacheguard.config import Settings
from cacheguard.service import CacheGuardService, build_default_service


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000, examples=["space shuttle launch delays"])

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
    cache_hit: bool
    matched_query: str | None
    cache_similarity: float | None = Field(
        description="Cosine similarity to the nearest cached query (null when the cache is empty)."
    )
    dominant_cluster: int | None
    answer: str
    sources: list[SourceOut]
    latency_ms: float


def create_app(service: CacheGuardService) -> FastAPI:
    app = FastAPI(title="CacheGuard", version=__version__)
    app.state.service = service

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "backend": service.backend.name, "cache_entries": len(service.cache)}

    @app.post("/query", response_model=QueryResponse)
    def query(request: QueryRequest) -> QueryResponse:
        result = service.handle(request.query)
        return QueryResponse(
            query=result.query,
            cache_hit=result.cache_hit,
            matched_query=result.matched_query,
            cache_similarity=result.cache_similarity,
            dominant_cluster=result.dominant_cluster,
            answer=result.response.answer,
            sources=[SourceOut(**vars(s)) for s in result.response.sources],
            latency_ms=round(result.latency_ms, 2),
        )

    @app.get("/cache/stats")
    def cache_stats() -> dict:
        return service.cache.stats()

    @app.delete("/cache")
    def clear_cache() -> dict:
        service.cache.clear()
        return {"message": "Cache cleared"}

    return app


def create_default_app() -> FastAPI:
    return create_app(build_default_service(Settings.from_env()))
