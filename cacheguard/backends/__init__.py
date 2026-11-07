"""Answer backends: what CacheGuard calls on a cache miss.

Phase 0 has a document-search backend. The Groq LLM backend added later
implements the same ``Backend`` protocol, so the cache does not change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence

import numpy as np


@dataclass(frozen=True)
class Source:
    doc_id: int
    newsgroup: str
    score: float
    snippet: str


@dataclass(frozen=True)
class BackendResponse:
    answer: str
    sources: tuple[Source, ...] = field(default_factory=tuple)
    # Backend-specific facts worth keeping with a cached answer, e.g. the model
    # and token usage of the LLM call (so a later hit knows what it saved).
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"answer": self.answer, "sources": [vars(s) for s in self.sources], "meta": self.meta}

    @classmethod
    def from_dict(cls, data: dict) -> BackendResponse:
        return cls(
            answer=data["answer"],
            sources=tuple(Source(**s) for s in data.get("sources", ())),
            meta=data.get("meta", {}),
        )


class BackendError(RuntimeError):
    """The backend could not answer. ``retryable`` marks upstream overload/timeouts."""

    def __init__(self, message: str, status: int | None = None, retryable: bool = False):
        super().__init__(message)
        self.status = status
        self.retryable = retryable


class Backend(Protocol):
    name: str

    def cache_settings(self) -> dict:
        """Everything besides the question that shapes the answer.

        Becomes part of the exact-match partition key, so answers produced
        with a different model, prompt or temperature are never reused.
        """
        ...

    def respond(
        self, query: str, query_embedding: np.ndarray, history: Sequence[dict] = ()
    ) -> BackendResponse: ...
