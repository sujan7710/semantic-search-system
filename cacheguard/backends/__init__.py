"""Answer backends: what CacheGuard calls on a cache miss.

Phase 0 has a document-search backend. The Groq LLM backend added later
implements the same ``Backend`` protocol, so the cache does not change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

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

    def to_dict(self) -> dict:
        return {"answer": self.answer, "sources": [vars(s) for s in self.sources]}

    @classmethod
    def from_dict(cls, data: dict) -> BackendResponse:
        return cls(answer=data["answer"], sources=tuple(Source(**s) for s in data.get("sources", ())))


class Backend(Protocol):
    name: str

    def respond(self, query: str, query_embedding: np.ndarray) -> BackendResponse: ...
