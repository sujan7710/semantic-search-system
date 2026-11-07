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


class Backend(Protocol):
    name: str

    def respond(self, query: str, query_embedding: np.ndarray) -> BackendResponse: ...
