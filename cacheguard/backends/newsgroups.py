"""Backend that answers a query with the most similar newsgroup posts."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from cacheguard.backends import BackendResponse, Source
from cacheguard.data.newsgroups import Document
from cacheguard.vector_index import VectorIndex


class NewsgroupsSearchBackend:
    name = "newsgroups-search"

    def __init__(self, documents: Sequence[Document], index: VectorIndex, top_k: int = 5, snippet_chars: int = 300):
        if len(documents) != len(index):
            raise ValueError(f"{len(documents)} documents but {len(index)} vectors in the index")
        self._documents = documents
        self._index = index
        self.top_k = top_k
        self.snippet_chars = snippet_chars

    def _snippet(self, text: str) -> str:
        if len(text) <= self.snippet_chars:
            return text
        return text[: self.snippet_chars].rsplit(" ", 1)[0] + "..."

    def respond(self, query: str, query_embedding: np.ndarray) -> BackendResponse:
        scores, ids = self._index.search(query_embedding, self.top_k)
        sources = tuple(
            Source(
                doc_id=int(i),
                newsgroup=self._documents[i].newsgroup,
                score=float(s),
                snippet=self._snippet(self._documents[i].text),
            )
            for s, i in zip(scores, ids)
            if i >= 0
        )
        answer = sources[0].snippet if sources else ""
        return BackendResponse(answer=answer, sources=sources)
