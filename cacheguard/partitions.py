"""Partition keys: which cached answers are even allowed to match?

Two requests may only share a cached answer if everything that shapes the
answer, apart from the question wording, is identical: the tenant/namespace,
the backend, and (for LLMs, from Phase 4) the model, system prompt and
sampling settings. These are matched *exactly*, never by similarity.
"""

from __future__ import annotations

import hashlib
import json


def partition_key(namespace: str, backend: str, **settings: object) -> str:
    """Readable prefix plus a short hash of the exact settings.

    >>> partition_key("acme", "groq", model="llama", temperature=0)
    'acme|groq|...'
    """
    if not settings:
        return f"{namespace}|{backend}"
    canonical = json.dumps(settings, sort_keys=True, separators=(",", ":"), default=str)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    return f"{namespace}|{backend}|{digest}"
