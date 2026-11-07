"""CacheGuard: a semantic cache for LLM APIs that knows when it's wrong.

Questions are matched by meaning (MiniLM + FAISS) inside exact partitions,
and a calibrated guard (similarity gate, swap rules, cross-encoder verifier)
decides whether a cached answer may be reused. Backends: Groq, the original
20 Newsgroups search, and a mock for load tests.
"""

__version__ = "1.0.0"
