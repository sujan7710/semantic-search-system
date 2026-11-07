"""CacheGuard: a semantic cache that sits in front of an answer backend.

Phase 0 ships the original 20 Newsgroups search as the backend; later phases
swap in an LLM (Groq) behind the same interface.
"""

__version__ = "0.1.0"
