"""Runtime settings, overridable with CACHEGUARD_* environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env(name: str, default: object) -> str:
    return os.environ.get(f"CACHEGUARD_{name}", str(default))


def _optional_float(name: str, default: float | None) -> float | None:
    raw = _env(name, default)
    return None if raw.strip().lower() in {"", "none", "off"} else float(raw)


def _optional_int(name: str, default: int | None) -> int | None:
    raw = _env(name, default)
    return None if raw.strip().lower() in {"", "none", "off"} else int(raw)


@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path("data/20_newsgroups")
    artifact_dir: Path = Path("artifacts/newsgroups")
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    cache_threshold: float = 0.85
    cache_db_path: str = "artifacts/cache.sqlite3"  # ":memory:" disables persistence
    cache_ttl_seconds: float | None = 7 * 24 * 3600
    cache_max_entries: int | None = 50_000
    faiss_threads: int = 1  # per-search threads; concurrency comes from parallel requests
    guard_enabled: bool = True  # False falls back to the plain global threshold
    guard_dir: str = "configs/guard"  # calibrated guard settings (python -m cacheguard.bench.guard_eval)
    verifier_model: str = "cross-encoder/quora-distilroberta-base"
    top_k: int = 5
    snippet_chars: int = 300
    n_clusters: int = 10
    cluster_covariance: str = "full"
    random_seed: int = 42

    @classmethod
    def from_env(cls) -> Settings:
        d = cls()
        return cls(
            data_dir=Path(_env("DATA_DIR", d.data_dir)),
            artifact_dir=Path(_env("ARTIFACT_DIR", d.artifact_dir)),
            embedding_model=_env("EMBEDDING_MODEL", d.embedding_model),
            cache_threshold=float(_env("CACHE_THRESHOLD", d.cache_threshold)),
            cache_db_path=_env("CACHE_DB_PATH", d.cache_db_path),
            cache_ttl_seconds=_optional_float("CACHE_TTL_SECONDS", d.cache_ttl_seconds),
            cache_max_entries=_optional_int("CACHE_MAX_ENTRIES", d.cache_max_entries),
            faiss_threads=int(_env("FAISS_THREADS", d.faiss_threads)),
            guard_enabled=_env("GUARD", "on").strip().lower() not in {"off", "0", "false", "no"},
            guard_dir=_env("GUARD_DIR", d.guard_dir),
            verifier_model=_env("VERIFIER_MODEL", d.verifier_model),
            top_k=int(_env("TOP_K", d.top_k)),
            snippet_chars=int(_env("SNIPPET_CHARS", d.snippet_chars)),
            n_clusters=int(_env("N_CLUSTERS", d.n_clusters)),
            cluster_covariance=_env("CLUSTER_COVARIANCE", d.cluster_covariance),
            random_seed=int(_env("RANDOM_SEED", d.random_seed)),
        )
