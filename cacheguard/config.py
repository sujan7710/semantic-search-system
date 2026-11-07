"""Runtime settings, overridable with CACHEGUARD_* environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env(name: str, default: object) -> str:
    return os.environ.get(f"CACHEGUARD_{name}", str(default))


@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path("data/20_newsgroups")
    artifact_dir: Path = Path("artifacts/newsgroups")
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    cache_threshold: float = 0.85
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
            top_k=int(_env("TOP_K", d.top_k)),
            snippet_chars=int(_env("SNIPPET_CHARS", d.snippet_chars)),
            n_clusters=int(_env("N_CLUSTERS", d.n_clusters)),
            cluster_covariance=_env("CLUSTER_COVARIANCE", d.cluster_covariance),
            random_seed=int(_env("RANDOM_SEED", d.random_seed)),
        )
