"""Runtime settings, overridable with CACHEGUARD_* environment variables.

A ``.env`` file in the working directory is read first (values already in
the environment win). It is git-ignored; see ``.env.example``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def load_dotenv(path: str | Path = ".env") -> None:
    """Minimal .env reader: KEY=VALUE lines, # comments, optional quotes."""
    path = Path(path)
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip().removeprefix("export ").strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(key, value)


def _env(name: str, default: object) -> str:
    return os.environ.get(f"CACHEGUARD_{name}", str(default))


def _optional_float(name: str, default: float | None) -> float | None:
    raw = _env(name, default)
    return None if raw.strip().lower() in {"", "none", "off"} else float(raw)


def _optional_int(name: str, default: int | None) -> int | None:
    raw = _env(name, default)
    return None if raw.strip().lower() in {"", "none", "off"} else int(raw)


def _secret(raw: str) -> str:
    """Strip whitespace and one pair of surrounding quotes.

    `docker run --env-file` and some hosting dashboards pass quotes through
    literally (GROQ_API_KEY="gsk_..." arrives with the quotes), which Groq
    then rejects as an invalid key.
    """
    value = raw.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        value = value[1:-1].strip()
    return value


def _flag(name: str, default: bool) -> bool:
    return _env(name, "on" if default else "off").strip().lower() not in {"off", "0", "false", "no"}


@dataclass(frozen=True)
class Settings:
    # --- backend: what answers on a cache miss -------------------------------
    backend: str = "auto"  # auto (groq if GROQ_API_KEY is set, else newsgroups) | groq | newsgroups | mock
    mock_delay_ms: float = 600.0  # simulated LLM latency for the mock backend (load tests)
    groq_api_key: str = ""  # read from GROQ_API_KEY (no prefix), never logged
    groq_model: str = "openai/gpt-oss-20b"
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_reasoning_effort: str = "low"  # for gpt-oss models; empty to omit
    system_prompt: str = "You are a concise, accurate assistant. Answer in at most a few short paragraphs."
    temperature: float = 0.0  # cached answers assume deterministic generation
    max_tokens: int = 512
    # USD per 1M tokens, used to report money saved (gpt-oss-20b list price, Nov 2026)
    price_input_per_m: float = 0.075
    price_output_per_m: float = 0.30

    # --- newsgroups backend ---------------------------------------------------
    data_dir: Path = Path("data/20_newsgroups")
    artifact_dir: Path = Path("artifacts/newsgroups")
    top_k: int = 5
    snippet_chars: int = 300
    n_clusters: int = 10
    cluster_covariance: str = "full"
    random_seed: int = 42

    # --- cache and guard --------------------------------------------------------
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    cache_threshold: float = 0.85
    cache_db_path: str = "artifacts/cache.sqlite3"  # ":memory:" disables persistence
    cache_ttl_seconds: float | None = 7 * 24 * 3600
    cache_max_entries: int | None = 50_000
    faiss_threads: int = 1  # per-search threads; concurrency comes from parallel requests
    # Per-inference threads for the two models. 1 measured best under concurrent load
    # (+42% throughput at 32 users); "none" restores PyTorch's default of all cores.
    torch_threads: int | None = 1
    guard_enabled: bool = True  # False falls back to the plain global threshold
    guard_dir: str = "configs/guard"  # calibrated guard settings (python -m cacheguard.bench.guard_eval)
    verifier_model: str = "cross-encoder/quora-distilroberta-base"
    verifier_memo_size: int = 50_000  # remembered (question, candidate) scores; 0 disables

    # --- public demo protection ------------------------------------------------
    rate_limit_per_minute: int | None = 30  # per client IP, on /query, /explain and /v1/chat/completions
    daily_backend_calls: int | None = 500  # LLM calls per UTC day; cache hits are free
    # How many proxies the *host* puts in front of the app. The client IP is then
    # read that many entries from the right of X-Forwarded-For; entries further
    # left are whatever the visitor sent and cannot be trusted. 0 ignores the header.
    proxy_hops: int = 0
    # A header set by the proxy itself with the client IP (e.g. "cf-connecting-ip"
    # behind Cloudflare). Takes precedence over proxy_hops. Only safe when the app
    # cannot be reached except through that proxy.
    client_ip_header: str = ""
    admin_token: str = ""  # if set, cache admin endpoints require header X-Admin-Token
    allow_unguarded: bool = True  # demo may ask for a plain-threshold decision to compare

    @classmethod
    def from_env(cls, dotenv: str | Path | None = ".env") -> Settings:
        if dotenv:
            load_dotenv(dotenv)
        d = cls()
        return cls(
            backend=_env("BACKEND", d.backend).strip().lower(),
            mock_delay_ms=float(_env("MOCK_DELAY_MS", d.mock_delay_ms)),
            groq_api_key=_secret(os.environ.get("GROQ_API_KEY", "")),
            groq_model=_env("GROQ_MODEL", d.groq_model),
            groq_base_url=_env("GROQ_BASE_URL", d.groq_base_url),
            groq_reasoning_effort=_env("GROQ_REASONING_EFFORT", d.groq_reasoning_effort).strip(),
            system_prompt=_env("SYSTEM_PROMPT", d.system_prompt),
            temperature=float(_env("TEMPERATURE", d.temperature)),
            max_tokens=int(_env("MAX_TOKENS", d.max_tokens)),
            price_input_per_m=float(_env("PRICE_INPUT_PER_M", d.price_input_per_m)),
            price_output_per_m=float(_env("PRICE_OUTPUT_PER_M", d.price_output_per_m)),
            data_dir=Path(_env("DATA_DIR", d.data_dir)),
            artifact_dir=Path(_env("ARTIFACT_DIR", d.artifact_dir)),
            top_k=int(_env("TOP_K", d.top_k)),
            snippet_chars=int(_env("SNIPPET_CHARS", d.snippet_chars)),
            n_clusters=int(_env("N_CLUSTERS", d.n_clusters)),
            cluster_covariance=_env("CLUSTER_COVARIANCE", d.cluster_covariance),
            random_seed=int(_env("RANDOM_SEED", d.random_seed)),
            embedding_model=_env("EMBEDDING_MODEL", d.embedding_model),
            cache_threshold=float(_env("CACHE_THRESHOLD", d.cache_threshold)),
            cache_db_path=_env("CACHE_DB_PATH", d.cache_db_path),
            cache_ttl_seconds=_optional_float("CACHE_TTL_SECONDS", d.cache_ttl_seconds),
            cache_max_entries=_optional_int("CACHE_MAX_ENTRIES", d.cache_max_entries),
            faiss_threads=int(_env("FAISS_THREADS", d.faiss_threads)),
            torch_threads=_optional_int("TORCH_THREADS", d.torch_threads),
            guard_enabled=_flag("GUARD", d.guard_enabled),
            guard_dir=_env("GUARD_DIR", d.guard_dir),
            verifier_model=_env("VERIFIER_MODEL", d.verifier_model),
            verifier_memo_size=int(_env("VERIFIER_MEMO_SIZE", d.verifier_memo_size)),
            rate_limit_per_minute=_optional_int("RATE_LIMIT_PER_MINUTE", d.rate_limit_per_minute),
            daily_backend_calls=_optional_int("DAILY_BACKEND_CALLS", d.daily_backend_calls),
            proxy_hops=int(_env("PROXY_HOPS", d.proxy_hops)),
            client_ip_header=_env("CLIENT_IP_HEADER", d.client_ip_header).strip().lower(),
            admin_token=_secret(_env("ADMIN_TOKEN", d.admin_token)),
            allow_unguarded=_flag("ALLOW_UNGUARDED", d.allow_unguarded),
        )

    def resolved_backend(self) -> str:
        if self.backend == "auto":
            return "groq" if self.groq_api_key else "newsgroups"
        if self.backend not in {"groq", "newsgroups", "mock"}:
            raise ValueError(f"unknown backend {self.backend!r}; use auto, groq, newsgroups or mock")
        return self.backend
