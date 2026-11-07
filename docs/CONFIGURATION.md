# Configuration

Settings come from environment variables, read at startup. A `.env` file in the working directory is loaded first
(values already in the environment win); it is git-ignored, and [.env.example](../.env.example) lists the common ones.
Defaults live in [config.py](../cacheguard/config.py).

## Backend

| Variable | Default | Meaning |
|---|---|---|
| `GROQ_API_KEY` | (none) | Groq key. With it the backend is Groq; without it, the original newsgroups search |
| `CACHEGUARD_BACKEND` | `auto` | `auto`, `groq`, `newsgroups`, or `mock` (fake LLM for load tests) |
| `CACHEGUARD_GROQ_MODEL` | `openai/gpt-oss-20b` | Any Groq chat model id |
| `CACHEGUARD_GROQ_REASONING_EFFORT` | `low` | Sent for gpt-oss models; empty to omit |
| `CACHEGUARD_SYSTEM_PROMPT` | concise assistant | Part of the cache key: changing it starts a fresh partition |
| `CACHEGUARD_TEMPERATURE` | `0` | Cached answers assume deterministic generation; requests asking for another temperature bypass the cache |
| `CACHEGUARD_MAX_TOKENS` | `512` | Max completion tokens per answer |
| `CACHEGUARD_PRICE_INPUT_PER_M` / `_OUTPUT_PER_M` | `0.075` / `0.30` | USD per 1M tokens, for the "saved" numbers |
| `CACHEGUARD_MOCK_DELAY_MS` | `600` | Simulated LLM latency for the mock backend |

## Guard and cache

| Variable | Default | Meaning |
|---|---|---|
| `CACHEGUARD_GUARD` | `on` | `off` uses the plain similarity threshold instead of the guard |
| `CACHEGUARD_GUARD_DIR` | `configs/guard` | Calibrated guard settings (written by `python -m cacheguard.bench.guard_eval`) |
| `CACHEGUARD_VERIFIER_MODEL` | `cross-encoder/quora-distilroberta-base` | Used if the guard config does not name one |
| `CACHEGUARD_VERIFIER_MEMO_SIZE` | `50000` | Remembered verifier scores; `0` disables (load tests) |
| `CACHEGUARD_CACHE_THRESHOLD` | `0.85` | Plain threshold, used only when the guard is off |
| `CACHEGUARD_CACHE_DB_PATH` | `artifacts/cache.sqlite3` | SQLite file; `:memory:` disables persistence |
| `CACHEGUARD_CACHE_TTL_SECONDS` | `604800` (7 days) | Default entry lifetime; `none` disables expiry |
| `CACHEGUARD_CACHE_MAX_ENTRIES` | `50000` | LRU eviction cap; `none` disables |
| `CACHEGUARD_EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model for questions |

## Performance

| Variable | Default | Meaning |
|---|---|---|
| `CACHEGUARD_TORCH_THREADS` | `1` | Threads per model inference; 1 measured best under concurrent load; `none` for PyTorch's default |
| `CACHEGUARD_FAISS_THREADS` | `1` | Threads per FAISS search |

## Public demo protection

| Variable | Default | Meaning |
|---|---|---|
| `CACHEGUARD_RATE_LIMIT_PER_MINUTE` | `30` | Per client IP on `/query`, `/explain`, `/v1/chat/completions`; `none` disables |
| `CACHEGUARD_DAILY_BACKEND_CALLS` | `500` | AI calls per UTC day; cache hits keep working after it is spent; `none` disables |
| `CACHEGUARD_ADMIN_TOKEN` | (none) | If set, cache admin endpoints and `/admin/whoami` need header `X-Admin-Token` |
| `CACHEGUARD_PROXY_HOPS` | `0` | Hosting proxies in front; the client IP is that many entries from the right of `X-Forwarded-For` (never the forgeable left side) |
| `CACHEGUARD_CLIENT_IP_HEADER` | (none) | A header the proxy sets with the client IP, e.g. `cf-connecting-ip` behind Cloudflare; only safe when the app is reachable solely through that proxy |
| `CACHEGUARD_ALLOW_UNGUARDED` | `on` | Lets `/query` ask for a plain-threshold decision (`"guard": false`) for comparison |

## Newsgroups backend (no Groq key)

| Variable | Default | Meaning |
|---|---|---|
| `CACHEGUARD_DATA_DIR` | `data/20_newsgroups` | Raw dataset folders |
| `CACHEGUARD_ARTIFACT_DIR` | `artifacts/newsgroups` | Built index (`python -m cacheguard.build`) |
| `CACHEGUARD_TOP_K` | `5` | Documents returned per query |
