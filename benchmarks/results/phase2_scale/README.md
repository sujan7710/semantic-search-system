# Phase 2: lookup latency at scale

Single-query lookups, 384-dim vectors, 500 queries per size, AMD64 Family 25 Model 80 Stepping 0, AuthenticAMD, Python 3.12.0. FAISS threads: 1.

| Cached entries | Baseline list scan p50 / p99 | CacheStore (FAISS exact) p50 / p99 | Store add | Restart from SQLite | HNSW p50 / p99 (recall@1, build) |
|---|---|---|---|---|---|
| 10,000 | 0.55 / 1.00 ms | 0.48 / 0.76 ms | 44 µs | 0.19 s | 0.267 / 0.393 ms (66%, 1.5s) |
| 50,000 | 5.15 / 5.62 ms | 3.76 / 4.67 ms | 45 µs | 1.06 s | 0.495 / 0.721 ms (27%, 25.6s) |
| 200,000 | 22.13 / 22.85 ms | 14.52 / 22.63 ms | 45 µs | 5.00 s | 0.611 / 0.966 ms (7%, 176.1s) |

Notes:
- Store lookups include the lock, expiry check and SQLite hit-count write when they hit; these random queries all miss.
- HNSW recall on random vectors is a pessimistic case; real embeddings cluster and usually score higher.
- FAISS runs single-threaded here, as in the server. With 12 OpenMP threads a single-query search paid ~0.5 ms of thread start-up (0.56 ms vs 0.05 ms at 2k entries), so parallelism is left to concurrent requests.
- Sizes stop at 200k because the benchmark machine had under 1 GB of free RAM; exact search is linear, so 1M entries would take roughly 5x the 200k time.
