# Phase 5: load test

375 QQP questions (120 paraphrase groups + singletons; 36% have an earlier paraphrase in the run). Mock LLM backend with 600 ms latency (no API cost, no provider rate limits); guard on; verifier memo off (each level replays the same questions, and memoized scores would hide the verifier's cost); one uvicorn process on a 12-thread AMD laptop CPU. Every level starts with an empty cache.

Reproduce: start the server as described in [loadtest.py](../../../cacheguard/bench/loadtest.py), then `python -m cacheguard.bench.loadtest`.

## PyTorch threads per inference: PyTorch default (12 threads)

| Concurrency | Throughput | Hit rate | AI calls | p50 / p95 / p99 (all) | Hit p50 / p95 | Miss p50 / p95 | Verifier p50 / p95 | Errors |
|---|---|---|---|---|---|---|---|---|
| 1 | 2.1 req/s | 27.2% | 273 | 621 / 656 / 663 ms | 53 / 70 ms | 622 / 659 ms | 33 / 44 ms | 0 |
| 4 | 8.2 req/s | 26.4% | 276 | 624 / 665 / 687 ms | 63 / 87 ms | 629 / 667 ms | 36 / 56 ms | 0 |
| 16 | 26.0 req/s | 24.3% | 284 | 663 / 884 / 1041 ms | 200 / 431 ms | 692 / 956 ms | 90 / 231 ms | 0 |
| 32 | 26.6 req/s | 23.2% | 288 | 1118 / 1708 / 2084 ms | 869 / 1510 ms | 1137 / 1760 ms | 307 / 704 ms | 0 |

## PyTorch threads per inference: 2 threads

| Concurrency | Throughput | Hit rate | AI calls | p50 / p95 / p99 (all) | Hit p50 / p95 | Miss p50 / p95 | Verifier p50 / p95 | Errors |
|---|---|---|---|---|---|---|---|---|
| 1 | 2.1 req/s | 27.2% | 273 | 621 / 662 / 670 ms | 61 / 73 ms | 622 / 664 ms | 41 / 51 ms | 0 |
| 16 | 26.3 req/s | 24.5% | 283 | 651 / 873 / 1034 ms | 224 / 522 ms | 671 / 883 ms | 133 / 284 ms | 0 |
| 32 | 26.2 req/s | 23.2% | 288 | 1099 / 1789 / 2174 ms | 1033 / 1713 ms | 1104 / 1931 ms | 404 / 869 ms | 0 |

## PyTorch threads per inference: 1 thread (shipped default)

| Concurrency | Throughput | Hit rate | AI calls | p50 / p95 / p99 (all) | Hit p50 / p95 | Miss p50 / p95 | Verifier p50 / p95 | Errors |
|---|---|---|---|---|---|---|---|---|
| 1 | 2.1 req/s | 27.2% | 273 | 624 / 678 / 691 ms | 75 / 92 ms | 628 / 680 ms | 53 / 67 ms | 0 |
| 16 | 29.2 req/s | 24.5% | 283 | 633 / 773 / 847 ms | 120 / 203 ms | 642 / 813 ms | 81 / 145 ms | 0 |
| 32 | 37.7 req/s | 22.4% | 291 | 798 / 1202 / 1351 ms | 503 / 944 ms | 827 / 1233 ms | 231 / 465 ms | 0 |

## What this showed

- **The bottleneck is CPU inference, not the cache.** With PyTorch's default, every model call uses all 12 CPU threads. At 32 concurrent requests that is hundreds of threads on 12 cores: throughput flattened at ~26 req/s and a cache *hit* took 869 ms (p50), longer than the 600 ms AI call it replaces.
- **One thread per inference fixed most of it.** Requests already run in parallel in the server's thread pool, so per-call parallelism only adds contention. At 32 concurrent users: 37.7 req/s (+42%), hit p50 503 ms (-42%). The price is a slightly slower verifier for a lone user (53 ms vs 33 ms), small next to a ~600 ms AI call. Two threads was no better than the default. This is the same lesson as FAISS in Phase 2.
- **Still CPU-bound at 32 users.** The next step would be micro-batching: collect verifier pairs from concurrent requests for a few milliseconds and score them in one batch, or run the models on a GPU or as a separate inference service.
- **Cache stampede.** AI calls rise with concurrency (273 at 1 user, 284-288 at 16-32) and hit rate drops from 27.2% to 22-23%: when paraphrases arrive while the first copy is still waiting for the AI, they all miss. Request coalescing (letting later near-duplicates wait for the in-flight answer) would recover these; not implemented.
- **Hit rate vs opportunity.** 36% of questions had an earlier paraphrase, 27% were served from cache. The gap is the guard being strict: its 5% error budget requires verifier score 0.9.
- No errors in 3,750 requests across the three runs.
