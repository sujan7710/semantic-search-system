# CacheGuard: design, results and what didn't work

This is the engineering story behind CacheGuard, phase by phase: what was measured, what was decided and why, and
what failed along the way. The [README](../README.md) has the short version.

**Contents**

1. [Starting point](#1-starting-point-a-semantic-search-project-measured-not-read)
2. [Measuring the problem](#2-measuring-the-problem)
3. [The cache store](#3-the-cache-store)
4. [The guard](#4-the-guard)
5. [The gateway and demo](#5-the-gateway-and-demo)
6. [Operations](#6-operations)
7. [What didn't work](#7-what-didnt-work)
8. [Limitations and next steps](#8-limitations-and-next-steps)
9. [Lessons](#9-lessons)

---

## 1. Starting point: a semantic search project, measured, not read

CacheGuard grew out of a 20 Newsgroups semantic search system (preserved at the git tag `v0-newsgroups-search`): MiniLM
embeddings, a FAISS index, Gaussian-mixture "fuzzy" clustering, and a semantic cache with a fixed 0.85 threshold.
Before changing anything, the existing system was measured. That turned up problems a code read would have missed:

| Problem | Evidence | Fix |
|---|---|---|
| **Label leakage.** Cleaning removed only `From:`/`Subject:`, so `Xref`/`Path`/`Newsgroups` headers, which name the newsgroup, stayed in every document and were embedded. They also used up much of MiniLM's 256-token window. | 19,997 of 19,997 cleaned documents still contained them. | Drop the whole header block, quoted replies and signatures. |
| **"Soft" clustering that wasn't.** A full-covariance mixture in 384 dimensions. | 99.97% of documents got > 0.99 probability for one cluster. | Documented; per-cluster cache settings later used k-means on questions instead. |
| **Duplicate documents.** Cross-posts appear once per newsgroup. | 663 exact duplicate bodies. | De-duplicate after cleaning. |
| **Misleading score.** On a miss, the API's `similarity_score` was the nearest *cache* entry's, not a retrieval score. | Code review. | Separate cache similarity and per-source scores. |
| **Unsafe model loading.** The clustering model was a pickle, which can run code when loaded. | `joblib.load`. | Arrays in `.npz`, loaded with `allow_pickle=False`. |
| **Generated binaries in git**, a cache that wasn't thread-safe, a bare `except:`, unpinned dependencies, no tests. | Code review. | Untracked (rebuilt with a manifest that detects stale indexes), a lock, specific errors, pinned versions, tests. |

Effect of the leak fix: the share of a document's 10 nearest neighbours from the same newsgroup fell from
**0.77 to 0.67**. About 10 points of the original "quality" came from the leaked headers.

The most valuable part of the old project was the semantic cache, and the most interesting question about it was one
nobody had asked: **how often does it return the wrong answer?** That became the project.

## 2. Measuring the problem

A plain semantic cache reuses an answer when two questions are similar enough (cosine similarity ≥ a threshold).
The [benchmark harness](../cacheguard/bench/) tests that on:

- **QQP** (Quora Question Pairs, GLUE validation split): real user questions labeled duplicate or not. The validation
  set is split into calibration and test halves *by question group*, so no question appears on both sides.
- **PAWS**: pairs with nearly identical words and often different meaning (word-order swaps).
- **2,000 generated look-alikes**: one edit that changes the answer (negation, a number, a name, an antonym).
- **Workload replay**: a 26,858-question stream through the actual cache, so a wrong entry can be served many times,
  as in real traffic.

Results for the original 0.85 threshold ([report](../benchmarks/results/phase1_baseline/README.md)):

| At threshold 0.85 | Result |
|---|---|
| Negated question reused ("How do I *not* cancel…" → "How do I cancel…") | **99.4%** |
| Swapped number reused ("5 hours" → "10 hours") | **90.8%** |
| Swapped antonym reused ("best" → "worst") | **86.2%** |
| PAWS word-order pairs | AUC **0.63**, barely better than chance |
| Judged cache answers that were for a different question (replay) | **32.0%** |

No single threshold fixes this. On the test stream, every threshold from 0.75 to 0.99 serves 32–39% wrong answers
among judged hits. The median negated question scores 0.979 against its original and the median swapped number
0.952, while only 20% of real paraphrases reach 0.95: the dangerous look-alikes are *more* similar than most real
paraphrases.

**How errors are counted (a correction made mid-project).** QQP annotators labeled specific pairs, not all pairs. The
first replay counted every hit between questions from different labeled pairs as wrong and reported 43.5% at 0.85.
A manual check later showed that 64% of high-similarity "errors" were pairs annotators never compared, and most were
real paraphrases ("How do I increase my level of confidence?" / "How can I improve my confidence?"). All final
numbers therefore judge only pairs the labels cover:

- **correct**: same labeled duplicate group;
- **wrong**: labeled "not duplicate", or a generated look-alike of the cached question;
- **unjudged**: anything else, excluded from error rates (70–90% of hits are judged).

## 3. The cache store

The original cache was a Python list scanned on every request, lost on restart, unbounded and shared by everyone.
[`CacheStore`](../cacheguard/store.py) replaced it:

| Concern | Design | Why |
|---|---|---|
| Isolation | One FAISS index per **partition**: namespace, backend, model, system prompt, temperature, max tokens, and a hash of earlier conversation turns, all matched exactly. | Similarity must never be what decides whether one tenant gets another tenant's answer. Only the last question is matched by meaning. |
| Lookup | `IndexIDMap2(IndexFlatIP)`: exact inner product on unit vectors (= cosine). | ID mapping supports deletion, needed for eviction and invalidation. |
| Freshness | Per-entry TTL, checked lazily; the 8 nearest candidates are fetched so an expired nearest entry falls through. | No background sweeper thread. |
| Size | LRU eviction past `max_entries`; a hit refreshes recency. | Bounded memory under any traffic. |
| Invalidation | Tags: `namespace:…`, `model:…`, `source:<doc>`. | One call drops every answer derived from a changed source or model. |
| Durability | SQLite write-through (WAL); indexes rebuilt on startup in LRU order. | A restart keeps the cache warm. |
| Concurrency | `nearest()` and `record()` are separate calls; the guard decides in between, outside the lock. | A 30 ms verifier call must not serialize all requests. |

[Scale benchmark](../benchmarks/results/phase2_scale/README.md), single-query lookup, 384 dimensions:

| Cached entries | Old list scan p50 | CacheStore p50 | Restart from SQLite |
|---|---|---|---|
| 10,000 | 0.55 ms | 0.48 ms | 0.19 s |
| 50,000 | 5.15 ms | 3.76 ms | 1.06 s |
| 200,000 | 22.13 ms | 14.52 ms | 5.00 s |

Decisions the numbers made:

- **Exact search stays.** Both are linear; FAISS is 1.1–1.5x faster, not a different complexity class. Embedding the
  question takes ~10–20 ms and an LLM call hundreds, so exact search is fine for tens of thousands of entries per
  partition, which eviction and partitioning keep it under.
- **HNSW was measured and rejected.** 0.3–0.6 ms at every size, but it found the true nearest neighbour only 66% (10k)
  to 7% (200k) of the time on these vectors, takes minutes to build and cannot delete. For a cache, a missed neighbour
  is a missed hit (wasted money), not a wrong answer, so it is a cost trade-off to revisit only at much larger scale.
- **One FAISS thread per search.** With 12 OpenMP threads, a single-query search paid ~0.5 ms of thread start-up,
  10x the search itself at 2k entries. Parallelism comes from concurrent requests instead.
- **No duplicate vectors.** Sizing the benchmark showed entries kept a second copy of every vector; now vectors live
  only in FAISS and SQLite.

## 4. The guard

The [guard](../cacheguard/guard.py) decides whether the nearest cached question may answer the new one, cheapest
check first:

1. **Gate.** Below a similarity floor: a miss, no further work.
2. **Rules** ([verify.py](../cacheguard/verify.py), ~0.1 ms). Reject when negation differs, or when each side has a
   number or capitalized name the other lacks.
3. **Cross-encoder** (`cross-encoder/quora-distilroberta-base`, ~30 ms on CPU). Reads both questions together
   instead of comparing two separate embeddings. On the same 1,000 QQP calibration pairs: AUC **0.961** vs 0.875 for
   embedding similarity.

The thresholds are not hand-picked. [guard_eval.py](../cacheguard/bench/guard_eval.py) scores a grid of 348 settings
on the calibration stream and, for each error budget, ships the setting with the most hits within budget. The
calibration and test streams share no questions.

Results on the test stream, the same one the baseline was measured on ([report](../benchmarks/results/phase3_guard/README.md)):

| | Correct answers from cache | Wrong answers from cache | Look-alikes served wrong |
|---|---|---|---|
| Plain cache, threshold 0.85 | 4,377 | 2,061 | 1,303 of 1,998 |
| **Guard, 5% budget (shipped)** | 4,231 | **241 (−88%)** | **74 of 1,998** |
| Guard, 10% budget | 5,123 (+17%) | 634 (−69%) | 326 of 1,998 |

The 5% budget was chosen on calibration data and held on test (5.4% measured). The cost: the verifier runs on ~60% of
lookups.

What the ablations showed:

- **The cross-encoder does most of the work.** Without rules it reaches 23.2% hits at 6.6% judged error (5% budget).
- **Rules mainly cut look-alikes** (178 → 74 served wrong), which is partly circular: the look-alike generator makes
  the same edits the rules check. At 10% and 15% budgets, calibration switched the rules off.
- **Rules alone are not enough**: rules plus the best threshold reached 10.3% hits at a 15% budget.

## 5. The gateway and demo

- **Groq backend** ([groq.py](../cacheguard/backends/groq.py)): plain `httpx` against Groq's OpenAI-compatible API;
  retries with backoff on 429/5xx and timeouts (honoring `Retry-After`, capped at 5 s), none on client errors; token
  usage and cost stored with each answer, so a later hit reports exactly what it saved.
- **OpenAI-compatible endpoint** (`/v1/chat/completions`): any OpenAI client works by changing its base URL. Requests
  with a different temperature bypass the cache, since cached answers assume deterministic generation.
- **Demo page** ([demo.html](../cacheguard/static/demo.html)): a chat with a live "Why?" panel: nearest cached
  question, similarity against the gate, and which check passed or failed. Before each question it calls `/explain`,
  a dry run, so it can also say what a plain cache *would* have done, e.g. *"would have answered with the cached
  reply to 'Is 5 hours of sleep enough?'. That would have been the wrong answer."* Model answers are rendered from
  Markdown into text nodes only, so model output cannot inject markup.
- **Public demo protection**: per-IP rate limit; a daily AI-call budget that cache hits don't consume (and failed
  calls refund); admin token with constant-time comparison; visitors can only reset their own namespace.

Live run against Groq (`openai/gpt-oss-20b`, fresh cache):

| Question, in order | Result |
|---|---|
| How do I make my laptop battery last longer? | AI call (432 tokens, 595 ms) |
| How can I extend my laptop's battery life? | **cache hit**, verifier 0.975, **10 ms** |
| How do I make my phone battery last longer? | refused (verifier 0.002); a plain cache would have served the laptop answer |
| Why doesn't my laptop battery last long? | refused by the negation rule |
| Is 8 hours of sleep enough? (after "5 hours") | refused by the number rule; a plain cache would have served the 5-hour answer |
| Is five hours of sleep sufficient? | **cache hit**, verifier 0.988 ("five" = "5") |

## 6. Operations

- **Metrics** at `/metrics` in Prometheus format (written without a client library): lookups by outcome, stage and
  reason; latency histograms for requests, decisions and AI calls; errors; limit rejections; savings.
- **Logs**: one JSON line per question. The question text is never logged, only a 12-character hash.
- **Docker image**: CPU-only PyTorch, models baked in with the Hub offline at runtime, non-root, health check.
- **CI** ([ci.yml](../.github/workflows/ci.yml)): tests on CPU-only PyTorch with fakes (no downloads, no keys), then
  builds the image and checks the running container.

[Load test](../benchmarks/results/phase5_load/README.md) (real paraphrase groups, mock LLM with 600 ms latency):

| Concurrent users | Throughput | Cache-hit p50 | Verifier p50 |
|---|---|---|---|
| 1 | 2.1 req/s | 75 ms | 53 ms |
| 16 | 29.2 req/s | 120 ms | 81 ms |
| 32 | 37.7 req/s | 503 ms | 231 ms |

The first run found the real bottleneck: with PyTorch's default of all 12 threads per inference, throughput flattened
at ~26 req/s and a cache **hit** at 32 users took 869 ms, longer than the AI call it replaced. One thread per
inference (now the default) gave **+42% throughput** and halved hit latency.

**Deployment** ([DEPLOY.md](DEPLOY.md)): the public demo runs through a Cloudflare quick tunnel from a PC; a Cloud Run
script is ready for when a billing account exists. Getting there exposed three bugs that tests had not:

- A clean Docker build pulled newer `transformers`/`huggingface_hub` that fail to import with PyTorch 2.5.1:
  the model stack is now pinned to the tested versions.
- `docker --env-file` keeps quotes, so Groq rejected `GROQ_API_KEY="…"`: secrets are stripped of surrounding quotes.
- Rate limiting trusted the *first* `X-Forwarded-For` entry, and the server trusted forwarded headers from anyone,
  so a visitor could forge a new IP per request. The client IP now comes from the proxy's own entry (counted from the
  right) or a header the proxy sets (`CF-Connecting-IP`). Verified live: Cloudflare rejects a forged
  `CF-Connecting-IP`, and a forged `X-Forwarded-For` is ignored.

## 7. What didn't work

| Tried | What happened | Kept? |
|---|---|---|
| Counting every unlabeled hit as wrong | Inflated errors: 64% of high-similarity "errors" were unlabeled paraphrases | Replaced by judging labeled pairs only |
| Rules that rejected *any* differing number or name | Threw away 37% of real paraphrases ("best laptop?" vs "best laptop under 40k?") | Replaced by "swap" rules (10%) |
| Per-cluster guard settings (k-means, 8 clusters) | Within ±1.5 points of the global setting on test, better at one budget and worse at another: calibration noise | Not shipped |
| The mixture-model clusters from the original project | Posteriors saturated (> 0.99), so not usable as soft clusters | Only labels newsgroup topics |
| HNSW for cache lookup | Fast, but 7–66% nearest-neighbour recall and no deletion | Exact search kept |
| PyTorch's default threading | Oversubscribed the CPU under concurrent load | One thread per inference |
| First load test | The verifier's score memo made later levels look 0 ms | Rerun with the memo off |
| Free Docker hosting | Hugging Face Docker Spaces now need a paid plan; free tiers cap memory at 512 MB | Cloudflare tunnel for the demo |
| Installing the Cloud CLI to deploy | The installer stalled, then the account had no billing account | Deploy script kept for later |

## 8. Limitations and next steps

- **In-domain verifier.** The cross-encoder was trained on QQP's train split; tests use the separate validation
  split, so results are in-domain. Other domains need their own calibration data.
- **Word-order paraphrases are unsolved.** On PAWS the cross-encoder scores AUC 0.57, worse than embeddings (0.63).
- **Conservative misses.** At the 5% budget the verifier needs a score of 0.9; "How can I cancel an order I placed?"
  scored 0.82 against "How do I cancel my order?" and was not reused.
- **CPU-bound at high concurrency.** Next: micro-batching verifier calls across requests, a quantized verifier, or a
  separate inference service.
- **Cache stampede.** Paraphrases arriving while the first copy waits for the AI all miss (273 → 288 AI calls from 1 to
  32 users). Next: request coalescing, letting near-duplicates wait for the in-flight answer.
- **Memory.** 0.4–1 GB with both models. Running them through ONNX Runtime without PyTorch would fit 512 MB hosts,
  but the guard would need re-calibrating with the converted verifier.
- **Single process.** Rate limits, budget and counters are in memory, per process.

## 9. Lessons

1. **Measure before believing.** Label leakage, saturated clusters, the inflated error rate and the CPU bottleneck
   were all invisible from reading code; each showed up only when measured.
2. **Treat a threshold as a trade-off, not a constant.** Choosing settings for a stated error budget on held-out data,
   and checking they hold on test, turned "0.85 seems fine" into a number that can be defended.
3. **Question the ground truth too.** The largest single correction in the project came from checking the labels, not
   the model.
4. **The same lesson twice: thread oversubscription.** FAISS and PyTorch both defaulted to all cores per call, which
   hurts a server that already runs requests in parallel.
5. **The first clean build and first real deploy find bugs tests don't.** Unpinned transitive dependencies, quoted
   secrets and forgeable client IPs were all found that way.
