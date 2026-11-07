# CacheGuard

**A semantic cache that knows when it's wrong.**

CacheGuard sits in front of an answer backend (today a document search, later an LLM via Groq). When a new question means the same thing as one it has already answered, it returns the saved answer instantly instead of calling the backend again. The goal of the project is to do that *without* serving wrong answers for questions that look alike but are different ("cancel my order" vs "don't cancel my order"), and to measure exactly how often that happens.

> **Status: Phase 0 (foundation).** The project grew out of a 20 Newsgroups semantic search system (preserved at the git tag `v0-newsgroups-search`). This phase restructures it, fixes measured bugs, and adds tests. It does not add new features yet. See [Roadmap](#roadmap).

## How a request flows

```
question ─► embed (MiniLM) ─► nearest cached question ─► similarity ≥ threshold?
                                                            │yes          │no
                                                            ▼             ▼
                                                  return cached answer   ask backend
                                                                          │
                                                                  store answer in cache
```

## Quick start

```bash
pip install -e ".[dev]"
```

Download the [20 Newsgroups dataset](https://archive.ics.uci.edu/dataset/113/twenty+newsgroups) and extract it so the folders are at `data/20_newsgroups/<newsgroup>/`.

Build the search index (embeds ~19k documents; takes a few minutes on CPU):

```bash
python -m cacheguard.build
```

Run the API, then open http://127.0.0.1:8000/docs:

```bash
uvicorn cacheguard.api:create_default_app --factory
```

Or use the terminal client:

```bash
python -m cacheguard.cli
```

Run the tests (no model download needed; they use a small fake embedder):

```bash
pytest
```

## API

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/query` | Answer a query, from cache when possible |
| `GET` | `/cache/stats` | Entries, hits, misses, hit rate, threshold |
| `DELETE` | `/cache` | Clear the cache |
| `GET` | `/health` | Liveness and backend name |

`POST /query` with `{"query": "space shuttle launch delays"}` returns:

```json
{
  "query": "space shuttle launch delays",
  "cache_hit": false,
  "matched_query": null,
  "cache_similarity": 0.41,
  "dominant_cluster": 3,
  "answer": "…best matching post…",
  "sources": [{"doc_id": 1234, "newsgroup": "sci.space", "score": 0.62, "snippet": "…"}],
  "latency_ms": 18.4
}
```

`cache_similarity` is the cosine similarity to the nearest cached query, whether or not it was a hit (`null` when the cache is empty). `sources[].score` is the cosine similarity between the query and each document.

## Project layout

```
cacheguard/
  config.py          settings (override with CACHEGUARD_* env vars)
  embeddings.py      sentence-transformers wrapper, unit-normalized output
  vector_index.py    FAISS exact inner-product (cosine) index
  cache.py           baseline semantic cache: linear scan, one global threshold
  clustering.py      Gaussian mixture soft clustering, saved as .npz (no pickle)
  backends/          what answers on a cache miss (newsgroups search for now)
  data/newsgroups.py loading, cleaning and de-duplicating the corpus
  artifacts.py       build/save/load the precomputed index, with a manifest
  service.py         the request path: embed, look up, answer, store
  api.py             FastAPI app
  build.py, cli.py   command-line entry points
tests/               pytest suite
```

## What Phase 0 fixed, and why

These came from measuring the original system rather than reading it:

| Problem found | Evidence | Fix |
|---|---|---|
| **Label leakage.** Cleaning removed only `From:`/`Subject:`, so `Xref`/`Path`/`Newsgroups` headers (which name the newsgroup) stayed in every document and were embedded. They also used up much of MiniLM's 256-token window. | 19,997 of 19,997 cleaned docs still contained the headers. | Drop the whole header block, quoted replies and signatures (`data/newsgroups.py`). |
| **Duplicate documents.** Cross-posted messages appear once per newsgroup. | 663 exact duplicate bodies. | De-duplicate after cleaning. |
| **Misleading score.** On a cache miss, `similarity_score` reported the nearest *cache* similarity, not a retrieval score. | Code review. | Separate `cache_similarity` and per-source `score`, top-k results instead of one. |
| **Unsafe model loading.** The clustering model was a pickle file, which can execute code when loaded. | `joblib.load`. | Store the mixture's arrays in `.npz` and load with `allow_pickle=False`. |
| **Large binaries in git.** ~48 MB of generated files were committed. | `embeddings.npy`, `gmm_model.pkl`. | Untracked; rebuilt by `cacheguard.build` with a manifest that detects stale indexes. |
| **Cache not thread-safe.** FastAPI runs sync endpoints in a thread pool. | Code review. | Lock around cache reads and writes. |
| **Fragile loader.** Bare `except:`, label ids counted stray files, unpinned dependencies, no tests. | Code review. | Specific error handling, sorted folder-based labels, pinned `pyproject.toml`, 28 tests. |

**Known issue, deferred to Phase 3:** the Gaussian mixture is "soft" in name only. With full covariance in 384 dimensions, over 99.9% of documents get more than 0.99 probability for a single cluster. It will be reworked when clusters start driving per-cluster cache thresholds.

## Roadmap

| Phase | Goal |
|---|---|
| 0 | Foundation: restructure, fix measured bugs, tests *(this phase)* |
| 1 | Benchmark harness: paraphrase datasets (QQP, PAWS) and hard negatives; measure the baseline cache's hit rate vs wrong-hit rate |
| 2 | Cache core: FAISS-backed cache, persistence, expiry and eviction, partitioning by model/tenant |
| 3 | Correctness: calibrated and per-cluster thresholds, verifier for borderline matches |
| 4 | Gateway: Groq LLM backend, OpenAI-compatible API, demo chat page |
| 5 | Operations: metrics, dashboard, load tests, Docker, deployment |
| 6 | Write-up: results, design decisions, what didn't work |

## Tech

Python, FastAPI, sentence-transformers (`all-MiniLM-L6-v2`), FAISS, scikit-learn, NumPy, pytest.
