"""Phase 2 scale benchmark: lookup latency as the cache grows.

Compares, at several cache sizes:
* baseline  - the original list cache (NumPy matrix scan)
* store     - CacheStore: FAISS exact inner-product index, one partition
* hnsw      - FAISS HNSW (approximate), for reference only: it does not
              support deletion, so it is not used by the store

Vectors are random unit vectors; exact-search cost does not depend on the
data distribution. Usage:
    python -m cacheguard.bench.scale [--sizes 10000 50000 200000]
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import tempfile
import time
from pathlib import Path

import faiss
import numpy as np

from cacheguard.cache import SemanticCache
from cacheguard.embeddings import normalize_rows
from cacheguard.policy import GlobalThreshold
from cacheguard.store import CacheStore

DIM = 384


def _percentiles(samples_ms: list[float]) -> dict:
    a = np.asarray(samples_ms)
    return {"p50_ms": float(np.percentile(a, 50)), "p99_ms": float(np.percentile(a, 99)), "mean_ms": float(a.mean())}


def _time_lookups(fn, queries: np.ndarray) -> dict:
    for q in queries[:20]:  # warm up
        fn(q)
    samples = []
    for q in queries:
        start = time.perf_counter()
        fn(q)
        samples.append((time.perf_counter() - start) * 1000)
    return _percentiles(samples)


def bench_size(n: int, n_queries: int, rng: np.random.Generator, with_hnsw: bool, persist_dir: Path) -> dict:
    vectors = normalize_rows(rng.normal(size=(n, DIM)).astype(np.float32))
    queries = normalize_rows(rng.normal(size=(n_queries, DIM)).astype(np.float32))
    result: dict = {"entries": n}

    baseline = SemanticCache(threshold=0.85)
    for v in vectors:
        baseline.add("q", v, None)
    result["baseline"] = _time_lookups(baseline.lookup, queries)
    del baseline
    gc.collect()

    store = CacheStore(DIM, GlobalThreshold(0.85))
    start = time.perf_counter()
    for i, v in enumerate(vectors):
        store.add("p", f"q{i}", v, i)
    add_s = time.perf_counter() - start
    result["store"] = _time_lookups(lambda q: store.lookup("p", q), queries)
    result["store"]["add_us"] = add_s / n * 1e6
    store.close()
    del store
    gc.collect()

    # Restart cost: rebuild the in-memory index from SQLite.
    db = persist_dir / f"scale_{n}.sqlite3"
    persisted = CacheStore(DIM, GlobalThreshold(0.85), db_path=db)
    with persisted._lock:  # bulk insert in one transaction; per-add commits would dominate
        for i, v in enumerate(vectors):
            persisted._db.execute(
                "INSERT INTO entries (entry_id, partition, query, embedding, response, cluster, tags, created_at)"
                " VALUES (?, 'p', ?, ?, ?, NULL, '[]', ?)",
                (i + 1, f"q{i}", v.tobytes(), json.dumps(i), time.time()),
            )
        persisted._db.commit()
    persisted.close()
    start = time.perf_counter()
    restored = CacheStore(DIM, GlobalThreshold(0.85), db_path=db)
    result["restore_s"] = time.perf_counter() - start
    assert len(restored) == n
    restored.close()
    del restored
    gc.collect()

    if with_hnsw:
        hnsw = faiss.IndexHNSWFlat(DIM, 32, faiss.METRIC_INNER_PRODUCT)
        hnsw.hnsw.efSearch = 64
        start = time.perf_counter()
        hnsw.add(vectors)
        build_s = time.perf_counter() - start
        result["hnsw"] = _time_lookups(lambda q: hnsw.search(q.reshape(1, -1), 1), queries)
        result["hnsw"]["build_s"] = build_s
        exact = faiss.IndexFlatIP(DIM)
        exact.add(vectors)
        _, truth = exact.search(queries, 1)
        _, approx = hnsw.search(queries, 1)
        result["hnsw"]["recall_at_1"] = float(np.mean(truth[:, 0] == approx[:, 0]))
        del hnsw, exact
    del vectors
    gc.collect()
    return result


def render(report: dict) -> str:
    lines = [
        "# Phase 2: lookup latency at scale",
        "",
        f"Single-query lookups, {report['dim']}-dim vectors, {report['queries']} queries per size, "
        f"{report['machine']}. FAISS threads: {report['faiss_threads']}.",
        "",
        "| Cached entries | Baseline list scan p50 / p99 | CacheStore (FAISS exact) p50 / p99 | Store add | Restart from SQLite | HNSW p50 / p99 (recall@1, build) |",
        "|---|---|---|---|---|---|",
    ]
    for r in report["results"]:
        h = r.get("hnsw")
        hnsw = f"{h['p50_ms']:.3f} / {h['p99_ms']:.3f} ms ({h['recall_at_1']:.0%}, {h['build_s']:.1f}s)" if h else "-"
        lines.append(
            f"| {r['entries']:,} | {r['baseline']['p50_ms']:.2f} / {r['baseline']['p99_ms']:.2f} ms | "
            f"{r['store']['p50_ms']:.2f} / {r['store']['p99_ms']:.2f} ms | {r['store']['add_us']:.0f} µs | "
            f"{r['restore_s']:.2f} s | {hnsw} |"
        )
    lines += [
        "",
        "Notes:",
        "- Store lookups include the lock, expiry check and SQLite hit-count write when they hit; these random queries all miss.",
        "- HNSW recall on random vectors is a pessimistic case; real embeddings cluster and usually score higher.",
        "- FAISS runs single-threaded here, as in the server. With 12 OpenMP threads a single-query search paid "
        "~0.5 ms of thread start-up (0.56 ms vs 0.05 ms at 2k entries), so parallelism is left to concurrent requests.",
        "- Sizes stop at 200k because the benchmark machine had under 1 GB of free RAM; exact search is linear, "
        "so 1M entries would take roughly 5x the 200k time.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[10_000, 50_000, 200_000])
    parser.add_argument("--queries", type=int, default=500)
    parser.add_argument("--no-hnsw", action="store_true")
    parser.add_argument("--out", default="benchmarks/results/phase2_scale")
    parser.add_argument("--faiss-threads", type=int, default=1, help="1 matches the server default")
    args = parser.parse_args(argv)
    faiss.omp_set_num_threads(args.faiss_threads)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    report = {
        "dim": DIM,
        "queries": args.queries,
        "machine": f"{platform.processor() or platform.machine()}, Python {platform.python_version()}",
        "faiss_threads": faiss.omp_get_max_threads(),
        "results": [],
    }
    with tempfile.TemporaryDirectory() as tmp:
        for n in args.sizes:
            print(f"Benchmarking {n:,} entries...")
            r = bench_size(n, args.queries, rng, not args.no_hnsw, Path(tmp))
            print(f"  baseline p50 {r['baseline']['p50_ms']:.2f} ms | store p50 {r['store']['p50_ms']:.2f} ms")
            report["results"].append(r)
    (out / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out / "README.md").write_text(render(report), encoding="utf-8")
    print(f"Wrote {out}/README.md")


if __name__ == "__main__":
    main()
