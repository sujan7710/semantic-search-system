"""Phase 5 load test against a running server.

Start the server with the mock backend and limits off, e.g.:

    CACHEGUARD_BACKEND=mock CACHEGUARD_CACHE_DB_PATH=:memory: \
    CACHEGUARD_RATE_LIMIT_PER_MINUTE=off CACHEGUARD_DAILY_BACKEND_CALLS=off \
    uvicorn cacheguard.api:create_default_app --factory --port 8000

then:  python -m cacheguard.bench.loadtest --url http://127.0.0.1:8000

The workload is real paraphrase groups from QQP (test half): 40% of the
questions have an earlier paraphrase in the run, so roughly that share
could be served from cache. Each concurrency level uses a fresh namespace,
so every level starts with an empty cache and sees the same questions.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import random
import time
from collections import Counter
from pathlib import Path

import httpx
import numpy as np


def build_questions(data_cache: str, n_groups: int, n_singletons: int, seed: int) -> tuple[list[str], list[int]]:
    from cacheguard.bench.datasets import load_qqp, question_groups

    pairs = load_qqp(data_cache, "test")
    groups = question_groups(pairs.text_a, pairs.text_b, pairs.labels)
    by_group: dict[int, list[str]] = {}
    for q, g in groups.items():
        by_group.setdefault(g, []).append(q)
    rng = random.Random(seed)
    multi = [qs for qs in by_group.values() if len(qs) >= 2]
    single = [qs[0] for qs in by_group.values() if len(qs) == 1]
    rng.shuffle(multi)
    rng.shuffle(single)
    chosen = [(q, i) for i, qs in enumerate(multi[:n_groups]) for q in qs[:3]]
    chosen += [(q, n_groups + i) for i, q in enumerate(single[:n_singletons])]
    rng.shuffle(chosen)
    return [q for q, _ in chosen], [g for _, g in chosen]


async def run_level(client: httpx.AsyncClient, questions: list[str], concurrency: int, namespace: str) -> dict:
    queue: asyncio.Queue = asyncio.Queue()
    for i, q in enumerate(questions):
        queue.put_nowait((i, q))
    records: list[dict] = [None] * len(questions)  # type: ignore[list-item]

    async def worker():
        while True:
            try:
                i, q = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            start = time.perf_counter()
            try:
                r = await client.post("/query", json={"query": q, "namespace": namespace})
                ms = (time.perf_counter() - start) * 1000
                body = r.json() if r.status_code == 200 else {}
                records[i] = {"status": r.status_code, "ms": ms, "hit": body.get("cache_hit"),
                              "stage": body.get("decision", {}).get("stage"),
                              "decision_ms": body.get("decision", {}).get("decision_ms")}
            except httpx.HTTPError as exc:
                records[i] = {"status": type(exc).__name__, "ms": (time.perf_counter() - start) * 1000}

    start = time.perf_counter()
    await asyncio.gather(*(worker() for _ in range(concurrency)))
    wall = time.perf_counter() - start
    return {"records": records, "wall_s": wall}


def summarize(level: dict, groups: list[int], concurrency: int) -> dict:
    recs = level["records"]
    ok = [r for r in recs if r["status"] == 200]
    pct = lambda xs, p: float(np.percentile(xs, p)) if xs else None  # noqa: E731
    by = lambda pred: [r["ms"] for r in ok if pred(r)]  # noqa: E731
    hits = [r for r in ok if r["hit"]]
    # Questions whose meaning group was already asked earlier in the list: an ideal
    # sequential cache could answer them. With concurrency, some arrive while the
    # first copy is still waiting on the backend and miss anyway (cache stampede).
    seen, repeat = set(), 0
    for g in groups:
        repeat += g in seen
        seen.add(g)
    decision = [r["decision_ms"] for r in ok if r.get("stage") == "verifier"]
    return {
        "concurrency": concurrency,
        "requests": len(recs),
        "errors": dict(Counter(str(r["status"]) for r in recs if r["status"] != 200)),
        "throughput_rps": round(len(ok) / level["wall_s"], 2),
        "wall_s": round(level["wall_s"], 1),
        "hit_rate": round(len(hits) / len(ok), 3) if ok else 0,
        "repeat_share": round(repeat / len(groups), 3),
        "backend_calls": len(ok) - len(hits),
        "p50_ms": pct([r["ms"] for r in ok], 50),
        "p95_ms": pct([r["ms"] for r in ok], 95),
        "p99_ms": pct([r["ms"] for r in ok], 99),
        "hit_p50_ms": pct(by(lambda r: r["hit"]), 50),
        "hit_p95_ms": pct(by(lambda r: r["hit"]), 95),
        "miss_p50_ms": pct(by(lambda r: not r["hit"]), 50),
        "miss_p95_ms": pct(by(lambda r: not r["hit"]), 95),
        "verifier_share": round(len(decision) / len(ok), 3) if ok else 0,
        "verifier_decision_p50_ms": pct(decision, 50),
        "verifier_decision_p95_ms": pct(decision, 95),
    }


def render(report: dict) -> str:
    f = lambda x: "-" if x is None else f"{x:.0f}"  # noqa: E731
    lines = [
        "# Phase 5: load test",
        "",
        f"{report['questions']} QQP questions ({report['groups']} paraphrase groups + singletons; "
        f"{report['levels'][0]['repeat_share']:.0%} have an earlier paraphrase in the run). Mock LLM backend with "
        f"{report['mock_delay_ms']:.0f} ms latency; guard on; one server process on {report['machine']}. "
        "Each level starts with an empty cache.",
        "",
        "| Concurrency | Throughput | Hit rate | AI calls | p50 / p95 / p99 (all) | Hit p50 / p95 | Miss p50 / p95 | Verifier p50 / p95 | Errors |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for s in report["levels"]:
        lines.append(
            f"| {s['concurrency']} | {s['throughput_rps']:.1f} req/s | {s['hit_rate']:.1%} | {s['backend_calls']} | "
            f"{f(s['p50_ms'])} / {f(s['p95_ms'])} / {f(s['p99_ms'])} ms | {f(s['hit_p50_ms'])} / {f(s['hit_p95_ms'])} ms | "
            f"{f(s['miss_p50_ms'])} / {f(s['miss_p95_ms'])} ms | {f(s['verifier_decision_p50_ms'])} / {f(s['verifier_decision_p95_ms'])} ms | "
            f"{sum(s['errors'].values()) or 0} |"
        )
    lines.append("")
    return "\n".join(lines)


async def main_async(args) -> None:
    questions, groups = build_questions(args.data_cache, args.groups, args.singletons, args.seed)
    async with httpx.AsyncClient(base_url=args.url, timeout=120) as client:
        health = (await client.get("/health")).json()
        stats = (await client.get("/stats")).json()
        report = {
            "url": args.url,
            "backend": health["backend"],
            "mock_delay_ms": stats["backend"].get("delay_ms", args.mock_delay_ms),
            "guard": health.get("guard"),
            "questions": len(questions),
            "groups": args.groups,
            "machine": f"{platform.processor() or platform.machine()}",
            "levels": [],
        }
        for c in args.concurrency:
            ns = f"load-{c}-{int(time.time())}"
            print(f"concurrency {c}: {len(questions)} requests...")
            level = await run_level(client, questions, c, ns)
            s = summarize(level, groups, c)
            report["levels"].append(s)
            print(f"  {s['throughput_rps']} req/s, hit rate {s['hit_rate']:.1%}, p50 {s['p50_ms']:.0f} ms, p95 {s['p95_ms']:.0f} ms, errors {s['errors']}")
        report["metrics_sample"] = (await client.get("/metrics")).text[:4000]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out / "README.md").write_text(render(report), encoding="utf-8")
    print(f"Wrote {out}/README.md")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 16])
    parser.add_argument("--groups", type=int, default=120)
    parser.add_argument("--singletons", type=int, default=120)
    parser.add_argument("--mock-delay-ms", type=float, default=600.0, help="for the report only; set on the server")
    parser.add_argument("--data-cache", default="artifacts/bench/raw")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="benchmarks/results/phase5_load")
    asyncio.run(main_async(parser.parse_args(argv)))


if __name__ == "__main__":
    main()
