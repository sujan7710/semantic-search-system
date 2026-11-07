"""Run the Phase 1 baseline benchmark and write a report.

Usage:  python -m cacheguard.bench.run [--out benchmarks/results/phase1_baseline]
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

from cacheguard.bench.datasets import load_paws, load_qqp
from cacheguard.bench.hard_negatives import generate_hard_negatives
from cacheguard.bench.pairs import auc, calibrate_threshold, evaluate_threshold, pair_similarities, sweep
from cacheguard.bench.workload import build_workload, replay
from cacheguard.config import Settings
from cacheguard.embeddings import SentenceTransformerEmbedder

BASELINE_THRESHOLD = 0.85
ERROR_BUDGETS = [0.01, 0.02, 0.05, 0.10]
REPLAY_THRESHOLDS = [0.75, 0.80, 0.85, 0.90, 0.95]


def pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def main(argv: list[str] | None = None) -> None:
    settings = Settings.from_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="benchmarks/results/phase1_baseline")
    parser.add_argument("--data-cache", default="artifacts/bench/raw")
    parser.add_argument("--hard-negatives-per-kind", type=int, default=500)
    parser.add_argument("--max-queries", type=int, default=None, help="cap natural queries in the replay stream")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render-only", action="store_true", help="rebuild README.md and plot from results.json")
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.render_only:
        report = json.loads((out / "results.json").read_text(encoding="utf-8"))
        stream_len = report["replay"][0]["queries"] if report["replay"] else 0
        (out / "README.md").write_text(render_markdown(report, stream_len), encoding="utf-8")
        plot(report, out / "tradeoff.png")
        return
    t0 = time.perf_counter()

    print("Loading datasets...")
    data = {
        "qqp-calibration": load_qqp(args.data_cache, "calibration"),
        "qqp-test": load_qqp(args.data_cache, "test"),
        "paws-calibration": load_paws(args.data_cache, "calibration"),
        "paws-test": load_paws(args.data_cache, "test"),
    }
    embedder = SentenceTransformerEmbedder(settings.embedding_model)

    print("Embedding pairs...")
    sims = {name: pair_similarities(embedder, p.text_a, p.text_b) for name, p in data.items()}
    labels = {name: np.array(p.labels) for name, p in data.items()}

    print("Generating hard negatives...")
    test = data["qqp-test"]
    hard_negatives = generate_hard_negatives(test.text_a + test.text_b, per_kind=args.hard_negatives_per_kind, seed=args.seed)
    hn_sims = pair_similarities(embedder, [h.original for h in hard_negatives], [h.perturbed for h in hard_negatives])
    hn_kinds = np.array([h.kind for h in hard_negatives])

    report: dict = {
        "embedding_model": settings.embedding_model,
        "baseline_threshold": BASELINE_THRESHOLD,
        "seed": args.seed,
        "python": platform.python_version(),
        "datasets": {name: {"pairs": len(p), "same_meaning_share": float(np.mean(p.labels))} for name, p in data.items()},
        "pairs": {},
        "hard_negatives": {},
        "calibrated": {},
        "replay": [],
    }

    # 1. Pair-level: how well does similarity separate same vs different meaning?
    for name in ("qqp-test", "paws-test"):
        report["pairs"][name] = {
            "auc": auc(sims[name], labels[name]),
            "at_baseline": evaluate_threshold(sims[name], labels[name], BASELINE_THRESHOLD).as_dict(),
            "sweep": [p.as_dict() for p in sweep(sims[name], labels[name])],
        }

    # 2. Hard negatives: share wrongly reused at each threshold, per kind.
    for kind in sorted(set(hn_kinds)):
        s = hn_sims[hn_kinds == kind]
        report["hard_negatives"][kind] = {
            "pairs": int(len(s)),
            "median_similarity": float(np.median(s)),
            "wrongly_reused_at": {f"{t:.2f}": float(np.mean(s >= t)) for t in REPLAY_THRESHOLDS},
        }

    # 3. Calibrate a single global threshold on calibration data, judge on test.
    for budget in ERROR_BUDGETS:
        entry = {}
        for family in ("qqp", "paws"):
            t = calibrate_threshold(sims[f"{family}-calibration"], labels[f"{family}-calibration"], budget)
            entry[family] = {
                "threshold": t,
                "test": None if t is None else evaluate_threshold(sims[f"{family}-test"], labels[f"{family}-test"], t).as_dict(),
            }
        report["calibrated"][f"{budget:.2f}"] = entry

    # 4. Replay a QQP traffic stream (with hard negatives mixed in) through the real cache.
    print("Replaying workload...")
    workload = build_workload(test.text_a, test.text_b, test.labels, hard_negatives, args.max_queries, args.seed)
    stream_embeddings = embedder.encode(workload.queries)
    thresholds = sorted(
        set(REPLAY_THRESHOLDS)
        | {round(e["qqp"]["threshold"], 4) for e in report["calibrated"].values() if e["qqp"]["threshold"] is not None}
    )
    for t in thresholds:
        result = replay(workload, stream_embeddings, t)
        report["replay"].append(result.as_dict())
        print(f"  t={t:.3f}  hit {pct(result.hit_rate)}  false-hit {pct(result.false_hit_rate)}")

    report["runtime_seconds"] = round(time.perf_counter() - t0, 1)
    (out / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out / "README.md").write_text(render_markdown(report, len(workload)), encoding="utf-8")
    plot(report, out / "tradeoff.png")
    print(f"Wrote {out}/results.json, README.md, tradeoff.png in {report['runtime_seconds']}s")


def render_markdown(r: dict, stream_len: int) -> str:
    lines = [
        "# Phase 1 baseline: fixed-threshold semantic cache",
        "",
        f"Embedding model `{r['embedding_model']}`; baseline threshold **{r['baseline_threshold']}** (the original project's value).",
        "",
        "**Hit rate** = share of same-meaning questions answered from cache. "
        "**False-hit rate** = share of cache answers that were for a different question.",
        "",
        "## Datasets",
        "",
        "| Dataset | Pairs | Same-meaning share |",
        "|---|---|---|",
    ]
    for name, d in r["datasets"].items():
        lines.append(f"| {name} | {d['pairs']:,} | {pct(d['same_meaning_share'])} |")

    lines += ["", "## Pair level at the baseline threshold", "", "| Dataset | AUC | Hit rate | False-hit rate |", "|---|---|---|---|"]
    for name, p in r["pairs"].items():
        b = p["at_baseline"]
        lines.append(f"| {name} | {p['auc']:.3f} | {pct(b['hit_rate'])} | {pct(b['false_hit_rate'])} |")

    lines += ["", "## Hard negatives: share wrongly treated as the same question", "",
              "| Edit | Pairs | Median similarity | " + " | ".join(f"t={t}" for t in REPLAY_THRESHOLDS) + " |",
              "|---|---|---|" + "---|" * len(REPLAY_THRESHOLDS)]
    for kind, h in r["hard_negatives"].items():
        cells = " | ".join(pct(v) for v in h["wrongly_reused_at"].values())
        lines.append(f"| {kind} | {h['pairs']} | {h['median_similarity']:.3f} | {cells} |")

    lines += ["", "## Best single threshold for an error budget", "",
              "Chosen on the calibration split (lowest threshold meeting the budget), measured on the test split.", "",
              "| Budget (max false-hit rate) | QQP threshold | QQP test hit / false-hit | PAWS threshold | PAWS test hit / false-hit |",
              "|---|---|---|---|---|"]
    for budget, e in r["calibrated"].items():
        cells = []
        for fam in ("qqp", "paws"):
            t, test = e[fam]["threshold"], e[fam]["test"]
            cells += ["not reachable", "-"] if t is None else [f"{t:.3f}", f"{pct(test['hit_rate'])} / {pct(test['false_hit_rate'])}"]
        lines.append(f"| {pct(float(budget))} | " + " | ".join(cells) + " |")

    lines += ["", f"## Workload replay ({stream_len:,} queries, QQP test + hard negatives)", "",
              "Every hit on a hard negative is wrong (it has its own meaning group), so the false-hit rate is also shown "
              "for natural QQP queries only.", "",
              "| Threshold | Hit rate | False-hit rate | False-hit rate, natural only | Wrong answers / all queries | Hard negatives served wrong | Best possible hit rate | Lookup p50 / p99 |",
              "|---|---|---|---|---|---|---|---|"]
    for x in r["replay"]:
        natural_hits = x["hits"] - x["hard_negative_wrong_hits"]
        natural_wrong = x["wrong_hits"] - x["hard_negative_wrong_hits"]
        natural_fhr = natural_wrong / natural_hits if natural_hits else 0.0
        lines.append(
            f"| {x['threshold']:.3f} | {pct(x['hit_rate'])} | {pct(x['false_hit_rate'])} | {pct(natural_fhr)} | {pct(x['wrong_answer_rate'])} | "
            f"{x['hard_negative_wrong_hits']} / {x['hard_negative_queries']} | {pct(x['best_possible_hit_rate'])} | "
            f"{x['lookup_p50_ms']:.2f} / {x['lookup_p99_ms']:.2f} ms |"
        )
    lines += ["", "![Trade-off](tradeoff.png)", "",
              "Caveat: QQP 'not duplicate' labels are incomplete, so some counted wrong hits may be true paraphrases. "
              "False-hit rates here are an upper bound.", ""]
    return "\n".join(lines)


def plot(r: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, color in (("qqp-test", "#2a6fdb"), ("paws-test", "#d9480f")):
        pts = r["pairs"][name]["sweep"]
        ax.plot([100 * p["false_hit_rate"] for p in pts], [100 * p["hit_rate"] for p in pts], color=color, label=name)
        b = r["pairs"][name]["at_baseline"]
        ax.scatter([100 * b["false_hit_rate"]], [100 * b["hit_rate"]], color=color, zorder=3)
        ax.annotate(f"t={r['baseline_threshold']}", (100 * b["false_hit_rate"], 100 * b["hit_rate"]),
                    textcoords="offset points", xytext=(6, -12), fontsize=8, color=color)
    ax.set_xlabel("False-hit rate (% of cache answers that are wrong)")
    ax.set_ylabel("Hit rate (% of same-meaning questions reused)")
    ax.set_title("Fixed-threshold cache: reuse vs wrong answers")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
