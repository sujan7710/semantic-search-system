"""Workload replay: push a stream of questions through a real cache.

Pair-level numbers say how a threshold separates two questions. Replay says
what happens in traffic: the cache fills up, a new question is compared with
*everything* cached so far, and one wrong entry can be served many times.

Stream construction (deterministic for a seed):
* Questions come from labeled pairs; questions linked by "duplicate" labels
  form a meaning group. A hit is correct when the matched cached question is
  in the same group as the incoming one.
* Hard negatives (edited questions with a different meaning) are added as
  their own groups, placed after their original, so a cache that confuses
  them serves a wrong answer.
* An "opportunity" is a query whose group already appeared earlier in the
  stream: a perfect cache would answer it. Opportunities / queries is the
  best achievable hit rate.
"""

from __future__ import annotations

import random
import time
from dataclasses import asdict, dataclass

import numpy as np

from cacheguard.bench.datasets import question_groups
from cacheguard.bench.hard_negatives import HardNegative
from cacheguard.cache import SemanticCache


@dataclass(frozen=True)
class Workload:
    queries: list[str]
    groups: list[int]
    kinds: list[str]  # "natural" or the hard-negative kind

    def __len__(self) -> int:
        return len(self.queries)


def build_workload(
    text_a: list[str],
    text_b: list[str],
    labels: list[int],
    hard_negatives: list[HardNegative] = (),
    max_queries: int | None = None,
    seed: int = 0,
) -> Workload:
    rng = random.Random(seed)
    groups = question_groups(text_a, text_b, labels)
    questions = list(groups)
    rng.shuffle(questions)
    if max_queries is not None:
        questions = questions[:max_queries]

    present = set(questions)
    negatives_by_original: dict[str, list[HardNegative]] = {}
    for hn in hard_negatives:
        if hn.original in present and hn.perturbed not in groups:
            negatives_by_original.setdefault(hn.original, []).append(hn)

    next_group = max(groups.values(), default=-1) + 1
    queries, query_groups, kinds = [], [], []
    for q in questions:
        queries.append(q)
        query_groups.append(groups[q])
        kinds.append("natural")
    # Insert each hard negative at a random point after its original.
    for hn in (h for hs in negatives_by_original.values() for h in hs):
        after = queries.index(hn.original)
        pos = rng.randint(after + 1, len(queries))
        queries.insert(pos, hn.perturbed)
        query_groups.insert(pos, next_group)
        kinds.insert(pos, hn.kind)
        next_group += 1
    return Workload(queries, query_groups, kinds)


@dataclass(frozen=True)
class ReplayResult:
    threshold: float
    queries: int
    hits: int
    correct_hits: int
    wrong_hits: int
    opportunities: int
    hit_rate: float  # hits / queries = backend calls avoided
    false_hit_rate: float  # wrong_hits / hits
    wrong_answer_rate: float  # wrong_hits / queries
    best_possible_hit_rate: float  # opportunities / queries
    hard_negative_queries: int
    hard_negative_wrong_hits: int
    lookup_p50_ms: float
    lookup_p99_ms: float
    final_cache_entries: int

    def as_dict(self) -> dict:
        return asdict(self)


def replay(workload: Workload, embeddings: np.ndarray, threshold: float) -> ReplayResult:
    """Run the baseline cache over the stream. ``embeddings[i]`` belongs to query i."""
    cache = SemanticCache(threshold=threshold)
    seen_groups: set[int] = set()
    hits = correct = opportunities = hn_queries = hn_wrong = 0
    lookup_ms = np.empty(len(workload))

    for i, (query, group, kind) in enumerate(zip(workload.queries, workload.groups, workload.kinds)):
        if group in seen_groups:
            opportunities += 1
        if kind != "natural":
            hn_queries += 1

        start = time.perf_counter()
        lookup = cache.lookup(embeddings[i])
        lookup_ms[i] = (time.perf_counter() - start) * 1000

        if lookup.hit:
            hits += 1
            if lookup.entry.response == group:
                correct += 1
            elif kind != "natural":
                hn_wrong += 1
        else:
            # The cached "answer" is just the group id: enough to judge correctness.
            cache.add(query, embeddings[i], response=group)
        seen_groups.add(group)

    n = len(workload)
    wrong = hits - correct
    return ReplayResult(
        threshold=threshold,
        queries=n,
        hits=hits,
        correct_hits=correct,
        wrong_hits=wrong,
        opportunities=opportunities,
        hit_rate=hits / n,
        false_hit_rate=wrong / hits if hits else 0.0,
        wrong_answer_rate=wrong / n,
        best_possible_hit_rate=opportunities / n,
        hard_negative_queries=hn_queries,
        hard_negative_wrong_hits=hn_wrong,
        lookup_p50_ms=float(np.percentile(lookup_ms, 50)),
        lookup_p99_ms=float(np.percentile(lookup_ms, 99)),
        final_cache_entries=len(cache),
    )
