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
from dataclasses import asdict, dataclass, field

import numpy as np

from cacheguard.bench.datasets import question_groups
from cacheguard.bench.hard_negatives import HardNegative
from cacheguard.cache import SemanticCache


CORRECT, WRONG, UNJUDGED = 1, 0, -1


@dataclass(frozen=True)
class Workload:
    queries: list[str]
    groups: list[int]
    kinds: list[str]  # "natural" or the hard-negative kind
    # For a hard negative, the meaning group of the question it was edited from.
    origins: list[int | None] = field(default_factory=list)
    # Pairs annotators explicitly labeled "not duplicate" (both orders).
    not_duplicate: frozenset[tuple[str, str]] = frozenset()

    def __len__(self) -> int:
        return len(self.queries)

    def judge(self, i: int, j: int) -> int:
        """Is answering query i with cached query j's answer correct?

        Only pairs the labels actually cover are judged:
        * same duplicate group                       -> CORRECT
        * a hard negative vs its original's group    -> WRONG
        * explicitly labeled "not duplicate"         -> WRONG
        * anything else                              -> UNJUDGED

        QQP annotators compared specific pairs only. Two questions from
        different pairs that were never compared are often real paraphrases,
        so counting them as wrong (as the Phase 1 replay did) overstates errors.
        """
        if self.groups[i] == self.groups[j]:
            return CORRECT
        origins = self.origins or [None] * len(self.queries)
        if origins[i] is not None and origins[i] == self.groups[j]:
            return WRONG
        if origins[j] is not None and origins[j] == self.groups[i]:
            return WRONG
        if (self.queries[i], self.queries[j]) in self.not_duplicate:
            return WRONG
        return UNJUDGED


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
    queries, query_groups, kinds, origins = [], [], [], []
    for q in questions:
        queries.append(q)
        query_groups.append(groups[q])
        kinds.append("natural")
        origins.append(None)
    # Insert each hard negative at a random point after its original.
    for hn in (h for hs in negatives_by_original.values() for h in hs):
        after = queries.index(hn.original)
        pos = rng.randint(after + 1, len(queries))
        queries.insert(pos, hn.perturbed)
        query_groups.insert(pos, next_group)
        kinds.insert(pos, hn.kind)
        origins.insert(pos, groups[hn.original])
        next_group += 1
    not_duplicate = frozenset(
        pair for a, b, label in zip(text_a, text_b, labels) if label == 0 for pair in ((a, b), (b, a))
    )
    return Workload(queries, query_groups, kinds, origins, not_duplicate)


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


def nearest_previous(embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For each query i, the most similar earlier query j < i (j = -1 for the first).

    This is the candidate a cache holding every earlier question would offer.
    The real cache holds only the questions that missed, so its candidate can
    differ; this is used to calibrate and to pre-compute verifier scores.
    """
    import faiss

    index = faiss.IndexFlatIP(embeddings.shape[1])
    sims = np.full(len(embeddings), -1.0, dtype=np.float32)
    idx = np.full(len(embeddings), -1, dtype=np.int64)
    for i, v in enumerate(np.ascontiguousarray(embeddings, dtype=np.float32)):
        if i:
            s, j = index.search(v.reshape(1, -1), 1)
            sims[i], idx[i] = s[0, 0], j[0, 0]
        index.add(v.reshape(1, -1))
    return sims, idx


@dataclass(frozen=True)
class PolicyReplayResult:
    name: str
    queries: int
    hits: int
    judged_hits: int  # hits whose correctness the labels determine
    correct_hits: int
    wrong_hits: int
    hit_rate: float
    judged_share: float  # judged_hits / hits
    false_hit_rate: float  # wrong / judged hits
    natural_false_hit_rate: float  # same, natural queries only
    hard_negative_queries: int
    hard_negative_wrong_hits: int
    verifier_calls: int
    verifier_call_rate: float
    reasons: dict

    def as_dict(self) -> dict:
        return asdict(self)


def replay_policy(name: str, workload: Workload, embeddings: np.ndarray, decide) -> PolicyReplayResult:
    """Replay with an arbitrary decision function.

    ``decide(i, candidate_index, similarity)`` gets the incoming query index,
    the stream index of the nearest *cached* query, and their similarity, and
    returns ``(hit, reason, used_verifier)``. Queries that are not hits are
    added to the cache, as in the real service.
    """
    import faiss

    index = faiss.IndexFlatIP(embeddings.shape[1])
    cached_stream_ids: list[int] = []
    hits = judged = correct = wrong = hn_queries = hn_wrong = verifier_calls = 0
    natural_judged = natural_wrong = 0
    reasons: dict[str, int] = {}
    vectors = np.ascontiguousarray(embeddings, dtype=np.float32)

    for i, kind in enumerate(workload.kinds):
        if kind != "natural":
            hn_queries += 1
        hit = False
        if cached_stream_ids:
            s, j = index.search(vectors[i].reshape(1, -1), 1)
            candidate = cached_stream_ids[int(j[0, 0])]
            hit, reason, used_verifier = decide(i, candidate, float(s[0, 0]))
            reasons[reason] = reasons.get(reason, 0) + 1
            verifier_calls += int(used_verifier)
            if hit:
                hits += 1
                verdict = workload.judge(i, candidate)
                if verdict != UNJUDGED:
                    judged += 1
                    natural = kind == "natural" and workload.kinds[candidate] == "natural"
                    natural_judged += natural
                    if verdict == CORRECT:
                        correct += 1
                    else:
                        wrong += 1
                        natural_wrong += natural
                        hn_wrong += kind != "natural"
        if not hit:
            index.add(vectors[i].reshape(1, -1))
            cached_stream_ids.append(i)

    n = len(workload)
    return PolicyReplayResult(
        name=name,
        queries=n,
        hits=hits,
        judged_hits=judged,
        correct_hits=correct,
        wrong_hits=wrong,
        hit_rate=hits / n,
        judged_share=judged / hits if hits else 0.0,
        false_hit_rate=wrong / judged if judged else 0.0,
        natural_false_hit_rate=natural_wrong / natural_judged if natural_judged else 0.0,
        hard_negative_queries=hn_queries,
        hard_negative_wrong_hits=hn_wrong,
        verifier_calls=verifier_calls,
        verifier_call_rate=verifier_calls / n,
        reasons=reasons,
    )


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
