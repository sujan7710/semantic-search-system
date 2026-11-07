import random

import numpy as np
import pytest

from cacheguard.bench.datasets import question_groups
from cacheguard.bench.hard_negatives import (
    HardNegative,
    change_number,
    generate_hard_negatives,
    negate,
    swap_antonym,
    swap_entity,
)
from cacheguard.bench.pairs import calibrate_threshold, evaluate_threshold
from cacheguard.bench.workload import Workload, build_workload, replay
from cacheguard.cache import SemanticCache


# --- hard negatives ------------------------------------------------------------

def test_negation_rules():
    assert negate("How do I cancel my order?") == "How do I not cancel my order?"
    assert negate("Is it safe to swim?") == "Isn't it safe to swim?"
    assert negate("Why is Python so popular?") == "Why isn't Python so popular?"
    assert negate("Hello there") is None


def test_number_entity_and_antonym_rules():
    rng = random.Random(0)
    changed = change_number("Is 5 hours of sleep enough?", rng)
    assert changed != "Is 5 hours of sleep enough?" and "hours of sleep enough" in changed
    swapped = swap_entity("How do I get a job in India?", rng)
    assert "India" not in swapped and swapped.startswith("How do I get a job in ")
    assert swap_antonym("What is the best phone?") == "What is the worst phone?"
    assert swap_antonym("Best phone to buy?") == "Worst phone to buy?"


def test_entity_swap_does_not_match_inside_words():
    assert swap_entity("Is a Macbook good?", random.Random(0)) is None


def test_generate_hard_negatives_caps_each_kind():
    questions = [f"What is the best way to learn {n} languages in India?" for n in range(10)]
    out = generate_hard_negatives(questions, per_kind=3, seed=1)
    kinds = [h.kind for h in out]
    assert {k: kinds.count(k) for k in set(kinds)} == {"negation": 3, "number": 3, "entity": 3, "antonym": 3}
    assert all(h.original != h.perturbed for h in out)


# --- groups and pair metrics -------------------------------------------------

def test_question_groups_are_transitive_over_duplicates_only():
    groups = question_groups(["a", "b", "c"], ["b", "c", "d"], [1, 1, 0])
    assert groups["a"] == groups["b"] == groups["c"]
    assert groups["d"] != groups["a"]


def test_evaluate_threshold_counts():
    sims = np.array([0.95, 0.90, 0.88, 0.60])
    labels = np.array([1, 0, 1, 1])
    p = evaluate_threshold(sims, labels, 0.85)
    assert p.hits == 3
    assert p.hit_rate == pytest.approx(2 / 3)
    assert p.false_hit_rate == pytest.approx(1 / 3)
    assert p.false_positive_rate == 1.0


def test_calibrate_threshold_picks_lowest_within_budget():
    sims = np.array([0.99, 0.98, 0.97, 0.96, 0.90, 0.80])
    labels = np.array([1, 1, 1, 1, 0, 1])
    assert calibrate_threshold(sims, labels, max_false_hit_rate=0.0, min_hits=1) == pytest.approx(0.96)
    assert calibrate_threshold(sims, labels, max_false_hit_rate=0.2, min_hits=1) == pytest.approx(0.80)
    assert calibrate_threshold(sims, labels, max_false_hit_rate=0.0, min_hits=10) is None


# --- workload replay -----------------------------------------------------------

def test_hard_negatives_are_placed_after_their_original():
    hn = [HardNegative("q1", "q1 not", "negation")]
    wl = build_workload(["q1", "q2"], ["q1 again", "q3"], [1, 0], hn, seed=3)
    assert wl.queries.index("q1 not") > wl.queries.index("q1")
    assert len(set(wl.groups)) == 4  # {q1, q1 again}, q2, q3, q1 not


def test_replay_scores_correct_and_wrong_hits():
    # q1 and q1b mean the same; "q1 not" means something else but embeds close to q1.
    wl = Workload(["q1", "q1b", "q1 not", "other"], [0, 0, 1, 2], ["natural", "natural", "negation", "natural"])
    emb = np.array([[1, 0], [0.99, 0.14], [0.97, 0.24], [0, 1]], dtype=np.float32)
    r = replay(wl, emb, threshold=0.95)
    assert (r.hits, r.correct_hits, r.wrong_hits) == (2, 1, 1)
    assert r.hard_negative_wrong_hits == 1 and r.hard_negative_queries == 1
    assert r.opportunities == 1 and r.best_possible_hit_rate == 0.25
    assert r.final_cache_entries == 2


def test_judge_only_counts_labeled_pairs():
    from cacheguard.bench.workload import CORRECT, UNJUDGED, WRONG

    hn = [HardNegative("how do I cancel", "how do I not cancel", "negation")]
    wl = build_workload(
        ["how do I cancel", "best phone", "best laptop"],
        ["cancel my order how", "top phone", "cheap laptop"],
        [1, 1, 0],
        hn,
        seed=0,
    )
    i = wl.queries.index
    assert wl.judge(i("how do I cancel"), i("cancel my order how")) == CORRECT
    assert wl.judge(i("how do I not cancel"), i("cancel my order how")) == WRONG  # look-alike vs original's group
    assert wl.judge(i("cancel my order how"), i("how do I not cancel")) == WRONG  # served from a cached look-alike
    assert wl.judge(i("best laptop"), i("cheap laptop")) == WRONG  # labeled not duplicate
    assert wl.judge(i("best phone"), i("best laptop")) == UNJUDGED  # never compared by annotators


def test_replay_policy_counts_judged_hits_only():
    from cacheguard.bench.workload import replay_policy

    wl = Workload(
        ["a", "a2", "b", "c"], [0, 0, 1, 2], ["natural"] * 4, [None] * 4,
        frozenset({("b", "a"), ("a", "b")}),
    )
    emb = np.eye(4, dtype=np.float32)
    always = lambda i, j, s: (True, "always", False)  # noqa: E731
    r = replay_policy("always", wl, emb, always)
    # a2->a correct, b->a wrong (labeled), c->a unjudged
    assert (r.hits, r.judged_hits, r.correct_hits, r.wrong_hits) == (3, 2, 1, 1)
    assert r.false_hit_rate == 0.5 and r.judged_share == pytest.approx(2 / 3)


def test_cache_buffer_grows_past_initial_capacity():
    cache = SemanticCache(threshold=0.999)
    vectors = np.eye(40, dtype=np.float32)
    for i, v in enumerate(vectors):
        cache.add(f"q{i}", v, i)
    for i, v in enumerate(vectors):
        assert cache.lookup(v).entry.response == i
