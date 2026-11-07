import numpy as np
import pytest
from fastapi.testclient import TestClient

from cacheguard.api import create_app
from cacheguard.guard import Guard, GuardSettings, QueryClusterer
from cacheguard.verify import names, negation_parity, numbers, rule_mismatch


class FakeVerifier:
    name = "fake-verifier"

    def __init__(self, score: float = 0.9):
        self.value = score
        self.calls = 0

    def score(self, pairs):
        self.calls += len(pairs)
        return np.full(len(pairs), self.value)


EMB = np.array([1.0, 0.0], dtype=np.float32)


# --- rules ------------------------------------------------------------------------

def test_rule_features():
    assert negation_parity("How do I not cancel?") == 1
    assert negation_parity("Why can't I sleep?") == 1
    assert negation_parity("How do I cancel?") == 0
    assert numbers("Is 5 hours or five hours enough? 1,000 people") == {"5": 2, "1000": 1}
    assert names("What is it like to work at Google in India? I wonder.") == {"google", "india"}


@pytest.mark.parametrize(
    "a, b, expected",
    [
        ("How do I cancel my order?", "How do I not cancel my order?", "negation"),
        ("Is 5 hours of sleep enough?", "Is 7 hours of sleep enough?", "number"),
        ("How do I get a job in India?", "How do I get a job in Canada?", "name"),
        ("Best laptop?", "Best laptop under 40000?", None),  # an addition, not a swap
        ("What is GST?", "What is the GST bill?", None),
        ("How do I learn Python?", "What is the best way to learn python?", None),  # case-insensitive
    ],
)
def test_rule_mismatch(a, b, expected):
    assert rule_mismatch(a, b) == expected


# --- guard decisions ------------------------------------------------------------

def test_guard_stages():
    verifier = FakeVerifier(0.9)
    guard = Guard(GuardSettings(gate=0.7, skip_verify_above=0.97, min_verifier_score=0.5), verifier)
    assert guard.decide("a", EMB, "b", 0.6).reason == "below_gate"
    assert guard.decide("How do I cancel?", EMB, "How do I not cancel?", 0.99).reason == "negation"
    d = guard.decide("a b", EMB, "a c", 0.98)
    assert d.hit and d.reason == "high_similarity" and verifier.calls == 0
    d = guard.decide("a b", EMB, "a c", 0.8)
    assert d.hit and d.reason == "verified" and d.verifier_score == 0.9 and verifier.calls == 1
    verifier.value = 0.1
    d = guard.decide("a b", EMB, "a d", 0.8)
    assert not d.hit and d.reason == "verifier_rejected"


def test_guard_without_verifier_rejects_borderline():
    guard = Guard(GuardSettings(gate=0.7, skip_verify_above=0.95))
    assert guard.decide("a", EMB, "b", 0.96).hit
    assert guard.decide("a", EMB, "b", 0.9).reason == "no_verifier"


def test_rules_can_be_disabled():
    guard = Guard(GuardSettings(gate=0.7, skip_verify_above=0.9, use_rules=False))
    assert guard.decide("How do I cancel?", EMB, "How do I not cancel?", 0.99).hit


def test_per_cluster_settings_and_round_trip(tmp_path):
    clusterer = QueryClusterer(np.array([[1, 0], [0, 1]], dtype=np.float32))
    guard = Guard(
        GuardSettings(gate=0.7),
        FakeVerifier(),
        per_cluster={1: GuardSettings(gate=0.95)},
        clusterer=clusterer,
        metadata={"calibrated_for_false_hit_rate": 0.1},
    )
    assert guard.decide("x", np.array([1, 0.1], dtype=np.float32), "y", 0.8).hit  # cluster 0, gate 0.7
    d = guard.decide("x", np.array([0.1, 1], dtype=np.float32), "y", 0.8)  # cluster 1, gate 0.95
    assert not d.hit and d.cluster == 1 and d.gate == 0.95

    guard.save(tmp_path)
    loaded = Guard.load(tmp_path, FakeVerifier())
    assert loaded.per_cluster == guard.per_cluster and loaded.default == guard.default
    np.testing.assert_allclose(loaded.clusterer.centroids, clusterer.centroids)
    assert loaded.describe()["calibrated_for_false_hit_rate"] == 0.1


def test_per_cluster_requires_clusterer():
    with pytest.raises(ValueError):
        Guard(GuardSettings(gate=0.7), per_cluster={0: GuardSettings(gate=0.8)})


# --- end to end through the API --------------------------------------------------

def test_plain_cache_reuses_negated_question_but_guard_does_not(service):
    question = {"query": "how do I cancel my order"}
    negated = {"query": "how do I not cancel my order"}

    plain = TestClient(create_app(service))
    plain.post("/query", json=question)
    body = plain.post("/query", json=negated).json()
    assert body["cache_hit"] is True  # the baseline failure mode
    assert body["decision"]["stage"] == "threshold"

    service.store.clear()
    service.guard = Guard(GuardSettings(gate=0.6, skip_verify_above=0.99), FakeVerifier(0.9))
    guarded = TestClient(create_app(service))
    guarded.post("/query", json=question)
    body = guarded.post("/query", json=negated).json()
    assert body["cache_hit"] is False
    assert body["decision"] == {"stage": "rules", "reason": "negation", "verifier_score": None,
                                "decision_ms": body["decision"]["decision_ms"], "guarded": True}

    paraphrase = guarded.post("/query", json={"query": "how do I cancel my order please"}).json()
    assert paraphrase["cache_hit"] is True
    assert paraphrase["decision"]["reason"] == "verified" and paraphrase["decision"]["verifier_score"] == 0.9

    stats = guarded.get("/cache/stats").json()
    assert stats["decisions"]["negation"] == 1 and stats["decisions"]["verified"] == 1
    assert guarded.get("/health").json()["guard"]["verifier"] == "fake-verifier"
