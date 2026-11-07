"""The guard: decides whether a candidate cached answer may be reused.

Given the incoming question and its nearest cached question:

    similarity < gate                 -> miss   (not even close)
    rules find a mismatch             -> reject (negation / number / name differs)
    similarity >= skip_verify_above   -> hit    (so close the verifier is skipped)
    verifier score >= min_score       -> hit    (borderline, verified)
    otherwise                         -> reject (verifier disagrees)

The thresholds come from calibration (``cacheguard.bench.calibrate``) for a
target false-hit rate. They can differ per query cluster: some topics need a
stricter gate than others.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from cacheguard.embeddings import normalize_rows
from cacheguard.verify import Verifier, rule_mismatch


@dataclass(frozen=True)
class GuardSettings:
    gate: float
    skip_verify_above: float = 1.01  # > 1 means always verify
    min_verifier_score: float = 0.5
    use_rules: bool = True


@dataclass(frozen=True)
class Decision:
    hit: bool
    stage: str  # gate | rules | similarity | verifier
    reason: str  # below_gate | negation | number | name | high_similarity | verified | verifier_rejected | no_verifier
    similarity: float
    gate: float
    verifier_score: float | None = None
    cluster: int | None = None


class QueryClusterer:
    """Nearest-centroid assignment (centroids from k-means on question embeddings)."""

    def __init__(self, centroids: np.ndarray):
        self.centroids = normalize_rows(np.asarray(centroids, dtype=np.float32))

    def assign(self, embedding: np.ndarray) -> int:
        return int(np.argmax(self.centroids @ normalize_rows(embedding.reshape(1, -1))[0]))


class Guard:
    def __init__(
        self,
        default: GuardSettings,
        verifier: Verifier | None = None,
        per_cluster: dict[int, GuardSettings] | None = None,
        clusterer: QueryClusterer | None = None,
        metadata: dict | None = None,
    ):
        if per_cluster and clusterer is None:
            raise ValueError("per-cluster settings need a clusterer")
        self.default = default
        self.verifier = verifier
        self.per_cluster = per_cluster or {}
        self.clusterer = clusterer
        self.metadata = metadata or {}

    def settings_for(self, embedding: np.ndarray) -> tuple[int | None, GuardSettings]:
        if self.clusterer is None:
            return None, self.default
        cluster = self.clusterer.assign(embedding)
        return cluster, self.per_cluster.get(cluster, self.default)

    def decide(self, query: str, embedding: np.ndarray, candidate: str, similarity: float) -> Decision:
        cluster, s = self.settings_for(embedding)

        def decision(hit: bool, stage: str, reason: str, score: float | None = None) -> Decision:
            return Decision(hit, stage, reason, similarity, s.gate, score, cluster)

        if similarity < s.gate:
            return decision(False, "gate", "below_gate")
        if s.use_rules:
            rule = rule_mismatch(query, candidate)
            if rule is not None:
                return decision(False, "rules", rule)
        if similarity >= s.skip_verify_above:
            return decision(True, "similarity", "high_similarity")
        if self.verifier is None:
            # Without a verifier, borderline matches are not trusted.
            return decision(False, "verifier", "no_verifier")
        score = float(self.verifier.score([(query, candidate)])[0])
        if score >= s.min_verifier_score:
            return decision(True, "verifier", "verified", score)
        return decision(False, "verifier", "verifier_rejected", score)

    def describe(self) -> dict:
        return {
            "type": "guard",
            "default": asdict(self.default),
            "per_cluster": {str(k): asdict(v) for k, v in sorted(self.per_cluster.items())},
            "verifier": getattr(self.verifier, "name", None),
            **self.metadata,
        }

    # -- persistence: JSON settings + .npz centroids (no pickle) --------------

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        config = {
            "default": asdict(self.default),
            "per_cluster": {str(k): asdict(v) for k, v in self.per_cluster.items()},
            "metadata": self.metadata,
        }
        (directory / "guard.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        if self.clusterer is not None:
            np.save(directory / "centroids.npy", self.clusterer.centroids)

    @classmethod
    def load(cls, directory: str | Path, verifier: Verifier | None = None) -> Guard:
        directory = Path(directory)
        config = json.loads((directory / "guard.json").read_text(encoding="utf-8"))
        centroids = directory / "centroids.npy"
        clusterer = QueryClusterer(np.load(centroids, allow_pickle=False)) if centroids.exists() else None
        return cls(
            default=GuardSettings(**config["default"]),
            verifier=verifier,
            per_cluster={int(k): GuardSettings(**v) for k, v in config["per_cluster"].items()},
            clusterer=clusterer,
            metadata=config.get("metadata", {}),
        )
