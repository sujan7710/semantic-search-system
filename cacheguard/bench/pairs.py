"""Pair-level evaluation: for a threshold, what does 'similarity >= t' get right?

Vocabulary used throughout the benchmark:
* hit rate        share of same-meaning pairs the cache would reuse (recall)
* false-hit rate  share of cache hits that are wrong, FP / (TP + FP)
                  (what fraction of reused answers would be incorrect)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import roc_auc_score

from cacheguard.embeddings import Embedder

DEFAULT_THRESHOLDS = np.round(np.arange(0.50, 1.0, 0.01), 2)


def pair_similarities(embedder: Embedder, text_a: list[str], text_b: list[str]) -> np.ndarray:
    """Cosine similarity of each (a, b) pair, embedding each unique text once."""
    unique = list(dict.fromkeys(text_a + text_b))
    position = {t: i for i, t in enumerate(unique)}
    vectors = embedder.encode(unique)
    a = vectors[[position[t] for t in text_a]]
    b = vectors[[position[t] for t in text_b]]
    return np.einsum("ij,ij->i", a, b)


@dataclass(frozen=True)
class ThresholdPoint:
    threshold: float
    hit_rate: float
    false_hit_rate: float
    false_positive_rate: float  # share of different-meaning pairs wrongly reused
    hits: int

    def as_dict(self) -> dict:
        return asdict(self)


def evaluate_threshold(similarities: np.ndarray, labels: np.ndarray, threshold: float) -> ThresholdPoint:
    labels = np.asarray(labels).astype(bool)
    predicted = similarities >= threshold
    tp = int(np.sum(predicted & labels))
    fp = int(np.sum(predicted & ~labels))
    positives = int(labels.sum())
    negatives = int((~labels).sum())
    return ThresholdPoint(
        threshold=float(threshold),
        hit_rate=tp / positives if positives else 0.0,
        false_hit_rate=fp / (tp + fp) if tp + fp else 0.0,
        false_positive_rate=fp / negatives if negatives else 0.0,
        hits=tp + fp,
    )


def sweep(similarities: np.ndarray, labels: np.ndarray, thresholds=DEFAULT_THRESHOLDS) -> list[ThresholdPoint]:
    return [evaluate_threshold(similarities, labels, t) for t in thresholds]


def auc(similarities: np.ndarray, labels: np.ndarray) -> float:
    """Probability a random same-meaning pair scores above a random different one."""
    return float(roc_auc_score(labels, similarities))


def calibrate_threshold(
    similarities: np.ndarray, labels: np.ndarray, max_false_hit_rate: float, min_hits: int = 50
) -> float | None:
    """Lowest threshold whose false-hit rate is within budget on calibration data.

    Lower threshold means more hits, so the lowest one that meets the error
    budget maximizes reuse. ``min_hits`` avoids picking a threshold that meets
    the budget only because it almost never fires. Returns None if no
    threshold meets the budget.
    """
    candidates = np.unique(np.round(similarities, 4))
    labels = np.asarray(labels).astype(bool)
    order = np.argsort(-similarities)
    sorted_sim = similarities[order]
    fp_cum = np.cumsum(~labels[order])
    best = None
    for t in candidates[::-1]:  # high -> low
        n = int(np.searchsorted(-sorted_sim, -t, side="right"))
        if n < min_hits:
            continue
        fhr = fp_cum[n - 1] / n
        if fhr <= max_false_hit_rate:
            best = float(t)
    return best
