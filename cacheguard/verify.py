"""Verifiers: second opinions on a candidate cache match.

Two tiers, cheapest first:

1. ``rule_mismatch`` (microseconds). Flags pairs whose surface details that
   decide the answer differ: negation, numbers, capitalized names. A mismatch
   rejects the match; a pass proves nothing (rules only catch what they look for).

   Evaluation caveat: the Phase 1 hard-negative generator edits exactly these
   things, so rules scoring well on hard negatives is partly circular. The
   rules are written generically (any number, any capitalized name) rather
   than from the generator's word lists, and the benchmark reports the
   cross-encoder alone as the non-circular measure.

2. ``CrossEncoderVerifier`` (~40 ms per pair on CPU). Reads both questions
   together, so it sees word order and interactions a single embedding loses.
   Used only for borderline similarities.
"""

from __future__ import annotations

import re
import threading
from collections import Counter, OrderedDict
from typing import Protocol, Sequence

import numpy as np

_NEGATION = re.compile(r"\b(not|no|never|none|nobody|nothing|neither|nor|without|cannot)\b|n't\b", re.IGNORECASE)
_NUMBER = re.compile(r"(?<![\w.])\d+(?:[.,]\d+)*(?![\w])")
_NUMBER_WORDS = {
    w: str(i)
    for i, w in enumerate(
        "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
        "fifteen sixteen seventeen eighteen nineteen twenty".split()
    )
}
_NUMBER_WORD = re.compile(r"\b(" + "|".join(_NUMBER_WORDS) + r")\b", re.IGNORECASE)
_TOKEN = re.compile(r"[A-Za-z][\w+#'-]*")
_SENTENCE_START = re.compile(r"(^|[.?!]\s+)([A-Za-z][\w+#'-]*)")


def negation_parity(text: str) -> int:
    return len(_NEGATION.findall(text)) % 2


def numbers(text: str) -> Counter:
    found = [n.replace(",", "") for n in _NUMBER.findall(text)]
    found += [_NUMBER_WORDS[w.lower()] for w in _NUMBER_WORD.findall(text)]
    return Counter(found)


def names(text: str) -> set[str]:
    """Capitalized words that are not sentence-initial (and not 'I'), lowercased."""
    starts = {m.start(2) for m in _SENTENCE_START.finditer(text)}
    out = set()
    for m in _TOKEN.finditer(text):
        word = m.group()
        if m.start() in starts or word == "I" or not word[0].isupper():
            continue
        out.add(word.lower())
    return out


def _swapped(a_items: set[str], b_items: set[str], a_text: str, b_text: str) -> bool:
    """True when each side has an item the other lacks (a substitution).

    Mere additions ("best laptop" vs "best laptop under 40k") are common in
    paraphrases, so they do not count. An item also counts as present if it
    appears anywhere in the other text, case-insensitively ("GST" vs "gst").
    """
    a_low, b_low = a_text.lower(), b_text.lower()
    only_a = {x for x in a_items - b_items if not re.search(rf"(?<!\w){re.escape(x)}(?!\w)", b_low)}
    only_b = {x for x in b_items - a_items if not re.search(rf"(?<!\w){re.escape(x)}(?!\w)", a_low)}
    return bool(only_a) and bool(only_b)


def rule_mismatch(a: str, b: str) -> str | None:
    """Name of the first rule the pair fails, or None if no rule objects."""
    if negation_parity(a) != negation_parity(b):
        return "negation"
    if _swapped(set(numbers(a)), set(numbers(b)), a, b):
        return "number"
    if _swapped(names(a), names(b), a, b):
        return "name"
    return None


class Verifier(Protocol):
    name: str

    def score(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray:
        """Probability in [0, 1] that each pair asks the same thing."""
        ...


class CrossEncoderVerifier:
    """Cross-encoder with a small memo: repeated (query, candidate) pairs are free."""

    def __init__(self, model_name: str = "cross-encoder/quora-distilroberta-base", memo_size: int = 50_000):
        from sentence_transformers import CrossEncoder  # heavy import, load lazily

        self.name = model_name
        self._model = CrossEncoder(model_name)
        self._memo: OrderedDict[tuple[str, str], float] = OrderedDict()
        self._memo_size = memo_size
        self._lock = threading.Lock()

    def dump_memo(self) -> list[tuple[str, str, float]]:
        with self._lock:
            return [(a, b, v) for (a, b), v in self._memo.items()]

    def load_memo(self, rows) -> None:
        with self._lock:
            for a, b, v in rows:
                self._memo[(a, b)] = float(v)

    def score(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray:
        pairs = [(a, b) for a, b in pairs]
        with self._lock:
            missing = list(dict.fromkeys(p for p in pairs if p not in self._memo))
        if missing:
            scores = self._model.predict(missing, batch_size=32, show_progress_bar=False)
            with self._lock:
                for pair, s in zip(missing, np.asarray(scores, dtype=float)):
                    self._memo[pair] = float(s)
                    if len(self._memo) > self._memo_size:
                        self._memo.popitem(last=False)
        with self._lock:
            return np.array([self._memo[p] for p in pairs], dtype=float)
