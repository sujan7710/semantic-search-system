"""Generate 'looks the same, means something different' question pairs.

Each rule makes one small edit that changes the answer a user needs:
negation, a changed number, a swapped entity, or an antonym. A cache that
returns the original question's answer for the edited one is wrong.
These are synthetic and rule-based, so they test specific failure modes
rather than representing natural traffic.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

_NEGATABLE = re.compile(
    r"\b(is|are|was|were|can|could|should|would|will|do|does|did)\b(\s+(I|you|we|they|he|she|it)\b)?",
    re.IGNORECASE,
)
_CONTRACTED = {
    "is": "isn't", "are": "aren't", "was": "wasn't", "were": "weren't", "can": "can't",
    "could": "couldn't", "should": "shouldn't", "would": "wouldn't", "will": "won't",
    "do": "don't", "does": "doesn't", "did": "didn't",
}

_NUMBER = re.compile(r"\b\d+\b")

_ENTITY_GROUPS = [
    ["India", "China", "USA", "Canada", "Germany", "Japan", "Australia", "Pakistan", "UK", "France", "Brazil", "Russia"],
    ["Python", "Java", "JavaScript", "C++", "PHP", "Ruby", "Swift"],
    ["Google", "Facebook", "Amazon", "Apple", "Microsoft", "Quora", "Uber", "Netflix"],
    ["iPhone", "Android", "Windows", "Mac", "Linux"],
    ["Hindi", "English", "Spanish", "French", "German", "Chinese", "Japanese"],
    ["Delhi", "Mumbai", "Bangalore", "London", "Paris", "Tokyo", "Chennai", "Hyderabad"],
    ["Trump", "Hillary", "Obama", "Modi"],
]

_ANTONYMS = {
    "best": "worst", "worst": "best", "increase": "decrease", "decrease": "increase",
    "before": "after", "after": "before", "buy": "sell", "sell": "buy",
    "cheap": "expensive", "expensive": "cheap", "start": "stop", "stop": "start",
    "more": "less", "less": "more", "positive": "negative", "negative": "positive",
    "good": "bad", "bad": "good", "easy": "hard", "hard": "easy", "lose": "gain", "gain": "lose",
    "advantages": "disadvantages", "disadvantages": "advantages", "pros": "cons", "cons": "pros",
    "rich": "poor", "poor": "rich", "male": "female", "female": "male", "fast": "slow", "slow": "fast",
}
_ANTONYM_RE = re.compile(r"\b(" + "|".join(_ANTONYMS) + r")\b", re.IGNORECASE)


@dataclass(frozen=True)
class HardNegative:
    original: str
    perturbed: str
    kind: str  # negation | number | entity | antonym


def _match_case(word: str, template: str) -> str:
    return word.capitalize() if template[:1].isupper() else word


def negate(text: str) -> str | None:
    """'How do I cancel' -> 'How do I not cancel'; 'Is it safe' -> 'Isn't it safe';
    'Python is fast' -> 'Python isn't fast'."""
    m = _NEGATABLE.search(text)
    if not m:
        return None
    aux, subject = m.group(1), m.group(2)
    if subject and m.start() > 0:  # "How do I cancel" -> "How do I not cancel"
        return text[: m.end()] + " not" + text[m.end():]
    replacement = _match_case(_CONTRACTED[aux.lower()], aux)
    return text[: m.start()] + replacement + text[m.start() + len(aux):]


def change_number(text: str, rng: random.Random) -> str | None:
    m = _NUMBER.search(text)
    if not m:
        return None
    value = int(m.group())
    new = value + rng.choice([1, 2, 5, 10]) if value < 1000 else value + rng.choice([1, 10, 100])
    return text[: m.start()] + str(new) + text[m.end():]


def swap_entity(text: str, rng: random.Random) -> str | None:
    for group in _ENTITY_GROUPS:
        for name in group:
            m = re.search(rf"(?<!\w){re.escape(name)}(?!\w)", text)
            if m:
                other = rng.choice([g for g in group if g != name and g.lower() not in text.lower()] or [None])
                if other is None:
                    return None
                return text[: m.start()] + other + text[m.end():]
    return None


def swap_antonym(text: str) -> str | None:
    m = _ANTONYM_RE.search(text)
    if not m:
        return None
    word = m.group(1)
    return text[: m.start()] + _match_case(_ANTONYMS[word.lower()], word) + text[m.end():]


def generate_hard_negatives(questions: list[str], per_kind: int = 500, seed: int = 0) -> list[HardNegative]:
    """Up to ``per_kind`` perturbed pairs of each kind, from distinct questions."""
    rng = random.Random(seed)
    pool = list(dict.fromkeys(questions))
    rng.shuffle(pool)
    rules = {
        "negation": negate,
        "number": lambda t: change_number(t, rng),
        "entity": lambda t: swap_entity(t, rng),
        "antonym": swap_antonym,
    }
    out: list[HardNegative] = []
    for kind, rule in rules.items():
        made = 0
        for question in pool:
            if made >= per_kind:
                break
            perturbed = rule(question)
            if perturbed and perturbed != question:
                out.append(HardNegative(question, perturbed, kind))
                made += 1
    return out
