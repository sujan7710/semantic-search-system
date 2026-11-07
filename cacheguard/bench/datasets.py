"""Labeled question pairs for cache evaluation.

* QQP (Quora Question Pairs, GLUE validation split, ~40k pairs): realistic
  user questions labeled duplicate / not duplicate. The GLUE test split has no
  labels, so validation is split into calibration and test halves *by
  question group*, so no question appears on both sides.
* PAWS (labeled_final): pairs with almost identical words but often different
  meaning ("flights from NY to Florida" vs "flights from Florida to NY").
  This is the adversarial case a threshold-only cache gets wrong.

Label caveat: QQP "not duplicate" is not exhaustive. Two questions never
paired by annotators may still be paraphrases, so a measured "wrong hit" in
workload replay can sometimes be a correct hit. Error rates are therefore
conservative (an upper bound).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

_SOURCES = {
    "qqp": ("nyu-mll/glue", "qqp/validation-00000-of-00001.parquet", "question1", "question2"),
    "paws-test": ("google-research-datasets/paws", "labeled_final/test-00000-of-00001.parquet", "sentence1", "sentence2"),
    "paws-validation": ("google-research-datasets/paws", "labeled_final/validation-00000-of-00001.parquet", "sentence1", "sentence2"),
}


@dataclass(frozen=True)
class PairSet:
    name: str
    text_a: list[str]
    text_b: list[str]
    labels: list[int]  # 1 = same meaning (a correct cache hit), 0 = different

    def __len__(self) -> int:
        return len(self.labels)


def _download(source: str, cache_dir: Path) -> pd.DataFrame:
    from huggingface_hub import hf_hub_download

    repo, filename, _, _ = _SOURCES[source]
    path = hf_hub_download(repo, filename, repo_type="dataset", local_dir=cache_dir / repo.replace("/", "__"))
    return pd.read_parquet(path)


def _clean(df: pd.DataFrame, col_a: str, col_b: str) -> pd.DataFrame:
    df = df[[col_a, col_b, "label"]].rename(columns={col_a: "a", col_b: "b"})
    df = df.dropna()
    df["a"] = df["a"].astype(str).str.strip()
    df["b"] = df["b"].astype(str).str.strip()
    return df[(df["a"] != "") & (df["b"] != "") & (df["a"] != df["b"])].reset_index(drop=True)


def _to_pairset(name: str, df: pd.DataFrame) -> PairSet:
    return PairSet(name, df["a"].tolist(), df["b"].tolist(), df["label"].astype(int).tolist())


def question_groups(text_a: list[str], text_b: list[str], labels: list[int]) -> dict[str, int]:
    """Union-find over duplicate pairs: questions in one group share meaning.

    Every question gets a group id; questions with no duplicate are singletons.
    """
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b, label in zip(text_a, text_b, labels):
        ra, rb = find(a), find(b)
        if label == 1 and ra != rb:
            parent[ra] = rb

    roots: dict[str, int] = {}
    return {q: roots.setdefault(find(q), len(roots)) for q in list(parent)}


def _stable_half(key: str) -> int:
    return int(hashlib.sha1(key.encode("utf-8")).hexdigest(), 16) % 2


def load_qqp(cache_dir: str | Path, split: str) -> PairSet:
    """QQP validation, deterministically split into 'calibration' or 'test'.

    A pair goes to a side by the hash of the group of its first question; pairs
    whose two questions would land on different sides are dropped, so the
    halves share no questions.
    """
    if split not in {"calibration", "test"}:
        raise ValueError("split must be 'calibration' or 'test'")
    _, _, col_a, col_b = _SOURCES["qqp"]
    df = _clean(_download("qqp", Path(cache_dir)), col_a, col_b)
    groups = question_groups(df["a"].tolist(), df["b"].tolist(), df["label"].tolist())
    # Groups are connected by duplicate links only; non-duplicate pairs can still
    # bridge two groups, so the side is decided per question and both must agree.
    side_a = df["a"].map(lambda q: _stable_half(f"g{groups[q]}"))
    side_b = df["b"].map(lambda q: _stable_half(f"g{groups[q]}"))
    want = 0 if split == "calibration" else 1
    return _to_pairset(f"qqp-{split}", df[(side_a == want) & (side_b == want)].reset_index(drop=True))


def load_paws(cache_dir: str | Path, split: str) -> PairSet:
    """PAWS labeled_final: 'calibration' = validation split, 'test' = test split."""
    source = {"calibration": "paws-validation", "test": "paws-test"}.get(split)
    if source is None:
        raise ValueError("split must be 'calibration' or 'test'")
    _, _, col_a, col_b = _SOURCES[source]
    return _to_pairset(f"paws-{split}", _clean(_download(source, Path(cache_dir)), col_a, col_b))
