"""Build, save and load the precomputed search index files.

Layout of an artifact folder:
    manifest.json     model name, preprocessing version, counts
    documents.jsonl   one cleaned document per line
    embeddings.npy    float32 (n, dim), unit-normalized
    clusters.npz      Gaussian mixture parameters (no pickle)
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from cacheguard.clustering import SoftClusterer
from cacheguard.data.newsgroups import PREPROCESSING_VERSION, Document
from cacheguard.embeddings import Embedder

MANIFEST = "manifest.json"
DOCUMENTS = "documents.jsonl"
EMBEDDINGS = "embeddings.npy"
CLUSTERS = "clusters.npz"


class ArtifactError(RuntimeError):
    pass


@dataclass
class IndexArtifacts:
    documents: list[Document]
    embeddings: np.ndarray
    clusterer: SoftClusterer
    manifest: dict


def build_artifacts(
    documents: list[Document],
    embedder: Embedder,
    n_clusters: int,
    covariance_type: str,
    random_seed: int,
    show_progress: bool = False,
) -> IndexArtifacts:
    embeddings = embedder.encode([d.text for d in documents], show_progress=show_progress)
    clusterer = SoftClusterer(n_clusters, covariance_type, random_seed).fit(embeddings)
    manifest = {
        "embedding_model": embedder.model_name,
        "embedding_dim": int(embeddings.shape[1]),
        "preprocessing_version": PREPROCESSING_VERSION,
        "n_documents": len(documents),
        "n_clusters": n_clusters,
        "cluster_covariance": covariance_type,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    return IndexArtifacts(documents, embeddings, clusterer, manifest)


def save_artifacts(artifacts: IndexArtifacts, out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / DOCUMENTS, "w", encoding="utf-8") as f:
        for doc in artifacts.documents:
            f.write(json.dumps(asdict(doc), ensure_ascii=False) + "\n")
    np.save(out_dir / EMBEDDINGS, artifacts.embeddings.astype(np.float32))
    artifacts.clusterer.save(out_dir / CLUSTERS)
    # Manifest last: its presence marks a complete build.
    (out_dir / MANIFEST).write_text(json.dumps(artifacts.manifest, indent=2), encoding="utf-8")


def load_artifacts(out_dir: str | Path, expected_model: str | None = None) -> IndexArtifacts:
    out_dir = Path(out_dir)
    manifest_path = out_dir / MANIFEST
    if not manifest_path.exists():
        raise ArtifactError(f"no index found in {out_dir}. Build it first: python -m cacheguard.build")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if manifest.get("preprocessing_version") != PREPROCESSING_VERSION:
        raise ArtifactError(
            f"index in {out_dir} was built with preprocessing v{manifest.get('preprocessing_version')}, "
            f"code is v{PREPROCESSING_VERSION}. Rebuild: python -m cacheguard.build"
        )
    if expected_model and manifest.get("embedding_model") != expected_model:
        raise ArtifactError(
            f"index in {out_dir} was built with {manifest.get('embedding_model')}, "
            f"but settings use {expected_model}. Rebuild: python -m cacheguard.build"
        )

    with open(out_dir / DOCUMENTS, encoding="utf-8") as f:
        documents = [Document(**json.loads(line)) for line in f]
    embeddings = np.load(out_dir / EMBEDDINGS, allow_pickle=False)
    clusterer = SoftClusterer.load(out_dir / CLUSTERS)

    if len(documents) != embeddings.shape[0] or len(documents) != manifest.get("n_documents"):
        raise ArtifactError(f"index in {out_dir} is inconsistent; rebuild it")
    return IndexArtifacts(documents, embeddings, clusterer, manifest)
