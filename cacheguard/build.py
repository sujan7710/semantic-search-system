"""Build the search index files from the raw dataset.

Usage:  python -m cacheguard.build [--data-dir DIR] [--out DIR]
"""

from __future__ import annotations

import argparse
import time

from cacheguard.artifacts import build_artifacts, save_artifacts
from cacheguard.config import Settings
from cacheguard.data.newsgroups import load_newsgroups
from cacheguard.embeddings import SentenceTransformerEmbedder


def main(argv: list[str] | None = None) -> None:
    settings = Settings.from_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(settings.data_dir))
    parser.add_argument("--out", default=str(settings.artifact_dir))
    args = parser.parse_args(argv)

    t0 = time.perf_counter()
    documents, stats = load_newsgroups(args.data_dir)
    print(
        f"Loaded {len(documents)} documents from {stats.files_read} files "
        f"({stats.duplicates} duplicates, {stats.too_short} too short, {stats.unreadable} unreadable) "
        f"in {time.perf_counter() - t0:.1f}s"
    )

    t1 = time.perf_counter()
    embedder = SentenceTransformerEmbedder(settings.embedding_model)
    artifacts = build_artifacts(
        documents,
        embedder,
        n_clusters=settings.n_clusters,
        covariance_type=settings.cluster_covariance,
        random_seed=settings.random_seed,
        show_progress=True,
    )
    print(f"Embedded and clustered in {time.perf_counter() - t1:.1f}s")

    save_artifacts(artifacts, args.out)
    print(f"Saved index to {args.out}")


if __name__ == "__main__":
    main()
