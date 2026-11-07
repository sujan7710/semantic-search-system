"""Soft clustering with a Gaussian mixture, persisted without pickle.

Known limitation (to be addressed in Phase 3): with full covariance in 384
dimensions the posteriors saturate, so nearly every point gets >0.99 for one
cluster. It behaves like hard clustering despite producing probabilities.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.mixture import GaussianMixture


class SoftClusterer:
    def __init__(self, n_clusters: int = 10, covariance_type: str = "full", random_state: int = 42):
        self.n_clusters = n_clusters
        self.covariance_type = covariance_type
        self.random_state = random_state
        self._model: GaussianMixture | None = None

    @property
    def is_fitted(self) -> bool:
        return self._model is not None

    def fit(self, embeddings: np.ndarray) -> SoftClusterer:
        self._model = GaussianMixture(
            n_components=self.n_clusters,
            covariance_type=self.covariance_type,
            random_state=self.random_state,
        ).fit(embeddings)
        return self

    def _require_model(self) -> GaussianMixture:
        if self._model is None:
            raise RuntimeError("SoftClusterer is not fitted")
        return self._model

    def predict_proba(self, embeddings: np.ndarray) -> np.ndarray:
        return self._require_model().predict_proba(np.atleast_2d(embeddings))

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        return self._require_model().predict(np.atleast_2d(embeddings))

    # Plain .npz arrays instead of pickle: loading never executes code.
    def save(self, path: str | Path) -> None:
        m = self._require_model()
        np.savez(
            path,
            weights=m.weights_,
            means=m.means_,
            covariances=m.covariances_,
            precisions_cholesky=m.precisions_cholesky_,
            covariance_type=np.array(self.covariance_type),
            random_state=np.array(self.random_state),
        )

    @classmethod
    def load(cls, path: str | Path) -> SoftClusterer:
        with np.load(path, allow_pickle=False) as data:
            covariance_type = str(data["covariance_type"])
            weights = data["weights"]
            clusterer = cls(len(weights), covariance_type, int(data["random_state"]))
            model = GaussianMixture(n_components=len(weights), covariance_type=covariance_type)
            model.weights_ = weights
            model.means_ = data["means"]
            model.covariances_ = data["covariances"]
            model.precisions_cholesky_ = data["precisions_cholesky"]
        model.n_features_in_ = model.means_.shape[1]
        model.converged_ = True
        clusterer._model = model
        return clusterer
