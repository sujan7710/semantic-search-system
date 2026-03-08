import numpy as np
from sklearn.mixture import GaussianMixture
import joblib
import os


class FuzzyClusterer:

    def __init__(self, n_clusters=10, model_path="gmm_model.pkl"):

        self.n_clusters = n_clusters
        self.model_path = model_path
        self.model = None

    def fit(self, embeddings):

        # If saved model exists, load it
        if os.path.exists(self.model_path):

            print("Loading saved clustering model...")
            self.model = joblib.load(self.model_path)

        else:

            print("Training clustering model...")

            self.model = GaussianMixture(
                n_components=self.n_clusters,
                random_state=42
            )

            self.model.fit(embeddings)

            # Save trained model
            joblib.dump(self.model, self.model_path)

            print("Clustering model saved.")

    def get_cluster_probabilities(self, embeddings):

        probs = self.model.predict_proba(embeddings)

        return probs

    def get_dominant_clusters(self, embeddings):

        clusters = self.model.predict(embeddings)

        return clusters