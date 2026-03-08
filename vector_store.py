import faiss
import numpy as np


class VectorStore:

    def __init__(self, embeddings):

        # Convert embeddings to float32 (required by FAISS)
        self.embeddings = np.array(embeddings).astype("float32")

        # Dimension of embedding vectors
        dimension = self.embeddings.shape[1]

        # IndexFlatL2 performs simple nearest neighbor search
        self.index = faiss.IndexFlatL2(dimension)

        # Add embeddings to index
        self.index.add(self.embeddings)

    def search(self, query_embedding, top_k=5):

        # Convert query to correct format
        query_embedding = np.array([query_embedding]).astype("float32")

        # Search nearest vectors
        distances, indices = self.index.search(query_embedding, top_k)

        return distances[0], indices[0]