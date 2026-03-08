import os
import numpy as np

from dataset_loader import load_20newsgroups_dataset
from preprocessing import clean_text
from embeddings import EmbeddingModel
from vector_store import VectorStore
from clustering import FuzzyClusterer
from semantic_cache import SemanticCache


def main():

    print("Loading dataset...")

    documents, labels, label_names = load_20newsgroups_dataset()

    # Clean raw text before embedding
    cleaned_docs = [clean_text(doc) for doc in documents]

    embeddings_file = "embeddings.npy"

    embedding_model = EmbeddingModel()

    # Load embeddings if already computed
    if os.path.exists(embeddings_file):

        print("Loading saved embeddings...")
        embeddings = np.load(embeddings_file)

    else:

        print("Generating embeddings...")
        embeddings = embedding_model.encode_documents(cleaned_docs)

        np.save(embeddings_file, embeddings)
        print("Embeddings saved to disk.")

    print("Building FAISS index...")

    vector_store = VectorStore(embeddings)

    print("Training fuzzy clustering model...")

    clusterer = FuzzyClusterer(n_clusters=10)
    clusterer.fit(embeddings)

    # Initialize semantic cache
    cache = SemanticCache()

    print("\nSystem ready. Type queries (type 'exit' to quit)\n")

    while True:

        query = input("Query: ")

        if query.lower() == "exit":
            break

        # Convert query to embedding
        query_embedding = embedding_model.encode_query(query)

        # Check semantic cache first
        hit, entry, score = cache.search(query_embedding)

        if hit:

            print("\nCache hit!")
            print("Matched query:", entry["query"])
            print("Similarity:", round(score, 3))
            print("Result:\n", entry["result"][:300])

        else:

            print("\nCache miss. Searching documents...")

            distances, indices = vector_store.search(query_embedding)

            best_doc = documents[indices[0]]

            # Determine dominant cluster for the query
            query_cluster = clusterer.get_dominant_clusters(
                np.array([query_embedding])
            )[0]

            # Store result in cache
            cache.add(query, query_embedding, best_doc, query_cluster)

            print("Result:\n", best_doc[:300])

        print("\nCache stats:", cache.stats())
        print("-" * 50)


if __name__ == "__main__":
    main()