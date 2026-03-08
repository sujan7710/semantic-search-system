import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

from dataset_loader import load_20newsgroups_dataset
from preprocessing import clean_text
from embeddings import EmbeddingModel
from vector_store import VectorStore
from clustering import FuzzyClusterer
from semantic_cache import SemanticCache


app = FastAPI(title="Semantic Search System")


# Request body format for /query endpoint
class QueryRequest(BaseModel):
    query: str


print("Loading system...")

# Load dataset
documents, labels, label_names = load_20newsgroups_dataset()

# Clean raw text
cleaned_docs = [clean_text(doc) for doc in documents]

embedding_model = EmbeddingModel()

# Load saved embeddings to avoid recomputing
embeddings = np.load("embeddings.npy")

# Build vector search index
vector_store = VectorStore(embeddings)

# Train clustering model once during startup
clusterer = FuzzyClusterer(n_clusters=10)
clusterer.fit(embeddings)

# Initialize semantic cache
cache = SemanticCache()

print("System ready")


@app.post("/query")
def query_system(request: QueryRequest):

    query = request.query

    # Convert query to embedding
    query_embedding = embedding_model.encode_query(query)

    # Check semantic cache
    hit, entry, score = cache.search(query_embedding)

    if hit:
        return {
            "query": query,
            "cache_hit": True,
            "matched_query": entry["query"],
            "similarity_score": float(score),
            "result": entry["result"][:300],
            "dominant_cluster": entry["cluster"]
        }

    # If cache miss, search vector database
    distances, indices = vector_store.search(query_embedding)

    best_doc = documents[indices[0]]

    # Determine cluster for the query
    query_cluster = clusterer.get_dominant_clusters(
        np.array([query_embedding])
    )[0]

    # Store result in cache
    cache.add(query, query_embedding, best_doc, query_cluster)

    return {
        "query": query,
        "cache_hit": False,
        "matched_query": None,
        "similarity_score": float(score),
        "result": best_doc[:300],
        "dominant_cluster": int(query_cluster)
    }


@app.get("/cache/stats")
def cache_stats():
    return cache.stats()


@app.delete("/cache")
def clear_cache():
    cache.clear()
    return {"message": "Cache cleared"}