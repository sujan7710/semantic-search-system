# Semantic Search System with Fuzzy Clustering and Semantic Cache

This project implements a lightweight semantic search system using the **20 Newsgroups dataset**.  
The system combines **vector embeddings, fuzzy clustering, semantic caching, and a FastAPI service** to efficiently retrieve semantically similar documents.

---

# System Overview

The system pipeline:

User Query  
↓  
Sentence Embedding  
↓  
Semantic Cache Check  
↓  
Cache Hit → Return Cached Result  
Cache Miss → FAISS Vector Search  
↓  
Determine Dominant Cluster  
↓  
Store Result in Cache  

---

# Optimization & Design Decisions

Several design decisions were made to balance performance, scalability, and semantic accuracy.

## Embedding Model Selection

The model used for generating embeddings is:

sentence-transformers/all-MiniLM-L6-v2

This model was selected because:

It produces high-quality sentence embeddings suitable for semantic similarity tasks.

It is lightweight (~80MB) compared to larger transformer models.

It offers a strong balance between accuracy and inference speed, which is important for API-based systems.

It is widely used in production semantic search systems.

This allows the system to generate embeddings efficiently while still preserving semantic meaning between documents and queries.

## Vector Database Choice

The system uses FAISS (Facebook AI Similarity Search) for vector indexing.

FAISS was chosen because:

It is optimized for high-dimensional vector similarity search.

It provides extremely fast nearest-neighbor retrieval.

It is widely used in large-scale semantic search and recommendation systems.

For this dataset (~20k vectors), the IndexFlatL2 index was selected since it performs exact nearest neighbor search, which is sufficiently fast for datasets of this size.

## Fuzzy Clustering Approach

The assignment requires soft clustering rather than hard cluster assignments.

To satisfy this requirement, the system uses:

Gaussian Mixture Models (GMM)

Unlike algorithms such as K-Means that assign each document to a single cluster, GMM provides a probability distribution across clusters.

Example:

Cluster 1 → 0.12
Cluster 2 → 0.73
Cluster 3 → 0.15

This reflects the reality that many documents belong to multiple overlapping topics, which is common in discussion forums such as newsgroups.

## Choice of Number of Clusters (n = 10)

The dataset contains 20 labeled categories, but many of these categories are semantically related.

Examples include:

comp.graphics

comp.windows.x

comp.sys.mac.hardware

These categories belong to the broader computer technology domain.

Using 10 clusters allows the system to capture broader thematic groups while still preserving meaningful topic separation.

A smaller number of clusters also helps:

reduce model complexity

improve clustering stability

improve cache lookup efficiency

## Semantic Cache Design

The cache stores query embeddings instead of raw text queries.

When a new query arrives:

The query is converted into an embedding.

Cosine similarity is computed against cached embeddings.

If similarity exceeds a threshold (0.85), the cached result is reused.

This enables the system to recognize semantically similar queries such as:

"space rocket launch"
"satellite launch mission"

even though the wording differs.

## Persistence for Faster Startup

Two expensive computations are persisted to disk:

embeddings.npy  → document embeddings
gmm_model.pkl   → trained clustering model

Persisting these artifacts prevents recomputation on every server startup.

This reduces API startup time from approximately 40–60 seconds to under 3 seconds.

## Cluster-aware Cache Lookup

Each cached query is associated with its dominant cluster.

This allows future optimization where cache lookup can be restricted to queries within the same cluster, improving lookup efficiency as the cache grows.

This design ensures the system is:

fast

scalable

semantically meaningful

aligned with real-world search architectures

##  FastAPI Service

The system is exposed through a FastAPI API.

Endpoints:

### POST /query

**Input**

```json
{
  "query": "space rocket launch"
}
```

**Output**

```json
{
  "query": "space rocket launch",
  "cache_hit": true,
  "matched_query": "rocket launch in space",
  "similarity_score": 0.91,
  "result": "...",
  "dominant_cluster": 3
}
```

---

### GET /cache/stats

Returns cache statistics.

```json
{
  "total_entries": 42,
  "hit_count": 17,
  "miss_count": 25,
  "hit_rate": 0.405
}
```

---

### DELETE /cache

Clears the semantic cache.


## Dataset

The system uses the 20 Newsgroups dataset:

https://archive.ics.uci.edu/dataset/113/twenty+newsgroups

The dataset contains ~20,000 documents across 20 topic categories.

## Installation

Clone repository:

git clone https://github.com/sujan7710/semantic-search-system.git
cd semantic-search-system

## Virtual Environment Setup

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment.

Windows:
venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt

## Running the API

Start the server:

```bash
uvicorn app:app --reload
```

Open API docs:

http://127.0.0.1:8000/docs

## Technologies Used:

Python

FastAPI

Sentence Transformers

FAISS

Scikit-learn

NumPy



