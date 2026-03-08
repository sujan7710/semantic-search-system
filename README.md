# Semantic Search System with Fuzzy Clustering and Semantic Cache

This project implements a lightweight semantic search system using the **20 Newsgroups dataset**.  
The system combines **vector embeddings, fuzzy clustering, semantic caching, and a FastAPI service** to efficiently retrieve semantically similar documents.

This project was developed as part of the **Trademarkia AI/ML Engineer Assignment**.

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

# Key Components

## 1. Embedding & Vector Database

Documents are converted into vector embeddings using:
sentence-transformers/all-MiniLM-L6-v2


Embeddings are stored in a **FAISS vector index** for efficient similarity search.

Embeddings are persisted to disk (`embeddings.npy`) to avoid recomputation.

---

## 2. Fuzzy Clustering

A **Gaussian Mixture Model (GMM)** is used for fuzzy clustering.

Each document receives a **probability distribution across clusters**, allowing overlapping topics.

Example:
Cluster 1 → 0.12
Cluster 2 → 0.73
Cluster 3 → 0.15


The trained model is saved as:
gmm_model.pkl


This reduces server startup time significantly.

---

## 3. Semantic Cache

A custom semantic cache is implemented from scratch.

Instead of exact query matching, the cache compares **query embeddings using cosine similarity**.

If similarity exceeds a threshold (0.85), the cached result is reused.

This reduces repeated computation for semantically similar queries.

---

## 4. FastAPI Service

The system is exposed through a FastAPI API.

Endpoints:

### POST /query

Input:

```json
{
  "query": "space rocket launch"
}
output:
{
  "query": "...",
  "cache_hit": true,
  "matched_query": "...",
  "similarity_score": 0.91,
  "result": "...",
  "dominant_cluster": 3
}

GET /cache/stats

Returns cache statistics.

{
  "total_entries": 42,
  "hit_count": 17,
  "miss_count": 25,
  "hit_rate": 0.405
}

DELETE /cache

Clears the semantic cache.

Dataset

The system uses the 20 Newsgroups dataset:

https://archive.ics.uci.edu/dataset/113/twenty+newsgroups

The dataset contains ~20,000 documents across 20 topic categories.

Installation

Clone repository:

git clone https://github.com/sujan7710/semantic-search-system.git
cd semantic-search-system

Install dependencies:

pip install -r requirements.txt

Running the API

Start the server:

uvicorn app:app --reload

Open API docs:

http://127.0.0.1:8000/docs

Technologies Used

Python

FastAPI

Sentence Transformers

FAISS

Scikit-learn

NumPy

Optimization

To reduce startup time:

Document embeddings are saved to disk

The trained clustering model is persisted

This prevents expensive recomputation on each run.

