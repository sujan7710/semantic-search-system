import pytest
from fastapi.testclient import TestClient

from cacheguard.api import create_app


@pytest.fixture
def client(service):
    return TestClient(create_app(service))


def test_health(client):
    body = client.get("/health").json()
    assert body == {"status": "ok", "backend": "newsgroups-search", "cache_entries": 0}


def test_first_query_misses_and_returns_ranked_sources(client):
    body = client.post("/query", json={"query": "shuttle launch delayed by weather"}).json()
    assert body["cache_hit"] is False
    assert body["matched_query"] is None
    assert body["cache_similarity"] is None  # cache was empty
    assert body["sources"][0]["newsgroup"] == "sci.space"
    scores = [s["score"] for s in body["sources"]]
    assert scores == sorted(scores, reverse=True)
    assert body["answer"] == body["sources"][0]["snippet"]
    assert isinstance(body["dominant_cluster"], int)


def test_repeated_query_hits_the_cache(client):
    client.post("/query", json={"query": "shuttle launch delayed by weather"})
    repeat = client.post("/query", json={"query": "  shuttle launch delayed by weather "}).json()
    assert repeat["cache_hit"] is True
    assert repeat["matched_query"] == "shuttle launch delayed by weather"
    assert repeat["cache_similarity"] == pytest.approx(1.0, abs=1e-5)


def test_unrelated_query_misses_and_reports_nearest_similarity(client):
    client.post("/query", json={"query": "shuttle launch delayed by weather"})
    body = client.post("/query", json={"query": "brake pads for my sedan"}).json()
    assert body["cache_hit"] is False
    assert 0.0 <= body["cache_similarity"] < 0.85


def test_stats_and_clear(client):
    client.post("/query", json={"query": "vaccine trial results"})
    client.post("/query", json={"query": "vaccine trial results"})
    assert client.get("/cache/stats").json()["hit_count"] == 1
    assert client.delete("/cache").json() == {"message": "Cache cleared"}
    assert client.get("/cache/stats").json()["total_entries"] == 0


@pytest.mark.parametrize("payload", [{"query": ""}, {"query": "   "}, {"query": "x" * 2001}, {}])
def test_invalid_queries_are_rejected(client, payload):
    assert client.post("/query", json=payload).status_code == 422
