"""Phase 5: metrics, request logs, mock backend, settings."""

import json
import logging

import pytest
from fastapi.testclient import TestClient

from cacheguard.api import create_app
from cacheguard.backends.mock import MockLLMBackend
from cacheguard.config import Settings
from cacheguard.metrics import Counter, Histogram, Metrics
from cacheguard.service import CacheGuardService
from cacheguard.store import CacheStore


def test_histogram_buckets_are_cumulative():
    h = Histogram("lat", "help", ("route",), buckets=(10, 100))
    for v in (5, 10, 50, 500):
        h.observe(v, route="/q")
    text = "\n".join(h.render())
    assert 'lat_bucket{route="/q",le="10"} 2' in text  # 5 and 10 (le is inclusive)
    assert 'lat_bucket{route="/q",le="100"} 3' in text
    assert 'lat_bucket{route="/q",le="+Inf"} 4' in text
    assert 'lat_sum{route="/q"} 565' in text and 'lat_count{route="/q"} 4' in text


def test_counter_escapes_label_values():
    c = Counter("c_total", "help", ("reason",))
    c.inc(reason='a"b\\c')
    assert 'c_total{reason="a\\"b\\\\c"} 1' in "\n".join(c.render())


@pytest.fixture
def mock_client(embedder):
    service = CacheGuardService(embedder, CacheStore(embedder.dim), MockLLMBackend(delay_ms=0))
    return TestClient(create_app(service, Settings(rate_limit_per_minute=1)))


def test_metrics_endpoint_counts_lookups_requests_and_rejections(mock_client):
    mock_client.post("/query", json={"query": "what is a cache"})
    mock_client.post("/query", json={"query": "what is a cache"})  # rate limited (1/min)
    text = mock_client.get("/metrics").text
    assert 'cacheguard_lookups_total{outcome="miss",stage="none",reason="empty_cache"} 1' in text
    assert 'cacheguard_http_requests_total{route="/query",method="POST",status="200"} 1' in text
    assert 'cacheguard_http_requests_total{route="/query",method="POST",status="429"} 1' in text
    assert 'cacheguard_limit_rejections_total{limit="rate"} 1' in text
    assert 'cacheguard_backend_ms_count{backend="mock"} 1' in text
    assert "cacheguard_cache_entries 1" in text
    assert "# TYPE cacheguard_query_ms histogram" in text


def test_metrics_record_hits_and_savings(embedder):
    service = CacheGuardService(embedder, CacheStore(embedder.dim), MockLLMBackend(delay_ms=0, prompt_tokens=10, completion_tokens=20))
    client = TestClient(create_app(service, Settings(rate_limit_per_minute=None)))
    client.post("/query", json={"query": "what is a cache"})
    hit = client.post("/query", json={"query": "what is a cache"}).json()
    assert hit["cache_hit"] and hit["saved"]["tokens"] == 30
    text = client.get("/metrics").text
    assert 'cacheguard_lookups_total{outcome="hit",stage="threshold",reason="above_threshold"} 1' in text
    assert "cacheguard_tokens_saved_total 30" in text


def test_request_log_has_decision_but_not_the_question(mock_client, caplog):
    with caplog.at_level(logging.INFO, logger="cacheguard.requests"):
        mock_client.post("/query", json={"query": "my secret medical question"})
    records = [json.loads(r.getMessage()) for r in caplog.records if r.name == "cacheguard.requests"]
    assert records and records[0]["event"] == "query" and records[0]["outcome"] == "miss"
    assert len(records[0]["query_sha"]) == 12
    assert "secret" not in caplog.text


def test_mock_backend_and_settings():
    r = MockLLMBackend(delay_ms=0, prompt_tokens=1, completion_tokens=2).respond("hi")
    assert r.answer == "(mock answer) hi" and r.meta["usage"]["total_tokens"] == 3
    assert Settings(backend="mock").resolved_backend() == "mock"
    with pytest.raises(ValueError):
        Settings(backend="nope").resolved_backend()
    assert Settings().torch_threads == 1
