"""Phase 4: Groq backend (mocked HTTP), limits, OpenAI-compatible endpoint, demo page."""

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from cacheguard.api import create_app
from cacheguard.backends import BackendError
from cacheguard.backends.groq import GroqBackend
from cacheguard.config import Settings, load_dotenv
from cacheguard.guard import Guard, GuardSettings
from cacheguard.limits import BudgetExceeded, DailyBudget, RateLimiter
from cacheguard.policy import GlobalThreshold
from cacheguard.service import CacheGuardService
from cacheguard.store import CacheStore


class FakeGroq:
    """httpx transport that records requests and replays canned responses."""

    def __init__(self, responses=None):
        self.requests = []
        self.responses = list(responses or [])

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.requests.append({"headers": dict(request.headers), "body": body, "path": request.url.path})
        if self.responses:
            status, payload, headers = self.responses.pop(0)
            return httpx.Response(status, json=payload, headers=headers or {})
        question = body["messages"][-1]["content"]
        return httpx.Response(200, json={
            "model": body["model"],
            "choices": [{"message": {"role": "assistant", "content": f"  Answer to: {question}  "}}],
            "usage": {"prompt_tokens": 40, "completion_tokens": 60, "total_tokens": 100, "queue_time": 0.01},
        })


def make_groq(fake=None, **kwargs):
    fake = fake or FakeGroq()
    sleeps = []
    backend = GroqBackend(
        api_key="test-key",
        transport=httpx.MockTransport(fake),
        sleep=sleeps.append,
        price_input_per_m=1.0,
        price_output_per_m=2.0,
        **kwargs,
    )
    return backend, fake, sleeps


# --- Groq backend -------------------------------------------------------------------

def test_groq_request_shape_and_response_parsing():
    backend, fake, _ = make_groq(model="openai/gpt-oss-20b", system_prompt="Be brief.")
    r = backend.respond("What is RAM?", None, history=[{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}])
    sent = fake.requests[0]
    assert sent["path"] == "/openai/v1/chat/completions"
    assert sent["headers"]["authorization"] == "Bearer test-key"
    assert sent["body"]["model"] == "openai/gpt-oss-20b"
    assert sent["body"]["temperature"] == 0.0
    assert sent["body"]["reasoning_effort"] == "low"
    assert [m["role"] for m in sent["body"]["messages"]] == ["system", "user", "assistant", "user"]
    assert sent["body"]["messages"][0]["content"] == "Be brief."
    assert r.answer == "Answer to: What is RAM?"
    assert r.meta["usage"] == {"prompt_tokens": 40, "completion_tokens": 60, "total_tokens": 100}
    assert r.meta["cost_usd"] == pytest.approx((40 * 1.0 + 60 * 2.0) / 1e6)


def test_groq_keeps_a_system_message_from_history_and_omits_empty_reasoning_effort():
    backend, fake, _ = make_groq(reasoning_effort="")
    backend.respond("q", None, history=[{"role": "system", "content": "custom"}])
    body = fake.requests[0]["body"]
    assert [m["content"] for m in body["messages"] if m["role"] == "system"] == ["custom"]
    assert "reasoning_effort" not in body


def test_groq_retries_overload_then_succeeds():
    fake = FakeGroq([(429, {"error": {"message": "slow down"}}, {"retry-after": "2"}), (503, {}, None)])
    backend, fake, sleeps = make_groq(fake)
    assert backend.respond("q").answer == "Answer to: q"
    assert len(fake.requests) == 3
    assert sleeps == [2.0, 1.0]


def test_groq_does_not_retry_client_errors_and_reports_them():
    fake = FakeGroq([(401, {"error": {"message": "Invalid API Key"}}, None)])
    backend, fake, _ = make_groq(fake)
    with pytest.raises(BackendError, match="401: Invalid API Key") as err:
        backend.respond("q")
    assert not err.value.retryable and len(fake.requests) == 1


def test_groq_gives_up_after_retries():
    fake = FakeGroq([(500, {}, None)] * 3)
    backend, _, _ = make_groq(fake, max_retries=2)
    with pytest.raises(BackendError) as err:
        backend.respond("q")
    assert err.value.retryable and err.value.status == 500


def test_groq_requires_key():
    with pytest.raises(ValueError):
        GroqBackend(api_key="")


# --- limits and config -----------------------------------------------------------------

def test_rate_limiter_window():
    now = [0.0]
    limiter = RateLimiter(2, clock=lambda: now[0])
    assert limiter.check("a") is None and limiter.check("a") is None
    assert limiter.check("a") == pytest.approx(60)
    assert limiter.check("b") is None  # per client
    now[0] = 61
    assert limiter.check("a") is None


def test_daily_budget_resets_and_refunds():
    day = ["2026-11-07"]
    budget = DailyBudget(2, today=lambda: day[0])
    budget.consume()
    budget.consume()
    with pytest.raises(BudgetExceeded):
        budget.consume()
    budget.refund()
    budget.consume()
    day[0] = "2026-11-08"
    assert budget.status()["remaining"] == 2


def test_dotenv_does_not_override_environment(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text('# comment\nGROQ_API_KEY="from-file"\nexport CACHEGUARD_GROQ_MODEL=m1\nCACHEGUARD_TOP_K=9\n')
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("CACHEGUARD_GROQ_MODEL", raising=False)
    monkeypatch.setenv("CACHEGUARD_TOP_K", "3")
    s = Settings.from_env(dotenv=env)
    assert s.groq_api_key == "from-file" and s.groq_model == "m1" and s.top_k == 3
    assert s.resolved_backend() == "groq"
    monkeypatch.delenv("GROQ_API_KEY")
    assert Settings.from_env(dotenv=None).resolved_backend() == "newsgroups"


# --- service + API with the Groq backend --------------------------------------------------

@pytest.fixture
def groq_service(embedder):
    backend, fake, _ = make_groq()
    store = CacheStore(embedder.dim, GlobalThreshold(0.85))
    service = CacheGuardService(embedder, store, backend, budget=DailyBudget(100))
    service.fake = fake
    return service


def client_for(service, **settings):
    return TestClient(create_app(service, Settings(**settings)))


def test_query_hit_reports_savings(groq_service):
    client = client_for(groq_service)
    first = client.post("/query", json={"query": "how do I reset my password"}).json()
    assert first["cache_hit"] is False and first["backend"]["usage"]["total_tokens"] == 100
    again = client.post("/query", json={"query": "how do I reset my password"}).json()
    assert again["cache_hit"] is True and again["answer"] == first["answer"]
    assert again["saved"]["tokens"] == 100 and again["saved"]["cost_usd"] > 0
    assert len(groq_service.fake.requests) == 1
    stats = client.get("/stats").json()
    assert stats["savings"]["backend_calls"] == 1 and stats["savings"]["tokens_saved"] == 100
    assert stats["budget"]["used"] == 1
    assert stats["backend"]["model"] == "openai/gpt-oss-20b"


def test_bypass_and_unguarded_flags(groq_service):
    client = client_for(groq_service)
    client.post("/query", json={"query": "what is RAM"})
    bypass = client.post("/query", json={"query": "what is RAM", "cache": False}).json()
    assert bypass["decision"]["stage"] == "bypass" and bypass["entry_id"] is None
    assert len(groq_service.fake.requests) == 2
    assert client_for(groq_service, allow_unguarded=False).post(
        "/query", json={"query": "what is RAM", "guard": False}).status_code == 403


def test_explain_is_a_dry_run(groq_service):
    groq_service.guard = Guard(GuardSettings(gate=0.6, skip_verify_above=0.99))
    client = client_for(groq_service)
    assert client.post("/explain", json={"query": "how do I cancel my order"}).json()["plain"] is None
    client.post("/query", json={"query": "how do I cancel my order"})
    ex = client.post("/explain", json={"query": "how do I not cancel my order"}).json()
    assert ex["nearest_query"] == "how do I cancel my order"
    assert ex["plain"]["hit"] is True  # what a plain cache would wrongly do
    assert ex["guard"] == {"hit": False, "stage": "rules", "reason": "negation", "verifier_score": None, "gate": 0.6}
    assert len(groq_service.fake.requests) == 1 and len(groq_service.store) == 1  # no call, nothing stored


def test_budget_exhaustion_still_serves_cached_answers(embedder):
    backend, _, _ = make_groq()
    service = CacheGuardService(embedder, CacheStore(embedder.dim), backend, budget=DailyBudget(1))
    client = client_for(service)
    client.post("/query", json={"query": "what is a cache"})
    assert client.post("/query", json={"query": "what is a cache"}).json()["cache_hit"] is True
    r = client.post("/query", json={"query": "who won the world cup"})
    assert r.status_code == 503 and "daily limit" in r.json()["detail"]


def test_backend_errors_map_to_gateway_statuses(embedder):
    fake = FakeGroq([(401, {"error": {"message": "Invalid API Key"}}, None)] + [(503, {}, None)] * 3)
    backend, _, _ = make_groq(fake)
    budget = DailyBudget(10)
    client = client_for(CacheGuardService(embedder, CacheStore(embedder.dim), backend, budget=budget))
    assert client.post("/query", json={"query": "a"}).status_code == 502
    assert client.post("/query", json={"query": "b"}).status_code == 503
    assert budget.status()["used"] == 0  # failed calls are refunded


def test_rate_limit_returns_429(groq_service):
    client = client_for(groq_service, rate_limit_per_minute=2)
    assert client.post("/query", json={"query": "a"}).status_code == 200
    assert client.post("/explain", json={"query": "a"}).status_code == 200
    r = client.post("/query", json={"query": "a"})
    assert r.status_code == 429 and "Retry-After" in r.headers


def test_admin_token_protects_cache_admin_but_not_own_demo_namespace(groq_service):
    client = client_for(groq_service, admin_token="secret")
    client.post("/query", json={"query": "a question", "namespace": "demo-abc"})
    assert client.delete("/cache").status_code == 403
    assert client.post("/cache/invalidate", json={"namespace": "default"}).status_code == 403
    assert client.post("/cache/invalidate", json={"namespace": "demo-abc"}).json() == {"removed": 1}
    assert client.delete("/cache", headers={"X-Admin-Token": "secret"}).status_code == 200


# --- OpenAI-compatible endpoint ----------------------------------------------------------------

def chat(client, content, **extra):
    return client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": content}], **extra})


def test_chat_completions_shape_and_cache_hit(groq_service):
    client = client_for(groq_service)
    first = chat(client, "how do I reset my password").json()
    assert first["object"] == "chat.completion"
    assert first["choices"][0]["message"] == {"role": "assistant", "content": "Answer to: how do I reset my password"}
    assert first["usage"]["total_tokens"] == 100 and first["cacheguard"]["cache_hit"] is False
    second = chat(client, "how do I reset my password", model="openai/gpt-oss-20b").json()
    assert second["cacheguard"]["cache_hit"] is True and second["usage"]["total_tokens"] == 0
    assert second["cacheguard"]["saved"]["tokens"] == 100


def test_chat_completions_conversation_history_is_part_of_the_key(groq_service):
    client = client_for(groq_service)
    turn = [{"role": "user", "content": "tell me about Python"}, {"role": "assistant", "content": "It's a language."}]
    client.post("/v1/chat/completions", json={"messages": turn + [{"role": "user", "content": "how do I install it"}]})
    other = [{"role": "user", "content": "tell me about Docker"}, {"role": "assistant", "content": "It's a container tool."}]
    r = client.post("/v1/chat/completions", json={"messages": other + [{"role": "user", "content": "how do I install it"}]}).json()
    assert r["cacheguard"]["cache_hit"] is False  # same words, different conversation
    sent = groq_service.fake.requests[-1]["body"]["messages"]
    assert [m["content"] for m in sent][-3:] == ["tell me about Docker", "It's a container tool.", "how do I install it"]


def test_chat_completions_temperature_bypass_and_validation(groq_service):
    client = client_for(groq_service)
    chat(client, "a question")
    assert chat(client, "a question", temperature=0.9).json()["cacheguard"]["decision"]["stage"] == "bypass"
    assert chat(client, "a question", stream=True).status_code == 400
    assert chat(client, "a question", model="some-other-model").status_code == 400
    bad = client.post("/v1/chat/completions", json={"messages": [{"role": "assistant", "content": "hi"}]})
    assert bad.status_code == 400


def test_demo_page_is_served(groq_service):
    r = client_for(groq_service).get("/")
    assert r.status_code == 200 and "CacheGuard" in r.text and "/explain" in r.text
