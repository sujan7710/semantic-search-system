"""LLM backend using Groq's OpenAI-compatible chat completions API.

Plain ``httpx`` instead of an SDK: one endpoint, explicit retries and
timeouts, and an injectable transport so tests never hit the network.
"""

from __future__ import annotations

import time
from typing import Sequence

import httpx
import numpy as np

from cacheguard.backends import BackendError, BackendResponse

_RETRYABLE = {408, 409, 429, 500, 502, 503, 504}


class GroqBackend:
    name = "groq"

    def __init__(
        self,
        api_key: str,
        model: str = "openai/gpt-oss-20b",
        system_prompt: str = "You are a concise, accurate assistant.",
        temperature: float = 0.0,
        max_tokens: int = 512,
        reasoning_effort: str | None = "low",
        base_url: str = "https://api.groq.com/openai/v1",
        timeout_s: float = 30.0,
        max_retries: int = 2,
        price_input_per_m: float = 0.0,
        price_output_per_m: float = 0.0,
        transport: httpx.BaseTransport | None = None,
        sleep=time.sleep,
    ):
        if not api_key:
            raise ValueError("GROQ_API_KEY is not set")
        self.model = model
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort or None
        self.max_retries = max_retries
        self.price_input_per_m = price_input_per_m
        self.price_output_per_m = price_output_per_m
        self._sleep = sleep
        self._client = httpx.Client(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout_s,
            transport=transport,
        )

    def cache_settings(self) -> dict:
        return {
            "model": self.model,
            "system_prompt": self.system_prompt,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "reasoning_effort": self.reasoning_effort,
        }

    def cost_usd(self, usage: dict) -> float:
        return (
            usage.get("prompt_tokens", 0) * self.price_input_per_m
            + usage.get("completion_tokens", 0) * self.price_output_per_m
        ) / 1_000_000

    def _messages(self, query: str, history: Sequence[dict]) -> list[dict]:
        messages = [dict(m) for m in history]
        if not any(m.get("role") == "system" for m in messages):
            messages.insert(0, {"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": query})
        return messages

    def respond(self, query: str, query_embedding: np.ndarray | None = None, history: Sequence[dict] = ()) -> BackendResponse:
        payload = {
            "model": self.model,
            "messages": self._messages(query, history),
            "temperature": self.temperature,
            "max_completion_tokens": self.max_tokens,
        }
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort

        start = time.perf_counter()
        data = self._post_with_retries(payload)
        latency_ms = (time.perf_counter() - start) * 1000
        try:
            answer = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise BackendError(f"unexpected Groq response shape: {exc}") from exc
        usage = {k: int(v) for k, v in (data.get("usage") or {}).items() if k.endswith("_tokens") and isinstance(v, (int, float))}
        return BackendResponse(
            answer=answer.strip(),
            meta={
                "model": data.get("model", self.model),
                "usage": usage,
                "cost_usd": self.cost_usd(usage),
                "latency_ms": round(latency_ms, 1),
            },
        )

    def _post_with_retries(self, payload: dict) -> dict:
        last_error: BackendError | None = None
        for attempt in range(self.max_retries + 1):
            response: httpx.Response | None = None
            try:
                response = self._client.post("/chat/completions", json=payload)
            except httpx.TimeoutException:
                last_error = BackendError("Groq request timed out", retryable=True)
            except httpx.HTTPError as exc:
                last_error = BackendError(f"could not reach Groq: {type(exc).__name__}", retryable=True)
            else:
                if response.status_code == 200:
                    return response.json()
                last_error = BackendError(
                    f"Groq returned {response.status_code}: {_error_message(response)}",
                    status=response.status_code,
                    retryable=response.status_code in _RETRYABLE,
                )
                if not last_error.retryable:
                    raise last_error
            if attempt < self.max_retries:
                self._sleep(_backoff_seconds(attempt, response))
        raise last_error  # type: ignore[misc]

    def close(self) -> None:
        self._client.close()


def _backoff_seconds(attempt: int, response: httpx.Response | None) -> float:
    retry_after = response.headers.get("retry-after") if response is not None else None
    try:
        if retry_after is not None:
            return min(float(retry_after), 5.0)
    except ValueError:
        pass
    return 0.5 * (2**attempt)


def _error_message(response: httpx.Response) -> str:
    try:
        return str(response.json().get("error", {}).get("message", ""))[:300]
    except ValueError:
        return response.text[:300]
