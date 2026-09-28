"""LiteLLMBackend: usage/cost parsing, 429 retry, and request spacing (LiteLLM stubbed)."""

import time
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("litellm")

from graphwalk.llm import LLMError, Message
from graphwalk.llm.litellm_backend import LiteLLMBackend

MESSAGES = [Message(role="user", content="hi")]


def _response(text: str = "OK") -> Any:
    usage = SimpleNamespace(prompt_tokens=9, completion_tokens=2, cost=3.9e-06)
    choice = SimpleNamespace(message=SimpleNamespace(content=text))
    return SimpleNamespace(usage=usage, choices=[choice], model="openai/x")


def backend(monkeypatch: pytest.MonkeyPatch, fail_times: int, **kwargs: Any) -> LiteLLMBackend:
    llm = LiteLLMBackend("openrouter/openai/x", rate_limit_backoff_s=0.0, **kwargs)
    calls: list[dict[str, Any]] = []
    rate_limit_error = llm._litellm.RateLimitError

    async def fake(**request: Any) -> Any:
        calls.append(request)
        if len(calls) <= fail_times:
            raise rate_limit_error("429", llm_provider="openrouter", model="x")
        return _response()

    monkeypatch.setattr(llm._litellm, "acompletion", fake)
    llm.calls = calls  # type: ignore[attr-defined]
    return llm


async def test_parses_usage_and_cost(monkeypatch: pytest.MonkeyPatch) -> None:
    llm = backend(monkeypatch, fail_times=0)
    result = await llm.complete(MESSAGES)
    assert (result.text, result.input_tokens, result.output_tokens) == ("OK", 9, 2)
    assert result.cost_usd == 3.9e-06
    assert result.model == "openai/x"
    assert llm.calls[0]["usage"] == {"include": True}  # type: ignore[attr-defined]


async def test_retries_rate_limits_then_gives_up(monkeypatch: pytest.MonkeyPatch) -> None:
    ok = backend(monkeypatch, fail_times=2, rate_limit_retries=3)
    assert (await ok.complete(MESSAGES)).text == "OK"
    assert len(ok.calls) == 3  # type: ignore[attr-defined]
    failing = backend(monkeypatch, fail_times=10, rate_limit_retries=1)
    with pytest.raises(LLMError, match="rate-limited 2 times"):
        await failing.complete(MESSAGES)


async def test_max_rpm_spaces_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    llm = backend(monkeypatch, fail_times=0, max_rpm=1200)  # 50 ms apart
    started = time.monotonic()
    for _ in range(3):
        result = await llm.complete(MESSAGES)
        assert result.latency_s < 0.05  # the wait is not counted as latency
    assert time.monotonic() - started >= 0.09
