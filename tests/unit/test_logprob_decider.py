import json
import math
from typing import Any

import httpx2
import pytest

from graphwalk.decisions.base import (
    ChoiceQuestion,
    DecisionBackend,
    DecisionBackendError,
    DecisionRequest,
    JSONContent,
)
from graphwalk.decisions.logprob import (
    FLOOR,
    LogprobDecider,
    first_token_alternatives,
    letter_distribution,
    letter_mass,
    prompt,
)


def question(n: int = 3, key: str = "move") -> ChoiceQuestion:
    options: dict[str, JSONContent | None] = {f"rel_{i}": None for i in range(n - 1)}
    options["STOP"] = "the current nodes answer the question"
    return ChoiceQuestion(key=key, instructions="Pick the next relation.", options=options)


def reply(top: list[tuple[str, float]], **extra: Any) -> dict[str, Any]:
    alternatives = [{"token": t, "logprob": math.log(p)} for t, p in top]
    return {
        "model": "served-model",
        "choices": [{"logprobs": {"content": [{"top_logprobs": alternatives}]}}],
        "usage": {"prompt_tokens": 50, "completion_tokens": 1, **extra},
    }


def decider(
    responses: list[httpx2.Response], *, base_url: str = "http://localhost:8000/v1"
) -> tuple[LogprobDecider, list[dict[str, Any]]]:
    bodies: list[dict[str, Any]] = []
    queue = list(responses)

    def handle(request: httpx2.Request) -> httpx2.Response:
        bodies.append(json.loads(request.content))
        return queue.pop(0)

    backend = LogprobDecider(
        "some/model", base_url=base_url, api_key="sk-test-123456789", attempts=3,
        transport=httpx2.MockTransport(handle),
    )  # fmt: skip
    return backend, bodies


def test_prompt_letters_every_option() -> None:
    text = prompt({"question": "q"}, question())
    assert "A. rel_0" in text
    assert "C. STOP: " in text


def test_letter_variants_are_summed_and_missing_letters_floored() -> None:
    top = [
        {"token": "B", "logprob": math.log(0.6)},
        {"token": " b", "logprob": math.log(0.2)},
        {"token": "A", "logprob": math.log(0.2)},
        {"token": "<eos>", "logprob": math.log(0.01)},
    ]
    result = letter_distribution(question(), top)
    assert result.top == "rel_1"
    assert result.probabilities["rel_1"] == pytest.approx(0.8, abs=1e-4)
    assert result.probabilities["STOP"] == pytest.approx(FLOOR, rel=0.01)


def test_no_letter_gives_uniform() -> None:
    result = letter_distribution(question(), [{"token": "Hello", "logprob": 0.0}])
    assert list(result.probabilities.values()) == pytest.approx([1 / 3] * 3)


def test_first_token_alternatives() -> None:
    data = {"choices": [{"logprobs": {"content": [{"top_logprobs": [{"token": "A"}]}]}}]}
    assert first_token_alternatives(data) == [{"token": "A"}]
    assert first_token_alternatives({"choices": [{"logprobs": None}]}) is None
    assert first_token_alternatives({}) is None


async def test_decides_each_question_and_reports_usage() -> None:
    backend, bodies = decider(
        [
            httpx2.Response(200, json=reply([("A", 0.9), ("B", 0.1)], cost=0.001)),
            httpx2.Response(200, json=reply([("B", 0.7), ("A", 0.3)], cost=0.002)),
        ]
    )
    assert isinstance(backend, DecisionBackend)
    request = DecisionRequest(state="s", questions=(question(key="q1"), question(key="q2")))
    response = await backend.decide(request)
    assert response.results["q1"].top == "rel_0"
    assert response.results["q2"].top == "rel_1"
    assert response.usage.input_tokens == 100
    assert response.usage.cost_usd == pytest.approx(0.003)
    assert response.model == "logprob:served-model"
    assert all(b["max_tokens"] == 1 and b["logprobs"] is True for b in bodies)
    assert "provider" not in bodies[0]  # OpenRouter-only switches stay off elsewhere
    await backend.aclose()


async def test_cost_is_unknown_when_the_endpoint_reports_none() -> None:
    backend, _ = decider([httpx2.Response(200, json=reply([("A", 1.0)]))])
    response = await backend.decide(DecisionRequest(state="s", questions=(question(),)))
    assert response.usage.cost_usd is None


async def test_openrouter_requests_ask_for_providers_with_logprobs() -> None:
    backend, bodies = decider(
        [httpx2.Response(200, json=reply([("A", 1.0)]))], base_url="https://openrouter.ai/api/v1"
    )
    await backend.decide(DecisionRequest(state="s", questions=(question(),)))
    assert bodies[0]["provider"] == {"require_parameters": True}
    assert bodies[0]["reasoning"] == {"enabled": False}


async def test_missing_logprobs_fail_at_once() -> None:
    no_logprobs = {"choices": [{"message": {"content": "A"}, "logprobs": None}]}
    backend, bodies = decider([httpx2.Response(200, json=no_logprobs)])
    with pytest.raises(DecisionBackendError, match="no token log-probabilities"):
        await backend.decide(DecisionRequest(state="s", questions=(question(),)))
    assert len(bodies) == 1


async def test_openrouter_retries_a_provider_that_dropped_logprobs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr("graphwalk.decisions.logprob.asyncio.sleep", no_sleep)
    no_logprobs = {"provider": "SomeHost", "choices": [{"logprobs": None}]}
    backend, bodies = decider(
        [httpx2.Response(200, json=no_logprobs), httpx2.Response(200, json=reply([("A", 1.0)]))],
        base_url="https://openrouter.ai/api/v1",
    )
    response = await backend.decide(DecisionRequest(state="s", questions=(question(),)))
    assert response.results["move"].top == "rel_0"
    assert len(bodies) == 2
    assert "ignore" not in bodies[0]["provider"]
    assert bodies[1]["provider"] == {"require_parameters": True, "ignore": ["SomeHost"]}


async def test_a_provider_is_skipped_once_then_the_error_is_final(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr("graphwalk.decisions.logprob.asyncio.sleep", no_sleep)
    no_logprobs = {"provider": "SomeHost", "choices": [{"logprobs": None}]}
    backend, bodies = decider(
        [httpx2.Response(200, json=no_logprobs)] * 2, base_url="https://openrouter.ai/api/v1"
    )
    with pytest.raises(DecisionBackendError, match="SomeHost returned no token"):
        await backend.decide(DecisionRequest(state="s", questions=(question(),)))
    assert len(bodies) == 2


async def test_rate_limits_are_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    async def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr("graphwalk.decisions.logprob.asyncio.sleep", no_sleep)
    backend, bodies = decider(
        [httpx2.Response(429, text="slow down"), httpx2.Response(200, json=reply([("A", 1.0)]))]
    )
    response = await backend.decide(DecisionRequest(state="s", questions=(question(),)))
    assert response.results["move"].top == "rel_0"
    assert len(bodies) == 2


async def test_client_errors_are_not_retried_and_keys_are_redacted() -> None:
    backend, bodies = decider([httpx2.Response(401, text="bad key sk-test-123456789")])
    with pytest.raises(DecisionBackendError) as error:
        await backend.decide(DecisionRequest(state="s", questions=(question(),)))
    assert "sk-test-123456789" not in str(error.value)
    assert len(bodies) == 1


async def test_too_many_options_are_refused() -> None:
    backend, _ = decider([])
    with pytest.raises(DecisionBackendError, match="at most 20"):
        await backend.decide(DecisionRequest(state="s", questions=(question(n=21),)))


def test_letter_mass_counts_offered_letters_only() -> None:
    top = [
        {"token": "The", "logprob": math.log(0.5)},
        {"token": "A", "logprob": math.log(0.3)},
        {"token": "Z", "logprob": math.log(0.1)},  # not offered
    ]
    mass, top_is_letter = letter_mass(question(), top)
    assert mass == pytest.approx(0.3)
    assert not top_is_letter


async def test_diagnostics_track_the_format() -> None:
    backend, _ = decider(
        [
            httpx2.Response(200, json=reply([("A", 0.9), ("B", 0.05)])),
            httpx2.Response(200, json=reply([("Sure", 0.6), ("B", 0.4)])),
        ]
    )
    request = DecisionRequest(state="s", questions=(question(key="q1"), question(key="q2")))
    await backend.decide(request)
    stats = backend.diagnostics()
    assert stats["asked"] == 2
    assert stats["mean_letter_mass"] == pytest.approx((0.95 + 0.4) / 2)
    assert stats["top_not_letter"] == 1
