"""JevBackend through the real typesafe-sdk, with HTTP answered by an in-process mock."""

import json
from collections.abc import Callable
from typing import Any

import httpx2
import pytest

from graphwalk.config import GraphwalkSettings
from graphwalk.decisions import ChoiceQuestion, DecisionBackendError, DecisionRequest
from graphwalk.decisions.jev import JevBackend

Handler = Callable[[httpx2.Request], httpx2.Response]

REQUEST = DecisionRequest(
    state={"query": "Who directed Inception?"},
    questions=(
        ChoiceQuestion(
            key="b0",
            instructions={"task": "next hop", "current": "Inception"},
            options={"o1": "directed_by -> Christopher Nolan", "o2": None, "STOP": "stop here"},
        ),
        ChoiceQuestion(key="b1", instructions="other beam", options={"o1": None, "STOP": None}),
    ),
)


def answer(probabilities: dict[str, float]) -> dict[str, Any]:
    top = max(probabilities, key=lambda k: probabilities[k])
    return {
        "type": "choice",
        "choice": top,
        "confidence": probabilities[top],
        "probabilities": probabilities,
    }


def ok_body(**extra: Any) -> dict[str, Any]:
    return {
        "model": "jev-1.13.0",
        "answers": {
            "b0": answer({"o1": 0.8, "o2": 0.1, "STOP": 0.1}),
            "b1": answer({"o1": 0.3, "STOP": 0.7}),
        },
        "usage": {"input_tokens": 120, "output_tokens": 2},
        **extra,
    }


class Recorder:
    def __init__(self, respond: Handler) -> None:
        self.respond = respond
        self.requests: list[httpx2.Request] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        return self.respond(request)

    def body(self, i: int = 0) -> dict[str, Any]:
        loaded: dict[str, Any] = json.loads(self.requests[i].content)
        return loaded


def backend(recorder: Recorder, **kwargs: Any) -> JevBackend:
    params: dict[str, Any] = {
        "api_key": "test-key",
        "provider": "typesafe",
        "model": "jev-1.13.0",
        "max_retries": 0,
    }
    params.update(kwargs)
    return JevBackend(**params, transport=httpx2.MockTransport(recorder))


def respond_json(
    body: dict[str, Any], status: int = 200, headers: dict[str, str] | None = None
) -> Handler:
    return lambda _request: httpx2.Response(status, json=body, headers=headers or {})


async def test_sends_pinned_model_and_one_question_per_key() -> None:
    rec = Recorder(respond_json(ok_body(), headers={"x-typesafe-request-id": "req-1"}))
    response = await backend(rec).decide(REQUEST)

    sent = rec.requests[0]
    assert sent.method == "POST"
    assert str(sent.url) == "https://api.typesafe.ai/v1/systemone"
    assert sent.headers["authorization"] == "Bearer test-key"
    body = rec.body()
    assert body["model"] == "jev-1.13.0"
    assert body["state"] == {"query": "Who directed Inception?"}
    assert body["questions"]["b0"] == {
        "type": "choice",
        "instructions": {"task": "next hop", "current": "Inception"},
        "criteria": {"o1": "directed_by -> Christopher Nolan", "o2": None, "STOP": "stop here"},
    }
    assert body["questions"]["b1"]["criteria"] == {"o1": None, "STOP": None}

    assert response.results["b0"].probabilities == pytest.approx(
        {"o1": 0.8, "o2": 0.1, "STOP": 0.1}
    )
    assert response.results["b1"].top == "STOP"
    assert response.usage.input_tokens == 120
    assert response.usage.cost_usd is None  # TypeSafe direct reports no cost
    assert response.request_id == "req-1"
    assert response.model == "jev-1.13.0"
    assert response.provider == "typesafe"
    assert response.latency_s >= 0


async def test_openrouter_url_and_extras() -> None:
    body = ok_body(id="gen-123", provider="TypeSafe")
    body["model"] = "typesafe/jev-1.13"
    body["usage"]["cost"] = 0.00000504
    rec = Recorder(respond_json(body))
    jev = backend(rec, provider="openrouter", model="typesafe/jev-1.13")
    response = await jev.decide(REQUEST)
    assert str(rec.requests[0].url) == "https://openrouter.ai/api/v1/systemone"
    assert rec.body()["model"] == "typesafe/jev-1.13"
    assert response.usage.cost_usd == 0.00000504
    assert response.request_id == "gen-123"
    assert response.provider == "TypeSafe"


async def test_renormalizes_server_probabilities() -> None:
    body = ok_body()
    body["answers"]["b1"] = answer({"o1": 0.3, "STOP": 0.6})  # sums to 0.9
    response = await backend(Recorder(respond_json(body))).decide(REQUEST)
    assert response.results["b1"].probabilities == pytest.approx({"o1": 1 / 3, "STOP": 2 / 3})


async def test_missing_answer_is_an_error() -> None:
    body = ok_body()
    del body["answers"]["b1"]
    with pytest.raises(DecisionBackendError, match="no choice answer for question 'b1'"):
        await backend(Recorder(respond_json(body))).decide(REQUEST)


async def test_unoffered_label_is_an_error() -> None:
    body = ok_body()
    body["answers"]["b1"] = answer({"o1": 0.3, "STOP": 0.6, "o9": 0.1})
    with pytest.raises(DecisionBackendError, match="not offered"):
        await backend(Recorder(respond_json(body))).decide(REQUEST)


@pytest.mark.parametrize("status", [401, 422, 500])
async def test_http_errors_become_backend_errors(status: int) -> None:
    rec = Recorder(respond_json({"detail": "nope"}, status=status))
    with pytest.raises(DecisionBackendError, match="Jev request failed"):
        await backend(rec).decide(REQUEST)
    assert len(rec.requests) == 1  # max_retries=0


async def test_retries_transient_errors() -> None:
    responses = iter([httpx2.Response(503, json={}), httpx2.Response(200, json=ok_body())])
    rec = Recorder(lambda _r: next(responses))
    response = await backend(rec, max_retries=1).decide(REQUEST)
    assert len(rec.requests) == 2
    assert response.results["b0"].top == "o1"


async def test_rejects_too_many_options_before_sending() -> None:
    rec = Recorder(respond_json(ok_body()))
    big = ChoiceQuestion(key="q", instructions="x", options=dict.fromkeys(map(str, range(256))))
    with pytest.raises(DecisionBackendError, match="at most 255"):
        await backend(rec).decide(DecisionRequest(state="s", questions=(big,)))
    assert rec.requests == []


def test_refuses_latest_alias() -> None:
    with pytest.raises(ValueError, match="moving model alias"):
        JevBackend(api_key="k", provider="typesafe", model="jev-latest")


def test_from_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("GRAPHWALK_DECISION_PROVIDER", "openrouter")
    with pytest.raises(DecisionBackendError, match="set OPENROUTER_API_KEY"):
        JevBackend.from_settings(GraphwalkSettings(_env_file=None))  # pyright: ignore[reportCallIssue]
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    jev = JevBackend.from_settings(GraphwalkSettings(_env_file=None))  # pyright: ignore[reportCallIssue]
    assert jev.model_id == "typesafe/jev-1.13"
    assert jev.provider == "openrouter"


async def test_verify_model_typesafe() -> None:
    listing = {"models": [{"name": "jev-1.13.0", "description": "d", "release_date": "2026-09-15"}]}
    rec = Recorder(respond_json(listing))
    await backend(rec).verify_model()
    assert str(rec.requests[0].url) == "https://api.typesafe.ai/v1/models"
    with pytest.raises(DecisionBackendError, match="not offered"):
        await backend(rec, model="jev-1.14.0").verify_model()


def openrouter_endpoints(model: str, n_endpoints: int = 1) -> dict[str, Any]:
    # Shape captured from the live OpenRouter API (GET /api/v1/models/{id}/endpoints).
    return {
        "data": {
            "id": model,
            "name": "TypeSafe: Jev 1.13",
            "architecture": {"modality": "text->decisions"},
            "endpoints": [{"name": f"TypeSafe | {model}-20260917", "model_id": model, "status": 0}]
            * n_endpoints,
        }
    }


async def test_verify_model_openrouter() -> None:
    rec = Recorder(respond_json(openrouter_endpoints("typesafe/jev-1.13")))
    jev = backend(rec, provider="openrouter", model="typesafe/jev-1.13")
    await jev.verify_model()
    assert (
        str(rec.requests[0].url)
        == "https://openrouter.ai/api/v1/models/typesafe/jev-1.13/endpoints"
    )
    assert rec.requests[0].headers["authorization"] == "Bearer test-key"


async def test_verify_model_openrouter_unknown_model() -> None:
    rec = Recorder(respond_json({"error": {"message": "not found"}}, status=404))
    with pytest.raises(DecisionBackendError, match="not offered"):
        await backend(rec, provider="openrouter", model="typesafe/jev-9").verify_model()


async def test_verify_model_openrouter_without_endpoints() -> None:
    rec = Recorder(respond_json(openrouter_endpoints("typesafe/jev-1.13", n_endpoints=0)))
    with pytest.raises(DecisionBackendError, match="not offered"):
        await backend(rec, provider="openrouter", model="typesafe/jev-1.13").verify_model()


async def test_verify_model_listing_failure() -> None:
    rec = Recorder(respond_json({"unexpected": True}))
    with pytest.raises(DecisionBackendError, match="could not check model availability"):
        await backend(rec, provider="openrouter", model="typesafe/jev-1.13").verify_model()


async def test_verify_model_non_object_listing() -> None:
    rec = Recorder(lambda _r: httpx2.Response(200, json=["not", "an", "object"]))
    with pytest.raises(DecisionBackendError, match="could not check model availability"):
        await backend(rec, provider="openrouter", model="typesafe/jev-1.13").verify_model()


async def test_from_settings_honors_base_url_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GRAPHWALK_DECISION_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://gateway.internal/openrouter/")
    body = ok_body()
    body["model"] = "typesafe/jev-1.13"
    rec = Recorder(respond_json(body))
    jev = JevBackend.from_settings(
        GraphwalkSettings(_env_file=None),  # pyright: ignore[reportCallIssue]
        transport=httpx2.MockTransport(rec),
    )
    await jev.decide(REQUEST)
    assert str(rec.requests[0].url) == "https://gateway.internal/openrouter/v1/systemone"
