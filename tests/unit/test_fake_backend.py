import math

import pytest

from graphwalk.decisions import (
    ChoiceQuestion,
    DecisionBackend,
    DecisionBackendError,
    DecisionRequest,
    FakeDecisionBackend,
    JSONContent,
)


def q(key: str, *labels: str, instructions: str = "pick") -> ChoiceQuestion:
    return ChoiceQuestion(key=key, instructions=instructions, options=dict.fromkeys(labels))


def req(*questions: ChoiceQuestion, state: JSONContent = "s") -> DecisionRequest:
    return DecisionRequest(state=state, questions=questions)


def test_is_a_decision_backend() -> None:
    assert isinstance(FakeDecisionBackend(), DecisionBackend)


async def test_default_is_deterministic_and_order_independent() -> None:
    a, b = q("a", "x", "y", "z"), q("b", "x", "y")
    first = await FakeDecisionBackend(seed=7).decide(req(a, b))
    second = FakeDecisionBackend(seed=7)
    await second.decide(req(q("other", "p", "q")))  # an unrelated earlier call
    again = await second.decide(req(b, a))
    assert first.results["a"] == again.results["a"]
    assert first.results["b"] == again.results["b"]
    assert math.isclose(math.fsum(first.results["a"].probabilities.values()), 1.0)


async def test_seed_and_context_change_the_default() -> None:
    question = q("a", "x", "y", "z")
    base = (await FakeDecisionBackend(seed=1).decide(req(question))).results["a"]
    other_seed = (await FakeDecisionBackend(seed=2).decide(req(question))).results["a"]
    other_state = (await FakeDecisionBackend(seed=1).decide(req(question, state="t"))).results["a"]
    assert base != other_seed
    assert base != other_state


async def test_script_overrides_and_falls_back() -> None:
    def script(question: ChoiceQuestion, state: JSONContent) -> dict[str, float] | None:
        del state
        return {"x": 3, "y": 1} if question.instructions == "scripted" else None

    backend = FakeDecisionBackend(script=script)
    response = await backend.decide(
        req(q("s", "x", "y", instructions="scripted"), q("d", "x", "y"))
    )
    assert response.results["s"].probabilities == {"x": 0.75, "y": 0.25}
    assert (
        response.results["d"]
        == (await FakeDecisionBackend().decide(req(q("d", "x", "y")))).results["d"]
    )


async def test_records_requests_usage_and_metadata() -> None:
    backend = FakeDecisionBackend(cost_per_call=0.001, latency_s=0.25)
    request = req(q("a", "x", "y"), q("b", "x", "y"))
    response = await backend.decide(request)
    assert backend.requests == [request]
    assert backend.calls == 1
    assert response.usage.output_tokens == 2
    assert response.usage.input_tokens > 0
    assert response.usage.cost_usd == 0.001
    assert response.latency_s == 0.25
    assert response.model == backend.model_id == "fake-decider-1"
    # usage is a pure function of the request
    assert (await FakeDecisionBackend().decide(request)).usage.input_tokens == (
        response.usage.input_tokens
    )


async def test_scripted_failures() -> None:
    backend = FakeDecisionBackend(fail_on_calls={2})
    await backend.decide(req(q("a", "x", "y")))
    with pytest.raises(DecisionBackendError, match="call 2"):
        await backend.decide(req(q("a", "x", "y")))
    await backend.decide(req(q("a", "x", "y")))
    assert backend.calls == 3


async def test_enforces_max_options() -> None:
    backend = FakeDecisionBackend(max_options=2)
    with pytest.raises(DecisionBackendError, match="max_options"):
        await backend.decide(req(q("a", "x", "y", "z")))


async def test_aclose() -> None:
    backend = FakeDecisionBackend()
    await backend.aclose()
    assert backend.closed
