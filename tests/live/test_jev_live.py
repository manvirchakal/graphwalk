"""Live smoke test against the real Jev API.

Run with ``uv run pytest --run-live tests/live``. Needs network access to the provider
and ``OPENROUTER_API_KEY`` or ``TYPESAFE_API_KEY`` (select with
``GRAPHWALK_DECISION_PROVIDER``). Costs a fraction of a cent.
"""

import math

import pytest

from graphwalk.config import GraphwalkSettings
from graphwalk.decisions import ChoiceQuestion, DecisionRequest
from graphwalk.decisions.jev import JevBackend

pytestmark = pytest.mark.live


async def test_jev_smoke() -> None:
    settings = GraphwalkSettings()
    if settings.decision_api_key is None:
        pytest.skip(f"no API key for {settings.decision_provider}")
    jev = JevBackend.from_settings(settings)
    try:
        await jev.verify_model()
        request = DecisionRequest(
            state={"query": "Who directed the film Inception?"},
            questions=(
                ChoiceQuestion(
                    key="b0",
                    instructions={
                        "task": "Pick the edge to follow from the current node to answer the "
                        "query, or STOP if the current node is the answer.",
                        "current": {"name": "Inception", "type": "film"},
                    },
                    options={
                        "o1": {"relation": "directed_by", "node": "Christopher Nolan"},
                        "o2": {"relation": "starred_actors", "node": "Leonardo DiCaprio"},
                        "o3": {"relation": "release_year", "node": "2010"},
                        "STOP": "The current node itself answers the query.",
                    },
                ),
                ChoiceQuestion(
                    key="b1",
                    instructions={
                        "task": "Pick the edge to follow, or STOP if the current node answers.",
                        "current": {"name": "Christopher Nolan", "type": "person"},
                        "path": ["Inception --directed_by--> Christopher Nolan"],
                    },
                    options={
                        "o1": {"relation": "directed", "node": "Memento"},
                        "STOP": "The current node itself answers the query.",
                    },
                ),
            ),
        )
        response = await jev.decide(request)
    finally:
        await jev.aclose()

    print(response.model_dump_json(indent=2))  # noqa: T201 - smoke-test report
    assert set(response.results) == {"b0", "b1"}  # batching works
    for question in request.questions:
        result = response.results[question.key]
        assert list(result.probabilities) == list(question.options)
        assert math.isclose(math.fsum(result.probabilities.values()), 1.0)
    assert response.usage.input_tokens > 0
    # Sanity, not a benchmark: the obvious answers should win.
    assert response.results["b0"].top == "o1"
    assert response.results["b1"].top == "STOP"
