"""Live traversal smoke test: real Jev decisions over the small movie graph.

Run with ``uv run pytest --run-live tests/live -s``. A sanity check that the prompts
work end to end, not a benchmark. Costs well under a cent.
"""

import pytest

from graphwalk.config import GraphwalkSettings
from graphwalk.decisions.jev import JevBackend
from graphwalk.traversal import Budget, TraversalConfig, TraversalResult, Traverser
from kg_fixtures import movie_store

pytestmark = pytest.mark.live

CASES = [
    ("Where was the director of Inception born?", "entity", {"london"}),
    (
        "Which films were directed by the director of Inception?",
        "relation",
        {"memento", "interstellar"},
    ),
    ("Who starred in Titanic?", "entity", {"dicaprio"}),
]


def _report(label: str, result: TraversalResult) -> None:
    t = result.trace.totals
    print(  # noqa: T201 - smoke-test report
        f"\n[{label}] status={result.status} calls={t.decision_calls} "
        f"questions={t.questions} in_tokens={t.input_tokens} cost={t.cost_usd} "
        f"decision_latency={t.decision_latency_s:.2f}s wall={t.wall_s:.2f}s"
    )
    for answer in result.answers[:3]:
        print(  # noqa: T201
            f"   {answer.names} score={answer.score:.3f} min_conf={answer.min_confidence} "
            f"[{answer.terminated_by}] path={[h.relation for h in answer.path]}"
        )
    for step in result.trace.steps:
        if step.call_id is not None:
            top = sorted(step.distribution.items(), key=lambda kv: -kv[1])[:3]
            print(f"     d{step.depth} b{step.beam_id} conf={step.confidence} {top}")  # noqa: T201


@pytest.mark.parametrize("strategy", ["greedy", "beam"])
@pytest.mark.parametrize(("query", "mode", "expected"), CASES)
async def test_live_traversal(query: str, mode: str, expected: set[str], strategy: str) -> None:
    settings = GraphwalkSettings()
    if settings.decision_api_key is None:
        pytest.skip(f"no API key for {settings.decision_provider}")
    jev = JevBackend.from_settings(settings)
    config = TraversalConfig.model_validate(
        {
            "strategy": strategy,
            "beam_width": 3,
            "hop_mode": mode,
            "allow_stop_at_start": False,
            "budget": Budget(max_depth=4, max_decision_calls=8),
        }
    )
    try:
        await jev.verify_model()
        store = await movie_store()
        start = "titanic" if "Titanic" in query else "inception"
        result = await Traverser(store, jev, config=config).traverse(query, [start])
    finally:
        await jev.aclose()
    _report(f"{strategy}/{mode}: {query}", result)
    assert result.status == "ok"
    assert result.best is not None
    assert set(result.best.node_ids) == expected
