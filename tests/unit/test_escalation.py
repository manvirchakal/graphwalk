"""Escalation: low-confidence walks are redone with the fallback decider."""

import math

import pytest

from graphwalk import Index
from graphwalk.traversal import EscalatingTraverser, TraversalConfig, Traverser
from kg_fixtures import movie_store, oracle

BORN = "Where was the director of Inception born?"
SURE = {
    "Inception": {"Christopher Nolan": 0.98, "Leonardo DiCaprio": 0.02},
    "Christopher Nolan": {"London": 0.98, "STOP": 0.02},
    "London": {"STOP": 0.98, "United Kingdom": 0.02},
}
UNSURE = {
    "Inception": {"Christopher Nolan": 0.4, "Leonardo DiCaprio": 0.6},
    "Leonardo DiCaprio": {"STOP": 0.6, "Titanic": 0.4},
}
GREEDY = TraversalConfig(strategy="greedy")


async def _pair(primary_route: dict[str, dict[str, float]], threshold: float = 0.9):
    store = await movie_store()
    primary = oracle(primary_route, cost_per_call=0.001)
    fallback = oracle(SURE, cost_per_call=0.01)
    walker = EscalatingTraverser(
        Traverser(store, primary, config=GREEDY),
        Traverser(store, fallback, config=GREEDY),
        threshold=threshold,
    )
    return walker, primary, fallback


async def test_confident_walk_is_not_escalated() -> None:
    walker, _, fallback = await _pair(SURE)
    result = await walker.traverse(BORN, ("inception",))
    assert result.escalation is None
    assert result.best is not None
    assert result.best.names == ("London",)
    assert result.confidence == pytest.approx(math.exp(result.best.score))
    assert fallback.calls == 0


async def test_unsure_walk_is_escalated_and_costs_both() -> None:
    walker, primary, fallback = await _pair(UNSURE)
    result = await walker.traverse(BORN, ("inception",))
    assert result.best is not None
    assert result.best.names == ("London",)
    assert result.escalation is not None
    assert result.escalation.reason == "low_confidence"
    assert result.escalation.primary.best is not None
    assert result.escalation.primary.best.names == ("Leonardo DiCaprio",)
    assert primary.calls > 0
    assert fallback.calls > 0
    own, first = result.trace.totals.cost_usd, result.escalation.primary.trace.totals.cost_usd
    assert own is not None
    assert first is not None
    assert result.cost_usd == pytest.approx(own + first)
    # The result, escalation included, stays JSON-serializable.
    assert '"escalation"' in result.model_dump_json()


async def test_failed_walk_is_escalated() -> None:
    store = await movie_store()
    walker = EscalatingTraverser(
        Traverser(store, oracle(SURE, fail_on_calls=(1,)), config=GREEDY),
        Traverser(store, oracle(SURE), config=GREEDY),
        threshold=0.5,
    )
    result = await walker.traverse(BORN, ("inception",))
    assert result.status == "ok"
    assert result.escalation is not None
    assert result.escalation.reason == "not_ok"


@pytest.mark.parametrize("threshold", [0.0, 1.5])
async def test_threshold_is_validated(threshold: float) -> None:
    store = await movie_store()
    t = Traverser(store, oracle(SURE))
    with pytest.raises(ValueError, match="threshold"):
        EscalatingTraverser(t, t, threshold=threshold)


async def test_index_walk_escalates() -> None:
    primary, fallback = oracle(UNSURE), oracle(SURE)
    index = Index(
        await movie_store(),
        decider=primary,
        fallback_decider=fallback,
        escalate_below=0.9,
        traversal=GREEDY,
    )
    async with index:
        result = await index.walk(BORN)
    assert result.entries == ("inception",)
    assert result.escalated
    assert result.best is not None
    assert result.best.names == ("London",)
    assert result.confidence is not None
    assert result.confidence > 0.9


async def test_index_walk_without_escalation() -> None:
    fallback = oracle(SURE)
    async with Index(await movie_store(), decider=oracle(UNSURE), traversal=GREEDY) as index:
        result = await index.walk(BORN)
    assert not result.escalated
    assert result.best is not None
    assert result.best.names == ("Leonardo DiCaprio",)
    assert fallback.calls == 0


def test_index_validates_escalate_below() -> None:
    from graphwalk.stores import NetworkXStore

    with pytest.raises(ValueError, match="escalate_below"):
        Index(NetworkXStore(), escalate_below=0.0)


async def test_index_threshold_from_config() -> None:
    from graphwalk.config import resolve_config

    config = resolve_config({"escalate_below": 0.9})
    index = Index(
        await movie_store(),
        config=config,
        decider=oracle(UNSURE),
        fallback_decider=oracle(SURE),
        traversal=GREEDY,
    )
    async with index:
        result = await index.walk(BORN)
    assert result.escalated
    assert config.describe()["decision"]["escalate_below"] == 0.9  # pyright: ignore[reportIndexIssue]


async def test_explicit_backends_ignore_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GRAPHWALK_ESCALATE_BELOW", "0.9")
    fallback = oracle(SURE)
    index = Index(
        await movie_store(), decider=oracle(UNSURE), fallback_decider=fallback, traversal=GREEDY
    )
    async with index:
        result = await index.walk(BORN)
    assert not result.escalated
    assert fallback.calls == 0
