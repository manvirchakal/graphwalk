"""Traversal engine behavior against the fake decision backend."""

import math

import pytest

from factories import edge, node
from graphwalk.core.errors import NodeNotFoundError
from graphwalk.decisions import FakeDecisionBackend
from graphwalk.embeddings import FakeEmbedder
from graphwalk.stores.networkx_store import NetworkXStore
from graphwalk.traversal import Budget, TraversalConfig, TraversalResult, Traverser
from graphwalk.traversal.prompts import node_text
from kg_fixtures import movie_store, oracle

BORN = "Where was the director of Inception born?"
# Inception -> Nolan -> London -> STOP, each hop clear.
BORN_ROUTE = {
    "Inception": {"Christopher Nolan": 0.9, "Leonardo DiCaprio": 0.1},
    "Christopher Nolan": {"London": 0.8, "STOP": 0.1, "Memento": 0.05, "Interstellar": 0.05},
    "London": {"STOP": 0.95, "United Kingdom": 0.05},
}


def cfg(**kwargs: object) -> TraversalConfig:
    return TraversalConfig.model_validate(kwargs)


async def walk(
    route: dict[str, dict[str, float]],
    query: str = BORN,
    start: tuple[str, ...] = ("inception",),
    **config: object,
) -> tuple[TraversalResult, FakeDecisionBackend]:
    backend = oracle(route)
    traverser = Traverser(await movie_store(), backend, config=cfg(**config))
    return await traverser.traverse(query, start), backend


# -- greedy ----------------------------------------------------------------------------


async def test_greedy_follows_the_best_edges_and_stops() -> None:
    result, backend = await walk(BORN_ROUTE, strategy="greedy")
    assert result.status == "ok"
    best = result.best
    assert best is not None
    assert best.node_ids == ("london",)
    assert best.names == ("London",)
    assert [(h.relation, h.direction, h.targets) for h in best.path] == [
        ("directed_by", "out", ("nolan",)),
        ("born_in", "out", ("london",)),
    ]
    assert best.terminated_by == "stop"
    assert best.n_decisions == 3
    assert math.isclose(best.sum_logp, math.log(0.9) + math.log(0.8) + math.log(0.95))
    assert backend.calls == 3
    assert result.trace.totals.decision_calls == 3
    assert [s.chosen for s in result.trace.steps][-1] == ("STOP",)


async def test_question_shows_query_in_state_and_local_context() -> None:
    _, backend = await walk(BORN_ROUTE, strategy="greedy")
    second = backend.requests[1]
    assert second.state == {"query": BORN}
    question = second.questions[0]
    assert isinstance(question.instructions, dict)
    assert question.instructions["current"] == {"name": "Christopher Nolan", "type": "person"}
    assert question.instructions["path_so_far"] == ["Inception --directed_by--> Christopher Nolan"]
    assert "STOP" in question.options
    # Inception was visited, so the back edge is not offered; the other films are,
    # reached by following directed_by backwards.
    edges = [d["edge"] for d in question.options.values() if isinstance(d, dict)]
    assert "Memento --directed_by--> Christopher Nolan" in edges
    assert all("Inception" not in str(e) for e in edges)


async def test_revisiting_is_allowed_when_configured() -> None:
    _, backend = await walk(BORN_ROUTE, strategy="greedy", exclude_visited=False)
    edges = [str(d) for d in backend.requests[1].questions[0].options.values()]
    assert any("Inception" in e for e in edges)


async def test_direction_out_only() -> None:
    _, backend = await walk(BORN_ROUTE, strategy="greedy", direction="out")
    options = backend.requests[1].questions[0].options
    assert len(options) == 2  # born_in London + STOP


# -- beam ------------------------------------------------------------------------------

# A trap: B looks slightly better at the first hop but nothing good follows it.
TRAP_NODES = [node(n, name=n) for n in ("A", "B", "C", "D", "E")]
TRAP_EDGES = [
    edge("A", "r", "B"),
    edge("A", "r", "C"),
    edge("B", "r", "D"),
    edge("B", "r", "E"),
    edge("C", "r", "D"),
]
TRAP_ROUTE = {
    "A": {"B": 0.55, "C": 0.45},
    "B": {"D": 0.3, "E": 0.3, "STOP": 0.4},
    "C": {"STOP": 0.99, "D": 0.01},
    "D": {"STOP": 1.0},
    "E": {"STOP": 1.0},
}


async def trap_store() -> NetworkXStore:
    store = NetworkXStore()
    for n in TRAP_NODES:
        await store.upsert_node(n)
    for e in TRAP_EDGES:
        await store.upsert_edge(e)
    return store


async def run_trap(**config: object) -> TraversalResult:
    traverser = Traverser(await trap_store(), oracle(TRAP_ROUTE), config=cfg(**config))
    return await traverser.traverse("q", ["A"])


async def test_beam_recovers_from_a_greedy_mistake() -> None:
    greedy = await run_trap(strategy="greedy", allow_stop_at_start=False, direction="out")
    beam = await run_trap(strategy="beam", beam_width=3, allow_stop_at_start=False, direction="out")
    assert greedy.best is not None
    assert greedy.best.node_ids == ("B",)
    assert beam.best is not None
    assert beam.best.node_ids == ("C",)
    expected = (math.log(0.45) + math.log(0.99)) / 2
    assert math.isclose(beam.best.score, expected)
    assert beam.best.terminated_by == "stop"
    assert all(a.terminated_by in ("stop", "dead_end") for a in beam.answers)
    scores = [a.score for a in beam.answers]
    assert scores == sorted(scores, reverse=True)


async def test_beam_batches_one_call_per_depth() -> None:
    result, backend = await walk(BORN_ROUTE, strategy="beam", beam_width=3)
    per_call = [len(r.questions) for r in backend.requests]
    assert len(per_call) == result.trace.totals.depth_reached
    assert max(per_call) > 1
    assert result.best is not None
    assert result.best.node_ids == ("london",)


async def test_per_beam_batching_sends_one_question_per_call() -> None:
    result, backend = await walk(
        BORN_ROUTE, strategy="beam", beam_width=3, beam_batching="per_beam"
    )
    assert all(len(r.questions) == 1 for r in backend.requests)
    assert result.trace.totals.questions == backend.calls
    assert result.best is not None
    assert result.best.node_ids == ("london",)


async def test_token_budget_splits_a_depth_into_several_calls() -> None:
    result, backend = await walk(
        BORN_ROUTE,
        strategy="beam",
        beam_width=3,
        max_request_tokens=400,
        max_state_plus_question_tokens=400,
        token_safety_margin=0.0,
    )
    depths = [c.depth for c in result.trace.calls]
    assert len(depths) > len(set(depths))  # some depth needed more than one call
    assert backend.calls == len(depths)


async def test_beam_is_deterministic() -> None:
    first, _ = await walk({}, strategy="beam", beam_width=3)  # seeded Dirichlet answers
    second, _ = await walk({}, strategy="beam", beam_width=3)
    assert first.model_dump(exclude={"trace": {"totals": {"wall_s"}}}) == second.model_dump(
        exclude={"trace": {"totals": {"wall_s"}}}
    )


async def test_zero_probability_is_floored_not_minus_infinity() -> None:
    route = {"Inception": {"Christopher Nolan": 1.0}, "Christopher Nolan": {"STOP": 1.0}}
    result, _ = await walk(route, strategy="beam", beam_width=5, prob_floor=1e-3)
    assert all(math.isfinite(a.score) for a in result.answers)
    assert result.best is not None
    assert result.best.node_ids == ("nolan",)
    assert result.best.score == 0.0


# -- stopping, dead ends, forced moves, limits ----------------------------------------


async def test_dead_end_ends_without_a_call() -> None:
    store = NetworkXStore()
    await store.upsert_node(node("A"))
    await store.upsert_node(node("B"))
    await store.upsert_edge(edge("A", "r", "B"))
    backend = FakeDecisionBackend()
    config = cfg(strategy="greedy", allow_stop_at_start=False, direction="out")
    result = await Traverser(store, backend, config=config).traverse("q", ["A"])
    # Depth 0: A has one move and STOP is not offered -> forced move, no call.
    # Depth 1: B has no moves, only STOP -> dead end, no call.
    assert backend.calls == 0
    assert result.best is not None
    assert result.best.node_ids == ("B",)
    assert result.best.terminated_by == "dead_end"
    assert result.best.n_decisions == 1  # the forced STOP counts as a certain decision
    assert result.best.score == 0.0
    assert result.best.min_confidence is None
    assert result.trace.totals.cost_usd == 0.0  # no calls made: nothing to pay
    assert [s.call_id for s in result.trace.steps] == [None, None]


async def test_no_stop_at_start_when_disabled() -> None:
    _, backend = await walk(BORN_ROUTE, strategy="greedy", allow_stop_at_start=False)
    assert "STOP" not in backend.requests[0].questions[0].options
    assert "STOP" in backend.requests[1].questions[0].options


async def test_max_depth_ends_open_walks() -> None:
    route = {"Inception": {"Christopher Nolan": 1.0}, "Christopher Nolan": {"London": 1.0}}
    result, backend = await walk(route, strategy="greedy", budget=Budget(max_depth=2))
    assert result.status == "ok"
    assert backend.calls == 2
    assert result.best is not None
    assert result.best.node_ids == ("london",)
    assert result.best.terminated_by == "max_depth"


async def test_stopped_answers_rank_above_open_ones() -> None:
    route = {
        "Inception": {"Christopher Nolan": 0.5, "STOP": 0.5},
        "Christopher Nolan": {"London": 1.0},
    }
    result, _ = await walk(route, strategy="beam", beam_width=2, budget=Budget(max_depth=2))
    assert next(a.terminated_by for a in result.answers) == "stop"


async def test_call_budget_aborts_with_partial_answers() -> None:
    budget = Budget(max_depth=5, max_decision_calls=2)
    result, backend = await walk(BORN_ROUTE, strategy="greedy", budget=budget)
    assert result.status == "aborted"
    assert result.abort_reason is not None
    assert "max_decision_calls" in result.abort_reason
    assert backend.calls == 2
    assert result.best is not None
    assert result.best.node_ids == ("london",)
    assert result.best.terminated_by == "aborted"
    assert len(result.trace.calls) == 2


async def test_token_budget_aborts_after_the_call_that_crosses_it() -> None:
    budget = Budget(max_depth=5, max_input_tokens=1)
    result, backend = await walk(BORN_ROUTE, strategy="greedy", budget=budget)
    assert result.status == "aborted"
    assert result.abort_reason is not None
    assert "max_input_tokens" in result.abort_reason
    assert backend.calls == 1


async def test_backend_error_returns_error_status() -> None:
    backend = oracle(BORN_ROUTE, fail_on_calls=[2])
    traverser = Traverser(await movie_store(), backend, config=cfg(strategy="greedy"))
    result = await traverser.traverse(BORN, ["inception"])
    assert result.status == "error"
    assert result.error is not None
    assert "fake failure" in result.error
    assert result.answers  # the partial hypothesis is still reported


async def test_bad_start_nodes() -> None:
    traverser = Traverser(await movie_store(), FakeDecisionBackend())
    with pytest.raises(NodeNotFoundError):
        await traverser.traverse("q", ["nope"])
    with pytest.raises(ValueError, match="start node"):
        await traverser.traverse("q", [])


# -- sampling --------------------------------------------------------------------------


async def test_sample_is_seeded_batched_and_voted() -> None:
    route = {
        "Inception": {"Christopher Nolan": 0.6, "Leonardo DiCaprio": 0.4},
        "Christopher Nolan": {"STOP": 1.0},
        "Leonardo DiCaprio": {"STOP": 1.0},
    }
    config = {"strategy": "sample", "n_samples": 20, "seed": 1, "allow_stop_at_start": False}
    first, backend = await walk(route, **config)
    again, _ = await walk(route, **config)
    assert [a.node_ids for a in first.answers] == [a.node_ids for a in again.answers]
    assert [a.votes for a in first.answers] == [a.votes for a in again.answers]
    assert sum(a.votes for a in first.answers) == 20
    assert {a.node_ids for a in first.answers} == {("nolan",), ("dicaprio",)}
    votes = [a.votes for a in first.answers]
    assert votes == sorted(votes, reverse=True)
    assert backend.calls == 2  # 20 walks, one call per depth
    assert len(backend.requests[0].questions) == 20


async def test_sample_at_zero_temperature_matches_greedy() -> None:
    greedy, _ = await walk(BORN_ROUTE, strategy="greedy")
    sampled, _ = await walk(BORN_ROUTE, strategy="sample", n_samples=3, temperature=0.0)
    assert greedy.best is not None
    assert sampled.best is not None
    assert sampled.best.node_ids == greedy.best.node_ids
    assert sampled.best.votes == 3


async def test_sample_records_reshaped_distribution() -> None:
    result, _ = await walk(BORN_ROUTE, strategy="sample", n_samples=1, temperature=0.5)
    step = result.trace.steps[0]
    assert step.distribution != step.distribution_used
    assert math.isclose(sum(step.distribution_used.values()), 1.0)


# -- relation mode ---------------------------------------------------------------------

FILMS = "Which films were directed by the director of Inception?"
FILMS_ROUTE = {
    "Inception": {"directed_by": 1.0},
    "Christopher Nolan": {"directed_by^-1": 0.9, "STOP": 0.1},
}


async def test_relation_mode_moves_to_all_targets() -> None:
    result, backend = await walk(
        FILMS_ROUTE, FILMS, hop_mode="relation", strategy="greedy", allow_stop_at_start=False
    )
    best = result.best
    assert best is not None
    assert set(best.node_ids) == {"memento", "interstellar"}  # Inception excluded: visited
    assert best.terminated_by == "dead_end"  # their only edges lead back to visited Nolan
    options = backend.requests[1].questions[0].options
    described = {d["relation"]: d for d in options.values() if isinstance(d, dict)}
    assert described["directed_by"]["direction"] == "incoming"
    assert described["directed_by"]["count"] == 2


async def test_relation_mode_caps_the_frontier() -> None:
    result, _ = await walk(
        FILMS_ROUTE,
        FILMS,
        hop_mode="relation",
        strategy="greedy",
        allow_stop_at_start=False,
        max_frontier=1,
    )
    pruned = [p for s in result.trace.steps for p in s.pruned]
    assert any(p.reason == "frontier_truncated" for p in pruned)
    assert result.best is not None
    assert len(result.best.node_ids) == 1


# -- high-degree nodes -----------------------------------------------------------------


async def hub_store(n: int) -> NetworkXStore:
    store = NetworkXStore()
    await store.upsert_node(node("hub", name="Hub"))
    for i in range(n):
        name = "golden retriever puppy" if i == 37 else f"item number {i}"
        await store.upsert_node(node(f"n{i:03}", name=name))
        await store.upsert_edge(edge("hub", "has", f"n{i:03}"))
    return store


async def test_prefilter_keeps_the_query_relevant_options() -> None:
    backend = FakeDecisionBackend()
    config = cfg(
        strategy="greedy", prefilter_threshold=20, prefilter_top_n=5, budget=Budget(max_depth=1)
    )
    traverser = Traverser(await hub_store(60), backend, embedder=FakeEmbedder(), config=config)
    result = await traverser.traverse("find the golden retriever puppy", ["hub"])
    question = backend.requests[0].questions[0]
    assert len(question.options) == 6  # 5 kept + STOP
    assert any("golden retriever" in str(d) for d in question.options.values())
    step = result.trace.steps[0]
    assert len(step.pruned) == 55
    assert {p.reason for p in step.pruned} == {"prefilter"}
    assert result.trace.embedder_model == "fake-embedder-1"


async def test_without_embedder_options_are_truncated_to_the_backend_limit() -> None:
    backend = FakeDecisionBackend(max_options=10)
    config = cfg(strategy="greedy", budget=Budget(max_depth=1))
    result = await Traverser(await hub_store(30), backend, config=config).traverse("q", ["hub"])
    assert len(backend.requests[0].questions[0].options) == 10  # 9 moves + STOP
    assert {p.reason for p in result.trace.steps[0].pruned} == {"truncated"}
    assert len(result.trace.steps[0].pruned) == 21


# -- trace -----------------------------------------------------------------------------


async def test_trace_records_confidence_and_round_trips() -> None:
    result, _ = await walk(BORN_ROUTE, strategy="beam", beam_width=2, label_style="readable")
    decided = [s for s in result.trace.steps if s.call_id is not None]
    assert all(s.confidence is not None for s in decided)
    first = decided[0]
    assert first.confidence == pytest.approx(
        (len(first.distribution) * max(first.distribution.values()) - 1)
        / (len(first.distribution) - 1)
    )
    assert result.best is not None
    assert result.best.min_confidence is not None
    assert result.trace.config["label_style"] == "readable"
    assert "directed_by -> Christopher Nolan" in first.distribution
    restored = TraversalResult.model_validate_json(result.model_dump_json())
    assert restored == result


async def test_totals_sum_calls() -> None:
    result, _ = await walk(BORN_ROUTE, strategy="greedy")
    totals = result.trace.totals
    assert totals.input_tokens == sum(c.input_tokens for c in result.trace.calls)
    assert totals.cost_usd is None  # the oracle does not report cost
    assert totals.depth_reached == 3


async def test_cost_is_summed_when_reported() -> None:
    backend = FakeDecisionBackend(cost_per_call=0.25)
    config = cfg(strategy="greedy", budget=Budget(max_depth=2))
    result = await Traverser(await movie_store(), backend, config=config).traverse(
        BORN, ["inception"]
    )
    assert result.trace.totals.cost_usd == pytest.approx(0.25 * backend.calls)


async def test_preloaded_embeddings_avoid_embedding_calls() -> None:
    store = await hub_store(60)
    embedder = FakeEmbedder()
    config = cfg(
        strategy="greedy", prefilter_threshold=20, prefilter_top_n=5, budget=Budget(max_depth=1)
    )
    traverser = Traverser(store, FakeDecisionBackend(), embedder=embedder, config=config)
    texts = [node_text(n) async for n in store.iter_nodes()]
    traverser.preload_embeddings(texts, await embedder.embed(texts))
    embedder.calls.clear()
    await traverser.traverse("find the golden retriever puppy", ["hub"])
    assert embedder.calls == [["find the golden retriever puppy"]]  # only the query
    with pytest.raises(ValueError, match="no embedder"):
        Traverser(store, FakeDecisionBackend()).preload_embeddings([], await embedder.embed([]))
