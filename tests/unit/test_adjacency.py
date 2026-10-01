"""The SQLite fast path (``AdjacencyStore``) walks exactly like the ``neighbors`` path."""

import random
from typing import Any

import pytest

from factories import edge, node
from graphwalk.core.errors import NodeNotFoundError
from graphwalk.decisions import FakeDecisionBackend
from graphwalk.embeddings import FakeEmbedder
from graphwalk.stores.base import AdjacencyStore, NodeLabel
from graphwalk.stores.networkx_store import NetworkXStore
from graphwalk.stores.sqlite_store import SQLiteStore
from graphwalk.traversal import TraversalConfig, Traverser

TYPES = ("person", "film", "place")
RELATIONS = ("a", "b", "c", "d")


async def _stores() -> tuple[NetworkXStore, SQLiteStore]:
    """A random graph with two hubs, self-loops, and nodes with and without summaries."""
    rng = random.Random(7)
    nodes = [
        node(f"n{i}", type_=TYPES[i % 3], name=f"Node {i}",
             summary=None if i % 2 else f"summary {i}")
        for i in range(120)
    ]  # fmt: skip
    edges = {
        (f"n{rng.randrange(120)}", rng.choice(RELATIONS), f"n{rng.randrange(120)}")
        for _ in range(300)
    }
    edges |= {(f"n{i}", "member_of", f"n{i % 2}") for i in range(2, 120)}  # hubs n0, n1
    edges |= {("n5", "a", "n5"), ("n0", "b", "n0")}
    plain, fast = NetworkXStore(), SQLiteStore(":memory:")
    for store in (plain, fast):
        for n in nodes:
            await store.upsert_node(n)
        for source, relation, target in sorted(edges):
            await store.upsert_edge(edge(source, relation, target))
    return plain, fast


@pytest.mark.parametrize("direction", ["out", "in", "both"])
async def test_adjacency_matches_neighbors(direction: Any) -> None:
    _, fast = await _stores()
    assert isinstance(fast, AdjacencyStore)
    for i in range(120):
        expected = [
            (n.edge.type, n.direction, n.node.id)
            for n in await fast.neighbors(f"n{i}", direction=direction)
        ]
        found = await fast.adjacency(f"n{i}", direction=direction)
        assert sorted(found, key=lambda a: (a.direction != "out", a.relation, a.other)) == expected
    with pytest.raises(NodeNotFoundError):
        await fast.adjacency("missing", direction=direction)


async def test_node_labels() -> None:
    _, fast = await _stores()
    labels = await fast.node_labels(["n1", "n2", "missing", "n1"])
    assert labels == {
        "n1": NodeLabel("Node 1", "film", None),
        "n2": NodeLabel("Node 2", "place", "summary 2"),
    }


def _comparable(result: Any) -> Any:
    dumped = result.model_dump(mode="json")
    trace = dumped["trace"]
    for key in ("totals", "started_at", "wall_s"):
        trace.pop(key, None)
    for call in trace.get("calls", []):
        call.pop("latency_s", None)
    return dumped["status"], dumped["answers"], trace["steps"]


@pytest.mark.parametrize("hop_mode", ["entity", "relation"])
@pytest.mark.parametrize("embedder", [False, True])
async def test_walks_are_identical_on_both_paths(hop_mode: Any, embedder: bool) -> None:
    plain, fast = await _stores()
    config = TraversalConfig(
        strategy="beam", beam_width=3, hop_mode=hop_mode, prefilter_threshold=8,
        prefilter_top_n=5, max_frontier=6, budget={"max_depth": 3},  # pyright: ignore[reportArgumentType]
    )  # fmt: skip
    for start in ("n0", "n1", "n5", "n17"):
        results = [
            await Traverser(
                store,
                FakeDecisionBackend(seed=3),
                embedder=FakeEmbedder() if embedder else None,
                config=config,
            ).traverse("Which node is related?", (start,))
            for store in (plain, fast)
        ]
        assert _comparable(results[0]) == _comparable(results[1])
        assert results[0].trace.steps  # a real walk, not an immediate stop
