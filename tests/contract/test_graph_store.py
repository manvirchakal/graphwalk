"""Behavioral contract every GraphStore implementation must satisfy.

Add a backend by adding a param to the ``store`` fixture. Tests only use the protocol.
"""

from collections.abc import AsyncIterator, Callable, Sequence

import pytest

from factories import edge, node, prov
from graphwalk.core.errors import EdgeNotFoundError, NodeNotFoundError
from graphwalk.core.merge import merge_edge_data
from graphwalk.core.model import AttributeConflict, ConflictingValue, Direction, Neighbor, Node
from graphwalk.stores import GraphStore, NetworkXStore

STORE_FACTORIES: dict[str, Callable[[], GraphStore]] = {
    "networkx": NetworkXStore,
    # "neo4j": added in M4 with pytest.mark.neo4j
}


@pytest.fixture(params=list(STORE_FACTORIES))
async def store(request: pytest.FixtureRequest) -> AsyncIterator[GraphStore]:
    s = STORE_FACTORIES[request.param]()
    await s.clear()
    try:
        yield s
    finally:
        await s.clear()
        await s.close()


async def seed(store: GraphStore, *nodes: Node) -> None:
    for n in nodes:
        await store.upsert_node(n)


def triples(neighbors: Sequence[Neighbor]) -> list[tuple[str, str, str]]:
    return [(n.direction, n.edge.type, n.node.id) for n in neighbors]


# ---------------------------------------------------------------- basics


async def test_implements_protocol(store: GraphStore) -> None:
    assert isinstance(store, GraphStore)


async def test_empty_store(store: GraphStore) -> None:
    assert await store.counts() == (0, 0)
    assert await store.get_node("missing") is None
    assert await store.get_edge("missing") is None
    assert [n async for n in store.iter_nodes()] == []


async def test_node_round_trip_is_lossless(store: GraphStore) -> None:
    conflict = AttributeConflict(
        key="pop",
        values=(
            ConflictingValue(value=1, provenance=(prov("a"),)),
            ConflictingValue(value={"n": [1, 2.5, None, True]}, provenance=(prov("b"),)),
        ),
    )
    original = node(
        "n1",
        type_="city",
        name="Zürich 東京",
        summary="multi\nline",
        aliases=("Zurich",),
        pop=1,
        nested={"a": [1, 2.5, None, True, "x"]},
    ).model_copy(
        update={
            "conflicts": {"pop": conflict},
            "attribute_provenance": {"pop": (prov("a", confidence=0.25),)},
            "provenance": (prov("a"), prov("b", confidence=0.5, content="h1")),
        }
    )
    await store.upsert_node(original)
    assert await store.get_node("n1") == original


async def test_upsert_node_replaces(store: GraphStore) -> None:
    await store.upsert_node(node("n", color="red"))
    await store.upsert_node(node("n", color="blue"))
    got = await store.get_node("n")
    assert got is not None
    assert got.attributes == {"color": "blue"}
    assert await store.counts() == (1, 0)


async def test_upsert_node_keeps_edges(store: GraphStore) -> None:
    await seed(store, node("a"), node("b"))
    await store.upsert_edge(edge("a", "r", "b"))
    await store.upsert_node(node("a", changed=True))
    assert await store.counts() == (2, 1)


async def test_get_nodes_skips_missing(store: GraphStore) -> None:
    await seed(store, node("a"), node("b"))
    got = await store.get_nodes(["b", "zzz", "a"])
    assert set(got) == {"a", "b"}
    assert got["a"].id == "a"


async def test_returned_models_are_snapshots(store: GraphStore) -> None:
    await store.upsert_node(node("a", tags=["x"]))
    got = await store.get_node("a")
    assert got is not None
    got.attributes["tags"] = ["mutated"]
    again = await store.get_node("a")
    assert again is not None
    assert again.attributes == {"tags": ["x"]}


# ---------------------------------------------------------------- edges


async def test_edge_requires_endpoints(store: GraphStore) -> None:
    await seed(store, node("a"))
    with pytest.raises(NodeNotFoundError):
        await store.upsert_edge(edge("a", "r", "ghost"))
    with pytest.raises(NodeNotFoundError):
        await store.upsert_edge(edge("ghost", "r", "a"))
    assert await store.counts() == (1, 0)


async def test_edge_round_trip_and_replace(store: GraphStore) -> None:
    await seed(store, node("a"), node("b"))
    e1 = edge("a", "r", "b", w=1)
    await store.upsert_edge(e1)
    assert await store.get_edge(e1.id) == e1
    e2 = edge("a", "r", "b", w=2)
    await store.upsert_edge(e2)
    assert await store.get_edge(e1.id) == e2
    assert await store.counts() == (2, 1)


async def test_parallel_edges_of_different_types(store: GraphStore) -> None:
    await seed(store, node("a"), node("b"))
    await store.upsert_edge(edge("a", "r", "b"))
    await store.upsert_edge(edge("a", "s", "b"))
    await store.upsert_edge(edge("b", "r", "a"))
    assert await store.counts() == (2, 3)


async def test_delete_edge(store: GraphStore) -> None:
    await seed(store, node("a"), node("b"))
    e = edge("a", "r", "b")
    await store.upsert_edge(e)
    await store.delete_edge(e.id)
    assert await store.get_edge(e.id) is None
    assert await store.counts() == (2, 0)
    with pytest.raises(EdgeNotFoundError):
        await store.delete_edge(e.id)


# ---------------------------------------------------------------- neighbors


@pytest.fixture
async def star(store: GraphStore) -> GraphStore:
    """a -r-> c, a -r-> b, a -s-> b, d -r-> a, a -loop-> a, b -r-> c."""
    await seed(store, *(node(i) for i in "abcd"))
    for e in [
        edge("a", "r", "c"),
        edge("a", "r", "b"),
        edge("a", "s", "b"),
        edge("d", "r", "a"),
        edge("a", "loop", "a"),
        edge("b", "r", "c"),
    ]:
        await store.upsert_edge(e)
    return store


async def test_neighbors_out_are_ordered(star: GraphStore) -> None:
    assert triples(await star.neighbors("a")) == [
        ("out", "loop", "a"),
        ("out", "r", "b"),
        ("out", "r", "c"),
        ("out", "s", "b"),
    ]


async def test_neighbors_in(star: GraphStore) -> None:
    assert triples(await star.neighbors("a", direction="in")) == [
        ("in", "loop", "a"),
        ("in", "r", "d"),
    ]


async def test_neighbors_both_reports_self_loop_once(star: GraphStore) -> None:
    assert triples(await star.neighbors("a", direction="both")) == [
        ("out", "loop", "a"),
        ("out", "r", "b"),
        ("out", "r", "c"),
        ("out", "s", "b"),
        ("in", "r", "d"),
    ]


async def test_neighbors_edge_type_filter(star: GraphStore) -> None:
    got = await star.neighbors("a", direction="both", edge_types={"r"})
    assert triples(got) == [("out", "r", "b"), ("out", "r", "c"), ("in", "r", "d")]
    assert await star.neighbors("a", edge_types=[]) == []


async def test_neighbors_carry_full_models(star: GraphStore) -> None:
    first = (await star.neighbors("d"))[0]
    assert first.edge == edge("d", "r", "a")
    assert first.node == node("a")


async def test_neighbors_of_missing_node_raises(store: GraphStore) -> None:
    with pytest.raises(NodeNotFoundError):
        await store.neighbors("ghost")
    with pytest.raises(NodeNotFoundError):
        await store.degree("ghost")


@pytest.mark.parametrize("direction", ["out", "in", "both"])
@pytest.mark.parametrize("edge_types", [None, {"r"}, {"loop", "s"}])
async def test_degree_matches_neighbors(
    star: GraphStore, direction: Direction, edge_types: set[str] | None
) -> None:
    for node_id in "abcd":
        n = await star.neighbors(node_id, direction=direction, edge_types=edge_types)
        d = await star.degree(node_id, direction=direction, edge_types=edge_types)
        assert d == len(n), (node_id, direction, edge_types)


# ---------------------------------------------------------------- lookup & iteration


async def test_find_nodes(store: GraphStore) -> None:
    await seed(
        store,
        node("p2", type_="city", name="Paris"),
        node("p1", type_="person", name="paris hilton", aliases=("PARIS",)),
        node("x", type_="city", name="Lyon"),
    )
    assert [n.id for n in await store.find_nodes(name="paris")] == ["p1", "p2"]
    assert [n.id for n in await store.find_nodes(name="Paris", type="city")] == ["p2"]
    assert [n.id for n in await store.find_nodes(type="city")] == ["p2", "x"]
    assert [n.id for n in await store.find_nodes(type="city", limit=1)] == ["p2"]
    assert await store.find_nodes(name="par") == []  # exact, not substring


async def test_iteration_is_complete_and_ordered(star: GraphStore) -> None:
    assert [n.id async for n in star.iter_nodes(batch_size=2)] == ["a", "b", "c", "d"]
    edge_ids = [e.id async for e in star.iter_edges(batch_size=2)]
    assert edge_ids == sorted(edge_ids)
    assert len(edge_ids) == 6


# ---------------------------------------------------------------- deletion


async def test_delete_node_removes_incident_edges(star: GraphStore) -> None:
    await star.delete_node("a")
    assert await star.get_node("a") is None
    assert await star.counts() == (3, 1)  # only b -r-> c survives
    assert await star.get_edge(edge("d", "r", "a").id) is None
    assert await star.neighbors("d") == []
    with pytest.raises(NodeNotFoundError):
        await star.delete_node("a")


async def test_clear(star: GraphStore) -> None:
    await star.clear()
    assert await star.counts() == (0, 0)


# ---------------------------------------------------------------- merge


async def test_merge_nodes_rewires_and_preserves_provenance(store: GraphStore) -> None:
    await seed(
        store,
        node("k", name="Paris", source="s1", pop=1),
        node("x", name="Paris, France", source="s2", pop=2, country="FR"),
        node("b"),
        node("c"),
    )
    kb_existing = edge("k", "r", "b", src="s1")
    for e in [
        edge("x", "r", "b", src="s2"),  # collides with k -r-> b after rewiring
        kb_existing,
        edge("c", "s", "x"),  # becomes c -s-> k
        edge("x", "t", "k"),  # between the pair: dropped
        edge("x", "u", "x"),  # absorbed self-loop: becomes k -u-> k
    ]:
        await store.upsert_edge(e)

    merged = await store.merge_nodes("k", "x")

    assert await store.get_node("x") is None
    assert await store.get_node("k") == merged
    assert merged.aliases == ("Paris, France",)
    assert merged.attributes == {"pop": 1, "country": "FR"}
    assert [cv.value for cv in merged.conflicts["pop"].values] == [1, 2]
    assert [p.source_id for p in merged.provenance] == ["s1", "s2"]

    assert triples(await store.neighbors("k", direction="both")) == [
        ("out", "r", "b"),
        ("out", "u", "k"),
        ("in", "s", "c"),
    ]
    assert await store.counts() == (3, 3)
    kb = await store.get_edge(kb_existing.id)
    assert kb == merge_edge_data(kb_existing, edge("k", "r", "b", src="s2"))
    assert kb is not None
    assert [p.source_id for p in kb.provenance] == ["s1", "s2"]


async def test_merge_nodes_errors(store: GraphStore) -> None:
    await seed(store, node("a"))
    with pytest.raises(ValueError, match="itself"):
        await store.merge_nodes("a", "a")
    with pytest.raises(NodeNotFoundError):
        await store.merge_nodes("a", "ghost")
    with pytest.raises(NodeNotFoundError):
        await store.merge_nodes("ghost", "a")
    assert await store.counts() == (1, 0)
