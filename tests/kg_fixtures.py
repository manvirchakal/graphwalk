"""A small movie graph and a scripted decision oracle for traversal tests."""

from collections.abc import Mapping
from typing import Any, cast

from factories import edge, node
from graphwalk.decisions import ChoiceQuestion, FakeDecisionBackend, JSONContent
from graphwalk.stores.networkx_store import NetworkXStore

MOVIE_NODES = [
    node("inception", type_="film", name="Inception"),
    node("memento", type_="film", name="Memento"),
    node("interstellar", type_="film", name="Interstellar"),
    node("titanic", type_="film", name="Titanic"),
    node("nolan", type_="person", name="Christopher Nolan", aliases=("Nolan",)),
    node("cameron", type_="person", name="James Cameron"),
    node("dicaprio", type_="person", name="Leonardo DiCaprio"),
    node("london", type_="city", name="London"),
    node("uk", type_="country", name="United Kingdom"),
    node("y2010", type_="year", name="2010"),
]
MOVIE_EDGES = [
    edge("inception", "directed_by", "nolan"),
    edge("inception", "starred_actors", "dicaprio"),
    edge("inception", "release_year", "y2010"),
    edge("memento", "directed_by", "nolan"),
    edge("interstellar", "directed_by", "nolan"),
    edge("titanic", "directed_by", "cameron"),
    edge("titanic", "starred_actors", "dicaprio"),
    edge("nolan", "born_in", "london"),
    edge("london", "located_in", "uk"),
]


async def movie_store() -> NetworkXStore:
    store = NetworkXStore()
    for n in MOVIE_NODES:
        await store.upsert_node(n)
    for e in MOVIE_EDGES:
        await store.upsert_edge(e)
    return store


def option_key(description: JSONContent | None) -> str:
    """What an oracle route refers to: a target node name, a relation, or STOP."""
    if not isinstance(description, dict):
        return "STOP"
    if "node" in description:
        return str(cast("dict[str, object]", description["node"])["name"])
    direction = description["direction"]
    return str(description["relation"]) + ("" if direction == "outgoing" else "^-1")


def current_key(question: ChoiceQuestion) -> str:
    instructions = cast("dict[str, object]", question.instructions)
    current = cast("dict[str, object]", instructions["current"])
    if "name" in current:
        return str(current["name"])
    return ",".join(sorted(str(n) for n in cast("list[object]", current["nodes"])))


type Route = Mapping[str, Mapping[str, float]]
"""current node name (or sorted comma-joined frontier names) -> option key -> weight."""


def oracle(route: Route, *, default: float = 0.0, **backend: Any) -> FakeDecisionBackend:
    """A fake backend that answers from ``route``. Unlisted options get ``default``;
    an unlisted current node falls back to the seeded Dirichlet draw."""

    def script(question: ChoiceQuestion, state: JSONContent) -> dict[str, float] | None:
        del state
        weights = route.get(current_key(question))
        if weights is None:
            return None
        dist = {
            label: weights.get(option_key(desc), default)
            for label, desc in question.options.items()
        }
        if sum(dist.values()) <= 0:
            return dict.fromkeys(dist, 1.0)
        return dist

    return FakeDecisionBackend(script=script, **backend)
