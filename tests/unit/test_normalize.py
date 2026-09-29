from factories import prov
from graphwalk.core.model import Edge, Node
from graphwalk.decisions import ChoiceQuestion, FakeDecisionBackend, JSONContent
from graphwalk.ingest.normalize import (
    DEFAULT_SCHEMA,
    MAPPING_KEY,
    OTHER,
    REVERSED,
    attributes_to_edges,
    normalize_graph,
    schema_options,
)
from graphwalk.stores.networkx_store import NetworkXStore

CHOICES = {
    "directed_by": "director",
    "director_of": "director" + REVERSED,
    "married": "spouse",
    "born_in": "place_of_birth",
    "vibes_with": OTHER,
}


def script(question: ChoiceQuestion, _state: JSONContent) -> dict[str, float]:
    assert isinstance(question.instructions, dict)
    relation = str(question.instructions["relation"])
    pick = CHOICES.get(relation, OTHER)
    return {label: float(label == pick) for label in question.options}


def n(node_id: str, name: str, type_: str = "thing", **attributes: str) -> Node:
    return Node(id=node_id, type=type_, name=name, attributes=dict(attributes),
                provenance=(prov("s/" + node_id),))  # fmt: skip


def e(source: str, type_: str, target: str) -> Edge:
    return Edge(source=source, target=target, type=type_, provenance=(prov("s/e"),))


async def store() -> NetworkXStore:
    s = NetworkXStore()
    for node in (
        n("film", "Inception", "film"),
        n("nolan", "Christopher Nolan", "person", born_in="London", birth_date="1970-07-30"),
        n("london", "London", "city"),
        n("emma", "Emma Thomas", "person"),
        n("memento", "Memento", "film"),
    ):
        await s.upsert_node(node)
    for edge in (
        e("film", "directed_by", "nolan"),
        e("nolan", "director_of", "memento"),
        e("nolan", "married", "emma"),
        e("nolan", "vibes_with", "london"),
        e("nolan", "director_of", "film"),  # collides with film --director--> nolan
    ):
        await s.upsert_edge(edge)
    return s


def test_schema_options() -> None:
    options = schema_options(DEFAULT_SCHEMA)
    assert options["father"] == "target is the father of source (a person)"
    assert options["father" + REVERSED] == "(reversed) source is the father of target (a person)"
    assert "spouse" + REVERSED not in options  # symmetric
    assert OTHER in options
    assert len(options) <= 255


async def test_attributes_become_edges() -> None:
    s = await store()
    assert await attributes_to_edges(s) == 1  # born_in: London (the date is skipped)
    assert await attributes_to_edges(s) == 0  # idempotent
    types = [(x.source, x.type, x.target) async for x in s.iter_edges()]
    assert ("nolan", "born_in", "london") in types


async def test_normalize_maps_reverses_and_merges() -> None:
    s = await store()
    decider = FakeDecisionBackend(script=script, cost_per_call=0.001)
    report = await normalize_graph(s, decider)
    edges = sorted([(x.source, x.type, x.target) async for x in s.iter_edges()])
    assert edges == [
        ("film", "director", "nolan"),
        ("memento", "director", "nolan"),
        ("nolan", "place_of_birth", "london"),
        ("nolan", "spouse", "emma"),
        ("nolan", "vibes_with", "london"),  # OTHER keeps its extracted name
    ]
    assert report.attribute_edges == 1
    assert report.relation_types == 5
    assert report.mapped_types == 4
    assert report.edges_merged == 1
    assert report.spend.decision_calls == decider.calls == 1
    assert report.spend.cost_usd == 0.001
    (merged,) = [x async for x in s.iter_edges() if x.source == "film"]
    assert len(merged.provenance) == 1  # same provenance record, de-duplicated
    mapping = await s.get_metadata(MAPPING_KEY)
    assert isinstance(mapping, dict)
    assert mapping["director_of"] == ["director", True, 1.0]


async def test_low_confidence_and_failures_keep_extracted_names() -> None:
    s = await store()
    unsure = FakeDecisionBackend(script=lambda q, _s: dict.fromkeys(q.options, 1.0))
    report = await normalize_graph(s, unsure)
    assert report.mapped_types == 0
    failing = FakeDecisionBackend(fail_on_calls=[1])
    s2 = await store()
    report2 = await normalize_graph(s2, failing)
    assert report2.mapped_types == 0
    assert {x.type async for x in s2.iter_edges()} >= {"directed_by", "married"}
