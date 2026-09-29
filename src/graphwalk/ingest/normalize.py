"""Normalize an extracted graph onto a fixed relation schema.

LLM extraction invents relation names freely (``legally_acknowledged_as_son``,
``succeeded``, ``directed_by`` vs. ``director_of``), which sends walks astray. This pass:

1. turns entity-valued attributes into edges, when the value exactly names one node
   (``born_in: "Helsingfors"`` becomes an edge to the Helsingfors node), so a walk can
   reach it; and
2. maps every distinct relation type onto :data:`DEFAULT_SCHEMA` with one decision
   question per type (batched), shown a few example edges. Each schema relation can be
   chosen as-is or reversed, so ``directed_by`` and ``director_of`` both become
   ``director`` with the right direction. Types mapped to ``OTHER`` or below
   ``min_probability`` keep their extracted name.

The mapping is stored in the store's metadata under :data:`MAPPING_KEY`.
"""

import re
from collections import defaultdict
from dataclasses import dataclass, field

from pydantic import JsonValue

from graphwalk.core.merge import merge_edge_data
from graphwalk.core.model import Edge, Node, NodeId
from graphwalk.decisions.base import (
    ChoiceQuestion,
    DecisionBackend,
    DecisionBackendError,
    DecisionRequest,
    JSONContent,
)
from graphwalk.ingest.extraction import name_key, snake
from graphwalk.ingest.spend import Spend
from graphwalk.stores.base import GraphStore
from graphwalk.traversal.batching import split_questions

MAPPING_KEY = "normalize:relations"
OTHER = "OTHER"
REVERSED = "__rev"


@dataclass(frozen=True)
class SchemaRelation:
    name: str
    description: str
    """What an edge ``source --name--> target`` means."""
    symmetric: bool = False


def _r(name: str, description: str, *, symmetric: bool = False) -> SchemaRelation:
    return SchemaRelation(name, description, symmetric=symmetric)


DEFAULT_SCHEMA: tuple[SchemaRelation, ...] = (
    # Chosen a priori from common Wikidata properties for people, works, organizations,
    # and places (not from any eval's answers).
    _r("father", "target is the father of source (a person)"),
    _r("mother", "target is the mother of source (a person)"),
    _r("child", "target is a son or daughter of source"),
    _r("spouse", "source and target are or were married", symmetric=True),
    _r("sibling", "source and target are brothers or sisters", symmetric=True),
    _r("relative", "source and target are related by family in another way", symmetric=True),
    _r("director", "target directed source (a film, show, or other work)"),
    _r("producer", "target produced source (a work)"),
    _r("screenwriter", "target wrote the screenplay of source (a work)"),
    _r("cast_member", "target (an actor) appears in source (a work)"),
    _r("performer", "target performed or recorded source (a song, album, or work)"),
    _r("composer", "target composed the music of source (a work)"),
    _r("author", "target wrote source (a book or text)"),
    _r("publisher", "target published source (a work)"),
    _r("production_company", "target (a company) produced or distributed source (a work)"),
    _r("genre", "source (a work or artist) belongs to target (a genre)"),
    _r("based_on", "source (a work) is based on or adapted from target (a work)"),
    _r("place_of_birth", "source (a person) was born in target (a place)"),
    _r("place_of_death", "source (a person) died in target (a place)"),
    _r("place_of_burial", "source (a person) is buried in target (a place)"),
    _r("residence", "source (a person) lived in target (a place)"),
    _r("country_of_citizenship", "source (a person) is a citizen or national of target"),
    _r("educated_at", "source (a person) studied at target (a school or university)"),
    _r("employer", "source (a person) worked for target (an organization)"),
    _r("member_of", "source is a member of target (a group, band, team, or party)"),
    _r("position_held", "source (a person) held target (an office, title, or position)"),
    _r("occupation", "source (a person) works as target (an occupation)"),
    _r("award_received", "source received target (an award or honor)"),
    _r("nominated_for", "source was nominated for target (an award)"),
    _r("founded_by", "target founded source (an organization or place)"),
    _r("owned_by", "target owns or controls source"),
    _r("headquarters", "source (an organization) is headquartered in target (a place)"),
    _r("located_in", "source is located in target (a larger place or region)"),
    _r("country", "source belongs to or is in target (a country)"),
    _r("capital", "target is the capital of source (a country or region)"),
    _r("part_of", "source is part of target"),
    _r("follows", "source came after target (a successor in office, title, or series)"),
    _r("instance_of", "source is an instance or kind of target"),
)

TASK = (
    "A knowledge graph was extracted from text with free-form relation names. Choose "
    "the schema relation that means the same as the extracted relation, as shown by its "
    "example edges 'source --relation--> target'. Options ending in (reversed) mean the "
    "same relation with source and target swapped. Choose OTHER if no schema relation "
    "means the same."
)


def schema_options(schema: tuple[SchemaRelation, ...]) -> dict[str, JSONContent | None]:
    options: dict[str, JSONContent | None] = {}
    for relation in schema:
        options[relation.name] = relation.description
        if not relation.symmetric:
            flipped = relation.description.replace("target", "\0").replace("source", "target")
            options[relation.name + REVERSED] = "(reversed) " + flipped.replace("\0", "source")
    options[OTHER] = "none of these relations means the same"
    return options


_ENTITYISH = re.compile(r"[A-Za-z]")
_DATEISH = re.compile(r"^\W*\d")


@dataclass
class NormalizeReport:
    attribute_edges: int = 0
    relation_types: int = 0
    mapped_types: int = 0
    edges_rewritten: int = 0
    edges_merged: int = 0
    spend: Spend = field(default_factory=Spend)
    mapping: dict[str, tuple[str, bool, float]] = field(
        default_factory=dict[str, tuple[str, bool, float]]
    )
    """Extracted type -> (schema relation or itself, reversed, probability)."""


async def attributes_to_edges(store: GraphStore) -> int:
    """Add an edge for each string attribute whose value exactly names one node."""
    by_name: dict[str, list[NodeId]] = defaultdict(list)
    nodes: list[Node] = []
    async for node in store.iter_nodes():
        nodes.append(node)
        for name in {node.name, *node.aliases}:
            by_name[name_key(name)].append(node.id)
    added = 0
    for node in nodes:
        for key, value in node.attributes.items():
            if not isinstance(value, str) or not _ENTITYISH.search(value):
                continue
            if _DATEISH.match(value):
                continue
            targets = sorted(set(by_name.get(name_key(value), [])))
            relation = snake(key)
            if len(targets) != 1 or targets[0] == node.id or not relation:
                continue
            edge = Edge(
                source=node.id,
                target=targets[0],
                type=relation,
                provenance=node.sources_of(key),
            )
            if await store.get_edge(edge.id) is None:
                await store.upsert_edge(edge)
                added += 1
    return added


async def _examples(store: GraphStore, per_type: int) -> dict[str, list[str]]:
    examples: dict[str, list[str]] = defaultdict(list)
    ends: dict[str, list[tuple[NodeId, NodeId]]] = defaultdict(list)
    async for edge in store.iter_edges():
        pairs = ends[edge.type]  # creates the entry, so every type is listed
        if len(pairs) < per_type:
            pairs.append((edge.source, edge.target))
    ids = sorted({n for pairs in ends.values() for pair in pairs for n in pair})
    found = await store.get_nodes(ids)

    def card(node_id: NodeId) -> str:
        node = found.get(node_id)
        return node_id if node is None else f"{node.name} ({node.type})"

    for type_, pairs in ends.items():
        examples[type_] = [f"{card(s)} --{type_}--> {card(t)}" for s, t in pairs]
    return examples


async def map_relations(
    store: GraphStore,
    decider: DecisionBackend,
    *,
    schema: tuple[SchemaRelation, ...] = DEFAULT_SCHEMA,
    examples_per_type: int = 3,
    min_probability: float = 0.5,
    report: NormalizeReport | None = None,
) -> NormalizeReport:
    """Decide a schema relation for every distinct relation type in ``store``."""
    report = report or NormalizeReport()
    examples = await _examples(store, examples_per_type)
    options = schema_options(schema)
    questions = [
        ChoiceQuestion(
            key=f"r{i}",
            instructions={"relation": t, "examples": list[JsonValue](examples[t])},
            options=options,
        )
        for i, t in enumerate(sorted(examples))
    ]
    types = dict(zip((q.key for q in questions), sorted(examples), strict=True))
    report.relation_types = len(types)
    state: dict[str, JsonValue] = {"task": TASK}
    for group in split_questions(
        state,
        questions,
        max_request_tokens=64_000,
        max_state_plus_question_tokens=32_000,
        safety_margin=0.2,
    ):
        report.spend.decision_calls += 1
        try:
            response = await decider.decide(DecisionRequest(state=state, questions=tuple(group)))
        except DecisionBackendError:
            for question in group:
                report.mapping[types[question.key]] = (types[question.key], False, 0.0)
            continue
        report.spend.input_tokens += response.usage.input_tokens
        report.spend.output_tokens += response.usage.output_tokens
        report.spend.add_cost(response.usage.cost_usd)
        for question in group:
            type_ = types[question.key]
            result = response.results[question.key]
            p = result.probabilities[result.top]
            if result.top == OTHER or p < min_probability:
                report.mapping[type_] = (type_, False, p)
                continue
            reversed_ = result.top.endswith(REVERSED)
            report.mapping[type_] = (result.top.removesuffix(REVERSED), reversed_, p)
            report.mapped_types += 1
    return report


async def apply_mapping(store: GraphStore, report: NormalizeReport) -> None:
    """Rewrite edges to their schema relation (and direction); merge collisions."""
    edges = [e async for e in store.iter_edges()]
    for edge in edges:
        target_type, reversed_, _p = report.mapping.get(edge.type, (edge.type, False, 1.0))
        if target_type == edge.type and not reversed_:
            continue
        source, target = (edge.target, edge.source) if reversed_ else (edge.source, edge.target)
        rewritten = Edge.model_validate(
            {
                "source": source,
                "target": target,
                "type": target_type,
                "attributes": edge.attributes,
                "attribute_provenance": edge.attribute_provenance,
                "provenance": edge.provenance,
                "conflicts": edge.conflicts,
            }
        )
        await store.delete_edge(edge.id)
        existing = await store.get_edge(rewritten.id)
        if existing is None:
            await store.upsert_edge(rewritten)
        else:
            await store.upsert_edge(merge_edge_data(existing, rewritten))
            report.edges_merged += 1
        report.edges_rewritten += 1
    mapping: dict[str, JsonValue] = {
        t: [canon, reversed_, p] for t, (canon, reversed_, p) in sorted(report.mapping.items())
    }
    await store.set_metadata(MAPPING_KEY, mapping)


async def normalize_graph(
    store: GraphStore,
    decider: DecisionBackend,
    *,
    schema: tuple[SchemaRelation, ...] = DEFAULT_SCHEMA,
    examples_per_type: int = 3,
    min_probability: float = 0.5,
) -> NormalizeReport:
    """Attributes to edges, then relation types onto ``schema``, in place."""
    report = NormalizeReport()
    report.attribute_edges = await attributes_to_edges(store)
    await map_relations(
        store,
        decider,
        schema=schema,
        examples_per_type=examples_per_type,
        min_probability=min_probability,
        report=report,
    )
    await apply_mapping(store, report)
    return report
