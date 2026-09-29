"""Apply routing decisions to the store, and retract a document's contributions.

Nothing is silently overwritten: an entity routed to an existing node is folded in with
:func:`~graphwalk.core.merge.merge_node_data`, so disagreeing attribute values become
:class:`~graphwalk.core.model.AttributeConflict` records with their provenance.
"""

from collections.abc import Collection
from dataclasses import dataclass

from pydantic import JsonValue

from graphwalk.core.hashing import content_hash
from graphwalk.core.merge import TYPE_CONFLICT_KEY, merge_edge_data, merge_node_data
from graphwalk.core.model import (
    AttributeConflict,
    ConflictingValue,
    Edge,
    Node,
    NodeId,
    Provenance,
)
from graphwalk.ingest.extraction import GENERIC_TYPE, ExtractedEntity, name_key
from graphwalk.stores.base import GraphStore


def new_node_id(doc_source_id: str, entity: ExtractedEntity) -> NodeId:
    """Deterministic, so re-ingesting a document recreates the same ids."""
    digest = content_hash([doc_source_id, name_key(entity.name), entity.type])
    return f"n:{digest.removeprefix('sha256:')[:24]}"


def entity_node(node_id: NodeId, entity: ExtractedEntity, provenance: Provenance) -> Node:
    return Node(
        id=node_id,
        type=entity.type,
        name=entity.name,
        summary=entity.description,
        attributes=dict[str, JsonValue](entity.attributes),
        provenance=(provenance,),
    )


async def upsert_entity(
    store: GraphStore,
    target: NodeId | None,
    entity: ExtractedEntity,
    provenance: Provenance,
    doc_source_id: str,
) -> tuple[Node, bool]:
    """Create a node for ``entity`` or fold it into ``target``. Returns ``(node, created)``."""
    node_id = target or new_node_id(doc_source_id, entity)
    existing = await store.get_node(node_id)
    if existing is None:
        node = entity_node(node_id, entity, provenance)
        await store.upsert_node(node)
        return node, True
    if entity.type == GENERIC_TYPE:  # untyped mention: no type claim to conflict with
        entity = entity.model_copy(update={"type": existing.type})
    elif existing.type == GENERIC_TYPE:  # first real type for a generic node
        existing = existing.model_copy(update={"type": entity.type})
    merged = merge_node_data(existing, entity_node(node_id, entity, provenance))
    await store.upsert_node(merged)
    return merged, False


async def upsert_relation(
    store: GraphStore, source: NodeId, type_: str, target: NodeId, provenance: Provenance
) -> bool | None:
    """Add or reinforce an edge. Returns created (True), merged (False), or skipped (None)."""
    if source == target:
        return None
    edge = Edge(source=source, target=target, type=type_, provenance=(provenance,))
    existing = await store.get_edge(edge.id)
    if existing is None:
        await store.upsert_edge(edge)
        return True
    await store.upsert_edge(merge_edge_data(existing, edge))
    return False


@dataclass
class Retraction:
    nodes_deleted: int = 0
    nodes_updated: int = 0
    edges_deleted: int = 0
    edges_updated: int = 0


def _keep(provenance: tuple[Provenance, ...], gone: Collection[str]) -> tuple[Provenance, ...]:
    return tuple(p for p in provenance if p.source_id not in gone)


def _retract_element[T: Node | Edge](element: T, gone: Collection[str]) -> T | None:
    """``element`` without provenance from ``gone``; ``None`` if nothing supports it."""
    provenance = _keep(element.provenance, gone)
    if not provenance:
        return None
    attributes: dict[str, JsonValue] = {}
    sources: dict[str, tuple[Provenance, ...]] = {}
    for key, value in element.attributes.items():
        kept = _keep(element.sources_of(key), gone)
        if kept:
            attributes[key] = value
            if key in element.attribute_provenance:
                sources[key] = kept
    conflicts: dict[str, AttributeConflict] = {}
    for key, conflict in element.conflicts.items():
        values = [
            ConflictingValue(value=cv.value, provenance=kept)
            for cv in conflict.values
            if (kept := _keep(cv.provenance, gone))
        ]
        if len(values) >= 2:  # noqa: PLR2004 - still a conflict
            conflicts[key] = AttributeConflict(key=key, values=tuple(values))
        elif values and key != TYPE_CONFLICT_KEY and key not in attributes:
            # The current value's sources were retracted; the one survivor takes over.
            attributes[key] = values[0].value
            sources[key] = values[0].provenance
    return element.model_copy(
        update={
            "provenance": provenance,
            "attributes": attributes,
            "attribute_provenance": sources,
            "conflicts": conflicts,
        }
    )


async def retract(store: GraphStore, gone: Collection[str]) -> Retraction:
    """Remove every provenance record whose ``source_id`` is in ``gone``.

    Elements left with no provenance are deleted; attributes left with no source are
    dropped. Not re-derived: a surviving node keeps its name, summary, and aliases even
    if they came from a retracted document.
    """
    report = Retraction()
    if not gone:
        return report
    edges = [e async for e in store.iter_edges()]
    for edge in edges:
        if not any(p.source_id in gone for p in edge.provenance):
            continue
        kept = _retract_element(edge, gone)
        if kept is None:
            await store.delete_edge(edge.id)
            report.edges_deleted += 1
        else:
            await store.upsert_edge(kept)
            report.edges_updated += 1
    nodes = [n async for n in store.iter_nodes()]
    for node in nodes:
        if not any(p.source_id in gone for p in node.provenance):
            continue
        kept = _retract_element(node, gone)
        if kept is None:
            report.edges_deleted += await store.degree(node.id, direction="both")
            await store.delete_node(node.id)
            report.nodes_deleted += 1
        else:
            await store.upsert_node(kept)
            report.nodes_updated += 1
    return report
