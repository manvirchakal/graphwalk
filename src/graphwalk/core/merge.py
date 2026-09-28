"""Pure merge logic shared by every GraphStore implementation.

Merging never overwrites: the kept element's values stay in place, and any disagreeing
value is recorded as an :class:`AttributeConflict` together with its provenance.
"""

import json
from collections.abc import Iterable, Mapping

from pydantic import JsonValue

from graphwalk.core.model import AttributeConflict, ConflictingValue, Edge, Node, Provenance

TYPE_CONFLICT_KEY = "@type"
"""Conflict key used when merged nodes disagree on ``Node.type`` (not an attribute)."""


def _value_key(value: JsonValue) -> str:
    # Canonical JSON distinguishes 1 / 1.0 / True, which Python's == does not.
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def union_provenance(*groups: Iterable[Provenance]) -> tuple[Provenance, ...]:
    """Order-preserving, de-duplicated union of provenance records."""
    return tuple(dict.fromkeys(p for group in groups for p in group))


def _add_value(
    values: dict[str, ConflictingValue], value: JsonValue, provenance: Iterable[Provenance]
) -> None:
    key = _value_key(value)
    existing = values.get(key)
    if existing is None:
        values[key] = ConflictingValue(value=value, provenance=tuple(provenance))
    else:
        values[key] = existing.model_copy(
            update={"provenance": union_provenance(existing.provenance, provenance)}
        )


def _merge_conflict_maps(
    *maps: Mapping[str, AttributeConflict],
) -> dict[str, dict[str, ConflictingValue]]:
    merged: dict[str, dict[str, ConflictingValue]] = {}
    for conflicts in maps:
        for key, conflict in conflicts.items():
            bucket = merged.setdefault(key, {})
            for cv in conflict.values:
                _add_value(bucket, cv.value, cv.provenance)
    return merged


def _record(
    buckets: dict[str, dict[str, ConflictingValue]],
    key: str,
    kept: tuple[JsonValue, Iterable[Provenance]],
    incoming: tuple[JsonValue, Iterable[Provenance]],
) -> None:
    bucket = buckets.setdefault(key, {})
    _add_value(bucket, *kept)
    _add_value(bucket, *incoming)


def _freeze(buckets: dict[str, dict[str, ConflictingValue]]) -> dict[str, AttributeConflict]:
    return {
        key: AttributeConflict(key=key, values=tuple(values.values()))
        for key, values in buckets.items()
        if len(values) >= 2  # noqa: PLR2004 - a conflict needs two distinct values
    }


def _merge_attributes(
    keep: Node | Edge, absorb: Node | Edge, buckets: dict[str, dict[str, ConflictingValue]]
) -> tuple[dict[str, JsonValue], dict[str, tuple[Provenance, ...]]]:
    attributes = dict(keep.attributes)
    sources = {key: keep.sources_of(key) for key in attributes}
    for key, value in absorb.attributes.items():
        incoming = absorb.sources_of(key)
        if key not in attributes:
            attributes[key] = value
            sources[key] = incoming
        elif _value_key(attributes[key]) == _value_key(value):
            sources[key] = union_provenance(sources[key], incoming)
        else:
            _record(buckets, key, (attributes[key], sources[key]), (value, incoming))
    return attributes, sources


def merge_node_data(keep: Node, absorb: Node) -> Node:
    """Fold ``absorb`` into ``keep`` and return the merged node (``keep``'s id)."""
    buckets = _merge_conflict_maps(keep.conflicts, absorb.conflicts)
    attributes, sources = _merge_attributes(keep, absorb, buckets)
    if keep.type != absorb.type:
        bucket = buckets.setdefault(TYPE_CONFLICT_KEY, {})
        if _value_key(keep.type) not in bucket:  # earlier merges already attributed it
            _add_value(bucket, keep.type, keep.provenance)
        _add_value(bucket, absorb.type, absorb.provenance)
    aliases = tuple(
        a for a in dict.fromkeys((*keep.aliases, absorb.name, *absorb.aliases)) if a != keep.name
    )
    return keep.model_copy(
        update={
            "summary": keep.summary if keep.summary is not None else absorb.summary,
            "aliases": aliases,
            "attributes": attributes,
            "attribute_provenance": sources,
            "provenance": union_provenance(keep.provenance, absorb.provenance),
            "conflicts": _freeze(buckets),
        }
    )


def merge_edge_data(keep: Edge, absorb: Edge) -> Edge:
    """Fold two edges with the same id into one."""
    if keep.id != absorb.id:
        msg = f"cannot merge different edges {keep.id!r} and {absorb.id!r}"
        raise ValueError(msg)
    buckets = _merge_conflict_maps(keep.conflicts, absorb.conflicts)
    attributes, sources = _merge_attributes(keep, absorb, buckets)
    return keep.model_copy(
        update={
            "attributes": attributes,
            "attribute_provenance": sources,
            "provenance": union_provenance(keep.provenance, absorb.provenance),
            "conflicts": _freeze(buckets),
        }
    )


def rewire_edge(edge: Edge, absorbed: str, keep: str) -> Edge | None:
    """Re-point an edge from ``absorbed`` to ``keep``.

    Returns ``None`` for an edge that ran between the two merged nodes, since it would
    become a meaningless self-loop. A pre-existing self-loop on ``absorbed`` is kept.
    """
    was_self_loop = edge.source == edge.target
    source = keep if edge.source == absorbed else edge.source
    target = keep if edge.target == absorbed else edge.target
    if source == target and not was_self_loop:
        return None
    # The id is derived from (source, type, target), so construct a new edge.
    return Edge.model_validate(
        {
            "source": source,
            "target": target,
            "type": edge.type,
            "attributes": edge.attributes,
            "attribute_provenance": edge.attribute_provenance,
            "provenance": edge.provenance,
            "conflicts": edge.conflicts,
        }
    )
