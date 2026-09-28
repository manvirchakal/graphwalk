from datetime import datetime

import pytest
from pydantic import ValidationError

from factories import T0, edge, node, prov
from graphwalk.core.model import (
    AttributeConflict,
    ConflictingValue,
    Edge,
    Node,
    Provenance,
    edge_id,
)


def test_provenance_requires_timezone() -> None:
    with pytest.raises(ValidationError):
        Provenance(
            source_id="s",
            ingested_at=datetime(2026, 1, 1),
            confidence=1.0,
            content_hash="h",
        )


@pytest.mark.parametrize("confidence", [-0.01, 1.01])
def test_provenance_confidence_bounds(confidence: float) -> None:
    with pytest.raises(ValidationError):
        Provenance(source_id="s", ingested_at=T0, confidence=confidence, content_hash="h")


def test_nodes_and_edges_require_provenance() -> None:
    with pytest.raises(ValidationError):
        Node.model_validate({"id": "a", "type": "t", "name": "a", "provenance": []})
    with pytest.raises(ValidationError):
        Edge(source="a", target="b", type="r", provenance=())


def test_models_are_frozen() -> None:
    n = node("a")
    with pytest.raises(ValidationError):
        n.name = "b"  # type: ignore[misc]


def test_edge_id_is_deterministic_and_direction_sensitive() -> None:
    assert edge("a", "r", "b").id == edge("a", "r", "b", src="other").id == edge_id("a", "r", "b")
    assert edge("a", "r", "b").id != edge("b", "r", "a").id
    assert edge("a", "r", "b").id != edge("a", "s", "b").id


def test_edge_id_is_unambiguous_across_field_boundaries() -> None:
    assert edge_id("a", "bc", "d") != edge_id("ab", "c", "d")


def test_edge_round_trips_through_json_including_id() -> None:
    e = edge("a", "r", "b", weight=2)
    dumped = e.model_dump(mode="json")
    assert dumped["id"] == e.id
    assert Edge.model_validate(dumped) == e


def test_edge_rejects_inconsistent_serialized_id() -> None:
    dumped = edge("a", "r", "b").model_dump(mode="json")
    dumped["id"] = "e:forged"
    with pytest.raises(ValidationError, match="does not match"):
        Edge.model_validate(dumped)


def test_conflict_needs_two_values_and_matching_key() -> None:
    one = ConflictingValue(value=1, provenance=(prov(),))
    two = ConflictingValue(value=2, provenance=(prov("other"),))
    with pytest.raises(ValidationError):
        AttributeConflict(key="x", values=(one,))
    conflict = AttributeConflict(key="x", values=(one, two))
    with pytest.raises(ValidationError, match="stored under"):
        Node.model_validate({**node("a").model_dump(), "conflicts": {"y": conflict.model_dump()}})


def test_attribute_provenance_must_refer_to_existing_attributes() -> None:
    with pytest.raises(ValidationError, match="no attribute"):
        Node.model_validate(
            {**node("a").model_dump(), "attribute_provenance": {"ghost": [prov().model_dump()]}}
        )


def test_sources_of_falls_back_to_element_provenance() -> None:
    n = node("a", x=1, source="s1")
    assert n.sources_of("x") == n.provenance
