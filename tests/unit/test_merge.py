import pytest

from factories import edge, node, prov
from graphwalk.core.merge import (
    TYPE_CONFLICT_KEY,
    merge_edge_data,
    merge_node_data,
    rewire_edge,
    union_provenance,
)


def test_union_provenance_dedupes_preserving_order() -> None:
    a, b = prov("a"), prov("b")
    assert union_provenance((a, b), (b, a, prov("a"))) == (a, b)


def test_merge_keeps_ids_names_and_unions_provenance() -> None:
    keep = node("k", name="Paris", source="s1", aliases=("City of Light",))
    absorb = node("x", name="paris, fr", source="s2", aliases=("Paris",))
    merged = merge_node_data(keep, absorb)
    assert merged.id == "k"
    assert merged.name == "Paris"
    assert merged.aliases == ("City of Light", "paris, fr")  # keep.name is not an alias
    assert [p.source_id for p in merged.provenance] == ["s1", "s2"]


def test_merge_fills_missing_attributes_and_summary() -> None:
    keep = node("k", a=1)
    absorb = node("x", b=2, summary="from x")
    merged = merge_node_data(keep, absorb)
    assert merged.attributes == {"a": 1, "b": 2}
    assert merged.summary == "from x"
    assert merged.conflicts == {}


def test_merge_equal_values_are_not_conflicts() -> None:
    merged = merge_node_data(node("k", pop=5, source="s1"), node("x", pop=5, source="s2"))
    assert merged.conflicts == {}


def test_merge_records_conflict_without_overwriting() -> None:
    keep = node("k", population=2_100_000, source="census")
    absorb = node("x", population=2_200_000, source="wiki")
    merged = merge_node_data(keep, absorb)
    assert merged.attributes["population"] == 2_100_000  # kept value stays
    conflict = merged.conflicts["population"]
    assert [(v.value, [p.source_id for p in v.provenance]) for v in conflict.values] == [
        (2_100_000, ["census"]),
        (2_200_000, ["wiki"]),
    ]


def test_merge_distinguishes_int_from_bool() -> None:
    merged = merge_node_data(node("k", flag=1), node("x", flag=True, source="s2"))
    assert "flag" in merged.conflicts


def test_merge_accumulates_existing_conflicts() -> None:
    first = merge_node_data(node("k", v=1, source="s1"), node("x", v=2, source="s2"))
    second = merge_node_data(first, node("y", v=3, source="s3"))
    values = [cv.value for cv in second.conflicts["v"].values]
    assert values == [1, 2, 3]
    # re-merging a value already present unions its provenance instead of duplicating it
    third = merge_node_data(second, node("z", v=2, source="s4"))
    by_value = {
        cv.value: [p.source_id for p in cv.provenance] for cv in third.conflicts["v"].values
    }
    assert by_value == {1: ["s1"], 2: ["s2", "s4"], 3: ["s3"]}


def test_merge_records_type_disagreement() -> None:
    merged = merge_node_data(node("k", type_="city"), node("x", type_="person", source="s2"))
    assert merged.type == "city"
    assert [cv.value for cv in merged.conflicts[TYPE_CONFLICT_KEY].values] == ["city", "person"]


def test_merge_edge_data_requires_same_id() -> None:
    with pytest.raises(ValueError, match="different edges"):
        merge_edge_data(edge("a", "r", "b"), edge("a", "r", "c"))
    merged = merge_edge_data(edge("a", "r", "b", w=1), edge("a", "r", "b", src="s2", w=2))
    assert merged.attributes == {"w": 1}
    assert [cv.value for cv in merged.conflicts["w"].values] == [1, 2]
    assert len(merged.provenance) == 2


@pytest.mark.parametrize(
    ("original", "expected"),
    [
        (("x", "r", "b"), ("k", "r", "b")),  # outgoing from absorbed
        (("b", "r", "x"), ("b", "r", "k")),  # incoming to absorbed
        (("x", "r", "x"), ("k", "r", "k")),  # absorbed's own self-loop survives
        (("x", "r", "k"), None),  # edge between the merged pair is dropped
        (("k", "r", "x"), None),
    ],
)
def test_rewire_edge(original: tuple[str, str, str], expected: tuple[str, str, str] | None) -> None:
    result = rewire_edge(edge(*original), absorbed="x", keep="k")
    if expected is None:
        assert result is None
    else:
        assert result is not None
        assert (result.source, result.type, result.target) == expected


def test_merge_tracks_which_source_contributed_each_attribute() -> None:
    merged = merge_node_data(node("k", a=1, source="s1"), node("x", b=2, source="s2"))
    assert [p.source_id for p in merged.sources_of("a")] == ["s1"]
    assert [p.source_id for p in merged.sources_of("b")] == ["s2"]
    # agreeing sources are both credited
    again = merge_node_data(merged, node("y", a=1, source="s3"))
    assert [p.source_id for p in again.sources_of("a")] == ["s1", "s3"]


def test_repeated_type_conflicts_do_not_misattribute() -> None:
    first = merge_node_data(node("k", type_="city"), node("x", type_="person", source="s2"))
    second = merge_node_data(first, node("y", type_="place", source="s3"))
    by_value = {
        cv.value: [p.source_id for p in cv.provenance]
        for cv in second.conflicts[TYPE_CONFLICT_KEY].values
    }
    assert by_value == {"city": ["src"], "person": ["s2"], "place": ["s3"]}
