"""NetworkXStore-specific behavior (persistence). Shared behavior is in tests/contract."""

import json
from pathlib import Path

import pytest

from factories import edge, node
from graphwalk.stores import NetworkXStore


async def build() -> NetworkXStore:
    store = NetworkXStore()
    for n in (node("a", x=1), node("b", name="Bé", aliases=("B",)), node("c")):
        await store.upsert_node(n)
    for e in (edge("a", "r", "b", w=0.5), edge("b", "s", "c"), edge("c", "loop", "c")):
        await store.upsert_edge(e)
    await store.merge_nodes("a", "c")  # exercise conflicts / aliases / provenance
    return store


async def test_save_load_round_trip(tmp_path: Path) -> None:
    store = await build()
    await store.set_metadata("ledger", {"doc": ["sha256:1", 2]})
    path = tmp_path / "nested" / "g.graph.json"
    await store.save(path)
    loaded = await NetworkXStore.load(path)
    assert await loaded.get_metadata("ledger") == {"doc": ["sha256:1", 2]}
    assert [n async for n in loaded.iter_nodes()] == [n async for n in store.iter_nodes()]
    assert [e async for e in loaded.iter_edges()] == [e async for e in store.iter_edges()]
    assert await loaded.counts() == await store.counts()


async def test_save_is_deterministic_and_leaves_no_temp_file(tmp_path: Path) -> None:
    store = await build()
    await store.save(tmp_path / "one.json")
    await store.save(tmp_path / "two.json")
    assert (tmp_path / "one.json").read_bytes() == (tmp_path / "two.json").read_bytes()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["one.json", "two.json"]


async def test_load_rejects_foreign_files(tmp_path: Path) -> None:
    path = tmp_path / "other.json"
    path.write_text(json.dumps({"format": "something-else", "version": 1}))
    with pytest.raises(ValueError, match=r"not a graphwalk\.networkx"):
        await NetworkXStore.load(path)


async def test_load_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        await NetworkXStore.load(tmp_path / "nope.json")
