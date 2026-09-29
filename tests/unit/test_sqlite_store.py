"""SQLite-specific behavior: persistence, concurrent readers, schema, provenance index."""

import sqlite3
from pathlib import Path

import pytest

from factories import edge, node, prov
from graphwalk.core.model import Provenance
from graphwalk.stores import SQLiteStore


async def test_reopen_keeps_everything(tmp_path: Path) -> None:
    path = tmp_path / "sub" / "g.db"
    store = SQLiteStore(path)
    await store.upsert_node(node("a", name="Alpha", aliases=("A1",)))
    await store.upsert_node(node("b"))
    await store.upsert_edge(edge("a", "knows", "b"))
    await store.set_metadata("k", {"x": [1, None]})
    await store.close()
    again = SQLiteStore(path)
    assert await again.counts() == (2, 1)
    assert [n.id for n in await again.find_nodes(name="a1")] == ["a"]
    assert await again.get_metadata("k") == {"x": [1, None]}
    await again.close()


async def test_readers_see_committed_data_while_a_writer_is_open(tmp_path: Path) -> None:
    path = tmp_path / "g.db"
    writer = SQLiteStore(path)
    reader = SQLiteStore(path)
    await writer.upsert_node(node("a"))
    assert await reader.get_node("a") is not None
    # A long write transaction in progress does not block readers (WAL mode).
    raw = sqlite3.connect(path, isolation_level=None)
    raw.execute("BEGIN IMMEDIATE")
    raw.execute("DELETE FROM nodes")
    assert await reader.counts() == (1, 0)
    raw.execute("ROLLBACK")
    raw.close()
    mode = sqlite3.connect(path).execute("PRAGMA journal_mode").fetchone()[0]
    assert mode == "wal"
    await writer.close()
    await reader.close()


async def test_rejects_unknown_schema_version(tmp_path: Path) -> None:
    path = tmp_path / "g.db"
    raw = sqlite3.connect(path)
    raw.execute("PRAGMA user_version = 99")
    raw.close()
    with pytest.raises(ValueError, match="schema version 99"):
        SQLiteStore(path)


async def test_failed_merge_rolls_back() -> None:
    store = SQLiteStore(":memory:")
    await store.upsert_node(node("a"))
    with pytest.raises(Exception, match="missing"):
        await store.merge_nodes("a", "missing")
    assert await store.counts() == (1, 0)


async def test_provenance_index_finds_elements_by_source() -> None:
    store = SQLiteStore(":memory:")
    spanned = Provenance.model_validate({**prov("doc/1").model_dump(), "start": 3, "end": 9})
    await store.upsert_node(node("a", source="doc/1"))
    await store.upsert_node(node("b").model_copy(update={"provenance": (spanned, prov("x"))}))
    await store.upsert_node(node("c", source="doc/2"))
    await store.upsert_edge(edge("a", "r", "c", src="doc/2"))
    found = await store.elements_from(["doc/1", "doc/2"])
    assert ("node", "a") in found
    assert ("node", "b") in found
    assert ("node", "c") in found
    assert len([f for f in found if f[0] == "edge"]) == 1
    await store.delete_node("c")
    assert await store.elements_from(["doc/2"]) == []
