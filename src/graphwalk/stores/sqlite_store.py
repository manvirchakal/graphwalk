"""``GraphStore`` and ``DocumentStore`` on one SQLite file.

Nodes and edges are stored as their JSON dumps, with the columns queries filter on
(type, endpoints, casefolded names) broken out and indexed. Provenance spans get their
own indexed table, so everything derived from a document can be found without a scan.

The database runs in write-ahead-log mode: other connections (another process serving
``locate``, say) can read while ingestion writes. Each public method is one
transaction.
"""

import json
import os
import sqlite3
from collections.abc import AsyncIterator, Collection, Generator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

from pydantic import JsonValue

from graphwalk.core.errors import EdgeNotFoundError, NodeNotFoundError
from graphwalk.core.merge import merge_edge_data, merge_node_data, rewire_edge
from graphwalk.core.model import (
    Direction,
    Edge,
    EdgeId,
    Neighbor,
    Node,
    NodeId,
    StoredDocument,
)

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS nodes (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    data TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS nodes_type ON nodes (type, id);
CREATE TABLE IF NOT EXISTS node_names (
    name_key TEXT NOT NULL,
    node_id TEXT NOT NULL REFERENCES nodes (id) ON DELETE CASCADE,
    PRIMARY KEY (name_key, node_id)
);
CREATE INDEX IF NOT EXISTS node_names_node ON node_names (node_id);
CREATE TABLE IF NOT EXISTS edges (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL REFERENCES nodes (id),
    target TEXT NOT NULL REFERENCES nodes (id),
    type TEXT NOT NULL,
    data TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS edges_out ON edges (source, type);
CREATE INDEX IF NOT EXISTS edges_in ON edges (target, type);
CREATE TABLE IF NOT EXISTS provenance (
    element_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    source_id TEXT NOT NULL,
    start INTEGER,
    end INTEGER
);
CREATE INDEX IF NOT EXISTS provenance_element ON provenance (element_id);
CREATE INDEX IF NOT EXISTS provenance_source ON provenance (source_id, start);
CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS documents (
    key TEXT PRIMARY KEY,
    data TEXT NOT NULL
);
"""


def _name_keys(node: Node) -> set[str]:
    return {name.casefold() for name in (node.name, *node.aliases)}


class SQLiteStore:
    """A graph (and its source documents) in one SQLite database file.

    ``path=":memory:"`` gives a private in-memory database, handy for tests.
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.path, isolation_level=None)
        self._db.execute("PRAGMA foreign_keys = ON")
        self._db.execute("PRAGMA busy_timeout = 5000")
        if self.path != ":memory:":
            self._db.execute("PRAGMA journal_mode = WAL")
            self._db.execute("PRAGMA synchronous = NORMAL")
        version: int = self._db.execute("PRAGMA user_version").fetchone()[0]
        if version not in (0, SCHEMA_VERSION):
            self._db.close()
            msg = f"{self.path}: schema version {version}, expected {SCHEMA_VERSION}"
            raise ValueError(msg)
        # executescript commits by itself; every statement in the schema is idempotent.
        self._db.executescript(_SCHEMA)
        self._db.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    @contextmanager
    def _tx(self) -> Generator[sqlite3.Connection]:
        """One write transaction; nested use joins the outer one."""
        if self._db.in_transaction:
            yield self._db
            return
        self._db.execute("BEGIN IMMEDIATE")
        try:
            yield self._db
        except BaseException:
            self._db.execute("ROLLBACK")
            raise
        self._db.execute("COMMIT")

    # ------------------------------------------------------------------ internals

    def _node(self, node_id: NodeId) -> Node:
        row = self._db.execute("SELECT data FROM nodes WHERE id = ?", (node_id,)).fetchone()
        if row is None:
            raise NodeNotFoundError(node_id)
        return Node.model_validate_json(row[0])

    def _edge(self, edge_id: EdgeId) -> Edge | None:
        row = self._db.execute("SELECT data FROM edges WHERE id = ?", (edge_id,)).fetchone()
        return None if row is None else Edge.model_validate_json(row[0])

    def _index_provenance(self, element_id: str, kind: str, element: Node | Edge) -> None:
        self._db.execute("DELETE FROM provenance WHERE element_id = ?", (element_id,))
        self._db.executemany(
            "INSERT INTO provenance (element_id, kind, source_id, start, end) "
            "VALUES (?, ?, ?, ?, ?)",
            [(element_id, kind, p.source_id, p.start, p.end) for p in element.provenance],
        )

    def _put_node(self, node: Node) -> None:
        self._db.execute(
            "INSERT INTO nodes (id, type, data) VALUES (?, ?, ?) "
            "ON CONFLICT (id) DO UPDATE SET type = excluded.type, data = excluded.data",
            (node.id, node.type, node.model_dump_json()),
        )
        self._db.execute("DELETE FROM node_names WHERE node_id = ?", (node.id,))
        self._db.executemany(
            "INSERT INTO node_names (name_key, node_id) VALUES (?, ?)",
            [(key, node.id) for key in sorted(_name_keys(node))],
        )
        self._index_provenance(node.id, "node", node)

    def _put_edge(self, edge: Edge) -> None:
        for endpoint in (edge.source, edge.target):
            if self._db.execute("SELECT 1 FROM nodes WHERE id = ?", (endpoint,)).fetchone() is None:
                raise NodeNotFoundError(endpoint)
        self._db.execute(
            "INSERT INTO edges (id, source, target, type, data) VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT (id) DO UPDATE SET data = excluded.data",
            (edge.id, edge.source, edge.target, edge.type, edge.model_dump_json()),
        )
        self._index_provenance(edge.id, "edge", edge)

    def _remove_edge(self, edge_id: EdgeId) -> None:
        self._db.execute("DELETE FROM edges WHERE id = ?", (edge_id,))
        self._db.execute("DELETE FROM provenance WHERE element_id = ?", (edge_id,))

    def _incident(
        self, node_id: NodeId, direction: Direction, edge_types: Collection[str] | None
    ) -> list[tuple[Edge, NodeId, Literal["out", "in"]]]:
        self._node(node_id)  # raise if missing
        types = None if edge_types is None else sorted(set(edge_types))
        if types == []:
            return []
        type_filter = "" if types is None else f" AND type IN ({','.join('?' * len(types))})"
        args = [] if types is None else types
        found: list[tuple[Edge, NodeId, Literal["out", "in"]]] = []
        if direction in ("out", "both"):
            rows = self._db.execute(
                f"SELECT data FROM edges WHERE source = ?{type_filter}",  # noqa: S608 - placeholders only
                (node_id, *args),
            )
            for (data,) in rows:
                edge = Edge.model_validate_json(data)
                found.append((edge, edge.target, "out"))
        if direction in ("in", "both"):
            rows = self._db.execute(
                f"SELECT data FROM edges WHERE target = ?{type_filter}",  # noqa: S608 - placeholders only
                (node_id, *args),
            )
            for (data,) in rows:
                edge = Edge.model_validate_json(data)
                if direction == "both" and edge.source == edge.target:
                    continue  # self-loop already reported as "out"
                found.append((edge, edge.source, "in"))
        return found

    def _nodes_where(self, sql: str, args: Sequence[object]) -> list[Node]:
        return [Node.model_validate_json(data) for (data,) in self._db.execute(sql, args)]

    # ------------------------------------------------------------------ GraphStore

    async def upsert_node(self, node: Node) -> None:
        with self._tx():
            self._put_node(node)

    async def upsert_edge(self, edge: Edge) -> None:
        with self._tx():
            self._put_edge(edge)

    async def upsert_many(self, nodes: Sequence[Node], edges: Sequence[Edge]) -> None:
        """Upsert ``nodes``, then ``edges``, in one transaction (bulk import)."""
        with self._tx():
            for node in nodes:
                self._put_node(node)
            for edge in edges:
                self._put_edge(edge)

    async def get_node(self, node_id: NodeId) -> Node | None:
        try:
            return self._node(node_id)
        except NodeNotFoundError:
            return None

    async def get_nodes(self, node_ids: Sequence[NodeId]) -> dict[NodeId, Node]:
        found: dict[NodeId, Node] = {}
        unique = list(dict.fromkeys(node_ids))
        for start in range(0, len(unique), 500):
            batch = unique[start : start + 500]
            rows = self._db.execute(
                f"SELECT id, data FROM nodes WHERE id IN ({','.join('?' * len(batch))})",  # noqa: S608
                batch,
            )
            for node_id, data in rows:
                found[node_id] = Node.model_validate_json(data)
        return {node_id: found[node_id] for node_id in node_ids if node_id in found}

    async def get_edge(self, edge_id: EdgeId) -> Edge | None:
        return self._edge(edge_id)

    async def neighbors(
        self,
        node_id: NodeId,
        *,
        direction: Direction = "out",
        edge_types: Collection[str] | None = None,
    ) -> list[Neighbor]:
        incident = self._incident(node_id, direction, edge_types)
        others = await self.get_nodes([other for _, other, _ in incident])
        result = [
            Neighbor(edge=edge, node=others[other], direction=dir_)
            for edge, other, dir_ in incident
        ]
        result.sort(key=lambda n: (n.direction != "out", n.edge.type, n.node.id))
        return result

    async def degree(
        self,
        node_id: NodeId,
        *,
        direction: Direction = "out",
        edge_types: Collection[str] | None = None,
    ) -> int:
        """Counted in SQL: a hub's degree never parses its edges."""
        self._node(node_id)  # raise if missing
        types = None if edge_types is None else sorted(set(edge_types))
        if types == []:
            return 0
        type_filter = "" if types is None else f" AND type IN ({','.join('?' * len(types))})"
        args = [] if types is None else types
        total = 0
        if direction in ("out", "both"):
            total += self._db.execute(
                f"SELECT COUNT(*) FROM edges WHERE source = ?{type_filter}",  # noqa: S608 - placeholders only
                (node_id, *args),
            ).fetchone()[0]
        if direction in ("in", "both"):
            # With "both", a self-loop is counted once (as "out"), as neighbors reports it.
            loops = " AND source != target" if direction == "both" else ""
            total += self._db.execute(
                f"SELECT COUNT(*) FROM edges WHERE target = ?{type_filter}{loops}",  # noqa: S608 - placeholders only
                (node_id, *args),
            ).fetchone()[0]
        return total

    async def find_nodes(
        self,
        *,
        name: str | None = None,
        type: str | None = None,  # noqa: A002 - mirrors Node.type
        limit: int | None = None,
    ) -> list[Node]:
        clauses: list[str] = []
        args: list[object] = []
        if name is not None:
            clauses.append("id IN (SELECT node_id FROM node_names WHERE name_key = ?)")
            args.append(name.casefold())
        if type is not None:
            clauses.append("type = ?")
            args.append(type)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        bound = ""
        if limit is not None:
            bound = " LIMIT ?"
            args.append(max(limit, 0))
        return self._nodes_where(f"SELECT data FROM nodes{where} ORDER BY id{bound}", args)  # noqa: S608

    async def iter_nodes(self, *, batch_size: int = 1000) -> AsyncIterator[Node]:
        after = ""
        while True:
            batch = self._nodes_where(
                "SELECT data FROM nodes WHERE id > ? ORDER BY id LIMIT ?", (after, batch_size)
            )
            for node in batch:
                yield node
            if len(batch) < batch_size:
                return
            after = batch[-1].id

    async def iter_edges(self, *, batch_size: int = 1000) -> AsyncIterator[Edge]:
        after = ""
        while True:
            rows = self._db.execute(
                "SELECT data FROM edges WHERE id > ? ORDER BY id LIMIT ?", (after, batch_size)
            ).fetchall()
            batch = [Edge.model_validate_json(data) for (data,) in rows]
            for edge in batch:
                yield edge
            if len(batch) < batch_size:
                return
            after = batch[-1].id

    async def delete_node(self, node_id: NodeId) -> None:
        with self._tx():
            self._delete_node(node_id)

    def _delete_node(self, node_id: NodeId) -> None:
        self._node(node_id)
        rows = self._db.execute(
            "SELECT id FROM edges WHERE source = ? OR target = ?", (node_id, node_id)
        ).fetchall()
        for (edge_id,) in rows:
            self._remove_edge(edge_id)
        self._db.execute("DELETE FROM nodes WHERE id = ?", (node_id,))
        self._db.execute("DELETE FROM provenance WHERE element_id = ?", (node_id,))

    async def delete_edge(self, edge_id: EdgeId) -> None:
        with self._tx():
            if self._edge(edge_id) is None:
                raise EdgeNotFoundError(edge_id)
            self._remove_edge(edge_id)

    async def merge_nodes(self, keep: NodeId, absorb: NodeId) -> Node:
        if keep == absorb:
            msg = "cannot merge a node into itself"
            raise ValueError(msg)
        with self._tx():
            kept, absorbed = self._node(keep), self._node(absorb)
            merged = merge_node_data(kept, absorbed)
            rewired = [
                rewire_edge(edge, absorb, keep)
                for edge, _, _ in self._incident(absorb, "both", None)
            ]
            self._delete_node(absorb)
            self._put_node(merged)
            for moved in rewired:
                if moved is None:
                    continue
                existing = self._edge(moved.id)
                self._put_edge(moved if existing is None else merge_edge_data(existing, moved))
        return merged

    async def get_metadata(self, key: str) -> JsonValue | None:
        row = self._db.execute("SELECT value FROM metadata WHERE key = ?", (key,)).fetchone()
        if row is None:
            return None
        value: JsonValue = json.loads(row[0])
        return value

    async def set_metadata(self, key: str, value: JsonValue | None) -> None:
        with self._tx():
            if value is None:
                self._db.execute("DELETE FROM metadata WHERE key = ?", (key,))
            else:
                self._db.execute(
                    "INSERT INTO metadata (key, value) VALUES (?, ?) "
                    "ON CONFLICT (key) DO UPDATE SET value = excluded.value",
                    (key, json.dumps(value, ensure_ascii=False, allow_nan=False)),
                )

    async def counts(self) -> tuple[int, int]:
        nodes: int = self._db.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
        edges: int = self._db.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
        return nodes, edges

    async def clear(self) -> None:
        with self._tx():
            for table in ("provenance", "edges", "node_names", "nodes", "metadata", "documents"):
                self._db.execute(f"DELETE FROM {table}")  # noqa: S608 - fixed table names

    async def close(self) -> None:
        self._db.close()

    # ------------------------------------------------------------------ DocumentStore

    async def put_document(self, document: StoredDocument) -> None:
        with self._tx():
            self._db.execute(
                "INSERT INTO documents (key, data) VALUES (?, ?) "
                "ON CONFLICT (key) DO UPDATE SET data = excluded.data",
                (document.key, document.model_dump_json()),
            )

    async def get_document(self, key: str) -> StoredDocument | None:
        row = self._db.execute("SELECT data FROM documents WHERE key = ?", (key,)).fetchone()
        return None if row is None else StoredDocument.model_validate_json(row[0])

    async def delete_document(self, key: str) -> None:
        with self._tx():
            self._db.execute("DELETE FROM documents WHERE key = ?", (key,))

    async def iter_documents(self) -> AsyncIterator[StoredDocument]:
        rows = self._db.execute("SELECT data FROM documents ORDER BY key").fetchall()
        for (data,) in rows:
            yield StoredDocument.model_validate_json(data)

    # ------------------------------------------------------------------ extras

    async def elements_from(self, source_ids: Collection[str]) -> list[tuple[str, str]]:
        """``(kind, element id)`` of every node and edge with provenance from any of
        ``source_ids``, via the provenance index (no scan)."""
        ids = sorted(set(source_ids))
        found: list[tuple[str, str]] = []
        for start in range(0, len(ids), 500):
            batch = ids[start : start + 500]
            found += self._db.execute(
                "SELECT DISTINCT kind, element_id FROM provenance "  # noqa: S608
                f"WHERE source_id IN ({','.join('?' * len(batch))}) ORDER BY kind, element_id",
                batch,
            ).fetchall()
        return found
