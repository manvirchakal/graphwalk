"""In-memory ``GraphStore`` on a NetworkX ``MultiDiGraph``, with JSON save/load."""

import asyncio
import copy
import json
import os
from collections.abc import AsyncIterator, Collection, Iterator, Sequence
from pathlib import Path
from typing import Any, Literal, Self

import networkx as nx
from pydantic import JsonValue

from graphwalk.core.errors import EdgeNotFoundError, NodeNotFoundError
from graphwalk.core.merge import merge_edge_data, merge_node_data, rewire_edge
from graphwalk.core.model import Direction, Edge, EdgeId, Neighbor, Node, NodeId

FILE_FORMAT = "graphwalk.networkx"
FILE_VERSION = 1

_NODE = "node"
_EDGE = "edge"


class NetworkXStore:
    """In-memory graph store.

    Nodes and edges are stored as private copies of the frozen models; all reads return
    copies too, so the store behaves like an external database. NetworkX edge keys are
    the graphwalk edge ids.
    """

    def __init__(self) -> None:
        self._graph: nx.MultiDiGraph[NodeId] = nx.MultiDiGraph()
        self._edge_index: dict[EdgeId, tuple[NodeId, NodeId]] = {}
        self._metadata: dict[str, JsonValue] = {}

    # ------------------------------------------------------------------ internals

    def _node(self, node_id: NodeId) -> Node:
        try:
            node: Node = self._graph.nodes[node_id][_NODE]
        except KeyError:
            raise NodeNotFoundError(node_id) from None
        return node

    def _edge(self, edge_id: EdgeId) -> Edge | None:
        endpoints = self._edge_index.get(edge_id)
        if endpoints is None:
            return None
        edge: Edge = self._graph.edges[endpoints[0], endpoints[1], edge_id][_EDGE]
        return edge

    def _put_edge(self, edge: Edge) -> None:
        self._graph.add_edge(edge.source, edge.target, key=edge.id, **{_EDGE: _copy(edge)})
        self._edge_index[edge.id] = (edge.source, edge.target)

    def _remove_edge(self, edge: Edge) -> None:
        self._graph.remove_edge(edge.source, edge.target, key=edge.id)
        del self._edge_index[edge.id]

    def _incident(
        self, node_id: NodeId, direction: Direction, edge_types: Collection[str] | None
    ) -> Iterator[tuple[Edge, NodeId, Literal["out", "in"]]]:
        self._node(node_id)  # raise if missing
        types = None if edge_types is None else frozenset(edge_types)
        if direction in ("out", "both"):
            for _, other, data in self._graph.out_edges(node_id, data=True):
                edge: Edge = data[_EDGE]
                if types is None or edge.type in types:
                    yield edge, other, "out"
        if direction in ("in", "both"):
            for other, _, data in self._graph.in_edges(node_id, data=True):
                edge = data[_EDGE]
                if direction == "both" and edge.source == edge.target:
                    continue  # self-loop already reported as "out"
                if types is None or edge.type in types:
                    yield edge, other, "in"

    # ------------------------------------------------------------------ GraphStore

    async def upsert_node(self, node: Node) -> None:
        if node.id in self._graph:
            self._graph.nodes[node.id][_NODE] = _copy(node)
        else:
            self._graph.add_node(node.id, **{_NODE: _copy(node)})

    async def upsert_edge(self, edge: Edge) -> None:
        for endpoint in (edge.source, edge.target):
            self._node(endpoint)
        existing = self._edge(edge.id)
        if existing is not None:
            self._remove_edge(existing)
        self._put_edge(edge)

    async def get_node(self, node_id: NodeId) -> Node | None:
        data = self._graph.nodes.get(node_id)
        return None if data is None else _copy(data[_NODE])

    async def get_nodes(self, node_ids: Sequence[NodeId]) -> dict[NodeId, Node]:
        found: dict[NodeId, Node] = {}
        for node_id in node_ids:
            data = self._graph.nodes.get(node_id)
            if data is not None:
                found[node_id] = _copy(data[_NODE])
        return found

    async def get_edge(self, edge_id: EdgeId) -> Edge | None:
        edge = self._edge(edge_id)
        return None if edge is None else _copy(edge)

    async def neighbors(
        self,
        node_id: NodeId,
        *,
        direction: Direction = "out",
        edge_types: Collection[str] | None = None,
    ) -> list[Neighbor]:
        result = [
            Neighbor(edge=_copy(edge), node=_copy(self._node(other)), direction=dir_)
            for edge, other, dir_ in self._incident(node_id, direction, edge_types)
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
        return sum(1 for _ in self._incident(node_id, direction, edge_types))

    async def find_nodes(
        self,
        *,
        name: str | None = None,
        type: str | None = None,  # noqa: A002
        limit: int | None = None,
    ) -> list[Node]:
        wanted = None if name is None else name.casefold()
        matches: list[Node] = []
        for node_id in sorted(self._graph.nodes):
            node = self._node(node_id)
            if type is not None and node.type != type:
                continue
            if wanted is not None and wanted not in {
                n.casefold() for n in (node.name, *node.aliases)
            }:
                continue
            matches.append(_copy(node))
            if limit is not None and len(matches) >= limit:
                break
        return matches

    async def iter_nodes(self, *, batch_size: int = 1000) -> AsyncIterator[Node]:
        del batch_size  # everything is in memory
        for node_id in sorted(self._graph.nodes):
            yield _copy(self._node(node_id))

    async def iter_edges(self, *, batch_size: int = 1000) -> AsyncIterator[Edge]:
        del batch_size
        for edge_id in sorted(self._edge_index):
            edge = self._edge(edge_id)
            if edge is not None:
                yield _copy(edge)

    async def delete_node(self, node_id: NodeId) -> None:
        self._node(node_id)
        for edge, _, _ in list(self._incident(node_id, "both", None)):
            del self._edge_index[edge.id]
        self._graph.remove_node(node_id)

    async def delete_edge(self, edge_id: EdgeId) -> None:
        edge = self._edge(edge_id)
        if edge is None:
            raise EdgeNotFoundError(edge_id)
        self._remove_edge(edge)

    async def merge_nodes(self, keep: NodeId, absorb: NodeId) -> Node:
        if keep == absorb:
            msg = "cannot merge a node into itself"
            raise ValueError(msg)
        kept, absorbed = self._node(keep), self._node(absorb)
        merged = merge_node_data(kept, absorbed)
        rewired = [
            rewire_edge(edge, absorb, keep)
            for edge, _, _ in list(self._incident(absorb, "both", None))
        ]
        await self.delete_node(absorb)
        self._graph.nodes[keep][_NODE] = _copy(merged)
        for moved in rewired:
            if moved is None:
                continue
            existing = self._edge(moved.id)
            if existing is not None:
                self._remove_edge(existing)
            self._put_edge(moved if existing is None else merge_edge_data(existing, moved))
        return _copy(merged)

    async def get_metadata(self, key: str) -> JsonValue | None:
        return copy.deepcopy(self._metadata.get(key))

    async def set_metadata(self, key: str, value: JsonValue | None) -> None:
        if value is None:
            self._metadata.pop(key, None)
        else:
            self._metadata[key] = copy.deepcopy(value)

    async def counts(self) -> tuple[int, int]:
        return self._graph.number_of_nodes(), self._graph.number_of_edges()

    async def clear(self) -> None:
        self._graph.clear()
        self._edge_index.clear()
        self._metadata.clear()

    async def close(self) -> None:
        return None

    # ------------------------------------------------------------------ persistence

    def _to_document(self) -> dict[str, Any]:
        return {
            "format": FILE_FORMAT,
            "version": FILE_VERSION,
            "nodes": [self._node(n).model_dump(mode="json") for n in sorted(self._graph.nodes)],
            "edges": [
                edge.model_dump(mode="json")
                for edge_id in sorted(self._edge_index)
                if (edge := self._edge(edge_id)) is not None
            ],
            "metadata": copy.deepcopy(self._metadata),
        }

    async def save(self, path: str | os.PathLike[str]) -> None:
        """Write the graph to ``path`` as JSON, atomically (temp file + rename)."""
        document = self._to_document()
        await asyncio.to_thread(_write_json_atomic, Path(path), document)

    @classmethod
    async def load(cls, path: str | os.PathLike[str]) -> Self:
        """Read a graph written by :meth:`save`."""
        document = await asyncio.to_thread(_read_json, Path(path))
        if document.get("format") != FILE_FORMAT or document.get("version") != FILE_VERSION:
            msg = (
                f"{path}: not a {FILE_FORMAT} v{FILE_VERSION} file "
                f"(format={document.get('format')!r}, version={document.get('version')!r})"
            )
            raise ValueError(msg)
        store = cls()
        for raw in document["nodes"]:
            await store.upsert_node(Node.model_validate(raw))
        for raw in document["edges"]:
            await store.upsert_edge(Edge.model_validate(raw))
        store._metadata = dict(document.get("metadata") or {})  # absent in older files
        return store


def _copy[T: Node | Edge](model: T) -> T:
    """A snapshot that shares nothing mutable with ``model``.

    Models are frozen and their tuples hold frozen values, so only the dicts need fresh
    copies (and ``attributes`` a deep one, since JSON values nest). ~7x cheaper than
    ``model_copy(deep=True)``, which dominated traversal over hub nodes.
    """
    return model.model_copy(
        update={
            "attributes": copy.deepcopy(model.attributes),
            "conflicts": dict(model.conflicts),
            "attribute_provenance": dict(model.attribute_provenance),
        }
    )


def _write_json_atomic(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(document, fh, ensure_ascii=False, separators=(",", ":"))
        fh.flush()
        os.fsync(fh.fileno())
    tmp.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        loaded: dict[str, Any] = json.load(fh)
    return loaded
