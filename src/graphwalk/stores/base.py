"""The ``GraphStore`` protocol.

Every implementation must pass ``tests/contract/test_graph_store.py``. Behavior the
contract pins down, so traversal is reproducible across backends:

* ``upsert_*`` replaces the stored element with the given one (merging is explicit, via
  :meth:`GraphStore.merge_nodes`).
* Returned models are snapshots: mutating them never changes the store.
* ``neighbors`` is ordered by ``(direction, edge.type, node.id)`` with ``out`` before
  ``in``. With ``direction="both"`` a self-loop is reported once, as ``out``.
* ``iter_nodes`` yields nodes ordered by id; ``find_nodes`` returns them ordered by id.
* ``clear`` also removes stored documents (see :class:`DocumentStore`).
* Metadata (``get_metadata``/``set_metadata``) is a small key -> JSON map kept with the
  graph, e.g. the ingestion ledger. It is persisted with the graph and removed by
  ``clear``.
"""

from collections.abc import AsyncIterator, Collection, Sequence
from typing import Literal, NamedTuple, Protocol, runtime_checkable

from pydantic import JsonValue

from graphwalk.core.model import Direction, Edge, EdgeId, Neighbor, Node, NodeId, StoredDocument


@runtime_checkable
class GraphStore(Protocol):
    async def upsert_node(self, node: Node) -> None:
        """Insert ``node``, or replace the stored node with the same id."""
        ...

    async def upsert_edge(self, edge: Edge) -> None:
        """Insert ``edge``, or replace the stored edge with the same id.

        Raises :class:`~graphwalk.core.errors.NodeNotFoundError` if an endpoint is missing.
        """
        ...

    async def get_node(self, node_id: NodeId) -> Node | None: ...

    async def get_nodes(self, node_ids: Sequence[NodeId]) -> dict[NodeId, Node]:
        """Fetch several nodes at once; missing ids are absent from the result."""
        ...

    async def get_edge(self, edge_id: EdgeId) -> Edge | None: ...

    async def neighbors(
        self,
        node_id: NodeId,
        *,
        direction: Direction = "out",
        edge_types: Collection[str] | None = None,
    ) -> list[Neighbor]:
        """Edges incident to ``node_id`` with the node at the other end.

        Raises :class:`~graphwalk.core.errors.NodeNotFoundError` if the node is missing.
        """
        ...

    async def degree(
        self,
        node_id: NodeId,
        *,
        direction: Direction = "out",
        edge_types: Collection[str] | None = None,
    ) -> int:
        """``len(await neighbors(...))`` with the same arguments, without loading nodes."""
        ...

    async def find_nodes(
        self,
        *,
        name: str | None = None,
        type: str | None = None,  # noqa: A002 - mirrors Node.type
        limit: int | None = None,
    ) -> list[Node]:
        """Nodes matching every given filter.

        ``name`` matches ``Node.name`` or any alias exactly, ignoring case.
        """
        ...

    def iter_nodes(self, *, batch_size: int = 1000) -> AsyncIterator[Node]: ...

    def iter_edges(self, *, batch_size: int = 1000) -> AsyncIterator[Edge]:
        """Yield every edge, ordered by id."""
        ...

    async def delete_node(self, node_id: NodeId) -> None:
        """Delete a node and all its incident edges. Raises if the node is missing."""
        ...

    async def delete_edge(self, edge_id: EdgeId) -> None:
        """Delete an edge. Raises :class:`~graphwalk.core.errors.EdgeNotFoundError`."""
        ...

    async def merge_nodes(self, keep: NodeId, absorb: NodeId) -> Node:
        """Fold ``absorb`` into ``keep`` and delete ``absorb``.

        Uses :func:`graphwalk.core.merge.merge_node_data`: provenance is unioned,
        ``absorb``'s names become aliases, and disagreeing attributes are recorded as
        conflicts. ``absorb``'s edges are re-pointed at ``keep``; edges that ran between
        the two are dropped, and re-pointed edges that collide with an existing edge are
        merged with :func:`~graphwalk.core.merge.merge_edge_data`. Returns the merged node.
        """
        ...

    async def get_metadata(self, key: str) -> JsonValue | None:
        """The metadata value stored under ``key``, or ``None``."""
        ...

    async def set_metadata(self, key: str, value: JsonValue | None) -> None:
        """Store ``value`` under ``key``; ``None`` deletes the key."""
        ...

    async def counts(self) -> tuple[int, int]:
        """``(number of nodes, number of edges)``."""
        ...

    async def clear(self) -> None:
        """Remove everything."""
        ...

    async def close(self) -> None: ...


class Adjacent(NamedTuple):
    """One incident edge, without its models: what traversal needs to list options."""

    relation: str
    direction: Literal["out", "in"]
    other: NodeId


class NodeLabel(NamedTuple):
    """The fields of a node that option ranking reads."""

    name: str
    type: str
    summary: str | None


@runtime_checkable
class AdjacencyStore(Protocol):
    """Optional fast path for traversal over high-degree nodes.

    A hub with 20k edges costs seconds to load as models; traversal needs only the
    edge types and endpoint ids to list its options, and full nodes only for the options
    it keeps. Stores that implement this are walked through it (same results, same
    order: ``tests/unit/test_adjacency.py`` checks parity with ``neighbors``).
    """

    async def adjacency(self, node_id: NodeId, *, direction: Direction) -> list[Adjacent]:
        """``neighbors(node_id, direction=...)`` as ``(edge type, direction, other id)``,
        in any order (the caller sorts). Raises :class:`NodeNotFoundError` like it."""
        ...

    async def node_labels(self, node_ids: Sequence[NodeId]) -> dict[NodeId, NodeLabel]:
        """Name, type, and summary of each existing node in ``node_ids``."""
        ...


@runtime_checkable
class DocumentStore(Protocol):
    """Source documents kept next to the graph, so locations can be read back.

    Optional for a :class:`GraphStore`: ingestion records documents only in stores that
    implement this too. Contract: ``tests/contract/test_document_store.py``.
    """

    async def put_document(self, document: StoredDocument) -> None:
        """Insert ``document``, or replace the stored one with the same key."""
        ...

    async def get_document(self, key: str) -> StoredDocument | None: ...

    async def delete_document(self, key: str) -> None:
        """Delete the document under ``key``; a missing key is not an error."""
        ...

    def iter_documents(self) -> AsyncIterator[StoredDocument]:
        """Yield every document, ordered by key."""
        ...
