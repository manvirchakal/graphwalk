"""The ``GraphStore`` protocol.

Every implementation must pass ``tests/contract/test_graph_store.py``. Behavior the
contract pins down, so traversal is reproducible across backends:

* ``upsert_*`` replaces the stored element with the given one (merging is explicit, via
  :meth:`GraphStore.merge_nodes`).
* Returned models are snapshots: mutating them never changes the store.
* ``neighbors`` is ordered by ``(direction, edge.type, node.id)`` with ``out`` before
  ``in``. With ``direction="both"`` a self-loop is reported once, as ``out``.
* ``iter_nodes`` yields nodes ordered by id; ``find_nodes`` returns them ordered by id.
* Metadata (``get_metadata``/``set_metadata``) is a small key -> JSON map kept with the
  graph, e.g. the ingestion ledger. It is persisted with the graph and removed by
  ``clear``.
"""

from collections.abc import AsyncIterator, Collection, Sequence
from typing import Protocol, runtime_checkable

from pydantic import JsonValue

from graphwalk.core.model import Direction, Edge, EdgeId, Neighbor, Node, NodeId


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
