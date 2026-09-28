"""Core graph data model."""

from graphwalk.core.errors import EdgeNotFoundError, GraphwalkError, NodeNotFoundError
from graphwalk.core.hashing import content_hash
from graphwalk.core.model import (
    AttributeConflict,
    ConflictingValue,
    Direction,
    Edge,
    EdgeId,
    JSONValue,
    Neighbor,
    Node,
    NodeId,
    Provenance,
    edge_id,
)

__all__ = [
    "AttributeConflict",
    "ConflictingValue",
    "Direction",
    "Edge",
    "EdgeId",
    "EdgeNotFoundError",
    "GraphwalkError",
    "JSONValue",
    "Neighbor",
    "Node",
    "NodeId",
    "NodeNotFoundError",
    "Provenance",
    "content_hash",
    "edge_id",
]
