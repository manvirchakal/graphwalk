"""Exception hierarchy."""


class GraphwalkError(Exception):
    """Base class for all graphwalk errors."""


class NodeNotFoundError(GraphwalkError, KeyError):
    """A node id that must exist does not."""

    def __init__(self, node_id: str) -> None:
        super().__init__(node_id)
        self.node_id = node_id

    def __str__(self) -> str:
        return f"node not found: {self.node_id!r}"


class EdgeNotFoundError(GraphwalkError, KeyError):
    """An edge id that must exist does not."""

    def __init__(self, edge_id: str) -> None:
        super().__init__(edge_id)
        self.edge_id = edge_id

    def __str__(self) -> str:
        return f"edge not found: {self.edge_id!r}"
