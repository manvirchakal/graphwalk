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


class DocumentNotFoundError(GraphwalkError, KeyError):
    """A location points at a document that is not available."""

    def __init__(self, key: str, reason: str = "not in the store") -> None:
        super().__init__(key)
        self.key = key
        self.reason = reason

    def __str__(self) -> str:
        return f"document {self.key!r}: {self.reason}"


class StaleLocationError(GraphwalkError):
    """The document changed since the location was produced, so its offsets may be wrong."""

    def __init__(self, key: str, expected: str, actual: str) -> None:
        super().__init__(key)
        self.key = key
        self.expected = expected
        self.actual = actual

    def __str__(self) -> str:
        return (
            f"document {self.key!r} changed since it was located "
            f"(hash {self.expected} -> {self.actual}); run locate again"
        )
