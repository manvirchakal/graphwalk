"""Small constructors for test data."""

from datetime import UTC, datetime

from pydantic import JsonValue

from graphwalk.core.model import Edge, Node, Provenance

T0 = datetime(2026, 1, 1, tzinfo=UTC)


def prov(source: str = "src", *, confidence: float = 1.0, content: str = "h0") -> Provenance:
    return Provenance(
        source_id=source, ingested_at=T0, confidence=confidence, content_hash=f"sha256:{content}"
    )


def node(
    node_id: str,
    *,
    type_: str = "thing",
    name: str | None = None,
    source: str = "src",
    aliases: tuple[str, ...] = (),
    summary: str | None = None,
    **attributes: JsonValue,
) -> Node:
    return Node(
        id=node_id,
        type=type_,
        name=node_id if name is None else name,
        summary=summary,
        aliases=aliases,
        attributes=attributes,
        provenance=(prov(source),),
    )


def edge(
    source: str, type_: str, target: str, *, src: str = "src", **attributes: JsonValue
) -> Edge:
    return Edge(
        source=source, target=target, type=type_, attributes=attributes, provenance=(prov(src),)
    )
