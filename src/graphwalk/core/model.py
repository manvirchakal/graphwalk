"""Nodes, edges, and provenance.

Every node and edge carries at least one :class:`Provenance` record. Models are frozen:
derive changed copies with ``model_copy(update=...)``.
"""

import hashlib
from datetime import UTC, datetime
from typing import Any, Literal, Self, cast

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    computed_field,
    field_validator,
    model_validator,
)

type NodeId = str
type EdgeId = str
type JSONValue = JsonValue
type Direction = Literal["out", "in", "both"]

_FROZEN = ConfigDict(frozen=True, extra="forbid")


class Provenance(BaseModel):
    """Where a node, edge, or attribute value came from."""

    model_config = _FROZEN

    source_id: str = Field(min_length=1)
    ingested_at: AwareDatetime
    confidence: float = Field(ge=0.0, le=1.0)
    content_hash: str = Field(min_length=1)
    """Hash of the chunk or record this was derived from (see :func:`content_hash`)."""
    start: int | None = Field(default=None, ge=0)
    """Character offset in the source document where the supporting text begins."""
    end: int | None = Field(default=None, ge=0)
    """Offset just past the supporting text. ``start``/``end`` are both set or both
    ``None`` (records from before offsets were tracked, or with no text behind them)."""

    @model_validator(mode="after")
    def _span_is_valid(self) -> Self:
        if (self.start is None) != (self.end is None):
            msg = "start and end must both be set or both be None"
            raise ValueError(msg)
        if self.start is not None and self.end is not None and self.end < self.start:
            msg = f"end ({self.end}) is before start ({self.start})"
            raise ValueError(msg)
        return self


class ConflictingValue(BaseModel):
    """One of several disagreeing values for an attribute, with where it came from."""

    model_config = _FROZEN

    value: JsonValue
    provenance: tuple[Provenance, ...] = Field(min_length=1)


class AttributeConflict(BaseModel):
    """Two or more sources disagree on an attribute. Recorded as data, never resolved silently."""

    model_config = _FROZEN

    key: str
    values: tuple[ConflictingValue, ...] = Field(min_length=2)


class _GraphElement(BaseModel):
    model_config = _FROZEN

    attributes: dict[str, JsonValue] = Field(default_factory=dict[str, JsonValue])
    provenance: tuple[Provenance, ...] = Field(min_length=1)
    attribute_provenance: dict[str, tuple[Provenance, ...]] = Field(
        default_factory=dict[str, tuple[Provenance, ...]]
    )
    """Per-attribute sources. A key absent here is attributed to ``provenance``."""
    conflicts: dict[str, AttributeConflict] = Field(default_factory=dict[str, AttributeConflict])

    def sources_of(self, key: str) -> tuple[Provenance, ...]:
        """The provenance of attribute ``key``'s current value."""
        return self.attribute_provenance.get(key, self.provenance)

    @model_validator(mode="after")
    def _attribute_provenance_keys_exist(self) -> Self:
        unknown = self.attribute_provenance.keys() - self.attributes.keys()
        if unknown:
            msg = f"attribute_provenance has keys with no attribute: {sorted(unknown)}"
            raise ValueError(msg)
        return self

    @field_validator("conflicts")
    @classmethod
    def _conflict_keys_match(
        cls, conflicts: dict[str, AttributeConflict]
    ) -> dict[str, AttributeConflict]:
        for key, conflict in conflicts.items():
            if conflict.key != key:
                msg = f"conflict stored under {key!r} has key {conflict.key!r}"
                raise ValueError(msg)
        return conflicts


class Node(_GraphElement):
    """A typed entity."""

    id: NodeId = Field(min_length=1)
    type: str = Field(min_length=1)
    name: str
    summary: str | None = None
    aliases: tuple[str, ...] = ()
    """Names absorbed from merged duplicates."""


def edge_id(source: NodeId, type_: str, target: NodeId) -> EdgeId:
    """Deterministic edge id: one edge per ``(source, type, target)`` triple."""
    digest = hashlib.sha256("\x1f".join((source, type_, target)).encode("utf-8")).hexdigest()
    return f"e:{digest[:32]}"


class Edge(_GraphElement):
    """A typed, directed relation between two nodes.

    The id is derived from ``(source, type, target)``, so re-adding the same relation
    always addresses the same edge.
    """

    source: NodeId = Field(min_length=1)
    target: NodeId = Field(min_length=1)
    type: str = Field(min_length=1)

    @model_validator(mode="before")
    @classmethod
    def _accept_serialized_id(cls, data: Any) -> Any:
        # ``id`` is derived, but it is included in dumps; accept it back if it is consistent.
        if not isinstance(data, dict):
            return data
        fields = dict(cast("dict[str, Any]", data))
        if "id" not in fields:
            return fields
        given = fields.pop("id")
        expected = edge_id(
            str(fields.get("source")), str(fields.get("type")), str(fields.get("target"))
        )
        if given != expected:
            msg = f"edge id {given!r} does not match its endpoints; expected {expected!r}"
            raise ValueError(msg)
        return fields

    @computed_field
    @property
    def id(self) -> EdgeId:
        return edge_id(self.source, self.type, self.target)


class Neighbor(BaseModel):
    """An edge incident to some node, paired with the node at its other end."""

    model_config = _FROZEN

    edge: Edge
    node: Node
    direction: Literal["out", "in"]
    """``out``: ``edge.source`` is the queried node; ``in``: ``edge.target`` is."""


def utc_now() -> datetime:
    """Current time as an aware UTC datetime (for provenance timestamps)."""
    return datetime.now(UTC)


class StoredDocument(BaseModel):
    """A source document as ingested: what locations point into.

    ``key`` is the provenance ``source_id`` of everything derived from the document
    (``<source id>/<doc id>``). ``text`` is ``None`` when only the hash was kept; the
    text is then re-read from where it came from (see ``graphwalk.documents``).
    """

    model_config = _FROZEN

    key: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    doc_id: str = Field(min_length=1)
    title: str | None = None
    text: str | None = None
    text_hash: str = Field(min_length=1)
    """``content_hash(text)`` of the text at ingestion."""
    length: int = Field(ge=0)
    ingested_at: AwareDatetime
