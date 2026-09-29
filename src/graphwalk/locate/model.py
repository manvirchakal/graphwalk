"""What ``locate`` returns and ``read`` gives back."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

_FROZEN = ConfigDict(frozen=True, extra="forbid")

type Via = Literal["graph", "dense", "hybrid"]


class Location(BaseModel):
    """A span of a source document that likely answers (part of) a query.

    ``start``/``end`` are character offsets into the document text; both ``None`` means
    the whole document (graphs built before offsets were recorded). Pass the location
    to ``read`` for the text.
    """

    model_config = _FROZEN

    key: str = Field(min_length=1)
    """The document's key: ``<source id>/<doc id>``, as in provenance."""
    title: str | None = None
    start: int | None = Field(default=None, ge=0)
    end: int | None = Field(default=None, ge=0)
    doc_hash: str | None = None
    """Hash of the document text this location was computed against; ``read`` uses it
    to detect a document that changed since. ``None`` if the document is not stored."""
    snippet: str | None = None
    """The start of the located text, for display (see ``read`` for all of it)."""
    score: float = 0.0
    """Informational: path probability x provenance confidence (graph), cosine
    similarity (dense), or the fused reciprocal-rank score (hybrid). The list order is
    the ranking."""
    via: Via
    path: tuple[str, ...] = ()
    """The walk that reached this span, one hop per item (``A --rel--> B``)."""
    element: Literal["edge", "node", "chunk"]
    """What the span supports: a walked edge, a node, or (dense) a text chunk."""
    element_id: str | None = None

    def overlaps(self, other: "Location") -> bool:
        """Same document and intersecting spans (a whole-document span overlaps all)."""
        if self.key != other.key:
            return False
        a_start, a_end, b_start, b_end = self.start, self.end, other.start, other.end
        if a_start is None or a_end is None or b_start is None or b_end is None:
            return True
        return a_start < b_end and b_start < a_end


class Passage(BaseModel):
    """The text at a location, as read now."""

    model_config = _FROZEN

    location: Location
    text: str
    start: int = Field(ge=0)
    """Offset of ``text`` in the document (``location.start`` minus any context)."""
    end: int = Field(ge=0)
    title: str | None = None
    stale: bool = False
    """The document changed since the location was produced: the offsets may no longer
    point at the same text."""
