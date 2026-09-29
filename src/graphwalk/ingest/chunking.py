"""Split document text into chunks small enough for one extraction call.

Chunks remember where their text came from: :func:`chunk_spans` returns
:class:`TextChunk` s whose pieces map back to character offsets in the document, so
provenance can point at the exact span an entity or relation was read from.
"""

import re
from dataclasses import dataclass

_PARAGRAPH = re.compile(r"\n\s*\n")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_JOIN = "\n\n"


@dataclass(frozen=True)
class Piece:
    """A run of chunk text copied verbatim from the document."""

    chunk_start: int
    """Offset in the chunk's text."""
    doc_start: int
    """Offset in the document's text."""
    length: int


@dataclass(frozen=True)
class TextChunk:
    """One chunk. ``text`` is its pieces joined by blank lines; each piece is an exact
    substring of the document, so offsets in ``text`` map back to the document."""

    text: str
    pieces: tuple[Piece, ...]

    @property
    def start(self) -> int:
        """Document offset where the chunk begins."""
        return self.pieces[0].doc_start

    @property
    def end(self) -> int:
        """Document offset just past the chunk's last character."""
        last = self.pieces[-1]
        return last.doc_start + last.length

    def to_doc(self, start: int, end: int) -> tuple[int, int]:
        """Map the chunk-text span ``[start, end)`` to a document span.

        An endpoint inside a joining blank line snaps to the nearest piece.
        """
        if not 0 <= start < end <= len(self.text):
            msg = f"span [{start}, {end}) is outside the chunk (length {len(self.text)})"
            raise ValueError(msg)
        first = next(p for p in self.pieces if start < p.chunk_start + p.length)
        last = next(p for p in reversed(self.pieces) if end > p.chunk_start)
        doc_start = first.doc_start + max(0, start - first.chunk_start)
        doc_end = last.doc_start + min(last.length, end - last.chunk_start)
        return doc_start, doc_end


def _spans(pattern: re.Pattern[str], text: str, offset: int) -> list[tuple[int, int]]:
    """``pattern.split(text)`` as document spans (``offset`` = where ``text`` starts)."""
    spans: list[tuple[int, int]] = []
    at = 0
    for match in pattern.finditer(text):
        spans.append((offset + at, offset + match.start()))
        at = match.end()
    spans.append((offset + at, offset + len(text)))
    return spans


def _strip(text: str, start: int, end: int) -> tuple[int, int]:
    raw = text[start:end]
    stripped = raw.strip()
    if not stripped:
        return start, start
    lead = len(raw) - len(raw.lstrip())
    return start + lead, start + lead + len(stripped)


def _pieces(text: str, max_chars: int) -> list[tuple[int, int]]:
    """Paragraphs; an over-long paragraph is split at sentences, then hard-wrapped."""
    out: list[tuple[int, int]] = []
    for raw_start, raw_end in _spans(_PARAGRAPH, text, 0):
        start, end = _strip(text, raw_start, raw_end)
        if start == end:
            continue
        if end - start <= max_chars:
            out.append((start, end))
            continue
        for s_start, s_end in _spans(_SENTENCE, text[start:end], start):
            out.extend((i, min(i + max_chars, s_end)) for i in range(s_start, s_end, max_chars))
    return out


def chunk_spans(text: str, *, max_chars: int = 2000) -> list[TextChunk]:
    """Greedily pack paragraphs into chunks of at most ``max_chars`` characters.

    Paragraph boundaries are kept where possible, so a chunk is a run of whole
    paragraphs unless one paragraph alone is too long.
    """
    if max_chars < 1:
        msg = f"max_chars must be positive, got {max_chars}"
        raise ValueError(msg)
    chunks: list[TextChunk] = []
    current: list[tuple[int, int]] = []
    size = 0
    for start, end in _pieces(text, max_chars):
        length = end - start
        joined = size + len(_JOIN) + length if current else length
        if joined <= max_chars:
            current.append((start, end))
            size = joined
        else:
            chunks.append(_chunk(text, current))
            current, size = [(start, end)], length
    if current:
        chunks.append(_chunk(text, current))
    return chunks


def _chunk(text: str, spans: list[tuple[int, int]]) -> TextChunk:
    pieces: list[Piece] = []
    at = 0
    for start, end in spans:
        if pieces:
            at += len(_JOIN)
        pieces.append(Piece(chunk_start=at, doc_start=start, length=end - start))
        at += end - start
    return TextChunk(text=_JOIN.join(text[s:e] for s, e in spans), pieces=tuple(pieces))


def chunk_text(text: str, *, max_chars: int = 2000) -> list[str]:
    """The texts of :func:`chunk_spans`."""
    return [chunk.text for chunk in chunk_spans(text, max_chars=max_chars)]
