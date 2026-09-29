"""Split document text into chunks small enough for one extraction call."""

import re

_PARAGRAPH = re.compile(r"\n\s*\n")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def _pieces(text: str, max_chars: int) -> list[str]:
    """Paragraphs; an over-long paragraph is split at sentences, then hard-wrapped."""
    out: list[str] = []
    for raw in _PARAGRAPH.split(text):
        paragraph = raw.strip()
        if not paragraph:
            continue
        if len(paragraph) <= max_chars:
            out.append(paragraph)
            continue
        for sentence in _SENTENCE.split(paragraph):
            out.extend(sentence[i : i + max_chars] for i in range(0, len(sentence), max_chars))
    return out


def chunk_text(text: str, *, max_chars: int = 2000) -> list[str]:
    """Greedily pack paragraphs into chunks of at most ``max_chars`` characters.

    Paragraph boundaries are kept where possible, so a chunk is a run of whole
    paragraphs unless one paragraph alone is too long.
    """
    if max_chars < 1:
        msg = f"max_chars must be positive, got {max_chars}"
        raise ValueError(msg)
    chunks: list[str] = []
    current = ""
    for piece in _pieces(text, max_chars):
        joined = f"{current}\n\n{piece}" if current else piece
        if len(joined) <= max_chars:
            current = joined
        else:
            chunks.append(current)
            current = piece
    if current:
        chunks.append(current)
    return chunks
