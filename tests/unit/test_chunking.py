"""Chunk offsets: every chunk maps back to the exact document text it came from."""

import random
import re

import pytest

from graphwalk.ingest.chunking import chunk_spans, chunk_text

_PARAGRAPH = re.compile(r"\n\s*\n")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def legacy_chunk_text(text: str, max_chars: int) -> list[str]:
    """The pre-offset implementation: chunk texts (and so extraction cache keys) must not
    change."""
    pieces: list[str] = []
    for raw in _PARAGRAPH.split(text):
        paragraph = raw.strip()
        if not paragraph:
            continue
        if len(paragraph) <= max_chars:
            pieces.append(paragraph)
            continue
        for sentence in _SENTENCE.split(paragraph):
            pieces.extend(sentence[i : i + max_chars] for i in range(0, len(sentence), max_chars))
    chunks: list[str] = []
    current = ""
    for piece in pieces:
        joined = f"{current}\n\n{piece}" if current else piece
        if len(joined) <= max_chars:
            current = joined
        else:
            chunks.append(current)
            current = piece
    if current:
        chunks.append(current)
    return chunks


def random_text(rng: random.Random) -> str:
    words = ["alpha", "beta.", "gamma!", "delta?", "  ", "\n", "\n\n", " \n \n", "é", "x" * 30]
    return "".join(rng.choice(words) + rng.choice(["", " "]) for _ in range(rng.randint(0, 120)))


@pytest.mark.parametrize("seed", range(200))
def test_chunks_match_legacy_and_map_back(seed: int) -> None:
    rng = random.Random(seed)
    text = random_text(rng)
    max_chars = rng.choice([5, 20, 60, 200, 2000])
    chunks = chunk_spans(text, max_chars=max_chars)
    assert [c.text for c in chunks] == legacy_chunk_text(text, max_chars)
    assert chunk_text(text, max_chars=max_chars) == legacy_chunk_text(text, max_chars)
    for chunk in chunks:
        for piece in chunk.pieces:
            in_chunk = chunk.text[piece.chunk_start : piece.chunk_start + piece.length]
            assert in_chunk == text[piece.doc_start : piece.doc_start + piece.length]
        assert chunk.to_doc(0, len(chunk.text)) == (chunk.start, chunk.end)


def test_to_doc_maps_spans_inside_and_across_pieces() -> None:
    text = "  First para here.\n\n\n   Second one.  "
    (chunk,) = chunk_spans(text)
    assert chunk.text == "First para here.\n\nSecond one."
    at = chunk.text.index("para")
    start, end = chunk.to_doc(at, at + 4)
    assert text[start:end] == "para"
    at = chunk.text.index("Second")
    assert text[slice(*chunk.to_doc(at, at + 6))] == "Second"
    # a span across the joining blank line covers both pieces in the document
    start, end = chunk.to_doc(chunk.text.index("here"), chunk.text.index(" one"))
    assert text[start:end] == "here.\n\n\n   Second"
    with pytest.raises(ValueError, match="outside"):
        chunk.to_doc(3, 3)
