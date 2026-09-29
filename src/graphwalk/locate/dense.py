"""Dense ``locate`` (embedding similarity over text chunks), and hybrid fusion.

The dense index is built lazily from the stored documents: each is chunked with
offsets, the chunks are embedded once, and a query costs one embedding plus a dot
product. Call :meth:`DenseLocator.refresh` after ingesting.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from graphwalk.core.errors import DocumentNotFoundError
from graphwalk.embeddings.base import Embedder, Vectors
from graphwalk.ingest.chunking import chunk_spans
from graphwalk.locate.documents import DocumentSource
from graphwalk.locate.graph import describe
from graphwalk.locate.model import Location
from graphwalk.stores.base import DocumentStore

RRF_K = 60
"""The usual reciprocal-rank-fusion constant (Cormack et al., 2009)."""


@dataclass(frozen=True)
class _Chunk:
    key: str
    start: int
    end: int


class DenseLocator:
    def __init__(
        self,
        store: DocumentStore,
        documents: DocumentSource,
        embedder: Embedder,
        *,
        chunk_chars: int = 800,
        batch_size: int = 64,
    ) -> None:
        self._store = store
        self._documents = documents
        self._embedder = embedder
        self._chunk_chars = chunk_chars
        self._batch_size = batch_size
        self._chunks: list[_Chunk] | None = None
        self._vectors: Vectors | None = None

    def refresh(self) -> None:
        """Drop the index; it is rebuilt on the next query."""
        self._chunks = None
        self._vectors = None

    async def _build(self) -> tuple[list[_Chunk], Vectors]:
        if self._chunks is not None and self._vectors is not None:
            return self._chunks, self._vectors
        chunks: list[_Chunk] = []
        texts: list[str] = []
        async for record in self._store.iter_documents():
            try:
                found = await self._documents.fetch(record.key)
            except DocumentNotFoundError:
                continue
            for chunk in chunk_spans(found.text, max_chars=self._chunk_chars):
                chunks.append(_Chunk(record.key, chunk.start, chunk.end))
                texts.append(chunk.text)
        parts = [
            await self._embedder.embed(texts[i : i + self._batch_size])
            for i in range(0, len(texts), self._batch_size)
        ]
        vectors = np.concatenate(parts) if parts else np.zeros((0, 1), dtype=np.float32)
        self._chunks, self._vectors = chunks, vectors
        return chunks, vectors

    async def locate(self, query: str, k: int = 5) -> list[Location]:
        chunks, vectors = await self._build()
        if not chunks:
            return []
        query_vector = (await self._embedder.embed([query]))[0]
        scores = vectors @ query_vector
        order = sorted(range(len(chunks)), key=lambda i: (-float(scores[i]), i))[:k]
        return [
            await describe(
                self._documents,
                Location(
                    key=chunks[i].key,
                    start=chunks[i].start,
                    end=chunks[i].end,
                    score=float(scores[i]),
                    via="dense",
                    element="chunk",
                ),
            )
            for i in order
        ]


def fuse(rankings: Sequence[Sequence[Location]], k: int, *, rrf_k: int = RRF_K) -> list[Location]:
    """Reciprocal rank fusion of several ranked lists.

    Overlapping locations (same document, intersecting spans) count as the same item: a
    location joins the best-ranked group it overlaps, and each group is represented by
    its first member from the earliest list (the graph's sentence rather than the dense
    chunk, when the graph list comes first). Each list scores a group at most once.
    """
    groups: list[tuple[Location, float]] = []
    for ranking in rankings:
        credited: set[int] = set()
        for rank, location in enumerate(ranking):
            index = next((g for g, (head, _) in enumerate(groups) if head.overlaps(location)), None)
            if index is None:
                groups.append((location, 1.0 / (rrf_k + rank + 1)))
                credited.add(len(groups) - 1)
            elif index not in credited:
                head, score = groups[index]
                groups[index] = (head, score + 1.0 / (rrf_k + rank + 1))
                credited.add(index)
    order = sorted(range(len(groups)), key=lambda g: (-groups[g][1], g))[:k]
    return [groups[g][0].model_copy(update={"score": groups[g][1], "via": "hybrid"}) for g in order]
