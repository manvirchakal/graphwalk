"""Embedding prefilter: keep the options most similar to the query."""

from collections.abc import Sequence

import numpy as np

from graphwalk.embeddings.base import Embedder, Vectors


class EmbeddingCache:
    """Caches vectors by text for one embedder. Texts include node content, so a changed
    node gets a new entry. Cleared wholesale when it grows past ``max_entries``."""

    def __init__(self, embedder: Embedder, *, max_entries: int = 100_000) -> None:
        self.embedder = embedder
        self._max_entries = max_entries
        self._vectors: dict[str, np.ndarray] = {}

    async def embed(self, texts: Sequence[str]) -> Vectors:
        missing = list(dict.fromkeys(t for t in texts if t not in self._vectors))
        if missing:
            if len(self._vectors) + len(missing) > self._max_entries:
                self._vectors.clear()
            fresh = await self.embedder.embed(missing)
            for text, vector in zip(missing, fresh, strict=True):
                self._vectors[text] = vector
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        return np.stack([self._vectors[t] for t in texts]).astype(np.float32, copy=False)


async def rank_by_similarity(
    cache: EmbeddingCache, query: str, texts: Sequence[str]
) -> list[tuple[int, float]]:
    """Indices of ``texts`` with cosine similarity to ``query``, best first; ties keep order."""
    if not texts:
        return []
    vectors = await cache.embed([query, *texts])
    sims = vectors[1:] @ vectors[0]
    order = sorted(range(len(texts)), key=lambda i: (-float(sims[i]), i))
    return [(i, float(sims[i])) for i in order]
