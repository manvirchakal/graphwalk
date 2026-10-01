"""The embedding cache behind the prefilter."""

import numpy as np

from graphwalk.embeddings import FakeEmbedder
from graphwalk.traversal.prefilter import EmbeddingCache, rank_by_similarity


async def test_eviction_never_drops_rows_from_the_result() -> None:
    # Regression: a call that both hit the cache and overflowed it lost the hits (the
    # cache was cleared before the result was assembled), so rows and texts misaligned.
    embedder = FakeEmbedder()
    cache = EmbeddingCache(embedder, max_entries=3)
    await cache.embed(["a", "b"])
    texts = ["a", "b", "c", "d"]
    vectors = await cache.embed(texts)
    assert vectors.shape[0] == len(texts)
    expected = await FakeEmbedder().embed(texts)
    assert np.allclose(vectors, expected)


async def test_rank_by_similarity_after_overflow() -> None:
    cache = EmbeddingCache(FakeEmbedder(), max_entries=2)
    texts = [f"option {i}" for i in range(10)]
    await cache.embed(texts[:2])
    ranked = await rank_by_similarity(cache, "option 3", texts)
    assert sorted(i for i, _ in ranked) == list(range(10))
    assert ranked[0][0] == 3


async def test_repeated_texts_and_empty() -> None:
    cache = EmbeddingCache(FakeEmbedder())
    assert (await cache.embed(["x", "x", "y"])).shape[0] == 3
    assert (await cache.embed([])).shape == (0, 0)
