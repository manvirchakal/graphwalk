"""FastEmbedEmbedder keeps input order despite length-sorted batching (model stubbed)."""

import asyncio
import threading
from typing import Any

import numpy as np
import pytest

pytest.importorskip("fastembed")

from graphwalk.embeddings.fastembed_embedder import FastEmbedEmbedder


class _StubModel:
    def embed(self, texts: list[str], batch_size: int) -> Any:
        del batch_size
        return [np.array([len(t), 1.0], dtype=np.float32) for t in texts]


async def test_order_is_preserved() -> None:
    embedder = object.__new__(FastEmbedEmbedder)  # skip loading a real model
    embedder._model = _StubModel()
    embedder._model_name = "stub"
    embedder._batch_size = 2
    embedder._lock = threading.Lock()
    vectors = await embedder.embed(["ccc", "a", "bb"])
    expected = np.array([[3, 1], [1, 1], [2, 1]], dtype=np.float32)
    expected /= np.linalg.norm(expected, axis=1, keepdims=True)
    assert np.allclose(vectors, expected)
    assert embedder.model_id == "fastembed:stub"


def test_usable_from_several_event_loops() -> None:
    embedder = object.__new__(FastEmbedEmbedder)
    embedder._model = _StubModel()
    embedder._model_name = "stub"
    embedder._batch_size = 2
    embedder._lock = threading.Lock()
    for _ in range(2):  # the eval CLI once failed here with a loop-bound asyncio.Lock
        assert asyncio.run(embedder.embed(["a", "bb"])).shape == (2, 2)
