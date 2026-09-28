"""FastEmbedEmbedder keeps input order despite length-sorted batching (model stubbed)."""

import asyncio
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
    embedder._lock = asyncio.Lock()
    vectors = await embedder.embed(["ccc", "a", "bb"])
    expected = np.array([[3, 1], [1, 1], [2, 1]], dtype=np.float32)
    expected /= np.linalg.norm(expected, axis=1, keepdims=True)
    assert np.allclose(vectors, expected)
    assert embedder.model_id == "fastembed:stub"
