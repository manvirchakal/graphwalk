"""Local CPU embedder via fastembed (ONNX; the ``embeddings`` extra).

fastembed instead of sentence-transformers: the same BGE/MiniLM models without a
multi-gigabyte torch install. Models download from Hugging Face on first use.
"""

import asyncio
from collections.abc import Sequence
from typing import Any

import numpy as np

from graphwalk.embeddings.base import Vectors, l2_normalize

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


class FastEmbedEmbedder:
    def __init__(self, model_name: str = DEFAULT_MODEL, *, batch_size: int = 32) -> None:
        try:
            from fastembed import TextEmbedding  # noqa: PLC0415 - optional dependency
        except ImportError as error:  # pragma: no cover - depends on the environment
            msg = "FastEmbedEmbedder needs the 'embeddings' extra: uv sync --extra embeddings"
            raise ImportError(msg) from error
        self._model: Any = TextEmbedding(model_name)
        self._model_name = model_name
        self._batch_size = batch_size
        self._lock = asyncio.Lock()

    @property
    def model_id(self) -> str:
        return f"fastembed:{self._model_name}"

    def _embed_sync(self, texts: list[str]) -> Vectors:
        # Batches are padded to their longest text, so embed in length order (8x faster
        # on mixed-length documents in our measurements) and restore the input order.
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        rows: list[Any] = list(
            self._model.embed([texts[i] for i in order], batch_size=self._batch_size)
        )
        out = np.empty((len(texts), len(rows[0])), dtype=np.float32)
        for position, index in enumerate(order):
            out[index] = rows[position]
        return l2_normalize(out)

    async def embed(self, texts: Sequence[str]) -> Vectors:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        async with self._lock:  # the ONNX session is shared; one batch at a time
            return await asyncio.to_thread(self._embed_sync, list(texts))
