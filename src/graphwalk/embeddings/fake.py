"""Deterministic hashing embedder for tests and offline development.

Each lowercase word is hashed to a signed bucket of a ``dim``-sized vector (the
"hashing trick"), so texts sharing words have positive cosine similarity. It has no
semantics beyond word overlap, which is exactly what tests need: predictable rankings.
"""

import hashlib
import re
from collections.abc import Sequence

import numpy as np

from graphwalk.embeddings.base import Vectors, l2_normalize

_WORD = re.compile(r"[a-z0-9]+")


class FakeEmbedder:
    def __init__(self, *, dim: int = 256, model_id: str = "fake-embedder-1") -> None:
        if dim < 1:
            msg = f"dim must be positive, got {dim}"
            raise ValueError(msg)
        self._dim = dim
        self._model_id = model_id
        self.calls: list[list[str]] = []

    @property
    def model_id(self) -> str:
        return self._model_id

    async def embed(self, texts: Sequence[str]) -> Vectors:
        self.calls.append(list(texts))
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in _WORD.findall(text.lower()):
                digest = hashlib.blake2b(word.encode("utf-8"), digest_size=8).digest()
                value = int.from_bytes(digest, "big")
                sign = 1.0 if value & 1 else -1.0
                out[row, (value >> 1) % self._dim] += sign
        return l2_normalize(out)
