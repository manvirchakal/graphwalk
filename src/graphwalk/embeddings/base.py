"""The ``Embedder`` protocol: text -> L2-normalized vectors."""

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

type Vectors = NDArray[np.float32]
"""Shape ``(n, d)``; every row has unit L2 norm (or is all zeros for empty text)."""


@runtime_checkable
class Embedder(Protocol):
    @property
    def model_id(self) -> str:
        """Recorded in traces and eval runs."""
        ...

    async def embed(self, texts: Sequence[str]) -> Vectors:
        """Embed ``texts`` in order. Rows are L2-normalized, so dot product = cosine."""
        ...


def l2_normalize(vectors: Vectors) -> Vectors:
    """Scale each row to unit length; all-zero rows stay zero."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms > 0)
