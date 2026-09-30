"""Remote embeddings from any OpenAI-compatible ``/embeddings`` endpoint (OpenAI,
OpenRouter), via LiteLLM (the ``llm`` extra)."""

from collections.abc import Sequence
from typing import Any, cast

import numpy as np

from graphwalk.core.errors import GraphwalkError
from graphwalk.core.redact import redact
from graphwalk.embeddings.base import Vectors, l2_normalize
from graphwalk.llm.litellm_import import import_litellm


class EmbeddingError(GraphwalkError):
    """An embedding request failed."""


class OpenAICompatibleEmbedder:
    def __init__(
        self,
        model: str,
        *,
        api_key: str,
        api_base: str,
        provider: str = "openai",
        batch_size: int = 128,
        timeout_s: float = 60.0,
    ) -> None:
        """``api_base`` includes the version path (``https://api.openai.com/v1``)."""
        self._litellm: Any = import_litellm()
        self._model = model
        self._api_key = api_key
        self._api_base = api_base
        self._provider = provider
        self._batch_size = batch_size
        self._timeout_s = timeout_s

    @property
    def model_id(self) -> str:
        return f"{self._provider}:{self._model}"

    async def embed(self, texts: Sequence[str]) -> Vectors:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        rows: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = list(texts[start : start + self._batch_size])
            try:
                # "openai/" routes to LiteLLM's OpenAI-compatible client at api_base.
                response: Any = await self._litellm.aembedding(
                    model=f"openai/{self._model}",
                    input=batch,
                    api_key=self._api_key,
                    api_base=self._api_base,
                    timeout=self._timeout_s,
                )
            except Exception as error:  # noqa: BLE001 - any provider error becomes ours
                detail = redact(str(error), [self._api_key])
                raise EmbeddingError(f"embedding request failed: {detail}") from None
            data = sorted(response.data, key=lambda item: _get(item, "index"))
            rows += [list(_get(item, "embedding")) for item in data]
        return l2_normalize(np.asarray(rows, dtype=np.float32))


def _get(item: Any, name: str) -> Any:
    if isinstance(item, dict):
        return cast("dict[str, Any]", item)[name]
    return getattr(item, name)
