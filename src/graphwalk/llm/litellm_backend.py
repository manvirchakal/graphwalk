"""LLM backend via LiteLLM (the ``llm`` extra). Defaults to OpenRouter.

Cost is taken from the provider's ``usage.cost`` (OpenRouter reports it), never
estimated from a price table.
"""

import time
from collections.abc import Sequence
from typing import Any

from graphwalk.llm.base import LLMError, LLMResult, Message

DEFAULT_MODEL = "openrouter/openai/gpt-6-luna"


class LiteLLMBackend:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 256,
        timeout_s: float = 60.0,
        num_retries: int = 2,
    ) -> None:
        try:
            import litellm  # noqa: PLC0415 - optional, heavy dependency
        except ImportError as error:  # pragma: no cover - depends on the environment
            msg = "LiteLLMBackend needs the 'llm' extra: uv sync --extra llm"
            raise ImportError(msg) from error
        litellm.suppress_debug_info = True
        self._litellm: Any = litellm
        self._model = model
        self._api_key = api_key
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout_s = timeout_s
        self._num_retries = num_retries

    @property
    def model_id(self) -> str:
        return self._model

    async def complete(self, messages: Sequence[Message]) -> LLMResult:
        extra: dict[str, Any] = {}
        if self._model.startswith("openrouter/"):
            extra["usage"] = {"include": True}
        started = time.perf_counter()
        try:
            response: Any = await self._litellm.acompletion(
                model=self._model,
                messages=[m.model_dump() for m in messages],
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                timeout=self._timeout_s,
                num_retries=self._num_retries,
                api_key=self._api_key,
                **extra,
            )
        except Exception as error:
            raise LLMError(f"LLM call failed: {error}") from error
        latency = time.perf_counter() - started
        usage: Any = getattr(response, "usage", None)
        cost = getattr(usage, "cost", None)
        text = response.choices[0].message.content or ""
        return LLMResult(
            text=str(text),
            input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            cost_usd=float(cost) if isinstance(cost, int | float) else None,
            latency_s=latency,
            model=str(getattr(response, "model", self._model)),
        )
