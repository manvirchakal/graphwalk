"""Run a scripted decision backend *through* the LLM-as-decider.

``ViaLLM(fake)`` is an :class:`LLMDecider` whose chat model answers each prompt by
parsing it back into the ``DecisionRequest`` and asking ``fake``. If the decider's
prompt, parsing, and normalization are faithful, a test gets the same distributions
as with ``fake`` directly, which is how the traversal suite checks the fallback.
"""

import json
from collections.abc import Sequence
from typing import Any

from graphwalk.decisions import ChoiceQuestion, DecisionBackend, DecisionRequest
from graphwalk.decisions.llm_decider import LLMDecider
from graphwalk.llm import LLMResult, Message


class DeciderAsLLM:
    def __init__(self, inner: DecisionBackend) -> None:
        self.inner = inner
        self.calls: list[list[Message]] = []

    @property
    def model_id(self) -> str:
        return f"scripted-{self.inner.model_id}"

    async def complete(self, messages: Sequence[Message]) -> LLMResult:
        self.calls.append(list(messages))
        payload: dict[str, Any] = json.loads(messages[1].content)
        request = DecisionRequest(
            state=payload["state"],
            questions=tuple(ChoiceQuestion.model_validate(q) for q in payload["questions"]),
        )
        response = await self.inner.decide(request)
        scores = {key: result.probabilities for key, result in response.results.items()}
        return LLMResult(
            text=json.dumps(scores),
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cost_usd=response.usage.cost_usd,
            latency_s=response.latency_s,
            model=response.model,
        )


class ViaLLM(LLMDecider):
    """An LLM decider over ``inner``; other attributes (``calls``, ``requests``, ...)
    are the inner backend's, so tests can inspect it as before."""

    def __init__(self, inner: DecisionBackend) -> None:
        super().__init__(DeciderAsLLM(inner), max_options=inner.max_options)
        self.inner = inner

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)
