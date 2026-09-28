"""Scriptable LLM for tests."""

from collections.abc import Callable, Sequence

from graphwalk.llm.base import LLMResult, Message


class FakeLLM:
    def __init__(
        self,
        respond: Callable[[Sequence[Message]], str] = lambda _messages: "unknown",
        *,
        model_id: str = "fake-llm-1",
        cost_per_call: float | None = None,
    ) -> None:
        self._respond = respond
        self._model_id = model_id
        self._cost = cost_per_call
        self.calls: list[list[Message]] = []

    @property
    def model_id(self) -> str:
        return self._model_id

    async def complete(self, messages: Sequence[Message]) -> LLMResult:
        self.calls.append(list(messages))
        text = self._respond(messages)
        chars = sum(len(m.content) for m in messages)
        return LLMResult(
            text=text,
            input_tokens=max(1, chars // 4),
            output_tokens=max(1, len(text) // 4),
            cost_usd=self._cost,
            latency_s=0.01,
            model=self._model_id,
        )
