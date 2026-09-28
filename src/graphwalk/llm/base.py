"""The ``LLMBackend`` protocol: chat completion with usage accounting."""

from collections.abc import Sequence
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from graphwalk.core.errors import GraphwalkError


class LLMError(GraphwalkError):
    """An LLM call failed."""


class Message(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    role: Literal["system", "user", "assistant"]
    content: str


class LLMResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    cost_usd: float | None = None
    """Only when the provider reports it. Never estimated."""
    latency_s: float = Field(ge=0.0)
    model: str


@runtime_checkable
class LLMBackend(Protocol):
    @property
    def model_id(self) -> str: ...

    async def complete(self, messages: Sequence[Message]) -> LLMResult: ...
