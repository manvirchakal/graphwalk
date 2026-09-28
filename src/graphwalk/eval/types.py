"""Eval questions, system answers, and the ``QASystem`` adapter protocol."""

from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from graphwalk.core.model import NodeId

_FROZEN = ConfigDict(frozen=True, extra="forbid")

type AnswerKind = Literal["set", "text"]
"""``set``: gold is a set of entity names (MetaQA). ``text``: one answer string (2Wiki)."""


class EvalQuestion(BaseModel):
    model_config = _FROZEN

    id: str
    dataset: str
    question: str
    answers: tuple[str, ...] = Field(min_length=1)
    kind: AnswerKind
    start: tuple[NodeId, ...] | None = None
    """Given entry nodes (MetaQA). ``None``: the system must resolve them."""
    gold_start: tuple[NodeId, ...] | None = None
    """Known entry nodes, for scoring entity linking separately from walking."""
    meta: dict[str, JsonValue] = Field(default_factory=dict[str, JsonValue])


class SystemAnswer(BaseModel):
    model_config = _FROZEN

    answers: tuple[str, ...]
    """Ranked answer strings; ``answers[0]`` is the top-1 prediction."""
    answer_set: tuple[str, ...]
    """What the system asserts is the complete answer (for set EM/F1)."""
    status: Literal["ok", "aborted", "error", "no_entry"] = "ok"
    error: str | None = None
    latency_s: float = 0.0
    decision_calls: int = 0
    llm_calls: int = 0
    embed_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None
    start: tuple[NodeId, ...] | None = None
    """Entry nodes actually used."""
    detail: dict[str, JsonValue] = Field(default_factory=dict[str, JsonValue])
    """System-specific extras (graphwalk: path, confidence, trace summary)."""


@runtime_checkable
class QASystem(Protocol):
    @property
    def name(self) -> str: ...

    def describe(self) -> dict[str, JsonValue]:
        """Config and model ids, recorded with every run."""
        ...

    async def answer(self, question: EvalQuestion) -> SystemAnswer: ...
