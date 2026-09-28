"""Traversal traces and results. Everything here is JSON-serializable."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from graphwalk.core.model import NodeId

_FROZEN = ConfigDict(frozen=True, extra="forbid")

type Status = Literal["ok", "aborted", "error"]
type Termination = Literal["stop", "dead_end", "max_depth", "aborted"]
type PruneReason = Literal["prefilter", "truncated", "frontier_prefilter", "frontier_truncated"]


class OptionInfo(BaseModel):
    """What one option label meant at one step."""

    model_config = _FROZEN

    label: str
    kind: Literal["move", "stop"]
    relation: str | None = None
    direction: Literal["out", "in"] | None = None
    targets: tuple[NodeId, ...] = ()
    """Entity mode: one node. Relation mode: every (unvisited) target of the relation."""
    description: JsonValue = None
    """Exactly what the decision backend was shown for this option."""


class Pruned(BaseModel):
    model_config = _FROZEN

    reason: PruneReason
    targets: tuple[NodeId, ...]
    relation: str | None = None
    similarity: float | None = None


class Call(BaseModel):
    """One decision-backend request. Latency and usage belong here, not to steps, because
    one call can serve several beams."""

    model_config = _FROZEN

    call_id: int
    depth: int
    n_questions: int
    latency_s: float
    input_tokens: int
    output_tokens: int
    cost_usd: float | None = None
    request_id: str | None = None
    model: str


class Step(BaseModel):
    """One decision (or forced move) for one beam at one depth."""

    model_config = _FROZEN

    depth: int
    beam_id: int
    parent_beam_id: int | None
    frontier: tuple[NodeId, ...]
    """Where the beam stood when deciding (one node in entity mode)."""
    options: tuple[OptionInfo, ...]
    pruned: tuple[Pruned, ...] = ()
    call_id: int | None
    """``None`` for a forced move (fewer than two legal options): no backend call."""
    question_key: str | None = None
    distribution: dict[str, float] = Field(default_factory=dict[str, float])
    """As returned by the backend (normalized)."""
    distribution_used: dict[str, float] = Field(default_factory=dict[str, float])
    """After temperature/top-p (sample strategy); equals ``distribution`` otherwise."""
    confidence: float | None = None
    chosen: tuple[str, ...]
    """Labels expanded into candidates: one for greedy/sample, every label for beam."""


class Hop(BaseModel):
    model_config = _FROZEN

    relation: str
    direction: Literal["out", "in"]
    targets: tuple[NodeId, ...]


class Answer(BaseModel):
    """One finished (or best partial) hypothesis."""

    model_config = _FROZEN

    beam_id: int
    node_ids: tuple[NodeId, ...]
    names: tuple[str, ...]
    start: tuple[NodeId, ...]
    path: tuple[Hop, ...]
    score: float
    """Length-normalized log-probability (higher is better, max 0)."""
    sum_logp: float
    n_decisions: int
    min_confidence: float | None
    """Lowest per-decision confidence along the path (``None`` if no decisions)."""
    terminated_by: Termination
    votes: int = 1
    """Sample strategy: how many walks ended on the same answer."""


class Totals(BaseModel):
    model_config = _FROZEN

    decision_calls: int = 0
    questions: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None
    decision_latency_s: float = 0.0
    """Sum of per-call latencies (concurrent calls overlap, so this can exceed wall time)."""
    wall_s: float = 0.0
    depth_reached: int = 0


class Trace(BaseModel):
    model_config = _FROZEN

    query: str
    config: dict[str, JsonValue]
    decision_model: str
    embedder_model: str | None
    start: tuple[NodeId, ...]
    steps: tuple[Step, ...]
    calls: tuple[Call, ...]
    totals: Totals
    answer_type: dict[str, float] | None = None
    """The speculative answer-type distribution, when asked."""


class TraversalResult(BaseModel):
    model_config = _FROZEN

    status: Status
    answers: tuple[Answer, ...]
    """Ranked best first. For ``aborted``/``error`` these are the best partial hypotheses."""
    abort_reason: str | None = None
    error: str | None = None
    trace: Trace

    @property
    def best(self) -> Answer | None:
        return self.answers[0] if self.answers else None
