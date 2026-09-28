"""Traversal settings. One immutable object, recorded verbatim in every trace."""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from graphwalk.core.model import Direction

type Strategy = Literal["greedy", "beam", "sample"]
type HopMode = Literal["entity", "relation"]
type BeamBatching = Literal["per_depth", "per_beam"]
type LabelStyle = Literal["opaque", "readable"]


class Budget(BaseModel):
    """Per-query guardrails.

    Depth and call limits are checked *before* a call; the token limit *after* it (the
    count is only known from the response), so a query can overshoot ``max_input_tokens``
    by at most one depth's calls. ``None`` disables a limit.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_depth: int = Field(default=3, ge=1)
    """Maximum number of decision depths (hops, including the final STOP)."""
    max_decision_calls: int | None = Field(default=None, ge=1)
    max_input_tokens: int | None = Field(default=None, ge=1)


class TraversalConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    strategy: Strategy = "beam"
    beam_width: int = Field(default=3, ge=1)
    """Beams kept per depth. ``greedy`` forces 1."""
    hop_mode: HopMode = "entity"
    """``entity``: options are (edge, neighbor) pairs. ``relation``: options are distinct
    edge types, and choosing one moves to *all* its targets."""
    direction: Direction = "both"
    """Which edges count as moves. ``both`` lets a walk follow edges backwards, which most
    KG-QA needs (``Nolan <-directed_by- Inception``)."""
    exclude_visited: bool = True
    allow_stop_at_start: bool = True
    """Offer STOP before the first hop (the start node itself may answer)."""

    budget: Budget = Budget()

    # Scoring (see scoring.py)
    length_alpha: float = Field(default=1.0, ge=0.0)
    """Score = sum(log p) / L**alpha. 1.0 = mean log-prob (TypeSafe's geometric mean)."""
    prob_floor: float = Field(default=1e-3, gt=0.0, lt=1.0)
    """Probabilities are floored before ``log``. Jev rounds to 2 decimals, so a reported 0
    means < 0.005, not impossible."""

    # Sampling (strategy="sample")
    temperature: float = Field(default=1.0, ge=0.0)
    """``p ** (1/T)``, renormalized. 0 = argmax."""
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    n_samples: int = Field(default=5, ge=1)
    seed: int = 0

    # Batching
    beam_batching: BeamBatching = "per_depth"
    """``per_depth``: one call per depth with one question per beam (Jev shares ``state``
    per call). ``per_beam``: one concurrent call per beam."""
    max_request_tokens: int = Field(default=64_000, ge=1)
    max_state_plus_question_tokens: int = Field(default=32_000, ge=1)
    token_safety_margin: float = Field(default=0.2, ge=0.0, lt=1.0)
    """Batches are split when the *estimated* tokens exceed (1 - margin) x a limit."""

    # Prompt layout
    label_style: LabelStyle = "opaque"
    """``opaque``: labels o1..oN, meaning in descriptions. ``readable``: labels are the
    relation/node text. Which classifies better is an eval question."""
    include_summaries: bool = True

    # High-degree handling
    prefilter_threshold: int = Field(default=50, ge=1)
    """Above this many move options, keep only the ``prefilter_top_n`` most similar to the
    query by embedding. Needs an embedder; without one, options are truncated in store
    order and the trace says so."""
    prefilter_top_n: int = Field(default=30, ge=1)
    max_frontier: int = Field(default=50, ge=1)
    """Relation mode: a frontier larger than this is cut to the most query-similar nodes."""
    frontier_preview: int = Field(default=5, ge=0)
    """Relation mode: how many target names each relation option shows."""

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.prefilter_top_n > self.prefilter_threshold:
            msg = "prefilter_top_n must be <= prefilter_threshold"
            raise ValueError(msg)
        return self

    @property
    def width(self) -> int:
        """Effective number of parallel hypotheses."""
        if self.strategy == "greedy":
            return 1
        if self.strategy == "sample":
            return self.n_samples
        return self.beam_width
