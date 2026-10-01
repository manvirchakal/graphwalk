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
    show_types: bool = False
    """Relation mode: show the current nodes' types and each relation's target types."""
    stop_style: Literal["v1", "literal"] = "v1"
    """``literal`` states the STOP condition first ("if the current nodes are the things the
    query asks for, choose STOP")."""
    relation_glosses: dict[str, str] = Field(default_factory=dict[str, str])
    """Optional plain-language meaning per relation name (graph schema documentation)."""
    answer_type: Literal["off", "hint", "gate"] = "off"
    """Ask Jev once, batched into the first call, what node type the query asks for.
    ``hint`` shows the predicted type in later questions; ``gate`` also withholds STOP
    from beams whose current nodes have none of the predicted type (when its probability
    is at least ``answer_type_gate_min_p``). Needs ``Traverser(node_types=...)``."""
    answer_type_gate_min_p: float = Field(default=0.6, ge=0.0, le=1.0)

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

    @classmethod
    def kgqa(cls, **overrides: object) -> Self:
        """The settings measured on curated knowledge graphs (MetaQA, the 2Wiki gold
        graph, Freebase subgraphs): greedy relation hops with typed options, a literal
        STOP, and an answer-type hint (pass the graph's node types to ``Index`` or
        ``Traverser`` for the hint). The eval suite's ``relation-v2`` preset, without
        dataset glosses; pass ``relation_glosses=`` if your schema has them."""
        settings: dict[str, object] = {
            "strategy": "greedy",
            "hop_mode": "relation",
            "show_types": True,
            "stop_style": "literal",
            "answer_type": "hint",
            "allow_stop_at_start": False,
            "max_frontier": 2000,
            "budget": {"max_depth": 4, "max_decision_calls": 12},
        }
        return cls.model_validate({**settings, **overrides})

    @property
    def width(self) -> int:
        """Effective number of parallel hypotheses."""
        if self.strategy == "greedy":
            return 1
        if self.strategy == "sample":
            return self.n_samples
        return self.beam_width
