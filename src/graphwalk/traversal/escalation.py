"""Escalation: redo a walk with a stronger (slower, pricier) decider when the cheap one
is unsure.

The cheap decider's confidence in its best path (``exp(score)``) separates right from
wrong walks well enough on curated graphs (paper table A1) that re-walking only the
low-confidence queries with an LLM decider matches the LLM decider's accuracy at well
under its cost. On graphs extracted from text it does not help: there the walks fail
because facts are missing, not because the decider is unsure.
"""

from collections.abc import Sequence
from typing import Protocol

from graphwalk.core.model import NodeId
from graphwalk.embeddings.base import Embedder
from graphwalk.traversal.engine import Traverser
from graphwalk.traversal.trace import Escalation, TraversalResult


class Walker(Protocol):
    """What :class:`~graphwalk.locate.graph.GraphLocator` needs from a traverser."""

    @property
    def embedder(self) -> Embedder | None: ...

    async def traverse(self, query: str, start: Sequence[NodeId]) -> TraversalResult: ...


class EscalatingTraverser:
    """Walks with ``primary``; re-walks with ``fallback`` when the primary walk has no
    answer, did not finish (``aborted``/``error``), or its best answer's confidence is
    below ``threshold``. The fallback's result is returned, with the primary walk in
    ``result.escalation``."""

    def __init__(self, primary: Traverser, fallback: Traverser, *, threshold: float) -> None:
        if not 0.0 < threshold <= 1.0:
            msg = f"threshold must be in (0, 1], got {threshold}"
            raise ValueError(msg)
        self.primary = primary
        self.fallback = fallback
        self.threshold = threshold

    @property
    def embedder(self) -> Embedder | None:
        return self.primary.embedder

    async def traverse(self, query: str, start: Sequence[NodeId]) -> TraversalResult:
        first = await self.primary.traverse(query, start)
        if first.status != "ok":
            reason = "not_ok"
        elif first.confidence is None:
            reason = "no_answer"
        elif first.confidence < self.threshold:
            reason = "low_confidence"
        else:
            return first
        second = await self.fallback.traverse(query, start)
        escalation = Escalation(threshold=self.threshold, reason=reason, primary=first)
        return second.model_copy(update={"escalation": escalation})
