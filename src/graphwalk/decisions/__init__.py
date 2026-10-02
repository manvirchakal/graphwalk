"""Decision backends: constrained choices with full probability distributions.

Bring your own decider: anything with ``model_id``, ``max_options``, an async
``decide(DecisionRequest) -> DecisionResponse`` returning a distribution over each
question's offered labels (:func:`normalize_distribution` validates one), and an async
``aclose()`` satisfies :class:`DecisionBackend`; pass it to ``Index.open(decider=...)``.
Built in: :class:`LogprobDecider` (token probabilities from any OpenAI-compatible
endpoint), ``JevBackend`` (``graphwalk.decisions.jev``), and :class:`LLMDecider` (a chat
model's stated scores; not calibrated).
"""

from graphwalk.decisions.base import (
    ChoiceQuestion,
    ChoiceResult,
    DecisionBackend,
    DecisionBackendError,
    DecisionRequest,
    DecisionResponse,
    JSONContent,
    Usage,
    choice_confidence,
    normalize_distribution,
)
from graphwalk.decisions.fake import FakeDecisionBackend
from graphwalk.decisions.llm_decider import LLMDecider
from graphwalk.decisions.logprob import LogprobDecider

__all__ = [
    "ChoiceQuestion",
    "ChoiceResult",
    "DecisionBackend",
    "DecisionBackendError",
    "DecisionRequest",
    "DecisionResponse",
    "FakeDecisionBackend",
    "JSONContent",
    "LLMDecider",
    "LogprobDecider",
    "Usage",
    "choice_confidence",
    "normalize_distribution",
]
# JevBackend is imported from graphwalk.decisions.jev so the SDK loads only when used.
