"""The ``DecisionBackend`` protocol and its request/response types.

A decision backend answers *choice* questions: given shared ``state`` and, per
question, ``instructions`` plus a set of labeled ``options``, it returns a probability
distribution over exactly those labels. Traversal and ingestion depend only on this
module, never on a concrete backend.
"""

import logging
import math
import re
from collections.abc import Mapping
from typing import Protocol, Self, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator

from graphwalk.core.errors import GraphwalkError

logger = logging.getLogger(__name__)

type JSONContent = str | dict[str, JsonValue] | list[JsonValue]
"""What the wire format accepts for state, instructions, and option descriptions."""

_KEY = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")
_FROZEN = ConfigDict(frozen=True, extra="forbid")


class DecisionBackendError(GraphwalkError):
    """A decision backend failed or returned an unusable answer."""


class ChoiceQuestion(BaseModel):
    """One constrained choice. ``options`` maps label -> description (``None`` = label only)."""

    model_config = _FROZEN

    key: str
    instructions: JSONContent
    options: dict[str, JSONContent | None] = Field(min_length=2)

    @field_validator("key")
    @classmethod
    def _key_is_simple(cls, key: str) -> str:
        if not _KEY.match(key):
            msg = f"question key must match {_KEY.pattern}, got {key!r}"
            raise ValueError(msg)
        return key

    @field_validator("options")
    @classmethod
    def _labels_nonempty(
        cls, options: dict[str, JSONContent | None]
    ) -> dict[str, JSONContent | None]:
        if any(not label for label in options):
            msg = "option labels must be non-empty"
            raise ValueError(msg)
        return options


class DecisionRequest(BaseModel):
    """Several questions answered in one backend call, all about the same ``state``."""

    model_config = _FROZEN

    state: JSONContent
    questions: tuple[ChoiceQuestion, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_keys(self) -> Self:
        keys = [q.key for q in self.questions]
        if len(set(keys)) != len(keys):
            msg = f"question keys must be unique, got {keys}"
            raise ValueError(msg)
        return self


class ChoiceResult(BaseModel):
    """The distribution for one question. Keys are exactly the offered labels; sums to 1."""

    model_config = _FROZEN

    key: str
    probabilities: dict[str, float]
    top: str
    confidence: float = Field(ge=0.0, le=1.0)
    """How peaked the distribution is: 1 = all mass on one label, 0 = uniform.

    Jev reports it; per TypeSafe's docs it is derived from the probabilities as
    ``(n * p_max - 1) / (n - 1)``, so it adds no information beyond ``probabilities``.
    Backends that do not report it get that formula (see :func:`choice_confidence`).
    """


class Usage(BaseModel):
    model_config = _FROZEN

    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    cost_usd: float | None = None
    """Only when the provider reports it (OpenRouter). Never estimated."""


class DecisionResponse(BaseModel):
    model_config = _FROZEN

    results: dict[str, ChoiceResult]
    usage: Usage
    latency_s: float = Field(ge=0.0)
    model: str
    """Model id as echoed by the server."""
    request_id: str | None = None
    provider: str | None = None


@runtime_checkable
class DecisionBackend(Protocol):
    @property
    def model_id(self) -> str:
        """The pinned model id requests are sent with."""
        ...

    @property
    def max_options(self) -> int:
        """Maximum options per question."""
        ...

    async def decide(self, request: DecisionRequest) -> DecisionResponse: ...

    async def aclose(self) -> None: ...


def choice_confidence(probabilities: Mapping[str, float]) -> float:
    """TypeSafe's Choice confidence: ``(n * p_max - 1) / (n - 1)``, clamped to [0, 1]."""
    n = len(probabilities)
    if n < 2:  # noqa: PLR2004 - a single option is certain by construction
        return 1.0
    peak = max(probabilities.values())
    return min(1.0, max(0.0, (n * peak - 1) / (n - 1)))


def normalize_distribution(
    question: ChoiceQuestion,
    probabilities: Mapping[str, float],
    *,
    confidence: float | None = None,
) -> ChoiceResult:
    """Validate a raw distribution against the offered labels and renormalize it to sum 1.

    * A label that was not offered is an error (the backend broke its contract).
    * An offered label the backend omitted gets probability 0 and a warning.
    * Negative, NaN, or all-zero mass is an error.

    ``top`` is the highest-probability label, ties broken by the order options were offered.
    ``confidence`` is the backend-reported value if given (clamped to [0, 1]), otherwise
    :func:`choice_confidence` of the normalized distribution.
    """
    offered = list(question.options)
    unknown = set(probabilities) - set(offered)
    if unknown:
        msg = f"question {question.key!r}: backend returned labels not offered: {sorted(unknown)}"
        raise DecisionBackendError(msg)
    missing = [label for label in offered if label not in probabilities]
    if missing:
        logger.warning("question %r: no probability for %s; using 0", question.key, missing)
    raw = [float(probabilities.get(label, 0.0)) for label in offered]
    if any(not math.isfinite(p) or p < 0 for p in raw):
        msg = f"question {question.key!r}: invalid probabilities {raw}"
        raise DecisionBackendError(msg)
    total = math.fsum(raw)
    if total <= 0:
        msg = f"question {question.key!r}: distribution has no mass"
        raise DecisionBackendError(msg)
    normalized = {label: p / total for label, p in zip(offered, raw, strict=True)}
    top = max(offered, key=lambda label: normalized[label])  # max keeps the first on ties
    if confidence is None or not math.isfinite(confidence):
        confidence = choice_confidence(normalized)
    return ChoiceResult(
        key=question.key,
        probabilities=normalized,
        top=top,
        confidence=min(1.0, max(0.0, confidence)),
    )
