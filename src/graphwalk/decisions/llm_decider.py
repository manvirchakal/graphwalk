"""LLM-as-decider: answers choice questions with a chat model instead of Jev.

The fallback for when no Jev key is configured (``GRAPHWALK_DECISION_FALLBACK=llm``).
One chat call per request covers all its questions: the model scores every option from
0 to 100, and the distribution is the scores normalized to sum to 1. That makes it a
drop-in ``DecisionBackend``, but not an equivalent one:

* the probabilities are **not calibrated**: they are the model's stated scores;
* it is slower and usually costlier per decision than Jev.

Its ``model_id`` and every response's ``model`` start with ``llm-decider:`` and its
``provider`` is ``llm-fallback``, so traces and results show which decider ran.
"""

import asyncio
import json
import math
from typing import Any, cast

from pydantic import JsonValue

from graphwalk.decisions.base import (
    ChoiceQuestion,
    ChoiceResult,
    DecisionBackendError,
    DecisionRequest,
    DecisionResponse,
    Usage,
    normalize_distribution,
)
from graphwalk.llm.base import LLMBackend, LLMError, Message

MODEL_PREFIX = "llm-decider:"
PROVIDER = "llm-fallback"

SYSTEM = """\
You answer multiple-choice questions by scoring every option. The user message is JSON:
{"state": ..., "questions": [{"key": str, "instructions": ..., "options": {label: description}}]}
For each question, read its instructions in the light of the state and score each
option from 0 (certainly wrong) to 100 (certainly right). Use the whole range; scores
of different options may be equal. Reply with one JSON object and nothing else:
{"<question key>": {"<option label>": <score>, ...}, ...}
Score every option of every question, using the labels exactly as given."""


def request_payload(request: DecisionRequest) -> dict[str, JsonValue]:
    """What the model is shown (the user message is this, as JSON)."""
    return {
        "state": cast("JsonValue", request.state),
        "questions": [
            {
                "key": q.key,
                "instructions": cast("JsonValue", q.instructions),
                "options": cast("JsonValue", q.options),
            }
            for q in request.questions
        ],
    }


def _scores(question: ChoiceQuestion, raw: object) -> dict[str, float]:
    """Offered labels -> non-negative scores; unscored or invalid labels get 0, and a
    question with no usable score gets a uniform distribution."""
    given = cast("dict[str, Any]", raw) if isinstance(raw, dict) else {}
    scores: dict[str, float] = {}
    for label in question.options:
        value = given.get(label)
        number = (
            float(value) if isinstance(value, int | float) and not isinstance(value, bool) else 0.0
        )
        scores[label] = number if math.isfinite(number) and number > 0 else 0.0
    if math.fsum(scores.values()) <= 0:
        return dict.fromkeys(question.options, 1.0)
    return scores


def parse_scores(request: DecisionRequest, text: str) -> dict[str, ChoiceResult]:
    """The reply's outermost JSON object as one distribution per question.

    Raises ``ValueError`` if there is no JSON object or no question is answered.
    """
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        msg = "no JSON object in the reply"
        raise ValueError(msg)
    body: object = json.loads(text[start : end + 1])
    if not isinstance(body, dict):
        msg = "the reply is not a JSON object"
        raise ValueError(msg)
    answers = cast("dict[str, Any]", body)
    if not any(q.key in answers for q in request.questions):
        msg = f"no scores for any question key ({', '.join(q.key for q in request.questions)})"
        raise ValueError(msg)
    return {
        q.key: normalize_distribution(q, _scores(q, answers.get(q.key))) for q in request.questions
    }


class LLMDecider:
    def __init__(self, llm: LLMBackend, *, max_options: int = 64, attempts: int = 2) -> None:
        self._llm = llm
        self._max_options = max_options
        self._attempts = max(1, attempts)

    @property
    def model_id(self) -> str:
        return MODEL_PREFIX + self._llm.model_id

    @property
    def max_options(self) -> int:
        return self._max_options

    async def decide(self, request: DecisionRequest) -> DecisionResponse:
        for question in request.questions:
            if len(question.options) > self._max_options:
                msg = (
                    f"question {question.key!r} has {len(question.options)} options; "
                    f"the LLM decider allows at most {self._max_options}"
                )
                raise DecisionBackendError(msg)
        messages = [
            Message(role="system", content=SYSTEM),
            Message(role="user", content=json.dumps(request_payload(request), ensure_ascii=False)),
        ]
        input_tokens = output_tokens = 0
        latency = 0.0
        costs: list[float | None] = []
        error = "no attempts made"
        for _ in range(self._attempts):
            try:
                reply = await self._llm.complete(messages)
            except LLMError as exc:
                raise DecisionBackendError(f"LLM decider call failed: {exc}") from exc
            input_tokens += reply.input_tokens
            latency += reply.latency_s
            output_tokens += reply.output_tokens
            costs.append(reply.cost_usd)
            try:
                results = parse_scores(request, reply.text)
            except (ValueError, DecisionBackendError) as exc:
                error = str(exc)[:300]
                messages = [
                    *messages,
                    Message(role="assistant", content=reply.text),
                    Message(
                        role="user",
                        content=f"That reply was not valid ({error}). "
                        "Reply with the JSON object only.",
                    ),
                ]
                continue
            return DecisionResponse(
                results=results,
                usage=Usage(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost_usd=None
                    if any(c is None for c in costs)
                    else math.fsum(c for c in costs if c is not None),
                ),
                latency_s=latency,  # the model's own call times, retries included
                model=MODEL_PREFIX + reply.model,
                provider=PROVIDER,
            )
        msg = f"LLM decider gave no usable scores after {self._attempts} attempts: {error}"
        raise DecisionBackendError(msg)

    async def aclose(self) -> None:
        close = getattr(self._llm, "aclose", None)
        if close is not None:
            result = close()
            if asyncio.iscoroutine(result):
                await result
