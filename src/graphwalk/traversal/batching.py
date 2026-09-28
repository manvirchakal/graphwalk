"""Split a depth's questions into requests that fit Jev's context limits.

Jev allows 64k tokens per request and 32k for ``state`` + the longest single question
(TypeSafe docs). We have no tokenizer, so tokens are *estimated* conservatively at
3 characters per token of compact JSON, and limits are shrunk by a safety margin.
"""

import json
import math
from collections.abc import Sequence

from pydantic import JsonValue

from graphwalk.decisions.base import ChoiceQuestion, DecisionBackendError

CHARS_PER_TOKEN = 3.0


def estimate_tokens(value: JsonValue) -> int:
    text = json.dumps(value, separators=(",", ":"), ensure_ascii=False)
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def question_tokens(question: ChoiceQuestion) -> int:
    return estimate_tokens(question.model_dump(mode="json"))


def split_questions(
    state: JsonValue,
    questions: Sequence[ChoiceQuestion],
    *,
    max_request_tokens: int,
    max_state_plus_question_tokens: int,
    safety_margin: float,
) -> list[list[ChoiceQuestion]]:
    """Greedy in-order packing. Raises if one question cannot fit on its own."""
    request_cap = max_request_tokens * (1.0 - safety_margin)
    single_cap = max_state_plus_question_tokens * (1.0 - safety_margin)
    state_tokens = estimate_tokens(state)
    groups: list[list[ChoiceQuestion]] = []
    current: list[ChoiceQuestion] = []
    current_tokens = state_tokens
    for question in questions:
        tokens = question_tokens(question)
        if state_tokens + tokens > single_cap:
            msg = (
                f"question {question.key!r} is ~{tokens} tokens; with the state that exceeds "
                f"the ~{int(single_cap)}-token per-question budget (lower prefilter_top_n)"
            )
            raise DecisionBackendError(msg)
        if current and current_tokens + tokens > request_cap:
            groups.append(current)
            current, current_tokens = [], state_tokens
        current.append(question)
        current_tokens += tokens
    if current:
        groups.append(current)
    return groups
