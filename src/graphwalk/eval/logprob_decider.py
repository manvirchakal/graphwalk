"""A token-probability decider: a chat model picks an option letter, and the
distribution is read from the first token's log-probabilities (research baseline).

Unlike :class:`graphwalk.decisions.llm_decider.LLMDecider`, which normalizes the scores a
model states, this reads the model's own token probabilities, the usual way to get a
confidence out of an LLM for a multiple-choice question. Used to test whether Jev's
calibration comes from the classification framing or from Jev itself (paper P1/P2).

Calls OpenRouter's chat API directly (not LiteLLM, which drops ``top_logprobs`` for
some providers), one call per question, ``max_tokens=1``, temperature 0, reasoning off,
and only providers that support the parameters. Options are labeled A, B, C, ...; at
most 20, the number of alternatives the API returns.
"""

import asyncio
import json
import math
import time
from typing import Any, cast

import httpx
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

URL = "https://openrouter.ai/api/v1/chat/completions"
LETTERS = "ABCDEFGHIJKLMNOPQRST"
FLOOR = 1e-6
"""Probability for a letter outside the returned alternatives."""
PROVIDER = "logprob-decider"

SYSTEM = """\
You answer one multiple-choice question about the state given. Read the instructions,
then reply with the letter of the single best option and nothing else."""


def prompt(state: JsonValue, question: ChoiceQuestion) -> str:
    lines = [
        "State:",
        json.dumps(state, ensure_ascii=False),
        "",
        "Instructions:",
        json.dumps(cast("JsonValue", question.instructions), ensure_ascii=False),
        "",
        "Options:",
    ]
    for letter, (label, description) in zip(LETTERS, question.options.items(), strict=False):
        detail = "" if description is None else f": {json.dumps(description, ensure_ascii=False)}"
        lines.append(f"{letter}. {label}{detail}")
    lines += ["", "Answer with one letter."]
    return "\n".join(lines)


def letter_distribution(question: ChoiceQuestion, top: list[dict[str, Any]]) -> ChoiceResult:
    """Sum the probability of each letter over token variants (" B", "b"); letters not
    among the alternatives get ``FLOOR``. No letter at all -> uniform."""
    letters = LETTERS[: len(question.options)]
    mass = dict.fromkeys(letters, 0.0)
    for alt in top:
        token = str(alt.get("token", "")).strip().upper()
        logprob = alt.get("logprob")
        if token in mass and isinstance(logprob, int | float):
            mass[token] += math.exp(float(logprob))
    if math.fsum(mass.values()) <= 0:
        mass = dict.fromkeys(letters, 1.0)
    probabilities = {
        label: max(mass[letter], FLOOR)
        for letter, label in zip(letters, question.options, strict=True)
    }
    return normalize_distribution(question, probabilities)


def first_token_alternatives(data: dict[str, Any]) -> list[dict[str, Any]] | None:
    """The first generated token's ``top_logprobs``, or ``None`` if the reply has none."""
    choices = cast("list[dict[str, Any]]", data.get("choices") or [])
    if not choices:
        return None
    logprobs = cast("dict[str, Any]", choices[0].get("logprobs") or {})
    content = cast("list[dict[str, Any]]", logprobs.get("content") or [])
    return cast("list[dict[str, Any]]", content[0].get("top_logprobs", [])) if content else None


class LogprobDecider:
    def __init__(
        self,
        model: str,
        *,
        api_key: str,
        max_rpm: float = 120.0,
        timeout_s: float = 60.0,
        attempts: int = 6,
    ) -> None:
        self._model = model.removeprefix("openrouter/")
        self._client = httpx.AsyncClient(
            timeout=timeout_s, headers={"Authorization": f"Bearer {api_key}"}
        )
        self._interval = 60.0 / max_rpm
        self._next = 0.0
        self._lock = asyncio.Lock()
        self._attempts = attempts
        self.failures = 0
        """Calls with no letter among the alternatives (answered uniform)."""

    @property
    def model_id(self) -> str:
        return f"logprob:{self._model}"

    @property
    def max_options(self) -> int:
        return len(LETTERS)

    async def _slot(self) -> None:
        async with self._lock:
            now = time.monotonic()
            wait = self._next - now
            self._next = max(now, self._next) + self._interval
        if wait > 0:
            await asyncio.sleep(wait)

    async def _ask(
        self, state: JsonValue, question: ChoiceQuestion
    ) -> tuple[ChoiceResult, int, float, float]:
        body = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt(state, question)},
            ],
            "max_tokens": 1,
            "temperature": 0,
            "logprobs": True,
            "top_logprobs": 20,
            "reasoning": {"enabled": False},
            "provider": {"require_parameters": True},
        }
        error = ""
        for attempt in range(self._attempts):
            await self._slot()
            start = time.monotonic()
            try:
                response = await self._client.post(URL, json=body)
            except httpx.HTTPError as exc:
                error = type(exc).__name__
            else:
                if response.status_code == 200:  # noqa: PLR2004
                    data = cast("dict[str, Any]", response.json())
                    top = first_token_alternatives(data)
                    if top is not None:
                        result = letter_distribution(question, top)
                        if len(set(result.probabilities.values())) == 1:
                            self.failures += 1
                        usage = cast("dict[str, Any]", data.get("usage") or {})
                        return (
                            result,
                            int(usage.get("prompt_tokens", 0)),
                            float(usage.get("cost", 0.0)),
                            time.monotonic() - start,
                        )
                    error = f"no logprobs from {data.get('provider')}"
                else:
                    error = f"HTTP {response.status_code}: {response.text[:200]}"
            await asyncio.sleep(min(30.0, 2.0 * 2**attempt))
        msg = f"logprob decider failed after {self._attempts} attempts: {error}"
        raise DecisionBackendError(msg)

    async def decide(self, request: DecisionRequest) -> DecisionResponse:
        for question in request.questions:
            if len(question.options) > self.max_options:
                msg = f"question {question.key!r} has {len(question.options)} options (max 20)"
                raise DecisionBackendError(msg)
        state = cast("JsonValue", request.state)
        answers = await asyncio.gather(*(self._ask(state, q) for q in request.questions))
        return DecisionResponse(
            results={q.key: a[0] for q, a in zip(request.questions, answers, strict=True)},
            usage=Usage(
                input_tokens=sum(a[1] for a in answers),
                output_tokens=len(answers),
                cost_usd=math.fsum(a[2] for a in answers),
            ),
            latency_s=max(a[3] for a in answers),
            model=self.model_id,
            provider=PROVIDER,
        )

    async def aclose(self) -> None:
        await self._client.aclose()
