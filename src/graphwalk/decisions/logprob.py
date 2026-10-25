"""The token-probability decider: any chat model that returns token log-probabilities.

The model sees the options lettered A, B, C, ... and answers with one letter; the
distribution over options is read from that first token's ``top_logprobs``. The
probabilities are the model's own, which is what makes a walk's confidence meaningful:
in the paper experiments (``docs/results-paper.md``, P1), an open-weights model decided
this way ranked walk answers as well as Jev did, where the same kind of model *stating*
scores (:class:`~graphwalk.decisions.llm_decider.LLMDecider`) barely did.

Works with any OpenAI-compatible ``/chat/completions`` endpoint that supports
``logprobs`` and ``top_logprobs``: OpenRouter (the default), OpenAI, vLLM, SGLang,
llama.cpp's server, and others. One call per question, ``max_tokens=1``, temperature
0. Through OpenRouter, requests also ask for providers that honor every parameter and
turn reasoning off. At most 20 options per question: the number of alternatives the
OpenAI API returns.

An endpoint that answers without log-probabilities raises
:class:`~graphwalk.decisions.base.DecisionBackendError` at once, rather than deciding
uniformly: a decider without probabilities is not this decider. Through OpenRouter,
some providers of a model silently drop the parameters: the decider then asks
OpenRouter to skip that provider (``provider.ignore``) for the rest of its life and
retries, failing only when no provider left returns log-probabilities.
"""

import asyncio
import json
import logging
import math
import time
from collections.abc import Mapping
from typing import Any, cast
from urllib.parse import urlsplit

import httpx2
from pydantic import JsonValue

from graphwalk.core.redact import redact
from graphwalk.decisions.base import (
    ChoiceQuestion,
    ChoiceResult,
    DecisionBackendError,
    DecisionRequest,
    DecisionResponse,
    Usage,
    normalize_distribution,
)

logger = logging.getLogger(__name__)

LETTERS = "ABCDEFGHIJKLMNOPQRST"
FLOOR = 1e-6
"""Probability for a letter outside the returned alternatives."""
PROVIDER = "logprob"
MODEL_PREFIX = "logprob:"
OPENROUTER_V1 = "https://openrouter.ai/api/v1"
DEFAULT_OPENROUTER_MODEL = "qwen/qwen3.8-27b"
"""Open weights; the model measured in the paper experiments (P1)."""
_RETRY_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})

SYSTEM = """\
You answer one multiple-choice question about the state given. Read the instructions,
then reply with the letter of the single best option and nothing else."""


def prompt(state: JsonValue, question: ChoiceQuestion) -> str:
    """The user message: the state, the instructions, and the options as letters."""
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
    """Sum each letter's probability over its token variants (" B", "b"); letters not
    among the alternatives get ``FLOOR``. No letter at all: uniform."""
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


def letter_mass(question: ChoiceQuestion, top: list[dict[str, Any]]) -> tuple[float, bool]:
    """The probability the alternatives put on the offered letters, and whether the most
    likely token is one of them: how well the model kept to the format."""
    letters = set(LETTERS[: len(question.options)])
    mass = 0.0
    best: tuple[float, str] | None = None
    for alt in top:
        token = str(alt.get("token", "")).strip().upper()
        logprob = alt.get("logprob")
        if not isinstance(logprob, int | float):
            continue
        if token in letters:
            mass += math.exp(float(logprob))
        if best is None or logprob > best[0]:
            best = (float(logprob), token)
    return min(mass, 1.0), best is not None and best[1] in letters


def first_token_alternatives(data: Mapping[str, Any]) -> list[dict[str, Any]] | None:
    """The first generated token's ``top_logprobs``, or ``None`` if the reply has none."""
    choices = cast("list[dict[str, Any]]", data.get("choices") or [])
    if not choices:
        return None
    logprobs = cast("dict[str, Any]", choices[0].get("logprobs") or {})
    content = cast("list[dict[str, Any]]", logprobs.get("content") or [])
    if not content:
        return None
    top = cast("list[dict[str, Any]]", content[0].get("top_logprobs") or [])
    return top or None


def is_openrouter(base_url: str) -> bool:
    return urlsplit(base_url).hostname == "openrouter.ai"


class LogprobDecider:
    """A :class:`~graphwalk.decisions.base.DecisionBackend` over token probabilities.

    ``base_url`` is the API root including its version path
    (``https://openrouter.ai/api/v1``, ``http://localhost:8000/v1``). ``api_key`` may be
    ``None`` for a local server. ``extra_body`` is merged into every request (for
    provider-specific switches). ``max_rpm`` spaces request starts; 429s and 5xx
    responses are retried with backoff.
    """

    def __init__(
        self,
        model: str = DEFAULT_OPENROUTER_MODEL,
        *,
        base_url: str = OPENROUTER_V1,
        api_key: str | None = None,
        extra_body: Mapping[str, JsonValue] | None = None,
        max_rpm: float | None = None,
        timeout_s: float = 60.0,
        attempts: int = 5,
        transport: httpx2.AsyncBaseTransport | None = None,
    ) -> None:
        self._model = model
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._api_key = api_key
        headers = {} if api_key is None else {"Authorization": f"Bearer {api_key}"}
        self._client = httpx2.AsyncClient(timeout=timeout_s, headers=headers, transport=transport)
        self._routed = is_openrouter(base_url)
        self._extra: dict[str, JsonValue] = (
            {"reasoning": {"enabled": False}, "provider": {"require_parameters": True}}
            if self._routed
            else {}
        )
        self._extra.update(extra_body or {})
        self._skip: list[str] = []
        """OpenRouter providers seen dropping log-probabilities for this model."""
        self._interval = 0.0 if not max_rpm else 60.0 / max_rpm
        self._next = 0.0
        self._lock = asyncio.Lock()
        self._attempts = max(1, attempts)
        self.no_letter = 0
        """Questions answered uniformly because no option letter was among the
        alternatives (rare; a sign the model ignores the format)."""
        self.asked = 0
        """Questions answered."""
        self.letter_mass_total = 0.0
        """Summed probability on the offered letters (see :meth:`diagnostics`)."""
        self.top_not_letter = 0
        """Questions whose most likely first token was not an offered letter."""

    @property
    def model_id(self) -> str:
        return MODEL_PREFIX + self._model

    @property
    def max_options(self) -> int:
        return len(LETTERS)

    async def _slot(self) -> None:
        if not self._interval:
            return
        async with self._lock:
            now = time.monotonic()
            wait = self._next - now
            self._next = max(now, self._next) + self._interval
        if wait > 0:
            await asyncio.sleep(wait)

    def _error(self, text: str) -> DecisionBackendError:
        return DecisionBackendError(redact(text, [self._api_key]))

    def _body(self, messages: list[JsonValue]) -> dict[str, JsonValue]:
        body: dict[str, JsonValue] = {
            "model": self._model,
            "messages": messages,
            "max_tokens": 1,
            "temperature": 0,
            "logprobs": True,
            "top_logprobs": len(LETTERS),
            **self._extra,
        }
        if self._skip:
            routing = cast("dict[str, JsonValue]", body.get("provider") or {})
            body["provider"] = {**routing, "ignore": cast("list[JsonValue]", list(self._skip))}
        return body

    async def _ask(
        self, state: JsonValue, question: ChoiceQuestion
    ) -> tuple[ChoiceResult, Usage, float, str]:
        messages: list[JsonValue] = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt(state, question)},
        ]
        error = "no attempts made"
        for attempt in range(self._attempts):
            if attempt:
                await asyncio.sleep(min(30.0, 2.0 * 2 ** (attempt - 1)))
            body = self._body(messages)
            await self._slot()
            start = time.monotonic()
            try:
                response = await self._client.post(self._url, json=body)
            except httpx2.HTTPError as exc:
                error = f"{type(exc).__name__}: {exc}"
                continue
            if response.status_code in _RETRY_STATUS:
                error = f"HTTP {response.status_code}: {response.text[:300]}"
                continue
            if response.status_code != 200:  # noqa: PLR2004
                msg = (
                    f"decider endpoint returned HTTP {response.status_code}: {response.text[:300]}"
                )
                raise self._error(msg)
            data = cast("dict[str, Any]", response.json())
            top = first_token_alternatives(data)
            if top is None:
                served = data.get("provider") or data.get("model") or self._model
                msg = (
                    f"{served} returned no token log-probabilities; the logprob decider "
                    "needs a model and endpoint that support logprobs/top_logprobs"
                )
                provider = data.get("provider")
                routing = cast("dict[str, Any]", body.get("provider") or {})
                skipped = cast("list[str]", routing.get("ignore") or [])
                if self._routed and isinstance(provider, str) and provider not in skipped:
                    # OpenRouter served the model from a provider that drops the
                    # parameters; skip that provider from now on and ask again.
                    if provider not in self._skip:
                        logger.warning("%s; asking OpenRouter to skip it", msg)
                        self._skip.append(provider)
                    error = msg
                    continue
                raise self._error(msg)
            result = letter_distribution(question, top)
            mass, top_is_letter = letter_mass(question, top)
            self.asked += 1
            self.letter_mass_total += mass
            self.top_not_letter += not top_is_letter
            if len(set(result.probabilities.values())) == 1:
                self.no_letter += 1
                logger.debug("question %r: no option letter among the alternatives", question.key)
            raw_usage = cast("dict[str, Any]", data.get("usage") or {})
            cost = raw_usage.get("cost")
            usage = Usage(
                input_tokens=int(raw_usage.get("prompt_tokens") or 0),
                output_tokens=int(raw_usage.get("completion_tokens") or 0),
                cost_usd=float(cost) if isinstance(cost, int | float) else None,
            )
            return result, usage, time.monotonic() - start, str(data.get("model") or self._model)
        msg = f"logprob decider failed after {self._attempts} attempts: {error}"
        raise self._error(msg)

    def diagnostics(self) -> dict[str, float | int]:
        """How well the model kept to the one-letter format so far: questions asked, the
        mean probability on offered letters (the rest went to other tokens and is
        dropped by renormalizing), and how often the top token was not a letter."""
        return {
            "asked": self.asked,
            "mean_letter_mass": self.letter_mass_total / self.asked if self.asked else 0.0,
            "top_not_letter": self.top_not_letter,
            "no_letter": self.no_letter,
        }

    async def decide(self, request: DecisionRequest) -> DecisionResponse:
        for question in request.questions:
            if len(question.options) > self.max_options:
                msg = (
                    f"question {question.key!r} has {len(question.options)} options; "
                    f"the logprob decider allows at most {self.max_options}"
                )
                raise DecisionBackendError(msg)
        state = cast("JsonValue", request.state)
        answers = await asyncio.gather(*(self._ask(state, q) for q in request.questions))
        costs = [a[1].cost_usd for a in answers]
        return DecisionResponse(
            results={q.key: a[0] for q, a in zip(request.questions, answers, strict=True)},
            usage=Usage(
                input_tokens=sum(a[1].input_tokens for a in answers),
                output_tokens=sum(a[1].output_tokens for a in answers),
                cost_usd=None
                if any(c is None for c in costs)
                else math.fsum(cast("list[float]", costs)),
            ),
            latency_s=max(a[2] for a in answers),  # questions run concurrently
            model=MODEL_PREFIX + answers[0][3],
            provider=PROVIDER,
        )

    async def aclose(self) -> None:
        await self._client.aclose()
