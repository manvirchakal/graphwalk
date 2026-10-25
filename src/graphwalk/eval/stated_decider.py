"""Stated-confidence baselines for the token-probability decider (paper P8).

The ``logprob`` decider reads an option letter's token probabilities. A reviewer will
ask whether the gap to the ``llm`` decider's stated 0-100 scores is about *how* the
confidence is obtained or only about which model gave it. These deciders ask the *same*
model, through the same endpoint and with the same lettered prompt
(:func:`graphwalk.decisions.logprob.prompt`, with only the last line changed), for
confidence in three other ways:

* ``scores``: a 0-100 score for every option, normalized (what the ``llm`` decider does);
* ``top1``: one letter and a 0-100 confidence in it (the "verbalized top-1" format of
  Tian et al., 2023); the rest of the mass is spread evenly over the other options;
* ``vote``: ``samples`` one-letter answers at temperature 1; the distribution is the
  vote share, smoothed so unvoted options keep a little mass (sampling consistency).

Through OpenRouter, reasoning is turned off, as for the ``logprob`` decider. Replies that
cannot be parsed fall back to a uniform distribution and are counted in ``unparsed``.
"""

import asyncio
import math
import re
import time
from collections import Counter
from typing import Any, Literal, cast

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
from graphwalk.decisions.logprob import LETTERS, OPENROUTER_V1, is_openrouter, prompt

Mode = Literal["scores", "top1", "vote"]
FLOOR = 1e-6
VOTE_SMOOTHING = 0.1
"""Pseudo-votes added to every option before normalizing."""
SYSTEM = """\
You answer one multiple-choice question about the state given. Read the instructions,
then reply exactly in the format asked for and nothing else."""
_RETRY_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})

ASK = {
    "scores": (
        "Score every option from 0 (certainly wrong) to 100 (certainly right), one per "
        "line as LETTER=SCORE (for example A=80), and nothing else."
    ),
    "top1": (
        "Reply with the letter of the single best option, a space, and how confident you "
        "are that it is right, from 0 to 100 (for example: B 85). Nothing else."
    ),
    "vote": "Answer with one letter.",
}
_SCORE = re.compile(r"\b([A-T])\s*[=:]\s*(\d{1,3})")
_TOP1 = re.compile(r"^\W*([A-T])\b\W*(\d{1,3})")


def stated_prompt(state: JsonValue, question: ChoiceQuestion, mode: Mode) -> str:
    return prompt(state, question).removesuffix("Answer with one letter.") + ASK[mode]


def parse_scores(question: ChoiceQuestion, text: str) -> dict[str, float] | None:
    letters = LETTERS[: len(question.options)]
    found = {letter: min(100, int(score)) for letter, score in _SCORE.findall(text.upper())}
    scores = {
        label: max(float(found.get(letter, 0)), FLOOR)
        for letter, label in zip(letters, question.options, strict=True)
    }
    return scores if any(letter in found for letter in letters) else None


def parse_top1(question: ChoiceQuestion, text: str) -> dict[str, float] | None:
    letters = LETTERS[: len(question.options)]
    match = _TOP1.match(text.strip().upper())
    if match is None or match.group(1) not in letters:
        return None
    confidence = min(100, int(match.group(2))) / 100
    rest = (1.0 - confidence) / (len(letters) - 1)
    return {
        label: max(confidence if letter == match.group(1) else rest, FLOOR)
        for letter, label in zip(letters, question.options, strict=True)
    }


def vote_distribution(question: ChoiceQuestion, answers: list[str]) -> dict[str, float]:
    letters = LETTERS[: len(question.options)]
    votes = Counter(a.strip().upper()[:1] for a in answers)
    return {
        label: votes.get(letter, 0) + VOTE_SMOOTHING
        for letter, label in zip(letters, question.options, strict=True)
    }


class StatedDecider:
    """A :class:`~graphwalk.decisions.base.DecisionBackend` over stated confidence."""

    def __init__(
        self,
        model: str,
        mode: Mode,
        *,
        base_url: str = OPENROUTER_V1,
        api_key: str | None = None,
        samples: int = 5,
        max_rpm: float | None = None,
        timeout_s: float = 60.0,
        attempts: int = 5,
        transport: httpx2.AsyncBaseTransport | None = None,
    ) -> None:
        self._model = model
        self.mode: Mode = mode
        self._samples = samples
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._api_key = api_key
        headers = {} if api_key is None else {"Authorization": f"Bearer {api_key}"}
        self._client = httpx2.AsyncClient(timeout=timeout_s, headers=headers, transport=transport)
        self._extra: dict[str, JsonValue] = (
            {"reasoning": {"enabled": False}} if is_openrouter(base_url) else {}
        )
        self._interval = 0.0 if not max_rpm else 60.0 / max_rpm
        self._next = 0.0
        self._lock = asyncio.Lock()
        self._attempts = max(1, attempts)
        self.asked = 0
        self.unparsed = 0

    @property
    def model_id(self) -> str:
        return f"stated-{self.mode}:{self._model}"

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

    async def _complete(
        self, text: str, *, max_tokens: int, temperature: float
    ) -> tuple[str, Usage, float]:
        body: dict[str, JsonValue] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": text},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            **self._extra,
        }
        error = "no attempts made"
        for attempt in range(self._attempts):
            if attempt:
                await asyncio.sleep(min(30.0, 2.0 * 2 ** (attempt - 1)))
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
                msg = f"endpoint returned HTTP {response.status_code}: {response.text[:300]}"
                raise DecisionBackendError(redact(msg, [self._api_key]))
            data = cast("dict[str, Any]", response.json())
            choices = cast("list[dict[str, Any]]", data.get("choices") or [{}])
            message = cast("dict[str, Any]", choices[0].get("message") or {})
            raw = cast("dict[str, Any]", data.get("usage") or {})
            cost = raw.get("cost")
            usage = Usage(
                input_tokens=int(raw.get("prompt_tokens") or 0),
                output_tokens=int(raw.get("completion_tokens") or 0),
                cost_usd=float(cost) if isinstance(cost, int | float) else None,
            )
            return str(message.get("content") or ""), usage, time.monotonic() - start
        msg = f"stated decider failed after {self._attempts} attempts: {error}"
        raise DecisionBackendError(redact(msg, [self._api_key]))

    async def _ask(
        self, state: JsonValue, question: ChoiceQuestion
    ) -> tuple[ChoiceResult, list[Usage], float]:
        text = stated_prompt(state, question, self.mode)
        self.asked += 1
        if self.mode == "vote":
            replies = await asyncio.gather(
                *(self._complete(text, max_tokens=1, temperature=1.0)
                  for _ in range(self._samples))
            )  # fmt: skip
            scores: dict[str, float] | None = vote_distribution(question, [r[0] for r in replies])
        else:
            limit = 8 * len(question.options) if self.mode == "scores" else 8
            replies = [await self._complete(text, max_tokens=limit, temperature=0.0)]
            parse = parse_scores if self.mode == "scores" else parse_top1
            scores = parse(question, replies[0][0])
        if scores is None:
            self.unparsed += 1
            scores = dict.fromkeys(question.options, 1.0)
        result = normalize_distribution(question, scores)
        return result, [r[1] for r in replies], max(r[2] for r in replies)

    async def decide(self, request: DecisionRequest) -> DecisionResponse:
        state = cast("JsonValue", request.state)
        answers = await asyncio.gather(*(self._ask(state, q) for q in request.questions))
        usages = [u for a in answers for u in a[1]]
        costs = [u.cost_usd for u in usages]
        return DecisionResponse(
            results={q.key: a[0] for q, a in zip(request.questions, answers, strict=True)},
            usage=Usage(
                input_tokens=sum(u.input_tokens for u in usages),
                output_tokens=sum(u.output_tokens for u in usages),
                cost_usd=None
                if any(c is None for c in costs)
                else math.fsum(cast("list[float]", costs)),
            ),
            latency_s=max(a[2] for a in answers),
            model=self.model_id,
            provider="stated",
        )

    def diagnostics(self) -> dict[str, float | int]:
        return {"asked": self.asked, "unparsed": self.unparsed}

    async def aclose(self) -> None:
        await self._client.aclose()
