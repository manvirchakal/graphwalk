"""Deterministic, scriptable decision backend for tests and offline development."""

import asyncio
import json
import random
from collections.abc import Callable, Collection, Mapping

from graphwalk.core.hashing import content_hash
from graphwalk.decisions.base import (
    ChoiceQuestion,
    ChoiceResult,
    DecisionBackendError,
    DecisionRequest,
    DecisionResponse,
    JSONContent,
    Usage,
    normalize_distribution,
)

type Script = Callable[[ChoiceQuestion, JSONContent], Mapping[str, float] | None]
"""Return a (possibly unnormalized) distribution for a question, or ``None`` to fall back
to the seeded default."""


class FakeDecisionBackend:
    """A ``DecisionBackend`` with fully reproducible answers.

    * ``script`` decides distributions; unscripted questions get a Dirichlet(1, ..., 1) draw
      seeded by ``(seed, question content, state)``, so answers do not depend on call order.
    * Usage is synthetic but deterministic: ~4 characters per input token, one output token
      per question. ``cost_per_call`` simulates a provider that reports cost.
    * ``latency_s`` is *reported*; ``delay_s`` actually sleeps (for concurrency tests).
    * ``fail_on_calls`` (1-based) raise :class:`DecisionBackendError` on those calls.
    * Every request is recorded in ``requests``.
    """

    def __init__(
        self,
        *,
        seed: int = 0,
        script: Script | None = None,
        model_id: str = "fake-decider-1",
        max_options: int = 255,
        cost_per_call: float | None = None,
        latency_s: float = 0.01,
        delay_s: float = 0.0,
        fail_on_calls: Collection[int] = (),
    ) -> None:
        self._seed = seed
        self._script = script
        self._model_id = model_id
        self._max_options = max_options
        self._cost_per_call = cost_per_call
        self._latency_s = latency_s
        self._delay_s = delay_s
        self._fail_on_calls = frozenset(fail_on_calls)
        self.requests: list[DecisionRequest] = []
        self.closed = False

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def max_options(self) -> int:
        return self._max_options

    @property
    def calls(self) -> int:
        return len(self.requests)

    def default_distribution(
        self, question: ChoiceQuestion, state: JSONContent
    ) -> dict[str, float]:
        """The seeded Dirichlet draw used for unscripted questions."""
        material = json.dumps(
            {"seed": self._seed, "state": state, "question": question.model_dump(mode="json")},
            sort_keys=True,
        )
        rng = random.Random(content_hash(material))  # noqa: S311 - not cryptographic
        return {label: rng.gammavariate(1.0, 1.0) for label in question.options}

    async def decide(self, request: DecisionRequest) -> DecisionResponse:
        self.requests.append(request)
        if self.calls in self._fail_on_calls:
            msg = f"fake failure on call {self.calls}"
            raise DecisionBackendError(msg)
        for question in request.questions:
            if len(question.options) > self._max_options:
                msg = f"{len(question.options)} options exceeds max_options={self._max_options}"
                raise DecisionBackendError(msg)
        if self._delay_s:
            await asyncio.sleep(self._delay_s)
        results: dict[str, ChoiceResult] = {}
        for question in request.questions:
            scripted = None if self._script is None else self._script(question, request.state)
            raw = (
                self.default_distribution(question, request.state) if scripted is None else scripted
            )
            results[question.key] = normalize_distribution(question, raw)
        payload = json.dumps(request.model_dump(mode="json"), separators=(",", ":"))
        return DecisionResponse(
            results=results,
            usage=Usage(
                input_tokens=max(1, len(payload) // 4),
                output_tokens=len(request.questions),
                cost_usd=self._cost_per_call,
            ),
            latency_s=self._latency_s,
            model=self._model_id,
            request_id=f"fake-{self.calls}",
            provider="fake",
        )

    async def aclose(self) -> None:
        self.closed = True
