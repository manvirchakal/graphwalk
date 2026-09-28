"""Jev (TypeSafe System One) decision backend, via the official ``typesafe-sdk``.

One class serves both providers: TypeSafe direct (``https://api.typesafe.ai``) and
OpenRouter's TypeSafe-compatible endpoint (``https://openrouter.ai/api``). Both expose
``POST /v1/systemone``; only the base URL, key, and model id differ.
"""

import json
import logging
import time
from typing import Any, cast

import httpx2
from typesafe_sdk import (
    AsyncTypeSafeClient,
    Choice,
    ChoiceAnswer,
    RetryPolicy,
    SystemOneResponse,
    TypeSafeError,
)

from graphwalk.config import (
    OPENROUTER_BASE_URL,
    TYPESAFE_BASE_URL,
    DecisionProvider,
    GraphwalkSettings,
)
from graphwalk.decisions.base import (
    ChoiceResult,
    DecisionBackendError,
    DecisionRequest,
    DecisionResponse,
    Usage,
    normalize_distribution,
)

logger = logging.getLogger(__name__)

JEV_MAX_OPTIONS = 255
"""Maximum options per choice question (from TypeSafe's published limits; unverified live)."""

_HTTP_NOT_FOUND = 404

_BASE_URLS: dict[DecisionProvider, str] = {
    "typesafe": TYPESAFE_BASE_URL,
    "openrouter": OPENROUTER_BASE_URL,
}


class JevBackend:
    """``DecisionBackend`` backed by Jev.

    ``transport`` is for tests: pass an ``httpx2.MockTransport`` to exercise the real SDK
    code path without a network.
    """

    def __init__(
        self,
        *,
        api_key: str,
        provider: DecisionProvider,
        model: str,
        base_url: str | None = None,
        timeout_s: float = 10.0,
        max_retries: int = 2,
        transport: httpx2.AsyncBaseTransport | None = None,
    ) -> None:
        if model.endswith("-latest"):
            msg = f"refusing moving model alias {model!r}; pin an exact version"
            raise ValueError(msg)
        self._provider: DecisionProvider = provider
        self._model = model
        self._base_url = (base_url or _BASE_URLS[provider]).rstrip("/")
        self._api_key = api_key
        self._timeout_s = timeout_s
        self._transport = transport
        self._warned_model_mismatch = False
        try:
            self._client = AsyncTypeSafeClient(
                api_key=api_key,
                model=model,
                base_url=self._base_url,
                timeout=timeout_s,
                retry=RetryPolicy(max_retries=max_retries),
                transport=transport,
            )
        except TypeSafeError as error:
            raise DecisionBackendError(str(error)) from error

    @classmethod
    def from_settings(
        cls, settings: GraphwalkSettings, *, transport: httpx2.AsyncBaseTransport | None = None
    ) -> "JevBackend":
        key = settings.decision_api_key
        if key is None:
            env = (
                "OPENROUTER_API_KEY"
                if settings.decision_provider == "openrouter"
                else "TYPESAFE_API_KEY"
            )
            msg = f"no API key for provider {settings.decision_provider!r}; set {env}"
            raise DecisionBackendError(msg)
        return cls(
            api_key=key.get_secret_value(),
            provider=settings.decision_provider,
            model=settings.jev_model,
            base_url=settings.decision_base_url,
            timeout_s=settings.jev_timeout_s,
            max_retries=settings.jev_max_retries,
            transport=transport,
        )

    @property
    def model_id(self) -> str:
        return self._model

    @property
    def max_options(self) -> int:
        return JEV_MAX_OPTIONS

    @property
    def provider(self) -> DecisionProvider:
        return self._provider

    async def decide(self, request: DecisionRequest) -> DecisionResponse:
        for question in request.questions:
            if len(question.options) > JEV_MAX_OPTIONS:
                msg = (
                    f"question {question.key!r} has {len(question.options)} options; "
                    f"Jev allows at most {JEV_MAX_OPTIONS}"
                )
                raise DecisionBackendError(msg)
        questions = {
            q.key: Choice(instructions=q.instructions, criteria=q.options)
            for q in request.questions
        }
        started = time.perf_counter()
        try:
            # The SDK's recursive JSON alias is opaque to pyright strict; the call is typed.
            response = await self._client.system_one(  # pyright: ignore[reportUnknownMemberType]
                state=request.state, questions=questions, model=self._model
            )
        except TypeSafeError as error:
            raise DecisionBackendError(f"Jev request failed: {error}") from error
        latency = time.perf_counter() - started

        results: dict[str, ChoiceResult] = {}
        for question in request.questions:
            answer = response.answers.get(question.key)
            if not isinstance(answer, ChoiceAnswer):
                msg = f"Jev returned no choice answer for question {question.key!r}"
                raise DecisionBackendError(msg)
            results[question.key] = normalize_distribution(question, answer.probabilities)

        self._check_model(response.model)
        extras = _openrouter_extras(response)
        return DecisionResponse(
            results=results,
            usage=Usage(
                input_tokens=response.usage.input_tokens or 0,
                output_tokens=response.usage.output_tokens or 0,
                cost_usd=extras.get("cost"),
            ),
            latency_s=latency,
            model=response.model,
            request_id=_request_id(response) or extras.get("id"),
            provider=extras.get("provider") or self._provider,
        )

    async def verify_model(self) -> None:
        """Fail fast if the pinned model is not offered by the provider.

        * TypeSafe: ``GET /v1/models`` -> ``{"models": [{"name": ...}]}``.
        * OpenRouter: ``GET /api/v1/models/{id}/endpoints`` -> ``{"data": {"id": ...,
          "endpoints": [...]}}``. OpenRouter's general ``/models`` list omits
          decision-modality models such as Jev, so it cannot be used for this check.
        """
        try:
            offered = (
                await self._typesafe_offers_model()
                if self._provider == "typesafe"
                else await self._openrouter_offers_model()
            )
        except (TypeSafeError, httpx2.HTTPError, ValueError, KeyError, TypeError) as error:
            raise DecisionBackendError(f"could not check model availability: {error}") from error
        if not offered:
            msg = f"pinned model {self._model!r} is not offered by {self._provider}"
            raise DecisionBackendError(msg)

    async def _typesafe_offers_model(self) -> bool:
        listing = await self._client.models.list()
        return self._model in {m.name for m in listing.models}

    async def _openrouter_offers_model(self) -> bool:
        async with httpx2.AsyncClient(transport=self._transport, timeout=self._timeout_s) as client:
            response = await client.get(
                f"{self._base_url}/v1/models/{self._model}/endpoints",
                headers={"Authorization": f"Bearer {self._api_key}"},
            )
            if response.status_code == _HTTP_NOT_FOUND:
                return False
            response.raise_for_status()
            data = cast("dict[str, Any]", cast("dict[str, Any]", response.json())["data"])
            endpoints = cast("list[Any]", data.get("endpoints") or [])
            return data.get("id") == self._model and len(endpoints) > 0

    def _check_model(self, echoed: str) -> None:
        if echoed != self._model and not self._warned_model_mismatch:
            self._warned_model_mismatch = True
            logger.warning("requested model %r but server answered with %r", self._model, echoed)

    async def aclose(self) -> None:
        await self._client.aclose()


def _request_id(response: SystemOneResponse) -> str | None:
    try:
        return response.request_id
    except TypeSafeError:
        return None


def _openrouter_extras(response: SystemOneResponse) -> dict[str, Any]:
    """OpenRouter adds ``id``, ``provider`` and ``usage.cost``; the SDK's models drop them."""
    try:
        body = cast("dict[str, Any]", json.loads(response.raw_http_response.content))
    except (TypeSafeError, ValueError):
        return {}
    usage = cast("dict[str, Any]", body.get("usage") or {})
    cost = usage.get("cost")
    return {
        "id": body.get("id") if isinstance(body.get("id"), str) else None,
        "provider": body.get("provider") if isinstance(body.get("provider"), str) else None,
        "cost": float(cost) if isinstance(cost, int | float) else None,
    }
