"""Build backends for each role from a :class:`~graphwalk.config.ResolvedConfig`.

* :func:`make_decider`: the configured decider (``decider``): Jev, the token-probability
  decider at any OpenAI-compatible endpoint, or the LLM-as-decider. With Jev and no Jev
  key, ``decision_fallback`` picks another, or a :class:`ConfigError` names the
  variables to set.
* :func:`make_escalation_decider`: what escalated walks decide with.
* :func:`make_llm`: any chat provider, through LiteLLM.
* :func:`make_embedder`: local fastembed, or OpenAI-compatible remote embeddings.

Keys and base URLs are always passed explicitly, never left for a client library to
find in the environment, so a remote server with ``env_keys=False`` cannot fall back
to its own keys by accident.
"""

from dataclasses import dataclass
from typing import Literal

from graphwalk.config import (
    ChatProvider,
    ConfigError,
    Provider,
    ResolvedConfig,
    env_name,
    resolve_config,
)
from graphwalk.decisions.base import DecisionBackend
from graphwalk.embeddings.base import Embedder
from graphwalk.llm.base import LLMBackend

type Role = Literal["decision", "llm", "embedding"]


@dataclass(frozen=True)
class ProviderSpec:
    name: Provider
    roles: frozenset[Role]
    litellm_prefix: str | None
    """LiteLLM's model prefix for chat, if the provider serves chat."""
    version_path: str
    """Appended to the base URL for OpenAI-style endpoints (chat, embeddings)."""


REGISTRY: dict[Provider, ProviderSpec] = {
    "openrouter": ProviderSpec(
        "openrouter", frozenset({"decision", "llm", "embedding"}), "openrouter", "/v1"
    ),
    "typesafe": ProviderSpec("typesafe", frozenset({"decision"}), None, ""),
    "openai": ProviderSpec("openai", frozenset({"llm", "embedding"}), "openai", ""),
    "anthropic": ProviderSpec("anthropic", frozenset({"llm"}), "anthropic", ""),
    "xai": ProviderSpec("xai", frozenset({"llm"}), "xai", ""),
}  # fmt: skip
"""Which provider serves which role. Base URLs follow each SDK's convention, so only
OpenRouter's (``https://openrouter.ai/api``, shared with Jev) needs ``/v1`` added."""


def supports(provider: Provider, role: Role) -> bool:
    return role in REGISTRY[provider].roles


def _key(config: ResolvedConfig, provider: Provider, role: Role) -> str:
    key = config.settings.api_key(provider)
    if key is None:
        msg = (
            f"the {role} role uses {provider}, which has no key: set {env_name(provider)} "
            "(or send it as a header to a remote server)"
        )
        raise ConfigError(msg)
    return key.get_secret_value()


def make_logprob_decider(
    config: ResolvedConfig | None = None, *, model: str | None = None
) -> DecisionBackend:
    """The token-probability decider at ``decider_base_url`` (default: OpenRouter)."""
    from graphwalk.decisions.logprob import (  # noqa: PLC0415
        DEFAULT_OPENROUTER_MODEL,
        LogprobDecider,
    )

    config = config or resolve_config()
    s = config.settings
    model = model or s.decider_model
    if s.decider_base_url is None:
        return LogprobDecider(
            model or DEFAULT_OPENROUTER_MODEL,
            base_url=s.openrouter_base_url + REGISTRY["openrouter"].version_path,
            api_key=_key(config, "openrouter", "decision"),
            max_rpm=s.decider_max_rpm,
        )
    if model is None:
        msg = "decider_base_url is set but no model: set GRAPHWALK_DECIDER_MODEL"
        raise ConfigError(msg)
    key = s.decider_api_key
    return LogprobDecider(
        model,
        base_url=s.decider_base_url,
        api_key=None if key is None else key.get_secret_value(),
        max_rpm=s.decider_max_rpm,
    )


def _jev(config: ResolvedConfig) -> DecisionBackend | None:
    s = config.settings
    key = s.decision_api_key
    if key is None:
        return None
    from graphwalk.decisions.jev import JevBackend  # noqa: PLC0415 - loads the SDK lazily

    return JevBackend(
        api_key=key.get_secret_value(),
        provider=s.decision_provider,
        model=s.jev_model,
        base_url=s.decision_base_url,
        timeout_s=s.jev_timeout_s,
        max_retries=s.jev_max_retries,
    )


def _llm_decider(config: ResolvedConfig, role: Literal["llm", "escalation"]) -> DecisionBackend:
    from graphwalk.decisions.llm_decider import LLMDecider  # noqa: PLC0415

    return LLMDecider(make_llm(config, role=role))


def make_decider(config: ResolvedConfig | None = None) -> DecisionBackend:
    """The decider walks use, per ``decider`` (and ``decision_fallback`` for Jev)."""
    config = config or resolve_config()
    s = config.settings
    if s.decider == "logprob":
        return make_logprob_decider(config)
    if s.decider == "llm":
        return _llm_decider(config, "llm")
    jev = _jev(config)
    if jev is not None:
        return jev
    if s.decision_fallback == "logprob":
        return make_logprob_decider(config)
    if s.decision_fallback == "llm":
        return _llm_decider(config, "llm")
    msg = (
        f"no key for the decision provider {s.decision_provider!r}: set "
        f"{env_name(s.decision_provider)}, or choose another decider: "
        "GRAPHWALK_DECIDER=logprob (token probabilities from any OpenAI-compatible "
        "endpoint) or GRAPHWALK_DECISION_FALLBACK=logprob to use it only without a Jev key"
    )
    raise ConfigError(msg)


def make_escalation_decider(config: ResolvedConfig | None = None) -> DecisionBackend:
    """The decider escalated walks use (``escalation_decider``)."""
    config = config or resolve_config()
    s = config.settings
    if s.escalation_decider == "logprob":
        return make_logprob_decider(config, model=s.escalation_model)
    return _llm_decider(config, "escalation")


def litellm_model(provider: ChatProvider, model: str) -> str:
    prefix = REGISTRY[provider].litellm_prefix
    assert prefix is not None  # noqa: S101 - every ChatProvider serves chat
    return f"{prefix}/{model}"


def make_llm(
    config: ResolvedConfig | None = None,
    *,
    role: Literal["llm", "escalation"] = "llm",
    max_tokens: int = 4096,
    max_rpm: float | None = None,
) -> LLMBackend:
    """The chat model for extraction (``llm``) or escalated routing (``escalation``)."""
    config = config or resolve_config()
    s = config.settings
    provider = s.llm_provider
    model = s.resolved_llm_model
    if role == "escalation" and s.escalation_model:
        model = s.escalation_model
    from graphwalk.llm.litellm_backend import LiteLLMBackend  # noqa: PLC0415 - the llm extra

    return LiteLLMBackend(
        litellm_model(provider, model),
        api_key=_key(config, provider, "llm"),
        api_base=s.base_url(provider) + REGISTRY[provider].version_path,
        max_tokens=max_tokens,
        max_rpm=max_rpm,
    )


def make_embedder(config: ResolvedConfig | None = None) -> Embedder:
    config = config or resolve_config()
    s = config.settings
    model = s.resolved_embedding_model
    if s.embedding_provider == "fastembed":
        from graphwalk.embeddings.fastembed_embedder import FastEmbedEmbedder  # noqa: PLC0415

        return FastEmbedEmbedder(model)
    provider: Provider = s.embedding_provider
    from graphwalk.embeddings.openai_compatible import OpenAICompatibleEmbedder  # noqa: PLC0415

    return OpenAICompatibleEmbedder(
        model,
        api_key=_key(config, provider, "embedding"),
        api_base=s.base_url(provider) + REGISTRY[provider].version_path,
        provider=provider,
    )
