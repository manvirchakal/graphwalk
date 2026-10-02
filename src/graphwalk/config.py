"""Runtime configuration: providers, keys, base URLs, and models per role.

Three roles, five providers:

========== ============================================ =================================
Role       Providers                                    Default
========== ============================================ =================================
decision   ``jev`` (via ``openrouter`` or ``typesafe``),   ``jev`` via ``openrouter``
           ``logprob`` (any OpenAI-compatible endpoint
           with token log-probabilities), or ``llm``
           (a chat model's stated scores)
llm        ``openrouter``, ``openai``, ``anthropic``,     ``openrouter``
           ``xai``
embedding  ``fastembed`` (local), ``openai``,           ``fastembed``
           ``openrouter``
========== ============================================ =================================

Values come from four places. :func:`resolve_config` applies them in this order, later
winning: defaults, the environment (plus an optional ``.env`` file), request headers
(remote MCP sessions), and explicit arguments. Keys are :class:`SecretStr` everywhere,
so they never show up in reprs, logs, or traces.

Environment names: provider keys and base URLs use their conventional unprefixed names
(``OPENAI_API_KEY``, ``ANTHROPIC_BASE_URL``, ...) and also accept a ``GRAPHWALK_``
prefix. Everything else uses the prefix (``GRAPHWALK_LLM_MODEL``). Base URLs follow
each provider's SDK convention: OpenAI and x.ai include ``/v1``; Anthropic, OpenRouter,
and TypeSafe do not.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, cast, get_args
from urllib.parse import urlsplit

from pydantic import AliasChoices, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

from graphwalk.core.errors import GraphwalkError

type Provider = Literal["openrouter", "typesafe", "openai", "anthropic", "xai"]
type DecisionProvider = Literal["openrouter", "typesafe"]
type ChatProvider = Literal["openrouter", "openai", "anthropic", "xai"]
type EmbeddingProvider = Literal["fastembed", "openai", "openrouter"]
type DecisionFallback = Literal["off", "llm", "logprob"]
type DeciderKind = Literal["jev", "logprob", "llm"]
type EscalationDecider = Literal["llm", "logprob"]
type Source = Literal["argument", "header", "environment", "default"]

PROVIDERS: tuple[Provider, ...] = get_args(Provider.__value__)

# Pinned model ids per provider. Never use a moving alias such as ``jev-latest``.
JEV_MODEL_TYPESAFE = "jev-1.13.0"
JEV_MODEL_OPENROUTER = "typesafe/jev-1.13"

TYPESAFE_BASE_URL = "https://api.typesafe.ai"
OPENROUTER_BASE_URL = "https://openrouter.ai/api"
OPENAI_BASE_URL = "https://api.openai.com/v1"
ANTHROPIC_BASE_URL = "https://api.anthropic.com"
XAI_BASE_URL = "https://api.x.ai/v1"

DEFAULT_LLM_MODELS: dict[ChatProvider, str | None] = {
    "openrouter": "openai/gpt-6-luna",
    "openai": "gpt-6-luna",
    "anthropic": "claude-haiku-4-5-20251001",
    "xai": None,  # no default: set GRAPHWALK_LLM_MODEL
}
"""Extraction and escalation defaults: cheap models, since ingestion makes many calls."""

DEFAULT_EMBEDDING_MODELS: dict[EmbeddingProvider, str] = {
    "fastembed": "BAAI/bge-small-en-v1.5",
    "openai": "text-embedding-3-small",
    "openrouter": "openai/text-embedding-3-small",
}


class ConfigError(GraphwalkError):
    """The configuration cannot provide what was asked for (e.g. a missing key)."""


def _key(env: str) -> Any:
    return Field(default=None, validation_alias=AliasChoices(env, f"GRAPHWALK_{env}"))


def _url(env: str, default: str) -> Any:
    return Field(default=default, validation_alias=AliasChoices(env, f"GRAPHWALK_{env}"))


class GraphwalkSettings(BaseSettings):
    """Every setting, read from the environment (and ``.env``) when constructed.

    :func:`settings_from` builds one from explicit values only, without reading the
    environment; :func:`resolve_config` uses that to layer sources.
    """

    model_config = SettingsConfigDict(
        env_prefix="GRAPHWALK_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    # -- keys and base URLs, per provider
    openrouter_api_key: SecretStr | None = _key("OPENROUTER_API_KEY")
    typesafe_api_key: SecretStr | None = _key("TYPESAFE_API_KEY")
    openai_api_key: SecretStr | None = _key("OPENAI_API_KEY")
    anthropic_api_key: SecretStr | None = _key("ANTHROPIC_API_KEY")
    xai_api_key: SecretStr | None = _key("XAI_API_KEY")
    openrouter_base_url: str = _url("OPENROUTER_BASE_URL", OPENROUTER_BASE_URL)
    typesafe_base_url: str = _url("TYPESAFE_BASE_URL", TYPESAFE_BASE_URL)
    openai_base_url: str = _url("OPENAI_BASE_URL", OPENAI_BASE_URL)
    anthropic_base_url: str = _url("ANTHROPIC_BASE_URL", ANTHROPIC_BASE_URL)
    xai_base_url: str = _url("XAI_BASE_URL", XAI_BASE_URL)

    # -- decision role
    decider: DeciderKind = "jev"
    """Which decider walks: ``jev``; ``logprob`` (a chat model's token probabilities,
    any OpenAI-compatible endpoint; see :mod:`graphwalk.decisions.logprob`); or ``llm``
    (a chat model's stated scores; not calibrated). Or pass your own ``DecisionBackend``
    to ``Index.open(decider=...)``."""
    decider_model: str | None = None
    """For ``logprob``: the model id at the endpoint. ``None``: an open-weights default
    through OpenRouter; required with ``decider_base_url``."""
    decider_base_url: str | None = None
    """For ``logprob``: an OpenAI-compatible API root including ``/v1``
    (``http://localhost:8000/v1`` for vLLM). ``None``: OpenRouter."""
    decider_api_key: SecretStr | None = None
    """For ``logprob`` at ``decider_base_url``; ``None`` for a local server. Through
    OpenRouter, ``OPENROUTER_API_KEY`` is used."""
    decider_max_rpm: float | None = Field(default=None, gt=0)
    """For ``logprob``: space request starts to this rate (requests per minute)."""
    decision_provider: DecisionProvider = "openrouter"
    """Where Jev is served (``decider="jev"``)."""
    decision_fallback: DecisionFallback = "off"
    """With ``decider="jev"`` and no Jev key: ``logprob`` or ``llm`` decides instead.
    Off by default so nobody gets another decider by accident."""
    jev_model_openrouter: str = JEV_MODEL_OPENROUTER
    jev_model_typesafe: str = JEV_MODEL_TYPESAFE
    escalate_below: float | None = Field(default=None, gt=0.0, le=1.0)
    """Re-walk with the escalation decider when the best answer's confidence is below
    this (``Index.walk``; see :mod:`graphwalk.traversal.escalation`). ``None``: never."""
    escalation_decider: EscalationDecider = "llm"
    """What escalated walks decide with: ``llm`` (the chat model's stated scores) or
    ``logprob`` (``escalation_model`` or ``decider_model`` at the logprob endpoint, so
    the escalated answer's confidence is a probability too)."""
    jev_timeout_s: float = Field(default=10.0, gt=0)
    jev_max_retries: int = Field(default=2, ge=0)

    # -- llm role (extraction, escalation, the fallback decider)
    llm_provider: ChatProvider = "openrouter"
    llm_model: str | None = None
    """Model id as the provider names it; ``None`` = the provider's default."""
    escalation_model: str | None = None
    """For escalated routing decisions (same provider); ``None`` = ``llm_model``."""

    # -- embedding role
    embedding_provider: EmbeddingProvider = "fastembed"
    embedding_model: str | None = None

    @field_validator(
        "openrouter_base_url",
        "typesafe_base_url",
        "openai_base_url",
        "anthropic_base_url",
        "xai_base_url",
        "decider_base_url",
    )
    @classmethod
    def _http_url(cls, url: str | None) -> str | None:
        if url is None:
            return None
        url = url.strip().rstrip("/")
        if not url.startswith(("https://", "http://")):
            msg = f"base URL must start with http:// or https://, got {url!r}"
            raise ValueError(msg)
        return url

    # -- per-provider access

    def api_key(self, provider: Provider) -> SecretStr | None:
        key: SecretStr | None = getattr(self, f"{provider}_api_key")
        return key

    def base_url(self, provider: Provider) -> str:
        url: str = getattr(self, f"{provider}_base_url")
        return url

    # -- decision role (kept for existing callers)

    @property
    def jev_model(self) -> str:
        """The pinned Jev model id for the selected provider."""
        if self.decision_provider == "openrouter":
            return self.jev_model_openrouter
        return self.jev_model_typesafe

    @property
    def decision_base_url(self) -> str:
        return self.base_url(self.decision_provider)

    @property
    def decision_api_key(self) -> SecretStr | None:
        return self.api_key(self.decision_provider)

    # -- llm and embedding roles

    @property
    def resolved_llm_model(self) -> str:
        model = self.llm_model or DEFAULT_LLM_MODELS[self.llm_provider]
        if model is None:
            msg = (
                f"llm_provider={self.llm_provider!r} has no default model; set GRAPHWALK_LLM_MODEL"
            )
            raise ConfigError(msg)
        return model

    @property
    def resolved_embedding_model(self) -> str:
        return self.embedding_model or DEFAULT_EMBEDDING_MODELS[self.embedding_provider]


class _ExplicitSettings(GraphwalkSettings):
    """Reads nothing from the environment: only the values it is given."""

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        del settings_cls, env_settings, dotenv_settings, file_secret_settings
        return (init_settings,)


def settings_from(values: Mapping[str, object] | None = None) -> GraphwalkSettings:
    """Settings from ``values`` (field names) and defaults only; the environment and
    ``.env`` are not read."""
    return _ExplicitSettings(**cast("dict[str, Any]", dict(values or {})))


def _env_names(name: str) -> tuple[str, ...]:
    alias = GraphwalkSettings.model_fields[name].validation_alias
    if isinstance(alias, AliasChoices):
        return tuple(str(choice) for choice in alias.choices)
    return (f"GRAPHWALK_{name.upper()}",)


FIELD_NAMES: dict[str, str] = {
    env: name for name in GraphwalkSettings.model_fields for env in _env_names(name)
}
"""Every accepted environment (and header) name -> settings field."""

SECRET_FIELDS = frozenset({*(f"{p}_api_key" for p in PROVIDERS), "decider_api_key"})
BASE_URL_FIELDS = frozenset({*(f"{p}_base_url" for p in PROVIDERS), "decider_base_url"})


def header_field(header: str) -> str | None:
    """The settings field a request header sets, if any.

    Header names are the environment names, matched ignoring case, with ``-`` for
    ``_`` and an optional ``X-`` prefix: ``OPENAI_API_KEY``, ``openai-api-key``, and
    ``X-OpenAI-API-Key`` all set ``openai_api_key``. (Use hyphens: proxies such as
    nginx drop headers with underscores by default.)
    """
    name = header.strip().upper().replace("-", "_")
    return FIELD_NAMES.get(name) or FIELD_NAMES.get(name.removeprefix("X_"))


def url_allowed(url: str, allowed: tuple[str, ...]) -> bool:
    """``url`` matches an allowlist entry: same scheme and host (and port), and under
    the entry's path."""
    target = urlsplit(url)
    for entry in allowed:
        rule = urlsplit(entry.strip().rstrip("/"))
        if (target.scheme, target.netloc.lower()) != (rule.scheme, rule.netloc.lower()):
            continue
        if target.path.rstrip("/").startswith(rule.path):
            return True
    return False


@dataclass(frozen=True)
class ResolvedConfig:
    """Settings after layering, with where each value came from."""

    settings: GraphwalkSettings
    sources: Mapping[str, Source]
    warnings: tuple[str, ...] = field(default=())

    def describe(self) -> dict[str, object]:
        """A summary safe to log or show: providers, models, and which keys are
        present, never the keys themselves."""
        s = self.settings
        return {
            "decision": {
                "decider": s.decider,
                "provider": s.decision_provider,
                "model": s.jev_model,
                "decider_model": s.decider_model,
                "decider_base_url": s.decider_base_url,
                "decider_key": s.decider_api_key is not None,
                "fallback": s.decision_fallback,
                "escalate_below": s.escalate_below,
                "escalation_decider": s.escalation_decider,
            },
            "llm": {
                "provider": s.llm_provider,
                "model": s.llm_model or DEFAULT_LLM_MODELS[s.llm_provider],
                "escalation_model": s.escalation_model,
            },
            "embedding": {"provider": s.embedding_provider, "model": s.resolved_embedding_model},
            "keys": {p: s.api_key(p) is not None for p in PROVIDERS},
            "base_urls": {p: s.base_url(p) for p in PROVIDERS},
            "sources": dict(self.sources),
            "warnings": list(self.warnings),
        }


def resolve_config(
    arguments: Mapping[str, object] | None = None,
    *,
    headers: Mapping[str, str] | None = None,
    environment: GraphwalkSettings | None = None,
    env_keys: bool = True,
    allowed_base_urls: tuple[str, ...] = (),
) -> ResolvedConfig:
    """Layer the configuration: defaults < environment < headers < ``arguments``.

    * ``arguments`` are settings field names (``llm_model="..."``); unknown names raise.
    * ``headers`` (remote MCP sessions) use the environment names; see
      :func:`header_field`. Unknown headers are ignored. A base URL from a header is
      used only if it matches ``allowed_base_urls``, so a client cannot point the server
      at an internal host; otherwise it is dropped with a warning.
    * ``environment`` defaults to ``GraphwalkSettings()`` (the process environment and
      ``.env``). With ``env_keys=False`` its API keys are ignored: a remote server then
      uses only keys its clients send.
    """
    env = GraphwalkSettings() if environment is None else environment
    values: dict[str, object] = {
        name: getattr(env, name) for name in GraphwalkSettings.model_fields
    }
    sources: dict[str, Source] = {
        name: "environment" if name in env.model_fields_set else "default" for name in values
    }
    warnings: list[str] = []
    if not env_keys:
        for name in SECRET_FIELDS:
            values[name] = None
            sources[name] = "default"
    for header, value in (headers or {}).items():
        name = header_field(header)
        if name is None:
            continue
        if name in BASE_URL_FIELDS and not url_allowed(value.strip(), allowed_base_urls):
            warnings.append(f"ignored {name} from a header: not in GRAPHWALK_ALLOWED_BASE_URLS")
            continue
        values[name] = value
        sources[name] = "header"
    for name, value in (arguments or {}).items():
        if name not in GraphwalkSettings.model_fields:
            msg = f"unknown setting {name!r}"
            raise ConfigError(msg)
        values[name] = value
        sources[name] = "argument"
    try:
        settings = settings_from(values)
    except ValueError as error:
        raise ConfigError(str(error)) from error
    return ResolvedConfig(settings=settings, sources=sources, warnings=tuple(warnings))


def env_name(provider: Provider, what: Literal["key", "base_url"] = "key") -> str:
    """The environment variable for a provider's key or base URL."""
    return f"{provider.upper()}_{'API_KEY' if what == 'key' else 'BASE_URL'}"
