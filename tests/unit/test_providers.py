"""Offline matrix: every provider x role x configuration path, with fakes.

LiteLLM is exercised for real up to the network call, which is monkeypatched to record
what it would have sent.
"""

import json
import logging
from types import SimpleNamespace
from typing import Any

import pytest

from graphwalk.config import (
    PROVIDERS,
    ConfigError,
    GraphwalkSettings,
    Provider,
    header_field,
    resolve_config,
    settings_from,
    url_allowed,
)
from graphwalk.core.redact import redact
from graphwalk.decisions import ChoiceQuestion, DecisionBackendError, DecisionRequest
from graphwalk.decisions.jev import JevBackend
from graphwalk.decisions.llm_decider import LLMDecider, parse_scores
from graphwalk.decisions.logprob import DEFAULT_OPENROUTER_MODEL, LogprobDecider
from graphwalk.llm import FakeLLM, LLMError, Message
from graphwalk.llm.litellm_import import import_litellm
from graphwalk.providers import (
    REGISTRY,
    make_decider,
    make_embedder,
    make_escalation_decider,
    make_llm,
    supports,
)

SECRET = "sk-test-SECRET-0123456789abcdef"  # noqa: S105 - a fake key

ROLES = ("decision", "llm", "embedding")
ROLE_FIELD = {
    "decision": "decision_provider",
    "llm": "llm_provider",
    "embedding": "embedding_provider",
}


def config(**values: object) -> Any:
    return resolve_config(values, environment=settings_from())


# -- precedence -------------------------------------------------------------------------


def test_precedence_argument_over_header_over_environment_over_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert resolve_config().sources["llm_model"] == "default"
    monkeypatch.setenv("GRAPHWALK_LLM_MODEL", "from-env")
    env = resolve_config()
    assert (env.settings.llm_model, env.sources["llm_model"]) == ("from-env", "environment")
    headers = {"X-Graphwalk-LLM-Model": "from-header"}
    hdr = resolve_config(headers=headers)
    assert (hdr.settings.llm_model, hdr.sources["llm_model"]) == ("from-header", "header")
    arg = resolve_config({"llm_model": "from-arg"}, headers=headers)
    assert (arg.settings.llm_model, arg.sources["llm_model"]) == ("from-arg", "argument")


@pytest.mark.parametrize("provider", PROVIDERS)
def test_every_provider_key_and_base_url_from_env_header_and_argument(
    monkeypatch: pytest.MonkeyPatch, provider: Provider
) -> None:
    upper = provider.upper()
    monkeypatch.setenv(f"{upper}_API_KEY", "env-key")
    monkeypatch.setenv(f"GRAPHWALK_{upper}_BASE_URL", "https://env.example/x/")
    env = resolve_config().settings
    assert env.api_key(provider).get_secret_value() == "env-key"  # type: ignore[union-attr]
    assert env.base_url(provider) == "https://env.example/x"
    header = resolve_config(
        headers={
            f"{provider}-api-key": "hdr-key",
            f"x-{provider}-base-url": "https://ok.example/v",
        },
        allowed_base_urls=("https://ok.example",),
    ).settings
    assert header.api_key(provider).get_secret_value() == "hdr-key"  # type: ignore[union-attr]
    assert header.base_url(provider) == "https://ok.example/v"
    arg = resolve_config({f"{provider}_api_key": "arg-key"}).settings
    assert arg.api_key(provider).get_secret_value() == "arg-key"  # type: ignore[union-attr]


def test_header_names_forms() -> None:
    for header in (
        "OPENAI_API_KEY",
        "openai-api-key",
        "X-OpenAI-API-Key",
        "graphwalk-openai-api-key",
    ):
        assert header_field(header) == "openai_api_key"
    assert header_field("GRAPHWALK_EMBEDDING_PROVIDER") == "embedding_provider"
    assert header_field("Authorization") is None
    assert header_field("content-type") is None


def test_header_base_urls_need_the_allowlist() -> None:
    blocked = resolve_config(headers={"openai-base-url": "http://169.254.169.254/latest"})
    assert blocked.settings.openai_base_url == "https://api.openai.com/v1"
    assert blocked.warnings
    assert url_allowed("https://gw.example/v1", ("https://gw.example",))
    assert url_allowed("https://gw.example/team/v1", ("https://gw.example/team",))
    assert not url_allowed("https://gw.example.evil/v1", ("https://gw.example",))
    assert not url_allowed("http://gw.example/v1", ("https://gw.example",))
    assert not url_allowed("https://gw.example/other", ("https://gw.example/team",))


def test_env_keys_can_be_disabled_for_remote_servers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "server-key")
    monkeypatch.setenv("GRAPHWALK_LLM_MODEL", "server-model")
    remote = resolve_config(headers={"anthropic-api-key": "client-key"}, env_keys=False)
    assert remote.settings.openrouter_api_key is None
    assert remote.settings.anthropic_api_key is not None
    assert remote.settings.llm_model == "server-model"  # non-secret settings still apply


def test_invalid_values_raise_config_error() -> None:
    with pytest.raises(ConfigError, match="unknown setting"):
        resolve_config({"llm_modle": "x"})
    with pytest.raises(ConfigError, match="llm_provider"):
        resolve_config({"llm_provider": "typesafe"})
    with pytest.raises(ConfigError, match="must start with"):
        resolve_config({"openai_base_url": "api.openai.com"})
    with pytest.raises(ConfigError, match="no default model"):
        _ = config(llm_provider="xai").settings.resolved_llm_model


# -- provider x role matrix -------------------------------------------------------------


@pytest.mark.parametrize("role", ROLES)
@pytest.mark.parametrize("provider", PROVIDERS)
def test_role_support_matrix(provider: Provider, role: str) -> None:
    """A provider is accepted for a role exactly when the registry says it serves it."""
    field = ROLE_FIELD[role]
    if supports(provider, role):  # type: ignore[arg-type]
        assert getattr(config(**{field: provider}).settings, field) == provider
    else:
        with pytest.raises(ConfigError):
            config(**{field: provider})


def test_registry_covers_every_provider() -> None:
    assert set(REGISTRY) == set(PROVIDERS)
    assert GraphwalkSettings.model_fields.keys() >= {f"{p}_api_key" for p in PROVIDERS}


@pytest.mark.parametrize("provider", ["openrouter", "typesafe"])
def test_decision_role_builds_jev_with_key_model_and_base_url(provider: str) -> None:
    decider = make_decider(
        config(
            decision_provider=provider,
            **{f"{provider}_api_key": SECRET, f"{provider}_base_url": "https://gw.example/jev"},
        )
    )
    assert isinstance(decider, JevBackend)
    assert decider.provider == provider
    assert decider.model_id == ("typesafe/jev-1.13" if provider == "openrouter" else "jev-1.13.0")
    assert decider._base_url == "https://gw.example/jev"


def test_decision_fallback_is_opt_in() -> None:
    with pytest.raises(ConfigError, match="GRAPHWALK_DECIDER=logprob"):
        make_decider(config(openai_api_key=SECRET, llm_provider="openai"))
    pytest.importorskip("litellm")
    fallback = make_decider(
        config(decision_fallback="llm", llm_provider="anthropic", anthropic_api_key=SECRET)
    )
    assert isinstance(fallback, LLMDecider)
    assert fallback.model_id == "llm-decider:anthropic/claude-haiku-4-5-20251001"
    # A Jev key wins over the fallback flag.
    jev = make_decider(config(decision_fallback="llm", openrouter_api_key=SECRET))
    assert isinstance(jev, JevBackend)


def test_logprob_decider_defaults_to_an_open_model_on_openrouter() -> None:
    decider = make_decider(config(decider="logprob", openrouter_api_key=SECRET))
    assert isinstance(decider, LogprobDecider)
    assert decider.model_id == f"logprob:{DEFAULT_OPENROUTER_MODEL}"
    with pytest.raises(ConfigError, match="OPENROUTER_API_KEY"):
        make_decider(config(decider="logprob"))


def test_logprob_decider_at_a_local_endpoint_needs_a_model_not_a_key() -> None:
    local = config(decider="logprob", decider_base_url="http://localhost:8000/v1")
    with pytest.raises(ConfigError, match="GRAPHWALK_DECIDER_MODEL"):
        make_decider(local)
    decider = make_decider(
        config(decider="logprob", decider_base_url="http://localhost:8000/v1", decider_model="m")
    )
    assert decider.model_id == "logprob:m"


def test_logprob_is_a_fallback_for_jev_and_an_escalation_decider() -> None:
    local = {"decider_base_url": "http://localhost:8000/v1", "decider_model": "m"}
    fallback = make_decider(config(decision_fallback="logprob", **local))
    assert fallback.model_id == "logprob:m"  # no Jev key: the fallback decides
    jev = make_decider(config(decision_fallback="logprob", openrouter_api_key=SECRET, **local))
    assert isinstance(jev, JevBackend)
    escalation = make_escalation_decider(
        config(escalation_decider="logprob", escalation_model="big", **local)
    )
    assert escalation.model_id == "logprob:big"


def test_decider_endpoint_from_headers_is_allowlisted_and_its_key_is_secret() -> None:
    resolved = resolve_config(
        headers={"X-GraphWalk-Decider-Base-URL": "http://169.254.169.254/v1"},
        environment=settings_from(),
    )
    assert resolved.settings.decider_base_url is None
    assert resolved.warnings
    env = settings_from({"decider_api_key": SECRET})
    assert resolve_config(environment=env, env_keys=False).settings.decider_api_key is None
    assert SECRET not in json.dumps(resolve_config(environment=env).describe())


def recorder(monkeypatch: pytest.MonkeyPatch, name: str, reply: Any) -> list[dict[str, Any]]:
    pytest.importorskip("litellm")
    litellm = import_litellm()
    sent: list[dict[str, Any]] = []

    async def fake(**kwargs: Any) -> Any:
        sent.append(kwargs)
        if isinstance(reply, Exception):
            raise reply
        return reply

    monkeypatch.setattr(litellm, name, fake)
    return sent


CHAT_REPLY = SimpleNamespace(
    choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
    usage=SimpleNamespace(prompt_tokens=3, completion_tokens=1, cost=None),
    model="served-model",
)

CHAT_CASES = [
    ("openrouter", None, "openrouter/openai/gpt-6-luna", "https://openrouter.ai/api/v1"),
    ("openai", None, "openai/gpt-6-luna", "https://api.openai.com/v1"),
    ("anthropic", None, "anthropic/claude-haiku-4-5-20251001", "https://api.anthropic.com"),
    ("xai", "grok-test", "xai/grok-test", "https://api.x.ai/v1"),
]


@pytest.mark.parametrize(("provider", "model", "litellm_model", "api_base"), CHAT_CASES)
async def test_llm_role_sends_model_key_and_base_url(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    model: str | None,
    litellm_model: str,
    api_base: str,
) -> None:
    sent = recorder(monkeypatch, "acompletion", CHAT_REPLY)
    values: dict[str, object] = {"llm_provider": provider, f"{provider}_api_key": SECRET}
    if model:
        values["llm_model"] = model
    llm = make_llm(config(**values))
    result = await llm.complete([Message(role="user", content="hi")])
    assert result.model == "served-model"
    (call,) = sent
    assert (call["model"], call["api_key"], call["api_base"]) == (litellm_model, SECRET, api_base)


async def test_llm_base_url_override_and_escalation_model(monkeypatch: pytest.MonkeyPatch) -> None:
    sent = recorder(monkeypatch, "acompletion", CHAT_REPLY)
    cfg = config(
        llm_provider="openrouter",
        openrouter_api_key=SECRET,
        openrouter_base_url="https://gw.example/or",
        escalation_model="big/model",
    )
    await make_llm(cfg, role="escalation").complete([Message(role="user", content="hi")])
    assert sent[0]["model"] == "openrouter/big/model"
    assert sent[0]["api_base"] == "https://gw.example/or/v1"


def test_llm_role_without_key_names_the_variable() -> None:
    with pytest.raises(ConfigError, match="set ANTHROPIC_API_KEY"):
        make_llm(config(llm_provider="anthropic"))


@pytest.mark.parametrize(
    ("provider", "model", "api_base"),
    [
        ("openai", "text-embedding-3-small", "https://api.openai.com/v1"),
        ("openrouter", "openai/text-embedding-3-small", "https://openrouter.ai/api/v1"),
    ],
)
async def test_remote_embedding_role(
    monkeypatch: pytest.MonkeyPatch, provider: str, model: str, api_base: str
) -> None:
    reply = SimpleNamespace(
        data=[{"index": 1, "embedding": [0.0, 2.0]}, {"index": 0, "embedding": [3.0, 4.0]}]
    )
    sent = recorder(monkeypatch, "aembedding", reply)
    embedder = make_embedder(config(embedding_provider=provider, **{f"{provider}_api_key": SECRET}))
    assert embedder.model_id == f"{provider}:{model}"
    vectors = await embedder.embed(["a", "b"])
    assert vectors.tolist() == [[0.6000000238418579, 0.800000011920929], [0.0, 1.0]]
    (call,) = sent
    assert (call["model"], call["api_key"], call["api_base"]) == (
        f"openai/{model}",
        SECRET,
        api_base,
    )


def test_local_embedding_role_is_the_default() -> None:
    pytest.importorskip("fastembed")
    from graphwalk.embeddings.fastembed_embedder import FastEmbedEmbedder

    built: list[str] = []
    original = FastEmbedEmbedder.__init__

    def spy(self: FastEmbedEmbedder, model_name: str = "", **kw: Any) -> None:
        built.append(model_name)
        raise RuntimeError("stop before loading the model")

    FastEmbedEmbedder.__init__ = spy  # type: ignore[method-assign]
    try:
        with pytest.raises(RuntimeError, match="stop"):
            make_embedder(config())
    finally:
        FastEmbedEmbedder.__init__ = original  # type: ignore[method-assign]
    assert built == ["BAAI/bge-small-en-v1.5"]


# -- secrets ----------------------------------------------------------------------------


def test_redact() -> None:
    assert redact(f"bad key {SECRET}!", [SECRET]) == "bad key ***!"
    assert redact(f"key ending ...{SECRET[-8:]}", [SECRET]) == "key ending ...***"
    assert redact("short abc", ["abc"]) == "short abc"
    assert redact("x", [None]) == "x"


async def test_secrets_never_reach_errors_logs_or_descriptions(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    recorder(monkeypatch, "acompletion", RuntimeError(f"401: invalid key {SECRET}"))
    cfg = config(llm_provider="openai", openai_api_key=SECRET, decision_fallback="llm")
    shown = [repr(cfg), str(cfg.describe()), repr(cfg.settings), cfg.settings.model_dump_json()]
    with pytest.raises(LLMError) as raised:
        await make_llm(cfg).complete([Message(role="user", content="hi")])
    shown.append(str(raised.value))
    assert raised.value.__cause__ is None  # the provider exception (and request) is dropped
    decider = make_decider(cfg)
    request = DecisionRequest(
        state="s",
        questions=(ChoiceQuestion(key="q", instructions="i", options={"a": None, "b": None}),),
    )
    with pytest.raises(DecisionBackendError) as failed:
        await decider.decide(request)
    shown += [str(failed.value), caplog.text]
    for text in shown:
        assert SECRET not in text
        assert SECRET[-8:] not in text


# -- the LLM decider --------------------------------------------------------------------

QUESTION = ChoiceQuestion(
    key="q1", instructions="Pick", options={"A": "first", "B": None, "C": None}
)
REQUEST = DecisionRequest(state={"query": "x"}, questions=(QUESTION,))


def scripted(*replies: str) -> FakeLLM:
    queue = list(replies)
    return FakeLLM(lambda _m: queue.pop(0), cost_per_call=0.002)


def test_parse_scores_tolerates_fences_missing_and_unknown_labels() -> None:
    results = parse_scores(REQUEST, '```json\n{"q1": {"A": 60, "B": 20, "Z": 99}}\n```')
    assert results["q1"].probabilities == {"A": 0.75, "B": 0.25, "C": 0.0}
    assert results["q1"].top == "A"
    uniform = parse_scores(REQUEST, '{"q1": {"A": 0, "B": -5, "C": "high"}}')
    assert uniform["q1"].probabilities == pytest.approx(dict.fromkeys("ABC", 1 / 3))
    with pytest.raises(ValueError, match="no scores"):
        parse_scores(REQUEST, '{"other": {}}')
    with pytest.raises(ValueError, match="no JSON"):
        parse_scores(REQUEST, "A")


async def test_llm_decider_prompt_retry_usage_and_marking() -> None:
    llm = scripted("I think A.", json.dumps({"q1": {"A": 1, "B": 3, "C": 0}}))
    decider = LLMDecider(llm)
    response = await decider.decide(REQUEST)
    assert response.results["q1"].probabilities == {"A": 0.25, "B": 0.75, "C": 0.0}
    assert response.model == "llm-decider:fake-llm-1"
    assert response.provider == "llm-fallback"
    assert response.usage.cost_usd == pytest.approx(0.004)  # both attempts are paid for
    payload = json.loads(llm.calls[0][1].content)
    assert payload["questions"][0]["options"] == {"A": "first", "B": None, "C": None}
    assert "not valid" in llm.calls[1][-1].content


async def test_llm_decider_gives_up_and_enforces_max_options() -> None:
    with pytest.raises(DecisionBackendError, match="no usable scores after 2"):
        await LLMDecider(scripted("no", "still no")).decide(REQUEST)
    with pytest.raises(DecisionBackendError, match="at most 2"):
        await LLMDecider(scripted(), max_options=2).decide(REQUEST)
    unknown_cost = FakeLLM(lambda _m: '{"q1": {"A": 1}}')
    assert (await LLMDecider(unknown_cost).decide(REQUEST)).usage.cost_usd is None
