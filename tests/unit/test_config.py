from pathlib import Path

import pytest

from graphwalk.config import (
    JEV_MODEL_OPENROUTER,
    JEV_MODEL_TYPESAFE,
    OPENROUTER_BASE_URL,
    TYPESAFE_BASE_URL,
    GraphwalkSettings,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for var in ("OPENROUTER_API_KEY", "TYPESAFE_API_KEY", "GRAPHWALK_DECISION_PROVIDER"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.chdir(tmp_path)  # no stray .env


def test_defaults_to_openrouter_with_pinned_model() -> None:
    settings = GraphwalkSettings()
    assert settings.decision_provider == "openrouter"
    assert settings.jev_model == JEV_MODEL_OPENROUTER == "typesafe/jev-1.13"
    assert settings.decision_base_url == OPENROUTER_BASE_URL
    assert settings.decision_api_key is None


def test_typesafe_provider_selects_its_key_model_and_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GRAPHWALK_DECISION_PROVIDER", "typesafe")
    monkeypatch.setenv("TYPESAFE_API_KEY", "ts-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    settings = GraphwalkSettings()
    assert settings.jev_model == JEV_MODEL_TYPESAFE == "jev-1.13.0"
    assert settings.decision_base_url == TYPESAFE_BASE_URL
    key = settings.decision_api_key
    assert key is not None
    assert key.get_secret_value() == "ts-key"


def test_api_key_is_not_leaked_in_repr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "super-secret")
    assert "super-secret" not in repr(GraphwalkSettings())


def test_unknown_provider_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GRAPHWALK_DECISION_PROVIDER", "nope")
    with pytest.raises(ValueError, match="decision_provider"):
        GraphwalkSettings()
