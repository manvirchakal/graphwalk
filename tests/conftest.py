"""Shared pytest configuration.

Sockets are disabled for every test (``--disable-socket`` in pyproject). Tests marked
``live`` or ``neo4j`` are skipped unless explicitly enabled, and get network access back
when they are.
"""

import os

import pytest

from graphwalk.config import FIELD_NAMES, GraphwalkSettings

# Before anything imports LiteLLM: it would otherwise fetch its price map on import.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

_OPT_IN_MARKERS = {"live": "--run-live", "neo4j": "--run-neo4j"}


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--run-live", action="store_true", help="run tests that call real APIs")
    parser.addoption("--run-neo4j", action="store_true", help="run tests that need Neo4j")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    for item in items:
        for marker, flag in _OPT_IN_MARKERS.items():
            if marker not in item.keywords:
                continue
            if config.getoption(flag):
                item.add_marker(pytest.mark.enable_socket)
            else:
                item.add_marker(pytest.mark.skip(reason=f"needs {flag}"))


@pytest.fixture(autouse=True)
def _hermetic_config(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Offline tests never see the developer's provider keys, base URLs, or settings
    (a real key would otherwise be picked up, and some hosts set ANTHROPIC_BASE_URL).
    ``.env`` files are ignored the same way. Live tests keep the real environment."""
    if request.node.get_closest_marker("live") is not None:
        return
    for name in FIELD_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setitem(GraphwalkSettings.model_config, "env_file", None)
