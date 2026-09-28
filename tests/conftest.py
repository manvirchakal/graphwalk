"""Shared pytest configuration.

Sockets are disabled for every test (``--disable-socket`` in pyproject). Tests marked
``live`` or ``neo4j`` are skipped unless explicitly enabled, and get network access back
when they are.
"""

import pytest

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
