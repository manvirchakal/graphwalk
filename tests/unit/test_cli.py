import asyncio
from pathlib import Path

import pytest
from typer.testing import CliRunner

from graphwalk import __version__, cli
from graphwalk.cli import app
from graphwalk.decisions import FakeDecisionBackend
from graphwalk.traversal import TraversalResult
from kg_fixtures import movie_store, oracle

runner = CliRunner()


def test_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.output.strip() == __version__


@pytest.mark.parametrize("command", ["ingest", "eval"])
def test_commands_exist_and_report_not_implemented(command: str) -> None:
    result = runner.invoke(app, [command])
    assert result.exit_code == 2
    assert "not implemented yet" in result.output


@pytest.fixture
def graph_file(tmp_path: Path) -> Path:
    path = tmp_path / "movies.json"
    asyncio.run(_save(path))
    return path


async def _save(path: Path) -> None:
    await (await movie_store()).save(path)


@pytest.fixture
def fake_backend(monkeypatch: pytest.MonkeyPatch) -> FakeDecisionBackend:
    backend = oracle(
        {
            "Inception": {"Christopher Nolan": 0.9, "Leonardo DiCaprio": 0.1},
            "Christopher Nolan": {"STOP": 0.9, "London": 0.1},
        },
        cost_per_call=0.001,
    )
    monkeypatch.setattr(cli, "_make_backend", lambda: backend)
    return backend


def test_query_resolves_start_and_prints_answers(
    graph_file: Path, fake_backend: FakeDecisionBackend, tmp_path: Path
) -> None:
    trace = tmp_path / "trace.json"
    result = runner.invoke(
        app,
        [
            "query",
            str(graph_file),
            "Who directed Inception?",
            "--strategy",
            "greedy",
            "--trace",
            str(trace),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "status: ok" in result.output
    assert "1. Christopher Nolan" in result.output
    assert "-> directed_by" in result.output
    assert "calls=2" in result.output
    assert "$0.002000" in result.output
    restored = TraversalResult.model_validate_json(trace.read_text(encoding="utf-8"))
    assert restored.best is not None
    assert restored.best.node_ids == ("nolan",)
    assert fake_backend.closed


def test_query_without_a_known_name_asks_for_start(
    graph_file: Path, fake_backend: FakeDecisionBackend
) -> None:
    result = runner.invoke(app, ["query", str(graph_file), "who knows?"])
    assert result.exit_code == 1
    assert "--start" in result.output
    assert fake_backend.calls == 0


def test_query_with_explicit_start_and_bad_options(
    graph_file: Path, fake_backend: FakeDecisionBackend
) -> None:
    ok = runner.invoke(
        app, ["query", str(graph_file), "q", "--start", "inception", "--max-depth", "1"]
    )
    assert ok.exit_code == 0, ok.output
    assert fake_backend.calls == 1
    bad = runner.invoke(app, ["query", str(graph_file), "q", "--strategy", "nope"])
    assert bad.exit_code == 2
    missing = runner.invoke(app, ["query", str(graph_file), "q", "--start", "ghost"])
    assert missing.exit_code == 1
    assert "ghost" in missing.output
