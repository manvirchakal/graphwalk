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


def test_ingest_builds_and_updates_a_graph(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import json

    from graphwalk.llm import FakeLLM
    from graphwalk.stores.networkx_store import NetworkXStore

    reply = {
        "entities": [{"name": "Ada Lovelace", "type": "person"}, {"name": "London"}],
        "relations": [{"source": "Ada Lovelace", "type": "born_in", "target": "London"}],
    }
    llm = FakeLLM(lambda _m: json.dumps(reply))
    monkeypatch.setattr(cli, "_ingest_backends", lambda *_a: (llm, None, None))
    monkeypatch.setattr(cli, "_make_backend", FakeDecisionBackend)
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "ada.txt").write_text("Ada Lovelace was born in London.", encoding="utf-8")
    graph = tmp_path / "g.json"
    report = tmp_path / "report.json"
    args = ["ingest", str(docs), "--graph", str(graph), "--source-id", "docs"]
    result = runner.invoke(app, [*args, "--report", str(report)])
    assert result.exit_code == 0, result.output
    assert "1 new" in result.output
    assert "(2 nodes, 1 edges)" in result.output
    assert json.loads(report.read_text(encoding="utf-8"))["nodes_created"] == 2
    store = asyncio.run(NetworkXStore.load(graph))
    assert asyncio.run(store.get_metadata("ingest:ledger:docs")) is not None

    again = runner.invoke(app, args)
    assert again.exit_code == 0, again.output
    assert "1 unchanged" in again.output
    assert len(llm.calls) == 1

    bad = runner.invoke(app, [*args, "--routing", "nope"])
    assert bad.exit_code == 2
    missing = runner.invoke(app, ["ingest", str(tmp_path / "nope"), "--graph", str(graph)])
    assert missing.exit_code == 2
