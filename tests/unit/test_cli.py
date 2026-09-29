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


def test_ingest_into_sqlite_then_locate_and_migrate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json

    from test_index import PIERRE, QUERY, extract, walker

    from graphwalk.embeddings import FakeEmbedder
    from graphwalk.llm import FakeLLM

    monkeypatch.setattr(cli, "_ingest_backends", lambda *_a: (FakeLLM(extract), None, None))
    monkeypatch.setattr(cli, "_make_backend", walker)
    monkeypatch.setattr(cli, "_locate_embedder", lambda _m: FakeEmbedder())
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "pierre.txt").write_text(PIERRE, encoding="utf-8")
    db = tmp_path / "g.db"
    args = ["ingest", str(docs), "--graph", str(db), "--routing", "exact", "--source-id", "d"]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    assert "(2 nodes, 1 edges)" in result.output

    query = "Where was Pierre Curie born?"
    located = runner.invoke(app, ["locate", str(db), query, "-k", "2"])
    assert located.exit_code == 0, located.output
    assert "1. d/pierre.txt" in located.output
    assert "via Pierre Curie --born_in--> Paris" in located.output
    assert "Pierre Curie was born in Paris, France, in 1859." in located.output
    as_json = runner.invoke(app, ["locate", str(db), query, "--json", "--mode", "hybrid"])
    assert as_json.exit_code == 0, as_json.output
    first = json.loads(as_json.output.splitlines()[0])
    assert first["location"]["via"] == "hybrid"
    nothing = runner.invoke(app, ["locate", str(db), QUERY.replace("Marie", "Nobody")])
    assert nothing.exit_code == 0
    assert runner.invoke(app, ["locate", str(db), query, "--mode", "x"]).exit_code == 2
    assert runner.invoke(app, ["locate", str(tmp_path / "no.db"), query]).exit_code == 2

    graph = tmp_path / "g.json"
    json_args = ["ingest", str(docs), "--graph", str(graph), "--routing", "exact"]
    assert runner.invoke(app, json_args).exit_code == 0
    target = tmp_path / "migrated.db"
    migrated = runner.invoke(app, ["migrate", str(graph), str(target)])
    assert migrated.exit_code == 0, migrated.output
    assert "2 nodes, 1 edges, 1 metadata entries, 1 documents" in migrated.output
    exists = runner.invoke(app, ["migrate", str(graph), str(target)])
    assert exists.exit_code == 1
    assert "already exists" in exists.output
    assert runner.invoke(app, ["migrate", str(graph), str(target), "--overwrite"]).exit_code == 0
    again = runner.invoke(app, ["locate", str(target), query])
    assert "Pierre Curie was born in Paris" in again.output
