"""Importing existing knowledge graphs from triples files."""

import asyncio
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from factories import node, prov
from graphwalk import Index
from graphwalk.cli import app
from graphwalk.ingest.triples import (
    TripleFormatError,
    import_file,
    literal_id,
    local_name,
    read_triples,
)
from graphwalk.stores import NetworkXStore, SQLiteStore
from graphwalk.stores.base import GraphStore
from graphwalk.traversal import TraversalConfig
from kg_fixtures import oracle

NT = """\
# A comment line.
<http://ex.org/Inception> <http://ex.org/directed_by> <http://ex.org/Nolan> .
<http://ex.org/Inception> <http://www.w3.org/2000/01/rdf-schema#label> "Inception"@en .
<http://ex.org/Nolan> <http://www.w3.org/2000/01/rdf-schema#label> "Christopher Nolan" .
<http://ex.org/Nolan> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://ex.org/Person> .
<http://ex.org/Nolan> <http://ex.org/born> "1970-07-30"^^<http://www.w3.org/2001/XMLSchema#date> .
<http://ex.org/Nolan> <http://ex.org/quote> "say \\"hi\\"\\u00e9" .
_:b1 <http://ex.org/about> <http://ex.org/Nolan> .
this line is not a triple
"""


@pytest.fixture(params=["networkx", "sqlite"])
def store(request: pytest.FixtureRequest, tmp_path: Path) -> GraphStore:
    if request.param == "sqlite":
        return SQLiteStore(tmp_path / "kg.db")
    return NetworkXStore()


def _write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_csv_header_aliases_and_optional_columns(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "kg.csv",
        "head,predicate,tail,subject_type,object_name\n"
        "Q1,directed_by,Q2,film,Christopher Nolan\n"
        ",missing,subject\n"
        "\n",
    )
    triples = list(read_triples(path))
    assert len(triples) == 1
    t = triples[0]
    assert (t.subject, t.relation, t.object) == ("Q1", "directed_by", "Q2")
    assert t.subject_type == "film"
    assert t.object_name == "Christopher Nolan"


def test_tsv_without_header_and_pipe(tmp_path: Path) -> None:
    tsv = _write(tmp_path, "kg.tsv", "Inception\tdirected_by\tNolan\nshort\trow\n")
    assert [(t.subject, t.object) for t in read_triples(tsv)] == [("Inception", "Nolan")]
    pipe = _write(tmp_path, "kb.txt", "Inception|directed_by|Nolan\nInception|year|2010\n")
    assert [t.relation for t in read_triples(pipe)] == ["directed_by", "year"]


def test_jsonl(tmp_path: Path) -> None:
    lines = [
        {"subject": "Inception", "relation": "directed_by", "object": "Nolan",
         "object_type": "person"},
        {"s": "Nolan", "p": "born_in", "o": "London"},
        {"subject": "no relation"},
        [1, 2, 3],
    ]  # fmt: skip
    path = _write(tmp_path, "kg.jsonl", "\n".join(json.dumps(x) for x in lines) + "\nnot json\n")
    triples = list(read_triples(path))
    assert [(t.subject, t.relation, t.object) for t in triples] == [
        ("Inception", "directed_by", "Nolan"),
        ("Nolan", "born_in", "London"),
    ]
    assert triples[0].object_type == "person"


def test_ntriples_labels_types_and_literals(tmp_path: Path) -> None:
    path = _write(tmp_path, "kg.nt", NT)
    triples = list(read_triples(path))
    by_rel = {t.relation: t for t in triples}
    assert set(by_rel) == {"directed_by", "born", "quote", "about"}
    directed = by_rel["directed_by"]
    assert directed.subject == "http://ex.org/Inception"
    assert directed.subject_name == "Inception"
    assert directed.object_name == "Christopher Nolan"
    assert directed.object_type == "Person"
    born = by_rel["born"]
    assert born.object == literal_id("1970-07-30")
    assert born.object_name == "1970-07-30"
    assert born.object_type == "literal"
    assert by_rel["quote"].object_name == 'say "hi"é'
    assert by_rel["about"].subject == "_:b1"


def test_local_name() -> None:
    assert local_name("http://ex.org/a/b") == "b"
    assert local_name("http://ex.org/onto#Person") == "Person"
    assert local_name("http://ex.org/a/") == "a"


def test_unknown_suffix(tmp_path: Path) -> None:
    with pytest.raises(TripleFormatError, match="unknown triples format"):
        read_triples(_write(tmp_path, "kg.xml", ""))


async def test_import_file_is_idempotent(store: GraphStore, tmp_path: Path) -> None:
    path = _write(tmp_path, "kg.nt", NT)
    first = await import_file(store, path)
    assert first.skipped == 1
    assert first.edges == 4
    nodes, edges = await store.counts()
    assert edges == 4
    nolan = await store.get_node("http://ex.org/Nolan")
    assert nolan is not None
    assert (nolan.name, nolan.type) == ("Christopher Nolan", "Person")
    assert nolan.provenance[0].source_id == "kg.nt"
    again = await import_file(store, path)
    assert again.nodes == 0
    assert await store.counts() == (nodes, edges)
    await store.close()


async def test_import_keeps_existing_nodes(store: GraphStore, tmp_path: Path) -> None:
    await store.upsert_node(node("Nolan", type_="director", name="C. Nolan"))
    path = _write(tmp_path, "kg.tsv", "Inception\tdirected_by\tNolan\n")
    report = await import_file(store, path, default_type="thing")
    assert report.nodes == 1
    nolan = await store.get_node("Nolan")
    inception = await store.get_node("Inception")
    assert nolan is not None
    assert inception is not None
    assert (nolan.name, nolan.type) == ("C. Nolan", "director")
    assert inception.type == "thing"
    assert nolan.provenance == (prov(),)
    await store.close()


async def test_index_import_then_walk(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "kg.csv",
        "subject,relation,object\nInception,directed_by,Christopher Nolan\n"
        "Christopher Nolan,born_in,London\n",
    )
    decider = oracle(
        {
            "Inception": {"Christopher Nolan": 1.0},
            "Christopher Nolan": {"London": 1.0},
            "London": {"STOP": 1.0},
        }
    )
    config = TraversalConfig(strategy="greedy")
    async with Index.open(tmp_path / "kg.db", decider=decider, traversal=config) as index:
        # Link once before importing, so the import must refresh the name index.
        assert (await index.walk("Where was the director of Inception born?")).entries == ()
        report = await index.import_triples(path)
        assert (report.nodes, report.edges) == (3, 2)
        result = await index.walk("Where was the director of Inception born?")
    assert result.best is not None
    assert result.best.names == ("London",)


def test_cli_import(tmp_path: Path) -> None:
    path = _write(tmp_path, "kg.txt", "Inception|directed_by|Nolan\n")
    for graph in (tmp_path / "kg.db", tmp_path / "kg.json"):
        result = CliRunner().invoke(app, ["import", str(path), "--graph", str(graph)])
        assert result.exit_code == 0, result.output
        assert "2 nodes, 1 edges" in result.output
    loaded = asyncio.run(NetworkXStore.load(tmp_path / "kg.json"))
    assert asyncio.run(loaded.counts()) == (2, 1)
    bad = CliRunner().invoke(app, ["import", str(path), "--graph", "x.db", "--format", "xml"])
    assert bad.exit_code == 2
