import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from factories import prov
from graphwalk.core.merge import merge_node_data
from graphwalk.core.model import Node
from graphwalk.decisions import ChoiceQuestion, FakeDecisionBackend, JSONContent
from graphwalk.embeddings import FakeEmbedder
from graphwalk.ingest import (
    ExtractedEntity,
    FileSource,
    IngestConfig,
    IngestPipeline,
    JsonFileCache,
    NodeIndex,
    RouteRecord,
    SourceDocument,
    TextSource,
    chunk_text,
    extract,
    provenance_id,
    retract,
)
from graphwalk.ingest.extraction import EXTRACT_SYSTEM, clean, parse_extraction
from graphwalk.ingest.routing import NEW_LABEL, content_words, parse_label
from graphwalk.llm import FakeLLM, Message
from graphwalk.stores.networkx_store import NetworkXStore

if TYPE_CHECKING:
    from pydantic import JsonValue

# ---------------------------------------------------------------- chunking & sources


def test_chunk_text_packs_paragraphs() -> None:
    text = "aaa\n\nbbb\n\n\nccc"
    assert chunk_text(text, max_chars=8) == ["aaa\n\nbbb", "ccc"]
    assert chunk_text(text, max_chars=100) == ["aaa\n\nbbb\n\nccc"]
    assert chunk_text("") == []


def test_chunk_text_splits_long_paragraphs() -> None:
    chunks = chunk_text("One two. Three four five. " + "x" * 25, max_chars=10)
    assert all(len(c) <= 10 for c in chunks)
    assert "".join(c.replace("\n\n", "") for c in chunks).startswith("One two.")
    with pytest.raises(ValueError, match="positive"):
        chunk_text("a", max_chars=0)


def test_file_source_reads_every_format(tmp_path: Path) -> None:
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "a.md").write_text("# Title\n\nBody", encoding="utf-8")
    (tmp_path / "people.csv").write_text("id,name,born\n7,Ada,1815\n,Bob,\n", encoding="utf-8")
    (tmp_path / "rows.jsonl").write_text(
        '{"title": "T", "text": "Hello"}\n\n{"x": 1}\n', encoding="utf-8"
    )
    (tmp_path / "one.json").write_text('{"id": "k", "body": "Text"}', encoding="utf-8")
    (tmp_path / "skip.bin").write_bytes(b"\0")
    source = FileSource(tmp_path, source_id="files")
    docs = {d.doc_id: d for d in source.documents()}
    assert sorted(docs) == ["notes/a.md", "one.json#k", "people.csv#1", "people.csv#7",
                            "rows.jsonl#0", "rows.jsonl#1"]  # fmt: skip
    assert docs["notes/a.md"].title == "a"
    assert docs["people.csv#7"].text == "id: 7\nname: Ada\nborn: 1815"
    assert docs["people.csv#7"].title == "Ada"
    assert docs["rows.jsonl#0"].text == "Hello"
    assert docs["rows.jsonl#1"].text == "x: 1"
    assert docs["one.json#k"].text == "Text"
    single = FileSource(tmp_path / "one.json")
    assert single.source_id.startswith("file:")
    assert [d.doc_id for d in single.documents()] == ["one.json#k"]
    with pytest.raises(FileNotFoundError):
        FileSource(tmp_path / "missing")


def test_text_source_rejects_duplicate_ids() -> None:
    doc = SourceDocument(doc_id="d", text="t")
    with pytest.raises(ValueError, match="duplicate"):
        TextSource("s", [doc, doc])


# ---------------------------------------------------------------- extraction


def test_parse_and_clean_extraction() -> None:
    reply = """```json
    {"entities": [{"name": " Marie  Curie ", "type": "Person", "attributes": {"born": 1867}},
                  {"name": "marie curie", "type": "scientist", "description": "Physicist"},
                  {"name": "Warsaw", "type": "", "extra": 1}],
     "relations": [{"source": "Marie Curie", "type": "Born In", "target": "Warsaw"},
                   {"source": "Marie Curie", "type": "born_in", "target": "warsaw"},
                   {"source": "Marie Curie", "type": "spouse of", "target": "Pierre Curie"},
                   {"source": "Warsaw", "type": "x", "target": "warsaw"},
                   {"source": "A", "type": "!!", "target": "B"}]}
    ```"""
    extraction, dropped = clean(parse_extraction(reply))
    names = {e.name: e for e in extraction.entities}
    assert list(names) == ["Marie Curie", "Warsaw", "Pierre Curie"]
    curie = names["Marie Curie"]
    assert (curie.type, curie.description, curie.attributes) == (
        "person",
        "Physicist",
        {"born": 1867},
    )
    assert names["Warsaw"].type == "entity"
    assert names["Pierre Curie"].type == "entity"  # added for a relation endpoint
    assert [(r.source, r.type, r.target) for r in extraction.relations] == [
        ("Marie Curie", "born_in", "Warsaw"),
        ("Marie Curie", "spouse_of", "Pierre Curie"),
    ]
    assert len(dropped) == 2
    with pytest.raises(ValueError, match="no JSON"):
        parse_extraction("sorry")


async def test_extract_retries_an_unparsable_reply() -> None:
    replies = iter(["not json", '{"entities": [{"name": "Ada"}]}'])
    llm = FakeLLM(lambda _m: next(replies), cost_per_call=0.001)
    result = await extract(llm, "Ada wrote notes.", title="Ada")
    assert result.error is None
    assert [e.name for e in result.extraction.entities] == ["Ada"]
    assert result.spend.llm_calls == 2
    assert result.spend.cost_usd == pytest.approx(0.002)
    assert llm.calls[0][1].content.startswith("Title: Ada")
    assert "not valid" in llm.calls[1][-1].content

    failing = await extract(FakeLLM(lambda _m: "{bad"), "text")
    assert failing.error is not None
    assert failing.extraction.entities == ()
    assert failing.spend.cost_usd is None  # the fake reports no cost


# ---------------------------------------------------------------- pipeline

EXTRACTIONS: dict[str, dict[str, object]] = {
    "Marie Curie was born in Warsaw.": {
        "entities": [
            {"name": "Marie Curie", "type": "person", "description": "A physicist.",
             "attributes": {"born": "1867"}},
            {"name": "Warsaw", "type": "city"},
        ],
        "relations": [{"source": "Marie Curie", "type": "born_in", "target": "Warsaw"}],
    },
    "Marie Curie was born in Warsaw in 1868.": {
        "entities": [{"name": "Marie Curie", "type": "person", "attributes": {"born": "1868"}},
                     {"name": "Warsaw", "type": "city"}],
        "relations": [{"source": "Marie Curie", "type": "born_in", "target": "Warsaw"}],
    },
    "M. Curie married Pierre Curie.": {
        "entities": [{"name": "M. Curie", "type": "person"}, {"name": "Pierre Curie"}],
        "relations": [{"source": "M. Curie", "type": "spouse_of", "target": "Pierre Curie"}],
    },
    "Marie Curie won a Nobel Prize.": {
        "entities": [{"name": "Marie Curie", "type": "person", "attributes": {"born": "1867"}},
                     {"name": "Nobel Prize", "type": "award"}],
        "relations": [{"source": "Marie Curie", "type": "won", "target": "Nobel Prize"}],
    },
}  # fmt: skip


def last_curie(routes: Sequence[RouteRecord]) -> RouteRecord:
    return [r for r in routes if r.mention == "Marie Curie"][-1]


def _passage(messages: Sequence[Message]) -> str:
    return messages[1].content.split("Passage:\n", 1)[1]


class Scripted:
    """LLM: extraction replies from EXTRACTIONS; escalation replies ``escalate``."""

    def __init__(self, escalate: str = "c1", fail: frozenset[str] = frozenset()) -> None:
        self.escalate = escalate
        self.fail = set(fail)
        self.llm = FakeLLM(self.respond, cost_per_call=0.001)

    def respond(self, messages: Sequence[Message]) -> str:
        if messages[0].content != EXTRACT_SYSTEM:
            return self.escalate
        passage = _passage(messages)
        if passage in self.fail:
            return "oops"
        return json.dumps(EXTRACTIONS[passage])

    def extraction_calls(self) -> int:
        return sum(1 for m in self.llm.calls if m[0].content == EXTRACT_SYSTEM)


def choose(label: str) -> FakeDecisionBackend:
    def script(question: ChoiceQuestion, _state: JSONContent) -> Mapping[str, float]:
        pick = label if label in question.options else NEW_LABEL
        return {
            o: (0.9 if o == pick else 0.1 / (len(question.options) - 1)) for o in question.options
        }

    return FakeDecisionBackend(script=script, cost_per_call=0.0001)


def docs(**texts: str) -> TextSource:
    return TextSource("src", [SourceDocument(doc_id=k, text=v) for k, v in texts.items()])


async def names(store: NetworkXStore) -> list[str]:
    return sorted([n.name async for n in store.iter_nodes()])


async def the(store: NetworkXStore, name: str) -> Node:
    (found,) = await store.find_nodes(name=name)
    return found


async def test_ingest_creates_and_merges() -> None:
    store = NetworkXStore()
    llm = Scripted()
    decider = choose("c1")
    pipeline = IngestPipeline(store, llm.llm, decider, embedder=FakeEmbedder())
    report = await pipeline.ingest(
        docs(a="Marie Curie was born in Warsaw.", b="Marie Curie won a Nobel Prize.")
    )
    assert (report.documents, report.new, report.chunks) == (2, 2, 2)
    assert await names(store) == ["Marie Curie", "Nobel Prize", "Warsaw"]
    assert (report.nodes_created, report.nodes_merged) == (3, 1)
    assert (report.edges_created, report.edges_merged) == (2, 0)
    # Doc b's "Marie Curie" had doc a's node as a candidate: one decision call, merged.
    assert report.decision_calls == 1
    assert [r.method for r in report.routes] == ["no_candidates"] * 2 + [
        "decision",
        "no_candidates",
    ]
    assert last_curie(report.routes).chosen == "Marie Curie"
    assert report.cost_usd == pytest.approx(2 * 0.001 + 0.0001)
    curie = await the(store, "Marie Curie")
    assert {p.source_id for p in curie.provenance} == {"src/a", "src/b"}
    assert curie.summary == "A physicist."
    assert curie.attributes == {"born": "1867"}
    assert await store.get_metadata("ingest:ledger:src") is not None


async def test_ingest_records_conflicts_and_is_idempotent() -> None:
    store = NetworkXStore()
    llm = Scripted()
    pipeline = IngestPipeline(store, llm.llm, choose("c1"))
    source = docs(a="Marie Curie was born in Warsaw.", b="Marie Curie was born in Warsaw in 1868.")
    await pipeline.ingest(source)
    curie = await the(store, "Marie Curie")
    conflict = curie.conflicts["born"]
    assert sorted(str(v.value) for v in conflict.values) == ["1867", "1868"]
    assert [p.source_id for v in conflict.values for p in v.provenance] == ["src/a", "src/b"]
    (edge,) = [e async for e in store.iter_edges()]
    assert [p.source_id for p in edge.provenance] == ["src/a", "src/b"]

    calls = len(llm.llm.calls)
    again = await pipeline.ingest(source)
    assert (again.unchanged, again.new, again.changed, again.chunks) == (2, 0, 0, 0)
    assert len(llm.llm.calls) == calls
    assert again.llm_calls == again.decision_calls == 0


async def test_changed_and_removed_documents_are_retracted() -> None:
    store = NetworkXStore()
    llm = Scripted()
    pipeline = IngestPipeline(store, llm.llm, choose("c1"), config=IngestConfig(prune_missing=True))
    await pipeline.ingest(
        docs(a="Marie Curie was born in Warsaw in 1868.", b="Marie Curie won a Nobel Prize.")
    )
    assert "born" in (await the(store, "Marie Curie")).conflicts

    changed = await pipeline.ingest(
        docs(a="Marie Curie was born in Warsaw.", b="Marie Curie won a Nobel Prize.")
    )
    assert (changed.changed, changed.unchanged) == (1, 1)
    curie = await the(store, "Marie Curie")
    assert curie.conflicts == {}  # 1868 came only from the old version of a
    assert curie.attributes == {"born": "1867"}
    assert await names(store) == ["Marie Curie", "Nobel Prize", "Warsaw"]

    removed = await pipeline.ingest(docs(b="Marie Curie won a Nobel Prize."))
    assert removed.removed == 1
    assert await names(store) == ["Marie Curie", "Nobel Prize"]
    assert removed.nodes_deleted == 1
    assert removed.edges_deleted == 1
    assert {p.source_id for p in (await the(store, "Marie Curie")).provenance} == {"src/b"}


async def test_routing_new_keeps_entities_apart() -> None:
    store = NetworkXStore()
    pipeline = IngestPipeline(store, Scripted().llm, choose(NEW_LABEL))
    await pipeline.ingest(
        docs(a="Marie Curie was born in Warsaw.", b="Marie Curie won a Nobel Prize.")
    )
    assert await names(store) == ["Marie Curie", "Marie Curie", "Nobel Prize", "Warsaw"]


async def test_exact_routing_makes_no_decision_calls() -> None:
    store = NetworkXStore()
    decider = choose("c1")
    pipeline = IngestPipeline(
        store, Scripted().llm, decider, embedder=FakeEmbedder(),
        config=IngestConfig(routing="exact", min_similarity=-1.0),
    )  # fmt: skip
    await pipeline.ingest(
        docs(
            a="Marie Curie was born in Warsaw.",
            b="Marie Curie won a Nobel Prize.",
            c="M. Curie married Pierre Curie.",
        )
    )
    assert decider.calls == 0
    # Exact names merge; "M. Curie" is only a fuzzy candidate, so it stays separate.
    assert (await names(store)).count("Marie Curie") == 1
    assert "M. Curie" in await names(store)
    assert IngestPipeline(store, Scripted().llm, None, config=IngestConfig(routing="exact"))
    with pytest.raises(ValueError, match="decision backend"):
        IngestPipeline(store, Scripted().llm, None)


async def test_low_confidence_decisions_escalate_to_the_llm() -> None:
    store = NetworkXStore()
    llm = Scripted(escalate="c1")
    uniform = FakeDecisionBackend(script=lambda q, _s: dict.fromkeys(q.options, 1.0))
    pipeline = IngestPipeline(store, llm.llm, uniform)
    report = await pipeline.ingest(
        docs(a="Marie Curie was born in Warsaw.", b="Marie Curie won a Nobel Prize.")
    )
    assert report.escalations == 1
    assert report.escalation_model == "fake-llm-1"  # defaults to the extraction LLM
    assert report.escalation_calls == 1
    assert report.escalation_cost_usd == pytest.approx(0.001)
    assert last_curie(report.routes).method == "llm"
    assert last_curie(report.routes).probability == pytest.approx(0.5)
    assert (await names(store)).count("Marie Curie") == 1

    store2 = NetworkXStore()
    no_escalation = IngestPipeline(
        store2, Scripted().llm, uniform, config=IngestConfig(escalate=False)
    )
    report2 = await no_escalation.ingest(
        docs(a="Marie Curie was born in Warsaw.", b="Marie Curie won a Nobel Prize.")
    )
    assert report2.escalations == 0
    assert last_curie(report2.routes).method == "decision"

    bad = IngestPipeline(NetworkXStore(), Scripted(escalate="dunno").llm, uniform)
    report3 = await bad.ingest(
        docs(a="Marie Curie was born in Warsaw.", b="Marie Curie won a Nobel Prize.")
    )
    assert last_curie(report3.routes).method == "decision"
    assert last_curie(report3.routes).error is not None


async def test_decision_failure_falls_back_to_exact_names() -> None:
    store = NetworkXStore()
    failing = FakeDecisionBackend(fail_on_calls=[1])
    pipeline = IngestPipeline(store, Scripted().llm, failing)
    report = await pipeline.ingest(
        docs(a="Marie Curie was born in Warsaw.", b="Marie Curie won a Nobel Prize.")
    )
    assert last_curie(report.routes).method == "fallback"
    assert last_curie(report.routes).error is not None
    assert (await names(store)).count("Marie Curie") == 1


async def test_failed_extraction_is_retried_next_run() -> None:
    store = NetworkXStore()
    llm = Scripted(fail=frozenset({"Marie Curie won a Nobel Prize."}))
    pipeline = IngestPipeline(store, llm.llm, choose("c1"))
    source = docs(a="Marie Curie was born in Warsaw.", b="Marie Curie won a Nobel Prize.")
    first = await pipeline.ingest(source)
    assert first.failed == 1
    assert len(first.errors) == 1
    assert first.errors[0].startswith("b:")
    assert "Nobel Prize" not in await names(store)

    llm.fail.clear()
    second = await pipeline.ingest(source)
    assert (second.changed, second.unchanged) == (1, 1)
    assert "Nobel Prize" in await names(store)


async def test_extraction_cache_skips_llm_calls() -> None:
    cache: dict[str, JsonValue] = {}
    source = docs(a="Marie Curie was born in Warsaw.")
    llm = Scripted()
    await IngestPipeline(NetworkXStore(), llm.llm, choose("c1"), extraction_cache=cache).ingest(
        source
    )
    assert len(cache) == 1
    report = await IngestPipeline(
        NetworkXStore(),
        llm.llm,
        choose("c1"),
        extraction_cache=cache,
    ).ingest(source)
    assert report.extraction_cache_hits == 1
    assert report.llm_calls == 0
    assert llm.extraction_calls() == 1


# ---------------------------------------------------------------- index & retraction


async def test_node_index_candidates() -> None:
    index = NodeIndex(FakeEmbedder(), min_similarity=0.99)
    a = Node(id="a", type="person", name="Marie Curie", aliases=("Maria Sklodowska",),
             provenance=(prov(),))  # fmt: skip
    b = Node(id="b", type="person", name="Pierre Curie", provenance=(prov(),))
    c = Node(id="c", type="city", name="Warsaw", provenance=(prov(),))
    for n in (a, b, c):
        index.add(n)
    assert len(index) == 3
    assert index.exact("maria  SKLODOWSKA") == ["a"]
    found = await index.candidates(ExtractedEntity(name="Marie Curie"), 5)
    assert found[0] == "a"
    assert "b" in found  # shares "curie"
    assert "c" not in found
    assert await index.candidates(ExtractedEntity(name="Curie"), 1) in (["a"], ["b"])
    index.remove("a")
    assert index.exact("Marie Curie") == []
    renamed = b.model_copy(update={"name": "P. Curie"})
    index.add(renamed)
    assert index.exact("Pierre Curie") == []
    assert index.exact("P. Curie") == ["b"]


async def test_retract_promotes_the_surviving_conflict_value() -> None:
    store = NetworkXStore()
    keep = Node(id="x", type="t", name="X", attributes={"k": 1}, provenance=(prov("s/a"),))
    other = Node(id="x", type="t", name="X", attributes={"k": 2}, provenance=(prov("s/b"),))
    await store.upsert_node(merge_node_data(keep, other))
    result = await retract(store, {provenance_id("s", "a")})
    assert (result.nodes_updated, result.nodes_deleted) == (1, 0)
    node = await the(store, "X")
    assert node.attributes == {"k": 2}
    assert node.conflicts == {}
    assert [p.source_id for p in node.sources_of("k")] == ["s/b"]
    assert await retract(store, set()) == type(result)()


async def test_escalation_can_use_a_separate_model() -> None:
    extractor = Scripted(escalate="NEW")  # would split the entity if asked
    judge = FakeLLM(lambda _m: "Reasoning... the answer is c1", model_id="judge",
                    cost_per_call=0.01)  # fmt: skip
    uniform = FakeDecisionBackend(
        script=lambda q, _s: dict.fromkeys(q.options, 1.0), cost_per_call=0.0
    )
    pipeline = IngestPipeline(NetworkXStore(), extractor.llm, uniform, escalation_llm=judge)
    report = await pipeline.ingest(
        docs(a="Marie Curie was born in Warsaw.", b="Marie Curie won a Nobel Prize.")
    )
    assert len(judge.calls) == 1
    assert extractor.extraction_calls() == len(extractor.llm.calls) == 2
    assert (report.escalation_model, report.escalation_calls) == ("judge", 1)
    assert report.escalation_cost_usd == pytest.approx(0.01)
    assert report.cost_usd == pytest.approx(0.002 + 0.01)
    assert last_curie(report.routes).chosen == "Marie Curie"


def test_parse_label() -> None:
    labels = ("c1", "c2", "NEW")
    assert parse_label(" `c2`. ", labels) == "c2"
    assert parse_label("Not c1; it is c2", labels) == "c2"
    assert parse_label("NEW", labels) == "NEW"
    assert parse_label("c7", labels) is None
    assert parse_label("unsure", labels) is None


# ---------------------------------------------------------------- performance paths


async def test_windowed_routing_matches_sequential_routing() -> None:
    texts = {
        "a": "Marie Curie was born in Warsaw.",
        "b": "Marie Curie won a Nobel Prize.",
        "c": "Marie Curie was born in Warsaw in 1868.",
    }
    graphs = []
    for window in (1, 4):
        store = NetworkXStore()
        pipeline = IngestPipeline(
            store, Scripted().llm, choose("c1"), config=IngestConfig(route_concurrency=window)
        )
        report = await pipeline.ingest(docs(**texts))
        assert report.route_s >= 0.0
        assert report.apply_s >= 0.0
        assert report.extract_wait_s >= 0.0
        graphs.append(await names(store))
    assert graphs[0] == graphs[1]


async def test_fuzzy_repeats_within_a_window_are_not_matched() -> None:
    # Documented limitation: "M. Curie" only fuzzy-matches "Marie Curie", which the
    # same window created, so windowed routing never offers it as a candidate.
    texts = {"a": "Marie Curie was born in Warsaw.", "c": "M. Curie married Pierre Curie."}
    sequential = NetworkXStore()
    await IngestPipeline(
        sequential, Scripted().llm, choose("c1"), config=IngestConfig(route_concurrency=1)
    ).ingest(docs(**texts))
    windowed = NetworkXStore()
    await IngestPipeline(windowed, Scripted().llm, choose("c1")).ingest(docs(**texts))
    assert "M. Curie" not in await names(sequential)
    assert "M. Curie" in await names(windowed)


async def test_windowed_new_decision_is_respected() -> None:
    # Both chunks route in one window; the second "Marie Curie" is re-routed against
    # the node the first created, and the decider's NEW keeps them apart.
    store = NetworkXStore()
    decider = choose(NEW_LABEL)
    pipeline = IngestPipeline(store, Scripted().llm, decider)
    report = await pipeline.ingest(
        docs(a="Marie Curie was born in Warsaw.", b="Marie Curie won a Nobel Prize.")
    )
    assert (await names(store)).count("Marie Curie") == 2
    assert report.decision_calls == 1  # the re-route
    assert decider.calls == 1


async def test_node_index_batches_and_reuses_vectors() -> None:
    class Counting(FakeEmbedder):
        def __init__(self) -> None:
            super().__init__()
            self.texts = 0
            self.batches = 0

        async def embed(self, texts: Sequence[str]) -> Any:
            self.batches += 1
            self.texts += len(texts)
            return await super().embed(texts)

    embedder = Counting()
    index = NodeIndex(embedder, min_similarity=-1.0)
    mentions = [ExtractedEntity(name=f"Person {i}", type="person") for i in range(100)]
    found = await index.candidates_many(mentions, 3)
    assert found == [[] for _ in mentions]
    assert (embedder.batches, embedder.texts) == (1, 100)
    for i, mention in enumerate(mentions):  # nodes created from the mentions: same text
        index.add(Node(id=f"n{i}", type="person", name=mention.name, provenance=(prov(),)))
    await index.prime()
    assert embedder.texts == 100  # no re-embedding
    again = await index.candidates_many(mentions[:2], 3)
    assert again[0][0] == "n0"
    assert len(again[0]) == 3
    index.remove("n1")
    assert "n1" not in (await index.candidates_many([mentions[1]], 5))[0]


def test_lexical_pool_skips_very_common_words() -> None:
    index = NodeIndex(max_word_df=2)
    for i in range(5):
        index.add(Node(id=f"j{i}", type="person", name=f"John Smith{i}", provenance=(prov(),)))
    index.add(Node(id="x", type="person", name="John Rarename", provenance=(prov(),)))
    assert set(index._lexical(content_words("John Rarename"), [])) == {"x"}
    # Every word is common: the rarest one still opens the pool.
    assert len(index._lexical(content_words("John"), [])) == 6


def test_json_file_cache_flushes_incrementally(tmp_path: Path) -> None:
    path = tmp_path / "c" / "cache.json"
    cache = JsonFileCache(path, flush_every=2)
    cache["a"] = 1
    assert not path.exists()
    cache["b"] = 2
    assert json.loads(path.read_text(encoding="utf-8")) == {"a": 1, "b": 2}
    cache["c"] = 3
    del cache["a"]
    cache.flush()
    reloaded = JsonFileCache(path)
    assert dict(reloaded) == {"b": 2, "c": 3}
    assert len(reloaded) == 2
    reloaded.flush()  # nothing unsaved: no-op


async def test_ingest_checkpoints_every_n_windows() -> None:
    calls: list[int] = []

    async def checkpoint() -> None:
        calls.append(1)

    texts = dict.fromkeys("abcde", "Marie Curie was born in Warsaw.")
    pipeline = IngestPipeline(
        NetworkXStore(), Scripted().llm, choose("c1"), config=IngestConfig(route_concurrency=1)
    )
    await pipeline.ingest(docs(**texts), checkpoint=checkpoint, checkpoint_every=2)
    assert len(calls) == 2  # after windows 2 and 4 of 5
