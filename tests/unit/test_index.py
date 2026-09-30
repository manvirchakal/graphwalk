"""Phase 1 exit: ``Index.open(db).locate(q)`` returns locations whose ``read()`` text
contains the gold evidence. Fully offline: scripted extraction and decisions."""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from graphwalk import (
    DocumentNotFoundError,
    FileDocuments,
    FileSource,
    Index,
    IngestConfig,
    Location,
    SourceDocument,
    StaleLocationError,
    TextSource,
)
from graphwalk.config import ConfigError, resolve_config
from graphwalk.decisions.base import ChoiceQuestion, JSONContent
from graphwalk.decisions.fake import FakeDecisionBackend
from graphwalk.embeddings.fake import FakeEmbedder
from graphwalk.llm.base import Message
from graphwalk.llm.fake import FakeLLM
from graphwalk.locate import fuse
from graphwalk.stores import NetworkXStore, SQLiteStore

CURIE = (
    "Marie Curie was a physicist who worked in Paris.\n\n"
    "Marie Curie was married to Pierre Curie in 1895. They shared a laboratory."
)
PIERRE = "Pierre Curie was a French physicist.\n\nPierre Curie was born in Paris, France, in 1859."
EINSTEIN = "Albert Einstein was born in Ulm. He developed the theory of relativity."

GOLD = ("Marie Curie was married to Pierre Curie", "Pierre Curie was born in Paris")
QUERY = "Where was the husband of Marie Curie born?"

EXTRACTIONS: dict[str, dict[str, object]] = {
    CURIE: {
        "entities": [
            {"name": "Marie Curie", "type": "person",
             "evidence": "Marie Curie was a physicist who worked in Paris."},
            {"name": "Pierre Curie", "type": "person",
             "evidence": "MARIE CURIE was   married to Pierre Curie"},  # fuzzy match
            {"name": "Paris", "type": "city", "evidence": "worked in Paris"},
        ],
        "relations": [
            {"source": "Marie Curie", "type": "spouse_of", "target": "Pierre Curie",
             "evidence": "Marie Curie was married to Pierre Curie in 1895."},
            {"source": "Marie Curie", "type": "worked_in", "target": "Paris",
             "evidence": "not in the passage at all"},  # falls back to the chunk
        ],
    },
    PIERRE: {
        "entities": [
            {"name": "Pierre Curie", "type": "person",
             "evidence": "Pierre Curie was a French physicist."},
            {"name": "Paris", "type": "city"},
        ],
        "relations": [
            {"source": "Pierre Curie", "type": "born_in", "target": "Paris",
             "evidence": "Pierre Curie was born in Paris, France, in 1859."},
        ],
    },
    EINSTEIN: {
        "entities": [{"name": "Albert Einstein", "type": "person"}, {"name": "Ulm"}],
        "relations": [{"source": "Albert Einstein", "type": "born_in", "target": "Ulm",
                       "evidence": "Albert Einstein was born in Ulm."}],
    },
}  # fmt: skip


def extract(messages: Sequence[Message]) -> str:
    passage = messages[1].content.split("Passage:\n", 1)[1]
    found = [reply for text, reply in EXTRACTIONS.items() if text in passage]
    return json.dumps(found[0] if found else {"entities": [], "relations": []})


def walker() -> FakeDecisionBackend:
    """Follows spouse_of, then born_in, then stops."""

    def script(question: ChoiceQuestion, _state: JSONContent) -> Mapping[str, float]:
        texts = {label: json.dumps(desc) for label, desc in question.options.items()}
        for relation in ("spouse_of", "born_in"):
            hits = [label for label, text in texts.items() if f"--{relation}-->" in text]
            if hits:
                return {label: (10.0 if label in hits else 0.1) for label in question.options}
        return {label: (10.0 if label == "STOP" else 0.1) for label in question.options}

    return FakeDecisionBackend(script=script)


def corpus() -> list[SourceDocument]:
    return [
        SourceDocument("curie", CURIE, title="Marie Curie"),
        SourceDocument("pierre", PIERRE, title="Pierre Curie"),
        SourceDocument("einstein", EINSTEIN, title="Albert Einstein"),
    ]


def make_index(path: Path | str, **kwargs: object) -> Index:
    return Index.open(
        path,
        llm=FakeLLM(extract),
        decider=walker(),
        ingest=IngestConfig(routing="exact"),
        **kwargs,
    )


async def test_exit_criterion_locate_then_read_finds_gold_evidence(tmp_path: Path) -> None:
    db = tmp_path / "my.db"
    async with make_index(db) as index:
        report = await index.ingest(TextSource("wiki", corpus()))
        assert report.evidence_found == 7
        assert report.evidence_missing == 4  # three entities without evidence, one bad quote
    # A fresh process: open the file again, only a decider needed.
    async with Index.open(db, decider=walker()) as index:
        locations = await index.locate(QUERY, k=5)
        texts = [(await index.read(loc)).text for loc in locations]
    for gold in GOLD:
        assert any(gold in text for text in texts), (gold, texts)
    first = locations[0]
    assert first.via == "graph"
    assert first.element == "edge"
    assert first.path == ("Marie Curie --spouse_of--> Pierre Curie",)
    assert texts[0] == "Marie Curie was married to Pierre Curie in 1895."
    assert first.title == "Marie Curie"
    assert first.snippet == texts[0]
    second = locations[1]
    assert second.path[-1] == "Pierre Curie --born_in--> Paris"
    assert texts[1] == "Pierre Curie was born in Paris, France, in 1859."
    assert all("Einstein" not in text for text in texts)


async def test_locations_do_not_overlap_and_respect_k(tmp_path: Path) -> None:
    async with make_index(tmp_path / "g.db") as index:
        await index.ingest(TextSource("wiki", corpus()))
        locations = await index.locate(QUERY, k=10)
        assert 2 <= len(locations) <= 10
        for i, a in enumerate(locations):
            for b in locations[i + 1 :]:
                assert not a.overlaps(b)
        assert len(await index.locate(QUERY, k=1)) == 1
        assert await index.locate("Nothing the graph knows about?") == []
        with pytest.raises(ValueError, match="k must be positive"):
            await index.locate(QUERY, k=0)


async def test_read_with_context_and_stale_detection(tmp_path: Path) -> None:
    async with make_index(tmp_path / "g.db") as index:
        await index.ingest(TextSource("wiki", corpus()))
        (first, *_) = await index.locate(QUERY, k=3)
        wide = await index.read(first, context=10)
        assert "Marie Curie was married to Pierre Curie in 1895." in wide.text
        assert wide.start == first.start - 10  # type: ignore[operator]
        # Re-ingest an edited document: old locations into it are stale.
        edited = [
            SourceDocument("curie", "Preface.\n\n" + CURIE, title="Marie Curie"),
            *corpus()[1:],
        ]
        await index.ingest(TextSource("wiki", edited))
        passage = await index.read(first)
        assert passage.stale
        with pytest.raises(StaleLocationError, match="changed since"):
            await index.read(first, on_stale="refuse")
        fresh = (await index.locate(QUERY, k=1))[0]
        assert (await index.read(fresh, on_stale="refuse")).text.startswith("Marie Curie was")


async def test_file_documents_read_from_disk(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "curie.txt").write_text(CURIE, encoding="utf-8")
    (docs / "pierre.txt").write_text(PIERRE, encoding="utf-8")
    source = FileSource(docs, source_id="files")
    store = SQLiteStore(tmp_path / "g.db")
    async with Index(
        store,
        llm=FakeLLM(extract),
        decider=walker(),
        ingest=IngestConfig(routing="exact", store_text=False),
        documents=FileDocuments(store, [source]),
    ) as index:
        await index.ingest(source)
        record = await store.get_document("files/curie.txt")
        assert record is not None
        assert record.text is None
        locations = await index.locate(QUERY, k=3)
        assert (await index.read(locations[0])).text == (
            "Marie Curie was married to Pierre Curie in 1895."
        )
        (docs / "curie.txt").write_text("Rewritten.\n\n" + CURIE, encoding="utf-8")
        assert (await index.read(locations[0])).stale
        (docs / "curie.txt").unlink()
        with pytest.raises(DocumentNotFoundError, match="no longer"):
            await index.read(locations[0])


async def test_dense_and_hybrid(tmp_path: Path) -> None:
    async with make_index(tmp_path / "g.db", embedder=FakeEmbedder()) as index:
        await index.ingest(TextSource("wiki", corpus()))
        dense = await index.locate("Albert Einstein relativity", k=2, mode="dense")
        assert dense[0].via == "dense"
        assert dense[0].key == "wiki/einstein"
        assert dense[0].element == "chunk"
        hybrid = await index.locate(QUERY, k=4, mode="hybrid")
        assert {loc.via for loc in hybrid} == {"hybrid"}
        texts = [(await index.read(loc)).text for loc in hybrid]
        for gold in GOLD:
            assert any(gold in text for text in texts)
        with pytest.raises(ValueError, match="mode must be"):
            await index.locate(QUERY, mode="bogus")  # type: ignore[arg-type]
    remote = resolve_config({"embedding_provider": "openai"})
    async with make_index(tmp_path / "h.db", config=remote) as index:
        with pytest.raises(ConfigError, match="set OPENAI_API_KEY"):
            await index.locate(QUERY, mode="dense")


def loc(key: str, start: int, end: int, via: str = "graph") -> Location:
    return Location(key=key, start=start, end=end, via=via, element="chunk")  # type: ignore[arg-type]


def test_fuse_merges_overlaps_and_ranks_by_rrf() -> None:
    graph = [loc("a", 0, 10), loc("b", 0, 5)]
    dense = [loc("b", 0, 50, "dense"), loc("c", 0, 5, "dense")]
    fused = fuse([graph, dense], k=3)
    # b is in both lists, so it wins; it keeps the graph's (first list's) span.
    assert [(f.key, f.end) for f in fused] == [("b", 5), ("a", 10), ("c", 5)]
    assert all(f.via == "hybrid" for f in fused)


async def test_neighbors_by_name_or_id_and_ingest_inputs(tmp_path: Path) -> None:
    async with make_index(tmp_path / "g.db") as index:
        await index.ingest(corpus())  # plain documents: source "memory"
        around = await index.neighbors("pierre curie")
        assert {n.edge.type for n in around} == {"spouse_of", "born_in"}
        assert await index.neighbors(around[0].node.id)
        assert isinstance(index.store, SQLiteStore)
        assert await index.store.get_document("memory/curie") is not None
    async with Index(NetworkXStore(), decider=walker()) as index:
        with pytest.raises(ConfigError, match="set OPENROUTER_API_KEY"):
            await index.ingest(corpus())
    async with Index(NetworkXStore()) as index:
        with pytest.raises(ConfigError, match="GRAPHWALK_DECISION_FALLBACK=llm"):
            await index.locate(QUERY)
