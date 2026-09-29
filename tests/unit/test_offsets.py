"""Offset provenance: span validation, evidence matching, cache compatibility."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from factories import T0, node, prov
from graphwalk.core.hashing import content_hash
from graphwalk.core.model import Provenance, StoredDocument
from graphwalk.eval.text_qa import legacy_config_dump
from graphwalk.ingest.extraction import (
    EXTRACT_SYSTEM,
    EXTRACT_SYSTEM_EVIDENCE,
    Extraction,
    clean,
    extraction_messages,
    find_evidence,
)
from graphwalk.ingest.pipeline import IngestConfig, IngestPipeline, chunk_key
from graphwalk.ingest.sources import FileSource, SourceDocument, TextSource
from graphwalk.llm import FakeLLM
from graphwalk.stores import NetworkXStore


def test_provenance_span_validation() -> None:
    base = prov().model_dump()
    assert Provenance.model_validate(base).start is None  # old records still load
    assert Provenance.model_validate({**base, "start": 2, "end": 2}).end == 2
    with pytest.raises(ValidationError, match="both be set"):
        Provenance.model_validate({**base, "start": 2})
    with pytest.raises(ValidationError, match="before start"):
        Provenance.model_validate({**base, "start": 5, "end": 3})


@pytest.mark.parametrize(
    ("quote", "expected"),
    [
        ("was born in Paris", "was born in Paris"),
        ('"Was  BORN\nin paris."', "was born in Paris."),
        ("born in Paris...", "born in Paris"),
        ("“born in Paris”", "born in Paris"),
        ("he", None),
        ("born in London", None),
        (None, None),
    ],
)
def test_find_evidence(quote: str | None, expected: str | None) -> None:
    text = "Pierre Curie was born in Paris. He was a physicist."
    found = find_evidence(text, quote)
    assert (None if found is None else text[found[0] : found[1]]) == expected


def test_clean_keeps_first_evidence_when_merging_duplicates() -> None:
    raw = Extraction.model_validate(
        {
            "entities": [
                {"name": "Ada", "type": "person"},
                {"name": "ada", "evidence": "Ada was here"},
            ],
            "relations": [
                {"source": "Ada", "type": "Born In", "target": "London", "evidence": "q1"},
                {"source": "Ada", "type": "born_in", "target": "London", "evidence": "q2"},
            ],
        }
    )
    cleaned, _ = clean(raw)
    assert cleaned.entities[0].evidence == "Ada was here"
    assert [r.evidence for r in cleaned.relations] == ["q1"]


def test_evidence_prompt_is_opt_in_per_call() -> None:
    assert extraction_messages("x", None)[0].content == EXTRACT_SYSTEM
    assert extraction_messages("x", None, evidence=True)[0].content == EXTRACT_SYSTEM_EVIDENCE
    assert '"evidence": str' in EXTRACT_SYSTEM_EVIDENCE


def test_legacy_cache_keys_are_unchanged() -> None:
    legacy = IngestConfig(evidence=False)
    assert chunk_key("text", "t", legacy.extractor) == chunk_key("text", "t")
    assert chunk_key("text", "t", IngestConfig().extractor) != chunk_key("text", "t")
    dump = legacy_config_dump(legacy)
    assert "evidence" not in dump
    assert "store_text" not in dump
    assert legacy_config_dump(IngestConfig())["evidence"] is True


async def test_without_evidence_provenance_spans_the_chunk() -> None:
    reply = '{"entities": [{"name": "Ada Lovelace", "evidence": "ignored"}], "relations": []}'
    store = NetworkXStore()
    text = "  Intro.\n\nAda Lovelace wrote notes.  "
    pipeline = IngestPipeline(
        store, FakeLLM(lambda _m: reply), None, config=IngestConfig(routing="exact", evidence=False)
    )
    report = await pipeline.ingest(TextSource("s", [SourceDocument("d", text)]))
    assert (report.evidence_found, report.evidence_missing) == (0, 0)
    (ada,) = await store.find_nodes(name="Ada Lovelace")
    (p,) = ada.provenance
    assert text[p.start : p.end] == "Intro.\n\nAda Lovelace wrote notes."  # type: ignore[misc]
    stored = await store.get_document("s/d")
    assert stored is not None
    assert stored.text == text
    assert stored.text_hash == content_hash(text)


async def test_networkx_save_load_keeps_documents(tmp_path: Path) -> None:
    store = NetworkXStore()
    await store.upsert_node(node("a"))
    doc = StoredDocument(
        key="s/d", source_id="s", doc_id="d", text="hi", text_hash="h", length=2, ingested_at=T0
    )
    await store.put_document(doc)
    await store.save(tmp_path / "g.json")
    loaded = await NetworkXStore.load(tmp_path / "g.json")
    assert await loaded.get_document("s/d") == doc


def test_file_source_document_lookup(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("alpha", encoding="utf-8")
    (tmp_path / "x#y.md").write_text("hash in name", encoding="utf-8")
    (tmp_path / "r.jsonl").write_text('{"id": "7", "text": "seven"}\n', encoding="utf-8")
    (tmp_path.parent / "outside.txt").write_text("secret", encoding="utf-8")
    source = FileSource(tmp_path)
    assert {d.doc_id for d in source.documents()} == {"a.txt", "x#y.md", "r.jsonl#7"}
    for doc in source.documents():
        assert source.document(doc.doc_id) == doc
    assert source.document("../outside.txt") is None
    assert source.document("missing.txt") is None
    assert source.document("r.jsonl#8") is None
    single = FileSource(tmp_path / "a.txt")
    assert single.document("a.txt") is not None
    assert single.document("b.txt") is None
