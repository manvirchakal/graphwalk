import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from graphwalk.decisions import ChoiceQuestion, FakeDecisionBackend, JSONContent
from graphwalk.embeddings import FakeEmbedder
from graphwalk.eval.datasets import hotpotqa
from graphwalk.eval.suite import preset_config
from graphwalk.eval.systems import GraphReaderSystem, node_line
from graphwalk.eval.text_qa import (
    GRAPHWALK,
    READER,
    TEXT_ITER_RAG,
    TEXT_RAG,
    build_graph,
    build_text_systems,
    by_type,
    graph_key,
    paragraph_docs,
    run_text_qa,
    stratified,
    to_questions,
)
from graphwalk.eval.types import EvalQuestion
from graphwalk.ingest import IngestConfig
from graphwalk.llm import FakeLLM, Message
from graphwalk.stores.networkx_store import NetworkXStore
from graphwalk.traversal import ChoiceEntryResolver, NameEntryResolver, Traverser
from graphwalk.traversal.entry import ENTRY_KEY
from kg_fixtures import current_key, movie_store, option_key

ROUTE = {
    "Inception": {"Christopher Nolan": 0.9, "Leonardo DiCaprio": 0.1},
    "Christopher Nolan": {"London": 0.9, "STOP": 0.1},
    "London": {"STOP": 1.0},
    "Titanic": {"James Cameron": 1.0},
    "James Cameron": {"STOP": 1.0},
}


def decide(question: ChoiceQuestion, _state: JSONContent) -> Mapping[str, float]:
    """Entry questions: the first candidate. Walk questions: follow ``ROUTE``."""
    if question.key == ENTRY_KEY:
        return {label: float(i == 0) for i, label in enumerate(question.options)}
    instructions = question.instructions
    if isinstance(instructions, dict) and "mention" in instructions:  # ingestion routing
        return {label: float(label == "NEW") for label in question.options}
    weights = ROUTE.get(current_key(question), {})
    dist = {label: weights.get(option_key(d), 0.0) for label, d in question.options.items()}
    return dist if sum(dist.values()) > 0 else dict.fromkeys(dist, 1.0)


def question(text: str, answer: str, type_: str = "compositional") -> EvalQuestion:
    return EvalQuestion(
        id=text, dataset="movies", question=text, answers=(answer,), kind="text",
        meta={"type": type_},
    )  # fmt: skip


async def reader_system(reply: str = "London") -> tuple[GraphReaderSystem, FakeLLM]:
    store = await movie_store()
    decider = FakeDecisionBackend(script=decide, cost_per_call=0.001)
    traverser = Traverser(store, decider, config=preset_config("greedy"))
    llm = FakeLLM(lambda _m: reply, cost_per_call=0.01)
    system = GraphReaderSystem(
        store, traverser, llm, ChoiceEntryResolver(store, decider), names=NameEntryResolver(store)
    )
    return system, llm


async def test_reader_reads_the_walk_and_node_facts() -> None:
    system, llm = await reader_system()
    answer = await system.answer(question("Where was the director of Inception born?", "London"))
    assert answer.status == "ok"
    assert answer.answers == ("London",)
    assert answer.start == ("inception",)
    assert answer.llm_calls == 1
    walk_calls = answer.decision_calls
    assert answer.cost_usd == pytest.approx(0.01 + 0.001 * walk_calls)
    prompt = llm.calls[0][1].content
    assert "Inception --directed_by--> Christopher Nolan --born_in--> London" in prompt
    assert "Christopher Nolan (person) [" not in prompt  # no attributes on this node
    assert "  - Christopher Nolan born_in London" in prompt
    assert system.describe()["system"] == "graphwalk-reader"


async def test_reader_walks_from_every_named_entity() -> None:
    system, llm = await reader_system("Inception")
    answer = await system.answer(
        question("Who directed Inception and who directed Titanic?", "x", "comparison")
    )
    assert set(answer.start or ()) == {"inception", "titanic"}
    walks = answer.detail["walks"]
    assert isinstance(walks, list)
    assert len(walks) == 2
    assert "James Cameron" in llm.calls[0][1].content


async def test_reader_without_entry_or_with_llm_failure() -> None:
    system, llm = await reader_system()
    none = await system.answer(question("Who painted the Mona Lisa?", "Leonardo da Vinci"))
    assert none.status == "no_entry"
    assert llm.calls == []

    class Broken(FakeLLM):
        async def complete(self, messages: Sequence[Message]) -> Any:
            from graphwalk.llm import LLMError

            raise LLMError("down")

    store = await movie_store()
    decider = FakeDecisionBackend(script=decide)
    broken = GraphReaderSystem(
        store,
        Traverser(store, decider, config=preset_config("greedy")),
        Broken(),
        ChoiceEntryResolver(store, decider),
    )
    failed = await broken.answer(question("Where was the director of Inception born?", "London"))
    assert failed.status == "error"
    assert failed.error == "down"


def test_node_line() -> None:
    from factories import node

    assert node_line(node("a", name="Ada", type_="person", summary="A poet.", born=1815)) == (
        "Ada (person): A poet. [born: 1815]"
    )


RECORDS: list[dict[str, Any]] = [
    {"_id": str(i), "type": t, "question": f"q{i}", "answer": f"a{i}",
     "context": [[f"T{i}", [f"Text {i}."]], ["Shared", ["Same."]]]}
    for i, t in enumerate(["x", "y", "x", "y", "x"])
]  # fmt: skip


def test_sampling_questions_and_documents() -> None:
    sample = stratified(RECORDS, ["x", "y"], 1, seed=0)
    assert sorted(r["type"] for r in sample) == ["x", "y"]
    assert stratified(RECORDS, ["x", "y"], 1, seed=0) == sample
    assert len(stratified(RECORDS, ["x"], 10, seed=0)) == 3
    questions = to_questions(RECORDS[:2], "d")
    assert (questions[0].id, questions[0].answers, questions[0].meta) == (
        "d-0",
        ("a0",),
        {"type": "x"},
    )
    assert paragraph_docs(RECORDS[:2]) == [
        ("T0", "T0: Text 0."), ("Shared", "Shared: Same."), ("T1", "T1: Text 1.")
    ]  # fmt: skip
    base = graph_key("d", RECORDS, IngestConfig())
    assert base == graph_key("d", RECORDS, IngestConfig())
    assert base != graph_key("d", RECORDS, IngestConfig(escalate=False))
    assert base != graph_key("d", RECORDS[:2], IngestConfig())


def test_hotpotqa_read_records(tmp_path: Path) -> None:
    row = {
        "id": "h1", "question": "Q?", "answer": "yes", "type": "comparison", "level": "hard",
        "context": {"title": ["A", "B"], "sentences": [["a1", "a2"], ["b1"]]},
    }  # fmt: skip
    path = tmp_path / "v.json"
    path.write_text(json.dumps([row]), encoding="utf-8")
    (record,) = hotpotqa.read_records(path)
    assert record["_id"] == "h1"
    assert record["context"] == [["A", ["a1", "a2"]], ["B", ["b1"]]]


EXTRACT = {
    "entities": [{"name": "Christopher Nolan", "type": "person"}, {"name": "London"}],
    "relations": [{"source": "Christopher Nolan", "type": "born_in", "target": "London"}],
}


async def test_text_qa_end_to_end(tmp_path: Path) -> None:
    records = [
        {"_id": "1", "type": "compositional", "question": "Where was Christopher Nolan born?",
         "answer": "London", "context": [["Christopher Nolan", ["Nolan was born in London."]]]},
        {"_id": "2", "type": "comparison", "question": "Is London in England?",
         "answer": "yes", "context": [["London", ["London is in England."]]]},
    ]  # fmt: skip

    def respond(messages: Sequence[Message]) -> str:
        if "extract a knowledge graph" in messages[0].content:
            return json.dumps(EXTRACT)
        return "London"

    llm = FakeLLM(respond, cost_per_call=0.001)
    decider = FakeDecisionBackend(script=decide, cost_per_call=0.0001)
    embedder = FakeEmbedder()
    graph = tmp_path / "g.graph.json"
    store, report = await build_graph(
        "movies", records, llm=llm, decider=decider, embedder=embedder, cache={},
        config=IngestConfig(escalate=False), graph_path=graph,
    )  # fmt: skip
    assert report is not None
    assert graph.exists()
    _, again = await build_graph(
        "movies", records, llm=llm, decider=decider, embedder=embedder, cache={},
        config=IngestConfig(), graph_path=graph,
    )  # fmt: skip
    assert again is None  # loaded from the cached graph

    systems = await build_text_systems(
        [GRAPHWALK, READER, TEXT_RAG, TEXT_ITER_RAG],
        store,
        records,
        decider=decider,
        embedder=embedder,
        llm=llm,
        config=preset_config("greedy-v2"),
        rag_k=1,
    )
    assert [s.name for s in systems] == [GRAPHWALK, READER, TEXT_RAG, TEXT_ITER_RAG]
    assert systems[2].describe()["document"] == "one per paragraph: title and text"
    with pytest.raises(ValueError, match="unknown system"):
        await build_text_systems(
            ["nope"], store, records, decider=decider, embedder=embedder, llm=llm,
            config=preset_config("greedy"),
        )  # fmt: skip

    seen: list[str] = []
    out, runs = await run_text_qa(
        "movies", records, systems, out_dir=tmp_path / "out", params={"n": 2},
        ingest=report, on_progress=lambda name, done, total: seen.append(name),
    )  # fmt: skip
    assert len(runs) == 4
    assert seen.count(TEXT_RAG) == 2
    rag_prompt = next(
        c
        for c in llm.calls
        if c[0].content.startswith("You answer questions using only the given passages")
    )
    assert "Passages:\n- Christopher Nolan: Nolan was born in London." in rag_prompt[1].content
    summary = (out / "summary.md").read_text(encoding="utf-8")
    assert "## By question type" in summary
    assert "| text-rag | " in by_type(runs)
    assert "## Ingestion" in summary
    rag = runs[2]
    assert rag.summary.f1 == pytest.approx(0.5)  # "London" is right once, wrong once

    # With checkpoints, a re-run reuses finished systems instead of answering again.
    checkpoints = tmp_path / "ckpt"
    await run_text_qa(
        "movies", records, systems[2:3], out_dir=tmp_path / "o1", params={}, ingest=None,
        checkpoint_dir=checkpoints,
    )  # fmt: skip
    calls = len(llm.calls)
    _, resumed = await run_text_qa(
        "movies", records, systems[2:3], out_dir=tmp_path / "o2", params={}, ingest=None,
        checkpoint_dir=checkpoints,
    )  # fmt: skip
    assert len(llm.calls) == calls
    assert resumed[0].summary.f1 == pytest.approx(0.5)


async def test_build_graph_resumes_from_a_partial_checkpoint(tmp_path: Path) -> None:
    records = [
        {"_id": str(i), "type": "x", "question": "q", "answer": "a",
         "context": [[f"Doc {i}", [f"Text {i}."]]]}
        for i in range(3)
    ]  # fmt: skip
    llm = FakeLLM(lambda _m: json.dumps(EXTRACT))
    decider = FakeDecisionBackend(script=decide)
    config = IngestConfig(escalate=False, route_concurrency=1)
    graph = tmp_path / "g.graph.json"
    # A run interrupted after the first document: its checkpoint has that doc's ledger.
    partial_store, _ = await build_graph(
        "d", records[:1], llm=llm, decider=decider, embedder=None, cache={}, config=config
    )
    await partial_store.save(graph.with_suffix(".partial.json"))
    store, report = await build_graph(
        "d", records, llm=llm, decider=decider, embedder=None, cache={}, config=config,
        graph_path=graph,
    )  # fmt: skip
    assert report is not None
    assert (report.unchanged, report.new) == (1, 2)
    assert graph.exists()
    assert not graph.with_suffix(".partial.json").exists()
    assert await store.counts() == await (await NetworkXStore.load(graph)).counts()
