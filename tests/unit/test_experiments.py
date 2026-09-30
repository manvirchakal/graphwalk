"""Phase 4 experiment code: evidence retrieval (E1), LLM-written queries (E2),
calibration (E3), bootstrap CIs (E4), and the compact score files."""

import json
import math
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from graphwalk.core.hashing import content_hash
from graphwalk.core.model import StoredDocument
from graphwalk.decisions import FakeDecisionBackend
from graphwalk.embeddings import FakeEmbedder
from graphwalk.eval.calibration import auroc, calibration_row, confidence, ece
from graphwalk.eval.evidence import (
    EvidenceQuestion,
    EvidenceRow,
    complete_at,
    gold_titles,
    hybrid_keys,
    names_title,
    paired_table,
    recall_at,
    recall_table,
    retrieval_rows,
    type_table,
)
from graphwalk.eval.metrics import Score, bootstrap_ci
from graphwalk.eval.query_writer import (
    PathQuery,
    PathQuerySystem,
    execute,
    parse_path,
    schema_from_store,
)
from graphwalk.eval.runner import SCORES_FILE, Record, run_system, write_results
from graphwalk.eval.suite import preset_config
from graphwalk.eval.systems import TEXT_PROMPTS
from graphwalk.eval.text_qa import TEXT_ITER_RAG, build_text_systems, to_questions
from graphwalk.eval.types import EvalQuestion, SystemAnswer
from graphwalk.llm import FakeLLM, Message
from graphwalk.locate.dense import DenseLocator
from graphwalk.locate.documents import StoredDocuments
from graphwalk.locate.model import Location
from graphwalk.stores.networkx_store import NetworkXStore
from graphwalk.traversal import NameEntryResolver
from kg_fixtures import movie_store

if TYPE_CHECKING:
    from pydantic import JsonValue

# -- E1: evidence metrics ------------------------------------------------------------


def test_gold_titles_prefer_supporting_facts() -> None:
    record = {
        "supporting_facts": [["B", 0], ["A", 2], ["B", 1]],
        "context": [["A", ["a"]], ["B", ["b"]], ["C", ["c"]]],
    }
    assert gold_titles(record) == ["B", "A"]
    assert gold_titles({"context": [["P", ["x"]], ["Q", ["y"]]]}) == ["P", "Q"]


def test_recall_and_complete() -> None:
    gold = ["a", "b"]
    assert recall_at(gold, ["x", "a", "b"], 2) == 0.5
    assert recall_at(gold, ["x", "a", "b"], 3) == 1.0
    assert complete_at(gold, ["a", "x", "b"], 2) == 0.0
    assert complete_at(gold, ["a", "x", "b"], 3) == 1.0
    assert math.isnan(recall_at([], ["a"], 1))


def loc(key: str, start: int | None = None, end: int | None = None) -> Location:
    return Location(key=key, start=start, end=end, score=1.0, via="graph", element="node")


def test_hybrid_fuses_documents() -> None:
    graph = [loc("g1"), loc("both")]
    dense = [loc("both", 0, 10), loc("d1", 0, 10)]
    # "both" is in both lists, so it outranks everything else.
    assert hybrid_keys(graph, dense, 3) == ["both", "g1", "d1"]
    assert hybrid_keys(graph, dense, 1) == ["g1"]  # only the first of each list


def test_names_title_ignores_case_and_disambiguator() -> None:
    assert names_title("Who directed polish-russian war?", "Polish-Russian War (film)")
    assert not names_title("A warm day", "War")
    assert not names_title("Is it Al?", "Al")  # too short to count


def row(method: str, qid: str, ranked: Sequence[str], type_: str = "t") -> EvidenceRow:
    return EvidenceRow(
        question_id=qid, type=type_, method=method, gold=("a", "b"), ranked=tuple(ranked)
    )


def test_tables_render() -> None:
    rows = [
        row("dense", "q1", ["a", "x"]),
        row("dense", "q2", ["x", "y"]),
        row("graph", "q1", ["a", "b"]),
        row("graph", "q2", ["b"]),
    ]
    text = recall_table(rows, [1, 2], 2)
    assert "| dense | 0.250 | 0.250 |" in text
    assert "| graph | 0.500 | 0.750 |" in text
    assert "| graph - dense | +0.250" in paired_table(rows, [("graph", "dense")], [1])
    assert "0.750 (2)" in type_table(rows, 2)
    hybrid = row("hybrid", "q1", ["x"]).model_copy(update={"by_k": {1: ("a",)}})
    assert hybrid.at(1) == ("a",)
    assert hybrid.at(2) == ("x",)


async def test_retrieval_rows_end_to_end(tmp_path: Path) -> None:
    store = await movie_store()
    now = datetime.now(UTC)
    texts = {
        "src/inception": "Inception is a 2010 film directed by Christopher Nolan.",
        "src/nolan": "Christopher Nolan is a director born in London.",
        "src/other": "Titanic is a film by James Cameron.",
    }
    for key, text in texts.items():
        await store.put_document(
            StoredDocument(
                key=key, source_id="src", doc_id=key.split("/")[1], title=key.split("/")[1],
                text=text, text_hash=content_hash(text), length=len(text), ingested_at=now,
            )
        )  # fmt: skip
    questions = [
        EvidenceQuestion(
            id="q1", text="Where was the director of Inception born?", type="compositional",
            gold=("src/nolan",),
        )
    ]  # fmt: skip
    decider = FakeDecisionBackend()
    kwargs = {
        "decider": decider, "embedder": FakeEmbedder(), "config": preset_config("greedy"),
        "ks": (1, 2), "checkpoints": tmp_path,
    }  # fmt: skip
    rows = await retrieval_rows(store, questions, {"nolan": "src/nolan"}, **kwargs)  # pyright: ignore[reportArgumentType]
    methods = {r.method for r in rows}
    assert methods == {"dense", "title+dense", "graph", "graph-choice", "hybrid", "hybrid-choice"}
    graph = next(r for r in rows if r.method == "graph")
    assert graph.ranked  # the walk reached documents via provenance ("src" keys)
    assert not graph.locations  # dropped from the reported rows
    assert (tmp_path / "graph.jsonl").exists()
    calls = len(decider.requests)
    again = await retrieval_rows(store, questions, {"nolan": "src/nolan"}, **kwargs)  # pyright: ignore[reportArgumentType]
    assert len(decider.requests) == calls  # resumed from the checkpoints
    assert [r.ranked for r in again] == [r.ranked for r in rows]


async def test_dense_chunks_are_titled() -> None:
    store = NetworkXStore()
    text = "He was born in London."
    await store.put_document(
        StoredDocument(
            key="s/d", source_id="s", doc_id="d", title="Christopher Nolan", text=text,
            text_hash=content_hash(text), length=len(text), ingested_at=datetime.now(UTC),
        )
    )  # fmt: skip
    embedder = FakeEmbedder()
    found = await DenseLocator(store, StoredDocuments(store), embedder).locate("Nolan", k=1)
    assert embedder.calls[0] == ["Christopher Nolan\n\nHe was born in London."]
    assert (found[0].start, found[0].end) == (0, len(text))  # offsets are the text's


# -- E2: LLM-written relation paths ---------------------------------------------------

RELATIONS = {"directed_by": ("film", "person"), "starred_actors": ("film", "person")}


def test_parse_path() -> None:
    query = parse_path(
        'Sure: {"path": [{"relation": "directed_by", "direction": "in"}]}', RELATIONS
    )
    assert query.path[0].relation == "directed_by"
    for bad, message in (
        ("no json", "no JSON"),
        ('{"path": []}', "empty"),
        ('{"path": [{"relation": "born_in", "direction": "out"}]}', "unknown relations"),
        ('{"path": [{"relation": "directed_by", "direction": "up"}]}', "invalid path"),
    ):
        with pytest.raises(ValueError, match=message):
            parse_path(bad, RELATIONS)


async def test_execute_takes_every_target_and_skips_visited() -> None:
    store = await movie_store()
    films = PathQuery.model_validate({"path": [{"relation": "directed_by", "direction": "in"}]})
    assert sorted(await execute(store, ["nolan"], films)) == [
        "inception",
        "interstellar",
        "memento",
    ]
    co_stars = PathQuery.model_validate(
        {"path": [{"relation": "starred_actors", "direction": "in"},
                  {"relation": "starred_actors", "direction": "out"}]}
    )  # fmt: skip
    assert await execute(store, ["dicaprio"], co_stars) == []  # only DiCaprio, visited


async def test_path_query_system_retries() -> None:
    store = await movie_store()
    replies = iter([
        "not json",
        '{"path": [{"relation": "directed_by", "direction": "out"}]}',  # from a person: nothing
        '{"path": [{"relation": "directed_by", "direction": "in"}]}',
    ])  # fmt: skip
    llm = FakeLLM(lambda _m: next(replies), cost_per_call=0.001)
    system = PathQuerySystem(store, llm, RELATIONS, retries=2)
    question = EvalQuestion(
        id="q", dataset="movies", question="Which films did Christopher Nolan direct?",
        answers=("Inception", "Memento", "Interstellar"), kind="set", start=("nolan",),
    )  # fmt: skip
    answer = await system.answer(question)
    assert sorted(answer.answer_set) == ["Inception", "Interstellar", "Memento"]
    assert answer.llm_calls == 3
    assert answer.cost_usd == pytest.approx(0.003)
    assert "person --directed_by-->" not in llm.calls[0][1].content  # schema is film-first
    assert "film --directed_by--> person" in llm.calls[0][1].content
    assert "Christopher Nolan (person)" in llm.calls[0][1].content
    assert "returns no entities" in llm.calls[2][-1].content
    no_retry = PathQuerySystem(store, FakeLLM(lambda _m: "{}"), RELATIONS)
    assert (await no_retry.answer(question)).answer_set == ()


# -- E3: calibration ------------------------------------------------------------------


def test_calibration_metrics() -> None:
    perfect = [(1.0, 1.0), (0.0, 0.0)]
    assert ece(perfect) == 0.0
    assert auroc(perfect) == 1.0
    assert ece([(0.9, 0.0)]) == pytest.approx(0.9)
    assert auroc([(0.5, 1.0), (0.5, 0.0)]) == 0.5
    assert math.isnan(auroc([(0.5, 1.0)]))


def record(score_: float | None, em: float) -> Record:
    question = EvalQuestion(id="q", dataset="d", question="?", answers=("a",), kind="text")
    detail: dict[str, JsonValue] = {} if score_ is None else {"score": score_}
    answer = SystemAnswer(answers=("a",), answer_set=("a",), detail=detail)
    return Record(
        question=question,
        answer=answer,
        score=Score(hits1=em, em=em, precision=em, recall=em, f1=em),
        linked=None,
    )


def test_calibration_row() -> None:
    records = [record(math.log(0.9), 1.0), record(math.log(0.2), 0.0), record(None, 0.0)]
    assert confidence(records[0]) == pytest.approx(0.9)
    assert confidence(records[2]) is None
    line = calibration_row("jev", records)
    assert line.startswith("| jev | 3 | 0.367 | 0.333 |")
    assert "| 1.000 |" in line  # AUROC: the right answer is the most confident


# -- E4: bootstrap CIs ----------------------------------------------------------------


def test_bootstrap_ci() -> None:
    assert bootstrap_ci([0.5] * 10) == (0.5, 0.5)
    values = [0.0, 1.0] * 50
    lo, hi = bootstrap_ci(values)
    assert lo < 0.5 < hi
    assert bootstrap_ci(values) == (lo, hi)  # seeded
    assert all(math.isnan(v) for v in bootstrap_ci([]))


# -- multi-step replay and score files -------------------------------------------------


async def test_multi_step_replay_matches_the_run(tmp_path: Path) -> None:
    records = [
        {"_id": "1", "type": "compositional", "question": "Where was Inception's director born?",
         "answer": "London", "context": [["Inception", ["Inception was directed by Nolan."]],
                                         ["Christopher Nolan", ["Nolan was born in London."]],
                                         ["Titanic", ["Titanic is a film."]]]},
    ]  # fmt: skip

    def respond(messages: Sequence[Message]) -> str:
        if "Christopher Nolan: Nolan was born" in messages[-1].content:
            return "ANSWER: London"
        return "SEARCH: Christopher Nolan"

    store = await movie_store()
    (system,) = await build_text_systems(
        [TEXT_ITER_RAG], store, records, decider=FakeDecisionBackend(), embedder=FakeEmbedder(),
        llm=FakeLLM(respond), config=preset_config("greedy"), rag_k=1, prompts=TEXT_PROMPTS,
    )  # fmt: skip
    run = await run_system(system, to_questions(records, "movies"))
    detail = run.records[0].answer.detail
    retrieved = detail["retrieved"]
    assert isinstance(retrieved, list)
    assert retrieved[-1] == "Christopher Nolan"
    replay = await system.replay(records[0]["question"], detail["steps"])  # pyright: ignore[reportAttributeAccessIssue, reportArgumentType]
    assert replay == retrieved

    out = write_results(tmp_path / "run", dataset="movies", runs=[run], params={})
    (row_,) = [json.loads(line) for line in (out / SCORES_FILE).read_text().splitlines()]
    assert row_["system"] == TEXT_ITER_RAG
    assert row_["answers"] == ["London"]
    assert row_["f1"] == 1.0
    assert row_["type"] == "compositional"
    assert row_["confidence"] is None


async def test_path_query_links_and_reads_the_store_schema() -> None:
    store = await movie_store()
    relations, counts = await schema_from_store(store)
    assert relations["directed_by"] == ("film", "person")
    assert counts["directed_by"] == 4
    llm = FakeLLM(lambda _m: '{"path": [{"relation": "directed_by", "direction": "in"}]}')
    system = PathQuerySystem(
        store, llm, relations, counts=counts, resolver=NameEntryResolver(store)
    )
    question = EvalQuestion(
        id="q", dataset="movies", question="Which films did Christopher Nolan direct?",
        answers=("Memento",), kind="text",
    )  # fmt: skip
    answer = await system.answer(question)
    assert answer.start == ("nolan",)
    assert "Memento" in answer.answer_set
    assert "(4 edges)" in llm.calls[0][1].content
    unlinked = question.model_copy(update={"question": "Nothing here matches."})
    assert (await system.answer(unlinked)).status == "no_entry"
