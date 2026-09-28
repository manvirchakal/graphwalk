"""Eval harness: metrics, loaders (from local files), systems, runner, CLI. No network."""

import json
import math
from collections.abc import Sequence
from pathlib import Path

import pytest
from typer.testing import CliRunner

from graphwalk import cli
from graphwalk.embeddings import FakeEmbedder
from graphwalk.eval import (
    DocIndex,
    EvalQuestion,
    GraphwalkSystem,
    QASystem,
    SystemAnswer,
    VectorRAGSystem,
    normalize,
    run_system,
    sample_questions,
    score,
)
from graphwalk.eval.datasets import metaqa, twowiki
from graphwalk.eval.metrics import percentile
from graphwalk.eval.runner import summary_table, write_results
from graphwalk.eval.suite import Factories, build_systems, preset_config
from graphwalk.eval.systems import entity_documents, parse_answers
from graphwalk.llm import FakeLLM, Message
from graphwalk.traversal import NameEntryResolver, Traverser
from kg_fixtures import movie_store, oracle

# -- metrics ---------------------------------------------------------------------------


def test_normalize() -> None:
    assert normalize("The  Dark Knight!") == "dark knight"
    assert normalize("An apple, a day") == "apple day"


def test_set_scores() -> None:
    s = score(["Y", "X"], ["X", "Y", "Z"], ["x", "y"], "set")
    assert s.hits1 == 1.0
    assert s.em == 0.0
    assert s.precision == pytest.approx(2 / 3)
    assert s.recall == 1.0
    assert s.f1 == pytest.approx(0.8)
    assert score(["x"], ["x", "y"], ["y", "x"], "set").em == 1.0
    empty = score([], [], ["x"], "set")
    assert (empty.hits1, empty.em, empty.f1) == (0.0, 0.0, 0.0)


def test_text_scores() -> None:
    s = score(["Missoula, Montana"], ["Missoula, Montana"], ["Missoula, Montana"], "text")
    assert (s.hits1, s.em, s.f1) == (1.0, 1.0, 1.0)
    partial = score(["Missoula"], ["Missoula"], ["Missoula, Montana"], "text")
    assert partial.em == 0.0
    assert partial.f1 == pytest.approx(2 / 3)


def test_percentile() -> None:
    assert percentile([3, 1, 2, 4], 50) == 2
    assert percentile([3, 1, 2, 4], 95) == 4
    assert math.isnan(percentile([], 50))


# -- loaders ---------------------------------------------------------------------------


async def test_metaqa_loader(tmp_path: Path) -> None:
    kb = tmp_path / "kb.txt"
    kb.write_text(
        "Inception|directed_by|Christopher Nolan\n"
        "Inception|release_year|2010\n"
        "Memento|directed_by|Christopher Nolan\n"
        "Inception|directed_by|Christopher Nolan\n"  # duplicate
        "bad line\n",
        encoding="utf-8",
    )
    triples = metaqa.read_triples(kb)
    assert len(triples) == 3
    store = await metaqa.build_store(triples, "metaqa:test")
    assert await store.counts() == (4, 3)
    nolan = await store.get_node("Christopher Nolan")
    assert nolan is not None
    assert nolan.type == "person"
    film = await store.get_node("Inception")
    assert film is not None
    assert film.type == "film"
    qa = tmp_path / "qa.txt"
    qa.write_text(
        "who directed [Inception]\tChristopher Nolan\n"
        "what films did [Christopher Nolan] direct\tInception|Memento\n"
        "no topic here\tX\n",
        encoding="utf-8",
    )
    questions = metaqa.read_questions(qa, hops=1)
    assert len(questions) == 2
    assert questions[1].question == "what films did Christopher Nolan direct"
    assert questions[1].start == ("Christopher Nolan",)
    assert questions[1].answers == ("Inception", "Memento")
    assert questions[1].kind == "set"


async def test_twowiki_pools_evidence_and_keeps_walkable_types(tmp_path: Path) -> None:
    records = [
        {
            "_id": "a",
            "type": "compositional",
            "question": "Where was the director of film X born?",
            "evidences": [["X", "director", "D"], ["D", "place of birth", "Paris"]],
            "answer": "Paris",
        },
        {
            "_id": "b",
            "type": "comparison",
            "question": "Which came first, X or Y?",
            "evidences": [["X", "publication date", "1990"], ["Y", "publication date", "1991"]],
            "answer": "X",
        },
    ]
    path = tmp_path / "dev.json"
    path.write_text(json.dumps(records), encoding="utf-8")
    loaded = twowiki.read_records(path)
    store = await twowiki.build_graph(loaded)
    assert await store.counts() == (6, 4)  # comparison evidence still pooled as distractors
    questions = twowiki.read_questions(loaded)
    assert [q.id for q in questions] == ["2wiki-a"]
    assert questions[0].gold_start == ("X",)
    assert questions[0].start is None
    assert questions[0].kind == "text"


def test_sample_questions_is_seeded_and_ordered() -> None:
    qs = [
        EvalQuestion(id=str(i), dataset="d", question="q", answers=("a",), kind="set")
        for i in range(50)
    ]
    first = sample_questions(qs, 10, seed=3)
    assert first == sample_questions(qs, 10, seed=3)
    assert [int(q.id) for q in first] == sorted(int(q.id) for q in first)
    assert sample_questions(qs, None, seed=0) == qs


# -- systems ---------------------------------------------------------------------------

BORN = EvalQuestion(
    id="q1",
    dataset="movies",
    question="Where was the director of Inception born?",
    answers=("London",),
    kind="text",
    start=("inception",),
    gold_start=("inception",),
)
ROUTE = {
    "Inception": {"Christopher Nolan": 0.9, "Leonardo DiCaprio": 0.1},
    "Christopher Nolan": {"London": 0.9, "STOP": 0.1},
    "London": {"STOP": 1.0},
}


async def graphwalk_system(linking: str = "given") -> GraphwalkSystem:
    store = await movie_store()
    traverser = Traverser(store, oracle(ROUTE, cost_per_call=0.001), config=preset_config("beam"))
    return GraphwalkSystem(
        store,
        traverser,
        linking=linking,  # pyright: ignore[reportArgumentType]
        resolver=NameEntryResolver(store),
    )


async def test_graphwalk_system_answers_and_accounts() -> None:
    system = await graphwalk_system()
    assert isinstance(system, QASystem)
    answer = await system.answer(BORN)
    assert answer.status == "ok"
    assert answer.answers[0] == "London"
    assert answer.answer_set == ("London",)
    assert answer.decision_calls >= 3
    assert answer.cost_usd == pytest.approx(0.001 * answer.decision_calls)
    assert answer.detail["path"] == ["directed_by:out", "born_in:out"]
    assert system.describe()["traversal"]["strategy"] == "beam"  # type: ignore[index]


async def test_graphwalk_linking_modes() -> None:
    no_start = BORN.model_copy(update={"start": None})
    resolved = await (await graphwalk_system("resolve")).answer(no_start)
    assert resolved.start == ("inception",)
    assert resolved.answers[0] == "London"
    missing = await (await graphwalk_system("given")).answer(no_start)
    assert missing.status == "no_entry"
    ghost = BORN.model_copy(update={"start": ("ghost",)})
    assert (await (await graphwalk_system()).answer(ghost)).status == "no_entry"


async def test_entity_documents_and_rag() -> None:
    store = await movie_store()
    docs = dict(await entity_documents(store, max_facts=2))
    assert docs["nolan"].startswith("Christopher Nolan (person). Christopher Nolan born_in London")
    assert docs["nolan"].count(";") == 1  # capped at 2 facts
    embedder = FakeEmbedder()
    index = await DocIndex.build(list(docs.items()), embedder, batch=3)
    assert index.vectors.shape[0] == len(docs)
    seen: list[Sequence[Message]] = []

    def respond(messages: Sequence[Message]) -> str:
        seen.append(messages)
        return "London | unknown\n- Paris"

    rag = VectorRAGSystem(index, embedder, FakeLLM(respond, cost_per_call=0.002), k=3)
    answer = await rag.answer(BORN)
    assert answer.answers == ("London", "Paris")
    assert answer.llm_calls == 1
    assert answer.cost_usd == 0.002
    assert len(answer.detail["retrieved"]) == 3  # type: ignore[arg-type]
    assert "Question: Where was the director of Inception born?" in seen[0][1].content
    with pytest.raises(ValueError, match="index built with"):
        VectorRAGSystem(index, FakeEmbedder(model_id="other"), FakeLLM())


def test_parse_answers() -> None:
    assert parse_answers("unknown") == []
    assert parse_answers("A | B | A") == ["A", "B"]
    assert parse_answers("* A\n* B") == ["A", "B"]


# -- runner ----------------------------------------------------------------------------


class Exploding:
    name = "exploding"

    def describe(self) -> dict[str, object]:
        return {}

    async def answer(self, question: EvalQuestion) -> SystemAnswer:
        raise RuntimeError(question.id)


async def test_run_system_scores_and_survives_exceptions(tmp_path: Path) -> None:
    good = await run_system(await graphwalk_system(), [BORN, BORN], concurrency=2)
    assert good.summary.n == 2
    assert good.summary.hits1 == 1.0
    assert good.summary.linking_accuracy == 1.0
    assert good.summary.cost_per_query_usd is not None
    bad = await run_system(Exploding(), [BORN])  # pyright: ignore[reportArgumentType]
    assert bad.summary.status == {"error": 1}
    assert bad.records[0].answer.error == "RuntimeError: q1"
    table = summary_table([good, bad])
    assert "graphwalk" in table
    assert "exploding" in table
    out = write_results(tmp_path / "run", dataset="movies", runs=[good], params={"n": 2})
    document = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert document["runs"][0]["summary"]["hits1"] == 1.0
    assert "packages" in document["environment"]
    assert "| graphwalk |" in (out / "summary.md").read_text(encoding="utf-8")


async def test_build_systems_from_presets() -> None:
    store = await movie_store()
    factories = Factories(
        decider=lambda: oracle(ROUTE),
        embedder=FakeEmbedder,
        llm=lambda: FakeLLM(lambda _: "London"),
    )
    systems = await build_systems(
        ["greedy", "relation", "rag"],
        store,
        "movies",
        factories,
        linking="given",
        cache_index=False,
    )
    assert [s.name for s in systems] == ["graphwalk-greedy", "graphwalk-relation", "vector-rag"]
    results = [await s.answer(BORN) for s in systems]
    assert results[0].answers[0] == "London"
    assert results[2].answers == ("London",)
    with pytest.raises(ValueError, match="unknown preset"):
        preset_config("nope")


def test_eval_cli_rejects_bad_input() -> None:
    runner = CliRunner()
    assert runner.invoke(cli.app, ["eval", "nope"]).exit_code == 2
    assert runner.invoke(cli.app, ["eval", "metaqa", "--system", "bogus"]).exit_code == 2
