"""Per-question KG-QA graphs (WebQSP/CWQ releases) and the per-question system wrapper.
Offline: tiny hand-written subgraphs, no download."""

from pathlib import Path

from graphwalk.eval.datasets import rog
from graphwalk.eval.per_question import PerQuestionSystem
from graphwalk.eval.query_writer import local_relations
from graphwalk.eval.systems import GraphwalkSystem
from graphwalk.eval.types import EvalQuestion, QASystem
from graphwalk.stores.base import GraphStore
from graphwalk.traversal import TraversalConfig, Traverser
from kg_fixtures import oracle

GRAPHS = {
    "q1": [
        ("Jamaica", "location.country.languages_spoken", "Jamaican English"),
        ("Jamaica", "location.statistical_region.gdp", "m.0nf4wmg"),
    ],
    "q2": [("France", "location.country.capital", "Paris")],
}


def question(qid: str, text: str, start: str, answer: str) -> EvalQuestion:
    return EvalQuestion(
        id=qid, dataset="webqsp", question=text, answers=(answer,), kind="set",
        start=(start,), gold_start=(start,),
    )  # fmt: skip


async def test_build_store_types_cvt_nodes() -> None:
    store = await rog.build_store(GRAPHS["q1"], "webqsp:q1")
    cvt = await store.get_node("m.0nf4wmg")
    country = await store.get_node("Jamaica")
    assert cvt is not None
    assert country is not None
    assert cvt.type == rog.CVT_TYPE
    assert cvt.summary == rog.CVT_SUMMARY
    assert country.type == rog.ENTITY_TYPE
    assert await store.counts() == (3, 2)


def test_relation_signature_and_node_type() -> None:
    assert rog.node_type("g.12tb6gh4f") == rog.CVT_TYPE
    assert rog.node_type("m.0k8nh0b") == rog.CVT_TYPE
    assert rog.node_type("m.a. smith") == rog.ENTITY_TYPE
    assert rog.relation_signature(GRAPHS["q1"]) == {
        "location.country.languages_spoken": (rog.ENTITY_TYPE, rog.ENTITY_TYPE),
        "location.statistical_region.gdp": (rog.ENTITY_TYPE, rog.CVT_TYPE),
    }


async def test_per_question_system_uses_each_questions_graph() -> None:
    decider = oracle(
        {
            "Jamaica": {"location.country.languages_spoken": 1.0},
            "Jamaican English": {"STOP": 1.0},
            "France": {"location.country.capital": 1.0},
            "Paris": {"STOP": 1.0},
        }
    )
    config = TraversalConfig(strategy="greedy", hop_mode="relation", allow_stop_at_start=False)
    built: list[GraphStore] = []

    async def graph(q: EvalQuestion) -> GraphStore:
        store = await rog.build_store(GRAPHS[q.id], q.id)
        built.append(store)
        return store

    def system(store: GraphStore, q: EvalQuestion) -> QASystem:
        del q
        return GraphwalkSystem(store, Traverser(store, decider, config=config), name="walk")

    wrapped = PerQuestionSystem("walk", graph, system, {"system": "graphwalk"})
    assert wrapped.name == "walk"
    assert wrapped.describe()["graph"] == "per question"
    a1 = await wrapped.answer(question("q1", "what do jamaicans speak", "Jamaica", "x"))
    a2 = await wrapped.answer(question("q2", "capital of france", "France", "Paris"))
    assert a1.answer_set == ("Jamaican English",)
    assert a2.answer_set == ("Paris",)
    assert isinstance(a2.detail["graph_build_s"], float)
    assert len(built) == 2
    assert await built[1].get_node("Jamaica") is None


async def test_global_store_merges_subgraphs_and_is_reused(tmp_path: Path) -> None:
    path = tmp_path / "global.db"
    store = await rog.build_global_store(GRAPHS.values(), path, source_id="t")
    assert await store.counts() == (5, 3)
    cvt = await store.get_node("m.0nf4wmg")
    assert cvt is not None
    assert cvt.summary == rog.CVT_SUMMARY
    await store.close()
    again = await rog.build_global_store([], path, source_id="t")  # built: not rebuilt
    assert await again.counts() == (5, 3)
    await again.close()


async def test_local_relations_counts_edges_within_hops(tmp_path: Path) -> None:
    store = await rog.build_global_store(
        [[*GRAPHS["q1"], ("Jamaican English", "language.human_language.region", "Caribbean")]],
        tmp_path / "g.db",
        source_id="t",
    )
    one = await local_relations(store, ["Jamaica"], hops=1)
    assert one == {"location.country.languages_spoken": 1, "location.statistical_region.gdp": 1}
    two = await local_relations(store, ["Jamaica"], hops=2)
    assert two["language.human_language.region"] == 1
    assert two["location.country.languages_spoken"] == 2  # seen from both ends
    await store.close()
