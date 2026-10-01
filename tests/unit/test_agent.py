"""The A8 agent harness: the loop, the graph/walk/search tools (no network)."""

import json
from pathlib import Path
from typing import Any

import numpy as np

from graphwalk.eval.agent import (
    ANSWER_SPEC,
    AgentSystem,
    Turn,
    graph_tools,
    search_tool,
    triple_passages,
    walk_tool,
)
from graphwalk.eval.datasets import rog
from graphwalk.eval.systems import DocIndex
from graphwalk.eval.types import EvalQuestion
from graphwalk.stores.sqlite_store import SQLiteStore
from graphwalk.traversal import TraversalConfig, Traverser
from kg_fixtures import oracle

TRIPLES = [
    ("Jamaica", "location.country.languages_spoken", "Jamaican English"),
    ("Jamaica", "location.country.languages_spoken", "Jamaican Creole"),
    ("Jamaica", "location.statistical_region.gdp", "m.0nf4wmg"),
    ("m.0nf4wmg", "measurement.dated_money_value.amount", "14.4"),
]
QUESTION = EvalQuestion(
    id="q1", dataset="webqsp", question="what do jamaicans speak",
    answers=("Jamaican English", "Jamaican Creole"), kind="set", start=("Jamaica",),
)  # fmt: skip


async def store_at(tmp_path: Path) -> SQLiteStore:
    return await rog.build_global_store([TRIPLES], tmp_path / "g.db", source_id="t")


def call(name: str, args: dict[str, Any], cid: str = "c") -> dict[str, Any]:
    return {
        "id": cid,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


class ScriptedChat:
    """Replays assistant turns; records the tools offered each turn."""

    def __init__(self, turns: list[list[dict[str, Any]]]) -> None:
        self._turns = turns
        self.offered: list[list[str]] = []
        self.seen: list[list[dict[str, Any]]] = []

    async def __call__(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> Turn:
        self.offered.append([t["function"]["name"] for t in tools])
        self.seen.append(list(messages))
        calls = self._turns[len(self.offered) - 1]
        message = {"role": "assistant", "content": "", "tool_calls": calls}
        return Turn(message, input_tokens=100, output_tokens=10, cost_usd=0.001, latency_s=0.5)


async def test_agent_explores_then_answers(tmp_path: Path) -> None:
    store = await store_at(tmp_path)
    chat = ScriptedChat(
        [
            [call("relations", {"node": "jamaica"})],
            [
                call(
                    "neighbors",
                    {"node": "Jamaica", "relation": "location.country.languages_spoken"},
                )
            ],
            [call("answer", {"answers": ["Jamaican English", "Jamaican Creole"]})],
        ]
    )
    agent = AgentSystem("agent-graph", chat, lambda _q: graph_tools(store), max_turns=5)
    answer = await agent.answer(QUESTION)
    assert answer.answer_set == ("Jamaican English", "Jamaican Creole")
    assert answer.llm_calls == 3
    assert answer.input_tokens == 300
    assert answer.cost_usd is not None
    assert abs(answer.cost_usd - 0.003) < 1e-9
    assert answer.detail["tool_calls"] == {"relations": 1, "neighbors": 1, "answer": 1}
    relations_out = chat.seen[1][-1]["content"]
    assert "location.country.languages_spoken (out): 2" in relations_out  # name resolved
    neighbors_out = chat.seen[2][-1]["content"]
    assert "-> Jamaican Creole" in neighbors_out
    assert "gdp" not in neighbors_out  # filtered to the relation
    assert "Topic entities: Jamaica" in chat.seen[0][1]["content"]
    await store.close()


async def test_last_turn_offers_only_answer_and_errors_go_back(tmp_path: Path) -> None:
    store = await store_at(tmp_path)
    chat = ScriptedChat(
        [
            [call("neighbors", {"node": "Atlantis"})],
            [call("answer", {"answers": "Jamaican English"})],
        ]
    )
    agent = AgentSystem("agent-graph", chat, lambda _q: graph_tools(store), max_turns=2)
    answer = await agent.answer(QUESTION)
    assert chat.offered[0] == ["relations", "neighbors", "answer"]
    assert chat.offered[1] == [ANSWER_SPEC["function"]["name"]]
    assert "error: no node named 'Atlantis'" in chat.seen[1][-1]["content"]
    assert answer.answer_set == ("Jamaican English",)  # a bare string is accepted
    await store.close()


async def test_unanswered_agent_returns_nothing(tmp_path: Path) -> None:
    store = await store_at(tmp_path)
    chat = ScriptedChat([[call("relations", {"node": "Jamaica"})]] * 2)
    agent = AgentSystem("agent-graph", chat, lambda _q: graph_tools(store), max_turns=2)
    answer = await agent.answer(QUESTION)
    assert answer.answer_set == ()
    assert answer.detail["answered"] is False
    await store.close()


async def test_walk_tool_reports_answers_and_confidence(tmp_path: Path) -> None:
    store = await store_at(tmp_path)
    decider = oracle(
        {
            "Jamaica": {"location.country.languages_spoken": 1.0},
            "Jamaican English": {"STOP": 1.0},
            "Jamaican Creole": {"STOP": 1.0},
        }
    )
    config = TraversalConfig(strategy="greedy", hop_mode="relation", allow_stop_at_start=False)
    tool = walk_tool(store, Traverser(store, decider, config=config))
    out = await tool.run({"question": QUESTION.question, "start": ["Jamaica"]})
    assert out.text.startswith("confidence: 1.00")
    assert "location.country.languages_spoken: Jamaican Creole, Jamaican English" in out.text
    assert out.detail["confidence"] == 1.0
    await store.close()


def test_triple_passages_group_records() -> None:
    passages = triple_passages(TRIPLES)
    assert "Jamaica location.country.languages_spoken Jamaican English" in passages
    assert (
        "Jamaica location.statistical_region.gdp (record); "
        "(record) measurement.dated_money_value.amount 14.4"
    ) in passages
    assert len(passages) == 3


class OneHot:
    model_id = "onehot"

    async def embed(self, texts: Any) -> Any:
        return np.array(
            [[1.0, 0.0] if "speak" in t or "language" in t else [0.0, 1.0] for t in texts]
        )


async def test_search_tool_returns_nearest_passages() -> None:
    texts = triple_passages(TRIPLES)
    index = await DocIndex.build([(t, t) for t in texts], OneHot())
    tool = search_tool(index, OneHot(), k=2)
    out = await tool.run({"query": "what do they speak"})
    assert out.text.count("languages_spoken") == 2
