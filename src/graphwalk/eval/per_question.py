"""Systems over per-question graphs (KG-QA releases that ship one subgraph per question).

:class:`PerQuestionSystem` builds the question's graph, builds the wrapped system on it,
and answers. The graph is rebuilt for every system and question: subgraphs are small
(thousands of triples), and nothing leaks from one question's graph into another's.
"""

import time
from collections.abc import Awaitable, Callable

from pydantic import JsonValue

from graphwalk.eval.types import EvalQuestion, QASystem, SystemAnswer
from graphwalk.stores.base import GraphStore

type GraphFactory = Callable[[EvalQuestion], Awaitable[GraphStore]]
type SystemFactory = Callable[[GraphStore, EvalQuestion], QASystem]


class PerQuestionSystem:
    def __init__(
        self,
        name: str,
        graph: GraphFactory,
        system: SystemFactory,
        describe: dict[str, JsonValue],
    ) -> None:
        self._name = name
        self._graph = graph
        self._system = system
        self._describe = describe

    @property
    def name(self) -> str:
        return self._name

    def describe(self) -> dict[str, JsonValue]:
        return {**self._describe, "graph": "per question"}

    async def answer(self, question: EvalQuestion) -> SystemAnswer:
        started = time.perf_counter()
        store = await self._graph(question)
        build_s = time.perf_counter() - started
        try:
            answer = await self._system(store, question).answer(question)
        finally:
            await store.close()
        return answer.model_copy(update={"detail": {**answer.detail, "graph_build_s": build_s}})
