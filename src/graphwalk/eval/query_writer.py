"""E2: the strongest practical competitor on a curated graph. An LLM reads the schema
and the question and writes the graph query (a relation path); code executes it.

This is text-to-query in its simplest executable form: the path language is exactly
what a relation-mode walk can express (a sequence of relation hops, each taking every
target), so the comparison isolates *who chooses the hops*: a calibrated classifier
deciding one hop at a time while seeing the graph, or an LLM planning the whole path up
front from the schema alone.

Execution matches relation-mode walks: each hop takes all neighbors over the relation
in the given direction, excluding nodes already visited; the answer is the final set.
With ``retries``, an invalid path or an empty result is shown back to the LLM, which
may try again (the cheap agentic fix).
"""

import json
import math
import re
import time
from collections.abc import Mapping, Sequence
from typing import Literal

from pydantic import BaseModel, JsonValue, ValidationError

from graphwalk.core.model import NodeId
from graphwalk.eval.types import EvalQuestion, SystemAnswer
from graphwalk.llm.base import LLMBackend, LLMError, Message
from graphwalk.stores.base import GraphStore

MAX_ANSWERS_SHOWN = 5

SYSTEM = (
    "You translate questions about a knowledge graph into a relation path that a program "
    "executes. Reply with JSON only."
)


class Hop(BaseModel):
    relation: str
    direction: Literal["out", "in"]


class PathQuery(BaseModel):
    path: list[Hop]


def schema_text(
    relations: Mapping[str, tuple[str, str]], glosses: Mapping[str, str] | None = None
) -> str:
    """One line per relation: ``subject_type --relation--> object_type: gloss``."""
    lines: list[str] = []
    for name, (subject, obj) in sorted(relations.items()):
        gloss = (glosses or {}).get(name)
        lines.append(f"- {subject} --{name}--> {obj}" + (f": {gloss}" if gloss else ""))
    return "\n".join(lines)


def instructions(schema: str) -> str:
    return (
        "The graph's relations (each edge points from its subject to its object):\n"
        f"{schema}\n\n"
        "Write the path from the start entity to the answer. Each hop follows one relation: "
        'direction "out" goes from subject to object, "in" goes from object back to '
        "subject. A hop takes every entity reachable that way, so a path returns a set. "
        "Entities already visited are never returned again.\n\n"
        'Reply with JSON only: {"path": [{"relation": "...", "direction": "out"|"in"}, ...]}'
    )


def parse_path(text: str, relations: Mapping[str, object]) -> PathQuery:
    """The first JSON object in ``text``; raises ``ValueError`` if it isn't a valid path
    over known relations."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match is None:
        msg = "no JSON object in the reply"
        raise ValueError(msg)
    try:
        query = PathQuery.model_validate(json.loads(match.group(0)))
    except (json.JSONDecodeError, ValidationError) as error:
        msg = f"invalid path JSON: {error}"
        raise ValueError(msg) from None
    if not query.path:
        msg = "the path is empty"
        raise ValueError(msg)
    unknown = [h.relation for h in query.path if h.relation not in relations]
    if unknown:
        msg = f"unknown relations {unknown}"
        raise ValueError(msg)
    return query


async def execute(store: GraphStore, start: Sequence[NodeId], query: PathQuery) -> list[NodeId]:
    """Run ``query`` from ``start`` as relation-mode hops (visited nodes excluded)."""
    frontier = list(dict.fromkeys(start))
    visited = set(frontier)
    for hop in query.path:
        targets: list[NodeId] = []
        for node_id in frontier:
            for neighbor in await store.neighbors(
                node_id, direction=hop.direction, edge_types=(hop.relation,)
            ):
                target = neighbor.node.id
                if target not in visited:
                    visited.add(target)
                    targets.append(target)
        frontier = targets
        if not frontier:
            break
    return frontier


class PathQuerySystem:
    """``QASystem``: the LLM writes a relation path from the given start entity."""

    def __init__(
        self,
        store: GraphStore,
        llm: LLMBackend,
        relations: Mapping[str, tuple[str, str]],
        *,
        glosses: Mapping[str, str] | None = None,
        retries: int = 0,
        name: str = "llm-path",
    ) -> None:
        """``relations`` maps each relation to ``(subject type, object type)``."""
        self._store = store
        self._llm = llm
        self._relations = dict(relations)
        self._schema = schema_text(relations, glosses)
        self._retries = retries
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def describe(self) -> dict[str, JsonValue]:
        return {
            "system": "llm-path",
            "llm": self._llm.model_id,
            "retries": self._retries,
            "relations": len(self._relations),
            "schema": self._schema,
        }

    async def answer(self, question: EvalQuestion) -> SystemAnswer:
        if not question.start:
            msg = "llm-path needs the question's start entity"
            raise ValueError(msg)
        start = list(question.start)
        nodes = await self._store.get_nodes(start)
        described = ", ".join(
            f"{nodes[s].name} ({nodes[s].type})" if s in nodes else s for s in start
        )
        messages = [
            Message(role="system", content=SYSTEM),
            Message(
                role="user",
                content=f"{instructions(self._schema)}\n\nStart entity: {described}\n"
                f"Question: {question.question}",
            ),
        ]
        calls, tokens_in, tokens_out, llm_s = 0, 0, 0, 0.0
        costs: list[float | None] = []
        attempts: list[JsonValue] = []
        answer_ids: list[NodeId] = []
        for _ in range(self._retries + 1):
            try:
                result = await self._llm.complete(messages)
            except LLMError as error:
                return SystemAnswer(
                    answers=(), answer_set=(), status="error", error=str(error),
                    latency_s=llm_s, llm_calls=calls + 1, input_tokens=tokens_in,
                    output_tokens=tokens_out, start=tuple(start),
                    detail={"attempts": attempts},
                )  # fmt: skip
            calls += 1
            tokens_in += result.input_tokens
            tokens_out += result.output_tokens
            llm_s += result.latency_s
            costs.append(result.cost_usd)
            messages.append(Message(role="assistant", content=result.text))
            try:
                query = parse_path(result.text, self._relations)
            except ValueError as error:
                attempts.append({"raw": result.text, "error": str(error)})
                messages.append(Message(role="user", content=f"{error}. Try again."))
                continue
            started = time.perf_counter()
            answer_ids = await execute(self._store, start, query)
            llm_s += time.perf_counter() - started
            path = [f"{h.relation}:{h.direction}" for h in query.path]
            attempts.append({"path": list[JsonValue](path), "answers": len(answer_ids)})
            if answer_ids:
                break
            messages.append(
                Message(
                    role="user",
                    content="That path returns no entities. Check each hop's direction "
                    "and relation against the schema, then try again.",
                )
            )
        found = await self._store.get_nodes(answer_ids)
        names = tuple(found[n].name for n in answer_ids if n in found)
        return SystemAnswer(
            answers=names,
            answer_set=names,
            # The LLM calls plus executing the path; excludes client-side rate-limit waits.
            latency_s=llm_s,
            llm_calls=calls,
            input_tokens=tokens_in,
            output_tokens=tokens_out,
            cost_usd=None if any(c is None for c in costs) else math.fsum(c or 0 for c in costs),
            start=tuple(start),
            detail={"attempts": attempts, "shown": list[JsonValue](names[:MAX_ANSWERS_SHOWN])},
        )
