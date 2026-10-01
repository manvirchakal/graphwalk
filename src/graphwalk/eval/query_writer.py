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
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Literal

from pydantic import BaseModel, JsonValue, ValidationError

from graphwalk.core.model import NodeId
from graphwalk.eval.types import EvalQuestion, SystemAnswer
from graphwalk.llm.base import LLMBackend, LLMError, Message
from graphwalk.stores.base import AdjacencyStore, GraphStore
from graphwalk.traversal.entry import EntryLink, EntryResolver

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
    relations: Mapping[str, tuple[str, str]],
    glosses: Mapping[str, str] | None = None,
    counts: Mapping[str, int] | None = None,
) -> str:
    """One line per relation: ``subject_type --relation--> object_type: gloss``, with
    the edge count when ``counts`` is given (most frequent first)."""
    names = sorted(relations)
    if counts is not None:
        names.sort(key=lambda n: -counts.get(n, 0))
    lines: list[str] = []
    for name in names:
        subject, obj = relations[name]
        line = f"- {subject} --{name}--> {obj}"
        if counts is not None:
            line += f" ({counts.get(name, 0)} edges)"
        gloss = (glosses or {}).get(name)
        lines.append(line + (f": {gloss}" if gloss else ""))
    return "\n".join(lines)


async def schema_from_store(store: GraphStore) -> tuple[dict[str, tuple[str, str]], dict[str, int]]:
    """Each relation's most common ``(subject type, object type)``, and its edge count:
    the schema of a graph that has none declared (e.g. one extracted from text)."""
    types = {node.id: node.type async for node in store.iter_nodes()}
    pairs: dict[str, Counter[tuple[str, str]]] = defaultdict(Counter)
    async for edge in store.iter_edges():
        pairs[edge.type][(types.get(edge.source, "entity"), types.get(edge.target, "entity"))] += 1
    relations = {name: c.most_common(1)[0][0] for name, c in pairs.items()}
    return relations, {name: c.total() for name, c in pairs.items()}


async def local_relations(
    store: AdjacencyStore, start: Sequence[NodeId], *, hops: int = 2, max_nodes: int = 5000
) -> Counter[str]:
    """Edge counts per relation within ``hops`` of ``start`` (both directions): the
    schema a practitioner can fetch around the question's entity when the whole schema
    is too large to show. Each hop expands at most ``max_nodes`` nodes (hubs are cut)."""
    counts: Counter[str] = Counter()
    frontier = list(dict.fromkeys(start))
    seen = set(frontier)
    for _ in range(hops):
        following: list[NodeId] = []
        for node_id in frontier[:max_nodes]:
            for edge in await store.adjacency(node_id, direction="both"):
                counts[edge.relation] += 1
                if edge.other not in seen:
                    seen.add(edge.other)
                    following.append(edge.other)
        frontier = following
    return counts


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
        counts: Mapping[str, int] | None = None,
        resolver: EntryResolver | None = None,
        retries: int = 0,
        name: str = "llm-path",
    ) -> None:
        """``relations`` maps each relation to ``(subject type, object type)``.
        ``resolver`` links questions that come without a start entity (the same linker
        graphwalk uses, so both start from the same place)."""
        self._store = store
        self._llm = llm
        self._relations = dict(relations)
        self._schema = schema_text(relations, glosses, counts)
        self._resolver = resolver
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
        link = EntryLink(nodes=tuple(question.start or ()))
        if not link.nodes and self._resolver is not None:
            link = await self._resolver.link(question.question)
        if not link.nodes:
            if question.start is None and self._resolver is None:
                msg = "llm-path needs the question's start entity or a resolver"
                raise ValueError(msg)
            return SystemAnswer(
                answers=(), answer_set=(), status="no_entry", start=(),
                decision_calls=link.decision_calls, cost_usd=link.cost_usd,
            )  # fmt: skip
        start = list(link.nodes)
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
        calls, tokens_in, tokens_out, llm_s = 0, 0, 0, link.latency_s
        costs: list[float | None] = [link.cost_usd] if link.decision_calls else []
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
            decision_calls=link.decision_calls,
            input_tokens=tokens_in + link.input_tokens,
            output_tokens=tokens_out + link.output_tokens,
            cost_usd=None if any(c is None for c in costs) else math.fsum(c or 0 for c in costs),
            start=tuple(start),
            detail={"attempts": attempts, "shown": list[JsonValue](names[:MAX_ANSWERS_SHOWN])},
        )
