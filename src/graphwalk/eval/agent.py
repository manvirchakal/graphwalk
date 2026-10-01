"""A minimal tool-calling agent, the same loop for every tool set (the harness for A8).

The question is how a capable agent answers knowledge-graph questions with different
tools, not how good one framework is, so the loop is deliberately plain: the model must
call a tool every turn; tool results go back verbatim; it ends by calling ``answer`` (or
is made to on its last turn). Every arm gets the same system prompt, model, and turn
budget; only the tools differ:

* ``graph``: :func:`graph_tools` (``relations`` and ``neighbors`` around a node);
* ``walk``: the same plus :func:`walk_tool` (a graphwalk walk with its confidence);
* ``search``: :func:`search_tool` only (dense search over the same facts as text).

Latency is the sum of model-call and tool latencies (rate-limiter waits excluded), so
it does not depend on the provider's request cap.
"""

import asyncio
import json
import logging
import time
from collections import Counter
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from pydantic import JsonValue

from graphwalk.core.redact import redact
from graphwalk.embeddings.base import Embedder
from graphwalk.eval.datasets import rog
from graphwalk.eval.systems import DocIndex
from graphwalk.eval.types import EvalQuestion, SystemAnswer
from graphwalk.llm.base import LLMError
from graphwalk.llm.litellm_import import import_litellm
from graphwalk.stores.sqlite_store import SQLiteStore
from graphwalk.traversal import Traverser

logger = logging.getLogger(__name__)

type Args = dict[str, Any]
type ToolSpec = dict[str, Any]

MAX_LISTED = 50
"""Most targets ``neighbors`` lists, and most relations ``relations`` lists."""

SYSTEM_PROMPT = """\
You answer questions using tools over a knowledge graph derived from Freebase. Nodes are \
named entities, plus anonymous record nodes with ids like m.0abc12 that tie several facts \
together (for example a marriage with its spouse and dates, or a government position \
with its office holder and term). Use the tools to find the answer, then call `answer` \
with the answer entities' names exactly as they appear in the graph (all of them if \
several are correct). Do not answer from memory: answer from what the tools return. You \
have at most {max_turns} turns; each turn is one round of tool calls."""

ANSWER_SPEC: ToolSpec = {
    "type": "function",
    "function": {
        "name": "answer",
        "description": "Give the final answer and stop.",
        "parameters": {
            "type": "object",
            "properties": {
                "answers": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Answer entity names, as they appear in the graph.",
                }
            },
            "required": ["answers"],
        },
    },
}


class ToolInputError(Exception):
    """A bad tool call; the message goes back to the model."""


@dataclass(frozen=True)
class ToolOutput:
    text: str
    cost_usd: float = 0.0
    decision_calls: int = 0
    detail: dict[str, JsonValue] = field(default_factory=dict[str, JsonValue])


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    run: Callable[[Args], Awaitable[ToolOutput]]

    def spec(self) -> ToolSpec:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass(frozen=True)
class Turn:
    """One model call: the assistant message (OpenAI format) and its usage."""

    message: dict[str, Any]
    input_tokens: int
    output_tokens: int
    cost_usd: float | None
    latency_s: float


class Chat:
    """Tool-calling chat completions via LiteLLM, with a request-rate limiter."""

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None,
        max_tokens: int = 1024,
        max_rpm: float | None = None,
        timeout_s: float = 90.0,
    ) -> None:
        self._litellm: Any = import_litellm()
        self.model = model
        self._api_key = api_key
        self._max_tokens = max_tokens
        self._timeout_s = timeout_s
        self._interval = 0.0 if max_rpm is None else 60.0 / max_rpm
        self._next_slot = 0.0
        self._lock = asyncio.Lock()

    async def _wait(self) -> None:
        async with self._lock:
            now = time.monotonic()
            wait = self._next_slot - now
            self._next_slot = max(now, self._next_slot) + self._interval
        if wait > 0:
            await asyncio.sleep(wait)

    async def __call__(self, messages: list[dict[str, Any]], tools: list[ToolSpec]) -> Turn:
        extra: dict[str, Any] = {}
        if self.model.startswith("openrouter/"):
            extra["usage"] = {"include": True}
        for attempt in range(6):
            await self._wait()
            started = time.perf_counter()
            try:
                response: Any = await self._litellm.acompletion(
                    model=self.model,
                    messages=messages,
                    tools=tools,
                    tool_choice="required",
                    temperature=0.0,
                    max_tokens=self._max_tokens,
                    timeout=self._timeout_s,
                    num_retries=2,
                    api_key=self._api_key,
                    **extra,
                )
                break
            except self._litellm.RateLimitError:
                logger.warning("%s rate-limited (attempt %d)", self.model, attempt + 1)
                await asyncio.sleep(10.0 * (attempt + 1))
            except Exception as error:  # noqa: BLE001 - provider errors carry the request
                raise LLMError(f"LLM call failed: {redact(str(error), [self._api_key])}") from None
        else:
            raise LLMError("LLM rate-limited 6 times")
        latency = time.perf_counter() - started
        usage: Any = getattr(response, "usage", None)
        cost = getattr(usage, "cost", None)
        message: Any = response.choices[0].message
        return Turn(
            message=_assistant_message(message),
            input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            cost_usd=float(cost) if isinstance(cost, int | float) else None,
            latency_s=latency,
        )


def _assistant_message(message: Any) -> dict[str, Any]:
    raw: list[Any] = getattr(message, "tool_calls", None) or []
    calls = [
        {
            "id": call.id,
            "type": "function",
            "function": {"name": call.function.name, "arguments": call.function.arguments},
        }
        for call in raw
    ]
    out: dict[str, Any] = {"role": "assistant", "content": message.content or ""}
    if calls:
        out["tool_calls"] = calls
    return out


type ChatFn = Callable[[list[dict[str, Any]], list[ToolSpec]], Awaitable[Turn]]


class AgentSystem:
    """A ``QASystem``: the agent loop over ``make_tools(question)``."""

    def __init__(
        self,
        name: str,
        chat: ChatFn,
        make_tools: Callable[[EvalQuestion], Sequence[Tool]],
        *,
        max_turns: int = 10,
        model: str = "",
    ) -> None:
        self._name = name
        self._chat = chat
        self._make_tools = make_tools
        self._max_turns = max_turns
        self._model = model

    @property
    def name(self) -> str:
        return self._name

    def describe(self) -> dict[str, JsonValue]:
        return {"system": "agent", "llm": self._model, "max_turns": self._max_turns}

    async def answer(self, question: EvalQuestion) -> SystemAnswer:  # noqa: PLR0915 - one loop, kept whole
        wall = time.perf_counter()
        tools = {t.name: t for t in self._make_tools(question)}
        specs = [t.spec() for t in tools.values()]
        topic = ", ".join(question.start or ()) or "(none given)"
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT.format(max_turns=self._max_turns)},
            {"role": "user", "content": f"Question: {question.question}\nTopic entities: {topic}"},
        ]
        counts: Counter[str] = Counter()
        transcript: list[JsonValue] = []
        in_tok = out_tok = decisions = 0
        agent_cost: float | None = 0.0
        tool_cost = latency = 0.0
        first_walk: float | None = None
        seen: list[str] = []
        final: list[str] | None = None
        status: Literal["ok", "error"] = "ok"
        error: str | None = None
        turns = 0
        try:
            while final is None and turns < self._max_turns:
                turns += 1
                last = turns == self._max_turns
                turn = await self._chat(messages, [ANSWER_SPEC] if last else [*specs, ANSWER_SPEC])
                in_tok += turn.input_tokens
                out_tok += turn.output_tokens
                latency += turn.latency_s
                agent_cost = (
                    None if agent_cost is None or turn.cost_usd is None
                    else agent_cost + turn.cost_usd
                )  # fmt: skip
                messages.append(turn.message)
                calls: list[dict[str, Any]] = turn.message.get("tool_calls") or []
                if not calls:  # ignored tool_choice: nudge once per turn
                    messages.append({"role": "user", "content": "Call a tool or `answer`."})
                    continue
                for call in calls:
                    name = str(call["function"]["name"])
                    counts[name] += 1
                    try:
                        args: Args = json.loads(str(call["function"]["arguments"] or "{}"))
                    except json.JSONDecodeError:
                        args = {}
                    if name == "answer":
                        raw: Any = args.get("answers", [])
                        items = cast("list[Any]", raw) if isinstance(raw, list) else [raw]
                        final = [str(a) for a in items if str(a).strip()]
                        transcript.append({"tool": "answer", "args": list[JsonValue](final)})
                        break
                    started = time.perf_counter()
                    tool = tools.get(name)
                    if tool is None:
                        output = ToolOutput(f"error: no tool named {name!r}")
                    else:
                        try:
                            output = await tool.run(args)
                        except ToolInputError as bad:
                            output = ToolOutput(f"error: {bad}")
                    latency += time.perf_counter() - started
                    tool_cost += output.cost_usd
                    decisions += output.decision_calls
                    walk_conf = output.detail.get("confidence")
                    if first_walk is None and name == "walk" and isinstance(walk_conf, float):
                        first_walk = walk_conf
                    seen.append(output.text.casefold())
                    transcript.append(
                        {"tool": name, "args": json.dumps(args)[:200], "out": output.text[:300]}
                    )
                    messages.append(
                        {"role": "tool", "tool_call_id": call["id"], "content": output.text}
                    )
                # A turn that called `answer` alongside other tools: those still ran and
                # need a reply only if the loop continued, and it does not.
        except LLMError as failure:
            status, error = "error", str(failure)
        answers = tuple(dict.fromkeys(final or ()))
        observed = "\n".join(seen)
        grounded = sum(a.casefold() in observed for a in answers)
        return SystemAnswer(
            answers=answers,
            answer_set=answers,
            status=status,
            error=error,
            latency_s=latency,
            decision_calls=decisions,
            llm_calls=turns,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=None if agent_cost is None else agent_cost + tool_cost,
            start=question.start,
            detail={
                "turns": turns,
                "answered": final is not None,
                "tool_calls": dict(counts),
                "agent_cost_usd": agent_cost,
                "tool_cost_usd": tool_cost,
                "wall_s": time.perf_counter() - wall,
                "first_walk_confidence": first_walk,
                "grounded": grounded / len(answers) if answers else None,
                "transcript": transcript,
            },
        )


# -- tools -----------------------------------------------------------------------------


def _object(properties: dict[str, Any], required: Sequence[str]) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": list(required)}


async def _resolve(store: SQLiteStore, node: object) -> str:
    """A node id from an id or a name (case-insensitive)."""
    if not isinstance(node, str) or not node.strip():
        raise ToolInputError("give a node name or id")
    if await store.node_labels([node]):
        return node
    matches = await store.find_nodes(name=node.strip(), limit=1)
    if not matches:
        raise ToolInputError(f"no node named {node!r}")
    return matches[0].id


def graph_tools(store: SQLiteStore) -> list[Tool]:
    """``relations`` (which relations a node has, with counts) and ``neighbors`` (the
    nodes one relation leads to): the low-level graph interface."""

    async def relations(args: Args) -> ToolOutput:
        node = await _resolve(store, args.get("node"))
        adjacent = await store.adjacency(node, direction="both")
        counts = Counter((a.relation, a.direction) for a in adjacent)
        lines = [f"{node}: {len(adjacent)} edges, {len(counts)} relation/direction pairs"]
        lines += [f"{r} ({d}): {n}" for (r, d), n in counts.most_common(MAX_LISTED)]
        if len(counts) > MAX_LISTED:
            lines.append(f"... {len(counts) - MAX_LISTED} more (rarest omitted)")
        return ToolOutput("\n".join(lines))

    async def neighbors(args: Args) -> ToolOutput:
        node = await _resolve(store, args.get("node"))
        relation = args.get("relation")
        direction = args.get("direction", "both")
        if direction not in {"out", "in", "both"}:
            raise ToolInputError("direction must be out, in, or both")
        adjacent = [
            a
            for a in await store.adjacency(node, direction=direction)
            if relation in {None, ""} or a.relation == relation
        ]
        labels = await store.node_labels([a.other for a in adjacent[:MAX_LISTED]])
        lines = [f"{node}: {len(adjacent)} edges" + (f" via {relation}" if relation else "")]
        for a in adjacent[:MAX_LISTED]:
            label = labels.get(a.other)
            name = a.other if label is None else label.name
            lines.append(f"{a.relation} ({a.direction}) -> {name}")
        if len(adjacent) > MAX_LISTED:
            lines.append(f"... {len(adjacent) - MAX_LISTED} more")
        return ToolOutput("\n".join(lines))

    node_param = {"type": "string", "description": "Node name or id."}
    return [
        Tool(
            "relations",
            "List the relations around a node (relation, direction, edge count), most "
            "frequent first. `out`: node --relation--> other; `in`: other --relation--> node.",
            _object({"node": node_param}, ["node"]),
            relations,
        ),
        Tool(
            "neighbors",
            f"The nodes a node is connected to, optionally only via one relation (up to "
            f"{MAX_LISTED}). Record nodes (m.…) can be explored the same way.",
            _object(
                {
                    "node": node_param,
                    "relation": {"type": "string", "description": "Only this relation."},
                    "direction": {"type": "string", "enum": ["out", "in", "both"]},
                },
                ["node"],
            ),
            neighbors,
        ),
    ]


def walk_tool(store: SQLiteStore, traverser: Traverser) -> Tool:
    """graphwalk's walk, as the MCP ``walk`` tool presents it, from given start nodes."""

    async def walk(args: Args) -> ToolOutput:
        question = args.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ToolInputError("give the question")
        raw: Any = args.get("start") or []
        names = cast("list[Any]", raw) if isinstance(raw, list) else [raw]
        start = tuple([await _resolve(store, n) for n in names])
        if not start:
            raise ToolInputError("give at least one start node")
        result = await traverser.traverse(question, start)
        cost = result.cost_usd or 0.0
        calls = result.trace.totals.decision_calls
        if not result.answers:
            return ToolOutput("no answer found", cost, calls, {"confidence": 0.0})
        lines = [f"confidence: {result.confidence:.2f}"]
        for answer in result.answers[:3]:
            path = " / ".join(
                h.relation + ("" if h.direction == "out" else "^-1") for h in answer.path
            )
            shown = ", ".join(answer.names[:10])
            more = f" (+{len(answer.names) - 10} more)" if len(answer.names) > 10 else ""  # noqa: PLR2004
            lines.append(f"- [{answer.confidence:.2f}] {path}: {shown}{more}")
        return ToolOutput("\n".join(lines), cost, calls, {"confidence": result.confidence})

    return Tool(
        "walk",
        "Answer from the graph in one call: walks relations from the start nodes, choosing "
        "each hop by the question, and returns the entities reached with each path and a "
        "confidence (0-1). Fast and cheap. It does not apply constraints (earliest, "
        "largest, both A and B): filter its answers yourself. Below 0.9 confidence, verify "
        "with the other tools before relying on it.",
        _object(
            {
                "question": {"type": "string", "description": "The question."},
                "start": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Start node names or ids (the topic entities).",
                },
            },
            ["question", "start"],
        ),
        walk,
    )


def triple_passages(triples: Sequence[tuple[str, str, str]], *, cvt_cap: int = 30) -> list[str]:
    """The facts as text: one ``head relation tail`` sentence per triple between named
    entities, and one passage per record (CVT) node gathering its triples, the record
    shown as ``(record)``."""
    passages: list[str] = []
    records: dict[str, list[str]] = {}

    def show(node: str) -> str:
        return "(record)" if rog.node_type(node) == rog.CVT_TYPE else node

    for h, r, t in triples:
        sentence = f"{show(h)} {r} {show(t)}"
        cvts = [n for n in (h, t) if rog.node_type(n) == rog.CVT_TYPE]
        if not cvts:
            passages.append(sentence)
        for node in cvts:
            records.setdefault(node, []).append(sentence)
    passages += ["; ".join(facts[:cvt_cap]) for facts in records.values()]
    return passages


def search_tool(index: DocIndex, embedder: Embedder, *, k: int = 10) -> Tool:
    """Dense search over :func:`triple_passages`: the retrieval tool of classic RAG."""

    async def search(args: Args) -> ToolOutput:
        query = args.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ToolInputError("give a query")
        hits = index.top_k((await embedder.embed([query]))[0], k)
        return ToolOutput("\n".join(f"- {index.texts[i]}" for i in hits))

    return Tool(
        "search",
        f"Search the knowledge graph's facts (as text) and return the {k} most relevant. "
        "Facts are `subject relation object`; facts about a record node are grouped, "
        "the record shown as (record). Search as many times as you need.",
        _object({"query": {"type": "string", "description": "What to look for."}}, ["query"]),
        search,
    )
