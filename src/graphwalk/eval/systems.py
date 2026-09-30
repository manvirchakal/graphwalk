"""QA systems under evaluation: graphwalk traversal and a vector-RAG baseline.

Both read the *same* knowledge: the dataset's graph. graphwalk walks it; the baseline
retrieves verbalized per-entity documents built from it and asks an LLM. That isolates
the question the eval exists to answer: walking with cheap decisions vs. retrieve+read.
"""

import asyncio
import math
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from pydantic import JsonValue

from graphwalk.core.model import Node, NodeId
from graphwalk.embeddings.base import Embedder, Vectors
from graphwalk.eval.metrics import normalize
from graphwalk.eval.types import EvalQuestion, SystemAnswer
from graphwalk.llm.base import LLMBackend, LLMError, Message
from graphwalk.stores.base import GraphStore
from graphwalk.traversal import TraversalConfig, TraversalResult, Traverser
from graphwalk.traversal.entry import EntryLink, EntryResolver, NameEntryResolver

type Linking = Literal["given", "gold", "resolve", "resolve-best", "choice"]
"""Where entry nodes come from: the question's ``start`` (MetaQA), its ``gold_start``
(traversal-only on 2Wiki), or an :class:`EntryResolver` (end-to-end): ``resolve`` =
every exact name mention, ``resolve-best`` = the top name candidate (fuzzy allowed),
``choice`` = a decision call picks among the name candidates."""
RESOLVED: tuple[Linking, ...] = ("resolve", "resolve-best", "choice")


class GraphwalkSystem:
    def __init__(
        self,
        store: GraphStore,
        traverser: Traverser,
        *,
        name: str = "graphwalk",
        linking: Linking = "given",
        resolver: EntryResolver | None = None,
    ) -> None:
        if linking in RESOLVED and resolver is None:
            msg = f"linking={linking!r} needs a resolver"
            raise ValueError(msg)
        self._store = store
        self._traverser = traverser
        self._name = name
        self._linking: Linking = linking
        self._resolver = resolver

    @property
    def name(self) -> str:
        return self._name

    @property
    def config(self) -> TraversalConfig:
        return self._traverser.config

    def describe(self) -> dict[str, JsonValue]:
        embedder = self._traverser.embedder
        return {
            "system": "graphwalk",
            "linking": self._linking,
            "decision_model": self._traverser.decider.model_id,
            "embedder": None if embedder is None else embedder.model_id,
            "traversal": self._traverser.config.model_dump(mode="json"),
        }

    async def _link(self, question: EvalQuestion) -> EntryLink:
        if self._linking == "given":
            return EntryLink(nodes=question.start or ())
        if self._linking == "gold":
            return EntryLink(nodes=question.gold_start or ())
        assert self._resolver is not None  # checked in __init__  # noqa: S101
        return await self._resolver.link(question.question)

    async def answer(self, question: EvalQuestion) -> SystemAnswer:
        started = time.perf_counter()
        link = await self._link(question)
        existing = await self._store.get_nodes(link.nodes)
        start = tuple(n for n in link.nodes if n in existing)
        if not start:
            return SystemAnswer(
                answers=(),
                answer_set=(),
                status="no_entry",
                latency_s=time.perf_counter() - started,
                start=(),
                decision_calls=link.decision_calls,
                input_tokens=link.input_tokens,
                output_tokens=link.output_tokens,
                cost_usd=link.cost_usd if link.decision_calls else None,
                detail={"linking": link.detail},
            )
        result = await self._traverser.traverse(question.question, start)
        ranked: list[str] = []
        for answer in result.answers:
            for name in answer.names:
                if name not in ranked:
                    ranked.append(name)
        best = result.best
        totals = result.trace.totals
        detail: dict[str, JsonValue] = {
            "depth_reached": totals.depth_reached,
            "questions": totals.questions,
            "decision_latency_s": totals.decision_latency_s,
        }
        if link.decision_calls or link.detail:
            detail["linking"] = {**link.detail, "decision_calls": link.decision_calls}
        if best is not None:
            detail.update(
                {
                    "path": [f"{h.relation}:{h.direction}" for h in best.path],
                    "terminated_by": best.terminated_by,
                    "score": best.score,
                    "min_confidence": best.min_confidence,
                    "n_answers": len(best.names),
                }
            )
        return SystemAnswer(
            answers=tuple(ranked),
            answer_set=() if best is None else best.names,
            status=result.status,
            error=result.error or result.abort_reason,
            latency_s=time.perf_counter() - started,
            decision_calls=totals.decision_calls + link.decision_calls,
            input_tokens=totals.input_tokens + link.input_tokens,
            output_tokens=totals.output_tokens + link.output_tokens,
            cost_usd=_add_costs(totals.cost_usd, link),
            start=start,
            detail=detail,
        )


def _add_costs(traversal: float | None, link: EntryLink) -> float | None:
    if not link.decision_calls:
        return traversal
    if traversal is None or link.cost_usd is None:
        return None
    return traversal + link.cost_usd


# -- vector RAG ------------------------------------------------------------------------


def entity_document(node: Node, facts: Sequence[str]) -> str:
    return f"{node.name} ({node.type}). " + "; ".join(facts)


async def entity_documents(store: GraphStore, *, max_facts: int = 40) -> list[tuple[NodeId, str]]:
    """One document per node: its name, type, and up to ``max_facts`` incident triples
    as ``subject relation object`` sentences (outgoing first, in store order)."""
    docs: list[tuple[NodeId, str]] = []
    async for node in store.iter_nodes():
        facts: list[str] = []
        for neighbor in await store.neighbors(node.id, direction="both"):
            if len(facts) >= max_facts:
                break
            other = neighbor.node.name
            if neighbor.direction == "out":
                facts.append(f"{node.name} {neighbor.edge.type} {other}")
            else:
                facts.append(f"{other} {neighbor.edge.type} {node.name}")
        docs.append((node.id, entity_document(node, facts)))
    return docs


@dataclass(frozen=True)
class DocIndex:
    ids: tuple[NodeId, ...]
    texts: tuple[str, ...]
    vectors: Vectors
    embedder_model: str

    @classmethod
    async def build(
        cls, docs: Sequence[tuple[NodeId, str]], embedder: Embedder, *, batch: int = 2048
    ) -> "DocIndex":
        parts: list[Vectors] = []
        texts = [text for _, text in docs]
        for i in range(0, len(texts), batch):
            parts.append(await embedder.embed(texts[i : i + batch]))
        vectors = np.concatenate(parts) if parts else np.zeros((0, 0), dtype=np.float32)
        return cls(
            ids=tuple(i for i, _ in docs),
            texts=tuple(texts),
            vectors=vectors.astype(np.float32, copy=False),
            embedder_model=embedder.model_id,
        )

    def top_k(self, query_vector: Vectors, k: int) -> list[int]:
        sims = self.vectors @ query_vector
        k = min(k, len(self.ids))
        if k <= 0:
            return []
        candidates = np.argpartition(-sims, k - 1)[:k]
        return sorted((int(i) for i in candidates), key=lambda i: (-float(sims[i]), i))


RAG_SYSTEM = "You answer questions using only the given facts from a knowledge graph."
RAG_INSTRUCTIONS = (
    "Answer with the answer entity names exactly as written in the facts. If there are "
    "several answers, list all of them separated by ' | '. Output only the answer(s), "
    "nothing else. If the facts do not contain the answer, output: unknown"
)


@dataclass(frozen=True)
class RAGPrompts:
    """The reader's wording. :data:`GRAPH_PROMPTS` (the default) reads facts verbalized
    from a graph and answers with entity names; :data:`TEXT_PROMPTS` reads original
    passages and answers with a short span (a name, date, number, or yes/no)."""

    system: str
    instructions: str
    iter_system: str
    iter_step: str
    iter_last: str
    context_label: str
    document: str
    """How the documents were made, recorded in ``describe()``."""


def parse_answers(text: str) -> list[str]:
    answers: list[str] = []
    for part in text.replace("\n", "|").split("|"):
        cleaned = part.strip().strip("-*•").strip()
        if cleaned and cleaned.casefold() != "unknown" and cleaned not in answers:
            answers.append(cleaned)
    return answers


class VectorRAGSystem:
    """Embed the question, retrieve the top-``k`` entity documents, ask the LLM."""

    def __init__(
        self,
        index: DocIndex,
        embedder: Embedder,
        llm: LLMBackend,
        *,
        k: int = 5,
        name: str = "vector-rag",
        prompts: RAGPrompts | None = None,
    ) -> None:
        if index.embedder_model != embedder.model_id:
            msg = f"index built with {index.embedder_model}, querying with {embedder.model_id}"
            raise ValueError(msg)
        self._index = index
        self._embedder = embedder
        self._llm = llm
        self._k = k
        self._name = name
        self._prompts = prompts or GRAPH_PROMPTS

    @property
    def name(self) -> str:
        return self._name

    def describe(self) -> dict[str, JsonValue]:
        return {
            "system": "vector-rag",
            "k": self._k,
            "llm": self._llm.model_id,
            "embedder": self._embedder.model_id,
            "documents": len(self._index.ids),
            "document": self._prompts.document,
        }

    def prompt(self, question: str, docs: Sequence[str]) -> list[Message]:
        p = self._prompts
        facts = "\n".join(f"- {d}" for d in docs)
        user = f"{p.context_label}:\n{facts}\n\nQuestion: {question}\n\n{p.instructions}"
        return [Message(role="system", content=p.system), Message(role="user", content=user)]

    async def answer(self, question: EvalQuestion) -> SystemAnswer:
        started = time.perf_counter()
        query_vector = (await self._embedder.embed([question.question]))[0]
        hits = self._index.top_k(query_vector, self._k)
        docs = [self._index.texts[i] for i in hits]
        retrieval_s = time.perf_counter() - started
        try:
            result = await self._llm.complete(self.prompt(question.question, docs))
        except LLMError as error:
            return SystemAnswer(
                answers=(),
                answer_set=(),
                status="error",
                error=str(error),
                latency_s=time.perf_counter() - started,
                llm_calls=1,
                embed_calls=1,
            )
        answers = tuple(parse_answers(result.text))
        return SystemAnswer(
            answers=answers,
            answer_set=answers,
            # Retrieval plus the LLM call itself; excludes client-side rate-limit waits.
            latency_s=retrieval_s + result.latency_s,
            llm_calls=1,
            embed_calls=1,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            cost_usd=result.cost_usd,
            detail={
                "retrieved": [self._index.ids[i] for i in hits],
                "raw": result.text,
                "llm_latency_s": result.latency_s,
            },
        )


# -- iterative (multi-step) RAG ----------------------------------------------------------

ITER_SYSTEM = (
    "You answer questions using facts from a knowledge graph. You can search the graph "
    "for more entities' facts, one step at a time."
)
ITER_STEP = (
    "Step {step} of {steps}. First write one short line starting with 'Thought:'. Then, if "
    "the facts answer the question, write one line 'ANSWER: <answer entity names exactly "
    "as written in the facts; several separated by ' | '>'. Otherwise write one line "
    "'SEARCH: <names of the entities whose facts you need next; several separated by "
    "' | '>'. Search for entity names, not questions."
)
ITER_LAST = (
    "Step {step} of {steps}, the last one. First write one short line starting with "
    "'Thought:'. Then write one line 'ANSWER: <answer entity names exactly as written in "
    "the facts; several separated by ' | '>', or 'ANSWER: unknown'."
)
_ACTION = ("ANSWER:", "SEARCH:")
ENTITY_DOCUMENT = "one per entity: name, type, <= 40 incident triples"

GRAPH_PROMPTS = RAGPrompts(
    system=RAG_SYSTEM,
    instructions=RAG_INSTRUCTIONS,
    iter_system=ITER_SYSTEM,
    iter_step=ITER_STEP,
    iter_last=ITER_LAST,
    context_label="Facts",
    document=ENTITY_DOCUMENT,
)
_SHORT = "a short answer only: the entity name, date, number, or yes/no, written as in the passages"
TEXT_PROMPTS = RAGPrompts(
    system="You answer questions using only the given passages.",
    instructions=(
        f"Answer with {_SHORT}. Output only the answer, nothing else. If the passages do "
        "not contain the answer, output: unknown"
    ),
    iter_system=(
        "You answer questions using passages from a document collection. You can search "
        "the collection for more passages, one step at a time."
    ),
    iter_step=(
        "Step {step} of {steps}. First write one short line starting with 'Thought:'. "
        "Then, if the passages answer the question, write one line 'ANSWER: <"
        + _SHORT
        + ">'. Otherwise write one line 'SEARCH: <titles or names of the entities whose "
        "passages you need next; several separated by ' | '>'. Search for entity names, "
        "not questions."
    ),
    iter_last=(
        "Step {step} of {steps}, the last one. First write one short line starting with "
        "'Thought:'. Then write one line 'ANSWER: <" + _SHORT + ">', or 'ANSWER: unknown'."
    ),
    context_label="Passages",
    document="one per paragraph: title and text",
)
_LIST = (
    "every item the question asks for, one per line; when it asks for something about "
    "each item, write 'Name: value'. Use names and values as written in the passages. "
    "If some items are missing from the passages, list the ones you can"
)
LIST_PROMPTS = RAGPrompts(
    system=TEXT_PROMPTS.system,
    instructions=(
        f"Answer with {_LIST}. Output only the answer. If the passages contain none of it, "
        "output: unknown"
    ),
    iter_system=TEXT_PROMPTS.iter_system,
    iter_step=(
        "Step {step} of {steps}. First write one short line starting with 'Thought:'. "
        "Then, if the passages answer the whole question, write one line 'ANSWER: <"
        "every item the question asks for, separated by ' | ', each as 'Name: value' when "
        "it asks for something about each item>'. Otherwise write one line 'SEARCH: <titles "
        "or names of the entities whose passages you need next; several separated by ' | "
        "'>'. Search for entity names, not questions."
    ),
    iter_last=(
        "Step {step} of {steps}, the last one. First write one short line starting with "
        "'Thought:'. Then write one line 'ANSWER: <every item the question asks for that "
        "the passages give, separated by ' | ', each as 'Name: value' when it asks for "
        "something about each item>', or 'ANSWER: unknown'."
    ),
    context_label="Passages",
    document="one per page: title and text",
)
"""For questions whose answer is a list or an item -> value map (FanOutQA)."""


def parse_step(text: str) -> tuple[str, list[str]]:
    """``("answer" | "search", items)`` from the last ANSWER:/SEARCH: line. Replies
    without either are read as a bare answer (minus any ``Thought:`` lines)."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in reversed(lines):
        head = line[:7].upper()
        if head in _ACTION:
            items = parse_answers(line[7:])
            return ("answer" if head == "ANSWER:" else "search"), items
    body = "\n".join(line for line in lines if not line.lower().startswith("thought:"))
    return "answer", parse_answers(body)


class IterativeRAGSystem:
    """Multi-step RAG: retrieve for the question, then let the LLM either answer or name
    entities to look up next (exact name match first, else dense retrieval), up to
    ``max_steps`` LLM calls. Same documents, embedder, and reader as
    :class:`VectorRAGSystem`, so the only difference is iteration."""

    def __init__(
        self,
        index: DocIndex,
        embedder: Embedder,
        llm: LLMBackend,
        *,
        names: dict[str, list[int]],
        k: int = 5,
        k_search: int = 2,
        max_steps: int = 5,
        max_searches: int = 8,
        max_docs: int = 40,
        name: str = "iter-rag",
        prompts: RAGPrompts | None = None,
        first: Callable[[str], Awaitable[Sequence[str]]] | None = None,
    ) -> None:
        """``first`` replaces the initial retrieval: given the question, it returns the
        document ids to start from (e.g. graphwalk's hybrid ``locate``), in rank order;
        ids not in the index are skipped."""
        if index.embedder_model != embedder.model_id:
            msg = f"index built with {index.embedder_model}, querying with {embedder.model_id}"
            raise ValueError(msg)
        self._index = index
        self._embedder = embedder
        self._llm = llm
        self._names = names
        self._k = k
        self._k_search = k_search
        self._max_steps = max_steps
        self._max_searches = max_searches
        self._max_docs = max_docs
        self._name = name
        self._prompts = prompts or GRAPH_PROMPTS
        self._first = first
        self._positions = {doc_id: i for i, doc_id in enumerate(index.ids)}

    async def _initial(self, question: str) -> list[int]:
        if self._first is None:
            return self._index.top_k((await self._embedder.embed([question]))[0], self._k)
        ids = await self._first(question)
        hits = [self._positions[d] for d in ids if d in self._positions]
        return list(dict.fromkeys(hits))[: self._k]

    @classmethod
    def name_index(cls, index: DocIndex, names: dict[NodeId, list[str]]) -> dict[str, list[int]]:
        """Casefolded name/alias -> document positions, for exact-name searches."""
        out: dict[str, list[int]] = {}
        for position, node_id in enumerate(index.ids):
            for name in names.get(node_id, []):
                out.setdefault(name.casefold(), []).append(position)
        return out

    @property
    def name(self) -> str:
        return self._name

    def describe(self) -> dict[str, JsonValue]:
        return {
            "system": "iter-rag",
            "k": self._k,
            "k_search": self._k_search,
            "first": "custom" if self._first is not None else "question embedding",
            "title_search": bool(self._names),
            "max_steps": self._max_steps,
            "max_searches": self._max_searches,
            "max_docs": self._max_docs,
            "llm": self._llm.model_id,
            "embedder": self._embedder.model_id,
            "documents": len(self._index.ids),
            "document": self._prompts.document,
        }

    def prompt(
        self,
        question: str,
        docs: Sequence[str],
        searched: Sequence[str],
        step: int,
        *,
        last: bool = False,
    ) -> list[Message]:
        p = self._prompts
        facts = "\n".join(f"- {d}" for d in docs) or "(none)"
        history = "; ".join(searched) if searched else "(none yet)"
        template = p.iter_last if last or step == self._max_steps else p.iter_step
        user = (
            f"{p.context_label}:\n{facts}\n\nSearched so far: {history}\n\n"
            f"Question: {question}\n\n" + template.format(step=step, steps=self._max_steps)
        )
        return [Message(role="system", content=p.iter_system), Message(role="user", content=user)]

    async def _search(self, query: str) -> list[int]:
        exact = self._names.get(query.strip().strip("\"'").casefold())
        if exact:
            return exact[:3]
        vector = (await self._embedder.embed([query]))[0]
        return self._index.top_k(vector, self._k_search)

    async def replay(self, question: str, steps: Sequence[Mapping[str, JsonValue]]) -> list[str]:
        """The documents a recorded run retrieved, in order: the question's own hits,
        then each step's searches. Retrieval is deterministic, so this reproduces
        ``detail["retrieved"]`` from a run's ``detail["steps"]`` without LLM calls."""
        seen = list(dict.fromkeys(await self._initial(question)))
        for step in steps:
            queries = step.get("search")
            if not isinstance(queries, list):
                continue
            for query in queries:
                for i in await self._search(str(query)):
                    if i not in seen:
                        seen.append(i)
        return [self._index.ids[i] for i in seen]

    async def answer(self, question: EvalQuestion) -> SystemAnswer:
        started = time.perf_counter()
        embed_calls = 1
        hits = await self._initial(question.question)
        retrieval_s = time.perf_counter() - started
        seen = list(dict.fromkeys(hits))
        searched: list[str] = []
        steps: list[JsonValue] = []
        calls, tokens_in, tokens_out, llm_s = 0, 0, 0, 0.0
        costs: list[float | None] = []
        answers: list[str] = []
        raw = ""
        last = False
        for step in range(1, self._max_steps + 1):
            last = last or step == self._max_steps
            # Over the cap: keep the question's own hits and the most recent searches.
            shown = seen[: self._k] + seen[self._k :][-(self._max_docs - self._k) :]
            docs = [self._index.texts[i] for i in shown]
            try:
                result = await self._llm.complete(
                    self.prompt(question.question, docs, searched, step, last=last)
                )
            except LLMError as error:
                return SystemAnswer(
                    answers=(),
                    answer_set=(),
                    status="error",
                    error=str(error),
                    latency_s=retrieval_s + llm_s,
                    llm_calls=calls + 1,
                    embed_calls=embed_calls,
                    input_tokens=tokens_in,
                    output_tokens=tokens_out,
                    cost_usd=None,
                    detail={"steps": steps},
                )
            calls += 1
            tokens_in += result.input_tokens
            tokens_out += result.output_tokens
            llm_s += result.latency_s
            costs.append(result.cost_usd)
            raw = result.text
            action, items = parse_step(result.text)
            if action == "answer" or last:
                answers = items if action == "answer" else []
                steps.append({"answer": list[JsonValue](items)} if action == "answer" else {})
                break
            new = [q for q in items if q not in searched][: self._max_searches]
            added = 0
            t0 = time.perf_counter()
            for query in new:
                searched.append(query)
                found = await self._search(query)
                embed_calls += query.strip().strip("\"'").casefold() not in self._names
                for i in found:
                    if i not in seen:
                        seen.append(i)
                        added += 1
            retrieval_s += time.perf_counter() - t0
            steps.append({"search": list[JsonValue](new), "added": added})
            # Nothing new was found: searching again won't help, so answer next.
            last = added == 0
        cost = None if not costs or any(c is None for c in costs) else sum(c or 0 for c in costs)
        return SystemAnswer(
            answers=tuple(answers),
            answer_set=tuple(answers),
            # Retrieval plus the LLM calls themselves; excludes client-side rate-limit waits.
            latency_s=retrieval_s + llm_s,
            llm_calls=calls,
            embed_calls=embed_calls,
            input_tokens=tokens_in,
            output_tokens=tokens_out,
            cost_usd=cost,
            detail={
                "steps": steps,
                "raw": raw,
                "docs": len(seen),
                "retrieved": [self._index.ids[i] for i in seen],
            },
        )


# -- graphwalk + LLM reader --------------------------------------------------------------

READER_SYSTEM = "You answer questions using only the given facts from a knowledge graph."
READER_INSTRUCTIONS = (
    "Answer with a short answer only: the entity name, date, number, or yes/no, written as "
    "in the facts. Output only the answer, nothing else. If the facts do not contain the "
    "answer, output: unknown"
)


def node_line(node: Node) -> str:
    """``Name (type): summary [key: value; ...]``."""
    text = f"{node.name} ({node.type})"
    if node.summary:
        text += f": {node.summary}"
    if node.attributes:
        attrs = "; ".join(f"{k}: {v}" for k, v in node.attributes.items())
        text += f" [{attrs}]"
    return text


async def node_facts(store: GraphStore, node: Node, limit: int) -> list[str]:
    facts: list[str] = []
    for neighbor in await store.neighbors(node.id, direction="both"):
        if len(facts) >= limit:
            break
        other = neighbor.node.name
        if neighbor.direction == "out":
            facts.append(f"{node.name} {neighbor.edge.type} {other}")
        else:
            facts.append(f"{other} {neighbor.edge.type} {node.name}")
    return facts


class GraphReaderSystem:
    """graphwalk, then an LLM reads what the walk reached.

    Entry nodes are the linked node plus other entities the question names exactly (up
    to ``max_entries``), so comparison questions get both sides. The walk runs from
    each entry separately. The reader sees each walk's path and, for every entry and
    walked node (at most ``max_nodes``), its summary, attributes, and up to
    ``max_facts`` incident facts. It answers in free text, so dates, yes/no, and
    comparisons are answerable, which graph-only walking cannot do.

    With ``documents`` (provenance source id -> ``(title, text)``), the reader reads
    source text instead of extracted facts: the walk only chooses *which* paragraphs,
    so facts that extraction dropped or garbled are still there. Up to ``max_docs``
    paragraphs, in this order: each entry node's own paragraph (title = its name), the
    paragraphs that support each walked edge, then walked nodes' own paragraphs.
    """

    def __init__(
        self,
        store: GraphStore,
        traverser: Traverser,
        llm: LLMBackend,
        resolver: EntryResolver,
        *,
        names: NameEntryResolver | None = None,
        max_entries: int = 2,
        max_nodes: int = 20,
        max_facts: int = 12,
        max_path_names: int = 5,
        documents: Mapping[str, tuple[str, str]] | None = None,
        max_docs: int = 5,
        instructions: str | None = None,
        max_answers: int | None = 1,
        name: str = "graphwalk-reader",
    ) -> None:
        """``instructions`` replaces the default short-answer instructions;
        ``max_answers=None`` keeps every answer line (for list answers)."""
        self._store = store
        self._traverser = traverser
        self._llm = llm
        self._resolver = resolver
        self._names = names
        self._max_entries = max_entries
        self._max_nodes = max_nodes
        self._max_facts = max_facts
        self._max_path_names = max_path_names
        self._documents = documents
        self._max_docs = max_docs
        self._instructions = instructions
        self._max_answers = max_answers
        self._own: dict[str, str] = {}
        """Normalized title -> provenance source id of the paragraph with that title."""
        for key, (title, _text) in (documents or {}).items():
            self._own.setdefault(normalize(title), key)
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def describe(self) -> dict[str, JsonValue]:
        embedder = self._traverser.embedder
        return {
            "system": "graphwalk-reader",
            "decision_model": self._traverser.decider.model_id,
            "llm": self._llm.model_id,
            "embedder": None if embedder is None else embedder.model_id,
            "max_entries": self._max_entries,
            "max_nodes": self._max_nodes,
            "max_facts": self._max_facts,
            "context": "facts" if self._documents is None else "source paragraphs",
            "max_docs": self._max_docs,
            "traversal": self._traverser.config.model_dump(mode="json"),
        }

    async def _entries(self, question: str) -> tuple[list[NodeId], EntryLink]:
        link = await self._resolver.link(question)
        entries = list(link.nodes)
        if self._names is not None:
            for candidate in await self._names.candidates(question, 8):
                if len(entries) >= self._max_entries:
                    break
                if candidate.exact and candidate.node_id not in entries:
                    entries.append(candidate.node_id)
        existing = await self._store.get_nodes(entries)
        return [e for e in entries if e in existing][: self._max_entries], link

    def _walk_line(self, result: TraversalResult, names: dict[NodeId, str]) -> str | None:
        best = result.best
        if best is None:
            return None
        parts = [" / ".join(names.get(s, s) for s in best.start)]
        for hop in best.path:
            shown = [names.get(t, t) for t in hop.targets[: self._max_path_names]]
            more = len(hop.targets) - len(shown)
            targets = ", ".join(shown) + (f" (+{more} more)" if more > 0 else "")
            arrow = f"--{hop.relation}-->" if hop.direction == "out" else f"<--{hop.relation}--"
            parts.append(f"{arrow} {targets}")
        return " ".join(parts)

    def prompt(self, question: str, walks: Sequence[str], nodes: Sequence[str]) -> list[Message]:
        walk_text = "\n".join(f"- {w}" for w in walks) or "(none)"
        node_text_ = "\n".join(nodes) or "(none)"
        if self._documents is None:
            body = f"Facts about the entities on those walks:\n{node_text_}"
            system, instructions = READER_SYSTEM, READER_INSTRUCTIONS
        else:
            body = f"Passages about the entities on those walks:\n{node_text_}"
            system, instructions = TEXT_PROMPTS.system, TEXT_PROMPTS.instructions
        if self._instructions is not None:
            instructions = self._instructions
            if self._documents is None:
                instructions = instructions.replace("passages", "facts")
        user = (
            f"Walks through a knowledge graph toward the answer:\n{walk_text}\n\n"
            f"{body}\n\nQuestion: {question}\n\n{instructions}"
        )
        return [Message(role="system", content=system), Message(role="user", content=user)]

    async def _walked_edge_sources(self, results: Sequence[TraversalResult]) -> list[str]:
        """Provenance source ids of the edges each best walk followed, in walk order."""
        sources: list[str] = []
        for result in results:
            best = result.best
            if best is None:
                continue
            frontier = list(best.start)
            for hop in best.path:
                targets = set(hop.targets[: self._max_path_names])
                for node_id in frontier:
                    for neighbor in await self._store.neighbors(
                        node_id, direction=hop.direction, edge_types=(hop.relation,)
                    ):
                        if neighbor.node.id in targets:
                            sources.extend(p.source_id for p in neighbor.edge.provenance)
                frontier = list(targets)
        return sources

    async def _passages(
        self, entries: Sequence[NodeId], order: Sequence[NodeId], results: Sequence[TraversalResult]
    ) -> list[str]:
        assert self._documents is not None  # noqa: S101 - source mode only
        found = await self._store.get_nodes(list(order))

        def own(node_id: NodeId) -> list[str]:
            node = found.get(node_id)
            if node is None:
                return []
            key = self._own.get(normalize(node.name))
            if key is not None:
                return [key]
            return [p.source_id for p in node.provenance[:1]]

        ranked: list[str] = []
        for node_id in entries:
            ranked += own(node_id)
        ranked += await self._walked_edge_sources(results)
        for node_id in order:
            ranked += own(node_id)
        keys = [k for k in dict.fromkeys(ranked) if k in self._documents][: self._max_docs]
        return [f"{self._documents[k][0]}: {self._documents[k][1]}" for k in keys]

    async def answer(self, question: EvalQuestion) -> SystemAnswer:
        started = time.perf_counter()
        entries, link = await self._entries(question.question)
        if not entries:
            return SystemAnswer(
                answers=(),
                answer_set=(),
                status="no_entry",
                latency_s=time.perf_counter() - started,
                start=(),
                decision_calls=link.decision_calls,
                input_tokens=link.input_tokens,
                output_tokens=link.output_tokens,
                cost_usd=link.cost_usd if link.decision_calls else None,
                detail={"linking": link.detail},
            )
        results = await asyncio.gather(
            *(self._traverser.traverse(question.question, (e,)) for e in entries)
        )
        order: list[NodeId] = list(entries)
        for result in results:
            best = result.best
            if best is None:
                continue
            for hop in best.path:
                order.extend(hop.targets[: self._max_path_names])
        order = list(dict.fromkeys(order))[: self._max_nodes]
        found = await self._store.get_nodes(order)
        names = {node_id: node.name for node_id, node in found.items()}
        walks = [w for r in results if (w := self._walk_line(r, names)) is not None]
        lines: list[str] = []
        if self._documents is not None:
            lines = [f"- {p}" for p in await self._passages(entries, order, results)]
        else:
            for node_id in order:
                node = found.get(node_id)
                if node is None:
                    continue
                facts = await node_facts(self._store, node, self._max_facts)
                lines.append(node_line(node) + "".join(f"\n  - {f}" for f in facts))
        decision_calls = link.decision_calls + sum(r.trace.totals.decision_calls for r in results)
        tokens_in = link.input_tokens + sum(r.trace.totals.input_tokens for r in results)
        tokens_out = link.output_tokens + sum(r.trace.totals.output_tokens for r in results)
        costs: list[float | None] = [r.trace.totals.cost_usd for r in results]
        if link.decision_calls:
            costs.append(link.cost_usd)
        walk_s = time.perf_counter() - started
        detail: dict[str, JsonValue] = {
            "entries": list[JsonValue](entries),
            "walks": list[JsonValue](walks),
            "nodes": len(lines),
            "walk_s": walk_s,
        }
        try:
            result = await self._llm.complete(self.prompt(question.question, walks, lines))
        except LLMError as error:
            return SystemAnswer(
                answers=(),
                answer_set=(),
                status="error",
                error=str(error),
                latency_s=walk_s,
                decision_calls=decision_calls,
                llm_calls=1,
                input_tokens=tokens_in,
                output_tokens=tokens_out,
                start=tuple(entries),
                detail=detail,
            )
        costs.append(result.cost_usd)
        answers = tuple(parse_answers(result.text))
        if self._max_answers is not None:
            answers = answers[: self._max_answers]
        detail["raw"] = result.text
        return SystemAnswer(
            answers=answers,
            answer_set=answers,
            # Walks plus the reader call itself; excludes client-side rate-limit waits.
            latency_s=walk_s + result.latency_s,
            decision_calls=decision_calls,
            llm_calls=1,
            input_tokens=tokens_in + result.input_tokens,
            output_tokens=tokens_out + result.output_tokens,
            cost_usd=None if any(c is None for c in costs) else math.fsum(c or 0 for c in costs),
            start=tuple(entries),
            detail=detail,
        )
