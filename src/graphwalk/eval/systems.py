"""QA systems under evaluation: graphwalk traversal and a vector-RAG baseline.

Both read the *same* knowledge: the dataset's graph. graphwalk walks it; the baseline
retrieves verbalized per-entity documents built from it and asks an LLM. That isolates
the question the eval exists to answer: walking with cheap decisions vs. retrieve+read.
"""

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from pydantic import JsonValue

from graphwalk.core.model import Node, NodeId
from graphwalk.embeddings.base import Embedder, Vectors
from graphwalk.eval.types import EvalQuestion, SystemAnswer
from graphwalk.llm.base import LLMBackend, LLMError, Message
from graphwalk.stores.base import GraphStore
from graphwalk.traversal import TraversalConfig, Traverser
from graphwalk.traversal.entry import EntryLink, EntryResolver

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
    ) -> None:
        if index.embedder_model != embedder.model_id:
            msg = f"index built with {index.embedder_model}, querying with {embedder.model_id}"
            raise ValueError(msg)
        self._index = index
        self._embedder = embedder
        self._llm = llm
        self._k = k
        self._name = name

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
            "document": "one per entity: name, type, <= 40 incident triples",
        }

    def prompt(self, question: str, docs: Sequence[str]) -> list[Message]:
        facts = "\n".join(f"- {d}" for d in docs)
        user = f"Facts:\n{facts}\n\nQuestion: {question}\n\n{RAG_INSTRUCTIONS}"
        return [Message(role="system", content=RAG_SYSTEM), Message(role="user", content=user)]

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
