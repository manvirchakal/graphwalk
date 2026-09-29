"""Routing: decide which existing node, if any, an extracted entity refers to.

For each mention, :class:`NodeIndex` proposes up to N existing nodes (exact name or
alias matches first, then shared name words and embedding similarity). One decision
call per chunk then asks, for every mention with candidates, a choice over
``{c1..cN, NEW}``. A choice whose probability is below ``route_threshold`` is escalated
to the LLM, if one is configured. Mentions with no candidates are NEW without a call.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from pydantic import JsonValue

from graphwalk.core.model import Node, NodeId
from graphwalk.decisions.base import (
    ChoiceQuestion,
    DecisionBackend,
    DecisionBackendError,
    DecisionRequest,
    JSONContent,
)
from graphwalk.embeddings.base import Embedder
from graphwalk.ingest.extraction import GENERIC_TYPE, ExtractedEntity, name_key
from graphwalk.ingest.spend import Spend
from graphwalk.llm.base import LLMBackend, LLMError, Message
from graphwalk.stores.base import GraphStore
from graphwalk.traversal.batching import split_questions
from graphwalk.traversal.entry import STOPWORDS
from graphwalk.traversal.prompts import edge_text, node_text

type RoutingMode = Literal["jev", "exact"]
type RouteMethod = Literal["no_candidates", "exact", "decision", "llm", "fallback"]

NEW_LABEL = "NEW"
ROUTE_TASK = (
    "The instructions give an entity mentioned in the passage in the state. Which existing "
    "knowledge-graph node is that same real-world entity? Names can be spelled, "
    "abbreviated, or titled differently (e.g. 'Count X of Y' and 'X, Count of Y'). A node "
    "with the same or an equivalent name and a compatible type IS the same entity, unless "
    "the passage or the node's facts contradict it (different dates, places, relatives, or "
    "roles that cannot both be true). A node need not repeat the passage's facts. Choose "
    "NEW only if no node is that entity."
)
NEW_DESCRIPTION = "None of these nodes is this entity: add it as a new node."
MAX_PASSAGE_CHARS = 4000

_WORD = re.compile(r"\w+")


def content_words(text: str) -> frozenset[str]:
    return frozenset(w for w in _WORD.findall(text.casefold()) if w not in STOPWORDS)


def mention_text(entity: ExtractedEntity) -> str:
    """Embedding text for a mention, shaped like :func:`node_text`."""
    text = f"{entity.name} ({entity.type})"
    return f"{text}: {entity.description}" if entity.description else text


class NodeIndex:
    """Incremental candidate index over the store's nodes (names, aliases, embeddings)."""

    def __init__(self, embedder: Embedder | None = None, *, min_similarity: float = 0.75) -> None:
        self._embedder = embedder
        self._min_similarity = min_similarity
        self._keys: dict[str, set[NodeId]] = {}
        self._postings: dict[str, set[NodeId]] = {}
        self._node_keys: dict[NodeId, list[str]] = {}
        self._node_words: dict[NodeId, list[frozenset[str]]] = {}
        self._texts: dict[NodeId, str] = {}
        self._vectors: dict[NodeId, np.ndarray] = {}
        self._pending: set[NodeId] = set()
        self._matrix: tuple[list[NodeId], np.ndarray] | None = None

    @classmethod
    async def build(
        cls, store: GraphStore, embedder: Embedder | None = None, *, min_similarity: float = 0.75
    ) -> "NodeIndex":
        index = cls(embedder, min_similarity=min_similarity)
        async for node in store.iter_nodes():
            index.add(node)
        return index

    def __len__(self) -> int:
        return len(self._node_keys)

    def remove(self, node_id: NodeId) -> None:
        for key in self._node_keys.pop(node_id, []):
            self._keys[key].discard(node_id)
        for words in self._node_words.pop(node_id, []):
            for word in words:
                self._postings[word].discard(node_id)
        self._texts.pop(node_id, None)
        self._vectors.pop(node_id, None)
        self._pending.discard(node_id)
        self._matrix = None

    def add(self, node: Node) -> None:
        """Index ``node``, replacing any earlier version of it."""
        self.remove(node.id)
        names = list(dict.fromkeys((node.name, *node.aliases)))
        keys = list(dict.fromkeys(name_key(n) for n in names))
        words = [content_words(n) for n in names]
        self._node_keys[node.id] = keys
        self._node_words[node.id] = words
        for key in keys:
            self._keys.setdefault(key, set()).add(node.id)
        for word in frozenset[str]().union(*words):
            self._postings.setdefault(word, set()).add(node.id)
        self._texts[node.id] = node_text(node)
        if self._embedder is not None:
            self._pending.add(node.id)

    async def _embeddings(self) -> tuple[list[NodeId], np.ndarray] | None:
        if self._embedder is None:
            return None
        if self._pending:
            ids = sorted(self._pending)
            vectors = await self._embedder.embed([self._texts[i] for i in ids])
            for node_id, vector in zip(ids, vectors, strict=True):
                self._vectors[node_id] = vector
            self._pending.clear()
            self._matrix = None
        if self._matrix is None and self._vectors:
            ids = sorted(self._vectors)
            self._matrix = (ids, np.stack([self._vectors[i] for i in ids]))
        return self._matrix

    def exact(self, name: str) -> list[NodeId]:
        return sorted(self._keys.get(name_key(name), ()))

    async def candidates(self, entity: ExtractedEntity, limit: int) -> list[NodeId]:
        """Exact name/alias matches, then the best of name-word overlap and similarity."""
        exact = self.exact(entity.name)
        out = exact[:limit]
        if len(out) >= limit:
            return out
        scores: dict[NodeId, float] = {}
        words = content_words(entity.name)
        pool: set[NodeId] = set()
        for word in words:
            pool |= self._postings.get(word, set())
        for node_id in pool.difference(out):
            best = max(
                (len(words & w) / len(words | w) for w in self._node_words[node_id] if w),
                default=0.0,
            )
            scores[node_id] = best
        matrix = await self._embeddings()
        if matrix is not None and self._embedder is not None:
            ids, vectors = matrix
            query = (await self._embedder.embed([mention_text(entity)]))[0]
            sims = vectors @ query
            top = np.argsort(-sims)[: limit * 2]
            for i in top.tolist():
                node_id: NodeId = ids[int(i)]
                similarity = float(sims[int(i)])
                if similarity >= self._min_similarity and node_id not in out:
                    scores[node_id] = max(scores.get(node_id, 0.0), similarity)
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        out.extend(node_id for node_id, _ in ranked[: limit - len(out)])
        return out


@dataclass
class Route:
    mention: ExtractedEntity
    candidates: tuple[NodeId, ...]
    target: NodeId | None
    """The existing node the mention refers to, or ``None`` for a new node."""
    probability: float
    """Probability of the chosen option (1.0 when there was nothing to choose)."""
    method: RouteMethod
    probabilities: dict[str, float] = field(default_factory=dict[str, float])
    """Candidate node id (or ``NEW``) -> probability, when a decision was made."""
    error: str | None = None


@dataclass
class RouteBatch:
    routes: list[Route]
    spend: Spend
    """Decision calls (and nothing else)."""
    escalation: Spend = field(default_factory=Spend)
    """LLM escalation calls, kept apart so their cost can be watched."""


class Router:
    def __init__(
        self,
        store: GraphStore,
        index: NodeIndex,
        decider: DecisionBackend | None,
        *,
        llm: LLMBackend | None = None,
        mode: RoutingMode = "jev",
        max_candidates: int = 5,
        route_threshold: float = 0.6,
        max_relations: int = 8,
    ) -> None:
        if mode == "jev" and decider is None:
            msg = "routing mode 'jev' needs a decision backend"
            raise ValueError(msg)
        self._store = store
        self._index = index
        self._decider = decider
        self._llm = llm
        self._mode = mode
        limit = max_candidates if decider is None else min(max_candidates, decider.max_options - 1)
        self._max = max(1, limit)
        self._threshold = route_threshold
        self._max_relations = max_relations

    async def _card(self, node_id: NodeId) -> dict[str, JsonValue]:
        node = await self._store.get_node(node_id)
        if node is None:  # pragma: no cover - the index mirrors the store
            return {"name": node_id}
        card: dict[str, JsonValue] = {"name": node.name, "type": node.type}
        if node.aliases:
            card["also_known_as"] = list[JsonValue](node.aliases[:5])
        if node.summary:
            card["summary"] = node.summary
        relations: list[JsonValue] = []
        for neighbor in await self._store.neighbors(node_id, direction="both"):
            relations.append(
                edge_text(node.name, neighbor.edge.type, neighbor.direction, neighbor.node.name)
            )
            if len(relations) >= self._max_relations:
                break
        if relations:
            card["relations"] = relations
        return card

    async def route(self, mentions: Sequence[ExtractedEntity], passage: str) -> RouteBatch:
        spend = Spend()
        routes: list[Route] = []
        for mention in mentions:
            candidates = tuple(await self._index.candidates(mention, self._max))
            routes.append(Route(mention, candidates, None, 1.0, "no_candidates"))
        pending = [r for r in routes if r.candidates]
        if self._mode == "exact":
            for route in pending:
                exact = self._index.exact(route.mention.name)
                route.target = exact[0] if exact else None
                route.method = "exact"
            return RouteBatch(routes, spend)
        if not pending:
            return RouteBatch(routes, spend)
        await self._decide(pending, passage, spend)
        escalation = Spend()
        if self._llm is not None:
            for route in pending:
                if route.method == "decision" and route.probability < self._threshold:
                    await self._escalate(route, passage, escalation)
        return RouteBatch(routes, spend, escalation)

    async def _question(self, key: str, route: Route) -> ChoiceQuestion:
        options: dict[str, JSONContent | None] = {}
        for i, node_id in enumerate(route.candidates):
            options[f"c{i + 1}"] = await self._card(node_id)
        options[NEW_LABEL] = NEW_DESCRIPTION
        mention: dict[str, JsonValue] = {"name": route.mention.name}
        if route.mention.type != GENERIC_TYPE:
            mention["type"] = route.mention.type
        if route.mention.description:
            mention["description"] = route.mention.description
        return ChoiceQuestion(
            key=key, instructions={"task": ROUTE_TASK, "mention": mention}, options=options
        )

    async def _decide(self, pending: list[Route], passage: str, spend: Spend) -> None:
        assert self._decider is not None  # noqa: S101 - checked in __init__
        state: dict[str, JsonValue] = {"passage": passage[:MAX_PASSAGE_CHARS]}
        questions = [await self._question(f"m{i}", r) for i, r in enumerate(pending)]
        by_key = {q.key: r for q, r in zip(questions, pending, strict=True)}
        for group in split_questions(
            state,
            questions,
            max_request_tokens=64_000,
            max_state_plus_question_tokens=32_000,
            safety_margin=0.2,
        ):
            spend.decision_calls += 1
            try:
                response = await self._decider.decide(
                    DecisionRequest(state=state, questions=tuple(group))
                )
            except DecisionBackendError as error:
                for question in group:
                    # Without a decision, merge only on an exact name match.
                    route = by_key[question.key]
                    exact = self._index.exact(route.mention.name)
                    route.target = exact[0] if exact else None
                    route.method, route.error = "fallback", str(error)
                    route.probability = 0.0
                continue
            spend.input_tokens += response.usage.input_tokens
            spend.output_tokens += response.usage.output_tokens
            spend.add_cost(response.usage.cost_usd)
            for question in group:
                route = by_key[question.key]
                result = response.results[question.key]
                labels = [f"c{i + 1}" for i in range(len(route.candidates))]
                route.probabilities = {
                    node_id: result.probabilities[label]
                    for label, node_id in zip(labels, route.candidates, strict=True)
                }
                route.probabilities[NEW_LABEL] = result.probabilities[NEW_LABEL]
                route.method = "decision"
                route.probability = result.probabilities[result.top]
                route.target = (
                    None if result.top == NEW_LABEL else route.candidates[labels.index(result.top)]
                )

    async def _escalate(self, route: Route, passage: str, spend: Spend) -> None:
        assert self._llm is not None  # noqa: S101 - checked by the caller
        lines = [f"Passage:\n{passage[:MAX_PASSAGE_CHARS]}", "", f"Mention: {route.mention.name}"]
        if route.mention.type != GENERIC_TYPE:
            lines.append(f"Mention type: {route.mention.type}")
        if route.mention.description:
            lines.append(f"Mention description: {route.mention.description}")
        lines += ["", "Existing nodes:"]
        labels = [f"c{i + 1}" for i in range(len(route.candidates))]
        for label, node_id in zip(labels, route.candidates, strict=True):
            lines.append(f"{label}: {await self._card(node_id)}")
        lines.append(f"{NEW_LABEL}: {NEW_DESCRIPTION}")
        messages = [
            Message(
                role="system",
                content="You resolve entity mentions against a knowledge graph. "
                + ROUTE_TASK.replace("instructions give", "user gives").replace("in the state", "")
                + " Reply with the option label only.",
            ),
            Message(role="user", content="\n".join(lines)),
        ]
        spend.llm_calls += 1
        try:
            reply = await self._llm.complete(messages)
        except LLMError as error:
            route.error = f"escalation failed: {error}"
            return
        spend.input_tokens += reply.input_tokens
        spend.output_tokens += reply.output_tokens
        spend.add_cost(reply.cost_usd)
        label = parse_label(reply.text, (*labels, NEW_LABEL))
        if label is None:
            route.error = f"unusable escalation reply: {reply.text[:100]!r}"
            return
        route.method = "llm"
        route.target = None if label == NEW_LABEL else route.candidates[labels.index(label)]
        route.probability = route.probabilities.get(label_of(route), route.probability)


def parse_label(text: str, labels: Sequence[str]) -> str | None:
    """The reply if it is exactly a label, else the last label it mentions."""
    stripped = text.strip().strip("`*.'\"").strip()
    if stripped in labels:
        return stripped
    found = [m for m in re.findall(r"\b(c\d+|NEW)\b", text) if m in labels]
    return found[-1] if found else None


def label_of(route: Route) -> str:
    """The key of the chosen option in ``route.probabilities``."""
    return NEW_LABEL if route.target is None else route.target
