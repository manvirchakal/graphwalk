"""Entry resolution: which node(s) a query starts from.

Kept separate from traversal so evals can report linking errors apart from walking
errors. Datasets that name the topic entity (MetaQA) should pass start nodes directly.

Two resolvers:

* :class:`NameEntryResolver`: free, deterministic. Exact whole-word name mentions; can
  also rank *fuzzy* candidates by IDF-weighted word overlap with the query.
* :class:`ChoiceEntryResolver`: takes the name resolver's candidates and spends one
  decision call to pick the entity the query is about (the framework's bet applied to
  linking: a cheap constrained choice instead of heuristics).
"""

import math
import re
import time
from collections import defaultdict
from collections.abc import Collection
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from pydantic import JsonValue

from graphwalk.core.model import NodeId
from graphwalk.decisions.base import (
    ChoiceQuestion,
    DecisionBackend,
    DecisionBackendError,
    DecisionRequest,
    JSONContent,
)
from graphwalk.embeddings.base import Embedder
from graphwalk.stores.base import GraphStore
from graphwalk.traversal.prefilter import EmbeddingCache, rank_by_similarity
from graphwalk.traversal.prompts import edge_text, node_card, node_text, state_for

_WORD = re.compile(r"\w+")

_FUNCTION_WORDS = (
    "a about an and are as at be by did do does for from had has have he her his how i "
    "in is it its me my of on or our s she that the their them they this to was we were "
    "what when where which who whom whose why will with you your"
)
STOPWORDS = frozenset(_FUNCTION_WORDS.split())
"""English function words. A mention made only of these (a film titled "Who...") is
ignored, and they never count as fuzzy-match evidence."""


def _words(text: str) -> tuple[str, ...]:
    return tuple(_WORD.findall(text.casefold()))


@dataclass(frozen=True)
class EntryLink:
    """Entry nodes for one query, plus what finding them cost."""

    nodes: tuple[NodeId, ...]
    decision_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None
    latency_s: float = 0.0
    detail: dict[str, JsonValue] = field(default_factory=dict[str, JsonValue])


@runtime_checkable
class EntryResolver(Protocol):
    async def resolve(self, query: str) -> list[NodeId]: ...

    async def link(self, query: str) -> EntryLink: ...


@dataclass(frozen=True)
class Candidate:
    node_id: NodeId
    score: float
    """Exact mentions: the mention's word count + 1000. Fuzzy: see ``candidates``."""
    exact: bool


@dataclass
class _Index:
    names: dict[tuple[str, ...], list[NodeId]]
    texts: list[tuple[NodeId, str]]
    postings: dict[str, set[NodeId]]
    idf: dict[str, float]
    variants: dict[NodeId, list[frozenset[str]]]
    """Content words of each name and alias of a node."""


class NameEntryResolver:
    """Finds node names and aliases mentioned in the query (whole words, ignoring case).

    Longest mentions win; overlapping shorter ones are dropped, as are mentions made only
    of ``stopwords``. Every remaining mention contributes all nodes with that name, in
    order of appearance. If nothing matches and an embedder is given, falls back to the
    single most similar node. With ``best_only``, returns just the top of
    :meth:`candidates` instead (one node, fuzzy matches allowed).

    The index is built on first use; call :meth:`refresh` after changing the graph.
    """

    def __init__(
        self,
        store: GraphStore,
        *,
        embedder: Embedder | None = None,
        min_name_chars: int = 3,
        max_name_words: int = 12,
        stopwords: Collection[str] = STOPWORDS,
        best_only: bool = False,
    ) -> None:
        self._store = store
        self._best_only = best_only
        self._cache = None if embedder is None else EmbeddingCache(embedder)
        self._min_chars = min_name_chars
        self._max_words = max_name_words
        self._stop = frozenset(stopwords)
        self._index: _Index | None = None

    def refresh(self) -> None:
        self._index = None

    async def _build(self) -> _Index:
        names: dict[tuple[str, ...], list[NodeId]] = defaultdict(list)
        postings: dict[str, set[NodeId]] = defaultdict(set)
        variants: dict[NodeId, list[frozenset[str]]] = defaultdict(list)
        texts: list[tuple[NodeId, str]] = []
        async for node in self._store.iter_nodes():
            texts.append((node.id, node_text(node)))
            for name in (node.name, *node.aliases):
                words = _words(name)
                if len(name.strip()) < self._min_chars or not words:
                    continue
                if len(words) <= self._max_words and node.id not in names[words]:
                    names[words].append(node.id)
                content = frozenset(w for w in words if w not in self._stop)
                if content:
                    variants[node.id].append(content)
                    for word in content:
                        postings[word].add(node.id)
        n = max(1, len(texts))
        idf = {word: math.log(1 + n / len(ids)) for word, ids in postings.items()}
        self._index = _Index(dict(names), texts, dict(postings), idf, dict(variants))
        return self._index

    async def _ensure(self) -> _Index:
        return self._index if self._index is not None else await self._build()

    def _mentions(self, index: _Index, words: tuple[str, ...]) -> list[tuple[int, int]]:
        spans: list[tuple[int, int]] = []
        for i in range(len(words)):
            for j in range(min(len(words), i + self._max_words), i, -1):
                if words[i:j] in index.names and not all(w in self._stop for w in words[i:j]):
                    spans.append((i, j))
                    break
        spans.sort(key=lambda s: (-(s[1] - s[0]), s[0]))
        taken: list[tuple[int, int]] = []
        for start, end in spans:
            if all(end <= s or start >= e for s, e in taken):
                taken.append((start, end))
        return sorted(taken)

    async def resolve(self, query: str) -> list[NodeId]:
        if self._best_only:
            return [c.node_id for c in await self.candidates(query, 1)]
        index = await self._ensure()
        words = _words(query)
        found: list[NodeId] = []
        for start, end in self._mentions(index, words):
            for node_id in index.names[words[start:end]]:
                if node_id not in found:
                    found.append(node_id)
        if found or self._cache is None or not index.texts:
            return found
        ranked = await rank_by_similarity(self._cache, query, [t for _, t in index.texts])
        return [index.texts[ranked[0][0]][0]]

    async def link(self, query: str) -> EntryLink:
        return EntryLink(nodes=tuple(await self.resolve(query)))

    async def candidates(self, query: str, limit: int = 8) -> list[Candidate]:
        """Exact mentions (longest first), then fuzzy matches, at most ``limit``.

        A fuzzy match scores ``m * m / t`` for its best name variant, where ``m`` is the
        summed IDF of the name's content words found in the query and ``t`` that of all
        its content words: rare shared words count, unmatched name words penalize. So
        "Reginald Ii, Count Of Bar" finds "Reginald II of Bar", and "Antonio Flores"
        finds "Antonio González Flores".
        """
        index = await self._ensure()
        words = _words(query)
        out: list[Candidate] = []
        seen: set[NodeId] = set()
        spans = sorted(self._mentions(index, words), key=lambda s: (s[0] - s[1], s[0]))
        for start, end in spans:
            for node_id in index.names[words[start:end]]:
                if node_id not in seen:
                    seen.add(node_id)
                    out.append(Candidate(node_id, 1000.0 + end - start, exact=True))
        present = {w for w in words if w not in self._stop and w in index.postings}
        pool: set[NodeId] = set()
        for word in present:
            pool |= index.postings[word]
        scored: list[tuple[float, NodeId]] = []
        for node_id in pool - seen:
            best = 0.0
            for variant in index.variants[node_id]:
                total = math.fsum(index.idf[w] for w in variant)
                matched = math.fsum(index.idf[w] for w in variant if w in present)
                best = max(best, matched * matched / total)
            scored.append((best, node_id))
        scored.sort(key=lambda s: (-s[0], s[1]))
        out.extend(Candidate(node_id, score, exact=False) for score, node_id in scored)
        return out[:limit]


ENTRY_KEY = "entry"
ENTRY_TASK = (
    "Which of these knowledge-graph nodes is the entity the query in the state is about, "
    "the node to start from to look up the answer? It is named in the query, possibly "
    "spelled or abbreviated a little differently. If several nodes share that name, choose "
    "the one whose relations include what the query asks about first."
)


class ChoiceEntryResolver:
    """Name-resolver candidates, then one decision call picks the entry node.

    No call when there is at most one candidate. If the call fails, falls back to the
    top candidate and records the error in ``EntryLink.detail``.
    """

    def __init__(
        self,
        store: GraphStore,
        decider: DecisionBackend,
        *,
        names: NameEntryResolver | None = None,
        max_candidates: int = 8,
        max_relations: int = 10,
    ) -> None:
        self._store = store
        self._decider = decider
        self._names = names or NameEntryResolver(store)
        self._max = max(1, min(max_candidates, decider.max_options))
        self._max_relations = max_relations

    @property
    def decider(self) -> DecisionBackend:
        return self._decider

    async def resolve(self, query: str) -> list[NodeId]:
        return list((await self.link(query)).nodes)

    async def _option(self, node_id: NodeId) -> dict[str, JsonValue]:
        node = await self._store.get_node(node_id)
        if node is None:  # pragma: no cover - candidates come from the store
            return {"name": node_id}
        relations: list[str] = []
        for neighbor in await self._store.neighbors(node_id, direction="both"):
            text = edge_text(node.name, neighbor.edge.type, neighbor.direction, "…")
            if text not in relations:
                relations.append(text)
            if len(relations) >= self._max_relations:
                break
        card = node_card(node, include_summary=True)
        card["relations"] = list[JsonValue](relations)
        return card

    async def link(self, query: str) -> EntryLink:
        candidates = await self._names.candidates(query, self._max)
        ids = [c.node_id for c in candidates]
        detail: dict[str, JsonValue] = {"candidates": list[JsonValue](ids)}
        if len(ids) <= 1:
            return EntryLink(nodes=tuple(ids), detail=detail)
        labels = [f"c{i + 1}" for i in range(len(ids))]
        options: dict[str, JSONContent | None] = {}
        for label, node_id in zip(labels, ids, strict=True):
            options[label] = await self._option(node_id)
        question = ChoiceQuestion(key=ENTRY_KEY, instructions=ENTRY_TASK, options=options)
        started = time.perf_counter()
        try:
            response = await self._decider.decide(
                DecisionRequest(state=state_for(query), questions=(question,))
            )
        except DecisionBackendError as error:
            detail["error"] = str(error)
            return EntryLink(
                nodes=(ids[0],),
                decision_calls=1,
                latency_s=time.perf_counter() - started,
                detail=detail,
            )
        result = response.results[ENTRY_KEY]
        chosen = ids[labels.index(result.top)]
        detail["probabilities"] = {
            node_id: result.probabilities[label] for label, node_id in zip(labels, ids, strict=True)
        }
        return EntryLink(
            nodes=(chosen,),
            decision_calls=1,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cost_usd=response.usage.cost_usd,
            latency_s=response.latency_s,
            detail=detail,
        )
