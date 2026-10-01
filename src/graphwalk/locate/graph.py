"""Graph ``locate``: walk from the entities a query names, return the text spans behind
what the walk touched.

The graph is used as an index over the text, not as the answer: every walked edge and
reached node carries provenance with character offsets, and those spans are what is
returned. Ranking, best first:

1. per answer (the traversal's ranking; answers of different entry nodes interleave):
   the spans of each walked edge in hop order, then the spans of the answer nodes;
2. after all answers, the entry nodes' own spans.

Ties break by provenance confidence. Spans overlapping a better-ranked span of the same
document are dropped, and each node contributes at most ``max_spans_per_node`` spans
(hub nodes are mentioned everywhere).
"""

import asyncio
import math
from collections.abc import Sequence
from dataclasses import dataclass, field

from graphwalk.core.errors import DocumentNotFoundError
from graphwalk.core.model import Node, NodeId, Provenance
from graphwalk.locate.documents import DocumentSource
from graphwalk.locate.model import Location
from graphwalk.stores.base import GraphStore
from graphwalk.traversal.entry import EntryLink, EntryResolver, NameEntryResolver
from graphwalk.traversal.escalation import Walker
from graphwalk.traversal.prompts import edge_text
from graphwalk.traversal.trace import Answer, TraversalResult

SNIPPET_CHARS = 240


@dataclass
class LocateResult:
    """Locations plus how they were found (for debugging and traces)."""

    locations: list[Location]
    entries: tuple[NodeId, ...] = ()
    link: EntryLink | None = None
    walks: list[TraversalResult] = field(default_factory=list[TraversalResult])

    @property
    def decision_calls(self) -> int:
        link = 0 if self.link is None else self.link.decision_calls
        return link + sum(w.trace.totals.decision_calls for w in self.walks)


@dataclass(frozen=True)
class _Candidate:
    rank: tuple[int, ...]
    provenance: Provenance
    score: float
    path: tuple[str, ...]
    element: str
    element_id: str


class GraphLocator:
    def __init__(
        self,
        store: GraphStore,
        traverser: Walker,
        *,
        resolver: EntryResolver | None = None,
        names: NameEntryResolver | None = None,
        documents: DocumentSource | None = None,
        max_entries: int = 2,
        max_answers: int = 3,
        max_spans_per_node: int = 3,
    ) -> None:
        """``resolver`` links the query to entry nodes (default: name matching);
        ``names`` adds other entities the query names exactly, up to ``max_entries``
        (so comparison questions start from both sides). ``documents`` fills in titles,
        hashes, and snippets."""
        self._store = store
        self._traverser = traverser
        self._names = names or NameEntryResolver(store, embedder=traverser.embedder)
        self._resolver = resolver or self._names
        self._documents = documents
        self._max_entries = max_entries
        self._max_answers = max_answers
        self._max_spans_per_node = max_spans_per_node

    def refresh(self) -> None:
        """Rebuild the name index after the graph changed."""
        self._names.refresh()
        refresh = getattr(self._resolver, "refresh", None)
        if refresh is not None and self._resolver is not self._names:
            refresh()

    async def entries(self, query: str) -> tuple[list[NodeId], EntryLink]:
        link = await self._resolver.link(query)
        entries = list(link.nodes)
        for candidate in await self._names.candidates(query, 8):
            if len(entries) >= self._max_entries:
                break
            if candidate.exact and candidate.node_id not in entries:
                entries.append(candidate.node_id)
        existing = await self._store.get_nodes(entries)
        return [e for e in entries if e in existing][: self._max_entries], link

    async def walk(self, query: str) -> tuple[list[NodeId], EntryLink, list[TraversalResult]]:
        """The entry nodes, how they were linked, and one walk per entry node."""
        entries, link = await self.entries(query)
        walks = list(
            await asyncio.gather(*(self._traverser.traverse(query, (e,)) for e in entries))
        )
        return entries, link, walks

    async def locate(self, query: str, k: int = 5) -> LocateResult:
        if k < 1:
            msg = f"k must be positive, got {k}"
            raise ValueError(msg)
        entries, link, walks = await self.walk(query)
        if not entries:
            return LocateResult(locations=[], link=link)
        nodes = await self._store.get_nodes(
            list({n for w in walks for a in w.answers[: self._max_answers] for n in a.node_ids})
            + entries
        )
        candidates: list[_Candidate] = []
        for e, walk in enumerate(walks):
            for a, answer in enumerate(walk.answers[: self._max_answers]):
                candidates += await self._answer_spans(answer, (a, e), nodes)
        for e, node_id in enumerate(entries):
            node = nodes[node_id]
            candidates += self._node_spans(node, (self._max_answers, 2, e), 1.0, (node.name,))
        locations = await self._select(candidates, k)
        return LocateResult(locations=locations, entries=tuple(entries), link=link, walks=walks)

    async def _answer_spans(
        self, answer: Answer, rank: tuple[int, int], nodes: dict[NodeId, Node]
    ) -> list[_Candidate]:
        a, e = rank
        probability = math.exp(answer.score)
        out: list[_Candidate] = []
        frontier = list(answer.start)
        path: list[str] = []
        for h, hop in enumerate(answer.path):
            targets = set(hop.targets)
            for node_id in frontier:
                for neighbor in await self._store.neighbors(
                    node_id, direction=hop.direction, edge_types=(hop.relation,)
                ):
                    if neighbor.node.id not in targets:
                        continue
                    here = await self._name(node_id, nodes)
                    step = edge_text(here, hop.relation, hop.direction, neighbor.node.name)
                    for p in neighbor.edge.provenance:
                        out.append(
                            _Candidate(
                                rank=(a, 0, h, e),
                                provenance=p,
                                score=probability * p.confidence,
                                path=(*path, step),
                                element="edge",
                                element_id=neighbor.edge.id,
                            )
                        )
            if hop.targets:
                first = await self._name(hop.targets[0], nodes)
                start = " / ".join([await self._name(n, nodes) for n in frontier])
                path.append(edge_text(start, hop.relation, hop.direction, first))
            frontier = list(hop.targets)
        for node_id in answer.node_ids:
            node = nodes.get(node_id)
            if node is not None:
                out += self._node_spans(node, (a, 1, len(answer.path), e), probability, tuple(path))
        return out

    async def _name(self, node_id: NodeId, nodes: dict[NodeId, Node]) -> str:
        node = nodes.get(node_id)
        if node is None:
            node = await self._store.get_node(node_id)
            if node is None:
                return node_id
            nodes[node_id] = node
        return node.name

    def _node_spans(
        self, node: Node, rank: tuple[int, ...], probability: float, path: tuple[str, ...]
    ) -> list[_Candidate]:
        best = sorted(node.provenance, key=lambda p: -p.confidence)[: self._max_spans_per_node]
        return [
            _Candidate(
                rank=rank,
                provenance=p,
                score=probability * p.confidence,
                path=path,
                element="node",
                element_id=node.id,
            )
            for p in best
        ]

    async def _select(self, candidates: Sequence[_Candidate], k: int) -> list[Location]:
        order = sorted(
            range(len(candidates)),
            key=lambda i: (candidates[i].rank, -candidates[i].provenance.confidence, i),
        )
        chosen: list[Location] = []
        for i in order:
            c = candidates[i]
            location = Location(
                key=c.provenance.source_id,
                start=c.provenance.start,
                end=c.provenance.end,
                score=c.score,
                via="graph",
                path=c.path,
                element="edge" if c.element == "edge" else "node",
                element_id=c.element_id,
            )
            if any(location.overlaps(kept) for kept in chosen):
                continue
            chosen.append(location)
            if len(chosen) == k:
                break
        return [await describe(self._documents, loc) for loc in chosen]


async def describe(documents: DocumentSource | None, location: Location) -> Location:
    """``location`` with its document's title, hash, and a snippet, when available."""
    if documents is None:
        return location
    try:
        found = await documents.fetch(location.key)
    except DocumentNotFoundError:
        return location
    start = location.start or 0
    end = len(found.text) if location.end is None else location.end
    snippet = found.text[start:end]
    if len(snippet) > SNIPPET_CHARS:
        snippet = snippet[: SNIPPET_CHARS - 1].rstrip() + "…"
    return location.model_copy(
        update={
            "title": found.record.title,
            "doc_hash": found.record.text_hash,  # what the offsets were computed against
            "snippet": " ".join(snippet.split()),
        }
    )
