"""Entry resolution: which node(s) a query starts from.

Kept separate from traversal so evals can report linking errors apart from walking
errors. Datasets that name the topic entity (MetaQA) should pass start nodes directly.
"""

import re
from collections import defaultdict
from typing import Protocol, runtime_checkable

from graphwalk.core.model import NodeId
from graphwalk.embeddings.base import Embedder
from graphwalk.stores.base import GraphStore
from graphwalk.traversal.prefilter import EmbeddingCache, rank_by_similarity
from graphwalk.traversal.prompts import node_text

_WORD = re.compile(r"\w+")


def _words(text: str) -> tuple[str, ...]:
    return tuple(_WORD.findall(text.casefold()))


@runtime_checkable
class EntryResolver(Protocol):
    async def resolve(self, query: str) -> list[NodeId]: ...


class NameEntryResolver:
    """Finds node names and aliases mentioned in the query (whole words, ignoring case).

    Longest mentions win; overlapping shorter ones are dropped. Every remaining mention
    contributes all nodes with that name, in order of appearance. If nothing matches and
    an embedder is given, falls back to the single most similar node.

    The name index is built on first use; call :meth:`refresh` after changing the graph.
    """

    def __init__(
        self,
        store: GraphStore,
        *,
        embedder: Embedder | None = None,
        min_name_chars: int = 3,
        max_name_words: int = 12,
    ) -> None:
        self._store = store
        self._cache = None if embedder is None else EmbeddingCache(embedder)
        self._min_chars = min_name_chars
        self._max_words = max_name_words
        self._index: dict[tuple[str, ...], list[NodeId]] | None = None
        self._texts: list[tuple[NodeId, str]] = []

    def refresh(self) -> None:
        self._index = None

    async def _build(self) -> dict[tuple[str, ...], list[NodeId]]:
        index: dict[tuple[str, ...], list[NodeId]] = defaultdict(list)
        texts: list[tuple[NodeId, str]] = []
        async for node in self._store.iter_nodes():
            texts.append((node.id, node_text(node)))
            for name in (node.name, *node.aliases):
                words = _words(name)
                if len(name.strip()) < self._min_chars or not words:
                    continue
                if len(words) <= self._max_words and node.id not in index[words]:
                    index[words].append(node.id)
        self._index, self._texts = dict(index), texts
        return self._index

    async def resolve(self, query: str) -> list[NodeId]:
        index = self._index if self._index is not None else await self._build()
        words = _words(query)
        spans: list[tuple[int, int]] = []
        for i in range(len(words)):
            for j in range(min(len(words), i + self._max_words), i, -1):
                if words[i:j] in index:
                    spans.append((i, j))
                    break
        spans.sort(key=lambda s: (-(s[1] - s[0]), s[0]))
        taken: list[tuple[int, int]] = []
        for start, end in spans:
            if all(end <= s or start >= e for s, e in taken):
                taken.append((start, end))
        taken.sort()
        found: list[NodeId] = []
        for start, end in taken:
            for node_id in index[words[start:end]]:
                if node_id not in found:
                    found.append(node_id)
        if found or self._cache is None or not self._texts:
            return found
        ranked = await rank_by_similarity(self._cache, query, [t for _, t in self._texts])
        return [self._texts[ranked[0][0]][0]]
