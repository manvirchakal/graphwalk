"""``Index``: the public entry point. Ingest documents, ``locate`` answers, ``read`` them.

    async with Index.open("my.db", llm=llm, decider=decider) as index:
        await index.ingest("docs/")
        for location in await index.locate("Where was Marie Curie's husband born?"):
            print((await index.read(location)).text)

The graph is an index over the text: ``locate`` walks it from the entities the query
names and returns the source spans behind the walk, with the path that reached them.
"""

import os
from collections.abc import Iterable
from pathlib import Path
from types import TracebackType
from typing import Literal, Self

from graphwalk.config import GraphwalkSettings
from graphwalk.core.model import Direction, Neighbor
from graphwalk.decisions.base import DecisionBackend
from graphwalk.embeddings.base import Embedder
from graphwalk.ingest.pipeline import ExtractionCache, IngestConfig, IngestPipeline, IngestReport
from graphwalk.ingest.sources import FileSource, Source, SourceDocument, TextSource
from graphwalk.llm.base import LLMBackend
from graphwalk.locate.dense import DenseLocator, fuse
from graphwalk.locate.documents import DocumentSource, OnStale, StoredDocuments, read
from graphwalk.locate.graph import GraphLocator, LocateResult
from graphwalk.locate.model import Location, Passage
from graphwalk.stores.base import DocumentStore, GraphStore
from graphwalk.stores.sqlite_store import SQLiteStore
from graphwalk.traversal.config import TraversalConfig
from graphwalk.traversal.engine import Traverser

type LocateMode = Literal["graph", "dense", "hybrid"]


def default_decider() -> DecisionBackend:
    """Jev from the environment (``GRAPHWALK_DECISION_PROVIDER`` and its API key)."""
    from graphwalk.decisions.jev import JevBackend  # noqa: PLC0415 - loads the SDK lazily

    return JevBackend.from_settings(GraphwalkSettings())


class Index:
    """A graph plus the documents it was built from.

    ``decider`` drives both the walk and ingestion routing; if omitted, one is built
    from the environment on first use (see :func:`default_decider`). ``llm`` extracts
    entities and relations and is needed only to ingest. ``embedder`` is optional for
    graph ``locate`` (it prefilters wide steps and backs name linking) and required for
    ``dense`` and ``hybrid``.
    """

    def __init__(
        self,
        store: GraphStore,
        *,
        decider: DecisionBackend | None = None,
        llm: LLMBackend | None = None,
        escalation_llm: LLMBackend | None = None,
        embedder: Embedder | None = None,
        traversal: TraversalConfig | None = None,
        ingest: IngestConfig | None = None,
        documents: DocumentSource | None = None,
        extraction_cache: ExtractionCache | None = None,
    ) -> None:
        self.store = store
        self._decider = decider
        self._owns_decider = False
        self._llm = llm
        self._escalation_llm = escalation_llm
        self._embedder = embedder
        self._traversal = traversal or TraversalConfig()
        self._ingest = ingest or IngestConfig()
        if documents is None and isinstance(store, DocumentStore):
            documents = StoredDocuments(store)
        self._documents = documents
        self._cache = extraction_cache
        self._graph: GraphLocator | None = None
        self._dense: DenseLocator | None = None

    @classmethod
    def open(cls, path: str | os.PathLike[str], **kwargs: object) -> Self:
        """An index in the SQLite file ``path`` (created if missing). Keyword arguments
        go to :class:`Index`."""
        return cls(SQLiteStore(path), **kwargs)  # pyright: ignore[reportArgumentType]

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.close()

    async def close(self) -> None:
        """Close the store, and the decider if the index built it."""
        if self._owns_decider and self._decider is not None:
            await self._decider.aclose()
        await self.store.close()

    @property
    def decider(self) -> DecisionBackend:
        if self._decider is None:
            self._decider = default_decider()
            self._owns_decider = True
        return self._decider

    # ------------------------------------------------------------------ ingest

    async def ingest(
        self, source: Source | str | os.PathLike[str] | Iterable[SourceDocument]
    ) -> IngestReport:
        """Ingest a :class:`Source`, a file or directory path, or documents in memory
        (source id ``"memory"``). Unchanged documents are skipped."""
        if self._llm is None:
            msg = "ingest needs an extraction model: Index(..., llm=...)"
            raise ValueError(msg)
        if isinstance(source, str | os.PathLike):
            source = FileSource(Path(source))
        elif not isinstance(source, Source):
            source = TextSource("memory", source)
        decider = self.decider if self._ingest.routing == "jev" else None
        pipeline = IngestPipeline(
            self.store,
            self._llm,
            decider,
            embedder=self._embedder,
            config=self._ingest,
            extraction_cache=self._cache,
            escalation_llm=self._escalation_llm,
        )
        report = await pipeline.ingest(source)
        if self._graph is not None:
            self._graph.refresh()
        if self._dense is not None:
            self._dense.refresh()
        return report

    # ------------------------------------------------------------------ locate / read

    def _graph_locator(self) -> GraphLocator:
        if self._graph is None:
            traverser = Traverser(
                self.store, self.decider, embedder=self._embedder, config=self._traversal
            )
            self._graph = GraphLocator(self.store, traverser, documents=self._documents)
        return self._graph

    def _dense_locator(self) -> DenseLocator:
        if self._dense is None:
            if self._embedder is None:
                msg = "dense and hybrid locate need an embedder: Index(..., embedder=...)"
                raise ValueError(msg)
            if self._documents is None or not isinstance(self.store, DocumentStore):
                msg = "dense and hybrid locate need a store that keeps documents"
                raise ValueError(msg)
            self._dense = DenseLocator(self.store, self._documents, self._embedder)
        return self._dense

    async def locate(self, query: str, k: int = 5, *, mode: LocateMode = "graph") -> list[Location]:
        """Up to ``k`` source spans likely to hold the answer to ``query``, best first.

        ``graph`` walks the graph from the entities the query names (empty if it names
        none the graph knows); ``dense`` ranks text chunks by embedding similarity;
        ``hybrid`` fuses the two by reciprocal rank.
        """
        return (await self.locate_detailed(query, k, mode=mode)).locations

    async def locate_detailed(
        self, query: str, k: int = 5, *, mode: LocateMode = "graph"
    ) -> LocateResult:
        """:meth:`locate`, plus the entry nodes and walks behind the result."""
        if mode == "dense":
            return LocateResult(locations=await self._dense_locator().locate(query, k))
        if mode == "graph":
            return await self._graph_locator().locate(query, k)
        if mode != "hybrid":
            msg = f"mode must be graph, dense, or hybrid, got {mode!r}"
            raise ValueError(msg)
        dense = await self._dense_locator().locate(query, k)
        graph = await self._graph_locator().locate(query, k)
        graph.locations = fuse([graph.locations, dense], k)
        return graph

    async def read(
        self, location: Location, *, context: int = 0, on_stale: OnStale = "flag"
    ) -> Passage:
        """The text at ``location`` (plus ``context`` characters either side)."""
        if self._documents is None:
            msg = "this store keeps no documents; pass documents=FileDocuments(...)"
            raise ValueError(msg)
        return await read(self._documents, location, context=context, on_stale=on_stale)

    # ------------------------------------------------------------------ graph

    async def neighbors(self, node: str, *, direction: Direction = "both") -> list[Neighbor]:
        """Edges around ``node``: a node id, or else a name or alias (every match)."""
        found = await self.store.get_node(node)
        ids = (
            [found.id]
            if found is not None
            else [n.id for n in await self.store.find_nodes(name=node)]
        )
        out: list[Neighbor] = []
        for node_id in ids:
            out += await self.store.neighbors(node_id, direction=direction)
        return out
