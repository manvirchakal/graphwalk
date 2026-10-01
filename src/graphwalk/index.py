"""``Index``: the public entry point. Ingest documents, ``locate`` answers, ``read`` them.

    async with Index.open("my.db", llm=llm, decider=decider) as index:
        await index.ingest("docs/")
        for location in await index.locate("Where was Marie Curie's husband born?"):
            print((await index.read(location)).text)

The graph is an index over the text: ``locate`` walks it from the entities the query
names and returns the source spans behind the walk, with the path that reached them.
"""

import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Literal, Self

from graphwalk.config import ResolvedConfig, resolve_config
from graphwalk.core.model import Direction, Neighbor, NodeId
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
from graphwalk.traversal.escalation import EscalatingTraverser, Walker
from graphwalk.traversal.trace import Answer, TraversalResult

type LocateMode = Literal["graph", "dense", "hybrid"]


@dataclass(frozen=True)
class WalkResult:
    """The graph's own answer to a query: one walk per entry node."""

    query: str
    entries: tuple[NodeId, ...]
    walks: tuple[TraversalResult, ...]

    @property
    def best(self) -> Answer | None:
        """The most confident answer over all walks."""
        answers = [w.best for w in self.walks if w.best is not None]
        return max(answers, key=lambda a: a.confidence, default=None)

    @property
    def confidence(self) -> float | None:
        best = self.best
        return None if best is None else best.confidence

    @property
    def escalated(self) -> bool:
        return any(w.escalation is not None for w in self.walks)

    @property
    def cost_usd(self) -> float | None:
        costs = [w.cost_usd for w in self.walks]
        return None if any(c is None for c in costs) else sum(c or 0.0 for c in costs)


class Index:
    """A graph plus the documents it was built from.

    Backends not passed are built on first use from ``config`` (by default the
    environment; see :mod:`graphwalk.config`):

    * ``decider`` drives the walk and ingestion routing (Jev, or the LLM fallback);
    * ``llm`` extracts entities and relations (needed only to ingest);
    * ``fallback_decider`` (with ``escalate_below``) re-walks queries whose best answer's
      confidence is below the threshold; by default the LLM-as-decider on the
      escalation model (see :mod:`graphwalk.traversal.escalation`);
    * ``embedder`` is optional for graph ``locate`` (it prefilters wide steps and backs
      name linking; not built by default, since fastembed downloads a model) and
      required for ``dense`` and ``hybrid`` (built from config if not passed).
    """

    def __init__(
        self,
        store: GraphStore,
        *,
        decider: DecisionBackend | None = None,
        fallback_decider: DecisionBackend | None = None,
        escalate_below: float | None = None,
        llm: LLMBackend | None = None,
        escalation_llm: LLMBackend | None = None,
        embedder: Embedder | None = None,
        traversal: TraversalConfig | None = None,
        ingest: IngestConfig | None = None,
        documents: DocumentSource | None = None,
        extraction_cache: ExtractionCache | None = None,
        config: ResolvedConfig | None = None,
        owns_store: bool = True,
    ) -> None:
        """``owns_store=False``: :meth:`close` leaves the store open (several indexes,
        e.g. one per remote session's keys, can share one store)."""
        self.store = store
        self._owns_store = owns_store
        self._config = config
        self._decider = decider
        self._fallback_decider = fallback_decider
        if escalate_below is not None and not 0.0 < escalate_below <= 1.0:
            msg = f"escalate_below must be in (0, 1], got {escalate_below}"
            raise ValueError(msg)
        self._escalate_below = escalate_below
        self._owns: list[object] = []
        """Backends this index built, closed with it."""
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
        for decider in (self._decider, self._fallback_decider):
            if decider is not None and decider in self._owns:
                await decider.aclose()
        if self._owns_store:
            await self.store.close()

    @property
    def decider(self) -> DecisionBackend:
        if self._decider is None:
            from graphwalk.providers import make_decider  # noqa: PLC0415

            self._decider = make_decider(self.config)
            self._owns.append(self._decider)
        return self._decider

    @property
    def fallback_decider(self) -> DecisionBackend:
        """The decider escalated walks use: as passed, or the LLM-as-decider."""
        if self._fallback_decider is None:
            from graphwalk.decisions.llm_decider import LLMDecider  # noqa: PLC0415
            from graphwalk.providers import make_llm  # noqa: PLC0415

            self._fallback_decider = LLMDecider(make_llm(self.config, role="escalation"))
            self._owns.append(self._fallback_decider)
        return self._fallback_decider

    @property
    def config(self) -> ResolvedConfig:
        if self._config is None:
            self._config = resolve_config()
        return self._config

    # ------------------------------------------------------------------ ingest

    async def ingest(
        self,
        source: Source | str | os.PathLike[str] | Iterable[SourceDocument],
        *,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> IngestReport:
        """Ingest a :class:`Source`, a file or directory path, or documents in memory
        (source id ``"memory"``). Unchanged documents are skipped."""
        if self._llm is None:
            from graphwalk.providers import make_llm  # noqa: PLC0415

            self._llm = make_llm(self.config)
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
        report = await pipeline.ingest(source, on_progress=on_progress)
        if self._graph is not None:
            self._graph.refresh()
        if self._dense is not None:
            self._dense.refresh()
        return report

    # ------------------------------------------------------------------ locate / read

    def _graph_locator(self) -> GraphLocator:
        if self._graph is None:
            traverser: Walker = Traverser(
                self.store, self.decider, embedder=self._embedder, config=self._traversal
            )
            if self._escalate_below is not None:
                fallback = Traverser(
                    self.store,
                    self.fallback_decider,
                    embedder=self._embedder,
                    config=self._traversal,
                )
                traverser = EscalatingTraverser(traverser, fallback, threshold=self._escalate_below)
            self._graph = GraphLocator(self.store, traverser, documents=self._documents)
        return self._graph

    def _dense_locator(self) -> DenseLocator:
        if self._dense is None:
            if self._embedder is None:
                from graphwalk.providers import make_embedder  # noqa: PLC0415

                self._embedder = make_embedder(self.config)
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

    async def walk(self, query: str) -> WalkResult:
        """Answer ``query`` from the graph itself: link the entities it names, walk from
        each, and return the answers with their confidence. For curated graphs, where
        the graph holds the facts; for text, :meth:`locate` and read the source."""
        entries, _, walks = await self._graph_locator().walk(query)
        return WalkResult(query=query, entries=tuple(entries), walks=tuple(walks))

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
