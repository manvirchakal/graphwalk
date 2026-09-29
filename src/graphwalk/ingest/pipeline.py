"""Idempotent ingestion: source -> chunks -> extraction -> routing -> graph.

A per-source ledger in the store's metadata maps each document id to a hash of its
content (and the extractor version). On each run:

* unchanged documents are skipped;
* changed documents have their provenance retracted, then are ingested again;
* documents that disappeared from the source are retracted if ``prune_missing``;
* new documents are ingested.

Extraction runs concurrently; routing and writes run one chunk at a time, in source
order, so a run is deterministic given the model outputs.
"""

import asyncio
import time
from collections.abc import Awaitable, Callable, MutableMapping
from dataclasses import dataclass
from typing import cast

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from graphwalk.core.hashing import content_hash
from graphwalk.core.model import NodeId, Provenance, utc_now
from graphwalk.decisions.base import DecisionBackend
from graphwalk.embeddings.base import Embedder
from graphwalk.ingest.chunking import chunk_text
from graphwalk.ingest.extraction import (
    EXTRACTOR_VERSION,
    Extraction,
    ExtractionResult,
    extract,
    name_key,
)
from graphwalk.ingest.resolution import Retraction, retract, upsert_entity, upsert_relation
from graphwalk.ingest.routing import NodeIndex, RouteBatch, Router, RoutingMode
from graphwalk.ingest.sources import Source, SourceDocument
from graphwalk.ingest.spend import Spend
from graphwalk.llm.base import LLMBackend
from graphwalk.stores.base import GraphStore

LEDGER_PREFIX = "ingest:ledger:"
INCOMPLETE = "incomplete"
"""Ledger value for a document whose ingestion partly failed: it counts as changed."""


class IngestConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    max_chunk_chars: int = Field(default=2000, ge=200)
    routing: RoutingMode = "jev"
    max_candidates: int = Field(default=5, ge=1)
    route_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    """Decisions whose chosen option has lower probability go to the LLM."""
    escalate: bool = True
    max_relations: int = Field(default=8, ge=0)
    """Relations shown per candidate node in a routing question."""
    concurrency: int = Field(default=8, ge=1)
    """Concurrent extraction calls."""
    route_concurrency: int = Field(default=4, ge=1)
    """Chunks routed at once, against the graph as it was before the window, then
    applied in order. A mention routed NEW whose exact name was created by an earlier
    chunk of the same window is routed again against the updated graph, so exact-name
    repeats still get a decision. Only fuzzy-name repeats within a window can end up
    as duplicates. 1 = strictly sequential."""
    prune_missing: bool = False
    """Retract documents in the ledger that the source no longer yields."""
    min_similarity: float = Field(default=0.75, ge=-1.0, le=1.0)
    """Embedding similarity for a node to be a routing candidate."""


class RouteRecord(BaseModel):
    """One routing decision, for analysis."""

    model_config = ConfigDict(frozen=True)

    doc_id: str
    mention: str
    type: str
    candidates: list[str]
    """Candidate node names, in the offered order."""
    chosen: str | None
    """Name of the node merged into; ``None`` for a new node."""
    probability: float
    method: str
    error: str | None = None


class IngestReport(BaseModel):
    source_id: str
    documents: int = 0
    new: int = 0
    changed: int = 0
    unchanged: int = 0
    removed: int = 0
    failed: int = 0
    """Documents with at least one chunk whose extraction failed (marked incomplete)."""
    chunks: int = 0
    entities: int = 0
    relations: int = 0
    nodes_created: int = 0
    nodes_merged: int = 0
    edges_created: int = 0
    edges_merged: int = 0
    nodes_deleted: int = 0
    edges_deleted: int = 0
    escalations: int = 0
    """Routing decisions changed or confirmed by the escalation LLM."""
    escalation_model: str | None = None
    escalation_calls: int = 0
    escalation_input_tokens: int = 0
    escalation_output_tokens: int = 0
    escalation_cost_usd: float | None = 0.0
    """Also included in the totals above; reported apart to watch its cost."""
    llm_calls: int = 0
    decision_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = 0.0
    extraction_cache_hits: int = 0
    elapsed_s: float = 0.0
    extract_wait_s: float = 0.0
    """Time routing spent waiting for extractions (extraction runs ahead concurrently)."""
    route_s: float = 0.0
    """Time in routing windows: candidates, decision calls, escalations."""
    apply_s: float = 0.0
    """Time writing routed entities and relations to the store."""
    errors: list[str] = Field(default_factory=list[str])
    routes: list[RouteRecord] = Field(default_factory=list[RouteRecord])

    def add_spend(self, spend: Spend) -> None:
        self.llm_calls += spend.llm_calls
        self.decision_calls += spend.decision_calls
        self.input_tokens += spend.input_tokens
        self.output_tokens += spend.output_tokens
        total = Spend(cost_usd=self.cost_usd)
        total.add_cost(spend.cost_usd)
        self.cost_usd = total.cost_usd


def document_hash(doc: SourceDocument) -> str:
    return content_hash({"title": doc.title, "text": doc.text, "extractor": EXTRACTOR_VERSION})


def chunk_key(chunk: str, title: str | None) -> str:
    return content_hash({"title": title, "chunk": chunk, "extractor": EXTRACTOR_VERSION})


def provenance_id(source_id: str, doc_id: str) -> str:
    """``Provenance.source_id`` for everything derived from one document."""
    return f"{source_id}/{doc_id}"


def _passage(chunk: "_Chunk") -> str:
    return f"{chunk.doc.title}\n\n{chunk.text}" if chunk.doc.title else chunk.text


@dataclass
class _Chunk:
    doc: SourceDocument
    text: str
    key: str
    result: ExtractionResult | None = None


class _Progress:
    """Marks a document ingested in the ledger once all its chunks are applied."""

    def __init__(
        self, todo: list[SourceDocument], chunks: list[_Chunk], ledger: dict[str, str]
    ) -> None:
        self._ledger = ledger
        self._docs = {doc.doc_id: doc for doc in todo}
        self._remaining: dict[str, int] = {}
        self._ok: dict[str, bool] = {}
        for chunk in chunks:
            doc_id = chunk.doc.doc_id
            self._remaining[doc_id] = self._remaining.get(doc_id, 0) + 1
        for doc in todo:  # no chunks (empty text): nothing to do
            if doc.doc_id not in self._remaining:
                ledger[doc.doc_id] = document_hash(doc)

    def done(self, chunk: _Chunk, *, ok: bool, report: IngestReport) -> None:
        doc_id = chunk.doc.doc_id
        self._ok[doc_id] = self._ok.get(doc_id, True) and ok
        self._remaining[doc_id] -= 1
        if self._remaining[doc_id] == 0:
            complete = self._ok[doc_id]
            report.failed += not complete
            self._ledger[doc_id] = document_hash(self._docs[doc_id]) if complete else INCOMPLETE


type ExtractionCache = MutableMapping[str, JsonValue]
"""Chunk key -> ``Extraction`` as JSON. Lets a re-run reuse extractions (e.g. across
routing configurations) without paying for them again."""


class IngestPipeline:
    def __init__(
        self,
        store: GraphStore,
        llm: LLMBackend,
        decider: DecisionBackend | None,
        *,
        embedder: Embedder | None = None,
        config: IngestConfig | None = None,
        extraction_cache: ExtractionCache | None = None,
        escalation_llm: LLMBackend | None = None,
    ) -> None:
        """``escalation_llm`` adjudicates low-confidence routing decisions; it defaults to
        ``llm`` (the extraction model). A stronger model is usually the better choice:
        it is called only for the few decisions below ``route_threshold``."""
        self._store = store
        self._llm = llm
        self._escalation_llm = escalation_llm or llm
        self._decider = decider
        self._embedder = embedder
        self.config = config or IngestConfig()
        self._cache = extraction_cache
        if self.config.routing == "jev" and decider is None:
            msg = "routing='jev' needs a decision backend"
            raise ValueError(msg)

    async def ledger(self, source_id: str) -> dict[str, str]:
        raw = await self._store.get_metadata(LEDGER_PREFIX + source_id)
        if not isinstance(raw, dict):
            return {}
        return {k: str(v) for k, v in raw.items()}

    async def ingest(
        self,
        source: Source,
        *,
        checkpoint: Callable[[], Awaitable[None]] | None = None,
        checkpoint_every: int = 25,
    ) -> IngestReport:
        """Ingest ``source``. ``checkpoint`` (e.g. saving the store) runs every
        ``checkpoint_every`` routing windows; since the ledger records finished
        documents, re-ingesting from a checkpointed store resumes where it stopped."""
        started = time.perf_counter()
        cfg = self.config
        report = IngestReport(source_id=source.source_id)
        if cfg.escalate:
            report.escalation_model = self._escalation_llm.model_id
        ledger = await self.ledger(source.source_id)
        todo, gone = self._plan(list(source.documents()), ledger, source.source_id, report)
        retraction: Retraction = await retract(self._store, gone)
        report.nodes_deleted += retraction.nodes_deleted
        report.edges_deleted += retraction.edges_deleted
        chunks = [
            _Chunk(doc, text, chunk_key(text, doc.title))
            for doc in todo
            for text in chunk_text(doc.text, max_chars=cfg.max_chunk_chars)
        ]
        report.chunks = len(chunks)
        index = await NodeIndex.build(
            self._store, self._embedder, min_similarity=cfg.min_similarity
        )
        router = self._router(index)
        progress = _Progress(todo, chunks, ledger)
        semaphore = asyncio.Semaphore(cfg.concurrency)
        tasks = [asyncio.create_task(self._extract_one(c, semaphore, report)) for c in chunks]
        try:
            for start in range(0, len(chunks), cfg.route_concurrency):
                window = chunks[start : start + cfg.route_concurrency]
                await self._window(
                    source.source_id,
                    window,
                    tasks[start : start + len(window)],
                    router=router,
                    index=index,
                    report=report,
                    progress=progress,
                )
                await self._save_ledger(source.source_id, ledger)
                windows = start // cfg.route_concurrency + 1
                if checkpoint is not None and windows % checkpoint_every == 0:
                    await checkpoint()
        finally:
            for task in tasks:
                task.cancel()
        await self._save_ledger(source.source_id, ledger)
        report.elapsed_s = time.perf_counter() - started
        return report

    def _plan(
        self,
        docs: list[SourceDocument],
        ledger: dict[str, str],
        source_id: str,
        report: IngestReport,
    ) -> tuple[list[SourceDocument], set[str]]:
        """Documents to ingest, and provenance ids to retract first."""
        report.documents = len(docs)
        todo: list[SourceDocument] = []
        gone: set[str] = set()
        for doc in docs:
            before = ledger.get(doc.doc_id)
            if before == document_hash(doc):
                report.unchanged += 1
                continue
            todo.append(doc)
            if before is None:
                report.new += 1
            else:
                report.changed += 1
                gone.add(provenance_id(source_id, doc.doc_id))
        if self.config.prune_missing:
            for doc_id in sorted(set(ledger) - {d.doc_id for d in docs}):
                gone.add(provenance_id(source_id, doc_id))
                del ledger[doc_id]
                report.removed += 1
        return todo, gone

    def _router(self, index: NodeIndex) -> Router:
        cfg = self.config
        return Router(
            self._store,
            index,
            self._decider if cfg.routing == "jev" else None,
            llm=self._escalation_llm if cfg.escalate else None,
            mode=cfg.routing,
            max_candidates=cfg.max_candidates,
            route_threshold=cfg.route_threshold,
            max_relations=cfg.max_relations,
        )

    async def _save_ledger(self, source_id: str, ledger: dict[str, str]) -> None:
        await self._store.set_metadata(LEDGER_PREFIX + source_id, cast("JsonValue", dict(ledger)))

    async def _window(
        self,
        source_id: str,
        window: list["_Chunk"],
        tasks: list["asyncio.Task[None]"],
        *,
        router: Router,
        index: NodeIndex,
        report: IngestReport,
        progress: "_Progress",
    ) -> None:
        """Route ``window``'s chunks concurrently, then apply them in order."""
        t0 = time.perf_counter()
        await asyncio.gather(*tasks)
        t1 = time.perf_counter()
        ready = [c for c in window if c.result is not None and c.result.error is None]
        batches = await asyncio.gather(
            *(router.route(c.result.extraction.entities, _passage(c)) for c in ready)  # type: ignore[union-attr]
        )
        t2 = time.perf_counter()
        routed = dict(zip((id(c) for c in ready), batches, strict=True))
        created: set[NodeId] = set()
        for chunk in window:
            assert chunk.result is not None  # noqa: S101 - awaited above
            error = chunk.result.error
            if error is not None:
                report.errors.append(f"{chunk.doc.doc_id}: {error}")
            else:
                await self._apply(
                    source_id, chunk, routed[id(chunk)], router=router, index=index,
                    report=report, created_in_window=created,
                )  # fmt: skip
            progress.done(chunk, ok=error is None, report=report)
        report.extract_wait_s += t1 - t0
        report.route_s += t2 - t1
        report.apply_s += time.perf_counter() - t2

    async def _extract_one(
        self, chunk: _Chunk, semaphore: asyncio.Semaphore, report: IngestReport
    ) -> None:
        if self._cache is not None and chunk.key in self._cache:
            extraction = Extraction.model_validate(self._cache[chunk.key])
            chunk.result = ExtractionResult(extraction=extraction)
            report.extraction_cache_hits += 1
            return
        async with semaphore:
            chunk.result = await extract(self._llm, chunk.text, title=chunk.doc.title)
        report.add_spend(chunk.result.spend)
        if self._cache is not None and chunk.result.error is None:
            self._cache[chunk.key] = chunk.result.extraction.model_dump(mode="json")

    async def _apply(
        self,
        source_id: str,
        chunk: _Chunk,
        batch: RouteBatch,
        *,
        router: Router,
        index: NodeIndex,
        report: IngestReport,
        created_in_window: set[NodeId],
    ) -> None:
        assert chunk.result is not None  # noqa: S101 - set by _extract_one
        extraction = chunk.result.extraction
        report.entities += len(extraction.entities)
        report.relations += len(extraction.relations)
        stale = [
            i
            for i, route in enumerate(batch.routes)
            if route.target is None
            and any(n in created_in_window for n in index.exact(route.mention.name))
        ]
        if stale:
            # Routed before an earlier chunk of this window created a same-name node:
            # route those mentions again against the graph as it is now.
            again = await router.route([batch.routes[i].mention for i in stale], _passage(chunk))
            batch.spend.add(again.spend)
            batch.escalation.add(again.escalation)
            for i, route in zip(stale, again.routes, strict=True):
                batch.routes[i] = route
        report.add_spend(batch.spend)
        report.add_spend(batch.escalation)
        report.escalation_calls += batch.escalation.llm_calls
        report.escalation_input_tokens += batch.escalation.input_tokens
        report.escalation_output_tokens += batch.escalation.output_tokens
        escalation_cost = Spend(cost_usd=report.escalation_cost_usd)
        escalation_cost.add_cost(batch.escalation.cost_usd if batch.escalation.llm_calls else 0.0)
        report.escalation_cost_usd = escalation_cost.cost_usd
        doc_source = provenance_id(source_id, chunk.doc.doc_id)
        now = utc_now()
        ids: dict[str, NodeId] = {}
        for route in batch.routes:
            if route.method == "llm":
                report.escalations += 1
            provenance = Provenance(
                source_id=doc_source,
                ingested_at=now,
                confidence=route.probability if route.method == "decision" else 1.0,
                content_hash=chunk.key,
            )
            chosen = None if route.target is None else await self._store.get_node(route.target)
            node, created = await upsert_entity(
                self._store, route.target, route.mention, provenance, doc_source
            )
            index.add(node)
            ids[name_key(route.mention.name)] = node.id
            if created:
                created_in_window.add(node.id)
                report.nodes_created += 1
            else:
                report.nodes_merged += 1
            found = await self._store.get_nodes(list(route.candidates))
            names = [found[c].name for c in route.candidates if c in found]
            report.routes.append(
                RouteRecord(
                    doc_id=chunk.doc.doc_id,
                    mention=route.mention.name,
                    type=route.mention.type,
                    candidates=names,
                    chosen=None if chosen is None else chosen.name,
                    probability=route.probability,
                    method=route.method,
                    error=route.error,
                )
            )
        for relation in extraction.relations:
            provenance = Provenance(
                source_id=doc_source, ingested_at=now, confidence=1.0, content_hash=chunk.key
            )
            outcome = await upsert_relation(
                self._store,
                ids[name_key(relation.source)],
                relation.type,
                ids[name_key(relation.target)],
                provenance,
            )
            if outcome is True:
                report.edges_created += 1
            elif outcome is False:
                report.edges_merged += 1
