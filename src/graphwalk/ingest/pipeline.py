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
from collections.abc import MutableMapping
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
from graphwalk.ingest.routing import NodeIndex, Router, RoutingMode
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
    concurrency: int = Field(default=4, ge=1)
    """Concurrent extraction calls."""
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
    llm_calls: int = 0
    decision_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = 0.0
    extraction_cache_hits: int = 0
    elapsed_s: float = 0.0
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


@dataclass
class _Chunk:
    doc: SourceDocument
    text: str
    key: str
    result: ExtractionResult | None = None


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
    ) -> None:
        self._store = store
        self._llm = llm
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

    async def ingest(self, source: Source) -> IngestReport:
        started = time.perf_counter()
        cfg = self.config
        report = IngestReport(source_id=source.source_id)
        ledger = await self.ledger(source.source_id)
        docs = list(source.documents())
        report.documents = len(docs)
        todo: list[SourceDocument] = []
        gone: set[str] = set()
        present = {d.doc_id for d in docs}
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
                gone.add(provenance_id(source.source_id, doc.doc_id))
        if cfg.prune_missing:
            for doc_id in sorted(set(ledger) - present):
                gone.add(provenance_id(source.source_id, doc_id))
                del ledger[doc_id]
                report.removed += 1
        retraction: Retraction = await retract(self._store, gone)
        report.nodes_deleted += retraction.nodes_deleted
        report.edges_deleted += retraction.edges_deleted

        chunks = [
            _Chunk(doc, text, chunk_key(text, doc.title))
            for doc in todo
            for text in chunk_text(doc.text, max_chars=cfg.max_chunk_chars)
        ]
        report.chunks = len(chunks)
        await self._extract_all(chunks, report)

        index = await NodeIndex.build(
            self._store, self._embedder, min_similarity=cfg.min_similarity
        )
        router = Router(
            self._store,
            index,
            self._decider if cfg.routing == "jev" else None,
            llm=self._llm if cfg.escalate else None,
            mode=cfg.routing,
            max_candidates=cfg.max_candidates,
            route_threshold=cfg.route_threshold,
            max_relations=cfg.max_relations,
        )
        by_doc: dict[str, list[_Chunk]] = {}
        for chunk in chunks:
            by_doc.setdefault(chunk.doc.doc_id, []).append(chunk)
        for doc in todo:
            complete = True
            for chunk in by_doc.get(doc.doc_id, []):
                assert chunk.result is not None  # noqa: S101 - set by _extract_all
                if chunk.result.error is not None:
                    complete = False
                    report.errors.append(f"{doc.doc_id}: {chunk.result.error}")
                    continue
                await self._apply(source.source_id, chunk, router, index, report)
            if not complete:
                report.failed += 1
            ledger[doc.doc_id] = document_hash(doc) if complete else INCOMPLETE
            await self._store.set_metadata(
                LEDGER_PREFIX + source.source_id, cast("JsonValue", dict(ledger))
            )
        await self._store.set_metadata(
            LEDGER_PREFIX + source.source_id, cast("JsonValue", dict(ledger))
        )
        report.elapsed_s = time.perf_counter() - started
        return report

    async def _extract_all(self, chunks: list[_Chunk], report: IngestReport) -> None:
        semaphore = asyncio.Semaphore(self.config.concurrency)

        async def one(chunk: _Chunk) -> None:
            if self._cache is not None and chunk.key in self._cache:
                extraction = Extraction.model_validate(self._cache[chunk.key])
                chunk.result = ExtractionResult(extraction=extraction)
                report.extraction_cache_hits += 1
                return
            async with semaphore:
                chunk.result = await extract(self._llm, chunk.text, title=chunk.doc.title)
            if self._cache is not None and chunk.result.error is None:
                self._cache[chunk.key] = chunk.result.extraction.model_dump(mode="json")

        await asyncio.gather(*(one(c) for c in chunks))
        for chunk in chunks:
            assert chunk.result is not None  # noqa: S101 - set above
            report.add_spend(chunk.result.spend)

    async def _apply(
        self,
        source_id: str,
        chunk: _Chunk,
        router: Router,
        index: NodeIndex,
        report: IngestReport,
    ) -> None:
        assert chunk.result is not None  # noqa: S101 - set by _extract_all
        extraction = chunk.result.extraction
        report.entities += len(extraction.entities)
        report.relations += len(extraction.relations)
        passage = f"{chunk.doc.title}\n\n{chunk.text}" if chunk.doc.title else chunk.text
        batch = await router.route(extraction.entities, passage)
        report.add_spend(batch.spend)
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
