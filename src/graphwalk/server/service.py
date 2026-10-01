"""What the MCP tools do, independent of the MCP SDK (so it is easy to test).

One :class:`Service` serves one SQLite graph. Each request's provider configuration
comes from the environment (stdio) or from the client's headers (remote); an
:class:`~graphwalk.index.Index` is kept per distinct configuration, so clients with
different keys never share backends. Nothing is written anywhere but the graph: keys
live in memory, in those indexes, and are dropped when the cache evicts them.
"""

import asyncio
import hashlib
import json
import secrets
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, JsonValue

from graphwalk.config import ResolvedConfig, resolve_config
from graphwalk.core.errors import GraphwalkError
from graphwalk.core.model import Node
from graphwalk.index import Index
from graphwalk.ingest.pipeline import IngestReport
from graphwalk.ingest.sources import FileSource, SourceDocument, TextSource
from graphwalk.locate.model import Location
from graphwalk.server.settings import ServerSettings
from graphwalk.stores.sqlite_store import SQLiteStore

type IndexFactory = Callable[[SQLiteStore, ResolvedConfig], Index]
type Mode = Literal["stdio", "http"]

INDEX_CACHE_SIZE = 32


class ToolError(GraphwalkError):
    """A tool call that cannot be served; the message is shown to the client."""


# -- tool outputs ---------------------------------------------------------------------


class LocationOut(BaseModel):
    key: str
    """The document; pass it, with ``start``/``end``/``doc_hash``, to ``read``."""
    start: int | None
    end: int | None
    doc_hash: str | None
    title: str | None
    snippet: str | None
    path: list[str]
    score: float
    via: str


class LocateOut(BaseModel):
    locations: list[LocationOut]
    entries: list[str] = Field(default_factory=list[str])
    """Names of the graph entities the query was linked to (graph and hybrid modes)."""
    decision_model: str | None = None
    decision_calls: int = 0


class ReadOut(BaseModel):
    key: str
    title: str | None
    text: str
    start: int
    end: int
    stale: bool
    """The document changed since it was located: re-run ``locate``."""
    truncated: bool = False


class NodeRef(BaseModel):
    id: str
    name: str
    type: str


class AnswerOut(BaseModel):
    names: list[str]
    node_ids: list[str]
    path: list[str]
    """The walk from the entry entity: ``relation`` (followed forwards) or
    ``relation^-1`` (backwards), one per hop."""
    confidence: float
    """The path's probability, length-normalized (``exp(score)``)."""


class WalkOut(BaseModel):
    answers: list[AnswerOut]
    """Best first; one walk per entry entity, up to three answers each."""
    entries: list[NodeRef]
    confidence: float | None
    """The best answer's confidence; ``None`` if no walk found an answer."""
    escalated: bool
    """A low-confidence walk was redone with the fallback decider."""
    decision_model: str | None = None
    decision_calls: int = 0
    cost_usd: float | None = None


class SourceRef(BaseModel):
    key: str
    start: int | None
    end: int | None
    confidence: float


class EdgeOut(BaseModel):
    relation: str
    direction: str
    """``out``: node --relation--> other; ``in``: other --relation--> node."""
    other: NodeRef
    sources: list[SourceRef]


class NeighborsOut(BaseModel):
    nodes: list[NodeRef]
    """Every node the name matched."""
    edges: list[EdgeOut]
    total: int
    """Edges before ``limit`` was applied."""


class NodeOut(BaseModel):
    id: str
    name: str
    type: str
    summary: str | None
    aliases: list[str]
    attributes: dict[str, JsonValue]
    sources: list[SourceRef]
    degree: int


class JobOut(BaseModel):
    job_id: str
    state: Literal["running", "done", "failed"]
    chunks_done: int = 0
    chunks_total: int = 0
    started_at: float
    finished_at: float | None = None
    report: dict[str, JsonValue] | None = None
    """A summary of the ingest report (counts, models, cost) when done."""
    error: str | None = None


class StatusOut(BaseModel):
    nodes: int
    edges: int
    documents: int
    sources: dict[str, int]
    """Documents per source id."""
    documents_without_text: int
    """Stored hash-only (``store_text=False``); ``read`` cannot serve them remotely."""
    config: dict[str, JsonValue]
    """Providers and models this client's requests would use; keys shown only as present
    or not."""
    jobs: list[JobOut]
    limits: dict[str, int]


class InlineDocument(BaseModel):
    id: str = Field(min_length=1, max_length=512)
    text: str
    title: str | None = None


# -- the service ----------------------------------------------------------------------


@dataclass
class _Job:
    job_id: str
    started_at: float
    task: "asyncio.Task[IngestReport] | None" = None
    index: Index | None = None
    done: int = 0
    total: int = 0
    finished_at: float | None = None
    report: IngestReport | None = None
    error: str | None = None


def _fingerprint(config: ResolvedConfig) -> str:
    """Identifies a configuration (keys included) without storing the keys."""
    s = config.settings
    material = {
        name: (value.get_secret_value() if hasattr(value, "get_secret_value") else value)
        for name, value in ((n, getattr(s, n)) for n in type(s).model_fields)
    }
    return hashlib.sha256(json.dumps(material, sort_keys=True, default=str).encode()).hexdigest()


def default_index(store: SQLiteStore, config: ResolvedConfig) -> Index:
    return Index(store, config=config, owns_store=False)


def _node_ref(node: Node) -> NodeRef:
    return NodeRef(id=node.id, name=node.name, type=node.type)


def _sources(element: Node, limit: int) -> list[SourceRef]:
    ranked = sorted(element.provenance, key=lambda p: -p.confidence)[:limit]
    return [
        SourceRef(key=p.source_id, start=p.start, end=p.end, confidence=p.confidence)
        for p in ranked
    ]


@dataclass
class Service:
    store: SQLiteStore
    settings: ServerSettings
    mode: Mode
    index_factory: IndexFactory = default_index
    _indexes: "OrderedDict[str, Index]" = field(default_factory=OrderedDict[str, Index])
    _jobs: dict[str, _Job] = field(default_factory=dict[str, _Job])
    _sessions: dict[str, asyncio.Semaphore] = field(default_factory=dict[str, asyncio.Semaphore])
    _stdio_config: ResolvedConfig | None = None

    # -- configuration and backends

    def config_for(self, headers: Mapping[str, str] | None) -> ResolvedConfig:
        if self.mode == "stdio":
            if self._stdio_config is None:
                self._stdio_config = resolve_config()
            return self._stdio_config
        return resolve_config(
            headers=headers or {},
            env_keys=self.settings.server_keys,
            allowed_base_urls=self.settings.base_url_allowlist,
        )

    async def index_for(self, headers: Mapping[str, str] | None) -> Index:
        config = self.config_for(headers)
        key = _fingerprint(config)
        index = self._indexes.get(key)
        if index is not None:
            self._indexes.move_to_end(key)
            return index
        index = self.index_factory(self.store, config)
        self._indexes[key] = index
        while len(self._indexes) > INDEX_CACHE_SIZE:
            _, evicted = self._indexes.popitem(last=False)
            if not self._in_use(evicted):
                await evicted.close()  # its backends; the store stays open
        return index

    def _in_use(self, index: Index) -> bool:
        """A running ingest job still needs it (it is closed with the job instead)."""
        return any(j.index is index and j.finished_at is None for j in self._jobs.values())

    def session_slot(self, session: str) -> asyncio.Semaphore:
        slot = self._sessions.get(session)
        if slot is None:
            slot = asyncio.Semaphore(self.settings.session_concurrency)
            self._sessions[session] = slot
        return slot

    async def close(self) -> None:
        for job in self._jobs.values():
            if job.task is not None and not job.task.done():
                job.task.cancel()
        for index in self._indexes.values():
            await index.close()
        self._indexes.clear()
        await self.store.close()

    # -- tools

    async def locate(
        self, headers: Mapping[str, str] | None, query: str, k: int, mode: str
    ) -> LocateOut:
        if not query.strip():
            raise ToolError("query is empty")
        if not 1 <= k <= self.settings.max_k:
            raise ToolError(f"k must be between 1 and {self.settings.max_k}")
        if mode not in ("graph", "dense", "hybrid"):
            raise ToolError("mode must be graph, dense, or hybrid")
        index = await self.index_for(headers)
        result = await index.locate_detailed(query, k, mode=mode)  # pyright: ignore[reportArgumentType]
        entries = await self.store.get_nodes(list(result.entries))
        walks = result.walks
        return LocateOut(
            locations=[_location_out(loc) for loc in result.locations],
            entries=[entries[e].name for e in result.entries if e in entries],
            decision_model=walks[0].trace.decision_model if walks else None,
            decision_calls=result.decision_calls,
        )

    async def walk(self, headers: Mapping[str, str] | None, query: str) -> WalkOut:
        if not query.strip():
            raise ToolError("query is empty")
        index = await self.index_for(headers)
        result = await index.walk(query)
        answers: list[AnswerOut] = []
        for walk in result.walks:
            answers += [
                AnswerOut(
                    names=list(a.names),
                    node_ids=list(a.node_ids),
                    path=[h.relation + ("" if h.direction == "out" else "^-1") for h in a.path],
                    confidence=a.confidence,
                )
                for a in walk.answers[:3]
            ]
        answers.sort(key=lambda a: -a.confidence)
        entries = await self.store.get_nodes(list(result.entries))
        return WalkOut(
            answers=answers,
            entries=[_node_ref(entries[e]) for e in result.entries if e in entries],
            confidence=result.confidence,
            escalated=result.escalated,
            decision_model=result.walks[0].trace.decision_model if result.walks else None,
            decision_calls=sum(
                w.trace.totals.decision_calls
                + (0 if w.escalation is None else w.escalation.primary.trace.totals.decision_calls)
                for w in result.walks
            ),
            cost_usd=result.cost_usd,
        )

    async def read(  # noqa: PLR0917 - mirrors the read tool
        self,
        headers: Mapping[str, str] | None,
        key: str,
        start: int | None,
        end: int | None,
        doc_hash: str | None,
        context: int,
    ) -> ReadOut:
        if context < 0 or (start is not None and end is not None and end < start):
            raise ToolError("invalid span: need 0 <= start <= end and context >= 0")
        if (start is None) != (end is None):
            raise ToolError("give both start and end, or neither (the whole document)")
        location = Location(
            key=key, start=start, end=end, doc_hash=doc_hash, via="graph", element="chunk"
        )
        index = await self.index_for(headers)
        passage = await index.read(location, context=context)
        text, truncated = passage.text, False
        if len(text) > self.settings.max_read_chars:
            text, truncated = text[: self.settings.max_read_chars], True
        return ReadOut(
            key=key,
            title=passage.title,
            text=text,
            start=passage.start,
            end=passage.start + len(text),
            stale=passage.stale,
            truncated=truncated,
        )

    async def _resolve_nodes(self, node: str) -> list[Node]:
        found = await self.store.get_node(node)
        if found is not None:
            return [found]
        matches = await self.store.find_nodes(name=node, limit=10)
        if not matches:
            raise ToolError(f"no node with id or name {node!r}")
        return matches

    async def neighbors(self, node: str, direction: str, limit: int) -> NeighborsOut:
        if direction not in ("out", "in", "both"):
            raise ToolError("direction must be out, in, or both")
        if not 1 <= limit <= 500:  # noqa: PLR2004
            raise ToolError("limit must be between 1 and 500")
        nodes = await self._resolve_nodes(node)
        edges: list[EdgeOut] = []
        for found in nodes:
            for neighbor in await self.store.neighbors(found.id, direction=direction):  # pyright: ignore[reportArgumentType]
                edges.append(
                    EdgeOut(
                        relation=neighbor.edge.type,
                        direction=neighbor.direction,
                        other=_node_ref(neighbor.node),
                        sources=[
                            SourceRef(
                                key=p.source_id, start=p.start, end=p.end, confidence=p.confidence
                            )
                            for p in neighbor.edge.provenance[:3]
                        ],
                    )
                )
        return NeighborsOut(
            nodes=[_node_ref(n) for n in nodes], edges=edges[:limit], total=len(edges)
        )

    async def get_node(self, node: str) -> list[NodeOut]:
        out: list[NodeOut] = []
        for found in await self._resolve_nodes(node):
            out.append(
                NodeOut(
                    id=found.id,
                    name=found.name,
                    type=found.type,
                    summary=found.summary,
                    aliases=list(found.aliases),
                    attributes=dict(found.attributes),
                    sources=_sources(found, 20),
                    degree=await self.store.degree(found.id, direction="both"),
                )
            )
        return out

    async def ingest(
        self,
        headers: Mapping[str, str] | None,
        documents: Sequence[InlineDocument] | None,
        path: str | None,
        source_id: str | None,
    ) -> JobOut:
        if (documents is None) == (path is None):
            raise ToolError("give either documents or path")
        if documents is not None:
            size = sum(len(d.text) for d in documents)
            if size > self.settings.max_ingest_chars:
                raise ToolError(
                    f"{size} characters exceeds the {self.settings.max_ingest_chars} limit"
                )
            ids = [d.id for d in documents]
            if len(set(ids)) != len(ids):
                raise ToolError("document ids must be unique")
            source = TextSource(
                source_id or "inline",
                [SourceDocument(doc_id=d.id, text=d.text, title=d.title) for d in documents],
            )
        else:
            assert path is not None  # noqa: S101 - checked above
            source = FileSource(self._allowed_path(path), source_id=source_id)
        index = await self.index_for(headers)
        job = _Job(job_id=secrets.token_hex(8), started_at=time.time(), index=index)

        def progress(done: int, total: int) -> None:
            job.done, job.total = done, total

        async def run() -> IngestReport:
            try:
                job.report = await index.ingest(source, on_progress=progress)
            except Exception as error:
                job.error = str(error)[:500]
                raise
            finally:
                job.finished_at = time.time()
                if index not in self._indexes.values():
                    await index.close()  # evicted while the job ran
            return job.report

        job.task = asyncio.create_task(run())
        job.task.add_done_callback(lambda t: t.cancelled() or t.exception())
        self._jobs[job.job_id] = job
        return self._job_out(job)

    def _allowed_path(self, path: str) -> Path:
        target = Path(path).expanduser()
        if self.mode == "http":
            root = self.settings.ingest_root
            if root is None:
                raise ToolError(
                    "this server does not ingest server paths; send documents inline "
                    "(the operator can allow a directory with GRAPHWALK_INGEST_ROOT)"
                )
            resolved_root = root.resolve()
            target = (resolved_root / target).resolve()
            if not target.is_relative_to(resolved_root):
                raise ToolError("path is outside GRAPHWALK_INGEST_ROOT")
        if not target.exists():
            raise ToolError(f"{path}: no such file or directory")
        return target

    def ingest_status(self, job_id: str) -> JobOut:
        job = self._jobs.get(job_id)
        if job is None:
            raise ToolError(f"no ingest job {job_id!r}")
        return self._job_out(job)

    def _job_out(self, job: _Job) -> JobOut:
        state: Literal["running", "done", "failed"] = "running"
        if job.error is not None:
            state = "failed"
        elif job.report is not None:
            state = "done"
        summary: dict[str, JsonValue] | None = None
        if job.report is not None:
            summary = job.report.model_dump(mode="json", exclude={"routes", "errors"})
            summary["errors"] = list[JsonValue](job.report.errors[:10])
        return JobOut(
            job_id=job.job_id,
            state=state,
            chunks_done=job.done,
            chunks_total=job.total,
            started_at=job.started_at,
            finished_at=job.finished_at,
            report=summary,
            error=job.error,
        )

    async def status(self, headers: Mapping[str, str] | None) -> StatusOut:
        nodes, edges = await self.store.counts()
        sources: dict[str, int] = {}
        documents = hash_only = 0
        async for document in self.store.iter_documents():
            documents += 1
            sources[document.source_id] = sources.get(document.source_id, 0) + 1
            hash_only += document.text is None
        config = self.config_for(headers).describe()
        return StatusOut(
            nodes=nodes,
            edges=edges,
            documents=documents,
            sources=sources,
            documents_without_text=hash_only,
            config=json.loads(json.dumps(config)),
            # Remote clients see only jobs they know the id of (via ingest_status).
            jobs=[self._job_out(j) for j in self._jobs.values()] if self.mode == "stdio" else [],
            limits={
                "max_k": self.settings.max_k,
                "max_read_chars": self.settings.max_read_chars,
                "session_concurrency": self.settings.session_concurrency,
                "max_ingest_chars": self.settings.max_ingest_chars,
            },
        )


def _location_out(location: Location) -> LocationOut:
    return LocationOut(
        key=location.key,
        start=location.start,
        end=location.end,
        doc_hash=location.doc_hash,
        title=location.title,
        snippet=location.snippet,
        path=list(location.path),
        score=round(location.score, 6),
        via=location.via,
    )
