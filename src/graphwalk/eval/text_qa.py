"""M7: question answering over graphs built from text.

For a seeded, type-stratified sample of 2Wiki or HotpotQA questions, all their
paragraphs (gold and distractor, de-duplicated by title) are pooled. The pool is:

* ingested into one graph (M6 pipeline) that graphwalk walks, and
* indexed as-is, one document per paragraph, for the RAG baselines.

So every system reads the same text; they differ in access. Systems:

* ``graphwalk``: graph-only walking; the answer is the reached node's name.
* ``graphwalk-reader``: the walks plus the reached nodes' facts, read by an LLM.
* ``text-rag``: top-k paragraphs, read by the same LLM.
* ``text-iter-rag``: multi-step retrieval over the paragraphs, same LLM.
"""

import asyncio
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from collections.abc import Awaitable, Callable, Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import Any

from pydantic import JsonValue

from graphwalk.decisions.base import DecisionBackend
from graphwalk.embeddings.base import Embedder
from graphwalk.eval.ingest_eval import documents
from graphwalk.eval.runner import SystemRun, run_system, write_results
from graphwalk.eval.systems import (
    TEXT_PROMPTS,
    DocIndex,
    GraphReaderSystem,
    GraphwalkSystem,
    IterativeRAGSystem,
    VectorRAGSystem,
)
from graphwalk.eval.types import EvalQuestion, QASystem
from graphwalk.ingest.pipeline import ExtractionCache, IngestConfig, IngestPipeline, IngestReport
from graphwalk.ingest.sources import TextSource
from graphwalk.llm.base import LLMBackend
from graphwalk.stores.networkx_store import NetworkXStore
from graphwalk.traversal import (
    ChoiceEntryResolver,
    NameEntryResolver,
    TraversalConfig,
    Traverser,
)
from graphwalk.traversal.prompts import node_text

GRAPHWALK = "graphwalk"
READER = "graphwalk-reader"
TEXT_RAG = "text-rag"
TEXT_ITER_RAG = "text-iter-rag"
TEXT_SYSTEMS = (GRAPHWALK, READER, TEXT_RAG, TEXT_ITER_RAG)
MAX_ANSWER_TYPES = 50


def stratified(
    records: Sequence[Mapping[str, Any]], types: Sequence[str], per_type: int, seed: int
) -> list[Mapping[str, Any]]:
    """``per_type`` records of each type, seeded; kept in dataset order."""
    rng = random.Random(seed)  # noqa: S311 - reproducible sampling
    chosen: set[int] = set()
    for type_ in types:
        pool = [i for i, r in enumerate(records) if r.get("type") == type_]
        chosen.update(rng.sample(pool, min(per_type, len(pool))))
    return [records[i] for i in sorted(chosen)]


def to_questions(records: Sequence[Mapping[str, Any]], dataset: str) -> list[EvalQuestion]:
    return [
        EvalQuestion(
            id=f"{dataset}-{r['_id']}",
            dataset=dataset,
            question=str(r["question"]),
            answers=(str(r["answer"]),),
            kind="text",
            meta={"type": str(r["type"])},
        )
        for r in records
    ]


def paragraph_docs(records: Sequence[Mapping[str, Any]]) -> list[tuple[str, str]]:
    """``(title, "title: text")`` per pooled paragraph, for the RAG index."""
    return [(d.doc_id, f"{d.title}: {d.text}") for d in documents(records)]


def graph_key(dataset: str, records: Sequence[Mapping[str, Any]], config: IngestConfig) -> str:
    """Names a cached ingested graph: the dataset, the pooled paragraphs, the config."""
    material = json.dumps(
        {
            "docs": [(d.doc_id, d.text) for d in documents(records)],
            "config": config.model_dump(mode="json"),
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return f"{dataset}-{hashlib.sha256(material.encode()).hexdigest()[:16]}"


async def build_graph(
    dataset: str,
    records: Sequence[Mapping[str, Any]],
    *,
    llm: LLMBackend,
    decider: DecisionBackend | None,
    embedder: Embedder | None,
    cache: ExtractionCache,
    config: IngestConfig,
    escalation_llm: LLMBackend | None = None,
    graph_path: Path | None = None,
) -> tuple[NetworkXStore, IngestReport | None]:
    """Ingest the pooled paragraphs, or load the graph from ``graph_path`` if it exists
    (then the report is ``None``). A fresh graph is saved to ``graph_path``; while it is
    being built, progress is checkpointed next to it and a re-run resumes from there."""
    if graph_path is not None and await asyncio.to_thread(graph_path.exists):
        return await NetworkXStore.load(graph_path), None
    partial = None if graph_path is None else graph_path.with_suffix(".partial.json")
    store = NetworkXStore()
    if partial is not None and await asyncio.to_thread(partial.exists):
        store = await NetworkXStore.load(partial)
    source = TextSource(f"{dataset}:paragraphs", documents(records))
    pipeline = IngestPipeline(
        store,
        llm,
        decider,
        embedder=embedder,
        config=config,
        extraction_cache=cache,
        escalation_llm=escalation_llm,
    )
    checkpoint = None if partial is None else partial_saver(store, partial)
    report = await pipeline.ingest(source, checkpoint=checkpoint)
    if graph_path is not None:
        await store.save(graph_path)
    if partial is not None:
        await asyncio.to_thread(partial.unlink, missing_ok=True)
    return store, report


def partial_saver(store: NetworkXStore, path: Path) -> Callable[[], Awaitable[None]]:
    async def save() -> None:
        await store.save(path)

    return save


async def build_text_systems(
    names: Sequence[str],
    store: NetworkXStore,
    records: Sequence[Mapping[str, Any]],
    *,
    decider: DecisionBackend,
    embedder: Embedder,
    llm: LLMBackend,
    config: TraversalConfig,
    rag_k: int = 5,
) -> list[QASystem]:
    systems: list[QASystem] = []
    # LLM-extracted graphs have hundreds of free-form types (337 on 2Wiki): offer the
    # answer-type question only the most common ones.
    type_counts = Counter([node.type async for node in store.iter_nodes()])
    node_types = sorted(t for t, _ in type_counts.most_common(MAX_ANSWER_TYPES))
    index: DocIndex | None = None
    traverser: Traverser | None = None
    for name in names:
        if name in (GRAPHWALK, READER):
            if traverser is None:
                traverser = Traverser(
                    store, decider, embedder=embedder, config=config, node_types=node_types
                )
                texts = sorted({node_text(n) async for n in store.iter_nodes()})
                if texts:
                    traverser.preload_embeddings(texts, await embedder.embed(texts))
            resolver = ChoiceEntryResolver(store, decider)
            if name == GRAPHWALK:
                systems.append(
                    GraphwalkSystem(
                        store, traverser, name=GRAPHWALK, linking="choice", resolver=resolver
                    )
                )
            else:
                systems.append(
                    GraphReaderSystem(
                        store, traverser, llm, resolver, names=NameEntryResolver(store), name=READER
                    )
                )
            continue
        if name in (TEXT_RAG, TEXT_ITER_RAG):
            if index is None:
                index = await DocIndex.build(paragraph_docs(records), embedder)
            if name == TEXT_RAG:
                systems.append(
                    VectorRAGSystem(
                        index, embedder, llm, k=rag_k, name=TEXT_RAG, prompts=TEXT_PROMPTS
                    )
                )
            else:
                titles = {title.casefold(): [i] for i, title in enumerate(index.ids)}
                systems.append(
                    IterativeRAGSystem(
                        index,
                        embedder,
                        llm,
                        names=titles,
                        k=rag_k,
                        name=TEXT_ITER_RAG,
                        prompts=TEXT_PROMPTS,
                    )
                )
            continue
        msg = f"unknown system {name!r}; choose from {TEXT_SYSTEMS}"
        raise ValueError(msg)
    return systems


def _fmt(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None or math.isnan(value) else f"{value:.{digits}f}"


def by_type(runs: Sequence[SystemRun]) -> str:
    """F1 (EM) per question type, per system, as a markdown table."""
    types = sorted({str(r.question.meta.get("type")) for run in runs for r in run.records})
    header = "| system | " + " | ".join(f"{t} F1 (EM)" for t in types) + " |"
    rows = [header, "|---" * (len(types) + 1) + "|"]
    for run in runs:
        groups: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for record in run.records:
            groups[str(record.question.meta.get("type"))].append((record.score.f1, record.score.em))
        cells: list[str] = []
        for type_ in types:
            values = groups.get(type_, [])
            if not values:
                cells.append("n/a")
                continue
            f1 = math.fsum(v[0] for v in values) / len(values)
            em = math.fsum(v[1] for v in values) / len(values)
            cells.append(f"{_fmt(f1)} ({_fmt(em)}) n={len(values)}")
        rows.append(f"| {run.system} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


async def run_text_qa(
    dataset: str,
    records: Sequence[Mapping[str, Any]],
    systems: Sequence[QASystem],
    *,
    out_dir: Path,
    params: dict[str, JsonValue],
    ingest: IngestReport | None,
    concurrency: int = 8,
    on_progress: Callable[[str, int, int], None] | None = None,
    checkpoint_dir: Path | None = None,
) -> tuple[Path, list[SystemRun]]:
    """Run each system on the records' questions and write results. With
    ``checkpoint_dir``, each finished system's run is saved there and reused on a
    re-run, so an interrupted evaluation resumes instead of starting over."""
    questions = to_questions(records, dataset)
    runs: list[SystemRun] = []
    for system in systems:
        saved = None if checkpoint_dir is None else checkpoint_dir / f"{system.name}.json"
        if saved is not None and await asyncio.to_thread(saved.exists):
            text = await asyncio.to_thread(saved.read_text, encoding="utf-8")
            runs.append(SystemRun.model_validate_json(text))
            continue
        progress = None if on_progress is None else partial(on_progress, system.name)
        run = await run_system(system, questions, concurrency=concurrency, on_done=progress)
        if saved is not None:
            await asyncio.to_thread(saved.parent.mkdir, parents=True, exist_ok=True)
            await asyncio.to_thread(saved.write_text, run.model_dump_json(), encoding="utf-8")
        runs.append(run)
    out = write_results(out_dir, dataset=f"{dataset}-text", runs=runs, params=params)
    extra = ["", "## By question type", "", by_type(runs), ""]
    if ingest is not None:
        cost = "n/a" if ingest.cost_usd is None else f"${ingest.cost_usd:.4f}"
        extra += [
            "## Ingestion",
            "",
            f"{ingest.documents} paragraphs, {ingest.chunks} chunks, {ingest.nodes_created} "
            f"nodes created, {ingest.nodes_merged} merged, {ingest.edges_created} edges; "
            f"{ingest.llm_calls} LLM + {ingest.decision_calls} decision calls "
            f"({ingest.extraction_cache_hits} cached extractions), {cost}, "
            f"{ingest.elapsed_s:.0f}s.",
            "",
        ]
    with (out / "summary.md").open("a", encoding="utf-8") as fh:
        fh.write("\n".join(extra))
    return out, runs
