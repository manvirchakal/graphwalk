"""E1: evidence retrieval. Graph `locate` vs dense vs multi-step retrieval vs hybrid.

    uv run python scripts/eval/evidence.py 2wiki
    uv run python scripts/eval/evidence.py hotpotqa
    uv run python scripts/eval/evidence.py fanoutqa

Uses the M7 question samples, pooled text, and cached graphs (run scripts/eval/text_qa.py
and scripts/eval/fanout.py first). No answers are generated; each method's ranked
documents are scored against the gold evidence (see graphwalk.eval.evidence). Methods:

* graph: `locate(mode="graph")` as the library ships it (entities linked by name);
* graph-choice: the same walks, entry entity chosen by Jev (the M7 reader's linking);
* dense: `locate(mode="dense")` (titled 800-character chunks, bge-small), one
  location per document;
* hybrid, hybrid-choice: reciprocal-rank fusion of graph (or graph-choice) and dense,
  as `locate(mode="hybrid")`;
* title+dense (control): documents whose title the question names, then dense;
* multi-step: the documents M7's multi-step RAG runs retrieved, in retrieval order,
  replayed from their checkpoints (deterministic; no LLM calls).

Only the graph methods make API calls (Jev). Per-method rows are checkpointed, so an
interrupted run resumes. Results go to results/<stamp>-<dataset>-evidence/.
"""

import argparse
import asyncio
import json
import random
import subprocess
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from text_qa import EMBED, TYPES, load  # scripts/eval sibling

from graphwalk.cli import _make_backend  # pyright: ignore[reportPrivateUsage]
from graphwalk.core.hashing import content_hash
from graphwalk.core.model import StoredDocument
from graphwalk.embeddings.fastembed_embedder import FastEmbedEmbedder
from graphwalk.eval.datasets import fanoutqa
from graphwalk.eval.datasets.cache import cache_dir
from graphwalk.eval.evidence import (
    EvidenceQuestion,
    EvidenceRow,
    gold_titles,
    paired_table,
    recall_table,
    retrieval_rows,
    type_table,
)
from graphwalk.eval.ingest_eval import documents
from graphwalk.eval.runner import SystemRun, environment
from graphwalk.eval.suite import preset_config
from graphwalk.eval.systems import LIST_PROMPTS, TEXT_PROMPTS, IterativeRAGSystem
from graphwalk.eval.text_qa import TEXT_ITER_RAG, build_text_systems, graph_key, stratified
from graphwalk.ingest import IngestConfig
from graphwalk.ingest.pipeline import provenance_id
from graphwalk.llm.base import LLMResult, Message
from graphwalk.stores.networkx_store import NetworkXStore

M7_INGEST = IngestConfig(routing="jev", escalate=True, evidence=False)
SETTINGS: dict[str, dict[str, Any]] = {
    # preset, ks, the k for CIs, RAG k and max docs of the M7 multi-step run
    "2wiki": {"preset": "greedy-v2", "ks": (2, 5, 10), "ci_k": 5, "rag_k": 5, "max_docs": 40},
    "hotpotqa": {"preset": "greedy-v2", "ks": (2, 5, 10), "ci_k": 5, "rag_k": 5, "max_docs": 40},
    "fanoutqa": {"preset": "relation-v2", "ks": (5, 10, 20), "ci_k": 10, "rag_k": 10,
                 "max_docs": 20},
}  # fmt: skip
PAIRS = (
    ("hybrid", "dense"), ("hybrid-choice", "dense"), ("graph-choice", "dense"),
    ("graph-choice", "graph"), ("graph-choice", "title+dense"),
    ("hybrid-choice", "title+dense"), ("multi-step", "hybrid-choice"),
)  # fmt: skip


class _NoLLM:
    """The multi-step system is only replayed, never asked."""

    model_id = "none"

    async def complete(self, messages: Sequence[Message]) -> LLMResult:
        raise NotImplementedError


def records_for(dataset: str, args: argparse.Namespace) -> list[Mapping[str, Any]]:
    if dataset == "fanoutqa":  # as scripts/eval/fanout.py samples
        questions = fanoutqa.read_questions(fanoutqa.download())
        rng = random.Random(args.seed)  # noqa: S311 - reproducible sampling
        chosen = sorted(rng.sample(range(len(questions)), args.fanout_n))
        sample = [questions[i] for i in chosen]
        return fanoutqa.records(sample, fanoutqa.load_corpus(), args.max_chars)
    per_type = args.per_type or (30 if dataset == "2wiki" else 60)
    return stratified(load(dataset), TYPES[dataset], per_type, args.seed)


def evidence_questions(
    dataset: str, records: Sequence[Mapping[str, Any]]
) -> list[EvidenceQuestion]:
    source_id = f"{dataset}:paragraphs"
    return [
        EvidenceQuestion(
            id=f"{dataset}-{r['_id']}", text=str(r["question"]), type=str(r.get("type", "fanout")),
            gold=tuple(provenance_id(source_id, t) for t in gold_titles(r)),
        )
        for r in records
    ]  # fmt: skip


async def attach_documents(
    store: NetworkXStore, dataset: str, records: Sequence[Any]
) -> dict[str, str]:
    """The M7 graphs predate stored documents: add the pooled text under the keys its
    provenance already uses. Returns title -> key."""
    source_id = f"{dataset}:paragraphs"
    now = datetime.now(UTC)
    titles: dict[str, str] = {}
    for doc in documents(records):
        key = provenance_id(source_id, doc.doc_id)
        titles[doc.title or doc.doc_id] = key
        await store.put_document(
            StoredDocument(
                key=key, source_id=source_id, doc_id=doc.doc_id, title=doc.title,
                text=doc.text, text_hash=content_hash(doc.text), length=len(doc.text),
                ingested_at=now,
            )
        )  # fmt: skip
    return titles


def iter_checkpoint(dataset: str, stem: str) -> Path:
    base = cache_dir().parent / "checkpoints"
    if dataset == "fanoutqa":
        return base / f"fanout-{stem}" / f"{TEXT_ITER_RAG}.json"
    matches = sorted(base.glob(f"{stem}-*/{TEXT_ITER_RAG}.json"))
    if len(matches) != 1:
        msg = f"expected one M7 multi-step checkpoint for {stem}, found {matches}"
        raise SystemExit(msg)
    return matches[0]


async def multi_step_rows(
    dataset: str, store: NetworkXStore, records: Sequence[Any], *,
    questions: Sequence[EvidenceQuestion], embedder: FastEmbedEmbedder, checkpoint: Path,
) -> list[EvidenceRow]:  # fmt: skip
    """The documents M7's multi-step RAG retrieved, replayed from its recorded searches
    (retrieval is deterministic); checked against the recorded document counts."""
    settings = SETTINGS[dataset]
    (system,) = await build_text_systems(
        [TEXT_ITER_RAG], store, records, decider=None, embedder=embedder,  # pyright: ignore[reportArgumentType]
        llm=_NoLLM(), config=preset_config(settings["preset"]), rag_k=settings["rag_k"],
        iter_max_docs=settings["max_docs"],
        prompts=LIST_PROMPTS if dataset == "fanoutqa" else TEXT_PROMPTS,
    )  # fmt: skip
    assert isinstance(system, IterativeRAGSystem)  # noqa: S101
    text = await asyncio.to_thread(checkpoint.read_text, encoding="utf-8")
    recorded = {r.question.id: r.answer for r in SystemRun.model_validate_json(text).records}
    source_id = f"{dataset}:paragraphs"
    rows: list[EvidenceRow] = []
    for question in questions:
        answer = recorded[question.id]
        steps = answer.detail.get("steps")
        titles = await system.replay(question.text, steps if isinstance(steps, list) else [])  # pyright: ignore[reportArgumentType]
        if answer.status == "ok" and len(titles) != answer.detail.get("docs"):
            recorded_docs = answer.detail.get("docs")
            msg = f"replay of {question.id}: {len(titles)} docs, M7 recorded {recorded_docs}"
            raise SystemExit(msg)
        rows.append(
            EvidenceRow(
                question_id=question.id, type=question.type, method="multi-step",
                gold=question.gold, ranked=tuple(provenance_id(source_id, t) for t in titles),
                cost_usd=answer.cost_usd, latency_s=answer.latency_s, error=answer.error,
            )
        )  # fmt: skip
    return rows


def summary_text(dataset: str, rows: Sequence[EvidenceRow], params: dict[str, Any]) -> str:
    settings = SETTINGS[dataset]
    ks, ci_k = settings["ks"], settings["ci_k"]
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,  # noqa: S607
                         check=False).stdout.strip()  # fmt: skip
    multi = [r for r in rows if r.method == "multi-step"]
    lines = [
        f"# {dataset} evidence retrieval (E1)", "",
        f"- params: `{json.dumps(params, sort_keys=True)}`",
        f"- git: `{sha}`",
        f"- environment: `{json.dumps(environment(), sort_keys=True)}`",
        f"- run at: {datetime.now(UTC).isoformat()}", "",
        "## Recall of gold evidence documents", "",
        recall_table(rows, ks, ci_k), "",
    ]  # fmt: skip
    if multi:
        recall = sum(sum(g in set(r.ranked) for g in r.gold) / len(r.gold) for r in multi)
        docs = sum(len(r.ranked) for r in multi) / len(multi)
        lines += [
            f"multi-step, over everything it read (mean {docs:.1f} docs): recall "
            f"{recall / len(multi):.3f}. Its cost is the M7 run's (LLM calls, including "
            "answering).", "",
        ]  # fmt: skip
    lines += [
        "## Paired differences (recall@k, 95% bootstrap CI)", "",
        paired_table(rows, PAIRS, ks), "",
        f"## recall@{ci_k} by question type", "",
        type_table(rows, ci_k), "",
    ]  # fmt: skip
    errors = Counter(r.method for r in rows if r.error)
    if errors:
        lines += [f"Errors (scored as empty): {dict(errors)}", ""]
    return "\n".join(lines)


async def main(args: argparse.Namespace) -> None:
    dataset: str = args.dataset
    settings = SETTINGS[dataset]
    records = records_for(dataset, args)
    graphs = cache_dir().parent / "graphs"
    graph_path = graphs / f"{graph_key(dataset, records, M7_INGEST)}.graph.json"
    if not graph_path.exists():
        msg = f"{graph_path} missing: build the M7 graph first"
        raise SystemExit(msg)
    questions = evidence_questions(dataset, records)
    print(f"{dataset}: {len(questions)} questions, graph {graph_path.name}", flush=True)  # noqa: T201
    store = await NetworkXStore.load(graph_path)
    titles = await attach_documents(store, dataset, records)
    embedder = FastEmbedEmbedder(EMBED)
    decider = _make_backend()

    def progress(method: str, done: int, total: int) -> None:
        if done % 20 == 0 or done == total:
            print(f"  {method}: {done}/{total}", flush=True)  # noqa: T201

    try:
        rows = await retrieval_rows(
            store, questions, titles, decider=decider, embedder=embedder,
            config=preset_config(settings["preset"]), ks=settings["ks"],
            concurrency=args.concurrency, on_progress=progress,
            checkpoints=cache_dir().parent / "checkpoints" / f"evidence-{graph_path.stem}",
        )  # fmt: skip
    finally:
        await decider.aclose()
    checkpoint = iter_checkpoint(dataset, graph_path.stem)
    rows += await multi_step_rows(
        dataset, store, records, questions=questions, embedder=embedder, checkpoint=checkpoint
    )
    params = {
        "dataset": dataset, "n": len(questions), "seed": args.seed, "ks": list(settings["ks"]),
        "preset": settings["preset"], "embedder": EMBED, "graph": graph_path.name,
        "decision_model": decider.model_id, "multi_step_run": checkpoint.parent.name,
    }  # fmt: skip
    summary = summary_text(dataset, rows, params)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = Path("results") / f"{stamp}-{dataset}-evidence"
    out.mkdir(parents=True)
    (out / "summary.md").write_text(summary, encoding="utf-8")
    (out / "rows.jsonl").write_text(
        "".join(r.model_dump_json() + "\n" for r in rows), encoding="utf-8"
    )
    print(summary)  # noqa: T201


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=sorted(SETTINGS))
    parser.add_argument("--per-type", type=int, default=None, help="default: M7's (30 / 60)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fanout-n", type=int, default=40)
    parser.add_argument("--max-chars", type=int, default=8000)
    parser.add_argument("--concurrency", type=int, default=8)
    asyncio.run(main(parser.parse_args()))
