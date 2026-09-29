"""Step 2: FanOutQA, where answers are sets spread over many Wikipedia pages.

    uv run python scripts/eval/fanout.py --n 40

Evidence pages (truncated to --max-chars) are pooled, ingested into one graph, and
indexed page-by-page for RAG, as in scripts/eval/text_qa.py. Systems:

* graphwalk-reader-facts: relation-mode walks (a hop takes *all* targets of a
  relation), reader reads the reached nodes' extracted facts;
* graphwalk-reader-source: the same walks, reader reads the reached pages;
* text-rag (top-k pages) and text-iter-rag (multi-step, searches by title).

Scored with FanOutQA's accuracy (share of reference strings, keys and values, found in
the answer; strict = all found), without its lemmatizer.
"""

import argparse
import asyncio
import json
import math
import random
import re
from datetime import UTC, datetime
from pathlib import Path

from graphwalk.cli import _make_backend  # pyright: ignore[reportPrivateUsage]
from graphwalk.config import GraphwalkSettings
from graphwalk.embeddings.fastembed_embedder import FastEmbedEmbedder
from graphwalk.eval.datasets import fanoutqa
from graphwalk.eval.datasets.cache import cache_dir
from graphwalk.eval.metrics import percentile
from graphwalk.eval.suite import preset_config
from graphwalk.eval.systems import LIST_PROMPTS
from graphwalk.eval.text_qa import (
    READER,
    TEXT_ITER_RAG,
    TEXT_RAG,
    build_graph,
    build_text_systems,
    graph_key,
    run_text_qa,
    source_documents,
)
from graphwalk.ingest import IngestConfig, JsonFileCache
from graphwalk.llm.litellm_backend import LiteLLMBackend

EMBED = "BAAI/bge-small-en-v1.5"
PRESET = "relation-v2"
INGEST = IngestConfig(routing="jev", escalate=True)


def score_table(runs, reference) -> str:
    rows = [
        "| system | loose acc | strict acc | $/1k q | p50 s | LLM calls/q | Jev calls/q |",
        "|---|---|---|---|---|---|---|",
    ]
    for run in runs:
        loose, strict, costs, lat = [], [], [], []
        for record in run.records:
            answer = record.answer
            text = " | ".join(answer.answers)
            lo, st = fanoutqa.answer_in_text(reference[record.question.id], text)
            loose.append(lo)
            strict.append(st)
            costs.append(answer.cost_usd)
            lat.append(answer.latency_s)
        known = [c for c in costs if c is not None]
        cost = "n/a" if len(known) < len(costs) else f"{1000 * math.fsum(known) / len(known):.3f}"
        n = len(loose)
        rows.append(
            f"| {run.system} | {sum(loose) / n:.3f} | {sum(strict) / n:.3f} | {cost} "
            f"| {percentile(lat, 50):.2f} | {run.summary.llm_calls_per_query:.2f} "
            f"| {run.summary.decision_calls_per_query:.2f} |"
        )
    return "\n".join(rows)


async def main(args: argparse.Namespace) -> None:
    questions = fanoutqa.read_questions(fanoutqa.download())
    rng = random.Random(args.seed)  # noqa: S311 - reproducible sampling
    chosen = sorted(rng.sample(range(len(questions)), args.n))
    sample = [questions[i] for i in chosen]
    records = fanoutqa.records(sample, fanoutqa.load_corpus(), args.max_chars)
    reference = {f"fanoutqa-{r['_id']}": r["answer"] for r in records}
    ceiling = [
        fanoutqa.answer_in_text(r["answer"], "\n".join(c[1][0] for c in r["context"]))[0]
        for r in records
    ]
    key = GraphwalkSettings().openrouter_api_key
    api_key = None if key is None else key.get_secret_value()

    def llm(model: str, max_tokens: int) -> LiteLLMBackend:
        return LiteLLMBackend(model, api_key=api_key, max_tokens=max_tokens, max_rpm=args.rpm)

    embedder = FastEmbedEmbedder(EMBED)
    decider = _make_backend()
    cache = JsonFileCache(
        cache_dir() / "extractions" / (re.sub(r"[^\w.-]+", "_", args.llm) + ".json")
    )
    graphs = cache_dir().parent / "graphs"
    graph_path = graphs / f"{graph_key('fanoutqa', records, INGEST)}.graph.json"
    try:
        store, report = await build_graph(
            "fanoutqa",
            records,
            llm=llm(args.llm, 4096),
            decider=decider,
            embedder=embedder,
            cache=cache,
            config=INGEST,
            escalation_llm=llm(args.escalation_model, 4096),
            graph_path=graph_path,
        )
        cache.flush()
        nodes, edges = await store.counts()
        print(f"graph: {nodes} nodes, {edges} edges ({graph_path})", flush=True)  # noqa: T201
        reader_llm = llm(args.llm, 1024)
        common = {"decider": decider, "embedder": embedder, "llm": reader_llm,
                  "config": preset_config(PRESET)}  # fmt: skip
        reader = {"instructions": LIST_PROMPTS.instructions, "max_answers": None,
                  "max_path_names": 25, "max_nodes": 40}  # fmt: skip
        systems = [
            *await build_text_systems(
                [READER], store, records, label="-facts", reader_options=reader, **common
            ),
            *await build_text_systems(
                [READER],
                store,
                records,
                label="-source",
                documents=source_documents("fanoutqa", records),
                reader_options={**reader, "max_docs": args.max_docs},
                **common,
            ),
            *await build_text_systems(
                [TEXT_RAG, TEXT_ITER_RAG],
                store,
                records,
                prompts=LIST_PROMPTS,
                rag_k=args.rag_k,
                iter_max_docs=args.max_docs,
                **common,
            ),
        ]

        def progress(name: str, done: int, total: int) -> None:
            if done == total or done % 10 == 0:
                print(f"  {name}: {done}/{total}", flush=True)  # noqa: T201

        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        out, runs = await run_text_qa(
            "fanoutqa",
            records,
            systems,
            out_dir=Path("results") / f"{stamp}-fanoutqa",
            params={
                "n": args.n,
                "seed": args.seed,
                "max_chars": args.max_chars,
                "max_docs": args.max_docs,
                "rag_k": args.rag_k,
                "llm": args.llm,
                "escalation_llm": args.escalation_model,
                "embedder": EMBED,
                "preset": PRESET,
                "graph": graph_path.name,
                "dev_sha256": fanoutqa.dev_sha256(fanoutqa.download()),
                "corpus": f"{fanoutqa.CORPUS_DATASET}@{fanoutqa.CORPUS_REVISION}",
            },
            ingest=report,
            concurrency=args.concurrency,
            on_progress=progress,
            checkpoint_dir=cache_dir().parent / "checkpoints" / f"fanout-{graph_path.stem}",
        )
    finally:
        cache.flush()
        await decider.aclose()
    table = score_table(runs, reference)
    extra = (
        "\n## FanOutQA accuracy\n\n"
        f"Ceiling (reference strings present in the truncated evidence): "
        f"{sum(ceiling) / len(ceiling):.3f} loose.\n\n{table}\n"
    )
    with (out / "summary.md").open("a", encoding="utf-8") as fh:
        fh.write(extra)
    (out / "reference.json").write_text(json.dumps(reference, ensure_ascii=False), "utf-8")
    print((out / "summary.md").read_text(encoding="utf-8"))  # noqa: T201


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-chars", type=int, default=8000)
    parser.add_argument("--max-docs", type=int, default=20)
    parser.add_argument("--rag-k", type=int, default=10)
    parser.add_argument("--llm", default="openrouter/openai/gpt-6-luna")
    parser.add_argument("--escalation-model", default="openrouter/xiaomi/mimo-v2.6-pro")
    parser.add_argument("--rpm", type=float, default=18.0)
    parser.add_argument("--concurrency", type=int, default=8)
    asyncio.run(main(parser.parse_args()))
