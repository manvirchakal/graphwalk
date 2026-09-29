"""M7 follow-up: walk-then-read-source, and a fixed relation schema.

    uv run python scripts/eval/walk_read.py 2wiki --per-type 30
    uv run python scripts/eval/walk_read.py hotpotqa --per-type 60

Reuses the M7 question samples and cached ingested graphs (run scripts/eval/text_qa.py
first). Normalizes each graph onto the fixed relation schema (cached next to it), then
runs:

* graphwalk-schema: graph-only walking on the normalized graph;
* graphwalk-reader-source: walks on the original graph, reader reads source paragraphs;
* graphwalk-reader-source-schema: the same on the normalized graph.

Finished systems are checkpointed, so an interrupted run resumes.
"""

import argparse
import asyncio
import json
import re
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from text_qa import EMBED, PRESET, TYPES, load  # scripts/ sibling

from graphwalk.cli import _make_backend  # pyright: ignore[reportPrivateUsage]
from graphwalk.config import GraphwalkSettings
from graphwalk.decisions import DecisionBackend
from graphwalk.embeddings.fastembed_embedder import FastEmbedEmbedder
from graphwalk.eval.datasets.cache import cache_dir
from graphwalk.eval.suite import preset_config
from graphwalk.eval.text_qa import (
    GRAPHWALK,
    READER,
    build_graph,
    build_text_systems,
    graph_key,
    run_text_qa,
    source_documents,
    stratified,
)
from graphwalk.ingest import IngestConfig, JsonFileCache
from graphwalk.ingest.normalize import normalize_graph
from graphwalk.llm.litellm_backend import LiteLLMBackend
from graphwalk.stores.networkx_store import NetworkXStore

M7_INGEST = IngestConfig(routing="jev", escalate=True, evidence=False)
"""The config scripts/eval/text_qa.py ingested with (it names the cached graph)."""


async def normalized(base: Path, store: NetworkXStore, decider: DecisionBackend) -> NetworkXStore:
    path = base.with_name(base.name.replace(".graph.json", ".schema.graph.json"))
    if path.exists():
        return await NetworkXStore.load(path)
    report = await normalize_graph(store, decider)
    await store.save(path)
    summary = {k: v for k, v in asdict(report).items() if k != "mapping"}
    path.with_suffix(".report.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"normalized: {summary}", flush=True)  # noqa: T201
    return store


async def main(args: argparse.Namespace) -> None:
    records = stratified(load(args.dataset), TYPES[args.dataset], args.per_type, args.seed)
    graph_path = (
        cache_dir().parent / "graphs" / f"{graph_key(args.dataset, records, M7_INGEST)}.graph.json"
    )
    if not graph_path.exists():
        msg = f"{graph_path} missing: run scripts/eval/text_qa.py {args.dataset} first"
        raise SystemExit(msg)
    key = GraphwalkSettings().openrouter_api_key
    llm = LiteLLMBackend(
        args.llm,
        api_key=None if key is None else key.get_secret_value(),
        max_tokens=512,
        max_rpm=args.rpm,
    )
    embedder = FastEmbedEmbedder(EMBED)
    decider = _make_backend()
    cache = JsonFileCache(
        cache_dir() / "extractions" / (re.sub(r"[^\w.-]+", "_", args.llm) + ".json")
    )
    try:
        base, _ = await build_graph(
            args.dataset, records, llm=llm, decider=decider, embedder=embedder, cache=cache,
            config=M7_INGEST, graph_path=graph_path,
        )  # fmt: skip
        schema = await normalized(graph_path, await NetworkXStore.load(graph_path), decider)
        docs = source_documents(args.dataset, records)
        common = {"decider": decider, "embedder": embedder, "llm": llm,
                  "config": preset_config(PRESET)}  # fmt: skip
        systems = [
            *await build_text_systems([GRAPHWALK], schema, records, label="-schema", **common),
            *await build_text_systems(
                [READER], base, records, documents=docs, label="-source", **common
            ),
            *await build_text_systems(
                [READER], schema, records, documents=docs, label="-source-schema", **common
            ),
        ]

        def progress(name: str, done: int, total: int) -> None:
            if done == total or done % 20 == 0:
                print(f"  {name}: {done}/{total}", flush=True)  # noqa: T201

        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        out, _ = await run_text_qa(
            args.dataset,
            records,
            systems,
            out_dir=Path("results") / f"{stamp}-{args.dataset}-walkread",
            params={
                "dataset": args.dataset,
                "per_type": args.per_type,
                "seed": args.seed,
                "n": len(records),
                "llm": args.llm,
                "embedder": EMBED,
                "preset": PRESET,
                "graph": graph_path.name,
                "max_docs": 5,
            },
            ingest=None,
            concurrency=args.concurrency,
            on_progress=progress,
            checkpoint_dir=cache_dir().parent / "checkpoints" / f"walkread-{graph_path.stem}",
        )
    finally:
        await decider.aclose()
    print((out / "summary.md").read_text(encoding="utf-8"))  # noqa: T201


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=sorted(TYPES))
    parser.add_argument("--per-type", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--llm", default="openrouter/openai/gpt-6-luna")
    parser.add_argument("--rpm", type=float, default=18.0)
    parser.add_argument("--concurrency", type=int, default=8)
    asyncio.run(main(parser.parse_args()))
