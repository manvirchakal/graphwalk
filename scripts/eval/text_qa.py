"""M7: QA over graphs ingested from text, vs. RAG over the same paragraphs.

    uv run python scripts/eval/text_qa.py 2wiki --per-type 30
    uv run python scripts/eval/text_qa.py hotpotqa --per-type 60

The ingested graph is cached (~/.cache/graphwalk/graphs/), as are extractions, so a
re-run pays only for answering. Results go to results/<stamp>-<dataset>-text/.
"""

import argparse
import asyncio
import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path

from graphwalk.cli import _make_backend  # pyright: ignore[reportPrivateUsage]
from graphwalk.config import GraphwalkSettings
from graphwalk.embeddings.fastembed_embedder import FastEmbedEmbedder
from graphwalk.eval.datasets import hotpotqa, twowiki
from graphwalk.eval.datasets.cache import cache_dir
from graphwalk.eval.suite import preset_config
from graphwalk.eval.text_qa import (
    TEXT_SYSTEMS,
    build_graph,
    build_text_systems,
    graph_key,
    run_text_qa,
    stratified,
)
from graphwalk.ingest import IngestConfig, JsonFileCache
from graphwalk.llm.litellm_backend import LiteLLMBackend

TYPES = {
    "2wiki": ("compositional", "inference", "comparison", "bridge_comparison"),
    "hotpotqa": hotpotqa.TYPES,
}
EMBED = "BAAI/bge-small-en-v1.5"
PRESET = "greedy-v2"


def load(dataset: str) -> list[dict[str, object]]:
    if dataset == "2wiki":
        return twowiki.read_records(twowiki.download())
    return hotpotqa.read_records(hotpotqa.download())


async def main(args: argparse.Namespace) -> None:
    records = stratified(load(args.dataset), TYPES[args.dataset], args.per_type, args.seed)
    key = GraphwalkSettings().openrouter_api_key
    api_key = None if key is None else key.get_secret_value()

    def llm(model: str, max_tokens: int) -> LiteLLMBackend:
        return LiteLLMBackend(model, api_key=api_key, max_tokens=max_tokens, max_rpm=args.rpm)

    embedder = FastEmbedEmbedder(EMBED)
    decider = _make_backend()
    config = IngestConfig(routing="jev", escalate=args.escalation_model is not None)
    cache_path = cache_dir() / "extractions" / (re.sub(r"[^\w.-]+", "_", args.llm) + ".json")
    cache = JsonFileCache(cache_path)
    graph_path = (
        cache_dir().parent / "graphs" / f"{graph_key(args.dataset, records, config)}.graph.json"
    )
    try:
        store, report = await build_graph(
            args.dataset,
            records,
            llm=llm(args.llm, 4096),
            decider=decider,
            embedder=embedder,
            cache=cache,
            config=config,
            escalation_llm=(
                None if args.escalation_model is None else llm(args.escalation_model, 4096)
            ),
            graph_path=graph_path,
        )
        cache.flush()
        nodes, edges = await store.counts()
        print(f"graph: {nodes} nodes, {edges} edges ({graph_path})", flush=True)  # noqa: T201
        systems = await build_text_systems(
            args.systems,
            store,
            records,
            decider=decider,
            embedder=embedder,
            llm=llm(args.llm, 512),
            config=preset_config(PRESET),
            rag_k=args.rag_k,
        )

        def progress(name: str, done: int, total: int) -> None:
            if done == total or done % 20 == 0:
                print(f"  {name}: {done}/{total}", flush=True)  # noqa: T201

        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        run_params = {
            "llm": args.llm, "rag_k": args.rag_k, "preset": PRESET, "systems": args.systems,
        }  # fmt: skip
        tag = hashlib.sha256(json.dumps(run_params, sort_keys=True).encode()).hexdigest()[:8]
        checkpoints = cache_dir().parent / "checkpoints" / f"{graph_path.stem}-{tag}"
        out, _ = await run_text_qa(
            args.dataset,
            records,
            systems,
            out_dir=Path("results") / f"{stamp}-{args.dataset}-text",
            params={
                "dataset": args.dataset,
                "per_type": args.per_type,
                "seed": args.seed,
                "n": len(records),
                "llm": args.llm,
                "escalation_llm": args.escalation_model,
                "embedder": EMBED,
                "preset": PRESET,
                "rag_k": args.rag_k,
                "graph": graph_path.name,
                "ingest": config.model_dump(mode="json"),
            },
            ingest=report,
            concurrency=args.concurrency,
            on_progress=progress,
            checkpoint_dir=checkpoints,
        )
    finally:
        cache.flush()
        await decider.aclose()
    print((out / "summary.md").read_text(encoding="utf-8"))  # noqa: T201


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=sorted(TYPES))
    parser.add_argument("--per-type", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--systems", nargs="+", default=list(TEXT_SYSTEMS))
    parser.add_argument("--llm", default="openrouter/openai/gpt-6-luna")
    parser.add_argument("--escalation-model", default="openrouter/xiaomi/mimo-v2.6-pro")
    parser.add_argument(
        "--rpm", type=float, default=18.0, help="OpenRouter new accounts: 20 req/min per model"
    )
    parser.add_argument("--rag-k", type=int, default=5)
    parser.add_argument("--concurrency", type=int, default=8)
    asyncio.run(main(parser.parse_args()))
