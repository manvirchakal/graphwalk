"""E2b: the LLM-written query on a large, noisy schema (graphs extracted from text).

    uv run python scripts/eval/query_writer_text.py 2wiki
    uv run python scripts/eval/query_writer_text.py hotpotqa

E2 found an LLM writing the relation path matches graphwalk on MetaQA (9 clean
relations). This repeats it where the schema is the M7 graph extracted from text:
1,476 (2Wiki) or 2,545 (HotpotQA) free-form relation types. The LLM sees the whole
schema (each relation's most common subject/object types and edge count) and starts
from the entity the M7 graphwalk run linked with the same Jev linker. Compare with the
M7 `graphwalk` (graph-only) rows on the same questions.
"""

import argparse
import asyncio
from datetime import UTC, datetime
from pathlib import Path

from text_qa import TYPES, load  # scripts/eval sibling

from graphwalk.cli import _make_backend  # pyright: ignore[reportPrivateUsage]
from graphwalk.config import GraphwalkSettings
from graphwalk.eval.datasets.cache import cache_dir
from graphwalk.eval.query_writer import PathQuerySystem, schema_from_store
from graphwalk.eval.runner import run_system, write_results
from graphwalk.eval.text_qa import graph_key, stratified, to_questions
from graphwalk.ingest import IngestConfig
from graphwalk.llm.litellm_backend import LiteLLMBackend
from graphwalk.stores.networkx_store import NetworkXStore
from graphwalk.traversal import ChoiceEntryResolver

M7_INGEST = IngestConfig(routing="jev", escalate=True, evidence=False)


async def main(args: argparse.Namespace) -> None:
    per_type = args.per_type or (30 if args.dataset == "2wiki" else 60)
    records = stratified(load(args.dataset), TYPES[args.dataset], per_type, args.seed)
    graphs = cache_dir().parent / "graphs"
    store = await NetworkXStore.load(
        graphs / f"{graph_key(args.dataset, records, M7_INGEST)}.graph.json"
    )
    relations, counts = await schema_from_store(store)
    key = GraphwalkSettings().openrouter_api_key
    llm = LiteLLMBackend(
        args.llm,
        api_key=None if key is None else key.get_secret_value(),
        max_tokens=args.max_tokens,
        max_rpm=args.rpm,
    )
    decider = _make_backend()
    try:
        system = PathQuerySystem(
            store, llm, relations, counts=counts, resolver=ChoiceEntryResolver(store, decider),
            retries=2, name="llm-path-retry",
        )  # fmt: skip

        def progress(done: int, total: int) -> None:
            if done == total or done % 20 == 0:
                print(f"  llm-path-retry: {done}/{total}", flush=True)  # noqa: T201

        run = await run_system(
            system, to_questions(records, args.dataset), concurrency=args.concurrency,
            on_done=progress,
        )  # fmt: skip
    finally:
        await decider.aclose()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = write_results(
        Path("results") / f"{stamp}-{args.dataset}-llmpath",
        dataset=f"{args.dataset}-text",
        runs=[run],
        params={
            "dataset": args.dataset, "per_type": per_type, "seed": args.seed,
            "n": len(records), "llm": args.llm, "relations": len(relations),
            "linking": "choice",
        },
    )  # fmt: skip
    print((out / "summary.md").read_text(encoding="utf-8"))  # noqa: T201


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=sorted(TYPES))
    parser.add_argument("--per-type", type=int, default=None, help="default: M7's (30 / 60)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--llm", default="openrouter/openai/gpt-6-luna")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--rpm", type=float, default=18.0)
    parser.add_argument("--concurrency", type=int, default=4)
    asyncio.run(main(parser.parse_args()))
