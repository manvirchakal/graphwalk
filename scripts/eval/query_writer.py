"""E2: an LLM writes the graph query (a relation path) on MetaQA; code executes it.

    # prompt check on dev (a handful of questions), then the report run on test:
    uv run python scripts/eval/query_writer.py --split dev --n 20
    uv run python scripts/eval/query_writer.py --split test --n 200

Same seeded test subsets (seed 0, n = 200 per hop count) and given start entities as
the M5 graphwalk runs, so the results compare directly with docs/results-m5.md.
Systems: `llm-path` (one attempt) and `llm-path-retry` (up to two retries when the
path is invalid or returns nothing). See graphwalk.eval.query_writer.
"""

import argparse
import asyncio
from datetime import UTC, datetime
from pathlib import Path

from graphwalk.config import GraphwalkSettings
from graphwalk.eval.datasets import metaqa
from graphwalk.eval.query_writer import PathQuerySystem
from graphwalk.eval.runner import run_system, sample_questions, write_results
from graphwalk.eval.suite import load_dataset
from graphwalk.llm.litellm_backend import LiteLLMBackend


async def main(args: argparse.Namespace) -> None:
    key = GraphwalkSettings().openrouter_api_key
    llm = LiteLLMBackend(
        args.llm,
        api_key=None if key is None else key.get_secret_value(),
        max_tokens=args.max_tokens,
        max_rpm=args.rpm,
    )
    relations = {name: ("film", obj) for name, obj in metaqa.OBJECT_TYPES.items()}
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    for hops in args.hops:
        store, questions, _ = await load_dataset("metaqa", hops, args.split)
        subset = sample_questions(questions, args.n, args.seed)
        systems = [
            PathQuerySystem(store, llm, relations, glosses=metaqa.RELATION_GLOSSES,
                            retries=retries, name=f"llm-path{suffix}{args.label}")
            for retries, suffix in ((0, ""), (2, "-retry"))
        ]  # fmt: skip
        runs = []
        for system in systems:

            def progress(done: int, total: int, name: str = f"{hops}-hop {system.name}") -> None:
                if done == total or done % 50 == 0:
                    print(f"  {name}: {done}/{total}", flush=True)  # noqa: T201

            runs.append(
                await run_system(system, subset, concurrency=args.concurrency, on_done=progress)
            )
        label = f"metaqa-{hops}hop-llmpath" + ("" if args.split == "test" else f"-{args.split}")
        root = Path("results") if args.split == "test" else Path("results") / "tuning"
        out = write_results(
            root / f"{stamp}-{label}",
            dataset=label,
            runs=runs,
            params={
                "dataset": "metaqa", "hops": hops, "n": len(subset), "seed": args.seed,
                "split": args.split, "linking": "given", "llm": args.llm,
                "total_questions": len(questions),
            },
        )  # fmt: skip
        print((out / "summary.md").read_text(encoding="utf-8"))  # noqa: T201


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hops", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--split", choices=["dev", "test"], default="test")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--llm", default="openrouter/openai/gpt-6-luna")
    parser.add_argument("--label", default="", help="appended to system names")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--rpm", type=float, default=18.0)
    parser.add_argument("--concurrency", type=int, default=4)
    asyncio.run(main(parser.parse_args()))
