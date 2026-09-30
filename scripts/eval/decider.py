"""E3: Jev vs an LLM as the decider. Same traversal, only the decision backend swapped.

    uv run python scripts/eval/decider.py --seed 0

Runs on the M5 test subsets (seed 0 by default): MetaQA 1/2/3-hop (n = 200 each,
relation-v2, given start entities) and 2Wiki's pooled gold-evidence graph (n = 300,
greedy-v2, gold start entities). Both deciders run in the same invocation, so they see
the same code and questions. The LLM decider (graphwalk.decisions.llm_decider) asks the
chat model to score every option 0-100 and normalizes the scores into a distribution.

Also reports the calibration of each walk's confidence (graphwalk.eval.calibration).
Other seeds (--seed 1, 2) are E4's re-samples of the same test sets.
"""

import argparse
import asyncio
from pathlib import Path

from graphwalk.cli import _make_backend  # pyright: ignore[reportPrivateUsage]
from graphwalk.config import GraphwalkSettings
from graphwalk.decisions.llm_decider import LLMDecider
from graphwalk.embeddings.fastembed_embedder import FastEmbedEmbedder
from graphwalk.eval.calibration import CALIBRATION_HEADER, calibration_row
from graphwalk.eval.suite import Factories, run_dataset
from graphwalk.llm.litellm_backend import LiteLLMBackend

EMBED = "BAAI/bge-small-en-v1.5"
RUNS = (("metaqa", 1, "relation-v2", 200), ("metaqa", 2, "relation-v2", 200),
        ("metaqa", 3, "relation-v2", 200), ("2wiki", 0, "greedy-v2", 300))  # fmt: skip


async def main(args: argparse.Namespace) -> None:
    key = GraphwalkSettings().openrouter_api_key
    api_key = None if key is None else key.get_secret_value()

    def llm(max_tokens: int) -> LiteLLMBackend:
        return LiteLLMBackend(args.llm, api_key=api_key, max_tokens=max_tokens, max_rpm=args.rpm)

    embedder = FastEmbedEmbedder(EMBED)
    deciders = {
        "jev": Factories(decider=_make_backend, embedder=lambda: embedder, llm=lambda: llm(512)),
        "llm": Factories(
            decider=lambda: LLMDecider(llm(args.max_tokens)),
            embedder=lambda: embedder,
            llm=lambda: llm(512),
        ),
    }
    selected = [r for r in RUNS if f"{r[0]}-{r[1]}" in args.only or not args.only]
    for dataset, hops, preset, n in selected:
        for label in args.deciders:
            factories = deciders[label]

            def progress(name: str, done: int, total: int) -> None:
                if done == total or done % 50 == 0:
                    print(f"  {name}: {done}/{total}", flush=True)  # noqa: T201

            out, runs = await run_dataset(
                dataset,  # pyright: ignore[reportArgumentType]
                [preset],
                factories,
                hops=max(hops, 1),
                n=n,
                seed=args.seed,
                concurrency=args.concurrency,
                out_root=Path("results") / "decider",
                on_progress=progress,
                variants=((f"-{label}", {}),),
            )
            table = "\n".join(
                [CALIBRATION_HEADER, *(calibration_row(r.system, r.records) for r in runs)]
            )
            with (out / "summary.md").open("a", encoding="utf-8") as fh:
                fh.write(f"\n## Calibration of walk confidence\n\n{table}\n")
            print((out / "summary.md").read_text(encoding="utf-8")[:1500], flush=True)  # noqa: T201


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deciders", nargs="+", default=["jev", "llm"], choices=["jev", "llm"])
    parser.add_argument("--only", nargs="*", default=[], help="e.g. metaqa-3 2wiki-0")
    parser.add_argument("--llm", default="openrouter/openai/gpt-6-luna")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--rpm", type=float, default=18.0)
    parser.add_argument("--concurrency", type=int, default=4)
    asyncio.run(main(parser.parse_args()))
