"""M5 tuning round: ablate prompt/decision knobs on MetaQA **dev** (never test).

    uv run python scripts/eval/tune_metaqa_dev.py [--n 200] [--preset relation]

Each variant runs the same seeded dev subset per hop count; summaries go to
results/tuning/. The chosen configuration is then evaluated once on test.
"""

import argparse
import asyncio
from pathlib import Path

from pydantic import JsonValue

from graphwalk.cli import _eval_factories  # pyright: ignore[reportPrivateUsage]
from graphwalk.eval.suite import GLOSSES, run_dataset

ALL_SOFT: dict[str, JsonValue] = {
    "show_types": True,
    "stop_style": "literal",
    "relation_glosses": GLOSSES,
    "answer_type": "hint",
}
VARIANTS: list[tuple[str, dict[str, JsonValue]]] = [
    ("-v1", {}),
    ("-types", {"show_types": True}),
    ("-literal", {"stop_style": "literal"}),
    ("-glosses", {"relation_glosses": GLOSSES}),
    ("-hint", {"answer_type": "hint"}),
    ("-gate", {"answer_type": "gate"}),
    ("-soft", ALL_SOFT),
    ("-soft+gate", {**ALL_SOFT, "answer_type": "gate"}),
]


async def main(n: int, preset: str, hops: list[int]) -> None:
    factories = _eval_factories("openrouter/openai/gpt-6-luna", "BAAI/bge-small-en-v1.5", 18.0)
    for hop in hops:
        path, _ = await run_dataset(
            "metaqa",
            [preset],
            factories,
            hops=hop,
            n=n,
            seed=0,
            split="dev",
            concurrency=4,
            out_root=Path("results/tuning"),
            variants=VARIANTS,
        )
        print((path / "summary.md").read_text(encoding="utf-8"))  # noqa: T201


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--preset", default="relation")
    parser.add_argument("--hops", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument(
        "--variants", action="extend", nargs="+", help="labels to run (default: all)"
    )
    args = parser.parse_args()
    if args.variants:
        VARIANTS[:] = [v for v in VARIANTS if v[0] in args.variants]
    asyncio.run(main(args.n, args.preset, args.hops))
