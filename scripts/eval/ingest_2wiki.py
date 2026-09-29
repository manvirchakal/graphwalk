"""M6: ingestion quality on 2Wiki paragraphs, per routing variant.

    uv run python scripts/eval/ingest_2wiki.py [--n 40] [--seed 0]

Extractions are cached in ~/.cache/graphwalk/extractions/, so later variants (and
re-runs) pay only for routing. Summaries go to results/<stamp>-2wiki-ingest/.
"""

import argparse
import asyncio
import re
from pathlib import Path

from graphwalk.cli import _make_backend  # pyright: ignore[reportPrivateUsage]
from graphwalk.config import GraphwalkSettings
from graphwalk.embeddings.fastembed_embedder import FastEmbedEmbedder
from graphwalk.eval.datasets.cache import cache_dir
from graphwalk.eval.datasets.twowiki import download, read_records
from graphwalk.eval.ingest_eval import (
    VariantResult,
    run_dir,
    run_variants,
    sample_records,
    summary_table,
    write_results,
)
from graphwalk.ingest import IngestConfig, JsonFileCache
from graphwalk.llm.litellm_backend import LiteLLMBackend

VARIANTS = {
    "exact": IngestConfig(routing="exact"),
    "jev": IngestConfig(routing="jev", escalate=False),
    "jev+llm": IngestConfig(routing="jev", escalate=True),
}


async def main(
    n: int, seed: int, model: str, variants: list[str], escalation_model: str | None
) -> None:
    records = sample_records(read_records(download()), n, seed)
    key = GraphwalkSettings().openrouter_api_key
    llm = LiteLLMBackend(
        model,
        api_key=None if key is None else key.get_secret_value(),
        max_tokens=4096,
        max_rpm=18.0,
    )
    escalation = (
        None
        if escalation_model is None
        else LiteLLMBackend(
            escalation_model,
            api_key=None if key is None else key.get_secret_value(),
            max_tokens=4096,  # reasoning models spend output tokens before the label
            max_rpm=18.0,
        )
    )
    cache_path = cache_dir() / "extractions" / (re.sub(r"[^\w.-]+", "_", model) + ".json")
    cache = JsonFileCache(cache_path)
    decider = _make_backend()
    out = run_dir(Path("results"), "2wiki-ingest")

    def done(result: VariantResult) -> None:
        cache.flush()
        print(summary_table([result]), flush=True)  # noqa: T201

    try:
        results = await run_variants(
            records,
            {k: VARIANTS[k] for k in variants},
            llm=llm,
            decider=decider,
            embedder=FastEmbedEmbedder("BAAI/bge-small-en-v1.5"),
            cache=cache,
            on_variant=done,
            escalation_llm=escalation,
        )
    finally:
        await decider.aclose()
        cache.flush()
    params = {
        "n": n,
        "seed": seed,
        "llm": model,
        "escalation_llm": escalation_model or model,
        "embedder": "BAAI/bge-small-en-v1.5",
    }
    path = write_results(out, results, params)
    print((path / "summary.md").read_text(encoding="utf-8"))  # noqa: T201


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default="openrouter/openai/gpt-6-luna")
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    parser.add_argument("--escalation-model", default=None, help="default: --model")
    args = parser.parse_args()
    asyncio.run(main(args.n, args.seed, args.model, args.variants, args.escalation_model))
