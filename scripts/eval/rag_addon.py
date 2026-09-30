"""E7: does graphwalk help multi-step RAG when retrieval is hard?

    uv run python scripts/eval/rag_addon.py

Multi-step RAG (M7's `text-iter-rag`) on the M7 2Wiki sample (120 questions, seed 0),
with and without the graph add-on, under two stressors:

* title lookup off: searches go through the embedding index only, as in corpora whose
  documents are not one-per-entity (news, filings, tickets);
* a smaller step budget: 2 LLM rounds (one search round) instead of 5.

The add-on replaces the first retrieval (top 5 paragraphs for the question) with
graphwalk's hybrid `locate` (Jev linking, graph walk fused with dense), taken from the
checkpointed E1 run on the same questions and graph, so its cost is E1's Jev cost
($0.245 per 1k questions), reported separately. Everything else is identical.
"""

import argparse
import asyncio
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from text_qa import EMBED, TYPES, load  # scripts/eval sibling

from graphwalk.config import GraphwalkSettings
from graphwalk.embeddings.fastembed_embedder import FastEmbedEmbedder
from graphwalk.eval.datasets.cache import cache_dir
from graphwalk.eval.evidence import EvidenceRow
from graphwalk.eval.systems import TEXT_PROMPTS, DocIndex, IterativeRAGSystem
from graphwalk.eval.text_qa import paragraph_docs, run_text_qa, stratified
from graphwalk.llm.litellm_backend import LiteLLMBackend

if TYPE_CHECKING:
    from graphwalk.eval.types import QASystem

E1_RUN = "20260930T124411Z-2wiki-evidence"
SOURCE = "2wiki:paragraphs/"


def hybrid_seeds(questions: dict[str, str], k: int) -> dict[str, list[str]]:
    """Question text -> the paragraph titles E1's hybrid (Jev linking) ranked first."""
    lines = (Path("results") / E1_RUN / "rows.jsonl").read_text(encoding="utf-8").splitlines()
    rows = [EvidenceRow.model_validate_json(line) for line in lines]
    seeds: dict[str, list[str]] = {}
    for row in rows:
        if row.method == "hybrid-choice":
            titles = [key.removeprefix(SOURCE) for key in row.at(k)]
            seeds[questions[row.question_id]] = titles
    return seeds


async def main(args: argparse.Namespace) -> None:
    records = stratified(load("2wiki"), TYPES["2wiki"], 30, 0)  # the M7 sample
    key = GraphwalkSettings().openrouter_api_key
    llm = LiteLLMBackend(
        args.llm,
        api_key=None if key is None else key.get_secret_value(),
        max_tokens=512,
        max_rpm=args.rpm,
    )
    embedder = FastEmbedEmbedder(EMBED)
    index = await DocIndex.build(paragraph_docs(records), embedder)
    titles = {title.casefold(): [i] for i, title in enumerate(index.ids)}
    questions = {f"2wiki-{r['_id']}": str(r["question"]) for r in records}
    seeds = hybrid_seeds(questions, args.k)

    async def graph_first(question: str) -> Sequence[str]:
        return seeds[question]

    systems: list[QASystem] = []
    for title_search in (True, False):
        for steps in (5, 2):
            for addon in (False, True):
                name = (f"iter-rag-steps{steps}" + ("" if title_search else "-notitles")
                        + ("+graph" if addon else ""))  # fmt: skip
                systems.append(
                    IterativeRAGSystem(
                        index, embedder, llm, names=titles if title_search else {},
                        k=args.k, max_steps=steps, max_docs=40, name=name,
                        prompts=TEXT_PROMPTS, first=graph_first if addon else None,
                    )
                )  # fmt: skip

    def progress(name: str, done: int, total: int) -> None:
        if done == total:
            print(f"  {name}: {done}/{total}", flush=True)  # noqa: T201

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out, _ = await run_text_qa(
        "2wiki", records, systems, out_dir=Path("results") / f"{stamp}-2wiki-ragaddon",
        params={"n": len(records), "seed": 0, "llm": args.llm, "k": args.k,
                "addon": f"hybrid-choice locate from {E1_RUN}", "embedder": EMBED},
        ingest=None, concurrency=args.concurrency, on_progress=progress,
        checkpoint_dir=cache_dir().parent / "checkpoints" / "e7-ragaddon-2wiki",
    )  # fmt: skip
    print((out / "summary.md").read_text(encoding="utf-8"))  # noqa: T201


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", default="openrouter/openai/gpt-6-luna")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--rpm", type=float, default=18.0)
    parser.add_argument("--concurrency", type=int, default=4)
    asyncio.run(main(parser.parse_args()))
