"""E5: extractor strength. Re-extract a 2Wiki slice with a stronger model, then re-run
QA and evidence retrieval on both graphs.

    uv run python scripts/eval/extractor.py --per-type 10

The slice is the first ``--per-type`` questions of each type in the M7 sample (seed 0),
and its pooled paragraphs. Two graphs are ingested from the same paragraphs with the
same pipeline (Jev routing, escalation model unchanged); only the extraction model
differs. Extractions are cached per model, so the base model's graph is almost free
(M7 already extracted these paragraphs).

On each graph: ingestion quality against 2Wiki's gold evidence triples (M6 metric),
evidence recall (E1 methods), and QA with the M7 systems. The text baselines read the
same paragraphs and don't depend on the graph, so they run once.
"""

import argparse
import asyncio
import json
import re
from datetime import UTC, datetime
from pathlib import Path

from evidence import SETTINGS, attach_documents, evidence_questions, summary_text
from text_qa import EMBED, PRESET, TYPES, load  # scripts/eval siblings

from graphwalk.cli import _make_backend  # pyright: ignore[reportPrivateUsage]
from graphwalk.config import GraphwalkSettings
from graphwalk.embeddings.fastembed_embedder import FastEmbedEmbedder
from graphwalk.eval.datasets.cache import cache_dir
from graphwalk.eval.evidence import retrieval_rows
from graphwalk.eval.ingest_eval import score
from graphwalk.eval.suite import preset_config
from graphwalk.eval.text_qa import (
    GRAPHWALK,
    READER,
    TEXT_ITER_RAG,
    TEXT_RAG,
    build_graph,
    build_text_systems,
    graph_key,
    run_text_qa,
    source_documents,
    stratified,
)
from graphwalk.ingest import IngestConfig, JsonFileCache
from graphwalk.llm.litellm_backend import LiteLLMBackend

INGEST = IngestConfig(routing="jev", escalate=True, evidence=False)  # as M7


def slug(model: str) -> str:
    return re.sub(r"[^\w.-]+", "_", model)


async def main(args: argparse.Namespace) -> None:  # noqa: PLR0915 - one linear experiment
    sample = stratified(load("2wiki"), TYPES["2wiki"], 30, 0)  # the M7 sample
    records = [
        r
        for type_ in TYPES["2wiki"]
        for r in [r for r in sample if r["type"] == type_][: args.per_type]
    ]
    key = GraphwalkSettings().openrouter_api_key
    api_key = None if key is None else key.get_secret_value()

    def llm(model: str, max_tokens: int) -> LiteLLMBackend:
        return LiteLLMBackend(model, api_key=api_key, max_tokens=max_tokens, max_rpm=args.rpm)

    embedder = FastEmbedEmbedder(EMBED)
    decider = _make_backend()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = Path("results") / f"{stamp}-2wiki-extractor"
    out.mkdir(parents=True)
    base_key = graph_key("2wiki", records, INGEST)
    reader = llm(args.llm, 512)
    common = {"decider": decider, "embedder": embedder, "llm": reader,
              "config": preset_config(PRESET)}  # fmt: skip
    ingest_lines = ["| extractor | cost | nodes | edges | entities found | triples recalled "
                    "| chains complete |", "|---|---|---|---|---|---|---|"]  # fmt: skip
    systems = []
    stores = {}
    try:
        for model in args.extractors:
            cache = JsonFileCache(cache_dir() / "extractions" / f"{slug(model)}.json")
            path = cache_dir().parent / "graphs" / f"{base_key}-{slug(model)}.graph.json"
            store, report = await build_graph(
                "2wiki", records, llm=llm(model, args.extract_max_tokens), decider=decider,
                embedder=embedder, cache=cache, config=INGEST,
                escalation_llm=llm(args.escalation_model, 4096), graph_path=path,
            )  # fmt: skip
            cache.flush()
            if report is not None:
                report_path = path.with_suffix(".report.json")
                report_path.write_text(report.model_dump_json(), "utf-8")
            report_file = path.with_suffix(".report.json")
            cost = "n/a"
            if report_file.exists():
                saved = json.loads(report_file.read_text("utf-8"))
                cost = "n/a" if saved.get("cost_usd") is None else f"${saved['cost_usd']:.3f}"
            quality = await score(store, records)
            rates = quality.rates()
            ingest_lines.append(
                f"| {model} | {cost} | {quality.nodes} | {quality.edges} "
                f"| {rates['entity_recall']:.3f} | {rates['triple_recall']:.3f} "
                f"| {rates['chain_recall']:.3f} |"
            )
            print(ingest_lines[-1], flush=True)  # noqa: T201
            stores[model] = store
            label = f"-{slug(model).split('_')[-1]}"
            systems += await build_text_systems([GRAPHWALK, READER], store, records,
                                                label=label, **common)  # fmt: skip
            systems += await build_text_systems(
                [READER], store, records, documents=source_documents("2wiki", records),
                label=f"{label}-source", **common,
            )  # fmt: skip

        (out / "ingest.md").write_text("\n".join(ingest_lines) + "\n", encoding="utf-8")
        if args.ingest_only:
            print("\n".join(ingest_lines))  # noqa: T201
            return
        systems += await build_text_systems([TEXT_RAG, TEXT_ITER_RAG], stores[args.extractors[0]],
                                            records, **common)  # fmt: skip

        # Evidence retrieval on each graph.
        questions = evidence_questions("2wiki", records)
        checkpoints = cache_dir().parent / "checkpoints"
        for model, store in stores.items():
            titles = await attach_documents(store, "2wiki", records)
            rows = await retrieval_rows(
                store, questions, titles, decider=decider, embedder=embedder,
                config=preset_config(SETTINGS["2wiki"]["preset"]), ks=SETTINGS["2wiki"]["ks"],
                checkpoints=checkpoints / f"e5-evidence-{slug(model)}-{base_key}",
            )  # fmt: skip
            text = summary_text("2wiki", rows, {"extractor": model, "n": len(questions)})
            (out / f"evidence-{slug(model)}.md").write_text(text, encoding="utf-8")
            (out / f"evidence-{slug(model)}.jsonl").write_text(
                "".join(r.model_dump_json() + "\n" for r in rows), encoding="utf-8"
            )

        def progress(name: str, done: int, total: int) -> None:
            if done == total:
                print(f"  {name}: {done}/{total}", flush=True)  # noqa: T201

        qa_out, _ = await run_text_qa(
            "2wiki", records, systems, out_dir=out / "qa",
            params={"slice_per_type": args.per_type, "n": len(records), "llm": args.llm,
                    "extractors": args.extractors, "escalation_llm": args.escalation_model,
                    "preset": PRESET, "embedder": EMBED},
            ingest=None, concurrency=args.concurrency, on_progress=progress,
            checkpoint_dir=cache_dir().parent / "checkpoints" / f"e5-qa-{base_key}",
        )  # fmt: skip
    finally:
        await decider.aclose()
    print((out / "ingest.md").read_text("utf-8"))  # noqa: T201
    print((qa_out / "summary.md").read_text("utf-8"))  # noqa: T201


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-type", type=int, default=10)
    parser.add_argument(
        "--extractors",
        nargs="+",
        default=["openrouter/openai/gpt-6-luna", "openrouter/openai/gpt-6.1-sol"],
    )
    parser.add_argument("--llm", default="openrouter/openai/gpt-6-luna", help="the reader")
    parser.add_argument("--escalation-model", default="openrouter/xiaomi/mimo-v2.6-pro")
    parser.add_argument("--extract-max-tokens", type=int, default=8192)
    parser.add_argument("--ingest-only", action="store_true", help="build and score graphs only")
    parser.add_argument("--rpm", type=float, default=18.0)
    parser.add_argument("--concurrency", type=int, default=4)
    asyncio.run(main(parser.parse_args()))
