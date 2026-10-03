"""A2: KG-QA on Freebase (WebQSP, CWQ), one subgraph per question.

    uv run python scripts/eval/credits.py && \\
    uv run --extra eval --extra llm --extra embeddings python scripts/eval/kgqa.py \\
        --dataset webqsp --n 50 --systems jev llm path

Systems, all starting from the release's topic entities:

* ``jev``: graphwalk, relation hops, greedy, Jev decisions (the ``relation-v2`` preset);
* ``llm``: the same walk with the LLM-as-decider;
* ``logprob``: the same walk, decisions from an option letter's token probabilities;
* ``path``: the LLM writes the relation path from the subgraph's schema and code runs
  it (two retries when it returns nothing), as in E2.

``--shuffle K`` presents each question's options in a random order (permutation seed K;
the option-order control, graphwalk.eval.shuffle); ``logprob`` runs also write the
decider's format diagnostics to ``diagnostics.json``.

Escalation (Jev, re-walked with the LLM decider below a confidence threshold) is
computed offline from ``jev`` and ``llm`` by ``scripts/paper/table_escalation.py``.
Wide steps (Freebase entities have hundreds of relations) are cut to the 30 most
query-similar options by a local embedder, as in every earlier experiment.
"""

import argparse
import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

from graphwalk.cli import _make_backend  # pyright: ignore[reportPrivateUsage]
from graphwalk.config import GraphwalkSettings
from graphwalk.decisions.base import DecisionBackend
from graphwalk.decisions.llm_decider import LLMDecider
from graphwalk.decisions.logprob import LogprobDecider
from graphwalk.embeddings.fastembed_embedder import FastEmbedEmbedder
from graphwalk.eval.calibration import CALIBRATION_HEADER, calibration_row
from graphwalk.eval.datasets import rog
from graphwalk.eval.per_question import PerQuestionSystem
from graphwalk.eval.query_writer import PathQuerySystem
from graphwalk.eval.runner import SystemRun, run_system, sample_questions, write_results
from graphwalk.eval.shuffle import ShuffledDecider
from graphwalk.eval.stated_decider import StatedDecider
from graphwalk.eval.suite import preset_config
from graphwalk.eval.systems import GraphwalkSystem
from graphwalk.eval.types import EvalQuestion, QASystem
from graphwalk.llm.litellm_backend import LiteLLMBackend
from graphwalk.stores.base import GraphStore
from graphwalk.traversal import Traverser

EMBED = "BAAI/bge-small-en-v1.5"
PRESET = "relation-v2"
NODE_TYPES = (rog.CVT_TYPE, rog.ENTITY_TYPE)


async def main(args: argparse.Namespace) -> None:
    questions, graphs = rog.load(args.dataset, args.split)
    subset = sample_questions(questions, args.n, args.seed)
    print(f"{args.dataset}: {len(subset)} of {len(questions)} questions", flush=True)  # noqa: T201
    key = GraphwalkSettings().openrouter_api_key
    api_key = None if key is None else key.get_secret_value()

    def llm(max_tokens: int) -> LiteLLMBackend:
        return LiteLLMBackend(args.llm, api_key=api_key, max_tokens=max_tokens, max_rpm=args.rpm)

    embedder = FastEmbedEmbedder(EMBED)
    config = preset_config(PRESET, budget={"max_depth": args.max_depth, "max_decision_calls": 12})

    async def graph(question: EvalQuestion) -> GraphStore:
        return await rog.build_store(graphs[question.id], f"{args.dataset}:{question.id}")

    made: list[DecisionBackend] = []
    suffix = f"-shuf{args.shuffle}" if args.shuffle else ""

    def walker(inner: DecisionBackend, name: str) -> PerQuestionSystem:
        made.append(inner)
        decider = ShuffledDecider(inner, args.shuffle) if args.shuffle else inner
        name += suffix

        def build(store: GraphStore, question: EvalQuestion) -> QASystem:
            del question
            traverser = Traverser(
                store, decider, embedder=embedder, config=config, node_types=NODE_TYPES
            )
            return GraphwalkSystem(store, traverser, name=name, linking="given")

        return PerQuestionSystem(
            name, graph, build,
            {"system": "graphwalk", "decision_model": decider.model_id, "embedder": EMBED,
             "traversal": config.model_dump(mode="json"), "shuffle": args.shuffle},
        )  # fmt: skip

    def path_writer() -> PerQuestionSystem:
        writer = llm(args.max_tokens)

        def build(store: GraphStore, question: EvalQuestion) -> QASystem:
            triples = graphs[question.id]
            counts: dict[str, int] = {}
            for _, relation, _ in triples:
                counts[relation] = counts.get(relation, 0) + 1
            return PathQuerySystem(
                store, writer, rog.relation_signature(triples), counts=counts,
                retries=2, name="llm-path-retry",
            )  # fmt: skip

        return PerQuestionSystem(
            "llm-path-retry", graph, build,
            {"system": "llm-path", "llm": writer.model_id, "retries": 2},
        )  # fmt: skip

    builders = {
        "jev": lambda: walker(_make_backend(), f"graphwalk-{PRESET}-jev"),
        "llm": lambda: walker(LLMDecider(llm(args.max_tokens)), f"graphwalk-{PRESET}-llm"),
        "logprob": lambda: walker(
            LogprobDecider(args.logprob_model, api_key=api_key or "", max_rpm=args.logprob_rpm),
            f"graphwalk-{PRESET}-logprob",
        ),
        "path": path_writer,
    }
    for mode in ("scores", "top1", "vote"):
        label = "vote" if mode == "vote" else f"stated-{mode}"
        builders[label] = lambda mode=mode, label=label: walker(
            StatedDecider(
                args.logprob_model, mode, api_key=api_key or "", max_rpm=args.logprob_rpm
            ),
            f"graphwalk-{PRESET}-{label}",
        )
    runs: list[SystemRun] = []
    for label in args.systems:
        system = builders[label]()

        def progress(done: int, total: int, name: str = system.name) -> None:
            if done == total or done % 25 == 0:
                print(f"  {name}: {done}/{total}", flush=True)  # noqa: T201

        runs.append(
            await run_system(system, subset, concurrency=args.concurrency, on_done=progress)
        )
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    label = f"{args.dataset}" + ("" if args.split == "test" else f"-{args.split}")
    out = write_results(
        Path("results") / "kgqa" / f"{stamp}-{label}",
        dataset=label,
        runs=runs,
        params={
            "dataset": args.dataset, "split": args.split, "n": len(subset), "seed": args.seed,
            "total_questions": len(questions), "preset": PRESET, "max_depth": args.max_depth,
            "answer_in_graph": sum(bool(q.meta.get("answer_in_graph")) for q in subset),
            "concurrency": args.concurrency,
        },
    )  # fmt: skip
    stats = {
        d.model_id: d.diagnostics() for d in made if isinstance(d, LogprobDecider | StatedDecider)
    }
    if stats:
        (out / "diagnostics.json").write_text(json.dumps(stats, indent=2) + "\n")
        print(stats, flush=True)  # noqa: T201
    walks = [r for r in runs if r.system.startswith("graphwalk")]
    if walks:
        table = "\n".join(
            [CALIBRATION_HEADER, *(calibration_row(r.system, r.records) for r in walks)]
        )
        with (out / "summary.md").open("a", encoding="utf-8") as fh:
            fh.write(f"\n## Calibration of walk confidence\n\n{table}\n")
    print((out / "summary.md").read_text(encoding="utf-8")[:3000], flush=True)  # noqa: T201


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["webqsp", "cwq"], default="webqsp")
    parser.add_argument("--split", default="test")
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--systems", nargs="+", default=["jev", "llm", "path"],
                        choices=["jev", "llm", "logprob", "path", "stated-scores",
                                 "stated-top1", "vote"])  # fmt: skip
    parser.add_argument("--llm", default="openrouter/openai/gpt-6-luna")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--logprob-model", default="qwen/qwen3.8-27b", help="open weights")
    parser.add_argument("--logprob-rpm", type=float, default=120.0)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--rpm", type=float, default=18.0)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--shuffle", type=int, default=0, help="option-order seed (0 = off)")
    asyncio.run(main(parser.parse_args()))
