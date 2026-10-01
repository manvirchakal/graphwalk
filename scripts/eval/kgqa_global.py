"""A6: KG-QA over one large graph whose schema is too big to curate per question.

    uv run python scripts/eval/credits.py && \\
    uv run --extra eval --extra llm --extra embeddings python scripts/eval/kgqa_global.py \\
        --n 100 --systems jev path-local path-global

A2 gave every system its own small per-question subgraph (median 288 relations). Here
all 1,628 WebQSP test subgraphs are merged into one SQLite graph (2.3M triples, 5,419
relations), the setting graphwalk's "large or messy schema" pitch is about. Systems,
all starting from the release's topic entities:

* ``jev``: graphwalk, ``TraversalConfig.kgqa()``, Jev decisions, embedder prefilter;
* ``path-global``: the LLM writes the relation path from the whole merged schema (all
  5,419 relations in one prompt; feasible because the model reads 1M tokens);
* ``path-local``: the LLM writes the path from the relations found within two hops of
  the topic entity in the merged graph (schema retrieval, what a practitioner would do
  when the full schema doesn't fit).

Both path writers get two retries when the path is invalid or returns nothing (as E2).
"""

import argparse
import asyncio
from datetime import UTC, datetime
from pathlib import Path

from pydantic import JsonValue

from graphwalk.cli import _make_backend  # pyright: ignore[reportPrivateUsage]
from graphwalk.config import GraphwalkSettings
from graphwalk.embeddings.fastembed_embedder import FastEmbedEmbedder
from graphwalk.eval.calibration import CALIBRATION_HEADER, calibration_row
from graphwalk.eval.datasets import rog
from graphwalk.eval.datasets.cache import cache_dir
from graphwalk.eval.query_writer import PathQuerySystem, local_relations
from graphwalk.eval.runner import SystemRun, run_system, sample_questions, write_results
from graphwalk.eval.systems import GraphwalkSystem
from graphwalk.eval.types import EvalQuestion, QASystem, SystemAnswer
from graphwalk.llm.base import LLMBackend
from graphwalk.llm.litellm_backend import LiteLLMBackend
from graphwalk.stores.sqlite_store import SQLiteStore
from graphwalk.traversal import TraversalConfig, Traverser

EMBED = "BAAI/bge-small-en-v1.5"
NODE_TYPES = (rog.CVT_TYPE, rog.ENTITY_TYPE)


class LocalPathSystem:
    """``path-local``: a path writer whose schema is the relations near the start."""

    def __init__(
        self,
        store: SQLiteStore,
        llm: LLMBackend,
        signature: dict[str, tuple[str, str]],
        *,
        hops: int,
    ) -> None:
        self._store = store
        self._llm = llm
        self._signature = signature
        self._hops = hops

    @property
    def name(self) -> str:
        return "llm-path-local"

    def describe(self) -> dict[str, JsonValue]:
        return {"system": "llm-path", "llm": self._llm.model_id, "retries": 2,
                "schema": f"relations within {self._hops} hops of the topic entity"}  # fmt: skip

    async def answer(self, question: EvalQuestion) -> SystemAnswer:
        counts = await local_relations(self._store, question.start or (), hops=self._hops)
        system = PathQuerySystem(
            self._store, self._llm, {r: self._signature[r] for r in counts},
            counts=counts, retries=2, name=self.name,
        )  # fmt: skip
        answer = await system.answer(question)
        return answer.model_copy(
            update={"detail": {**answer.detail, "schema_relations": len(counts)}}
        )


async def main(args: argparse.Namespace) -> None:
    questions, graphs = rog.load("webqsp", "test")
    subset = sample_questions(questions, args.n, args.seed)
    path = cache_dir() / "graphs" / "webqsp-test-global.db"
    path.parent.mkdir(parents=True, exist_ok=True)
    store = await rog.build_global_store(
        (graphs[q.id] for q in questions), path, source_id="rog-webqsp-test-global"
    )
    nodes, edges = await store.counts()
    signature = dict(rog.relation_signature([t for g in graphs.values() for t in g]))
    global_counts: dict[str, int] = {}
    for g in graphs.values():
        for _, relation, _ in g:
            global_counts[relation] = global_counts.get(relation, 0) + 1
    summary = f"{nodes:,} nodes, {edges:,} edges, {len(signature):,} relations"
    print(f"global graph: {summary}; {len(subset)} questions", flush=True)  # noqa: T201

    key = GraphwalkSettings().openrouter_api_key
    api_key = None if key is None else key.get_secret_value()

    def llm() -> LiteLLMBackend:
        return LiteLLMBackend(
            args.llm, api_key=api_key, max_tokens=args.max_tokens, max_rpm=args.rpm
        )

    def jev() -> QASystem:
        config = TraversalConfig.kgqa()
        traverser = Traverser(
            store, _make_backend(), embedder=FastEmbedEmbedder(EMBED), config=config,
            node_types=NODE_TYPES,
        )  # fmt: skip
        return GraphwalkSystem(store, traverser, name="graphwalk-kgqa-jev", linking="given")

    def path_global() -> QASystem:
        return PathQuerySystem(
            store, llm(), signature, counts=global_counts, retries=2, name="llm-path-global"
        )

    builders = {
        "jev": (jev, subset),
        "path-local": (lambda: LocalPathSystem(store, llm(), signature, hops=2), subset),
        "path-global": (path_global, subset[: args.n_global or None]),
    }
    runs: list[SystemRun] = []
    for label in args.systems:
        factory, questions_for = builders[label]
        system = factory()

        def progress(done: int, total: int, name: str = system.name) -> None:
            if done == total or done % 25 == 0:
                print(f"  {name}: {done}/{total}", flush=True)  # noqa: T201

        runs.append(
            await run_system(system, questions_for, concurrency=args.concurrency, on_done=progress)
        )
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = write_results(
        Path("results") / "kgqa" / f"{stamp}-webqsp-global",
        dataset="webqsp-global",
        runs=runs,
        params={
            "dataset": "webqsp", "graph": "all test subgraphs merged", "nodes": nodes,
            "edges": edges, "relations": len(signature), "split": "test", "n": len(subset),
            "n_global": args.n_global, "seed": args.seed, "total_questions": len(questions),
            "traversal": "kgqa", "llm": args.llm, "concurrency": args.concurrency,
        },
    )  # fmt: skip
    walks = [r for r in runs if r.system.startswith("graphwalk")]
    if walks:
        table = "\n".join(
            [CALIBRATION_HEADER, *(calibration_row(r.system, r.records) for r in walks)]
        )
        with (out / "summary.md").open("a", encoding="utf-8") as fh:
            fh.write(f"\n## Calibration of walk confidence\n\n{table}\n")
    await store.close()
    print((out / "summary.md").read_text(encoding="utf-8")[:2500], flush=True)  # noqa: T201


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--n-global", type=int, default=0, help="first N only (0 = all)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--systems", nargs="+", default=["jev", "path-local", "path-global"],
                        choices=["jev", "path-local", "path-global"])  # fmt: skip
    parser.add_argument("--llm", default="openrouter/openai/gpt-6-luna")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--rpm", type=float, default=18.0)
    parser.add_argument("--concurrency", type=int, default=4)
    asyncio.run(main(parser.parse_args()))
