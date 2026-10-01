"""A8: does a walk tool make an agent cheaper or better than graph primitives or RAG?

    uv run python scripts/eval/credits.py && \\
    uv run --extra eval --extra llm --extra embeddings python scripts/eval/agent_arms.py \\
        --n 100 --pilot 20 --arms graph walk search jev

One tool-calling loop (:mod:`graphwalk.eval.agent`), one model, one prompt, one turn
budget; only the tools differ:

* ``graph``: ``relations`` + ``neighbors`` (low-level graph tools);
* ``walk``: the same plus graphwalk's ``walk`` with its confidence;
* ``search``: dense search over the same facts as text (multi-step RAG);
* ``jev``: the walk alone, no agent (reference);
* ``closedbook``: no tools, one turn (what the model knows without the graph).

``grounded`` is the share of an arm's answers that appeared in some tool output: the
model knows much of WebQSP, so an answer can be right without the tools finding it.

All arms see the same facts: the WebQSP test subgraphs of the sampled questions, merged
into one graph (the search index embeds that graph's triples; the full 1,628-subgraph
graph of A6 is too large to embed here). ``--pilot`` runs the first N of the sample.
"""

import argparse
import asyncio
import json
import statistics
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from graphwalk.cli import _make_backend  # pyright: ignore[reportPrivateUsage]
from graphwalk.config import GraphwalkSettings
from graphwalk.embeddings.fastembed_embedder import FastEmbedEmbedder
from graphwalk.eval.agent import (
    AgentSystem,
    Chat,
    graph_tools,
    search_tool,
    triple_passages,
    walk_tool,
)
from graphwalk.eval.datasets import rog
from graphwalk.eval.datasets.cache import cache_dir
from graphwalk.eval.metrics import bootstrap_ci, percentile
from graphwalk.eval.runner import SystemRun, run_system, sample_questions, write_results
from graphwalk.eval.systems import DocIndex, GraphwalkSystem
from graphwalk.eval.types import QASystem
from graphwalk.traversal import TraversalConfig, Traverser

EMBED = "BAAI/bge-small-en-v1.5"
NODE_TYPES = (rog.CVT_TYPE, rog.ENTITY_TYPE)


async def passage_index(
    triples: list[tuple[str, str, str]], embedder: FastEmbedEmbedder, path: Path
) -> DocIndex:
    """Embedded once, then loaded from ``path`` (.npy + .json)."""
    texts_path = path.with_suffix(".json")
    if texts_path.exists():
        texts = json.loads(texts_path.read_text(encoding="utf-8"))
        vectors = np.load(path.with_suffix(".npy"))
        return DocIndex(tuple(texts), tuple(texts), vectors, embedder.model_id)
    texts = triple_passages(triples)
    print(f"embedding {len(texts):,} passages", flush=True)  # noqa: T201
    index = await DocIndex.build([(t, t) for t in texts], embedder)
    np.save(path.with_suffix(".npy"), index.vectors)
    texts_path.write_text(json.dumps(texts), encoding="utf-8")
    return index


def agent_table(runs: list[SystemRun]) -> str:
    lines = [
        "| arm | n | F1 | hits@1 | answered | turns (mean) | agent in-tok (mean) | "
        "agent out-tok | $/q (agent) | $/q (tools) | latency p50 / p90 s | grounded | tool calls |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for run in runs:
        records = run.records
        n = len(records)
        if not n:
            continue
        f1 = [r.score.f1 for r in records]
        lo, hi = bootstrap_ci(f1)
        hits = statistics.fmean(r.score.hits1 for r in records)
        lat = [r.answer.latency_s for r in records]
        detail = [r.answer.detail for r in records]
        turns = statistics.fmean(float(d.get("turns", 0) or 0) for d in detail)  # pyright: ignore[reportArgumentType]
        answered = statistics.fmean(float(bool(d.get("answered", True))) for d in detail)
        agent_cost = [d.get("agent_cost_usd") for d in detail]
        tool_cost = [d.get("tool_cost_usd") for d in detail]
        agent_q = (
            statistics.fmean(float(c) for c in agent_cost if isinstance(c, float))  # pyright: ignore[reportArgumentType]
            if any(isinstance(c, float) for c in agent_cost) else 0.0
        )  # fmt: skip
        tools_q = (
            statistics.fmean(float(c) for c in tool_cost if isinstance(c, float))  # pyright: ignore[reportArgumentType]
            if any(isinstance(c, float) for c in tool_cost)
            else statistics.fmean(r.answer.cost_usd or 0.0 for r in records)
        )
        shares = [float(g) for d in detail if isinstance(g := d.get("grounded"), float)]
        grounded = f"{statistics.fmean(shares):.2f}" if shares else "-"
        calls: dict[str, int] = {}
        for d in detail:
            raw = d.get("tool_calls")
            if isinstance(raw, dict):
                for k, v in raw.items():
                    calls[k] = calls.get(k, 0) + int(v)  # pyright: ignore[reportArgumentType]
        call_text = ", ".join(f"{k} {v / n:.1f}" for k, v in sorted(calls.items())) or "-"
        lines.append(
            f"| {run.system} | {n} | {statistics.fmean(f1):.3f} [{lo:.2f}, {hi:.2f}] | "
            f"{hits:.2f} | {answered:.2f} | {turns:.1f} | "
            f"{statistics.fmean(r.answer.input_tokens for r in records):,.0f} | "
            f"{statistics.fmean(r.answer.output_tokens for r in records):,.0f} | "
            f"{agent_q:.5f} | {tools_q:.5f} | "
            f"{percentile(lat, 50):.1f} / {percentile(lat, 90):.1f} | {grounded} | {call_text} |"
        )
    return "\n".join(lines)


def paired(runs: list[SystemRun]) -> str:
    by = {r.system: {x.question.id: x.score.f1 for x in r.records} for r in runs}
    lines: list[str] = []
    for a, b in (("agent-walk", "agent-graph"), ("agent-walk", "agent-search"),
                 ("agent-graph", "agent-search"), ("agent-walk", "jev"),
                 ("agent-walk", "closedbook")):  # fmt: skip
        if a in by and b in by:
            common = sorted(set(by[a]) & set(by[b]))
            diffs = [by[a][q] - by[b][q] for q in common]
            lo, hi = bootstrap_ci(diffs)
            lines.append(
                f"- {a} minus {b}: {statistics.fmean(diffs):+.3f} [{lo:+.2f}, {hi:+.2f}] F1"
                f" ({len(common)} q)"
            )
    return "\n".join(lines)


async def main(args: argparse.Namespace) -> None:
    questions, graphs = rog.load("webqsp", "test")
    sample = sample_questions(questions, args.n, args.seed)
    subset = sample[: args.pilot or None]
    root = cache_dir() / "graphs"
    root.mkdir(parents=True, exist_ok=True)
    tag = f"webqsp-test-agent-n{args.n}-s{args.seed}"
    store = await rog.build_global_store(
        (graphs[q.id] for q in sample), root / f"{tag}.db", source_id=f"rog-{tag}"
    )
    nodes, edges = await store.counts()
    print(f"graph: {nodes:,} nodes, {edges:,} edges; {len(subset)} questions", flush=True)  # noqa: T201
    embedder = FastEmbedEmbedder(EMBED)
    triples = list(dict.fromkeys(t for q in sample for t in graphs[q.id]))

    key = GraphwalkSettings().openrouter_api_key
    chat = Chat(
        args.llm,
        api_key=None if key is None else key.get_secret_value(),
        max_tokens=args.max_tokens,
        max_rpm=args.rpm,
    )

    def traverser() -> Traverser:
        return Traverser(
            store, _make_backend(), embedder=embedder, config=TraversalConfig.kgqa(),
            node_types=NODE_TYPES,
        )  # fmt: skip

    async def build(arm: str) -> QASystem:
        if arm == "jev":
            return GraphwalkSystem(store, traverser(), name="jev", linking="given")
        if arm == "closedbook":
            return AgentSystem("closedbook", chat, lambda _q: [], max_turns=1, model=args.llm)
        if arm == "search":
            index = await passage_index(triples, embedder, root / f"{tag}-passages")
            tools = [search_tool(index, embedder)]
        elif arm == "walk":
            tools = [*graph_tools(store), walk_tool(store, traverser())]
        else:
            tools = graph_tools(store)
        return AgentSystem(
            f"agent-{arm}", chat, lambda _q: tools, max_turns=args.max_turns, model=args.llm
        )

    runs: list[SystemRun] = []
    for arm in args.arms:
        system = await build(arm)

        def progress(done: int, total: int, name: str = system.name) -> None:
            if done == total or done % 10 == 0:
                print(f"  {name}: {done}/{total}", flush=True)  # noqa: T201

        runs.append(
            await run_system(system, subset, concurrency=args.concurrency, on_done=progress)
        )
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = write_results(
        Path("results") / "agent" / f"{stamp}-webqsp-agent",
        dataset="webqsp-agent",
        runs=runs,
        params={
            "dataset": "webqsp", "split": "test", "graph": f"subgraphs of {len(sample)} sampled q",
            "nodes": nodes, "edges": edges, "n": len(subset), "sample": len(sample),
            "seed": args.seed, "llm": args.llm, "max_turns": args.max_turns,
            "arms": args.arms, "concurrency": args.concurrency,
        },
    )  # fmt: skip
    with (out / "summary.md").open("a", encoding="utf-8") as fh:
        fh.write(f"\n## Agent arms\n\n{agent_table(runs)}\n\n## Paired F1\n\n{paired(runs)}\n")
    await store.close()
    print(agent_table(runs), "\n", paired(runs), flush=True)  # noqa: T201


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100, help="sample size (fixes the graph)")
    parser.add_argument("--pilot", type=int, default=0, help="run the first N only (0 = all)")
    parser.add_argument("--seed", type=int, default=0)
    arms = ["jev", "closedbook", "graph", "walk", "search"]
    parser.add_argument("--arms", nargs="+", default=arms, choices=arms)
    parser.add_argument("--llm", default="openrouter/openai/gpt-6-luna")
    parser.add_argument("--max-turns", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--rpm", type=float, default=18.0)
    parser.add_argument("--concurrency", type=int, default=4)
    asyncio.run(main(parser.parse_args()))
