"""A5: does traversal stay fast on a large SQLite graph with high-degree hubs?

    uv run python scripts/eval/scale.py --nodes 200000 --out-degree 6 --hubs 10

No API calls. A synthetic graph: ``--nodes`` entities, each with ``--out-degree``
random typed edges and one edge into one of ``--hubs`` hub nodes (so each hub has
about nodes / hubs incoming edges, like a country or a profession in Freebase). It is
imported with ``graphwalk.ingest.triples`` into a SQLite file, then timed:

* import (triples per second) and file size;
* ``neighbors`` and ``degree`` for random nodes and for hubs;
* name linking (index build, then per query);
* walks with a scripted decider (seeded random distributions, no I/O), entity and
  relation mode, from random nodes and from hubs, with and without an embedder
  prefilter (a fake embedder: this measures graphwalk's overhead, not the model's).

Decision latency is excluded by construction: these numbers are what graphwalk itself
adds to each walk. Writes ``results/<timestamp>-scale/results.json`` and
``summary.md``.
"""

import argparse
import asyncio
import json
import random
import resource
import statistics
import subprocess
import tempfile
import time
from collections.abc import Awaitable, Sequence
from datetime import UTC, datetime
from pathlib import Path

from graphwalk.decisions.fake import FakeDecisionBackend
from graphwalk.embeddings.fake import FakeEmbedder
from graphwalk.ingest.triples import Triple, import_triples
from graphwalk.stores.sqlite_store import SQLiteStore
from graphwalk.traversal import NameEntryResolver, TraversalConfig, Traverser

ROOT = Path(__file__).resolve().parents[2]
TYPES = ("person", "film", "place", "organization", "work")
WORDS = ("amber", "basil", "cedar", "delta", "ember", "fable", "garnet", "harbor", "iris",
         "juniper", "kestrel", "lumen", "maple", "nectar", "onyx", "pebble", "quartz",
         "raven", "sable", "tundra")  # fmt: skip


def name_of(i: int) -> str:
    """Distinct, wordy names (so name linking has realistic token overlap)."""
    a, b = WORDS[i % len(WORDS)], WORDS[(i // len(WORDS)) % len(WORDS)]
    return f"{a.title()} {b.title()} {i}"


def synthetic(nodes: int, out_degree: int, hubs: int, relations: int, seed: int) -> list[Triple]:
    rng = random.Random(seed)  # noqa: S311 - reproducible synthetic data
    rels = [f"rel_{r:02d}" for r in range(relations)]
    triples: list[Triple] = []
    for i in range(nodes):
        s_type = TYPES[i % len(TYPES)]
        for _ in range(out_degree):
            j = rng.randrange(nodes)
            if j != i:
                triples.append(Triple(f"n{i}", rng.choice(rels), f"n{j}", s_type,
                                      TYPES[j % len(TYPES)], name_of(i), name_of(j)))  # fmt: skip
        h = rng.randrange(hubs)
        triples.append(Triple(f"n{i}", "member_of", f"hub{h}", s_type, "hub", name_of(i),
                              f"Hub {WORDS[h % len(WORDS)].title()} {h}"))  # fmt: skip
    return triples


def pct(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def summary(values: Sequence[float]) -> dict[str, float]:
    return {
        "n": len(values),
        "p50_ms": 1000 * statistics.median(values),
        "p95_ms": 1000 * pct(values, 0.95),
        "max_ms": 1000 * max(values),
    }


async def timed[T](work: Awaitable[T]) -> tuple[float, T]:
    start = time.perf_counter()
    out = await work
    return time.perf_counter() - start, out


def git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],  # noqa: S607
        capture_output=True, text=True, check=True, cwd=ROOT,
    ).stdout.strip()  # fmt: skip


def dir_mb(path: Path) -> float:
    return sum(p.stat().st_size for p in path.iterdir()) / 1e6


def remove_dir(path: Path) -> None:
    for child in path.iterdir():
        child.unlink()
    path.rmdir()


def write_run(out: dict[str, object]) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = ROOT / "results" / f"{stamp}-scale"
    run_dir.mkdir(parents=True)
    (run_dir / "results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    (run_dir / "summary.md").write_text(render(out), encoding="utf-8")
    return run_dir


async def bench_import(store: SQLiteStore, args: argparse.Namespace) -> dict[str, float]:
    triples = synthetic(args.nodes, args.out_degree, args.hubs, args.relations, args.seed)
    seconds, _ = await timed(import_triples(store, triples, source_id="synthetic"))
    nodes, edges = await store.counts()
    return {"triples": len(triples), "nodes": nodes, "edges": edges, "seconds": seconds,
            "triples_per_s": len(triples) / seconds}  # fmt: skip


async def bench_lookups(store: SQLiteStore, groups: dict[str, list[str]]) -> dict[str, object]:
    lookups: dict[str, object] = {}
    for label, ids in groups.items():
        times: list[float] = []
        sizes: list[int] = []
        degree_times: list[float] = []
        for node_id in ids:
            t, found = await timed(store.neighbors(node_id, direction="both"))
            times.append(t)
            sizes.append(len(found))
            degree_times.append((await timed(store.degree(node_id, direction="both")))[0])
        lookups[label] = {"neighbors": summary(times), "degree": summary(degree_times),
                          "mean_degree": statistics.mean(sizes)}  # fmt: skip
    return lookups


async def bench_linking(store: SQLiteStore, ids: list[str]) -> dict[str, object]:
    resolver = NameEntryResolver(store)
    build_s, _ = await timed(resolver.resolve("warm up"))
    link_times: list[float] = []
    hits = 0
    for node_id in ids:
        query = f"Who is related to {name_of(int(node_id[1:]))}?"
        t, linked = await timed(resolver.resolve(query))
        link_times.append(t)
        hits += node_id in linked
    return {"index_build_s": build_s, "link": summary(link_times),
            "hit_rate": hits / len(link_times)}  # fmt: skip


async def bench_walks(
    store: SQLiteStore, groups: dict[str, list[str]], seed: int
) -> dict[str, object]:
    walks: dict[str, object] = {}
    for hop_mode in ("entity", "relation"):
        for prefilter in (False, True):
            config = TraversalConfig.model_validate(
                {"strategy": "beam", "beam_width": 3, "hop_mode": hop_mode,
                 "budget": {"max_depth": 3}}
            )  # fmt: skip
            traverser = Traverser(
                store,
                FakeDecisionBackend(seed=seed),
                embedder=FakeEmbedder() if prefilter else None,
                config=config,
            )
            for label, ids in groups.items():
                times: list[float] = []
                calls: list[int] = []
                for node_id in ids:
                    t, result = await timed(traverser.traverse("Which one is related?", (node_id,)))
                    times.append(t)
                    calls.append(result.trace.totals.decision_calls)
                key = f"{hop_mode}/{'prefilter' if prefilter else 'truncate'}/{label}"
                walks[key] = {**summary(times), "mean_decision_calls": statistics.mean(calls)}
                print(f"walk {key}: {walks[key]}")  # noqa: T201
    return walks


async def main(args: argparse.Namespace) -> None:
    out: dict[str, object] = {
        "args": {k: v for k, v in vars(args).items() if k not in {"tmp", "keep"}},
        "git_sha": git_sha(),
    }
    rng = random.Random(args.seed + 1)  # noqa: S311 - a benchmark's sample, not a secret
    workdir = Path(tempfile.mkdtemp(prefix="graphwalk-scale-", dir=args.tmp))
    store = SQLiteStore(workdir / "scale.db")
    imported = await bench_import(store, args)
    out["import"] = {**imported, "db_mb": await asyncio.to_thread(dir_mb, workdir)}
    print(f"import: {out['import']}")  # noqa: T201
    regular = [f"n{rng.randrange(args.nodes)}" for _ in range(args.samples)]
    hubs = [f"hub{h}" for h in range(args.hubs)]
    out["lookups"] = await bench_lookups(store, {"regular": regular, "hub": hubs})
    print(f"lookups: {out['lookups']}")  # noqa: T201
    out["linking"] = await bench_linking(store, regular[: args.walks])
    print(f"linking: {out['linking']}")  # noqa: T201
    out["walks"] = await bench_walks(
        store, {"regular": regular[: args.walks], "hub": hubs}, args.seed
    )
    out["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    await store.close()
    if not args.keep:
        await asyncio.to_thread(remove_dir, workdir)
    print(f"wrote {await asyncio.to_thread(write_run, out)}")  # noqa: T201


def render(out: dict[str, object]) -> str:
    imp = out["import"]
    assert isinstance(imp, dict)  # noqa: S101
    lines = [
        "# A5: scale benchmark (synthetic graph, scripted decider)\n",
        f"git {out['git_sha']}; args {json.dumps(out['args'])}\n",
        f"Import: {imp['triples']:,} triples -> {imp['nodes']:,} nodes, {imp['edges']:,} "
        f"edges in {imp['seconds']:.0f} s ({imp['triples_per_s']:,.0f} triples/s); "
        f"{imp['db_mb']:.0f} MB on disk. Peak RSS {out['peak_rss_mb']:.0f} MB.\n",
        "| operation | n | p50 ms | p95 ms | max ms | notes |",
        "|---|---|---|---|---|---|",
    ]
    lookups = out["lookups"]
    assert isinstance(lookups, dict)  # noqa: S101
    for label, data in lookups.items():
        for op in ("neighbors", "degree"):
            s = data[op]
            lines.append(f"| {op}, {label} nodes | {s['n']} | {s['p50_ms']:.1f} | "
                         f"{s['p95_ms']:.1f} | {s['max_ms']:.1f} | mean degree "
                         f"{data['mean_degree']:,.0f} |")  # fmt: skip
    linking = out["linking"]
    assert isinstance(linking, dict)  # noqa: S101
    s = linking["link"]
    lines.append(f"| name linking | {s['n']} | {s['p50_ms']:.1f} | {s['p95_ms']:.1f} | "
                 f"{s['max_ms']:.1f} | index build {linking['index_build_s']:.1f} s; "
                 f"right node linked {linking['hit_rate']:.0%} |")  # fmt: skip
    walks = out["walks"]
    assert isinstance(walks, dict)  # noqa: S101
    for key, s in walks.items():
        lines.append(f"| walk, {key} | {s['n']} | {s['p50_ms']:.1f} | {s['p95_ms']:.1f} | "
                     f"{s['max_ms']:.1f} | {s['mean_decision_calls']:.1f} decisions |")  # fmt: skip
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A5: scale benchmark (no API calls).")
    parser.add_argument("--nodes", type=int, default=200_000)
    parser.add_argument("--out-degree", type=int, default=6)
    parser.add_argument("--hubs", type=int, default=10)
    parser.add_argument("--relations", type=int, default=20)
    parser.add_argument("--samples", type=int, default=200, help="Nodes for lookup timings.")
    parser.add_argument("--walks", type=int, default=50, help="Walks per configuration.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tmp", type=Path, default=None, help="Where to put the database.")
    parser.add_argument("--keep", action="store_true", help="Keep the database.")
    asyncio.run(main(parser.parse_args()))
