"""Command-line interface: ``graphwalk ingest | locate | mcp | query | migrate | eval``."""

import asyncio
import math
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, get_args

import typer
from pydantic import ValidationError

from graphwalk import __version__
from graphwalk.config import GraphwalkSettings
from graphwalk.core.errors import GraphwalkError
from graphwalk.decisions import DecisionBackend
from graphwalk.stores.base import GraphStore
from graphwalk.stores.networkx_store import NetworkXStore
from graphwalk.stores.sqlite_store import SQLiteStore
from graphwalk.traversal import NameEntryResolver, TraversalConfig, TraversalResult, Traverser

if TYPE_CHECKING:
    from graphwalk.embeddings import Embedder
    from graphwalk.eval.suite import Factories
    from graphwalk.llm import LLMBackend

app = typer.Typer(
    name="graphwalk",
    help="Fast, probabilistic knowledge-graph traversal and ingestion.",
    no_args_is_help=True,
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(__version__)
        raise typer.Exit


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option("--version", callback=_version_callback, is_eager=True, help="Show version."),
    ] = False,
) -> None:
    """Fast, probabilistic knowledge-graph traversal and ingestion."""


SQLITE_SUFFIXES = frozenset({".db", ".sqlite", ".sqlite3"})


async def _open_graph(path: Path) -> GraphStore:
    """A SQLite database for ``.db``/``.sqlite`` paths, else a NetworkX JSON file."""
    if path.suffix.lower() in SQLITE_SUFFIXES:
        return SQLiteStore(path)
    exists = await asyncio.to_thread(path.exists)
    return await NetworkXStore.load(path) if exists else NetworkXStore()


def _make_backend() -> DecisionBackend:
    """The decision backend for CLI commands (replaced in tests): Jev, or the LLM
    fallback if ``GRAPHWALK_DECISION_FALLBACK=llm`` and no Jev key is set."""
    from graphwalk.providers import make_decider  # noqa: PLC0415

    return make_decider()


async def _run_query(
    graph: Path, question: str, start: list[str], config: TraversalConfig
) -> TraversalResult:
    store = await NetworkXStore.load(graph)
    backend = _make_backend()
    try:
        verify = getattr(backend, "verify_model", None)
        if verify is not None:
            await verify()
        if not start:
            start = await NameEntryResolver(store).resolve(question)
            if not start:
                typer.echo("No node name found in the query; pass --start.", err=True)
                raise typer.Exit(code=1)
        return await Traverser(store, backend, config=config).traverse(question, start)
    finally:
        await backend.aclose()


def _print_result(result: TraversalResult, top: int) -> None:
    typer.echo(f"status: {result.status}")
    if result.abort_reason:
        typer.echo(f"aborted: {result.abort_reason}")
    if result.error:
        typer.echo(f"error: {result.error}")
    for rank, answer in enumerate(result.answers[:top], start=1):
        names = ", ".join(answer.names)
        extra = f" votes={answer.votes}" if answer.votes > 1 else ""
        typer.echo(
            f"{rank}. {names}  score={answer.score:.3f} p={math.exp(answer.score):.2f} "
            f"[{answer.terminated_by}]{extra}"
        )
        for hop in answer.path:
            arrow = "->" if hop.direction == "out" else "<-"
            typer.echo(f"     {arrow} {hop.relation} ({len(hop.targets)} node(s))")
    t = result.trace.totals
    cost = "" if t.cost_usd is None else f", ${t.cost_usd:.6f}"
    typer.echo(
        f"calls={t.decision_calls} questions={t.questions} input_tokens={t.input_tokens}"
        f"{cost} decision_latency={t.decision_latency_s:.2f}s wall={t.wall_s:.2f}s"
    )


@app.command()
def query(  # noqa: PLR0917 - Typer maps parameters to CLI options
    graph: Annotated[Path, typer.Argument(help="Graph JSON written by NetworkXStore.save.")],
    question: Annotated[str, typer.Argument(help="The natural-language query.")],
    start: Annotated[
        list[str] | None,
        typer.Option(
            "--start", "-s", help="Start node id (repeatable). Default: names in the query."
        ),
    ] = None,
    strategy: Annotated[str, typer.Option(help="greedy | beam | sample")] = "beam",
    beam_width: Annotated[int, typer.Option(help="Beams kept per depth.")] = 3,
    max_depth: Annotated[int, typer.Option(help="Maximum decision depths (hops + STOP).")] = 4,
    hop_mode: Annotated[str, typer.Option(help="entity | relation")] = "entity",
    samples: Annotated[int, typer.Option(help="Walks for --strategy sample.")] = 5,
    temperature: Annotated[float, typer.Option(help="Sampling temperature.")] = 1.0,
    seed: Annotated[int, typer.Option(help="Sampling seed.")] = 0,
    label_style: Annotated[str, typer.Option(help="opaque | readable")] = "opaque",
    max_calls: Annotated[int | None, typer.Option(help="Abort after this many calls.")] = None,
    max_input_tokens: Annotated[
        int | None, typer.Option(help="Abort past this many tokens.")
    ] = None,
    top: Annotated[int, typer.Option(help="Answers to print.")] = 5,
    trace: Annotated[
        Path | None, typer.Option(help="Write the full result and trace as JSON here.")
    ] = None,
) -> None:
    """Answer a query by traversing the graph with Jev decisions."""
    try:
        config = TraversalConfig.model_validate(
            {
                "strategy": strategy,
                "beam_width": beam_width,
                "hop_mode": hop_mode,
                "n_samples": samples,
                "temperature": temperature,
                "seed": seed,
                "label_style": label_style,
                "budget": {
                    "max_depth": max_depth,
                    "max_decision_calls": max_calls,
                    "max_input_tokens": max_input_tokens,
                },
            }
        )
    except ValidationError as error:
        typer.echo(f"invalid options: {error}", err=True)
        raise typer.Exit(code=2) from error
    try:
        result = asyncio.run(_run_query(graph, question, start or [], config))
    except (GraphwalkError, ValueError, OSError) as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(code=1) from error
    _print_result(result, top)
    if trace is not None:
        trace.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        typer.echo(f"trace written to {trace}")
    if result.status == "error":
        raise typer.Exit(code=1)


def _eval_factories(llm_model: str, embed_model: str, llm_rpm: float | None) -> "Factories":
    """Real backends for ``graphwalk eval`` (replaced in tests)."""
    from graphwalk.embeddings.fastembed_embedder import FastEmbedEmbedder  # noqa: PLC0415
    from graphwalk.eval.suite import Factories  # noqa: PLC0415
    from graphwalk.llm.litellm_backend import LiteLLMBackend  # noqa: PLC0415

    settings = GraphwalkSettings()
    key = settings.openrouter_api_key
    return Factories(
        decider=_make_backend,
        embedder=lambda: FastEmbedEmbedder(embed_model),
        llm=lambda: LiteLLMBackend(
            llm_model,
            api_key=None if key is None else key.get_secret_value(),
            max_tokens=512,
            max_rpm=llm_rpm,
        ),
    )


def _ingest_backends(  # noqa: PLR0917 - replaced positionally in tests
    llm_provider: str | None,
    llm_model: str | None,
    embed_model: str | None,
    llm_rpm: float | None,
    escalation_model: str | None,
    use_embedder: bool,
) -> "tuple[LLMBackend, Embedder | None, LLMBackend | None]":
    """Extraction LLM, embedder, and escalation LLM for ``graphwalk ingest`` (replaced
    in tests). Options given on the command line override the environment."""
    from graphwalk.config import resolve_config  # noqa: PLC0415
    from graphwalk.providers import make_embedder, make_llm  # noqa: PLC0415

    arguments = {
        name: value
        for name, value in {
            "llm_provider": llm_provider,
            "llm_model": llm_model,
            "escalation_model": escalation_model,
            "embedding_model": embed_model,
        }.items()
        if value is not None
    }
    config = resolve_config(arguments)
    # 4096 tokens: extraction replies are long, and reasoning models also spend
    # output tokens on reasoning before the label.
    llm = make_llm(config, max_rpm=llm_rpm)
    escalation = (
        None if escalation_model is None else make_llm(config, role="escalation", max_rpm=llm_rpm)
    )
    return llm, make_embedder(config) if use_embedder else None, escalation


@app.command()
def ingest(  # noqa: PLR0917 - Typer maps parameters to CLI options
    path: Annotated[Path, typer.Argument(help="A file or directory (txt, md, json, jsonl, csv).")],
    graph: Annotated[
        Path,
        typer.Option(help="Graph to update (created if missing): SQLite (.db) or JSON."),
    ],
    source_id: Annotated[
        str | None, typer.Option(help="Ledger/provenance name (default: file:<abs path>).")
    ] = None,
    routing: Annotated[str, typer.Option(help="jev | exact")] = "jev",
    escalate: Annotated[
        bool, typer.Option(help="Send low-confidence routing decisions to the LLM.")
    ] = True,
    route_threshold: Annotated[float, typer.Option(help="Escalation threshold.")] = 0.6,
    prune: Annotated[
        bool, typer.Option(help="Retract documents that are no longer in the source.")
    ] = False,
    max_chunk_chars: Annotated[int, typer.Option(help="Chunk size.")] = 2000,
    concurrency: Annotated[int, typer.Option(help="Concurrent extraction calls.")] = 4,
    llm_provider: Annotated[
        str | None,
        typer.Option(
            help="openrouter | openai | anthropic | xai (default: GRAPHWALK_LLM_PROVIDER)."
        ),
    ] = None,
    llm_model: Annotated[
        str | None,
        typer.Option(help="Extraction model id, as the provider names it (default: per provider)."),
    ] = None,
    escalation_model: Annotated[
        str | None,
        typer.Option(help="Model for escalated routing decisions (default: --llm-model)."),
    ] = None,
    embed_model: Annotated[
        str | None,
        typer.Option(help="Embedding model for routing candidates (default: per provider)."),
    ] = None,
    no_embed: Annotated[
        bool, typer.Option("--no-embed", help="Name matching only for candidates.")
    ] = False,
    llm_rpm: Annotated[float | None, typer.Option(help="Cap LLM requests/min.")] = 18.0,
    report: Annotated[
        Path | None, typer.Option(help="Write the full report (with routes) as JSON.")
    ] = None,
    evidence: Annotated[
        bool, typer.Option(help="Record the supporting sentence of each fact (for locate).")
    ] = True,
) -> None:
    """Ingest a data source into the graph (idempotent; changed documents are re-derived)."""
    from graphwalk.ingest import FileSource, IngestConfig, IngestPipeline  # noqa: PLC0415

    if routing not in ("jev", "exact"):
        typer.echo("routing must be jev or exact", err=True)
        raise typer.Exit(code=2)
    try:
        source = FileSource(path, source_id=source_id)
        config = IngestConfig(
            routing=routing,  # pyright: ignore[reportArgumentType] - validated above
            escalate=escalate,
            route_threshold=route_threshold,
            prune_missing=prune,
            max_chunk_chars=max_chunk_chars,
            concurrency=concurrency,
            evidence=evidence,
        )
    except (FileNotFoundError, ValidationError) as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=2) from None
    try:
        llm, embedder, escalation_llm = _ingest_backends(
            llm_provider, llm_model, embed_model, llm_rpm, escalation_model, not no_embed
        )
    except GraphwalkError as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(code=2) from None

    async def run() -> None:
        store = await _open_graph(graph)
        decider = _make_backend() if routing == "jev" else None
        try:
            pipeline = IngestPipeline(
                store, llm, decider, embedder=embedder, config=config, escalation_llm=escalation_llm
            )
            result = await pipeline.ingest(source)
            if isinstance(store, NetworkXStore):
                await store.save(graph)
            nodes, edges = await store.counts()
        finally:
            if decider is not None:
                await decider.aclose()
            await store.close()
        cost = "n/a" if result.cost_usd is None else f"${result.cost_usd:.4f}"
        esc = result.escalation_cost_usd
        escalation_line = (
            f"escalation ({result.escalation_model}): {result.escalation_calls} calls, "
            f"{result.escalation_input_tokens} in / {result.escalation_output_tokens} out tokens, "
            f"{'n/a' if esc is None else f'${esc:.4f}'}\n"
            if result.escalation_model
            else ""
        )
        typer.echo(
            f"documents: {result.documents} ({result.new} new, {result.changed} changed, "
            f"{result.unchanged} unchanged, {result.removed} removed, {result.failed} failed)\n"
            f"chunks: {result.chunks}; entities: {result.entities}; "
            f"relations: {result.relations}\n"
            f"nodes: +{result.nodes_created} created, {result.nodes_merged} merged, "
            f"-{result.nodes_deleted} deleted; edges: +{result.edges_created} created, "
            f"{result.edges_merged} merged, -{result.edges_deleted} deleted\n"
            f"calls: {result.llm_calls} LLM, {result.decision_calls} decision "
            f"({result.escalations} escalations); cost: {cost}; {result.elapsed_s:.1f}s\n"
            f"{escalation_line}"
            f"graph: {graph} ({nodes} nodes, {edges} edges)"
        )
        for error in result.errors[:10]:
            typer.echo(f"error: {error}", err=True)
        if report is not None:
            report.write_text(result.model_dump_json(indent=2), encoding="utf-8")
            typer.echo(f"report written to {report}")

    try:
        asyncio.run(run())
    except GraphwalkError as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(code=1) from None


@app.command("eval")
def eval_(  # noqa: PLR0917 - Typer maps parameters to CLI options
    dataset: Annotated[str, typer.Argument(help="metaqa | 2wiki")],
    system: Annotated[
        list[str] | None,
        typer.Option(
            "--system",
            help=(
                "Repeatable: greedy, beam, relation, relation-beam, sample, rag, iter-rag; "
                "-v2 = tuned."
            ),
        ),
    ] = None,
    hops: Annotated[list[int] | None, typer.Option(help="MetaQA hops (repeatable).")] = None,
    n: Annotated[int, typer.Option(help="Seeded subset size (0 = all).")] = 100,
    seed: Annotated[int, typer.Option(help="Subset seed.")] = 0,
    linking: Annotated[
        str | None,
        typer.Option(help="given | gold | resolve | resolve-best | choice (default per dataset)."),
    ] = None,
    concurrency: Annotated[int, typer.Option(help="Questions in flight.")] = 8,
    rag_k: Annotated[int, typer.Option(help="Documents retrieved by the RAG baseline.")] = 5,
    llm_model: Annotated[
        str, typer.Option(help="LiteLLM model id for the RAG reader.")
    ] = "openrouter/openai/gpt-6-luna",
    embed_model: Annotated[str, typer.Option(help="fastembed model.")] = "BAAI/bge-small-en-v1.5",
    llm_rpm: Annotated[
        float | None, typer.Option(help="Cap LLM requests/min (OpenRouter new accounts: 20).")
    ] = 18.0,
    out: Annotated[Path, typer.Option(help="Results root directory.")] = Path("results"),
) -> None:
    """Run systems on a dataset subset; write results.json and summary.md."""
    from graphwalk.eval.suite import SYSTEMS, run_dataset  # noqa: PLC0415
    from graphwalk.eval.systems import Linking  # noqa: PLC0415

    if dataset not in ("metaqa", "2wiki"):
        typer.echo("dataset must be metaqa or 2wiki", err=True)
        raise typer.Exit(code=2)
    systems = system or ["greedy", "rag"]
    bad = [s for s in systems if s not in SYSTEMS]
    if bad or (linking is not None and linking not in get_args(Linking.__value__)):
        typer.echo(f"unknown system(s) {bad} or linking {linking!r}", err=True)
        raise typer.Exit(code=2)
    factories = _eval_factories(llm_model, embed_model, llm_rpm)

    def progress(name: str, done: int, total: int) -> None:
        if done == total or done % 25 == 0:
            typer.echo(f"  {name}: {done}/{total}", err=True)

    async def run_all() -> list[Path]:
        # One event loop for every run: backends hold loop-bound state (locks, clients).
        paths: list[Path] = []
        for hop in (hops or [1]) if dataset == "metaqa" else [0]:
            path, _runs = await run_dataset(
                "metaqa" if dataset == "metaqa" else "2wiki",
                systems,
                factories,
                hops=hop or 1,
                n=n or None,
                seed=seed,
                linking=linking,  # pyright: ignore[reportArgumentType] - validated above
                concurrency=concurrency,
                out_root=out,
                rag_k=rag_k,
                on_progress=progress,
            )
            typer.echo((path / "summary.md").read_text(encoding="utf-8"))
            typer.echo(f"results: {path}")
            paths.append(path)
        return paths

    asyncio.run(run_all())


def _locate_embedder(model: str) -> "Embedder":
    """The embedder for ``graphwalk locate --mode dense|hybrid`` (replaced in tests)."""
    from graphwalk.embeddings.fastembed_embedder import FastEmbedEmbedder  # noqa: PLC0415

    return FastEmbedEmbedder(model)


@app.command()
def locate(  # noqa: PLR0917 - Typer maps parameters to CLI options
    graph: Annotated[Path, typer.Argument(help="SQLite graph (.db) built by ingest.")],
    question: Annotated[str, typer.Argument(help="The natural-language query.")],
    k: Annotated[int, typer.Option("-k", help="Locations to return.")] = 5,
    mode: Annotated[str, typer.Option(help="graph | dense | hybrid")] = "graph",
    context: Annotated[int, typer.Option(help="Characters of context around each span.")] = 0,
    embed_model: Annotated[
        str, typer.Option(help="fastembed model (dense and hybrid modes).")
    ] = "BAAI/bge-small-en-v1.5",
    as_json: Annotated[bool, typer.Option("--json", help="Print JSON lines.")] = False,
) -> None:
    """Find the source passages that answer a question, and print them."""
    from graphwalk.index import Index  # noqa: PLC0415

    if mode not in ("graph", "dense", "hybrid"):
        typer.echo("mode must be graph, dense, or hybrid", err=True)
        raise typer.Exit(code=2)
    if graph.suffix.lower() not in SQLITE_SUFFIXES or not graph.exists():
        typer.echo(f"{graph}: not an existing SQLite graph (.db)", err=True)
        raise typer.Exit(code=2)

    async def run() -> None:
        decider = _make_backend() if mode != "dense" else None
        embedder = _locate_embedder(embed_model) if mode != "graph" else None
        index = Index(SQLiteStore(graph), decider=decider, embedder=embedder)
        try:
            locations = await index.locate(question, k, mode=mode)  # pyright: ignore[reportArgumentType]
            if not locations:
                typer.echo("no locations (the question names nothing the graph knows)")
            for rank, location in enumerate(locations, 1):
                passage = await index.read(location, context=context)
                if as_json:
                    typer.echo(passage.model_dump_json())
                    continue
                span = f"{passage.start}-{passage.end}"
                stale = " (STALE)" if passage.stale else ""
                typer.echo(f"{rank}. {location.key} [{span}] {location.title or ''}{stale}")
                for step in location.path:
                    typer.echo(f"   via {step}")
                typer.echo(f"   {' '.join(passage.text.split())}")
        finally:
            if decider is not None:
                await decider.aclose()
            await index.close()

    try:
        asyncio.run(run())
    except (GraphwalkError, ValueError, OSError) as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(code=1) from None


@app.command()
def migrate(
    source: Annotated[Path, typer.Argument(help="Graph JSON written by NetworkXStore.save.")],
    target: Annotated[Path, typer.Argument(help="SQLite database to create (.db).")],
    overwrite: Annotated[bool, typer.Option(help="Replace the target if it exists.")] = False,
) -> None:
    """Convert a NetworkX JSON graph to SQLite."""
    from graphwalk.stores.migrate import networkx_to_sqlite  # noqa: PLC0415

    try:
        done = asyncio.run(networkx_to_sqlite(source, target, overwrite=overwrite))
    except (FileExistsError, FileNotFoundError, ValueError, RuntimeError) as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(code=1) from None
    typer.echo(
        f"{target}: {done.nodes} nodes, {done.edges} edges, {done.metadata} metadata "
        f"entries, {done.documents} documents"
    )


@app.command()
def mcp(
    http: Annotated[
        bool, typer.Option("--http", help="Serve streamable HTTP instead of stdio.")
    ] = False,
    db: Annotated[
        Path | None, typer.Option(help="SQLite graph to serve (default: GRAPHWALK_DB).")
    ] = None,
    host: Annotated[str | None, typer.Option(help="HTTP bind address.")] = None,
    port: Annotated[int | None, typer.Option(help="HTTP port.")] = None,
) -> None:
    """Run the MCP server. stdio: provider keys from the environment. --http: keys from
    each client's headers; auth, limits, and allowlists from GRAPHWALK_* variables."""
    try:
        from graphwalk.server.app import run_http, run_stdio  # noqa: PLC0415
        from graphwalk.server.settings import ServerSettings  # noqa: PLC0415
    except ImportError:
        typer.echo("graphwalk mcp needs the 'mcp' extra: pip install 'graphwalk[mcp]'", err=True)
        raise typer.Exit(code=2) from None
    overrides = {k: v for k, v in {"db": db, "host": host, "port": port}.items() if v is not None}
    try:
        settings = ServerSettings(**overrides)  # pyright: ignore[reportArgumentType]
        if http:
            settings.check_remote()
    except ValueError as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(code=2) from None
    asyncio.run(run_http(settings) if http else run_stdio(settings))
