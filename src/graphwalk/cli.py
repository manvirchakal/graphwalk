"""Command-line interface: ``graphwalk ingest | query | eval``."""

import asyncio
import math
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer
from pydantic import ValidationError

from graphwalk import __version__
from graphwalk.config import GraphwalkSettings
from graphwalk.core.errors import GraphwalkError
from graphwalk.decisions import DecisionBackend
from graphwalk.stores.networkx_store import NetworkXStore
from graphwalk.traversal import NameEntryResolver, TraversalConfig, TraversalResult, Traverser

if TYPE_CHECKING:
    from graphwalk.eval.suite import Factories

app = typer.Typer(
    name="graphwalk",
    help="Fast, probabilistic knowledge-graph traversal and ingestion.",
    no_args_is_help=True,
)


def _not_implemented(command: str, milestone: str) -> None:
    typer.echo(f"`graphwalk {command}` is not implemented yet (planned for {milestone}).", err=True)
    raise typer.Exit(code=2)


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


@app.command()
def ingest() -> None:
    """Ingest a data source into the graph."""
    _not_implemented("ingest", "M6")


def _make_backend() -> DecisionBackend:
    """The decision backend for CLI commands (replaced in tests)."""
    from graphwalk.decisions.jev import JevBackend  # noqa: PLC0415 - loads the SDK lazily

    return JevBackend.from_settings(GraphwalkSettings())


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


@app.command("eval")
def eval_(  # noqa: PLR0917 - Typer maps parameters to CLI options
    dataset: Annotated[str, typer.Argument(help="metaqa | 2wiki")],
    system: Annotated[
        list[str] | None,
        typer.Option(
            "--system",
            help="Repeatable: greedy, beam, relation, relation-beam, sample, rag.",
        ),
    ] = None,
    hops: Annotated[list[int] | None, typer.Option(help="MetaQA hops (repeatable).")] = None,
    n: Annotated[int, typer.Option(help="Seeded subset size (0 = all).")] = 100,
    seed: Annotated[int, typer.Option(help="Subset seed.")] = 0,
    linking: Annotated[
        str | None, typer.Option(help="given | gold | resolve (default per dataset).")
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

    if dataset not in ("metaqa", "2wiki"):
        typer.echo("dataset must be metaqa or 2wiki", err=True)
        raise typer.Exit(code=2)
    systems = system or ["greedy", "rag"]
    bad = [s for s in systems if s not in SYSTEMS]
    if bad or (linking is not None and linking not in ("given", "gold", "resolve")):
        typer.echo(f"unknown system(s) {bad} or linking {linking!r}", err=True)
        raise typer.Exit(code=2)
    factories = _eval_factories(llm_model, embed_model, llm_rpm)

    def progress(name: str, done: int, total: int) -> None:
        if done == total or done % 25 == 0:
            typer.echo(f"  {name}: {done}/{total}", err=True)

    for hop in (hops or [1]) if dataset == "metaqa" else [0]:
        path, _runs = asyncio.run(
            run_dataset(
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
        )
        typer.echo((path / "summary.md").read_text(encoding="utf-8"))
        typer.echo(f"results: {path}")
