"""Command-line interface: ``graphwalk ingest | query | eval``."""

import asyncio
import math
from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError

from graphwalk import __version__
from graphwalk.config import GraphwalkSettings
from graphwalk.core.errors import GraphwalkError
from graphwalk.decisions import DecisionBackend
from graphwalk.stores.networkx_store import NetworkXStore
from graphwalk.traversal import NameEntryResolver, TraversalConfig, TraversalResult, Traverser

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


@app.command("eval")
def eval_() -> None:
    """Run an evaluation suite."""
    _not_implemented("eval", "M5")
