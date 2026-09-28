"""Command-line interface: ``graphwalk ingest | query | eval``."""

from typing import Annotated

import typer

from graphwalk import __version__

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


@app.command()
def query() -> None:
    """Answer a query by traversing the graph."""
    _not_implemented("query", "M3")


@app.command("eval")
def eval_() -> None:
    """Run an evaluation suite."""
    _not_implemented("eval", "M5")
