import pytest
from typer.testing import CliRunner

from graphwalk import __version__
from graphwalk.cli import app

runner = CliRunner()


def test_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.output.strip() == __version__


@pytest.mark.parametrize("command", ["ingest", "query", "eval"])
def test_commands_exist_and_report_not_implemented(command: str) -> None:
    result = runner.invoke(app, [command])
    assert result.exit_code == 2
    assert "not implemented yet" in result.output
