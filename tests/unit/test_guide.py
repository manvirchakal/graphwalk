"""The agent guide ships in the package and stays in step with the code."""

import re
from pathlib import Path

from typer.testing import CliRunner

import graphwalk
from graphwalk import Index, TraversalConfig, guide
from graphwalk.cli import app
from graphwalk.eval.suite import preset_config
from kg_fixtures import movie_store, oracle

ROOT = Path(__file__).resolve().parents[2]


def test_guide_and_skill_are_packaged() -> None:
    assert guide().startswith("# graphwalk")
    assert guide("skill").startswith("---\nname: graphwalk\n")


def test_cli_prints_the_guide() -> None:
    runner = CliRunner()
    assert runner.invoke(app, ["guide"]).output == guide()
    assert runner.invoke(app, ["guide", "--skill"]).output == guide("skill")


def test_skill_frontmatter_is_valid() -> None:
    match = re.match(r"---\n(.*?)\n---\n", guide("skill"), re.DOTALL)
    assert match is not None
    fields = dict(line.split(": ", 1) for line in match.group(1).splitlines())
    assert set(fields) == {"name", "description"}
    assert re.fullmatch(r"[a-z0-9-]{1,64}", fields["name"])
    assert 0 < len(fields["description"]) <= 1024  # the skill spec's limit


def test_guide_lists_exactly_the_public_api() -> None:
    section = guide().split("## API surface", 1)[1].split("`Index` methods", 1)[0]
    listed = set(re.findall(r"`(\w+)`", section)) - {"graphwalk"}
    assert listed == set(graphwalk.__all__) - {"__version__"}


def test_guide_names_only_real_index_methods_and_cli_commands() -> None:
    text = guide()
    methods = re.findall(r"`(\w+)\(", text.split("`Index` methods:", 1)[1].split("\n\n")[0])
    assert methods
    assert all(hasattr(Index, m) for m in methods)
    commands = {c.name for c in app.registered_commands}
    cli = text.split("CLI: ", 1)[1].split("\n\n")[0]
    assert set(re.findall(r"`graphwalk (\w+)`", cli)) <= commands


def test_llms_txt_links_resolve() -> None:
    text = (ROOT / "llms.txt").read_text(encoding="utf-8")
    assert text.startswith("# graphwalk\n\n> ")
    links = re.findall(r"\]\(([^)]+)\)", text)
    assert links
    assert all((ROOT / link).is_file() for link in links)


def test_kgqa_config_is_the_measured_preset() -> None:
    assert TraversalConfig.kgqa() == preset_config("relation-v2")
    assert TraversalConfig.kgqa(max_frontier=10).max_frontier == 10


async def test_index_passes_node_types_for_the_answer_type_hint() -> None:
    store = await movie_store()
    decider = oracle({})
    typed = Index(
        store, decider=decider, traversal=TraversalConfig.kgqa(), node_types=["film", "person"]
    )
    untyped = Index(store, decider=decider, traversal=TraversalConfig.kgqa(), owns_store=False)
    for index, expected in ((typed, "hint"), (untyped, "off")):
        locator = index._graph_locator()  # pyright: ignore[reportPrivateUsage]
        traverser = locator._traverser  # pyright: ignore[reportPrivateUsage]
        assert traverser.config.answer_type == expected  # pyright: ignore[reportAttributeAccessIssue]
    await untyped.close()
    await typed.close()
