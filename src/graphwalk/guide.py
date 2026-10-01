"""The guide for coding agents, shipped inside the package.

``graphwalk guide`` prints it; ``graphwalk guide --skill`` prints a ``SKILL.md`` for
agents that load skills. The repository's ``llms.txt`` links to the same files.
"""

from importlib.resources import files
from typing import Literal

type GuideKind = Literal["guide", "skill"]

_FILES: dict[GuideKind, str] = {"guide": "guide.md", "skill": "SKILL.md"}


def guide(kind: GuideKind = "guide") -> str:
    """When to use graphwalk and how, as Markdown (``kind="skill"``: a SKILL.md)."""
    return files("graphwalk").joinpath("agent_docs", _FILES[kind]).read_text(encoding="utf-8")
