"""Builds what the decision backend sees: shared ``state``, per-question ``instructions``,
and option descriptions.

Following TypeSafe's Jev 1.13 guidance (literal reading, weak at indirection, distracted
by irrelevant state): ``state`` is only the query; each question asks one *local*,
literally worded judgment about the current node; options carry real descriptions.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from pydantic import JsonValue

from graphwalk.core.model import Node, NodeId
from graphwalk.traversal.config import HopMode, LabelStyle

STOP = "STOP"

ENTITY_TASK = (
    "You are walking a knowledge graph to answer the query in the state. Choose the edge "
    "from the current node that moves one step closer to the answer. Choose STOP only if "
    "the current node itself is the answer to the query."
)
RELATION_TASK = (
    "You are walking a knowledge graph to answer the query in the state. Choose the "
    "relation to follow from the current nodes that moves one step closer to the answer. "
    "Choose STOP only if the current nodes themselves are the answer to the query."
)
ENTITY_STOP = "The current node itself is the answer to the query; stop here."
RELATION_STOP = "The current nodes themselves are the answer to the query; stop here."


@dataclass(frozen=True)
class Move:
    """A candidate hop. Entity mode: exactly one target."""

    relation: str
    direction: Literal["out", "in"]
    targets: tuple[NodeId, ...]


def state_for(query: str) -> dict[str, JsonValue]:
    return {"query": query}


def node_card(node: Node, *, include_summary: bool) -> dict[str, JsonValue]:
    card: dict[str, JsonValue] = {"name": node.name, "type": node.type}
    if include_summary and node.summary:
        card["summary"] = node.summary
    return card


def node_text(node: Node) -> str:
    """Plain text for embedding a node."""
    text = f"{node.name} ({node.type})"
    return f"{text}: {node.summary}" if node.summary else text


def edge_text(current: str, relation: str, direction: Literal["out", "in"], other: str) -> str:
    """``A --rel--> B`` with the arrow in the stored direction of the edge."""
    if direction == "out":
        return f"{current} --{relation}--> {other}"
    return f"{other} --{relation}--> {current}"


def names_text(names: Sequence[str], limit: int) -> str:
    shown = ", ".join(names[:limit])
    extra = len(names) - limit
    return f"{{{shown}, +{extra} more}}" if extra > 0 else f"{{{shown}}}"


def instructions_for(
    mode: HopMode,
    frontier: Sequence[Node],
    path: Sequence[str],
    *,
    include_summary: bool,
    preview: int,
) -> dict[str, JsonValue]:
    out: dict[str, JsonValue]
    if mode == "entity":
        out = {
            "task": ENTITY_TASK,
            "current": node_card(frontier[0], include_summary=include_summary),
        }
    else:
        names: list[JsonValue] = [node.name for node in frontier[: max(preview, 1) * 4]]
        out = {"task": RELATION_TASK, "current": {"nodes": names, "count": len(frontier)}}
    if path:
        out["path_so_far"] = list(path)
    return out


def entity_option(
    current: Node, move: Move, target: Node, *, include_summary: bool
) -> dict[str, JsonValue]:
    return {
        "edge": edge_text(current.name, move.relation, move.direction, target.name),
        "node": node_card(target, include_summary=include_summary),
    }


def relation_option(move: Move, target_names: Sequence[str], preview: int) -> dict[str, JsonValue]:
    shown: list[JsonValue] = list(target_names[:preview])
    return {
        "relation": move.relation,
        "direction": "outgoing" if move.direction == "out" else "incoming",
        "leads_to": shown,
        "count": len(target_names),
    }


def stop_description(mode: HopMode) -> str:
    return ENTITY_STOP if mode == "entity" else RELATION_STOP


def readable_label(move: Move, target_name: str | None) -> str:
    arrow = "->" if move.direction == "out" else "<-"
    if target_name is None:  # relation mode
        return move.relation if move.direction == "out" else f"{move.relation} (inverse)"
    return f"{move.relation} {arrow} {target_name}"


def make_labels(
    style: LabelStyle, moves: Sequence[Move], target_names: Sequence[str | None]
) -> list[str]:
    """One unique label per move, never equal to STOP."""
    if style == "opaque":
        return [f"o{i + 1}" for i in range(len(moves))]
    labels: list[str] = []
    seen = {STOP}
    for move, name in zip(moves, target_names, strict=True):
        base = readable_label(move, name)
        label, n = base, 1
        while label in seen:
            n += 1
            label = f"{base} #{n}"
        seen.add(label)
        labels.append(label)
    return labels
