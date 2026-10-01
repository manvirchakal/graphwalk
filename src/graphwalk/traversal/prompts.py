"""Builds what the decision backend sees: shared ``state``, per-question ``instructions``,
and option descriptions.

Following TypeSafe's Jev 1.13 guidance (literal reading, weak at indirection, distracted
by irrelevant state): ``state`` is only the query; each question asks one *local*,
literally worded judgment about the current node; options carry real descriptions.
"""

from collections.abc import Mapping, Sequence
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

# stop_style="literal": state the stop condition first and exactly (Jev reads literally).
ENTITY_TASK_LITERAL = (
    "You are walking a knowledge graph to answer the query in the state. If the current "
    "node is the thing the query asks for, choose STOP. Otherwise choose the edge from the "
    "current node that moves one step closer to the answer."
)
RELATION_TASK_LITERAL = (
    "You are walking a knowledge graph to answer the query in the state. If the current "
    "nodes are the things the query asks for, choose STOP. Otherwise choose the relation "
    "to follow from the current nodes that moves one step closer to the answer."
)
ENTITY_STOP_LITERAL = "The current node is what the query asks for. Return it as the answer."
RELATION_STOP_LITERAL = "The current nodes are what the query asks for. Return them as the answer."

ANSWER_TYPE_KEY = "answer_type"
ANSWER_TYPE_TASK = "What type of thing does the query in the state ask for?"


@dataclass(frozen=True)
class PromptOptions:
    """Prompt-layout switches (see ``TraversalConfig``); defaults reproduce v1 prompts."""

    include_summary: bool = True
    preview: int = 5
    show_types: bool = False
    stop_style: Literal["v1", "literal"] = "v1"
    relation_glosses: Mapping[str, str] | None = None
    expected_answer_type: str | None = None


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
    return label_text(node.name, node.type, node.summary)


def label_text(name: str, type_: str, summary: str | None) -> str:
    """:func:`node_text` from the fields alone (stores can read them without the node)."""
    text = f"{name} ({type_})"
    return f"{text}: {summary}" if summary else text


def edge_text(current: str, relation: str, direction: Literal["out", "in"], other: str) -> str:
    """``A --rel--> B`` with the arrow in the stored direction of the edge."""
    if direction == "out":
        return f"{current} --{relation}--> {other}"
    return f"{other} --{relation}--> {current}"


def names_text(names: Sequence[str], limit: int) -> str:
    shown = ", ".join(names[:limit])
    extra = len(names) - limit
    return f"{{{shown}, +{extra} more}}" if extra > 0 else f"{{{shown}}}"


def _types(nodes: Sequence[Node]) -> list[JsonValue]:
    return list(dict.fromkeys(node.type for node in nodes))


def instructions_for(
    mode: HopMode,
    frontier: Sequence[Node],
    path: Sequence[str],
    options: PromptOptions,
) -> dict[str, JsonValue]:
    literal = options.stop_style == "literal"
    out: dict[str, JsonValue]
    if mode == "entity":
        out = {
            "task": ENTITY_TASK_LITERAL if literal else ENTITY_TASK,
            "current": node_card(frontier[0], include_summary=options.include_summary),
        }
    else:
        names: list[JsonValue] = [node.name for node in frontier[: max(options.preview, 1) * 4]]
        current: dict[str, JsonValue] = {"nodes": names, "count": len(frontier)}
        if options.show_types:
            current["types"] = _types(frontier)
        out = {"task": RELATION_TASK_LITERAL if literal else RELATION_TASK, "current": current}
    if path:
        out["path_so_far"] = list(path)
    if options.expected_answer_type is not None:
        out["the_query_asks_for_a"] = options.expected_answer_type
    return out


def _gloss(relation: str, options: PromptOptions) -> str | None:
    return None if options.relation_glosses is None else options.relation_glosses.get(relation)


def entity_option(
    current: Node, move: Move, target: Node, options: PromptOptions
) -> dict[str, JsonValue]:
    out: dict[str, JsonValue] = {
        "edge": edge_text(current.name, move.relation, move.direction, target.name),
        "node": node_card(target, include_summary=options.include_summary),
    }
    gloss = _gloss(move.relation, options)
    if gloss:
        out["relation_meaning"] = gloss
    return out


def relation_option(
    move: Move, targets: Sequence[Node], options: PromptOptions
) -> dict[str, JsonValue]:
    shown: list[JsonValue] = [node.name for node in targets[: options.preview]]
    out: dict[str, JsonValue] = {
        "relation": move.relation,
        "direction": "outgoing" if move.direction == "out" else "incoming",
        "leads_to": shown,
        "count": len(targets),
    }
    gloss = _gloss(move.relation, options)
    if gloss:
        out["relation_meaning"] = gloss
    if options.show_types:
        out["leads_to_types"] = _types(targets)
    return out


def stop_description(mode: HopMode, options: PromptOptions | None = None) -> str:
    if options is not None and options.stop_style == "literal":
        return ENTITY_STOP_LITERAL if mode == "entity" else RELATION_STOP_LITERAL
    return ENTITY_STOP if mode == "entity" else RELATION_STOP


def answer_type_options(types: Sequence[str]) -> dict[str, str]:
    """Options for the speculative answer-type question: one label per node type."""
    return {t: f"a {t}" for t in types}


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
