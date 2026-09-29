"""LLM extraction: chunk text -> candidate entities and relations (validated JSON)."""

import json
import re
from dataclasses import dataclass, field

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from graphwalk.ingest.spend import Spend
from graphwalk.llm.base import LLMBackend, LLMError, Message

EXTRACTOR_VERSION = "1"
"""Bump when the prompt or parsing changes: it is part of each document's ledger hash,
so a new extractor re-ingests everything."""
EVIDENCE_EXTRACTOR_VERSION = "2"
"""The prompt that also asks for supporting evidence (``evidence=True``)."""

GENERIC_TYPE = "entity"

type Scalar = str | int | float | bool

EXTRACT_SYSTEM = """\
You extract a knowledge graph from a passage. Reply with one JSON object and nothing else:
{"entities": [{"name": str, "type": str, "description": str, "attributes": {str: str|number}}],
 "relations": [{"source": str, "type": str, "target": str}]}
Rules:
- Entities are specific, named things: people, organizations, places, works, events,
  products, and so on. Skip generic nouns and pronouns.
- "name": the most complete proper name used in the passage (e.g. "Marie Curie", not
  "Curie" or "she").
- "type": one lowercase word or snake_case phrase, e.g. person, film, city, country,
  organization, award, band, album, book.
- "description": one short sentence about the entity, from the passage only.
- "attributes": literal facts about the entity only (dates, numbers, short strings),
  e.g. {"date_of_birth": "1867-11-07"}. Use ISO dates when the passage gives a full date.
- Relations connect two entities by their exact "name". "type" is a lowercase
  snake_case verb phrase read from source to target, e.g. directed_by, born_in,
  spouse_of, member_of, located_in, award_received.
- Use only facts stated in the passage. Do not guess."""

EXTRACT_SYSTEM_EVIDENCE = (
    EXTRACT_SYSTEM.replace(
        '"attributes": {str: str|number}}]',
        '"attributes": {str: str|number}, "evidence": str}]',
    ).replace(
        '"target": str}]}',
        '"target": str, "evidence": str}]}',
    )
    + """
- "evidence": the shortest sentence or clause of the passage that states the entity or
  relation, copied exactly, character for character."""
)


def extractor_version(*, evidence: bool) -> str:
    return EVIDENCE_EXTRACTOR_VERSION if evidence else EXTRACTOR_VERSION


def snake(text: str) -> str:
    """``"Directed By"`` -> ``"directed_by"``; empty for text with no letters or digits."""
    return re.sub(r"[^0-9a-z]+", "_", text.strip().lower()).strip("_")


class ExtractedEntity(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    name: str = Field(min_length=1)
    type: str = GENERIC_TYPE
    description: str | None = None
    attributes: dict[str, Scalar] = Field(default_factory=dict[str, Scalar])
    evidence: str | None = None
    """The passage text the entity was read from, as quoted by the model (unverified)."""

    @field_validator("name")
    @classmethod
    def _strip_name(cls, name: str) -> str:
        return " ".join(name.split())

    @field_validator("type")
    @classmethod
    def _snake_type(cls, type_: str) -> str:
        return snake(type_) or GENERIC_TYPE


class ExtractedRelation(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    source: str = Field(min_length=1)
    type: str = Field(min_length=1)
    target: str = Field(min_length=1)
    evidence: str | None = None
    """The passage text stating the relation, as quoted by the model (unverified)."""

    @field_validator("source", "target")
    @classmethod
    def _strip(cls, name: str) -> str:
        return " ".join(name.split())


class Extraction(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    entities: tuple[ExtractedEntity, ...] = ()
    relations: tuple[ExtractedRelation, ...] = ()


@dataclass
class ExtractionResult:
    extraction: Extraction
    spend: Spend = field(default_factory=Spend)
    error: str | None = None
    """Set when every attempt failed; ``extraction`` is then empty."""
    dropped: list[str] = field(default_factory=list[str])
    """Relations dropped during cleanup, as ``source -type-> target``."""


def name_key(name: str) -> str:
    """Case- and whitespace-insensitive name key."""
    return " ".join(name.casefold().split())


def parse_extraction(text: str) -> Extraction:
    """Parse the model's reply: the outermost JSON object, code fences allowed."""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        msg = "no JSON object in the reply"
        raise ValueError(msg)
    return Extraction.model_validate(json.loads(text[start : end + 1]))


def clean(extraction: Extraction) -> tuple[Extraction, list[str]]:
    """Merge duplicate entities, normalize relation types, and repair relations.

    A relation endpoint not listed as an entity is added as a generic entity, so the
    fact is kept. Self-loops and relations with no usable type are dropped.
    """
    entities: dict[str, ExtractedEntity] = {}
    for entity in extraction.entities:
        key = name_key(entity.name)
        kept = entities.get(key)
        if kept is None:
            entities[key] = entity
            continue
        entities[key] = kept.model_copy(
            update={
                "type": kept.type if kept.type != GENERIC_TYPE else entity.type,
                "description": kept.description or entity.description,
                "attributes": {**entity.attributes, **kept.attributes},
                "evidence": kept.evidence or entity.evidence,
            }
        )
    relations: dict[tuple[str, str, str], ExtractedRelation] = {}
    dropped: list[str] = []
    for relation in extraction.relations:
        type_ = snake(relation.type)
        source, target = name_key(relation.source), name_key(relation.target)
        if not type_ or source == target:
            dropped.append(f"{relation.source} -{relation.type}-> {relation.target}")
            continue
        for name, key in ((relation.source, source), (relation.target, target)):
            if key not in entities:
                entities[key] = ExtractedEntity(name=name)
        relations.setdefault(
            (source, type_, target),
            ExtractedRelation(
                source=entities[source].name,
                type=type_,
                target=entities[target].name,
                evidence=relation.evidence,
            ),
        )
    cleaned = Extraction(entities=tuple(entities.values()), relations=tuple(relations.values()))
    return cleaned, dropped


def extraction_messages(chunk: str, title: str | None, *, evidence: bool = False) -> list[Message]:
    header = f"Title: {title}\n\n" if title else ""
    return [
        Message(role="system", content=EXTRACT_SYSTEM_EVIDENCE if evidence else EXTRACT_SYSTEM),
        Message(role="user", content=f"{header}Passage:\n{chunk}"),
    ]


async def extract(
    llm: LLMBackend,
    chunk: str,
    *,
    title: str | None = None,
    attempts: int = 2,
    evidence: bool = False,
) -> ExtractionResult:
    """Extract from one chunk. A reply that fails to parse is retried with the error.

    With ``evidence``, the prompt also asks for the passage text supporting each entity
    and relation (see :func:`find_evidence`)."""
    messages = extraction_messages(chunk, title, evidence=evidence)
    result = ExtractionResult(extraction=Extraction())
    error = "no attempts made"
    for _ in range(max(1, attempts)):
        try:
            reply = await llm.complete(messages)
        except LLMError as exc:
            error = f"LLM call failed: {exc}"
            continue
        result.spend.llm_calls += 1
        result.spend.input_tokens += reply.input_tokens
        result.spend.output_tokens += reply.output_tokens
        result.spend.add_cost(reply.cost_usd)
        try:
            parsed = parse_extraction(reply.text)
        except (ValueError, ValidationError) as exc:
            error = f"unparsable reply: {str(exc)[:300]}"
            messages = [
                *messages,
                Message(role="assistant", content=reply.text),
                Message(
                    role="user",
                    content=f"That reply was not valid ({error}). Reply with the JSON object only.",
                ),
            ]
            continue
        result.extraction, result.dropped = clean(parsed)
        return result
    result.error = error
    return result


_QUOTES = "\"'`\u201c\u201d\u2018\u2019"
MIN_EVIDENCE_CHARS = 4
"""Shorter quotes ("he", "it") match too easily to be trusted."""


def _folded(text: str) -> tuple[str, list[int]]:
    """``text`` lowercased with whitespace runs collapsed to one space, plus the offset
    in ``text`` of each output character."""
    out: list[str] = []
    index: list[int] = []
    space = False
    for i, char in enumerate(text):
        if char.isspace():
            space = bool(out)
            continue
        if space:
            out.append(" ")
            index.append(i - 1)
            space = False
        lower = char.lower()
        out.append(lower if len(lower) == 1 else char)
        index.append(i)
    return "".join(out), index


def find_evidence(text: str, quote: str | None) -> tuple[int, int] | None:
    """Where ``quote`` occurs in ``text``, as ``[start, end)``; ``None`` if it does not.

    Tries an exact match, then one ignoring case and whitespace differences. Surrounding
    quote marks and whitespace are ignored, as is a trailing ellipsis.
    """
    if quote is None:
        return None
    wanted = quote.strip().strip(_QUOTES).strip().removesuffix("...").removesuffix("\u2026")
    wanted = wanted.strip()
    if len(wanted) < MIN_EVIDENCE_CHARS:
        return None
    at = text.find(wanted)
    if at >= 0:
        return at, at + len(wanted)
    folded, index = _folded(text)
    needle, _ = _folded(wanted)
    at = folded.find(needle)
    if at < 0:
        return None
    return index[at], index[at + len(needle) - 1] + 1
