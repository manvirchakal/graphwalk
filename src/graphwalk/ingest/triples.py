"""Import an existing knowledge graph from triples: no extraction, no LLM.

Formats, chosen by file suffix (or ``fmt=``):

* ``.csv`` / ``.tsv``: a header naming the ``subject``, ``relation``, and ``object``
  columns (aliases: ``head``/``source``/``s``, ``predicate``/``type``/``p``,
  ``tail``/``target``/``o``), plus optional ``subject_type``, ``object_type``,
  ``subject_name``, and ``object_name``. Without a recognizable header, the first three
  columns are taken as subject, relation, object. ``.txt`` is ``|``-separated with no
  header (MetaQA's ``kb.txt``).
* ``.jsonl``: one object per line with the same keys.
* ``.nt``: N-Triples. IRIs become node ids; ``rdfs:label`` (or ``schema:name``,
  ``foaf:name``) sets a node's name and ``rdf:type`` its type, instead of becoming
  edges. Other literal objects become ``literal`` nodes, so a walk can end on a date or
  a number. A name defaults to the IRI's last path segment or fragment.

Node ids are the subject and object strings as given; names default to the ids. A node
seen with several types keeps the first one given. Re-importing the same file is
idempotent (edge ids are derived from their triple).
"""

import csv
import hashlib
import json
import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, cast, runtime_checkable

from graphwalk.core.hashing import content_hash
from graphwalk.core.model import Edge, Node, Provenance, utc_now
from graphwalk.stores.base import GraphStore

type TripleFormat = Literal["csv", "tsv", "pipe", "jsonl", "nt"]

DEFAULT_TYPE = "entity"
LITERAL_TYPE = "literal"

_SUFFIX_FORMATS: dict[str, TripleFormat] = {
    ".csv": "csv",
    ".tsv": "tsv",
    ".txt": "pipe",
    ".jsonl": "jsonl",
    ".nt": "nt",
}
_ALIASES: dict[str, tuple[str, ...]] = {
    "subject": ("subject", "head", "source", "s", "subj", "from"),
    "relation": ("relation", "predicate", "type", "p", "rel", "property", "edge"),
    "object": ("object", "tail", "target", "o", "obj", "to"),
}
_OPTIONAL = ("subject_type", "object_type", "subject_name", "object_name")
_LABELS = frozenset({
    "http://www.w3.org/2000/01/rdf-schema#label",
    "http://schema.org/name",
    "https://schema.org/name",
    "http://xmlns.com/foaf/0.1/name",
})  # fmt: skip
_RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"


@runtime_checkable
class BulkStore(Protocol):
    async def upsert_many(self, nodes: Sequence[Node], edges: Sequence[Edge]) -> None: ...


@dataclass(frozen=True)
class Triple:
    subject: str
    relation: str
    object: str
    subject_type: str | None = None
    object_type: str | None = None
    subject_name: str | None = None
    object_name: str | None = None


@dataclass(frozen=True)
class ImportReport:
    nodes: int
    edges: int
    skipped: int
    """Malformed lines or rows (missing a field, or not parseable)."""


class TripleFormatError(ValueError):
    """The file's format could not be determined or read."""


def detect_format(path: Path) -> TripleFormat:
    fmt = _SUFFIX_FORMATS.get(path.suffix.lower())
    if fmt is None:
        msg = (
            f"{path}: unknown triples format {path.suffix!r} "
            f"(expected one of {sorted(_SUFFIX_FORMATS)}, or pass a format)"
        )
        raise TripleFormatError(msg)
    return fmt


# ---------------------------------------------------------------------- readers


@dataclass
class _Counter:
    skipped: int = 0


def _clean(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _columns(header: list[str]) -> dict[str, int] | None:
    """Field -> column index, or ``None`` if the header names no subject/relation/object."""
    lowered = [h.strip().lower() for h in header]
    found: dict[str, int] = {}
    for field, aliases in _ALIASES.items():
        for alias in aliases:
            if alias in lowered:
                found[field] = lowered.index(alias)
                break
    if len(found) < len(_ALIASES):
        return None
    for field in _OPTIONAL:
        if field in lowered:
            found[field] = lowered.index(field)
    return found


def _from_fields(fields: Mapping[str, object]) -> Triple | None:
    s, r, o = (_clean(fields.get(k)) for k in ("subject", "relation", "object"))
    if s is None or r is None or o is None:
        return None
    return Triple(s, r, o, *(_clean(fields.get(k)) for k in _OPTIONAL))


def _read_delimited(path: Path, delimiter: str, counter: _Counter) -> Iterator[Triple]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = csv.reader(handle, delimiter=delimiter)
        first = next(rows, None)
        if first is None:
            return
        columns = None if delimiter == "|" else _columns(first)
        if columns is None:
            columns = {"subject": 0, "relation": 1, "object": 2}
            rows = _chain([first], rows)
        width = max(columns.values()) + 1
        for row in rows:
            if not row or all(not cell.strip() for cell in row):
                continue
            if len(row) < width:
                counter.skipped += 1
                continue
            triple = _from_fields({k: row[i] for k, i in columns.items()})
            if triple is None:
                counter.skipped += 1
                continue
            yield triple


def _chain[T](head: list[T], rest: Iterator[T]) -> Iterator[T]:
    yield from head
    yield from rest


def _read_jsonl(path: Path, counter: _Counter) -> Iterator[Triple]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                counter.skipped += 1
                continue
            if not isinstance(data, dict):
                counter.skipped += 1
                continue
            record = cast("dict[str, object]", data)
            fields = {
                field: next((record[a] for a in aliases if a in record), None)
                for field, aliases in _ALIASES.items()
            }
            fields.update({k: record.get(k) for k in _OPTIONAL})
            triple = _from_fields(fields)
            if triple is None:
                counter.skipped += 1
                continue
            yield triple


_IRI = r"<([^>]*)>"
_BNODE = r"(_:[A-Za-z0-9_\-.]+)"
_LITERAL = r'"((?:[^"\\]|\\.)*)"(?:@[A-Za-z0-9\-]+|\^\^<[^>]*>)?'
_NT_LINE = re.compile(
    rf"^\s*(?:{_IRI}|{_BNODE})\s+{_IRI}\s+(?:{_IRI}|{_BNODE}|{_LITERAL})\s*\.\s*(?:#.*)?$"
)
_ESCAPES = {"t": "\t", "n": "\n", "r": "\r", '"': '"', "\\": "\\", "b": "\b", "f": "\f"}


def _unescape(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        code = match.group(1)
        if code[0] in "uU":
            return chr(int(code[1:], 16))
        return _ESCAPES.get(code, code)

    return re.sub(r"\\(u[0-9A-Fa-f]{4}|U[0-9A-Fa-f]{8}|.)", replace, text)


def local_name(iri: str) -> str:
    """The readable tail of an IRI: after the last ``#`` or ``/`` (or the IRI itself)."""
    tail = re.split(r"[#/]", iri.rstrip("/#"))[-1]
    return tail or iri


def literal_id(value: str) -> str:
    return "lit:" + hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]


def _read_nt(path: Path, counter: _Counter) -> Iterator[Triple]:
    """Two passes: names and types first (they may come after the edges that use them)."""
    names: dict[str, str] = {}
    types: dict[str, str] = {}
    parsed: list[tuple[str, str, str | None, str | None]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            match = _NT_LINE.match(line)
            if match is None:
                counter.skipped += 1
                continue
            s_iri, s_bnode, predicate, o_iri, o_bnode, o_literal = match.groups()
            subject = s_iri if s_iri is not None else s_bnode
            obj = o_iri if o_iri is not None else o_bnode
            literal = None if o_literal is None else _unescape(o_literal)
            if predicate in _LABELS and literal is not None:
                names.setdefault(subject, literal)
            elif predicate == _RDF_TYPE and obj is not None:
                types.setdefault(subject, local_name(obj))
            else:
                parsed.append((subject, predicate, obj, literal))
    for subject, predicate, obj, literal in parsed:
        s_name = names.get(subject, local_name(subject))
        relation = local_name(predicate)
        if literal is not None:
            yield Triple(
                subject, relation, literal_id(literal), types.get(subject), LITERAL_TYPE,
                s_name, literal,
            )  # fmt: skip
        else:
            assert obj is not None  # noqa: S101 - the regex matched an IRI or blank node
            yield Triple(
                subject, relation, obj, types.get(subject), types.get(obj), s_name,
                names.get(obj, local_name(obj)),
            )  # fmt: skip


def read_triples(
    path: str | Path, *, fmt: TripleFormat | None = None, counter: _Counter | None = None
) -> Iterator[Triple]:
    """Triples from ``path`` (format from the suffix unless ``fmt`` is given)."""
    path = Path(path)
    counter = counter or _Counter()
    fmt = fmt or detect_format(path)
    if fmt == "csv":
        return _read_delimited(path, ",", counter)
    if fmt == "tsv":
        return _read_delimited(path, "\t", counter)
    if fmt == "pipe":
        return _read_delimited(path, "|", counter)
    if fmt == "jsonl":
        return _read_jsonl(path, counter)
    return _read_nt(path, counter)


# ---------------------------------------------------------------------- import


async def import_triples(
    store: GraphStore,
    triples: Iterable[Triple],
    *,
    source_id: str,
    default_type: str = DEFAULT_TYPE,
    batch_size: int = 5000,
) -> ImportReport:
    """Write ``triples`` into ``store``: nodes first, then edges, in batches.

    Every node and edge gets one provenance record naming ``source_id`` (confidence 1).
    Nodes already in the store keep their stored version (an import never overwrites
    a node's name, type, or summary).
    """
    provenance = (
        Provenance(
            source_id=source_id,
            ingested_at=utc_now(),
            confidence=1.0,
            content_hash=content_hash(source_id),
        ),
    )
    nodes: dict[str, tuple[str, str | None]] = {}
    edges: dict[tuple[str, str, str], None] = {}

    def see(node_id: str, name: str | None, type_: str | None) -> None:
        known = nodes.get(node_id)
        if known is None:
            nodes[node_id] = (name or node_id, type_)
        elif known[1] is None and type_ is not None:
            nodes[node_id] = (known[0], type_)

    for t in triples:
        see(t.subject, t.subject_name, t.subject_type)
        see(t.object, t.object_name, t.object_type)
        edges[(t.subject, t.relation, t.object)] = None

    existing: set[str] = set()
    ids = list(nodes)
    for start in range(0, len(ids), batch_size):
        existing.update(await store.get_nodes(ids[start : start + batch_size]))
    new_nodes = [
        Node(id=node_id, type=type_ or default_type, name=name, provenance=provenance)
        for node_id, (name, type_) in nodes.items()
        if node_id not in existing
    ]
    # Every node before any edge: an edge's endpoints must exist.
    for start in range(0, len(new_nodes), batch_size):
        await _write(store, new_nodes[start : start + batch_size], ())
    keys = list(edges)
    for start in range(0, len(keys), batch_size):
        batch = [
            Edge(source=s, target=o, type=r, provenance=provenance)
            for s, r, o in keys[start : start + batch_size]
        ]  # built per batch: a million Edge models at once is gigabytes
        await _write(store, (), batch)
    return ImportReport(nodes=len(new_nodes), edges=len(keys), skipped=0)


async def _write(store: GraphStore, nodes: Sequence[Node], edges: Sequence[Edge]) -> None:
    """One transaction per batch where the store supports it (SQLite)."""
    if isinstance(store, BulkStore):
        await store.upsert_many(nodes, edges)
        return
    for node in nodes:
        await store.upsert_node(node)
    for edge in edges:
        await store.upsert_edge(edge)


async def import_file(
    store: GraphStore,
    path: str | Path,
    *,
    fmt: TripleFormat | None = None,
    source_id: str | None = None,
    default_type: str = DEFAULT_TYPE,
) -> ImportReport:
    """Read ``path`` and import it; ``source_id`` defaults to the file name."""
    path = Path(path)
    counter = _Counter()
    triples = list(read_triples(path, fmt=fmt, counter=counter))
    report = await import_triples(
        store, triples, source_id=source_id or path.name, default_type=default_type
    )
    return ImportReport(nodes=report.nodes, edges=report.edges, skipped=counter.skipped)
