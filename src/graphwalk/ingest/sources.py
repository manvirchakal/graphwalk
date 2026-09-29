"""Sources: where documents come from.

A :class:`Source` yields :class:`SourceDocument` s with ids that are stable across runs,
so the ingestion ledger can tell new, changed, unchanged, and removed documents apart.
"""

import csv
import json
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast, runtime_checkable

TEXT_SUFFIXES = frozenset({".txt", ".md", ".markdown"})
RECORD_SUFFIXES = frozenset({".json", ".jsonl", ".csv"})
SUFFIXES = TEXT_SUFFIXES | RECORD_SUFFIXES

_TEXT_KEYS = ("text", "content", "body", "paragraph")
_TITLE_KEYS = ("title", "name", "heading")
_ID_KEYS = ("id", "_id", "doc_id", "uuid")


@dataclass(frozen=True)
class SourceDocument:
    doc_id: str
    """Unique and stable within its source."""
    text: str
    title: str | None = None


@runtime_checkable
class Source(Protocol):
    @property
    def source_id(self) -> str:
        """Names the source in the ledger and in provenance."""
        ...

    def documents(self) -> Iterable[SourceDocument]: ...


class TextSource:
    """Documents given in memory."""

    def __init__(self, source_id: str, documents: Iterable[SourceDocument]) -> None:
        self._source_id = source_id
        self._documents = list(documents)
        ids = [d.doc_id for d in self._documents]
        if len(set(ids)) != len(ids):
            msg = f"source {source_id!r}: duplicate document ids"
            raise ValueError(msg)

    @property
    def source_id(self) -> str:
        return self._source_id

    def documents(self) -> list[SourceDocument]:
        return list(self._documents)


class FileSource:
    """A file or a directory tree of ``.txt``/``.md`` (one document per file) and
    ``.json``/``.jsonl``/``.csv`` (one document per record).

    A record's text is its ``text``/``content``/``body`` field if it has one, else all its
    fields as ``key: value`` lines. Its id is its ``id`` field if present, else its index.
    Document ids are ``<path relative to the root>[#<record id>]``.
    """

    def __init__(self, path: str | Path, *, source_id: str | None = None) -> None:
        self._path = Path(path)
        if not self._path.exists():
            msg = f"{self._path}: no such file or directory"
            raise FileNotFoundError(msg)
        self._source_id = source_id or f"file:{self._path.resolve().as_posix()}"

    @property
    def source_id(self) -> str:
        return self._source_id

    def _files(self) -> list[tuple[Path, str]]:
        if self._path.is_file():
            return [(self._path, self._path.name)]
        files = sorted(
            p for p in self._path.rglob("*") if p.is_file() and p.suffix.lower() in SUFFIXES
        )
        return [(p, p.relative_to(self._path).as_posix()) for p in files]

    def documents(self) -> Iterator[SourceDocument]:
        for path, name in self._files():
            yield from _file_documents(path, name)

    def document(self, doc_id: str) -> SourceDocument | None:
        """Re-read one document by id, parsing only the file it lives in; ``None`` if it
        no longer exists."""
        candidates = [doc_id]
        if "#" in doc_id:  # ``<file>#<record id>``; the file name may contain '#' too
            candidates.append(doc_id.rsplit("#", 1)[0])
        for name in candidates:
            path = self._resolve(name)
            if path is None:
                continue
            for doc in _file_documents(path, name):
                if doc.doc_id == doc_id:
                    return doc
        return None

    def _resolve(self, name: str) -> Path | None:
        """The file named ``name`` in this source, if it exists inside the root."""
        if self._path.is_file():
            return self._path if name == self._path.name else None
        root = self._path.resolve()
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            return None
        return path if path.suffix.lower() in SUFFIXES else None


def _file_documents(path: Path, name: str) -> Iterator[SourceDocument]:
    suffix = path.suffix.lower()
    if suffix in TEXT_SUFFIXES:
        text = path.read_text(encoding="utf-8")
        yield SourceDocument(doc_id=name, text=text, title=path.stem)
    elif suffix == ".csv":
        with path.open(encoding="utf-8", newline="") as fh:
            yield from _records(name, csv.DictReader(fh))
    elif suffix == ".jsonl":
        with path.open(encoding="utf-8") as fh:
            rows = [json.loads(line) for line in fh if line.strip()]
        yield from _records(name, rows)
    elif suffix == ".json":
        with path.open(encoding="utf-8") as fh:
            loaded: Any = json.load(fh)
        rows = cast("list[Any]", loaded) if isinstance(loaded, list) else [loaded]
        yield from _records(name, rows)
    else:
        msg = f"{path}: unsupported file type {suffix!r}"
        raise ValueError(msg)


def _first(record: Mapping[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = record.get(key)
        if isinstance(value, str | int | float) and str(value).strip():
            return str(value).strip()
    return None


def record_text(record: Mapping[str, Any]) -> str:
    """A record as text: its text field, or ``key: value`` lines for every field."""
    text = _first(record, _TEXT_KEYS)
    if text is not None:
        return text
    lines: list[str] = []
    for key, value in record.items():
        if value is None or value == "":
            continue
        shown = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        lines.append(f"{key}: {shown}")
    return "\n".join(lines)


def _records(name: str, rows: Iterable[Any]) -> Iterator[SourceDocument]:
    seen: set[str] = set()
    for index, row in enumerate(rows):
        record: Mapping[str, Any] = (
            cast("Mapping[str, Any]", row) if isinstance(row, Mapping) else {"text": str(row)}
        )
        key = _first(record, _ID_KEYS) or str(index)
        if key in seen:
            key = f"{key}@{index}"
        seen.add(key)
        text = record_text(record)
        if text.strip():
            yield SourceDocument(
                doc_id=f"{name}#{key}", text=text, title=_first(record, _TITLE_KEYS)
            )
