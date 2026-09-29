"""Where ``read`` gets document text: the store itself, or the original files.

Both check the document's hash against the one recorded in the location, so a
document edited since ``locate`` ran is flagged (or refused) rather than silently read
at offsets that now point somewhere else.
"""

import asyncio
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from graphwalk.core.errors import DocumentNotFoundError, StaleLocationError
from graphwalk.core.hashing import content_hash
from graphwalk.core.model import StoredDocument
from graphwalk.ingest.sources import FileSource
from graphwalk.locate.model import Location, Passage
from graphwalk.stores.base import DocumentStore

type OnStale = Literal["flag", "refuse"]


@dataclass(frozen=True)
class DocumentText:
    record: StoredDocument
    text: str
    text_hash: str
    """Hash of ``text`` as read now."""


@runtime_checkable
class DocumentSource(Protocol):
    async def fetch(self, key: str) -> DocumentText:
        """The document's current text. Raises :class:`DocumentNotFoundError`."""
        ...


class StoredDocuments:
    """Text kept in the store at ingestion (``IngestConfig.store_text``, the default)."""

    def __init__(self, store: DocumentStore) -> None:
        self._store = store

    async def fetch(self, key: str) -> DocumentText:
        record = await self._store.get_document(key)
        if record is None:
            raise DocumentNotFoundError(key)
        if record.text is None:
            raise DocumentNotFoundError(
                key, "its text was not stored (store_text=False); read it with FileDocuments"
            )
        return DocumentText(record=record, text=record.text, text_hash=record.text_hash)


class FileDocuments:
    """Text re-read from the files it was ingested from. The store still supplies the
    document record; ``sources`` are the :class:`FileSource` s ingested (matched by
    ``source_id``)."""

    def __init__(self, store: DocumentStore, sources: Iterable[FileSource]) -> None:
        self._store = store
        self._sources: Mapping[str, FileSource] = {s.source_id: s for s in sources}

    async def fetch(self, key: str) -> DocumentText:
        record = await self._store.get_document(key)
        if record is None:
            raise DocumentNotFoundError(key)
        source = self._sources.get(record.source_id)
        if source is None:
            raise DocumentNotFoundError(key, f"no FileSource for {record.source_id!r}")
        doc = await asyncio.to_thread(source.document, record.doc_id)
        if doc is None:
            raise DocumentNotFoundError(key, "no longer in its source files")
        return DocumentText(record=record, text=doc.text, text_hash=content_hash(doc.text))


async def read(
    documents: DocumentSource,
    location: Location,
    *,
    context: int = 0,
    on_stale: OnStale = "flag",
) -> Passage:
    """The text at ``location``, plus up to ``context`` characters either side.

    If the document's text changed since ``locate`` (its hash differs from
    ``location.doc_hash``), the passage is marked ``stale``, or with
    ``on_stale="refuse"`` a :class:`StaleLocationError` is raised.
    """
    if context < 0:
        msg = f"context must be >= 0, got {context}"
        raise ValueError(msg)
    found = await documents.fetch(location.key)
    stale = location.doc_hash is not None and location.doc_hash != found.text_hash
    if stale and on_stale == "refuse":
        assert location.doc_hash is not None  # noqa: S101 - checked above
        raise StaleLocationError(location.key, location.doc_hash, found.text_hash)
    text = found.text
    start = 0 if location.start is None else min(location.start, len(text))
    end = len(text) if location.end is None else min(location.end, len(text))
    start, end = max(0, start - context), min(len(text), end + context)
    return Passage(
        location=location,
        text=text[start:end],
        start=start,
        end=end,
        title=found.record.title,
        stale=stale,
    )
