"""Behavioral contract for DocumentStore implementations."""

from collections.abc import AsyncIterator, Callable

import pytest

from factories import T0
from graphwalk.core.hashing import content_hash
from graphwalk.core.model import StoredDocument
from graphwalk.stores import DocumentStore, NetworkXStore, SQLiteStore

FACTORIES: dict[str, Callable[[], NetworkXStore | SQLiteStore]] = {
    "networkx": NetworkXStore,
    "sqlite": lambda: SQLiteStore(":memory:"),
}


@pytest.fixture(params=list(FACTORIES))
async def store(request: pytest.FixtureRequest) -> AsyncIterator[NetworkXStore | SQLiteStore]:
    s = FACTORIES[request.param]()
    try:
        yield s
    finally:
        await s.close()


def doc(
    key: str = "src/a", text: str | None = "Hello there.", title: str | None = "A"
) -> StoredDocument:
    source_id, doc_id = key.split("/", 1)
    body = text or ""
    return StoredDocument(
        key=key,
        source_id=source_id,
        doc_id=doc_id,
        title=title,
        text=text,
        text_hash=content_hash(body),
        length=len(body),
        ingested_at=T0,
    )


async def test_implements_protocol(store: DocumentStore) -> None:
    assert isinstance(store, DocumentStore)


async def test_round_trip_replace_delete(store: NetworkXStore | SQLiteStore) -> None:
    assert await store.get_document("src/a") is None
    original = doc(text="Zürich 東京\nline two")
    await store.put_document(original)
    assert await store.get_document("src/a") == original
    replaced = doc(text="new text", title=None)
    await store.put_document(replaced)
    assert await store.get_document("src/a") == replaced
    await store.delete_document("src/a")
    assert await store.get_document("src/a") is None
    await store.delete_document("src/a")  # missing: no error


async def test_hash_only_documents(store: NetworkXStore | SQLiteStore) -> None:
    await store.put_document(doc(text=None))
    got = await store.get_document("src/a")
    assert got is not None
    assert got.text is None


async def test_iter_documents_is_ordered_and_clear_removes(
    store: NetworkXStore | SQLiteStore,
) -> None:
    for key in ("src/c", "src/a", "other/b"):
        await store.put_document(doc(key))
    assert [d.key async for d in store.iter_documents()] == ["other/b", "src/a", "src/c"]
    await store.clear()
    assert [d async for d in store.iter_documents()] == []
