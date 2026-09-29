"""Convert a NetworkX JSON graph file to a SQLite database."""

import asyncio
import os
from dataclasses import dataclass
from pathlib import Path

from graphwalk.stores.networkx_store import NetworkXStore
from graphwalk.stores.sqlite_store import SQLiteStore


@dataclass(frozen=True)
class MigrationReport:
    nodes: int
    edges: int
    metadata: int
    documents: int


def _prepare_target(path: Path, *, overwrite: bool) -> None:
    if not path.exists():
        return
    if not overwrite:
        msg = f"{path} already exists (pass overwrite=True to replace it)"
        raise FileExistsError(msg)
    for suffix in ("", "-wal", "-shm"):
        Path(f"{path}{suffix}").unlink(missing_ok=True)


async def networkx_to_sqlite(
    source: str | os.PathLike[str], target: str | os.PathLike[str], *, overwrite: bool = False
) -> MigrationReport:
    """Copy every node, edge, metadata entry, and document from the graph file
    ``source`` into a new SQLite database at ``target``.

    Refuses to write into an existing file unless ``overwrite``; the copy is verified
    by counts before returning.
    """
    target_path = Path(target)
    await asyncio.to_thread(_prepare_target, target_path, overwrite=overwrite)
    graph = await NetworkXStore.load(source)
    store = SQLiteStore(target_path)
    try:
        async for node in graph.iter_nodes():
            await store.upsert_node(node)
        async for edge in graph.iter_edges():
            await store.upsert_edge(edge)
        metadata = graph.metadata_items()
        for key, value in metadata.items():
            await store.set_metadata(key, value)
        documents = 0
        async for document in graph.iter_documents():
            await store.put_document(document)
            documents += 1
        nodes, edges = await store.counts()
        expected = await graph.counts()
        if (nodes, edges) != expected:
            msg = f"migration lost data: {expected} -> {(nodes, edges)} (nodes, edges)"
            raise RuntimeError(msg)
        return MigrationReport(
            nodes=nodes, edges=edges, metadata=len(metadata), documents=documents
        )
    finally:
        await store.close()
