"""graphwalk: fast, probabilistic question answering over knowledge graphs.

When to use it and how: ``graphwalk.guide()`` (or ``graphwalk guide`` on the command
line). The public API is what this module exports. Everything else (``graphwalk.traversal``,
``graphwalk.ingest``, ...) is internal and may change between minor versions.
"""

from importlib.metadata import version

from graphwalk.core.errors import DocumentNotFoundError, GraphwalkError, StaleLocationError
from graphwalk.core.model import Edge, Neighbor, Node, Provenance
from graphwalk.guide import guide
from graphwalk.index import Index, WalkResult
from graphwalk.ingest.pipeline import IngestConfig, IngestReport
from graphwalk.ingest.sources import FileSource, SourceDocument, TextSource
from graphwalk.ingest.triples import ImportReport
from graphwalk.locate.documents import FileDocuments, StoredDocuments
from graphwalk.locate.model import Location, Passage
from graphwalk.traversal.config import TraversalConfig

__version__ = version("graphwalk")

__all__ = [
    "DocumentNotFoundError",
    "Edge",
    "FileDocuments",
    "FileSource",
    "GraphwalkError",
    "ImportReport",
    "Index",
    "IngestConfig",
    "IngestReport",
    "Location",
    "Neighbor",
    "Node",
    "Passage",
    "Provenance",
    "SourceDocument",
    "StaleLocationError",
    "StoredDocuments",
    "TextSource",
    "TraversalConfig",
    "WalkResult",
    "__version__",
    "guide",
]
