"""graphwalk: a knowledge graph as an index over text.

The public API is what this module exports. Everything else (``graphwalk.traversal``,
``graphwalk.ingest``, ...) is internal and may change between minor versions.
"""

from importlib.metadata import version

from graphwalk.core.errors import DocumentNotFoundError, GraphwalkError, StaleLocationError
from graphwalk.core.model import Edge, Neighbor, Node, Provenance
from graphwalk.index import Index
from graphwalk.ingest.pipeline import IngestConfig, IngestReport
from graphwalk.ingest.sources import FileSource, SourceDocument, TextSource
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
    "__version__",
]
