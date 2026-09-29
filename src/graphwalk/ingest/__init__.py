"""Ingestion: turn documents into graph nodes and edges, idempotently."""

from graphwalk.ingest.cache import JsonFileCache
from graphwalk.ingest.chunking import chunk_text
from graphwalk.ingest.extraction import (
    ExtractedEntity,
    ExtractedRelation,
    Extraction,
    ExtractionResult,
    extract,
)
from graphwalk.ingest.pipeline import (
    IngestConfig,
    IngestPipeline,
    IngestReport,
    RouteRecord,
    provenance_id,
)
from graphwalk.ingest.resolution import retract
from graphwalk.ingest.routing import NodeIndex, Route, Router
from graphwalk.ingest.sources import FileSource, Source, SourceDocument, TextSource
from graphwalk.ingest.spend import Spend

__all__ = [
    "ExtractedEntity",
    "ExtractedRelation",
    "Extraction",
    "ExtractionResult",
    "FileSource",
    "IngestConfig",
    "IngestPipeline",
    "IngestReport",
    "JsonFileCache",
    "NodeIndex",
    "Route",
    "RouteRecord",
    "Router",
    "Source",
    "SourceDocument",
    "Spend",
    "TextSource",
    "chunk_text",
    "extract",
    "provenance_id",
    "retract",
]
