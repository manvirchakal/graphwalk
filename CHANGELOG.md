# Changelog

All notable changes to graphwalk are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/). Before 1.0, minor versions may break the
API; the changelog says when.

## [Unreleased]

The proof of concept (milestones M0–M7). Nothing is published yet; v0.1 is planned
in [`roadmap.md`](roadmap.md).

### Added

- The retriever core (roadmap Phase 1):
  - `graphwalk.Index` with `ingest`, `locate`, `read`, and `neighbors`; the public
    API is now what `graphwalk` exports.
  - Offset provenance: `Provenance.start`/`end`; chunks map back to document
    offsets; extraction asks for each fact's supporting sentence (`IngestConfig.evidence`,
    on by default) and falls back to the chunk when the quote is not found.
  - `locate` in three modes: graph (walk, then rank the provenance spans of walked
    edges and reached nodes), dense (embedding similarity over chunks), and hybrid
    (reciprocal rank fusion).
  - `read` through `StoredDocuments` (text in the store) or `FileDocuments` (re-read
    from disk), with stale-document detection.
  - `SQLiteStore` (write-ahead logging, indexed provenance, a documents table) and a
    `DocumentStore` protocol, both covered by contract tests; retraction uses the
    provenance index when the store has one.
  - `graphwalk locate` and `graphwalk migrate` (NetworkX JSON to SQLite); `ingest`
    writes SQLite for `.db` paths.

- Core graph model with provenance, conflict-preserving merges, and retraction;
  `GraphStore` protocol with a NetworkX store, including store metadata.
- Decision-driven traversal: greedy, beam, and sampled strategies over calibrated
  choice questions (Jev via TypeSafe or OpenRouter), with batching, guardrails,
  and traces.
- Entity entry: embedding prefilter plus decision-based linking.
- Ingestion: sources, chunking, LLM extraction, candidate routing with optional LLM
  escalation, a hash ledger for incremental re-ingestion, checkpoints, and the
  `graphwalk ingest` command.
- Relation-schema normalization (experimental; no measured gain yet).
- Evaluation harness: MetaQA, 2WikiMultiHopQA, HotpotQA, and FanOutQA loaders;
  vector-RAG, multi-step RAG, and graph+reader systems; ingestion-quality metrics.
  Findings are in `docs/results-m5.md` to `docs/results-m7.md`.

### Changed

- Eval scripts moved to `scripts/eval/`.

### Removed

- The Wikipedia page fetcher and HTML-to-text converter for FanOutQA (superseded by
  the pinned corpus mirror) and unused JSON cache helpers in `eval/ingest_eval.py`.
