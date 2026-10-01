# Changelog

All notable changes to graphwalk are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/). Before 1.0, minor versions may break the
API; the changelog says when.

## [Unreleased]

The proof of concept (milestones M0–M7). Nothing is published yet; v0.1 is planned
in [`roadmap.md`](roadmap.md).

### Added

- Import existing knowledge graphs (roadmap Phase 8): `graphwalk import` and
  `Index.import_triples` read CSV, TSV, JSONL, `|`-separated text, and N-Triples
  (labels and `rdf:type` become names and types; literals become nodes).
  `SQLiteStore.upsert_many` writes a batch in one transaction. `graphwalk query`
  accepts SQLite graphs and `--escalate-below`.
- Escalation (roadmap Phase 7, A1): `Index(escalate_below=..., fallback_decider=...)`
  re-walks queries whose best answer's confidence is below the threshold with a
  fallback decider (default: the LLM-as-decider). `Index.walk(query)` returns the
  graph's own answers with their confidence (`WalkResult`). `EscalatingTraverser`,
  `Answer.confidence`, `TraversalResult.confidence`/`.cost_usd`/`.escalation`.
  Paper table: `scripts/paper/table_escalation.py`.
- MCP server and container (roadmap Phase 3):
  - `graphwalk mcp` (stdio) and `graphwalk mcp --http` (streamable HTTP) on the
    official MCP SDK, with tools `locate`, `read`, `neighbors`, `get_node`, `ingest`,
    `ingest_status`, and `status` (the `mcp` extra).
  - Remote mode: provider keys from each client's headers (one backend set per
    distinct configuration, in memory only); server keys only with
    `GRAPHWALK_SERVER_KEYS`; header base URLs only from an allowlist; DNS-rebinding
    protection; request limits; remote `ingest` of server paths only under
    `GRAPHWALK_INGEST_ROOT`.
  - Auth: `none` (localhost only), `bearer`, or `oauth` as an OAuth 2.1 resource
    server (RFC 9728 metadata; JWT via JWKS with RFC 8414/OIDC discovery, or RFC 7662
    introspection; audience, expiry, and scope checks).
  - `Dockerfile` (non-root, `/data` volume, health check), a CI container smoke test,
    and a tag-triggered GHCR publish workflow.
  - `examples/`: client configs, Claude Code commands, a from-scratch agent harness,
    and a sample folder.
  - Ingestion progress callbacks; `Index(owns_store=False)`.
- Providers and configuration (roadmap Phase 2):
  - Five providers (OpenRouter, TypeSafe, OpenAI, Anthropic, x.ai) across three roles
    (decision, LLM, embedding), with overridable base URLs everywhere.
  - `resolve_config`: arguments > headers > environment > defaults, with the source of
    every value recorded; header base URLs only from an allowlist; environment keys
    can be switched off for remote servers.
  - `graphwalk.providers`: `make_decider`, `make_llm`, `make_embedder`. Keys and base
    URLs are always passed explicitly, so no client library falls back to the
    process environment.
  - LLM-as-decider fallback (`GRAPHWALK_DECISION_FALLBACK=llm`), marked in traces and
    reports; the traversal test suite runs against it too.
  - OpenAI-compatible remote embeddings (OpenAI, OpenRouter).
  - Secrets are redacted from provider error messages; a test checks errors, logs,
    reprs, and summaries.
  - Ingest reports record the extraction and decision models.
  - Live smoke tests per provider and role (`--run-live`).
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

- `import litellm` no longer downloads LiteLLM's price map (graphwalk sets
  `LITELLM_LOCAL_MODEL_COST_MAP`); cost is only ever what the provider reports.
- `graphwalk ingest`: `--llm-provider`, and `--llm-model`/`--embed-model` now take
  the provider's own model id (defaults come from the configuration).

- Eval scripts moved to `scripts/eval/`.

### Removed

- The Wikipedia page fetcher and HTML-to-text converter for FanOutQA (superseded by
  the pinned corpus mirror) and unused JSON cache helpers in `eval/ingest_eval.py`.
