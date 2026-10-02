# Changelog

All notable changes to graphwalk are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/). Before 1.0, minor versions may break the
API; the changelog says when.

## [Unreleased]

### Fixed

- With an embedder, options over the decider's limit but under `prefilter_threshold`
  were cut in arbitrary order; they are now ranked by similarity to the question.

### Changed

- Evidence: an open-weights model's token probabilities make the walk's confidence as
  informative as Jev's (paper P1), so the docs no longer credit Jev alone; and over 100
  questions the strong agent's saving from `walk` is 17% (95% CI 8–27%), not 25% (P4).

### Added

- Paper experiments P0, P1, P4, P5 (`docs/results-paper.md`), the token-probability
  decider `graphwalk.eval.logprob_decider`, and `scripts/paper/figure_selective.py`
  and `table_agent.py`.

## [0.1.0] - 2026-10-02

The first public release: confidence-scored question answering over existing knowledge
graphs, as a library, a CLI, and an MCP server.

Highlights:

- `graphwalk import` / `Index.import_triples`: load a graph from CSV, TSV, JSONL, or
  N-Triples into SQLite, with no API key.
- `graphwalk query` / `Index.walk`: answers with their path and confidence, walked with
  the measured `TraversalConfig.kgqa()` setting; `escalate_below` re-walks unsure
  queries with an LLM decider.
- `graphwalk mcp`: the same as MCP tools (`walk`, `neighbors`, `get_node`, ...), over
  stdio or authenticated HTTP, and as a container.
- `graphwalk guide`: a packaged guide for coding agents, with the measured regime map.
- Text ingestion with `locate`/`read` (experimental: RAG answers documents better).
- A documentation site, and the evidence behind every claim in `docs/` (experiments
  E1–E7 and A1–A8, with the scripts and runs that regenerate them).

Details, by area:

### Changed

- Walks default to `TraversalConfig.kgqa()` (greedy relation hops), the setting
  measured on curated graphs: `Index.walk`, `graphwalk query` (its flags now override
  single settings of it), and the MCP `walk` tool. Previously they used a bare
  `TraversalConfig()` (beam search over individual neighbors), which no experiment
  found best. Graph `locate` keeps the bare default.

### Added

- Documentation site (MkDocs Material, built in CI, deployed to GitHub Pages).
- Experiments A7 (predicting walk success before walking: it barely works; walk
  first) and A8 (graphwalk as an agent's tool: a strong agent with `walk` cost 25%
  less at equal F1), with the agent harness in `graphwalk.eval.agent`.
- Faster walks at high-degree nodes: stores may implement `AdjacencyStore` (SQLite
  does), so traversal lists options from edge ids and loads only the nodes it keeps;
  relation hops no longer deduplicate targets in quadratic time. Walks from a
  20k-neighbor hub: 3.1 s → 0.11 s (relation hops), 1.3 s → 0.19 s (entity hops).

- A guide for coding agents, shipped in the package: `graphwalk guide` (and
  `graphwalk.guide()`) prints when to use graphwalk (the measured regime map), recipes,
  the public API, and common mistakes; `graphwalk guide --skill` prints a `SKILL.md`.
  `llms.txt` at the repository root indexes the docs and evidence. The MCP server
  serves the guide as the resource `graphwalk://guide`.
- MCP: the instructions state the escalation threshold (or that escalation is off),
  and `walk` results report the `threshold` they used; `Index.escalate_below` exposes
  the effective threshold.
- `TraversalConfig.kgqa()`: the configuration measured on curated graphs (the eval
  suite's `relation-v2` preset), and `Index(node_types=...)` so its answer-type hint
  works through `Index`.
- MCP `walk` tool: the graph's own answers with paths and confidence, escalated per
  `GRAPHWALK_ESCALATE_BELOW` (a new setting; also the `Index` default when it builds
  its own backends or is given a config).
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

[Unreleased]: https://github.com/manvirchakal/graphwalk/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/manvirchakal/graphwalk/releases/tag/v0.1.0
