# graphwalk roadmap: from proof of concept to public release and paper

## Goals

1. **A usable open-source library.** Install it with pip, point it at documents or an
   existing graph, and let any agent harness use it as a retriever: through Python,
   or through MCP (stdio or remote HTTP).
2. **An arXiv preprint.** An honest empirical study: *when does walking a graph with a
   calibrated classifier beat retrieval, and when doesn't it?* It should report the
   negative results and explain them.
3. **Reproducible evidence.** Every number in the paper can be regenerated from the
   repo with one command per table, within a stated API budget.

Decisions already made (see "Decisions" below):

- The graph is an **index over the source text**. graphwalk returns *locations*, and
  the harness reads the text.
- v0.1 ships the library **plus both MCP modes**: stdio and remote HTTP, with a
  Docker image.
- The paper is an arXiv preprint, framed as "when does it help".
- The remaining experiment budget is **about $41**: a $50 key limit, $8.89 already spent.
- The phases below are ordered by dependency. There are no calendar dates.

## Where we are (proof of concept, M0–M7)

**Built and tested** (263+ offline tests, strict pyright):

- Graph model with provenance, merging, and conflicts.
- NetworkX store.
- Jev decision backend.
- Traverser with greedy, beam, and sample strategies; entity and relation hop modes.
- Entry linking, including a Jev choice linker.
- Ingestion: extraction, Jev routing, escalation, a ledger with retraction,
  checkpoints, and a fast windowed router.
- Relation-schema normalization (experimental; showed no gain).
- Eval harness for MetaQA, 2Wiki, HotpotQA, and FanOutQA, with RAG and multi-step
  RAG baselines.

**Findings so far** (`docs/results-m5.md`, `docs/results-m6.md`, `docs/results-m7.md`):

| Setting | Result |
|---|---|
| Curated knowledge graph (MetaQA, 1/2/3-hop) | graphwalk beats multi-step RAG: 0.97/0.99/0.89 vs 0.89/0.81/0.41 F1, at 3–5× lower cost |
| 2Wiki, gold-triple graph | Tie (0.91 vs 0.92) |
| Graphs extracted from text (2Wiki, HotpotQA) | Multi-step RAG wins: 0.75/0.82 vs 0.65/0.67 F1, with graphwalk choosing which source text to read |
| FanOutQA, set-valued answers over text | Multi-step RAG wins: 0.67 vs 0.48 loose accuracy |
| Diagnosis | Walks reach the right entities and sets. Extraction drops the order, dates, and qualifiers the questions need. The graph works as a navigator, not as a replacement for the text. |

**Known gaps:**

- ~~Provenance is document- and chunk-level, with no character offsets.~~ Phase 1.
- ~~There is no `locate`/`read` API.~~ Phase 1.
- ~~Storage is one JSON file per graph.~~ SQLite, Phase 1.
- ~~Only OpenRouter and TypeSafe are supported.~~ Phase 2.
- ~~There is no MCP server and no CI.~~ Phases 0 and 3.
- Every result is one seed on small samples.
- There is no strong baseline for the curated-graph case (an LLM writing a graph query).

## Decisions

| Topic | Decision |
|---|---|
| Product shape | Layers: the Python library is the product; the MCP server is a thin wrapper; the container is the MCP server over HTTP plus a persistent volume. No separate REST API until someone asks. |
| Core API | `locate(query, k) -> [Location]` and `read(location) -> text`. A `Location` is `(doc_id, start, end, doc_hash, score, path)`: the walk path explains *why*. Plus `neighbors`, `get_node`, `ingest` (async jobs), and `status`. No answer generation inside graphwalk. |
| Storage | **SQLite is the default store**: nodes, edges, provenance with offsets, ledger, and document text, in one transactional file. NetworkX stays for tests and in-memory use. **Neo4j becomes a read-only adapter for existing knowledge graphs**, built only if the curated-graph case survives experiment E2. |
| Providers | OpenRouter, TypeSafe, OpenAI, Anthropic, and x.ai, each with an overridable base URL. |
| Roles | **Decisions**: Jev via TypeSafe or OpenRouter. If neither key is present, an **LLM-as-decider fallback** (any chat provider) runs behind an explicit flag and is documented as slower and uncalibrated. **LLM** (extraction, escalation): any chat provider. **Embeddings**: local fastembed by default; OpenAI or OpenRouter optional. |
| Keys | **stdio**: environment variables with conventional names (`OPENROUTER_API_KEY`, `TYPESAFE_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `XAI_API_KEY`, each with a `*_BASE_URL`). **Remote**: the same names as headers on the MCP initialize request, bound to that session only, held in memory, never logged or persisted. Precedence: headers, then the server's environment (only if the operator opts in), then a clear error. |
| Remote security | Server access is authenticated separately from provider keys, and the operator configures how: none (local only), a static bearer token, or **OAuth 2.1 per the MCP authorization spec** (graphwalk acts as a resource server that validates tokens from the operator's authorization server). All three ship in v0.1. **Base URLs sent in headers are ignored unless their host is on an operator allowlist** (`GRAPHWALK_ALLOWED_BASE_URLS`), which prevents server-side request forgery. Server-side keys are used only if the operator allows it. |
| Positioning | **Revised after Phase 7:** "Cheap, confidence-scored question answering over a knowledge graph you already have, as a library or an agent tool." The evidence: walks are ~6× cheaper than an LLM writing the query and their confidence is informative (AUROC up to 0.97); as an agent tool, `walk` cut a strong agent's cost by 17% at equal accuracy (A8b + P4, n=100). We do **not** claim more accuracy than an LLM, or to beat RAG on text; text ingestion is experimental. (The original framing, "a graph index over your documents", did not survive M7/E1: text is graphwalk's weakest case.) |
| Affiliation | None with TypeSafe. The paper says so, and the LLM-as-decider ablation makes the method vendor-neutral. |

## Phases

Each phase lists deliverables and an **exit criterion**. Phases 1–3 are the
engineering path to v0.1. Phase 4 (experiments) can start as soon as Phase 1 lands,
and runs in parallel with Phases 2–3.

### Phase 0: Repository foundations (done)

- GitHub Actions CI: ruff, ruff format, pyright, and offline pytest on Python 3.12 and
  3.13. It needs no network (pytest-socket already enforces this) and no secrets.
- pre-commit config, `CHANGELOG.md`, semantic versioning, issue and PR templates,
  `SECURITY.md` (how to report vulnerabilities; key handling).
- Repo hygiene:
  - Decide what stays in `results/`: keep committed summaries, keep large JSON
    ignored, and add a `results/README.md` index mapping each result to its script
    and commit.
  - Move the eval scripts under `scripts/eval/`.
  - Delete dead code paths.
- A `docs/` overhaul plan. Keep `design.md` as the internal design record; user docs
  come in Phase 5.

**Exit:** CI is green on the default branch; a fresh clone passes `uv sync && uv run
pytest`.

### Phase 1: The retriever core (`locate` / `read`) (done)

1. **Offset provenance.**
   - Chunks record `(doc_id, start, end)` in the source text.
   - Extraction additionally returns the supporting sentence span for each entity and
     relation. The prompt asks for it; it's validated against the chunk, with a
     fallback to the whole chunk.
   - `Provenance` gains `start` and `end`, and stays backward-compatible.
2. **`Location` model** plus **`locate(query, k)`**:
   - Walk from the linked entities.
   - Collect provenance spans from walked edges and visited nodes.
   - Rank them (walk order and score), de-duplicate overlapping spans, and return the
     top k with their paths.
3. **Hybrid `locate`**: optionally merge graph locations with dense-retrieval
   locations (reciprocal rank fusion). Experiment E1 decides the default.
4. **`read(location)`** through a `DocumentSource` interface:
   - `StoredDocuments`: the text is kept in the store (default).
   - `FileDocuments`: re-read from disk.
   - Both verify `doc_hash` and refuse or flag stale locations.
5. **SQLite store** implementing `GraphStore`:
   - Passes the existing store contract tests.
   - Adds a documents table and indexed provenance.
   - Uses write-ahead logging, so readers can query while ingestion writes.
   - A migration tool converts NetworkX JSON to SQLite.
6. **Library API cleanup**: a small, typed public surface (`graphwalk.Index` with
   `ingest/locate/read/neighbors`); everything else is internal.

**Exit:** `Index.open("my.db").locate("…")` returns locations whose `read()` text
contains the gold evidence on a small fixture. The store contract tests pass for both
NetworkX and SQLite.

**As built** (differences from the plan above):
- Evidence extraction is a new prompt version, so extractions cached under the old
  prompt are not reused when it is on. The eval scripts pin `evidence=False`, which
  keeps every existing cache key (extraction and graph) valid.
- Graph `locate` needs a decider for the walk; `Index` builds Jev from the environment
  if none is given. Provider configuration proper is Phase 2.
- Location scores are informational; the list order is the ranking. Graph spans rank
  per answer (walked edges in hop order, then answer nodes), then entry nodes.
- The exit test uses scripted extraction and decisions. Whether `locate` finds gold
  evidence on real data, and whether graph beats dense, is E1.

### Phase 2: Providers and configuration (done)

1. A provider registry covering the five providers × three roles, built on LiteLLM
   for chat and on provider SDKs or LiteLLM for embeddings. Base-URL overrides work
   everywhere.
2. **An LLM-as-decider backend** implementing the `DecisionBackend` protocol:
   - Asks the chat model to rank options with structured output; the distribution
     comes from its scores.
   - Enabled with `GRAPHWALK_DECISION_FALLBACK=llm`.
   - Clearly marked in results and traces.
3. **A configuration object** with sources and precedence: explicit arguments, then
   headers (in remote mode), then environment, then defaults. Secrets are
   `SecretStr` and never reach logs or traces; a test asserts this.
4. Model defaults per role. Every model id is recorded in traces and results.

**Exit:**
- An offline test matrix covers every provider, role, and configuration path using
  fakes.
- One live smoke test per provider runs manually, skipped by default.
- The fallback passes the traversal test suite.

**As built:**
- Default models: OpenRouter `openai/gpt-6-luna`, OpenAI `gpt-6-luna`, Anthropic
  `claude-haiku-4-5-20251001` (cheap, since ingestion makes many calls), x.ai none
  (set `GRAPHWALK_LLM_MODEL`; we did not want to guess a model id).
- The fallback asks for 0–100 scores for every option of every question in one call,
  and normalizes them. "Passes the traversal suite" means the full engine suite runs
  through its prompt, parsing, and normalization with scripted answers. It says
  nothing about how well a real model decides; E3 measures that.
- Header names are the environment names, matched ignoring case, with `-` for `_` and
  an optional `X-` prefix (nginx drops underscore headers by default).
- `import litellm` fetched a price map from GitHub; it no longer does.
- Live smoke tests were run for the OpenRouter roles (chat, Jev, fallback,
  embeddings); OpenAI, Anthropic, x.ai, and TypeSafe direct are written but unrun
  (no keys here).

### Phase 3: MCP server and container (done)

1. **`graphwalk mcp`** on the official MCP Python SDK.
   - Tools: `locate`, `read`, `neighbors`, `get_node`, `ingest` and `ingest_status`
     (async jobs with progress), and `status` (graph size, providers configured,
     staleness).
   - Outputs are compact: snippets plus positions, with full text via `read`.
2. **stdio mode**: keys from the environment; the graph is a local SQLite file.
3. **Remote HTTP mode (streamable HTTP)**:
   - Provider keys come from headers at initialize time, scoped to the session.
   - Server access auth is configurable (`GRAPHWALK_AUTH=none|bearer|oauth`):
     - `bearer`: a static token, for simple self-hosting;
     - `oauth`: graphwalk is an OAuth 2.1 resource server per the MCP authorization
       spec. It serves protected-resource metadata (RFC 9728) pointing at the
       operator's authorization server, validates access tokens (JWT via the issuer's
       JWKS, or introspection), and checks audience and scopes. graphwalk does not
       run its own authorization server.
     - `none` is allowed only when bound to localhost.
   - Base-URL headers are subject to the allowlist.
   - Request limits: maximum `k`, maximum `read` span, per-session concurrency.
4. **Docker image**: non-root, a volume for the SQLite file, a health endpoint,
   published to GHCR.
5. **Integration examples**:
   - a generic MCP client config (stdio and HTTP);
   - Claude Code;
   - a minimal from-scratch tool-calling harness that connects to both modes;
   - "index a folder, then ask an agent".
6. Tests:
   - in-process MCP client tests for every tool, for header handling, and for the
     rule that keys never leak into logs;
   - the SSRF allowlist;
   - each auth mode: missing, expired, wrong-audience and wrong-scope tokens are
     rejected; metadata discovery works with a stub authorization server;
   - a container smoke test in CI.

**Exit:** your own harness uses graphwalk over stdio and over remote HTTP to answer
questions about a folder of documents, with keys only ever passed in the
environment or headers.

**As built:**
- Exit check run live: `examples/harness.py` answered questions about
  `examples/data/curies` over stdio and over HTTP (bearer auth). The HTTP server had no
  provider key in its environment; the client's key arrived as a header, and neither
  the key nor the bearer token appears in the server log, even at INFO. Total cost
  under $0.01.
- Keys are read from headers on every request rather than bound once at `initialize`:
  clients send the same headers each time, so the effect is the same, and nothing is
  stored per session. Backends are cached per distinct configuration (at most 32, in
  memory).
- Remote clients see only ingest jobs whose id they hold; `status` lists jobs only in
  stdio mode.
- The container is about 850 MB, mostly onnxruntime (local embeddings) and LiteLLM. A
  slimmer image without local embeddings is possible later.
- The GHCR publish workflow runs on `v*` tags and has not run yet.
- "Staleness" in `status` is limited to counting hash-only documents; detecting edited
  source files needs a re-scan, which `ingest` already does (unchanged files are skipped).

### Phase 4: Experiments for the paper (done)

Each experiment reuses the cached graphs and extractions wherever possible. The
costs are estimates from the M5–M7 per-query costs. Order is by value to the paper.

| # | Experiment | Question it answers | Est. cost |
|---|---|---|---|
| E1 | **Evidence retrieval**: compare `locate`, dense top-k, multi-step retrieval, and hybrid on evidence recall@k (2Wiki and HotpotQA supporting facts; FanOutQA evidence pages), using the cached M7 graphs. | Is the graph a good *retriever*, alone or merged with dense retrieval? This decides the default for hybrid `locate`. | ~$2–4 |
| E2 | **An LLM writes the graph query** on MetaQA: the LLM proposes a relation path, which code executes; plus a Cypher-style variant if cheap. | Does the curated-graph win survive the strongest practical competitor? Decides whether the Neo4j adapter is worth building. | ~$1–3 |
| E3 | **Jev vs LLM-as-decider** (same traversal, only the decision backend swapped) on MetaQA and the 2Wiki gold graph. | Is the *calibrated classifier* part of the claim, or would any LLM do? Makes the paper vendor-neutral. | ~$3–6 |
| E4 | **Seeds and confidence intervals**: three seeds for the headline tables (MetaQA, 2Wiki, HotpotQA, FanOutQA), with bootstrap 95% confidence intervals. The cheap systems get all three seeds; the expensive ones fewer. | Are the headline gaps real? | ~$8–12 |
| E5 | **Extractor strength**: re-extract one small 2Wiki slice with a stronger model, then re-run QA and E1. | Is extraction the bottleneck, as the diagnosis claims? Would a better extractor flip the text results? | ~$4–8 |
| E6 | **Cost and latency accounting**: ingestion amortized over N queries, and per-query cost curves. No new calls. | When does paying for ingestion pay off? | $0 |
| — | Reserve for reruns and fixes | | ~$6–8 |

**Rules:**

- The existing test/dev discipline holds: tune on dev or on a separate seed, report
  on test.
- Every run records its git SHA, model ids, and costs from the provider's key-usage
  endpoint.
- Every run is checkpointed and resumable (container restarts are routine).
- One command per paper table: `scripts/paper/table_N.py`.
- If E2 shows the LLM-written query matching graphwalk on MetaQA, the paper says so
  plainly, and the curated-graph claim narrows to where query-writing breaks: messy
  or huge schemas and fuzzy relations. That would call for one more small
  experiment, E2b, on a large noisy schema, if budget remains.
- **What a larger budget would add** (not planned): full dev/test sets, a second
  reader-model tier, more text corpora (e.g. MuSiQue), and a non-Wikipedia domain
  corpus with schema-guided extraction.

**Exit:** every table and figure in the paper outline below has a script, a results
folder, and a confidence interval where applicable.

**Status: done** (details and verdicts in `docs/results-phase4.md`):
- E1: hybrid is the `locate` default to recommend. The graph adds 18 points of evidence
  recall on 2Wiki, ties on HotpotQA, and costs about 5 on FanOutQA.
- E2: the risk case happened. An LLM writing the relation path beats graphwalk on
  MetaQA (+0.055 F1 at 3 hops, 3 seeds) at about half the cost.
- A6 (Phase 8): on one curated Freebase graph with 5,419 relations, the LLM path writer
  still wins (+0.16 F1 [+0.07, +0.25]); schema size alone is not walking's niche.
- E2b: on large, noisy text-derived schemas, walking wins instead (+0.14 to +0.18 F1
  at 1/20 the cost). The curated-graph claim narrows to that.
- E3: an LLM decider is as accurate or better. Jev is 3–4× cheaper, 7–10× faster, and
  its confidence tells right from wrong on the harder walks.
- E4: seeds and bootstrap CIs for every table.
- E5: a 20× pricier extractor does not improve QA or retrieval.
- E6: cost and amortization.
- Budget: the first account held $10, not $41. Phase 4 spent about $5.90 across two
  keys.
- E7 (done): the graph as a first retrieval for multi-step RAG under hard retrieval.
- Next: Phase 7.

### Phase 5: Documentation and public release (v0.1)

**Status:** prepared. README, docs site (`mkdocs.yml`, `.github/workflows/docs.yml`),
changelog, metadata, and a history scan for secrets (none found) are done; PyPI trusted
publishing and the `pypi` environment are set up. Left: make the repo public, enable
GitHub Pages (source: GitHub Actions), and push the `v0.1.0` tag.

1. A README quickstart that gets someone from zero to `locate` in 5 minutes,
   covering both the library and MCP.
2. A docs site (mkdocs-material on GitHub Pages) with concepts (index-over-text,
   walks, calibration), configuration and keys, MCP setup, ingestion, and the results
   summary with its honest limits.
3. Examples folder: a folder of Markdown files, a CSV of records, and an existing
   curated knowledge graph (MetaQA subset).
4. Packaging:
   - PyPI release through trusted publishing from CI.
   - The optional dependency extras (`mcp`, `llm`, `embeddings`, `eval`; the unused `neo4j` extra is dropped until an adapter exists)
     are rationalized.
   - The GHCR image is tagged `v0.1.0`.
5. Licenses and attribution: Apache-2.0 for the code. Datasets aren't redistributed;
   the loaders fetch them. The FanOutQA mirror (CC BY-SA) is credited, and so are
   MetaQA, 2Wiki, and HotpotQA.
6. Before going public, audit git history for secrets (there should be none) and
   scrub internal notes.

**Exit:**
- `pip install graphwalk[mcp]` and `docker run ghcr.io/<you>/graphwalk` work from a
  clean machine.
- The docs build in CI.
- The repo is public and tagged v0.1.0.

### Phase 6: Paper (arXiv preprint)

Working title: *"Walking the graph or reading the text? When classifier-guided graph
traversal beats retrieval."*

1. **Outline:**
   - Introduction.
   - Method: calibrated choice decisions for traversal, entity and relation hops,
     index-over-text ingestion.
   - Setup: datasets, baselines, and budget-controlled cost accounting.
   - Results: curated graphs (MetaQA with E2 and E3), text-derived graphs (2Wiki,
     HotpotQA, FanOutQA, plus E1 and E5), cost and latency (E6).
   - Analysis: why extraction loses order and qualifiers, the failure taxonomy, the
     cases where walking wins.
   - Limitations.
   - Related work.
   - A disclosure (no affiliation with TypeSafe; which models were used).
2. **Related work to cover:**
   - GraphRAG, LightRAG, HippoRAG;
   - KG-QA traversal with LLMs (Think-on-Graph, ToG-2, KG-agent style methods);
   - multi-step retrieval (IRCoT, self-ask);
   - text-to-SPARQL and text-to-Cypher;
   - calibration and classifier-based decisions;
   - FanOutQA and the multi-hop QA benchmarks used.
3. **Figures:** the pipeline diagram; F1 vs cost frontiers per dataset; recall@k
   curves (E1); the failure taxonomy; the calibration of decision confidence.
4. **Reproducibility appendix:** exact commands, model ids, costs, seeds, and
   pointers to the dataset loaders.
5. **Process:** write in LaTeX in `paper/`, build it in CI, have external readers
   review a draft if you can find them, then post to arXiv. Linking the public repo
   requires v0.1 to be public first.

**Exit:** the preprint is on arXiv and links to the v0.1 repo and to reproduction
scripts.

### Phase 7: Experiments, round 2 (large curated graphs and escalation)

Phase 4 narrowed the claim: walking a graph with a cheap classifier pays off on
existing, large or messy knowledge graphs, especially when low-confidence walks are
escalated to an LLM. It does not pay off on graphs extracted from text. Phase 7 tests
the narrowed claim where it would matter: real KG-QA benchmarks on Freebase.

| # | Experiment | Question it answers | Est. cost |
|---|---|---|---|
| A1 | **Escalation**: re-walk with the LLM decider when Jev's confidence is below a threshold. An offline table from the E3 runs, then a library feature. | Does confidence routing buy LLM-decider accuracy at Jev's cost? | $0 |
| A5 | **Scale**: a SQLite graph with 1M+ edges and high-degree hubs; latency per neighbor lookup and per walk (scripted decider). | Is the "fast traversal engine" claim true at size? | $0 |
| A2 | **WebQSP and CWQ on Freebase**, using the published per-question subgraphs (not a full Freebase dump). Systems: Jev walk, Jev + escalation, LLM decider, LLM-written path; compared with published RoG and Think-on-Graph numbers. 50-question pilot first. | Does the narrowed claim hold on standard KG-QA benchmarks? | ~$10–20 |
| A3 | Three seeds and bootstrap CIs on A2. | Are the A2 gaps real? | ~$10–15 |
| A4 | Full MetaQA test sets instead of samples (time a pilot first; rate limits, not dollars, are the constraint). | Removes the small-sample caveat on the curated tables. | ~$10 |

**Order:** A1, A5, A2 pilot, A2, A3, A4.

**Kill criterion:** if the A2 pilot shows that Jev's confidence does not separate right
from wrong walks on Freebase (AUROC near 0.5), or that escalation saves no money at
equal accuracy, stop and reframe the paper before spending the rest.

**Budget:** about $35–60. Auto top-up is on; every script keeps its $1 balance floor.

### Phase 8: The library for existing knowledge graphs (v0.1 scope additions)

1. **Import existing graphs**: triples from CSV, JSONL, and N-Triples; a read-only
   Neo4j adapter (built after the A2 pilot, if it holds).
2. **Escalation in the API**: `Index.walk` returns answers with their confidence;
   `escalate_below=` re-walks with a fallback decider. Exposed over MCP too.
3. **Text ingestion is marked experimental.** Hybrid becomes the documented `locate`
   default for text (E1). Relation-schema normalization stays internal and
   undocumented: it showed no gain in M7, but `scripts/eval/walk_read.py` needs it to
   reproduce the M7 tables.
4. **Hardening**: hub capping and prefiltering at scale (from A5; **done**: an
   ids-only adjacency path and a quadratic-loop fix, hub walks 7–27× faster); live
   tests for the OpenAI, Anthropic, and x.ai providers, or a clear "untested" label.
5. **Agent guidance** (**done**): `graphwalk guide`, a packaged `SKILL.md`, `llms.txt`,
   and the guide as an MCP resource, with tests that keep it in step with the API.
6. Then Phase 5 (docs, PyPI, GHCR, public repo). v0.1 is released after the A2 pilot,
   so its headline use case has evidence behind it.

## Dependencies at a glance

```
Phase 0 ──► Phase 1 ──► Phase 2 ──► Phase 3 ──► Phase 5 (v0.1 public) ──► Phase 6 (arXiv)
              │                                        ▲                      ▲
              └──────────► Phase 4 (E1–E6) ────────────┴──────────────────────┘
```

Phase 7 (A2 pilot) gates the v0.1 release; Phase 8 runs alongside Phase 7.

E1 needs `locate` (Phase 1). E3 needs the LLM-as-decider (Phase 2). E2, E4, E5, and
E6 need neither and can run any time.

## Risks and how we handle them

| Risk | Mitigation |
|---|---|
| Jev availability, pricing, or model versions change | Pin model ids and record the server-echoed model. The LLM-as-decider fallback keeps the library usable. The paper reports the exact version. |
| OpenRouter's new-account limit (20 requests/min per model) slows experiments | Default to 18 requests/min, spread roles across models, keep checkpoints resumable, and budget wall-clock time, not just dollars. |
| Container restarts kill long runs | Everything is resumable: extraction cache, partial graphs, per-system checkpoints. Wait on runs in the foreground. |
| Wikipedia rate-limits bulk fetching | Use pinned mirrors (as for FanOutQA) and cache everything locally. |
| Results are noisy (Jev isn't deterministic; samples are small) | E4 seeds and confidence intervals. Claims are made only where the confidence intervals separate. |
| E2 erases the curated-graph advantage | The paper stays valuable as a "when does it help" study. The narrowed claim is tested in E2b if budget allows. |
| A $41 budget limits scope | Experiments are ordered by value, graphs are cached, and the paper states its scale. |
| Header-borne keys in remote mode | Keys are session-scoped and in memory only, logs are redacted (with tests), TLS is required in the docs, server access has its own auth (bearer or OAuth), and there's a base-URL allowlist. |

## Open questions (to revisit)

- GitHub org or user and the package name (`graphwalk` may be taken on PyPI; check
  before Phase 5).
- Whether the relation-schema normalization should ship as an experimental feature
  or be removed (it showed no gain in M7).
