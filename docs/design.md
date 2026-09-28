# graphwalk — design

Status: **draft; open questions answered (see §9).** Nothing is implemented yet.

## 0. What the Jev API actually is (and how I verified it)

`docs.typesafe.ai` and `openrouter.ai` are blocked by the build environment's network
policy, so I could not read those pages directly. What I relied on, in order of trust:

1. **The official Python SDK source**, `typesafe-sdk==0.7.2` from PyPI, read in full. Its
   `_schemas/models.py` is generated from `https://api.typesafe.ai/openapi.json`, so it
   is the wire contract.
2. **Search-result summaries** of the OpenRouter model page and OpenRouter's Jev guide.
   These are secondhand. Anything that comes only from them is marked *(unverified)*
   below.

### Wire contract (from the SDK)

`POST {base_url}/v1/systemone`, `Authorization: Bearer <key>`

```jsonc
// request
{
  "model": "jev-1.13.0",
  "state": "<str | object | array>",          // shared by ALL questions in the call
  "questions": {                               // >= 1, named by the caller
    "<name>": {
      "type": "choice",
      "instructions": "<str | object | array | null>",
      "criteria": { "<label>": "<description | object | null>", ... }
    }
    // also "noul" (yes/no -> P(yes)) and "score" (ordered rubric -> distribution)
  }
}
// response
{
  "model": "jev-1.13.0",
  "answers": {
    "<name>": { "type": "choice", "choice": "<label>", "confidence": 0.9,
                "probabilities": { "<label>": 0.8, ... } }   // full distribution, sums to about 1
  },
  "usage": { "input_tokens": 120, "output_tokens": 12 }
  // OpenRouter adds: "id", "provider", "usage.cost" (USD)   (unverified)
}
```

### Consequences for this spec

| Spec assumption | Reality | Impact |
|---|---|---|
| Full distribution access | Yes: `probabilities` over every offered label | Beam and sample work as specified |
| Several questions in one call | Yes: `questions` is a named map | Beam batching works, but see the next row |
| Per-question context | **No.** `state` is shared per call; only `instructions` differ per question | Per-beam context (path, current node) must go into each question's `instructions`. See §4 |
| `usage.cost` | **OpenRouter only** (unverified). TypeSafe direct returns only token counts | **Open question Q1** |
| Temperature or top-p on the server | None | Applied client-side to the returned distribution, as the spec already intends |
| Option limits | Choice supports at most 255 options *(unverified)*; context is 32k tokens *(unverified)* | Prefilter cap must be ≤ 254 (STOP uses one slot). Batches must be split by token budget |
| Model pin `jev-1.13.0` | Direct: `jev-1.13.0` *(unverified)*. OpenRouter: `typesafe/jev-1.13` (no patch version) *(unverified)* | **Open question Q2** |
| OpenRouter uses the Decisions API | OpenRouter has `POST /api/alpha/decisions` and a TypeSafe-compatible `POST /api/v1/systemone` *(unverified)* | One code path: the SDK with `base_url=https://openrouter.ai/api` |

### Integration path: the official SDK

I plan to use `typesafe-sdk>=0.7.2,<0.8` (`AsyncTypeSafeClient`) instead of raw HTTP:

- It already handles retries (429 and 5xx with `Retry-After`), typed errors, and request IDs.
- `system_one(..., response_model=...)` accepts our own Pydantic model. We need that
  because the SDK's `Usage` type ignores unknown fields and would drop OpenRouter's
  `usage.cost`. Our response model adds `usage.cost: float | None`, plus `id` and
  `provider`.
- `base_url` is configurable, so one backend class serves both providers. Only the
  base URL, key, and model id change.
- We always pass `model=` explicitly. The SDK defaults to `jev-latest`, which the spec
  forbids.
- Cost of this choice: it adds `httpx2`, `tenacity`, and `pydantic` (already needed) to
  the base install. That's light, so the SDK is a base dependency, not an extra.

## 1. Module layout

```
src/graphwalk/
  __init__.py            # public API re-exports
  config.py              # pydantic-settings: GraphwalkSettings (env prefix GRAPHWALK_)
  cli.py                 # Typer app: ingest / query / eval
  _sync.py               # run_sync() helper + thin sync wrappers
  core/
    model.py             # Node, Edge, Provenance, AttributeConflict, NodeId, EdgeId
    hashing.py           # stable content hashing (sha256 over canonical JSON)
  stores/
    base.py              # GraphStore protocol
    networkx_store.py    # in-memory MultiDiGraph + save/load (JSON)
    neo4j_store.py       # [neo4j extra] official async driver
  decisions/
    base.py              # DecisionBackend protocol + request/response types
    fake.py              # FakeDecisionBackend (seeded, scriptable)
    jev.py               # JevBackend (typesafe-sdk), provider = typesafe | openrouter
  embeddings/
    base.py              # Embedder protocol
    fake.py              # deterministic hash embedder (tests)
    sentence_transformers.py  # [embeddings extra]
  llm/
    base.py              # LLMBackend protocol
    fake.py
    litellm_backend.py
  traversal/
    engine.py            # Traverser: orchestrates steps, guardrails, trace
    strategies.py        # Greedy / Beam / Sample (pure functions over distributions)
    scoring.py           # log-prob accumulation, length normalization
    sampling.py          # temperature + top-p over a distribution
    prefilter.py         # embedding top-N neighbor prefilter
    prompts.py           # builds state/instructions/criteria for Jev from graph data
    guardrails.py        # Budget (depth, calls, cost) + BudgetExceeded
    trace.py             # Trace, Step, BeamState (pydantic, JSON-serializable)
    entry.py             # EntryResolver protocol: query -> start node(s)
  ingest/
    sources/base.py      # Source protocol, SourceDocument, SourceProfile
    sources/files.py     # txt, md, json, csv
    sources/sql.py       # SQLAlchemy: tables, columns, FKs -> profile + rows
    chunking.py
    extraction.py        # LLM -> candidate entities/relations (pydantic schema)
    routing.py           # Jev: candidate -> existing node X | NEW (+ LLM escalation)
    resolution.py        # merge nodes, record conflicts, preserve provenance
    pipeline.py          # idempotent orchestration (hash ledger)
  eval/
    datasets/{metaqa,twowiki,hotpotqa}.py   # download + cache + load into GraphStore
    metrics.py           # EM, F1, hits@1, latency percentiles
    baselines/base.py    # QASystem adapter protocol (GraphRAG/LightRAG can plug in later)
    baselines/vector_rag.py
    runner.py            # runs a QASystem over a dataset subset, records config + versions
    report.py            # results.json + summary.md
tests/
  contract/test_graph_store.py   # parametrized over [networkx, neo4j]
  unit/...  integration/...  live/...
```

Extras: `neo4j` (neo4j driver), `embeddings` (sentence-transformers), `eval`
(datasets or requests, the vector-RAG deps), `dev` (ruff, pyright, pytest,
pytest-asyncio, testcontainers). Base install: pydantic, pydantic-settings, typer,
networkx, numpy, typesafe-sdk, litellm, sqlalchemy.

LiteLLM is heavy. Ingestion needs it, but traversal does not. **Q6:** it may be better
as an `llm` extra.

## 2. Core model

```python
class Provenance(BaseModel, frozen=True):
    source_id: str
    ingested_at: datetime
    confidence: float            # [0, 1]
    content_hash: str            # hash of the chunk or record this came from

class AttributeConflict(BaseModel, frozen=True):
    key: str
    values: list[tuple[JSONValue, Provenance]]   # every competing value, with its source

class Node(BaseModel):
    id: NodeId
    type: str
    name: str
    summary: str | None = None
    attributes: dict[str, JSONValue] = {}
    provenance: list[Provenance]                  # >= 1; grows on merge
    conflicts: dict[str, AttributeConflict] = {}
    aliases: list[str] = []                       # absorbed names from merges

class Edge(BaseModel):
    id: EdgeId                                    # deterministic: hash(src, type, dst)
    source: NodeId
    target: NodeId
    type: str
    attributes: dict[str, JSONValue] = {}
    provenance: list[Provenance]
```

## 3. Protocols

```python
class GraphStore(Protocol):
    async def upsert_node(self, node: Node) -> None: ...
    async def upsert_edge(self, edge: Edge) -> None: ...
    async def get_node(self, node_id: NodeId) -> Node | None: ...
    async def get_nodes(self, ids: Sequence[NodeId]) -> dict[NodeId, Node]: ...
    async def neighbors(self, node_id: NodeId, *, direction: Direction = "out",
                        edge_types: Collection[str] | None = None) -> list[Neighbor]: ...
                        # Neighbor = (edge, node, direction)
    async def degree(self, node_id: NodeId, *, direction: Direction = "out") -> int: ...
    async def find_nodes(self, *, name: str | None = None, type: str | None = None,
                         limit: int | None = None) -> list[Node]: ...
    async def iter_nodes(self, *, batch_size: int = 1000) -> AsyncIterator[Node]: ...
    async def delete_node(self, node_id: NodeId) -> None: ...   # also removes incident edges
    async def merge_nodes(self, keep: NodeId, absorb: NodeId) -> Node: ...
            # rewire edges, union provenance, record conflicts, add alias
    async def counts(self) -> tuple[int, int]: ...
    async def close(self) -> None: ...
# NetworkXStore adds save(path) and load(path).

class DecisionBackend(Protocol):
    @property
    def model_id(self) -> str: ...                    # recorded in traces and eval runs
    @property
    def max_options(self) -> int: ...                 # 255 for Jev
    async def decide(self, request: DecisionRequest) -> DecisionResponse: ...

class ChoiceQuestion(BaseModel):     # the only question type v1 needs
    key: str                         # e.g. "b0"
    instructions: JSONContent
    options: dict[str, JSONContent | None]            # label -> description

class DecisionRequest(BaseModel):
    state: JSONContent
    questions: list[ChoiceQuestion]

class ChoiceResult(BaseModel):
    key: str
    probabilities: dict[str, float]  # renormalized to sum to exactly 1; keys == offered labels
    top: str

class DecisionResponse(BaseModel):
    results: dict[str, ChoiceResult]
    usage: Usage                     # input_tokens, output_tokens, cost_usd: float | None
    latency_s: float
    model: str                       # as echoed by the server
    request_id: str | None

class Embedder(Protocol):
    @property
    def model_id(self) -> str: ...
    async def embed(self, texts: Sequence[str]) -> np.ndarray: ...   # (n, d), L2-normalized

class LLMBackend(Protocol):
    @property
    def model_id(self) -> str: ...
    async def complete(self, messages: Sequence[Message], *,
                       schema: type[T] | None = None) -> LLMResult[T]: ...
                       # text, parsed, usage, cost_usd

class Source(Protocol):
    @property
    def source_id(self) -> str: ...
    async def profile(self) -> SourceProfile: ...
    def documents(self) -> AsyncIterator[SourceDocument]: ...   # id, text, metadata, content_hash

class EntryResolver(Protocol):   # query -> start nodes (see Q4)
    async def resolve(self, query: str, store: GraphStore) -> list[NodeId]: ...

class QASystem(Protocol):        # eval adapter; graphwalk, vector RAG, later GraphRAG/LightRAG
    name: str
    async def answer(self, question: EvalQuestion) -> SystemAnswer: ...
            # answers, latency_s, cost_usd, decision_calls, llm_calls, trace
```

`FakeDecisionBackend(seed, script)`: `script` is a callable
`(ChoiceQuestion, state) -> dict[label, float] | None`, or a dict keyed on
`(current node, label set)`. When nothing is scripted, it returns a seeded Dirichlet
draw over the offered labels. It records every request, reports fixed fake usage and
cost, and can be told to raise, which is how guardrail and error paths get tested.

## 4. Traversal and the beam-to-Jev mapping

### One step, one question

For a path that is currently at node `c`:

- **options:** one label per outgoing (edge type, neighbor) pair, plus `STOP`. The
  option is the edge, not just the neighbor. "directed_by → X" and "written_by → X"
  are different moves, and MetaQA depends on the relation. Visited nodes are excluded
  (configurable).
- **labels:** short, stable, and unique within the question: `o1..oN`, `STOP`.
  Descriptions carry the meaning, for example
  `{"relation": "directed_by", "node": "Christopher Nolan", "type": "person", "summary": "..."}`.
  *Alternative:* human-readable labels such as `directed_by: Christopher Nolan`.
  Jev reads unlabeled choices by name, so readable labels might classify better. I'll
  make this a prompt-layout setting and let the evals decide instead of guessing.
- **instructions** (per question):
  `{"task": "...pick the next hop or STOP if the current node answers the query...", "path": [...], "current": {...}}`
- **state** (shared): `{"query": "..."}`

### Beam batching

At depth `d` there are up to `k` active beams, and each needs its own decision.
Mapping: **one Jev call per depth; one choice question per active beam**, keyed `b{i}`,
all sharing `state={"query": q}`. Each question's `instructions` carries its own path
and current node. This is the only batching Jev allows, because `state` is per call.

Splits: if the estimated tokens for the batch exceed the budget (default 24k of 32k),
the batch is split into several calls. These run concurrently (`asyncio.gather`) and
each counts toward the call and cost guardrails. Beams at the same node with the same
option set still get separate questions, because their paths differ. There is no
dedup in v1.

Honest risk: stacking k beams in one prompt may make each decision worse than k solo
calls, because the model sees unrelated context. `beam_batching: "per_depth" | "per_beam"`
will be a setting so evals can measure the tradeoff instead of assuming it.

### Scoring (exact, unit-tested)

- Each decision adds `log p(chosen)`. The STOP decision counts as a step.
- Score = `Σ log p_i / L^α`, where `L` is the number of decisions including STOP.
  `α = 1.0` by default, which is the plain mean log-prob. `α` is configurable
  (GNMT-style length penalty). With `α = 0` there is no normalization.
- Expanding a beam creates one candidate per option. STOP candidates move to the
  *finished* pool. All non-STOP candidates across beams are ranked by normalized score,
  and the top `k` survive. Search ends when no active beams remain, when the best
  active score cannot beat the k-th finished score (only valid for `α = 0`; otherwise
  run to max depth), or when a guardrail trips.
- Result: finished beams ranked by normalized score. If none finished (max depth hit),
  fall back to active beams, flagged `terminated_by="max_depth"`.
- Greedy is beam search with `k = 1`, using argmax instead of a sampled choice. It
  shares one code path.

### Sampling

`p_i ∝ p_i^(1/T)`, then top-p (the smallest set whose cumulative mass is ≥ `top_p`
after sorting descending, ties broken by label), renormalized, and drawn with
`random.Random(seed)`. `T = 0` means argmax. Each query's RNG is derived from
`(seed, query hash)`, so results don't depend on query order. `n_samples` independent
walks can run, and answers are aggregated by vote.

### High-degree prefilter

If `degree(c) > prefilter_threshold` (default 50), embed the query plus path text and
the option texts, keep the top `prefilter_top_n` (default 30, must be ≤ `max_options - 1`)
by cosine similarity, then call Jev. Embeddings are cached by `(node_id, content_hash, model)`.
Pruned options are logged in the trace.

### Guardrails

`Budget(max_depth, max_decision_calls, max_input_tokens)` is checked *before* each
call, for depth and calls, and *after* each call, for tokens, since the token count is
known only from the response. So a query can overshoot the token cap by at most one
call. Per Q1, v1 logs token usage only; pricing, and a USD cap derived from it, comes
later. An abort
raises nothing to the caller: the result has `status="aborted"`, `abort_reason`, the
best partial answer, and the full trace.

### Trace

`Trace{query, strategy, config, model_ids, steps[], result, totals}`, where each
`Step` = `{depth, beam_id, parent_beam_id, current_node, options (label→node/edge), pruned_by_prefilter, distribution (raw), distribution_used (after T/top-p), chosen, step_logp, cum_logp, norm_score, call_id}`
and `Call` = `{call_id, n_questions, latency_s, input_tokens, cost_usd, request_id}`.
Latency and cost belong to the call, not the step, because one call serves several
beams. The trace is JSON-serializable and the CLI can pretty-print it.

## 5. Ingestion (M6, summarized)

Profile the source → chunk → LLM extraction (structured output: entities + relations)
→ for each candidate, the prefilter picks the top-N existing nodes → a Jev choice over
`{match_1..match_N, NEW}` → if `confidence < route_threshold`, escalate to the LLM to
adjudicate → upsert, merge, or create (the LLM writes a summary for new nodes).
Conflicting attribute values become `AttributeConflict` records with provenance.
Nothing is silently overwritten. A hash ledger `(source_id, doc_id) → content_hash`
is stored in the graph store, keyed per source. An unchanged hash is skipped. A
changed hash retracts that document's provenance: nodes whose provenance becomes empty
are deleted, and the rest are re-derived. Only affected nodes are touched.

## 6. Evals

- **MetaQA** (1/2/3-hop, `kb.txt` loaded as triples). The topic entity comes from the
  `[bracketed]` span, so the entry point is given. Metrics: hits@1, EM, and F1 over
  answer *sets*.
- **2WikiMultiHopQA:** a graph built per question from its evidence triples, with the
  entry point from entity matching.
- **HotpotQA (M7):** graph built by the ingestion pipeline from the question's
  paragraphs.
- **Vector RAG baseline:** chunks → embedder → top-k cosine → LLM answer. It sits
  behind the `QASystem` adapter.
- **Outputs:** `results/<run_id>/results.json` (per-question records plus config, git
  SHA, model ids, package versions) and `summary.md` (EM, F1, hits@1, p50/p95 latency,
  cost per query, decision calls per query). Datasets are cached in
  `~/.cache/graphwalk/datasets`. `--n` takes a seeded subset.

## 7. Testing

- The default `uv run pytest` uses fakes only. A conftest autouse fixture blocks
  sockets, so any network call fails the run.
- Markers: `live` (needs `--run-live` plus an API key) and `neo4j` (needs `--run-neo4j`,
  testcontainers or `docker-compose.yml`).
- Exact-value unit tests for beam scoring, length normalization, temperature and
  top-p, the STOP handling, each guardrail abort, and batch splitting.

## 8. Environment limits of this build container (not design issues)

These don't change the design, but they limit what I can *verify* here:

- `docs.typesafe.ai`, `openrouter.ai`, `huggingface.co`, Dropbox, and Google Drive are
  blocked. PyPI works.
- No Docker daemon, so the Neo4j tests can't run here.
- No `TYPESAFE_API_KEY` or `OPENROUTER_API_KEY`, so the live smoke test can't run here.

Everything will be written and unit-tested with fakes. The live, Neo4j, and eval runs
need either a broader network policy plus secrets on this environment, or a run on your
machine.

## 9. Decisions

- **Q1 → token usage only.** Traces, evals, and guardrails use `input_tokens` and
  `output_tokens`. `Usage.cost_usd` stays an optional field, filled only when a provider
  reports it (OpenRouter), and is never computed. The spec's "max cost per query"
  becomes `max_input_tokens` for v1.
- **Q2 → yes.** Pin `jev-1.13.0` (direct) and `typesafe/jev-1.13` (OpenRouter). Check
  the id at startup via `GET /v1/models` and fail if it is missing. Record the
  server-echoed `model` in every trace.
- **Q3 → yes.** `hop_mode: "entity" | "relation"`, default `"entity"` as in the spec.
  In relation mode, each step's options are the distinct outgoing edge types plus
  STOP. Picking one moves the frontier to *all* targets of that type (deduplicated).
  When STOP is chosen, the frontier set is the answer. If the frontier grows beyond
  `max_frontier`, it is prefiltered by embedding similarity to the query. Beam scoring
  is unchanged; a beam is now a relation path.
- **Q6 → yes.** LiteLLM goes in an `llm` extra.
- **Q4, Q5, Q7 → proposed defaults stand** unless revisited: `EntryResolver` with name
  match then embedding; graph-only and graph + LLM reader variants in the E2E eval;
  `per_depth` beam batching with a `per_beam` switch.

## 10. Original open questions (for reference)

**Q1 — Cost on TypeSafe direct.** It returns token counts but no `usage.cost`.
Proposal: `cost_usd = input_tokens × price_per_input_token` from config (default
$0.042/M, the OpenRouter-listed price, *unverified* for direct), and mark the trace
`cost_source="estimated"` vs `"reported"`. The alternative is to disable the cost
guardrail on direct. Which do you want?

**Q2 — Model id per provider.** Pin `jev-1.13.0` on direct and `typesafe/jev-1.13` on
OpenRouter, which appears to have no patch-level id. At startup, call `GET /v1/models`
and fail if the pinned id is missing, and record the server-echoed `model` in every
trace. OK?

**Q3 — Answers are sets.** Many MetaQA answers are sets ("films directed by X" → 12
films). A walk that picks one entity per hop can only get hits@1 right, and it caps EM
and F1 low. Options: (a) the answer set = the union of STOP nodes across finished
beams; (b) add a relation-level step (choose the edge *type*, then take all targets of
that type), which also shrinks high-degree fan-out. I recommend (b) as an optional
`hop_mode="relation"` alongside the spec's `hop_mode="entity"`, since it is likely
much stronger on MetaQA. This goes beyond the spec, so I need your OK.

**Q4 — Entry point.** The spec doesn't say how the start node is chosen. MetaQA gives
it. 2Wiki and HotpotQA don't. Proposal: an `EntryResolver` protocol with the default =
exact or alias name match, then embedding top-1. It stays configurable, and its errors
are reported separately in evals so traversal accuracy isn't conflated with linking
accuracy.

**Q5 — Non-entity answers and fair comparison.** HotpotQA and some 2Wiki questions
have yes/no, comparison, or date answers that no node "is". Vector RAG uses an LLM
reader. Proposal: report two graphwalk variants, *graph-only* (answer = node
name/attribute) and *graph + LLM reader* (the reached nodes and path given to the same
LLM as the baseline). That gives both the cheap number and an apples-to-apples one.
Without this, the E2E comparison will either look unfairly bad or hide LLM cost.

**Q6 — LiteLLM in the base install?** It pulls in many dependencies. I recommend an
`llm` extra (needed by `ingest` and the RAG baseline). This adds one extra to the four
you listed.

**Q7 — Beam batching default.** `per_depth` (the spec) or `per_beam`. I'll default to
`per_depth` per the spec and expose the switch, unless you object.
