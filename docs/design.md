# graphwalk — design

Status: **approved; open questions answered (see §9).** M0–M3 and M5 implemented (M4 deferred); M2 and M3 verified live on OpenRouter; M5 results in `docs/results-m5.md` (§0.1, §4 "As built in M3").

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
  // OpenRouter adds: "id", "provider", "usage.cost" (USD)   (verified live, see §0.1)
}
```

### Consequences for this spec

| Spec assumption | Reality | Impact |
|---|---|---|
| Full distribution access | Yes: `probabilities` over every offered label | Beam and sample work as specified |
| Several questions in one call | Yes: `questions` is a named map | Beam batching works, but see the next row |
| Per-question context | **No.** `state` is shared per call; only `instructions` differ per question | Per-beam context (path, current node) must go into each question's `instructions`. See §4 |
| `usage.cost` | **OpenRouter only** (verified live on OpenRouter). TypeSafe direct returns only token counts | Q1 decided: tokens only |
| Temperature or top-p on the server | None | Applied client-side to the returned distribution, as the spec already intends |
| Option limits | **Verified (docs):** Choice supports ≤ 255 options. Context is **64k tokens per request, and 32k for `state` + the longest single question** | Prefilter cap ≤ 254 (STOP uses one slot). Batches are split by that two-part budget |
| Model pin `jev-1.13.0` | **Verified:** direct `jev-1.13.0` (docs). OpenRouter `typesafe/jev-1.13` (live API), served by endpoint `typesafe/jev-1.13-20260917` | Q2 decided |
| OpenRouter uses the Decisions API | OpenRouter has `POST /api/alpha/decisions` and a TypeSafe-compatible `POST /api/v1/systemone` (**verified live**) | One code path: the SDK with `base_url=https://openrouter.ai/api` |

### 0.1 Live verification (OpenRouter, 2026-09-28)

`uv run pytest --run-live tests/live -s` passed on the first run with no code changes.
Three extra raw `curl` calls to `POST https://openrouter.ai/api/v1/systemone` confirmed
the following:

- **Path.** The SDK's `/v1/systemone` works under `base_url=https://openrouter.ai/api`.
- **Echoed model.** The response `model` is `typesafe/jev-1.13-20260917`, the dated
  endpoint id, not the requested `typesafe/jev-1.13`. `_check_model` accepts it, and
  traces record the dated id.
- **Top-level extras.** `id` (`gen-dec-…`) and `provider` (`"TypeSafe"`) are top-level
  fields. `cost` is at `usage.cost` in USD. `_openrouter_extras` already reads all three.
- **Billing.** Only input tokens are billed. `cost / input_tokens` = $0.042/M exactly
  (e.g. 588 input tokens → $0.000024696), while `output_tokens` (80–93) are reported
  but not charged.
- **Latency.** Two questions per call took 0.63–0.74 s wall-clock from this container
  (n = 3). About 550–590 input tokens per two-hop-sized call.
- **Probabilities are rounded to 2 decimals.** Easy hops come back as exactly
  `1.0 / 0.0 / 0.0`, and a genuinely ambiguous question gave `mixed 0.93, negative
  0.07, positive 0`. Consequences for M3: (a) beam scoring must floor probabilities
  (e.g. `max(p, 1e-3)`) before taking logs, because `log(0)` is `-inf`; (b) there are
  frequent exact ties at 0, so the tie-break must be deterministic (option order);
  (c) temperature sampling can't revive a 0 option, so exploration relies on beams.
  The API's resolution also limits how "calibrated" any evaluation can claim to be.
- **Near-deterministic.** Repeating the same request gave 0.93 vs 0.94 on the
  ambiguous question and identical results on the easy ones.
- **`confidence` is an extra field.** Each choice answer includes `confidence`, and it
  differs from max-p (0.89 vs 0.93). We currently drop it. It may be a better STOP or
  guardrail signal than max-p. Not yet used; proposed for M3.
- **Key order.** Probability keys come back shuffled. `normalize_distribution`
  reorders them to the caller's option order, and the smoke test asserts this.

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

### Verified from the TypeSafe docs (after network access was granted)

- **Questions in one call are evaluated in parallel against the shared `state`.** Adding
  questions barely changes latency. That supports per-depth beam batching, and it
  weakens my earlier worry that stacking beams would contaminate each decision. It
  still needs measuring.
- TypeSafe publishes a Hierarchical Classification cookbook that runs **beam search over
  Choice probabilities**. That's prior art for our beam strategy, and I'll read it
  before M3.
- The Jev 1.13 "jaggedness" page lists known weak spots:
  - literal reading of instructions
  - numbers and date comparison
  - **indirection / multi-hop reasoning**
  - large state full of irrelevant detail

  Implications for us: keep `state` = the query only, and put a precise, literal task in
  each question. Give options real descriptions, not bare opaque labels. Keep numeric and
  date comparisons in code. The indirection weakness is a direct risk to our bet: each
  hop must be asked as a *local* judgment, never as "which path answers the query".
- Rate limits: 250k tokens/s and 1,200 requests/min, and they currently change
  dynamically. The eval runner needs concurrency limits and has to respect 429s. The
  SDK already retries them.

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
(datasets or requests, the vector-RAG deps), `llm` (LiteLLM, per Q6). Dev tools
(ruff, pyright, pytest, pytest-asyncio, pytest-socket) live in a uv `dev` *dependency
group*, not an extra. That way plain `uv sync` / `uv run pytest` installs them, and
they never leak into the published package metadata. Base install: pydantic, pydantic-settings, typer,
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

**As built in M1:**

- `Node` and `Edge` also carry `attribute_provenance: dict[str, tuple[Provenance, ...]]`,
  which records which sources back each attribute value. A key that is absent falls back
  to the element's `provenance`. This is required for correct conflict attribution after
  more than one merge: node-level provenance alone would credit a value to every source
  the node ever absorbed.
- Models are frozen. `Edge.id` is derived from `(source, type, target)`.
- Merge logic is pure (`core/merge.py`) and shared by every store, so backends can't
  diverge on it.

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
    async def get_edge(self, edge_id: EdgeId) -> Edge | None: ...
    def iter_edges(self, *, batch_size: int = 1000) -> AsyncIterator[Edge]: ...
    async def delete_node(self, node_id: NodeId) -> None: ...   # also removes incident edges
    async def delete_edge(self, edge_id: EdgeId) -> None: ...
    async def clear(self) -> None: ...
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

**As built in M2:**

- `state`, `instructions` and option descriptions are typed `JSONContent` (text, an
  object, or an array), matching the wire format. A question needs at least 2 options:
  traversal must short-circuit a single-option step itself.
- `normalize_distribution` enforces the contract for every backend:
  - A label the backend was not offered is an **error**.
  - An offered label that is missing gets **0 and a warning**.
  - The result is renormalized to sum to 1. Ties break by offer order.
- `JevBackend` uses the SDK's default response type, so the SDK's validation and
  request id are kept. It reads OpenRouter's `id`, `provider` and `usage.cost` from the
  raw body. It refuses any `*-latest` model. `verify_model()` checks the pinned id:
  TypeSafe lists models as `{"models": [{"name"}]}` and OpenRouter as
  `{"data": {"id", "endpoints": [...]}}` via `GET /api/v1/models/{id}/endpoints`, verified
  against the live API. OpenRouter's general `/models` list **omits decision models**,
  so it can't be used for this check.
- `FakeDecisionBackend` answers from a script callable `(question, state) -> dist | None`.
  An unscripted question gets a Dirichlet draw seeded by `(seed, question, state)`, so
  answers don't depend on call order.

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

Splits: Jev's documented budget is 64k tokens per request and 32k for `state` + the longest
question. If the batch's estimated tokens exceed either (with a safety margin),
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

### As built in M3

Implemented in `src/graphwalk/traversal/` (`config`, `engine`, `scoring`, `sampling`,
`batching`, `prefilter`, `prompts`, `entry`, `trace`) and `src/graphwalk/embeddings/`.
TypeSafe's Hierarchical Classification cookbook, read before starting, shaped several of
these choices. Deviations from the plan above:

- **Beam selection follows the cookbook, not "finished pool + run to max depth".**
  Finished beams stay in the pool and compete with new candidates for the `k` slots.
  The walk ends when every surviving beam has finished. The plan would have spent
  `max_depth` calls on every query. Any finished beam that made the pool is kept for the
  answer list.
- **Probability floor.** Scoring uses `log(max(p, prob_floor))` with a default of `1e-3`.
  Jev rounds to 2 decimals, so a reported 0 means < 0.005 (§0.1). The cookbook uses
  `1e-9`, which treats a rounded 0 as near-impossible.
- **Forced steps.** One legal option means no call:
  - A forced *move* is not a decision, as in the cookbook.
  - A forced *STOP* (a dead end: no unvisited neighbors) counts as a decision with p = 1.
    It ranks alongside a chosen STOP.
  - Without this, walks ending at a leaf skip the final STOP decision and rank below
    walks that chose STOP. Under `exclude_visited`, leaves are often the correct answer
    (e.g. a film's release year).
- **Known quirk of `alpha = 1`.** Mean log-prob rewards appending near-certain decisions:
  a path gains score by adding a p ≈ 1 STOP. `length_alpha` is configurable, and evals
  should compare 1.0 against smaller values.
- **Direction.** The default is `both`: most KG-QA needs to follow edges backwards, e.g.
  `Nolan <-directed_by- Memento`. Options show the edge in its stored direction
  (`Memento --directed_by--> Christopher Nolan`).
- **Depth.** `max_depth` counts decision depths *including* STOP. The default is 4, i.e.
  3 hops plus STOP. A walk still open at the limit is reported with
  `terminated_by="max_depth"` and ranked after walks that ended on their own.
- **Confidence.** Every step records Jev's `confidence`, and each answer records the
  lowest confidence along its path (`min_confidence`). Per TypeSafe's docs, confidence
  is derived from the probabilities: `(n * p_max - 1) / (n - 1)` for a Choice. It carries
  no new information, but it is normalized for option count, which max-p is not. It is
  recorded only and not yet used for control. Using it is an eval question for M5,
  e.g. abstaining below a threshold.
- **Sampling.** `n_samples` independent walks run as parallel beams that don't compete.
  All walks at one depth go in one call. Each walk has its own RNG, seeded from
  `(seed, query, walk index)`. Answers are grouped and ranked by votes.
- **Prefilter.** The prefilter embeds the *query only*, not query plus path. Without an
  embedder, oversized option sets are truncated in store order to `max_options - 1`, and
  the trace records it (`reason="truncated"`).
- **Relation mode.** Each option is `{relation, direction, leads_to (first N names),
  count}`. The frontier is capped at `max_frontier` by query similarity, or by
  truncation without an embedder.
- **Guardrails.** `Budget(max_depth, max_decision_calls, max_input_tokens)` lives in
  `config.py`; there is no separate `guardrails.py`. A backend failure returns
  `status="error"` with the partial answers and trace, instead of raising.
- **Batch splitting.** Token counts are estimated at 3 characters per token of compact
  JSON, with a 20% margin. Live calls came in at ~500 tokens per question, so the
  estimate is conservative.
- **Entry.** `NameEntryResolver` finds whole-word name and alias mentions in the query,
  longest first, with an optional embedding fallback.
- **Not yet built.** The sentence-transformers `Embedder` is deferred to M5, where the
  RAG baseline needs it too; M3 has the protocol and a deterministic hashing fake.
  The `alpha = 0` early-termination bound is not implemented. There is no `_sync.py`
  wrapper yet; the CLI uses `asyncio.run`.
- **CLI.** `graphwalk query GRAPH.json "question" [--start ID] [--strategy ...]
  [--trace out.json]`.

**Live check (OpenRouter, 2026-09-28, `tests/live/test_traversal_live.py`).** The movie
fixture graph (10 nodes) was run through 3 queries × {greedy, beam k=3}: 6/6 correct.
- Greedy used 2–3 calls per query, ~1–1.2 s wall and ~500 input tokens per question.
  Cost was $0.00004–0.00007 per query.
- Beam cost 1.1–2× the tokens and calls of greedy for the same answers. It keeps
  expanding low-scoring alternatives until they finish.
- Per-call latency was 0.3–0.45 s. Depths are sequential, so wall time ≈ depth × latency.
- On these easy hops Jev is again saturated (1.0/0.0), and confidence only drops on
  genuinely ambiguous steps.

This proves plumbing, not accuracy: a 10-node graph can't show whether the bet holds.
That is M5's job.

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

### As built in M6

- **Modules:** `ingest/{sources,chunking,extraction,routing,resolution,pipeline}.py`.
  The CLI is `graphwalk ingest PATH --graph G.json`. The SQL source is not built yet:
  CSV/JSON records cover tabular data as text.
- **Extraction:** one LLM call per chunk (≤2,000 characters, packed from whole
  paragraphs) returns JSON with entities (name, type, description, literal attributes)
  and relations. A reply that fails validation is retried once with the error. Cleanup
  merges duplicate names, snake-cases types, drops self-loops, and adds untyped
  entities for relation endpoints that were not listed. The entity description becomes
  the node summary, so there is no separate summary call.
- **Routing:** `NodeIndex` proposes up to 5 candidates per mention: exact name or alias
  first, then name-word Jaccard and bge-small cosine ≥ 0.75. One Jev request per chunk
  asks one question per mention with candidates, over `{c1..cN, NEW}`. Each option
  card has the node's name, type, aliases, summary, and up to 8 relations; the state is
  the passage. If the chosen option has p < 0.6, the LLM adjudicates. If Jev fails,
  the mention merges only on an exact name. Mentions with no candidates are NEW with
  no call. `routing="exact"` (merge on identical normalized name, no calls) is kept
  as the baseline.
- **Writes:** a new node's id is a hash of (document, name, type), so re-ingesting
  recreates the same ids. A routed mention is folded in with `merge_node_data`, so
  disagreeing attributes become conflicts with provenance. A generic `entity` type
  never conflicts with a real one.
- **Ledger and retraction:** `GraphStore` gained `get_metadata`/`set_metadata`, which
  NetworkX persists in the graph file. The ledger maps doc id → hash(title, text,
  extractor version). `Provenance.source_id` is `<source>/<doc_id>`. Retraction
  removes that provenance everywhere. It deletes elements left without provenance,
  drops attributes left without sources, and promotes a lone surviving conflict value.
  A document with a failed chunk is marked `incomplete`, so the next run retries it.
- **Known limitation:** retraction does not re-derive a surviving node's name,
  summary, or aliases, even if they came from the retracted document. Routing within
  one chunk cannot merge two mentions with different names, since they are routed
  against the graph as it was before the chunk.
- **Throughput (M7 fix):** extraction runs ahead concurrently (8 calls in flight),
  and routing consumes chunks in order as they are ready, `route_concurrency` (4) at a
  time. Each window is routed against the graph as it was before the window, then
  applied in order. A mention routed NEW whose exact name was created earlier in the
  same window is routed again, so it still gets a decision. Only fuzzy-name repeats
  within one window can become duplicates; `route_concurrency=1` restores strictly
  sequential routing.

  Other changes:
  - The candidate index embeds a chunk's mentions in one batch, caches vectors by
    text (a new node reuses its mention's vector), and scores a chunk's mentions with
    one matrix product.
  - Name words on more than 200 nodes don't open the lexical pool.
  - Escalations within a chunk run concurrently.
  - Extractions persist every 25 (`JsonFileCache`).
  - The report splits time into extraction wait, routing, and apply.

  Result: CPU-only routing over 970 cached 2Wiki paragraphs is now linear (26 s / 51 s
  / 111 s for 278 / 505 / 970 paragraphs). It was superlinear before: 33 s / 104 s for
  278 / 505. Embedding the mentions is now about 80% of that CPU time.
- **Eval:** `scripts/ingest_2wiki.py` (see `docs/results-m6.md`).

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

### As built in M5

- **Datasets.**
  - **MetaQA:** the original KB (134,741 triples → 43,234 nodes, 133,582 unique edges),
    with vanilla test questions, from a pinned HF mirror (`camazlucas/MetaQA@f8385409`).
    Node ids are entity names, as in the original. Types come from relations: subjects
    are `film`, objects are `person`, `year`, `genre`, and so on.
  - **2Wiki:** the dev split from `voidful/2WikiMultihopQA@16852fde`.
- **2Wiki uses one pooled graph, not one graph per question.** A question's own
  evidence triples *are* its reasoning chain, so a per-question graph would make
  traversal trivial. The graph pools every dev question's evidence (33,091 nodes,
  26,403 edges). It is still sparse (average degree ≈ 1.6), and every walkable question
  is exactly 2 hops, so 2Wiki is a weak test here. MetaQA, with 4,000-edge hubs, is the
  real one.
- **2Wiki question types.** Only `compositional` and `inference` questions are used
  (6,785 of 12,576). `comparison` and `bridge_comparison` need a comparison computed in
  code, which graph-only traversal cannot do. That belongs to M7's graph + LLM reader
  variant.
- **2Wiki entry nodes.** `linking=gold` is traversal-only: it starts from the first
  evidence triple's subject. `linking=resolve` uses `NameEntryResolver`, and the
  summary reports linking accuracy separately. In only 86% of these questions does the
  gold start name appear verbatim in the question.
- **Baseline: same knowledge, different access.** Vector RAG retrieves from one document
  per node: its name, type, and up to 40 incident triples as sentences. It uses
  bge-small (fastembed) with top-k = 5, and the LLM reader answers with `' | '`-separated
  names. Both systems therefore see the same graph; the comparison is walk-with-Jev vs
  retrieve-and-read. Original-text RAG is M7's job.
- **Reader LLM.** `openrouter/openai/gpt-6-luna` ($0.10/M input, $0.50/M output) is the
  cheapest current general model on OpenRouter. It is configurable (`--llm-model`).
- **Cost.** Every cost comes from the provider's reported `usage.cost`, for Jev and the
  LLM alike. Nothing is estimated.
- **Latency.** Wall time per question at concurrency 4. For RAG it is retrieval plus
  the LLM call, excluding client-side rate-limit waits: OpenRouter caps new accounts at
  20 req/min per model. graphwalk latency includes the local prefilter and any CPU
  contention between concurrent questions.
- **Embedder.** fastembed (ONNX) replaced sentence-transformers in the `embeddings`
  extra, which avoids a 5 GB torch install. Batching in length order made document
  embedding 8x faster. Every node's `node_text` is embedded once per graph (cached) and
  preloaded, so the prefilter over hubs is dot products, not embedding calls.
- **Store performance.** `NetworkXStore` snapshots copy only the mutable dicts, ~7x
  cheaper than a deep copy. Deep copying dominated relation-mode expansion of hub
  frontiers.
- **Presets.** All presets use `allow_stop_at_start=False`, because neither dataset's
  answer is the topic entity. They also use `max_frontier=2000` and
  `Budget(max_depth=4, max_decision_calls=12)`. `greedy` and `beam` (k = 3) use entity
  hops. `relation` and `relation-beam` use relation hops.

- **Results.** See `docs/results-m5.md`. In short:
  - relation-mode graphwalk scores F1 0.81–0.91 across MetaQA 1–3 hops, vs
    0.89 → 0.27 → 0.11 for single-shot RAG over the same KG;
  - it is 2–3x faster at p50;
  - cost is roughly at parity (Jev is only ~2.4x cheaper per token than the cheapest
    current LLM, and a walk makes several calls);
  - the main weakness is STOP, and end to end, entity linking.
- **Multi-step RAG baseline (`iter-rag`).** Same documents, embedder and reader as the
  one-shot baseline. The reader answers or names entities to look up (exact name, else
  dense top 2), for up to 5 calls, 8 lookups per step and 40 documents in context.
- **Entity linking.** `NameEntryResolver` drops stopword-only mentions and ranks fuzzy
  candidates by IDF-weighted name-word overlap (`best_only` = top candidate).
  `ChoiceEntryResolver` spends one decision call to pick among up to 8 candidates, shown
  with their relations. Eval `--linking resolve | resolve-best | choice`; linking calls
  count toward each answer's calls, tokens and cost. Questions with no entry node count as
  linking misses.
- **Tuning round.** Chosen on MetaQA dev, applied once to test; knobs are in
  `TraversalConfig`, and `-v2` presets enable them.
  - The chosen knobs are `show_types`, `stop_style="literal"`, `relation_glosses` and
    `answer_type="hint"`.
  - The answer-type question is batched into the first call (TypeSafe's speculative
    fan-out pattern). It turns itself off on graphs with fewer than 2 node types.
  - The STOP gate (`answer_type="gate"`) is implemented but not chosen: it didn't help on
    dev.

### As built in M7

- `eval/text_qa.py` and `scripts/text_qa.py` run QA over a graph ingested from a
  question sample's pooled paragraphs, against RAG over the same paragraphs. The
  systems are graph-only graphwalk, graphwalk + LLM reader (`GraphReaderSystem`), and
  one-shot and multi-step RAG (`TEXT_PROMPTS`).
- The datasets are 2Wiki (all four question types) and HotpotQA distractor (a
  `datasets/hotpotqa.py` loader via the HF rows API, with its revision checked).
- Result: RAG over the paragraphs wins on both datasets. Multi-step RAG scores 0.75 /
  0.82 F1 against 0.54 / 0.55 for graphwalk + reader. See `docs/results-m7.md`.
- Follow-up: `GraphReaderSystem(documents=...)` reads the source paragraphs the walk
  reached (via provenance) instead of extracted facts. That gives +11/+12 F1 (0.65 /
  0.67), still 9–15 behind multi-step RAG. `ingest/normalize.py` maps free-form
  relation types onto a fixed schema with Jev; it had no measurable effect.

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
- ~~No API keys~~ `OPENROUTER_API_KEY` and `openrouter.ai` access were added on
  2026-09-28. The live smoke test now runs here (§0.1). There is still no `TYPESAFE_API_KEY`,
  so TypeSafe direct remains unexercised.

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
