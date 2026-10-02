# graphwalk: guide for coding agents

graphwalk answers questions over a knowledge graph by walking it. Each hop is a
classification decision: the current node's relations (plus STOP) are the options, and
a fast decision model (Jev) returns a probability for each. A walk's answer comes with a
confidence, so unsure walks can be re-run ("escalated") with a slower LLM decider.

This guide says when graphwalk is the right tool, when it is not, and how to use it.
Every claim below comes from a measured experiment. The `docs/results-*.md` files in the
repository hold the tables, CIs, and caveats. Samples are small (50–300 questions per
setting, mostly one seed), so treat the numbers as directions, not guarantees.

## Decide first: is graphwalk the right tool?

| Your situation | Use | Evidence |
|---|---|---|
| An **agent** that already explores a knowledge graph with low-level tools (`neighbors`, a relation list) | **Also give it `walk`** (the MCP server does). The agent calls `walk` first, then checks or continues with `neighbors`. | A8b: a strong agent with `walk` cost 17% less (95% CI 8–27%) at equal F1 (0.76 vs 0.75, n=100, WebQSP); with a cheap agent, F1 was equal and the cost a wash. Text search over the same facts tied on F1, but only while the model knew the names: with names replaced by aliases, graph tools beat it by 0.25 F1 (P3) |
| A knowledge graph you already have (triples, RDF), **any schema size**, and you want the best accuracy | **Have an LLM write the query** from the relations around the question's entity (fetch them with one adjacency query; for a whole large schema, it fits a 1M-token context too). | A6: Freebase, 5,419 relations in one graph: 0.66 vs 0.49 F1 for walking |
| The same, but **cost or latency matters more than ~15 F1 points**, or you need a **per-answer confidence** | **graphwalk `walk`**: 6× cheaper and ~4× faster than the LLM path writer, with a confidence that ranks its answers | A6; A2 (AUROC 0.92 on WebQSP) |
| A **noisy graph extracted from text** (inconsistent relation names, edges in either direction) | graphwalk `walk` beats LLM-written paths, but every method is weak here; prefer RAG over the source text | E2b: 3–5× the F1 of LLM-written paths at 1/20–1/35 the cost, absolute F1 0.18–0.25 |
| A graph with a **small, clean schema** (≲ 50 self-describing relations) | **Have an LLM write the query** (a relation path, Cypher, SPARQL). It is more accurate, and here also cheaper than walking with escalation. | E2: MetaQA 3-hop 0.955 vs 0.887 F1; A2: WebQSP 0.76 vs 0.54 F1 |
| You want a **cheap first pass that knows when it's wrong**, and to pay for a strong model only on hard queries | **`walk` with `escalate_below`**, but expect modest savings: escalating to an LLM-written path matched its accuracy for ~19% less on Freebase (A6, offline), and on MetaQA the path writer alone is cheaper | A1, A6 |
| **Compositional questions with constraints** ("the earliest...", "which X that also Y", superlatives, comparisons) | **Not graphwalk.** It follows one relation path from the topic entity and does no filtering, ranking, or intersection. Use a dedicated KG-QA method or LLM-written queries with filters. | A2: CWQ hits@1 ≈ 0.28 for every system tried, vs ≈ 0.63–0.69 published |
| **Questions over documents** (no curated graph) | **Multi-step RAG**, not graphwalk. Building a graph from text loses dates, order, and qualifiers. | M7, E4: RAG wins by 10–19 F1 points |
| Retrieval over documents where questions **chain through named entities** ("the director of the film X...") | graphwalk `locate(mode="hybrid")` as the retriever, then read the text | E1: +18 points recall over dense on 2Wiki; ties on HotpotQA; −5 on FanOutQA |
| Broad or list-style questions over documents | Plain dense retrieval | E1 |
| As an add-on to an existing multi-step RAG loop | Don't, unless your reader model is expensive. It cuts LLM rounds by 16–21% but not error, and its own calls cost more than the rounds saved at cheap-reader prices. | E7 |

Rules of thumb:

- **The graph must hold the answer.** graphwalk's answers are nodes. If facts live in
  free text, attributes, or qualifiers the graph lacks, no walk finds them.
- **A decider that reads probabilities buys cost, speed, and confidence, not
  accuracy.** An LLM writing the query is as accurate or better. Jev is 3–4× cheaper
  and 7–10× faster per decision than an LLM deciding, and its confidence separates
  right from wrong answers (AUROC 0.92–0.97 on MetaQA 2–3 hop and WebQSP; weaker,
  0.64–0.71, on MetaQA 1-hop, 2Wiki, and CWQ). The `logprob` decider with an
  open-weights model matched that (P1) at ~10× Jev's latency: the confidence comes from
  reading probabilities, not from Jev. The `llm` decider's stated scores barely do
  (0.50–0.69).
- **Walking is fast; the model calls dominate.** graphwalk adds ~10 ms per walk on
  typical nodes and ~0.1–0.5 s at nodes with 20k neighbors (SQLite, 1.4M edges).

## Install and keys

```bash
pip install 'graphwalk[embeddings]'   # or, from a clone: uv sync --extra embeddings
```

Extras: `embeddings` (local fastembed embedder: prefilter and dense `locate`), `llm`
(LiteLLM: text ingestion, the LLM decider, escalation), `mcp` (the MCP server), `eval`.

Each hop is decided by a **decider**, and you choose it (`GRAPHWALK_DECIDER`, or
`Index.open(decider=...)` with your own `DecisionBackend`):

| Decider | What it is | Keys and settings | Measured |
|---|---|---|---|
| `jev` (default) | TypeSafe's Jev classification model | `OPENROUTER_API_KEY` (model `typesafe/jev-1.13`) or `TYPESAFE_API_KEY` with `GRAPHWALK_DECISION_PROVIDER=typesafe` | Fastest: ~0.3–0.5 s per hop |
| `logprob` | Any chat model that returns token log-probabilities, at any OpenAI-compatible endpoint (OpenRouter, vLLM, llama.cpp, OpenAI) | `GRAPHWALK_DECIDER_MODEL`, `GRAPHWALK_DECIDER_BASE_URL` (default OpenRouter, with an open-weights model), `GRAPHWALK_DECIDER_API_KEY` | With Qwen3.8-27B (the default): as accurate as Jev, confidence as informative (P1); ~10× slower through OpenRouter. Other models vary (DeepSeek V4 Flash: EM 0.44 vs 0.80), so check yours |
| `llm` | A chat model that states a score per option | the LLM settings below | Works, but its confidence is barely informative (stated scores, not probabilities) |

`GRAPHWALK_DECISION_FALLBACK=logprob` (or `llm`) uses another decider only when no Jev
key is set. LLM (escalation, ingestion): `OPENROUTER_API_KEY`, or `OPENAI_API_KEY` /
`ANTHROPIC_API_KEY` / `XAI_API_KEY` with `GRAPHWALK_LLM_PROVIDER`; set
`GRAPHWALK_ESCALATION_DECIDER=logprob` so escalated answers carry probabilities too.

Importing triples needs no key at all.

## Recipe 1: question answering over an existing knowledge graph

```bash
graphwalk import kg.nt --graph kg.db      # .nt, .csv, .tsv, .jsonl, or a|b|c .txt
graphwalk query kg.db "Where was the director of Inception born?" --escalate-below 0.9
```

(`query` walks with `TraversalConfig.kgqa()` by default; flags such as `--hop-mode` or
`--strategy` override single settings.)

```python
import asyncio
from graphwalk import Index, TraversalConfig


async def main() -> None:
    async with Index.open(
        "kg.db",
        traversal=TraversalConfig.kgqa(),  # the default; shown for clarity
        node_types=["person", "film", "place"],  # your graph's types, for the type hint
        escalate_below=0.9,
    ) as index:
        await index.import_triples("kg.nt")  # once; idempotent
        result = await index.walk("Where was the director of Inception born?")
        if result.best is None:
            print("no answer")  # entities not linked, or nothing reached
        else:
            print(result.best.names, result.confidence, result.escalated, result.cost_usd)


asyncio.run(main())
```

- `TraversalConfig.kgqa()` is the configuration measured in the experiments (greedy
  relation hops, typed options, answer-type hint, depth ≤ 4), and what `Index`, the CLI,
  and the MCP server walk with unless given another. A bare `TraversalConfig()` (beam
  search over individual neighbors) is a low-level default for `Traverser`, not a
  measured setting.
- `result.best.names` is a **set** of entities (a relation hop moves to all its
  targets). `result.best.path` is the list of hops taken; show it to users as the
  justification.
- `result.confidence` is `exp(score)`, the length-normalized path probability. Treat it
  as a ranking signal. 0.9 was the best default threshold on MetaQA; on a graph with a
  different shape, check it on 50–100 labeled questions before relying on it.
- `escalate_below` re-walks unsure queries with the LLM decider (needs the `llm` extra
  and an LLM key), or with `fallback_decider=` if you pass one. `result.cost_usd`
  includes both walks.
- Start entities are found by matching names in the question. If you already know them
  (from your own entity linker or the user's selection), call the CLI with
  `--start <node id>` (repeatable).

Triples formats: CSV/TSV with a header (`subject,relation,object`, aliases
`head/source/s`, `predicate/type/p`, `tail/target/o`, plus optional `subject_type`,
`object_type`, `subject_name`, `object_name`); JSONL with the same keys; N-Triples
(`rdfs:label` becomes the name, `rdf:type` the type, literals become `literal` nodes).
Give nodes readable names and types: decisions are made from names, relation names,
and types, not from opaque ids.

## Recipe 2: locate passages in documents (experimental)

Only when your questions chain through named entities (see the table). Otherwise use
dense retrieval or multi-step RAG.

```python
from graphwalk import Index

async with Index.open("docs.db") as index:  # inside async code; needs [llm,embeddings]
    await index.ingest("docs/")  # LLM extraction; costs money; idempotent
    for location in await index.locate("Who directed ...?", k=5, mode="hybrid"):
        passage = await index.read(location, context=200)
        print(location.path, passage.text)  # answer from the text, not the graph
```

## Recipe 3: give an agent the graph (MCP)

```bash
pip install 'graphwalk[mcp,llm]'
graphwalk mcp --db kg.db            # stdio; keys from the environment
```

Tools: `walk` (answers with confidence and path), `locate` and `read` (documents),
`neighbors` and `get_node` (explore), `status`, `ingest` and `ingest_status`. The server
also serves this guide as the resource `graphwalk://guide`.

Set `GRAPHWALK_ESCALATE_BELOW=0.9` (after checking it on your graph) to escalate unsure
walks. The server's instructions then state the threshold, and every `walk` result
reports the threshold it used, so the agent knows which answers cleared it. Escalated
answers carry the LLM decider's confidence, which is not calibrated, so the agent
should verify them with `neighbors`.

## API surface

Public names are those exported from `graphwalk` (anything else may change):
`Index`, `WalkResult`, `TraversalConfig`, `ImportReport`, `IngestConfig`,
`IngestReport`, `Location`, `Passage`, `Node`, `Edge`, `Neighbor`, `Provenance`,
`FileSource`, `TextSource`, `SourceDocument`, `FileDocuments`, `StoredDocuments`,
`GraphwalkError`, `DocumentNotFoundError`, `StaleLocationError`, `guide`; and for
deciders, `DecisionBackend`, `DecisionRequest`, `DecisionResponse`, `ChoiceQuestion`,
`ChoiceResult`, `Usage`, `DecisionBackendError`, `normalize_distribution`,
`LogprobDecider`, `LLMDecider`.

`Index` methods: `open(path, **kwargs)`, `import_triples(path)`, `walk(query)`,
`ingest(source)`, `locate(query, k, mode=)`, `read(location, context=)`,
`neighbors(node)`, `close()`. All I/O methods are async.

CLI: `graphwalk import`, `query`, `ingest`, `locate`, `migrate`, `mcp`, `guide`.

## Performance

- Per query: one Jev call per hop (~0.3–0.5 s each) plus graphwalk's overhead.
  MetaQA walks cost $0.05–0.15 per 1,000 queries; Freebase subgraph walks ≈ $0.2.
- SQLite: imports ~9k triples/s; a 1.4M-edge graph is ~1.1 GB on disk. Listing a
  node's edges takes <0.1 ms (56 ms at a 20k-neighbor hub).
- Walk overhead, excluding model calls: ~8 ms on typical nodes; 0.1 s (relation hops)
  to 0.2 s (entity hops) from a 20k-neighbor hub. With an embedder, the first visit to a
  hub embeds its neighbors' names (seconds with a local CPU model; cached after), so
  prefer relation hops (`TraversalConfig.kgqa()`) on graphs with big hubs.

## Common mistakes

- Using graphwalk to answer questions over documents. Use RAG; use `locate` only as a
  retriever.
- Expecting constraint handling ("first", "largest", "both A and B"). graphwalk doesn't
  filter, rank, or intersect answer sets; do that in your own code, or pick another tool.
- Comparing confidence across deciders, or trusting the LLM-decider fallback's: it is
  the model's stated scores, not probabilities, and was found barely informative.
- Passing a bare `TraversalConfig()` to `Index` or `Traverser` for KG-QA. Use
  `TraversalConfig.kgqa()` (the `Index` default) and override single fields.
- Assuming the escalation threshold transfers. Check it on your own graph.
- Opaque node ids as names (`Q42`, `m.0abc`). Import names (`rdfs:label` or the name
  columns) so the decider can read the options.
