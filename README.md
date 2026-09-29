# graphwalk

Fast, probabilistic knowledge-graph traversal and ingestion.

graphwalk treats each hop through a knowledge graph as a **constrained classification
decision**: the current node's neighbors (plus `STOP`) are the options, and a fast
decision model returns a calibrated probability distribution over them. The bet is that
many cheap, calibrated decisions beat LLM-driven graph RAG on cost and latency at
competitive accuracy. The built-in eval harness exists to test exactly that bet.

> **Status: pre-alpha.** Ingestion, graph stores (NetworkX, SQLite), traversal, the
> `locate`/`read` retriever, and the eval harness work. Results so far:
> [`docs/results-m7.md`](docs/results-m7.md) (graph walking has not beaten multi-step
> RAG yet). Design: [`docs/design.md`](docs/design.md). Plan: [`roadmap.md`](roadmap.md).

```bash
graphwalk ingest docs/ --graph my.db
graphwalk locate my.db "Where was the director of Inception born?"
```

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12.

```bash
git clone <repo-url> graphwalk && cd graphwalk
uv sync                       # base install + dev tools
cp .env.example .env          # add OPENROUTER_API_KEY or TYPESAFE_API_KEY
uv run graphwalk --help
```

Optional extras:

| Extra        | Enables                                           |
|--------------|---------------------------------------------------|
| `embeddings` | local fastembed embedder (prefilter, dense locate) |
| `llm`        | LiteLLM backend for ingestion and the RAG baseline |
| `eval`       | dataset download for the eval harness             |

```bash
uv sync --extra llm --extra embeddings
```

## Ingestion

```bash
uv sync --extra llm --extra embeddings
graphwalk ingest docs/ --graph my.db --report ingest-report.json
graphwalk locate my.db "Who directed ...?" -k 5 --context 200
```

A `.db` graph is SQLite (write-ahead logging, so it can be read while ingestion
writes); any other path is NetworkX JSON. `graphwalk migrate graph.json graph.db`
converts one to the other.

`ingest` reads `.txt`/`.md` files (one document each) and `.json`/`.jsonl`/`.csv`
records (one document per record). An LLM extracts entities and relations, and Jev
decides for each entity whether it is an existing node or a new one. Low-confidence
decisions go to the LLM. Re-running is idempotent: unchanged documents are skipped, and
changed ones are retracted and re-ingested (`--prune` also retracts deleted ones).

## Library: `locate` and `read`

The graph is an index over your text. `locate` walks it from the entities a question
names and returns the source spans behind the walk (each extracted fact records the
sentence it came from), with the path that reached them; `read` returns the text.

```python
from graphwalk import Index

async with Index.open("my.db", llm=llm, decider=decider) as index:
    await index.ingest("docs/")
    for location in await index.locate("Where was Marie Curie's husband born?", k=5):
        passage = await index.read(location, context=100)
        print(location.key, location.path, passage.text)
```

`mode="dense"` ranks text chunks by embedding similarity instead, and `mode="hybrid"`
fuses both (needs `embedder=`). Which default is best is still an open experiment
(E1 in the roadmap). `read` detects documents that changed since `locate` ran and
flags the passage as `stale` (or refuses, with `on_stale="refuse"`). By default the
text is kept in the store; `IngestConfig(store_text=False)` with `FileDocuments`
re-reads it from disk instead.

Only names exported from `graphwalk` are public API; submodules may change.

## Decision backend

The v1 decision backend is TypeSafe's Jev (pinned: `jev-1.13.0` direct,
`typesafe/jev-1.13` on OpenRouter). Select the provider with
`GRAPHWALK_DECISION_PROVIDER=openrouter|typesafe`. Backends sit behind a
`DecisionBackend` protocol, so other classifiers can be added without touching traversal.

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache-2.0. See [LICENSE](LICENSE).
