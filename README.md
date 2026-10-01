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

## Existing knowledge graphs: `walk` and escalation

On a curated graph the graph holds the facts, so the walk's answer is the answer.
`walk` returns it with its confidence; `escalate_below` re-walks unsure queries with a
slower, stronger decider (the LLM-as-decider by default). On MetaQA, escalating below
0.9 matched the LLM decider's accuracy at about half its cost
([`docs/results-phase7.md`](docs/results-phase7.md)).

```python
async with Index.open("kg.db", escalate_below=0.9) as index:
    result = await index.walk("Where was the director of Inception born?")
    print(result.best.names, result.confidence, result.escalated, result.cost_usd)
```

## MCP server

```bash
pip install 'graphwalk[mcp,llm]'
graphwalk mcp --db my.db                      # stdio: keys from the environment
graphwalk mcp --http --db my.db               # streamable HTTP: keys from client headers
```

Tools: `locate`, `read`, `neighbors`, `get_node`, `ingest`/`ingest_status`, `status`.
The HTTP server supports bearer-token or OAuth 2.1 (resource server) auth, and the same
server ships as a container (`Dockerfile`). Client configs, Claude Code commands,
server settings, and a from-scratch agent harness are in [`examples/`](examples/).

## Providers and configuration

graphwalk uses models in three roles:

| Role | Providers | Default |
|---|---|---|
| **Decision** (each hop, entity routing) | Jev via OpenRouter or TypeSafe | OpenRouter, `typesafe/jev-1.13` |
| **LLM** (extraction, escalation) | OpenRouter, OpenAI, Anthropic, x.ai | OpenRouter, `openai/gpt-6-luna` |
| **Embedding** (prefilter, dense `locate`) | fastembed (local), OpenAI, OpenRouter | fastembed, `BAAI/bge-small-en-v1.5` |

Keys and base URLs use the conventional variables (`OPENAI_API_KEY`,
`ANTHROPIC_BASE_URL`, ...); everything else is `GRAPHWALK_*`. See
[`.env.example`](.env.example) for the full list. Precedence, highest first: explicit
arguments, request headers (remote MCP sessions), the environment, defaults. Keys are
kept as secrets and never appear in logs, traces, or error messages.

With no Jev key, `GRAPHWALK_DECISION_FALLBACK=llm` makes the chat model decide
instead: it scores the options and the scores become the distribution. It works
everywhere Jev does (it passes the same traversal test suite), but it is slower and
its probabilities are **not calibrated**. Traces and reports mark it (`llm-decider:`
model ids). It is off by default so nobody gets it by accident.

```python
from graphwalk.config import resolve_config
from graphwalk import Index

config = resolve_config({"llm_provider": "anthropic", "decision_fallback": "llm"})
index = Index.open("my.db", config=config)
```

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache-2.0. See [LICENSE](LICENSE).
