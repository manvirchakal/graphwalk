# graphwalk

Fast, probabilistic knowledge-graph traversal.

graphwalk answers questions over a knowledge graph by walking it: each hop is a
**constrained classification decision** (the current node's relations or neighbors,
plus `STOP`, are the options), made by a fast decision model that returns a probability
distribution over them. Walks are cheap and fast, and their confidence is informative,
so unsure walks can be escalated to a slower LLM decider only when needed.

**What it is for:** answering questions over a graph you already have (imported from
triples, or built with `ingest`), as a library or as a tool for an agent (MCP). **What
it is not:** a replacement for RAG over text. On graphs extracted from documents,
multi-step RAG answers better; there graphwalk is best used to *locate* passages, not
to answer from the graph.

> **Status: pre-alpha.** Results so far: on curated graphs (MetaQA, 2Wiki gold
> graph), walking with escalation matches an LLM decider's accuracy at about half its
> cost ([`docs/results-phase7.md`](docs/results-phase7.md)); on text-derived graphs it
> loses to multi-step RAG ([`docs/results-phase4.md`](docs/results-phase4.md)).
> Freebase KG-QA (WebQSP, CWQ) is next. Design: [`docs/design.md`](docs/design.md).
> Plan: [`roadmap.md`](roadmap.md).

```bash
graphwalk import kg.nt --graph kg.db          # an existing graph, no LLM needed
graphwalk query kg.db "Where was the director of Inception born?" --escalate-below 0.9
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

## Ingestion from documents (experimental)

Building a graph from text works, but walking a text-derived graph has not beaten
multi-step RAG at answering (extraction drops the dates, order, and qualifiers questions
need). Use it to `locate` passages, preferably in `hybrid` mode.

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
fuses both (needs `embedder=`). For text, prefer `hybrid` over `graph`: in E1 the graph
alone recalled less than dense retrieval on HotpotQA and FanOutQA. Hybrid beat dense by
18 points on 2Wiki (entity chains), tied on HotpotQA, and trailed by about 5 on
FanOutQA, so `dense` is the safer choice for broad questions
([`docs/results-phase4.md`](docs/results-phase4.md)). `read` detects documents that changed since `locate` ran and
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

Import a graph you already have from triples (CSV, TSV, JSONL, or N-Triples; no LLM
or API key needed), then walk it:

```bash
graphwalk import kg.nt --graph kg.db
graphwalk query kg.db "Where was the director of Inception born?" --escalate-below 0.9
```

```python
async with Index.open("kg.db", escalate_below=0.9) as index:
    await index.import_triples("kg.nt")
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
