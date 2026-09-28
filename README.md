# graphwalk

Fast, probabilistic knowledge-graph traversal and ingestion.

graphwalk treats each hop through a knowledge graph as a **constrained classification
decision**: the current node's neighbors (plus `STOP`) are the options, and a fast
decision model returns a calibrated probability distribution over them. The bet is that
many cheap, calibrated decisions beat LLM-driven graph RAG on cost and latency at
competitive accuracy. The built-in eval harness exists to test exactly that bet.

> **Status: pre-alpha.** Graph store, Jev decision backend, and the traversal engine
> (greedy / beam / sample, entity and relation hops) work. Ingestion, Neo4j, and evals are
> still being built. See [`docs/design.md`](docs/design.md).

```bash
uv run graphwalk query graph.json "Where was the director of Inception born?" \
    --strategy beam --beam-width 3 --trace trace.json
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
| `neo4j`      | `Neo4jStore` (official async driver)              |
| `embeddings` | local sentence-transformers embedder for prefilter |
| `llm`        | LiteLLM backend for ingestion and the RAG baseline |
| `eval`       | dataset download for the eval harness             |

```bash
uv sync --extra neo4j --extra embeddings
```

## Decision backend

The v1 decision backend is TypeSafe's Jev (pinned: `jev-1.13.0` direct,
`typesafe/jev-1.13` on OpenRouter). Select the provider with
`GRAPHWALK_DECISION_PROVIDER=openrouter|typesafe`. Backends sit behind a
`DecisionBackend` protocol, so other classifiers can be added without touching traversal.

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache-2.0. See [LICENSE](LICENSE).
