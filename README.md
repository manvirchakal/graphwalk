# graphwalk

[![CI](https://github.com/manvirchakal/graphwalk/actions/workflows/ci.yml/badge.svg)](https://github.com/manvirchakal/graphwalk/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/graphwalk)](https://pypi.org/project/graphwalk/)
[![Docs](https://img.shields.io/badge/docs-github.io-blue)](https://manvirchakal.github.io/graphwalk/)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](https://github.com/manvirchakal/graphwalk/blob/main/LICENSE)

**Cheap, confidence-scored question answering over a knowledge graph you already
have.** Python library, CLI, and MCP server for agents.

graphwalk answers a question by walking the graph from the entities it names. Each hop
is a **constrained classification decision**: the current node's relations, plus
`STOP`, are the options, and a fast decision model returns a probability for each. So
every answer comes with a path and a **confidence**, and you can escalate the unsure
ones to a stronger model.

```bash
pip install graphwalk
graphwalk import kg.nt --graph kg.db          # triples: CSV, TSV, JSONL, N-Triples; no key needed
graphwalk query kg.db "Where was the director of Inception born?" --escalate-below 0.9
```

## When to use it (measured)

Every number below comes from an experiment in [`docs/`](https://github.com/manvirchakal/graphwalk/tree/main/docs/), on samples of 30–300
questions. Treat them as directions, not guarantees.

| Situation | What we found |
|---|---|
| **An agent exploring a KG** with graph tools (MCP) | Adding graphwalk's `walk` tool cut a strong agent's cost by **17%** (95% CI 8–27%) at equal accuracy (0.76 vs 0.75 F1, n=100, WebQSP). Text search over the same facts was as accurate and a little cheaper, but slower. |
| **Existing KG, cost or latency first**, or you need a per-answer confidence | `walk` is ~6× cheaper and ~4× faster than an LLM writing the query, and its confidence ranks answers well (AUROC 0.92 on WebQSP, up to 0.97 on MetaQA). |
| **Existing KG, best accuracy** | Have an LLM write the query instead: 0.66 vs 0.49 F1 on a 5,419-relation Freebase graph. |
| Questions with constraints or superlatives (CWQ) | Not graphwalk. Every system we tried scored about 0.3. |
| **Questions over documents** | Not graphwalk. Multi-step RAG wins by 10–19 F1. Text ingestion exists but is experimental. |

graphwalk does not claim to be more accurate than an LLM. What it offers is a cheap first
answer that **knows when it might be wrong**.

## Quickstart

Decisions use the Jev model, through [OpenRouter](https://openrouter.ai)
(`OPENROUTER_API_KEY`) or TypeSafe (`TYPESAFE_API_KEY`). Without either,
`GRAPHWALK_DECISION_FALLBACK=llm` lets any chat model decide (slower, not calibrated).

```python
import asyncio
from graphwalk import Index


async def main() -> None:
    async with Index.open("kg.db", escalate_below=0.9) as index:
        await index.import_triples("kg.nt")  # idempotent
        result = await index.walk("Where was the director of Inception born?")
        print(result.best.names, result.confidence, result.escalated, result.cost_usd)


asyncio.run(main())
```

A runnable example over a 15-triple movie graph is in
[`examples/kg/`](https://github.com/manvirchakal/graphwalk/blob/main/examples/kg/walk.py).

## For agents (MCP)

```bash
pip install 'graphwalk[mcp]'
claude mcp add --env OPENROUTER_API_KEY=sk-or-... graphwalk -- graphwalk mcp --db kg.db
```

Tools: `walk` (answers, paths, confidence), `neighbors`, `get_node`, `status`, plus
`locate`/`read`/`ingest` for text. The usual loop: call `walk` first; at confidence 0.9
or above, take the answer; below it, check it with `neighbors` or escalate. The server
also runs over HTTP with bearer or OAuth 2.1 auth, and ships as a container. See
[MCP setup](https://manvirchakal.github.io/graphwalk/agents/).

The package carries its own guide for coding agents: `graphwalk guide` (or
`graphwalk guide --skill` for a `SKILL.md`), and [`llms.txt`](https://github.com/manvirchakal/graphwalk/blob/main/llms.txt).

## Install options

| Extra | Enables |
|---|---|
| (none) | import, walk, query, the CLI |
| `mcp` | the MCP server (stdio and HTTP) |
| `llm` | LiteLLM: escalation, the LLM decider, text ingestion |
| `embeddings` | local fastembed embedder (prefilter, dense `locate`) |
| `eval` | dataset loaders for the benchmarks |

## Documentation

- [Docs site](https://manvirchakal.github.io/graphwalk/): quickstart, agents and MCP,
  configuration, text ingestion, evidence.
- [Results](https://github.com/manvirchakal/graphwalk/blob/main/docs/evidence.md): every experiment, with CIs, costs, and the runs behind it.
- [Design record](https://github.com/manvirchakal/graphwalk/blob/main/docs/design.md) and [roadmap](https://github.com/manvirchakal/graphwalk/blob/main/roadmap.md).

## Status

Alpha (v0.1). The public API is what `graphwalk` exports; submodules may change between
minor versions. Results are from small samples, mostly one seed. The decision model
(Jev) is a hosted model from TypeSafe; this project has no affiliation with TypeSafe,
and the LLM-decider fallback keeps it usable without Jev.

## Development

See [CONTRIBUTING.md](https://github.com/manvirchakal/graphwalk/blob/main/CONTRIBUTING.md).

## License

Apache-2.0. See [LICENSE](https://github.com/manvirchakal/graphwalk/blob/main/LICENSE).
