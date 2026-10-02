# Configuration

## Roles and providers

graphwalk uses models in three roles:

| Role | Providers | Default |
|---|---|---|
| **Decision** (each hop, entity routing) | Jev via OpenRouter or TypeSafe | OpenRouter, `typesafe/jev-1.13` |
| **LLM** (escalation, text extraction) | OpenRouter, OpenAI, Anthropic, x.ai | OpenRouter, `openai/gpt-6-luna` |
| **Embedding** (type prefilter, dense `locate`) | fastembed (local), OpenAI, OpenRouter | fastembed, `BAAI/bge-small-en-v1.5` |

Keys and base URLs use the conventional variables (`OPENROUTER_API_KEY`,
`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `XAI_API_KEY`, `TYPESAFE_API_KEY`, each with a
`*_BASE_URL`). Everything else is `GRAPHWALK_*`; the full list is in
[`.env.example`](../.env.example).

Precedence, highest first: explicit arguments, request headers (remote MCP sessions),
the environment, defaults. Keys are held as secrets and never appear in logs, traces,
or error messages.

```python
from graphwalk import Index
from graphwalk.config import resolve_config

config = resolve_config({"llm_provider": "anthropic", "escalate_below": 0.9})
index = Index.open("kg.db", config=config)
```

## Without Jev: the LLM decider

`GRAPHWALK_DECISION_FALLBACK=llm` makes the chat model decide each hop: it scores the
options and the scores become the distribution. It passes the same traversal test
suite and is as accurate or better (+0.06 F1 at 3 hops on MetaQA), but it is 3–4×
more expensive, 7–10× slower per decision, and **its confidence is not calibrated**
(AUROC 0.50–0.69 vs Jev's 0.92–0.97), so escalation thresholds mean little with it.
Traces and reports mark it (`llm-decider:` model ids). It is off by default so nobody
gets it by accident.

## Escalation

`escalate_below` (or `GRAPHWALK_ESCALATE_BELOW`, or `--escalate-below`) re-walks a
query whose confidence is below the threshold with the LLM decider, or with
`fallback_decider=` if you pass one to `Index`. `result.escalated` says which walks
were redone, and `result.cost_usd` includes both.

## Traversal settings

`Index`, the CLI, and the MCP server walk with `TraversalConfig.kgqa()`: greedy
relation hops, typed options, an answer-type hint, at most 4 decisions. That is the
configuration measured in the experiments. Override single fields with
`TraversalConfig.kgqa(max_depth=3, ...)` or the CLI flags (`--strategy`,
`--hop-mode`, `--max-depth`, ...). A bare `TraversalConfig()` is a low-level default,
not a measured setting.

## Storage

A `.db` path is SQLite (one transactional file: nodes, edges, provenance, ingestion
ledger, and document text; write-ahead logging, so it can be read while it is
written). Any other path is NetworkX JSON, meant for tests and small graphs;
`graphwalk migrate graph.json graph.db` converts it. A 1.4M-edge graph is about 1.1 GB
of SQLite, and listing a node's edges takes under 0.1 ms.
