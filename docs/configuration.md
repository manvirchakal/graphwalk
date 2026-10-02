# Configuration

## Roles and providers

graphwalk uses models in three roles:

| Role | Providers | Default |
|---|---|---|
| **Decision** (each hop, entity routing) | Jev via OpenRouter or TypeSafe; any model with token probabilities at an OpenAI-compatible endpoint; a chat model's stated scores; or your own ([deciders](deciders.md)) | Jev via OpenRouter, `typesafe/jev-1.13` |
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

## Choosing the decider

`GRAPHWALK_DECIDER` is `jev` (default), `logprob`, or `llm`; see [deciders](deciders.md)
for what each measured and how to bring your own.

| Variable | Meaning |
|---|---|
| `GRAPHWALK_DECIDER` | `jev`, `logprob` (token probabilities), or `llm` (stated scores) |
| `GRAPHWALK_DECIDER_MODEL` | `logprob`: model id at the endpoint (default `qwen/qwen3.8-27b` on OpenRouter) |
| `GRAPHWALK_DECIDER_BASE_URL` | `logprob`: OpenAI-compatible API root with `/v1` (default OpenRouter) |
| `GRAPHWALK_DECIDER_API_KEY` | `logprob` at your own endpoint, if it needs one |
| `GRAPHWALK_DECIDER_MAX_RPM` | `logprob`: request rate limit |
| `GRAPHWALK_DECISION_FALLBACK` | With `jev` and no Jev key: `logprob` or `llm` decides instead (default `off`) |
| `GRAPHWALK_ESCALATION_DECIDER` | What escalated walks use: `llm` (default) or `logprob` |

The `llm` decider makes the chat model score each option; the scores become the
distribution. It is as accurate as Jev or better (+0.06 F1 at 3 hops on MetaQA), but
3–4× more expensive, 7–10× slower per decision, and **its confidence is barely
informative** (AUROC 0.50–0.69): stated scores are not probabilities. Prefer `logprob`.
Traces mark every decider by its model id (`logprob:`, `llm-decider:`).

On a remote MCP server, `GRAPHWALK_DECIDER_BASE_URL` sent as a header is honored only
if it matches `GRAPHWALK_ALLOWED_BASE_URLS`, and `GRAPHWALK_DECIDER_API_KEY` is a secret
like the provider keys.

## Escalation

`escalate_below` (or `GRAPHWALK_ESCALATE_BELOW`, or `--escalate-below`) re-walks a
query whose confidence is below the threshold with the escalation decider
(`GRAPHWALK_ESCALATION_DECIDER`: the chat model's stated scores by default, or
`logprob`), or with `fallback_decider=` if you pass one to `Index`. `result.escalated` says which walks
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
