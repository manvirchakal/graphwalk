# webqsp-agent

- params: `{"anonymized": true, "arms": ["search"], "concurrency": 4, "dataset": "webqsp", "edges": 302871, "graph": "subgraphs of 100 sampled q", "llm": "openrouter/openai/gpt-6-luna", "max_turns": 10, "n": 100, "nodes": 103194, "sample": 100, "seed": 0, "split": "test", "start": 0}`
- git: `a79a66d23dc10166bc19ea3b0ea4e77c61ddc37a`
- run at: 2026-10-02T15:46:50.728976+00:00

| system | n | hits@1 | EM | F1 | p50 s | p95 s | $/query | $/1k q | Jev calls/q | LLM calls/q | in tok/q | status | linking |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| agent-search | 100 | 0.270 | 0.130 | 0.199 | 28.27 | 48.31 | 0.001788 | 1.788 | 0.00 | 9.55 | 17819 | ok=100 | 1.000 |

## Systems

- **agent-search** (wall 1024s): `{"system": "agent", "llm": "openrouter/openai/gpt-6-luna", "max_turns": 10}`

## Agent arms

| arm | n | F1 | hits@1 | answered | turns (mean) | agent in-tok (mean) | agent out-tok | $/q (agent) | $/q (tools) | latency p50 / p90 s | grounded | tool calls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| agent-search | 100 | 0.199 [0.13, 0.27] | 0.27 | 0.98 | 9.6 | 17,819 | 1,220 | 0.00179 | 0.00000 | 28.3 / 43.0 | 0.63 | answer 1.0, search 13.4 |

## Paired F1


