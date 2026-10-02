# webqsp-agent

- params: `{"anonymized": true, "arms": ["graph"], "concurrency": 4, "dataset": "webqsp", "edges": 302871, "graph": "subgraphs of 100 sampled q", "llm": "openrouter/openai/gpt-6-luna", "max_turns": 10, "n": 100, "nodes": 103194, "sample": 100, "seed": 0, "split": "test", "start": 0}`
- git: `49b828058663d1843ed94df1d10be999eecdede8`
- run at: 2026-10-02T14:58:24.189156+00:00

| system | n | hits@1 | EM | F1 | p50 s | p95 s | $/query | $/1k q | Jev calls/q | LLM calls/q | in tok/q | status | linking |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| agent-graph | 100 | 0.470 | 0.370 | 0.448 | 26.68 | 50.73 | 0.001858 | 1.858 | 0.00 | 7.71 | 16898 | ok=100 | 1.000 |

## Systems

- **agent-graph** (wall 887s): `{"system": "agent", "llm": "openrouter/openai/gpt-6-luna", "max_turns": 10}`

## Agent arms

| arm | n | F1 | hits@1 | answered | turns (mean) | agent in-tok (mean) | agent out-tok | $/q (agent) | $/q (tools) | latency p50 / p90 s | grounded | tool calls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| agent-graph | 100 | 0.448 [0.36, 0.54] | 0.47 | 0.95 | 7.7 | 16,898 | 1,687 | 0.00186 | 0.00000 | 26.7 / 48.7 | 0.92 | answer 0.9, neighbors 11.8, relations 6.2 |

## Paired F1


