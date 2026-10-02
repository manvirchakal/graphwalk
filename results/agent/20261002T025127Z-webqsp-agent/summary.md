# webqsp-agent

- params: `{"arms": ["graph"], "concurrency": 4, "dataset": "webqsp", "edges": 302873, "graph": "subgraphs of 100 sampled q", "llm": "openrouter/google/gemini-3.1-pro-preview", "max_turns": 10, "n": 70, "nodes": 103253, "sample": 100, "seed": 0, "split": "test", "start": 30}`
- git: `ca64b4b58c76298b21777157409fcb2697a590c4` (dirty)
- run at: 2026-10-02T02:51:27.103285+00:00

| system | n | hits@1 | EM | F1 | p50 s | p95 s | $/query | $/1k q | Jev calls/q | LLM calls/q | in tok/q | status | linking |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| agent-graph | 70 | 0.800 | 0.629 | 0.761 | 20.30 | 45.23 | 0.035635 | 35.635 | 0.00 | 6.56 | 10589 | ok=70 | 1.000 |

## Systems

- **agent-graph** (wall 918s): `{"system": "agent", "llm": "openrouter/google/gemini-3.1-pro-preview", "max_turns": 10}`

## Agent arms

| arm | n | F1 | hits@1 | answered | turns (mean) | agent in-tok (mean) | agent out-tok | $/q (agent) | $/q (tools) | latency p50 / p90 s | grounded | tool calls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| agent-graph | 70 | 0.761 [0.67, 0.85] | 0.80 | 1.00 | 6.6 | 10,589 | 1,205 | 0.03564 | 0.00000 | 20.3 / 41.8 | 0.88 | answer 1.0, neighbors 6.1, relations 2.8 |

## Paired F1


