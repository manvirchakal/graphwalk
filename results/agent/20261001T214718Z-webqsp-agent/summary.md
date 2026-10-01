# webqsp-agent

- params: `{"arms": ["closedbook", "graph", "walk", "search"], "concurrency": 4, "dataset": "webqsp", "edges": 302873, "graph": "subgraphs of 100 sampled q", "llm": "openrouter/google/gemini-3.1-pro-preview", "max_turns": 10, "n": 30, "nodes": 103253, "sample": 100, "seed": 0, "split": "test"}`
- git: `bec586f3c4b6f1dec8eb3d8f86c61186b5289106`
- run at: 2026-10-01T21:47:18.583659+00:00

| system | n | hits@1 | EM | F1 | p50 s | p95 s | $/query | $/1k q | Jev calls/q | LLM calls/q | in tok/q | status | linking |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| closedbook | 30 | 0.767 | 0.333 | 0.568 | 7.22 | 26.25 | 0.010992 | 10.992 | 0.00 | 1.00 | 177 | ok=30 | 1.000 |
| agent-graph | 30 | 0.767 | 0.567 | 0.714 | 29.83 | 70.17 | 0.042258 | 42.258 | 0.00 | 6.37 | 9839 | ok=30 | 1.000 |
| agent-walk | 30 | 0.800 | 0.567 | 0.750 | 24.22 | 61.93 | 0.031541 | 31.541 | 2.57 | 6.43 | 8804 | ok=30 | 1.000 |
| agent-search | 30 | 0.833 | 0.533 | 0.717 | 40.10 | 76.65 | 0.027969 | 27.969 | 0.00 | 6.90 | 7171 | ok=30 | 1.000 |

## Systems

- **closedbook** (wall 116s): `{"system": "agent", "llm": "openrouter/google/gemini-3.1-pro-preview", "max_turns": 1}`
- **agent-graph** (wall 651s): `{"system": "agent", "llm": "openrouter/google/gemini-3.1-pro-preview", "max_turns": 10}`
- **agent-walk** (wall 644s): `{"system": "agent", "llm": "openrouter/google/gemini-3.1-pro-preview", "max_turns": 10}`
- **agent-search** (wall 721s): `{"system": "agent", "llm": "openrouter/google/gemini-3.1-pro-preview", "max_turns": 10}`

## Agent arms

| arm | n | F1 | hits@1 | answered | turns (mean) | agent in-tok (mean) | agent out-tok | $/q (agent) | $/q (tools) | latency p50 / p90 s | grounded | tool calls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| closedbook | 30 | 0.568 [0.43, 0.71] | 0.77 | 1.00 | 1.0 | 177 | 886 | 0.01099 | 0.00000 | 7.2 / 11.3 | 0.00 | answer 1.0 |
| agent-graph | 30 | 0.714 [0.57, 0.85] | 0.77 | 1.00 | 6.4 | 9,839 | 1,882 | 0.04226 | 0.00000 | 29.8 / 50.0 | 0.87 | answer 1.0, neighbors 4.8, relations 3.5 |
| agent-walk | 30 | 0.750 [0.62, 0.88] | 0.80 | 1.00 | 6.4 | 8,804 | 1,142 | 0.03131 | 0.00023 | 24.2 / 57.1 | 0.90 | answer 1.0, neighbors 4.1, relations 2.1, walk 1.2 |
| agent-search | 30 | 0.717 [0.57, 0.85] | 0.83 | 1.00 | 6.9 | 7,171 | 1,136 | 0.02797 | 0.00000 | 40.1 / 63.5 | 0.83 | answer 1.0, search 6.4 |

## Paired F1

- agent-walk minus agent-graph: +0.035 [-0.01, +0.12] F1 (30 q)
- agent-walk minus agent-search: +0.033 [-0.03, +0.11] F1 (30 q)
- agent-graph minus agent-search: -0.002 [-0.11, +0.09] F1 (30 q)
- agent-walk minus closedbook: +0.182 [+0.08, +0.30] F1 (30 q)
