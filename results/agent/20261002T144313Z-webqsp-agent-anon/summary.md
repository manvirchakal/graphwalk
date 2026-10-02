# webqsp-agent

- params: `{"anonymized": true, "arms": ["closedbook"], "concurrency": 4, "dataset": "webqsp", "edges": 302871, "graph": "subgraphs of 100 sampled q", "llm": "openrouter/openai/gpt-6-luna", "max_turns": 10, "n": 100, "nodes": 103194, "sample": 100, "seed": 0, "split": "test", "start": 0}`
- git: `6c9aa6d9b73a211c6b3927e5fed4282be5a5f30d`
- run at: 2026-10-02T14:43:13.502869+00:00

| system | n | hits@1 | EM | F1 | p50 s | p95 s | $/query | $/1k q | Jev calls/q | LLM calls/q | in tok/q | status | linking |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| closedbook | 100 | 0.010 | 0.000 | 0.005 | 2.96 | 5.78 | 0.000121 | 0.121 | 0.00 | 1.00 | 203 | ok=100 | 1.000 |

## Systems

- **closedbook** (wall 107s): `{"system": "agent", "llm": "openrouter/openai/gpt-6-luna", "max_turns": 1}`

## Agent arms

| arm | n | F1 | hits@1 | answered | turns (mean) | agent in-tok (mean) | agent out-tok | $/q (agent) | $/q (tools) | latency p50 / p90 s | grounded | tool calls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| closedbook | 100 | 0.005 [0.00, 0.01] | 0.01 | 1.00 | 1.0 | 203 | 201 | 0.00012 | 0.00000 | 3.0 / 5.0 | 0.00 | answer 1.0 |

## Paired F1


