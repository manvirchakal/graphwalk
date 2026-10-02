# webqsp-agent

- params: `{"anonymized": true, "arms": ["walk"], "concurrency": 4, "dataset": "webqsp", "edges": 302871, "graph": "subgraphs of 100 sampled q", "llm": "openrouter/openai/gpt-6-luna", "max_turns": 10, "n": 100, "nodes": 103194, "sample": 100, "seed": 0, "split": "test", "start": 0}`
- git: `1e7957f1735682b0268246d9df95ef31b8fa241d`
- run at: 2026-10-02T15:13:03.409153+00:00

| system | n | hits@1 | EM | F1 | p50 s | p95 s | $/query | $/1k q | Jev calls/q | LLM calls/q | in tok/q | status | linking |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| agent-walk | 100 | 0.530 | 0.440 | 0.521 | 25.72 | 53.68 | 0.002275 | 2.275 | 6.72 | 7.52 | 15240 | ok=100 | 1.000 |

## Systems

- **agent-walk** (wall 857s): `{"system": "agent", "llm": "openrouter/openai/gpt-6-luna", "max_turns": 10}`

## Agent arms

| arm | n | F1 | hits@1 | answered | turns (mean) | agent in-tok (mean) | agent out-tok | $/q (agent) | $/q (tools) | latency p50 / p90 s | grounded | tool calls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| agent-walk | 100 | 0.521 [0.43, 0.61] | 0.53 | 0.99 | 7.5 | 15,240 | 1,379 | 0.00161 | 0.00066 | 25.7 / 46.9 | 0.94 | answer 1.0, neighbors 8.8, relations 4.8, walk 3.3 |

## Paired F1


