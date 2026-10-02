# webqsp-agent

- params: `{"arms": ["walk"], "concurrency": 4, "dataset": "webqsp", "edges": 302873, "graph": "subgraphs of 100 sampled q", "llm": "openrouter/google/gemini-3.1-pro-preview", "max_turns": 10, "n": 70, "nodes": 103253, "sample": 100, "seed": 0, "split": "test", "start": 30}`
- git: `ca64b4b58c76298b21777157409fcb2697a590c4` (dirty)
- run at: 2026-10-02T03:07:06.063559+00:00

| system | n | hits@1 | EM | F1 | p50 s | p95 s | $/query | $/1k q | Jev calls/q | LLM calls/q | in tok/q | status | linking |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| agent-walk | 70 | 0.857 | 0.657 | 0.757 | 15.58 | 41.21 | 0.030961 | 30.961 | 2.66 | 6.53 | 9685 | ok=70 | 1.000 |

## Systems

- **agent-walk** (wall 915s): `{"system": "agent", "llm": "openrouter/google/gemini-3.1-pro-preview", "max_turns": 10}`

## Agent arms

| arm | n | F1 | hits@1 | answered | turns (mean) | agent in-tok (mean) | agent out-tok | $/q (agent) | $/q (tools) | latency p50 / p90 s | grounded | tool calls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| agent-walk | 70 | 0.757 [0.67, 0.85] | 0.86 | 1.00 | 6.5 | 9,685 | 946 | 0.03073 | 0.00024 | 15.6 / 34.7 | 0.91 | answer 1.0, neighbors 4.6, relations 2.4, walk 1.1 |

## Paired F1


