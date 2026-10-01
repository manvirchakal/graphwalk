# webqsp-agent

- params: `{"arms": ["jev", "graph", "walk"], "concurrency": 4, "dataset": "webqsp", "edges": 302873, "graph": "subgraphs of 100 sampled q", "llm": "openrouter/openai/gpt-6-luna", "max_turns": 10, "n": 2, "nodes": 103253, "sample": 100, "seed": 0, "split": "test"}`
- git: `21085ef5e7c7072b51aa273da8aac54a6e0958f1`
- run at: 2026-10-01T17:52:33.102346+00:00

| system | n | hits@1 | EM | F1 | p50 s | p95 s | $/query | $/1k q | Jev calls/q | LLM calls/q | in tok/q | status | linking |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| jev | 2 | 0.000 | 0.000 | 0.000 | 1.75 | 2.09 | 0.000286 | 0.286 | 3.00 | 0.00 | 6802 | ok=2 | 1.000 |
| agent-graph | 2 | 1.000 | 0.500 | 0.900 | 4.65 | 19.02 | 0.000780 | 0.780 | 0.00 | 6.50 | 6962 | ok=2 | 1.000 |
| agent-walk | 2 | 1.000 | 1.000 | 1.000 | 3.32 | 18.41 | 0.000997 | 0.997 | 5.00 | 6.00 | 5785 | ok=2 | 1.000 |

## Systems

- **jev** (wall 2s): `{"system": "graphwalk", "linking": "given", "decision_model": "typesafe/jev-1.13", "embedder": "fastembed:BAAI/bge-small-en-v1.5", "traversal": {"strategy": "greedy", "beam_width": 3, "hop_mode": "relation", "direction": "both", "exclude_visited": true, "allow_stop_at_start": false, "budget": {"max_depth": 4, "max_decision_calls": 12, "max_input_tokens": null}, "length_alpha": 1.0, "prob_floor": 0.001, "temperature": 1.0, "top_p": 1.0, "n_samples": 5, "seed": 0, "beam_batching": "per_depth", "max_request_tokens": 64000, "max_state_plus_question_tokens": 32000, "token_safety_margin": 0.2, "label_style": "opaque", "include_summaries": true, "show_types": true, "stop_style": "literal", "relation_glosses": {}, "answer_type": "hint", "answer_type_gate_min_p": 0.6, "prefilter_threshold": 50, "prefilter_top_n": 30, "max_frontier": 2000, "frontier_preview": 5}}`
- **agent-graph** (wall 46s): `{"system": "agent", "llm": "openrouter/openai/gpt-6-luna", "max_turns": 10}`
- **agent-walk** (wall 39s): `{"system": "agent", "llm": "openrouter/openai/gpt-6-luna", "max_turns": 10}`

## Agent arms

| arm | n | F1 | hits@1 | answered | turns (mean) | agent in-tok (mean) | agent out-tok | $/q (agent) | $/q (tools) | latency p50 / p90 s | tool calls |
|---|---|---|---|---|---|---|---|---|---|---|---|
| jev | 2 | 0.000 [0.00, 0.00] | 0.00 | 1.00 | 0.0 | 6,802 | 642 | 0.00000 | 0.00029 | 1.7 / 2.1 | - |
| agent-graph | 2 | 0.900 [0.80, 1.00] | 1.00 | 1.00 | 6.5 | 6,962 | 552 | 0.00078 | 0.00000 | 4.6 / 19.0 | answer 1.0, neighbors 3.0, relations 3.0 |
| agent-walk | 2 | 1.000 [1.00, 1.00] | 1.00 | 1.00 | 6.0 | 5,785 | 250 | 0.00062 | 0.00038 | 3.3 / 18.4 | answer 1.0, neighbors 1.5, relations 1.5, walk 2.0 |

## Paired F1

- agent-walk minus agent-graph: +0.100 [+0.00, +0.20] F1 (2 q)
- agent-walk minus jev: +1.000 [+1.00, +1.00] F1 (2 q)
