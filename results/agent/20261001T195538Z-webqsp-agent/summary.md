# webqsp-agent

- params: `{"arms": ["jev", "closedbook", "graph", "walk", "search"], "concurrency": 4, "dataset": "webqsp", "edges": 302873, "graph": "subgraphs of 100 sampled q", "llm": "openrouter/openai/gpt-6-luna", "max_turns": 10, "n": 100, "nodes": 103253, "sample": 100, "seed": 0, "split": "test"}`
- git: `88315131de4f4616ddedd10dfc0a092fe179bc7e`
- run at: 2026-10-01T19:55:38.350701+00:00

| system | n | hits@1 | EM | F1 | p50 s | p95 s | $/query | $/1k q | Jev calls/q | LLM calls/q | in tok/q | status | linking |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| jev | 100 | 0.520 | 0.410 | 0.511 | 0.54 | 1.54 | 0.000220 | 0.220 | 2.27 | 0.00 | 5231 | ok=100 | 1.000 |
| closedbook | 100 | 0.550 | 0.210 | 0.422 | 2.61 | 5.67 | 0.000085 | 0.085 | 0.00 | 1.00 | 197 | ok=100 | 1.000 |
| agent-graph | 100 | 0.830 | 0.550 | 0.739 | 7.58 | 29.61 | 0.000882 | 0.882 | 0.00 | 5.48 | 9421 | ok=100 | 1.000 |
| agent-walk | 100 | 0.760 | 0.530 | 0.721 | 6.50 | 28.15 | 0.001002 | 1.002 | 4.18 | 4.80 | 5979 | ok=100 | 1.000 |
| agent-search | 100 | 0.810 | 0.420 | 0.677 | 3.39 | 22.85 | 0.000442 | 0.442 | 0.00 | 4.11 | 4377 | ok=100 | 1.000 |

## Systems

- **jev** (wall 20s): `{"system": "graphwalk", "linking": "given", "decision_model": "typesafe/jev-1.13", "embedder": "fastembed:BAAI/bge-small-en-v1.5", "traversal": {"strategy": "greedy", "beam_width": 3, "hop_mode": "relation", "direction": "both", "exclude_visited": true, "allow_stop_at_start": false, "budget": {"max_depth": 4, "max_decision_calls": 12, "max_input_tokens": null}, "length_alpha": 1.0, "prob_floor": 0.001, "temperature": 1.0, "top_p": 1.0, "n_samples": 5, "seed": 0, "beam_batching": "per_depth", "max_request_tokens": 64000, "max_state_plus_question_tokens": 32000, "token_safety_margin": 0.2, "label_style": "opaque", "include_summaries": true, "show_types": true, "stop_style": "literal", "relation_glosses": {}, "answer_type": "hint", "answer_type_gate_min_p": 0.6, "prefilter_threshold": 50, "prefilter_top_n": 30, "max_frontier": 2000, "frontier_preview": 5}}`
- **closedbook** (wall 334s): `{"system": "agent", "llm": "openrouter/openai/gpt-6-luna", "max_turns": 1}`
- **agent-graph** (wall 1824s): `{"system": "agent", "llm": "openrouter/openai/gpt-6-luna", "max_turns": 10}`
- **agent-walk** (wall 1603s): `{"system": "agent", "llm": "openrouter/openai/gpt-6-luna", "max_turns": 10}`
- **agent-search** (wall 1373s): `{"system": "agent", "llm": "openrouter/openai/gpt-6-luna", "max_turns": 10}`

## Agent arms

| arm | n | F1 | hits@1 | answered | turns (mean) | agent in-tok (mean) | agent out-tok | $/q (agent) | $/q (tools) | latency p50 / p90 s | grounded | tool calls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| jev | 100 | 0.511 [0.42, 0.60] | 0.52 | 1.00 | 0.0 | 5,231 | 492 | 0.00000 | 0.00022 | 0.5 / 1.1 | - | - |
| closedbook | 100 | 0.422 [0.34, 0.50] | 0.55 | 1.00 | 1.0 | 197 | 131 | 0.00009 | 0.00000 | 2.6 / 5.0 | 0.00 | answer 1.0 |
| agent-graph | 100 | 0.739 [0.67, 0.81] | 0.83 | 1.00 | 5.5 | 9,421 | 620 | 0.00088 | 0.00000 | 7.6 / 25.2 | 0.91 | answer 1.0, neighbors 6.5, relations 3.2 |
| agent-walk | 100 | 0.721 [0.64, 0.80] | 0.76 | 1.00 | 4.8 | 5,979 | 438 | 0.00062 | 0.00039 | 6.5 / 25.3 | 0.94 | answer 1.0, neighbors 3.2, relations 1.4, walk 1.7 |
| agent-search | 100 | 0.677 [0.60, 0.75] | 0.81 | 1.00 | 4.1 | 4,377 | 254 | 0.00044 | 0.00000 | 3.4 / 17.0 | 0.92 | answer 1.0, search 3.4 |

## Paired F1

- agent-walk minus agent-graph: -0.018 [-0.09, +0.05] F1 (100 q)
- agent-walk minus agent-search: +0.045 [-0.02, +0.11] F1 (100 q)
- agent-graph minus agent-search: +0.062 [+0.00, +0.12] F1 (100 q)
- agent-walk minus jev: +0.210 [+0.12, +0.30] F1 (100 q)
- agent-walk minus closedbook: +0.299 [+0.21, +0.39] F1 (100 q)
