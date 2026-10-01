# webqsp-agent

- params: `{"arms": ["search", "jev", "closedbook", "graph", "walk"], "concurrency": 4, "dataset": "webqsp", "edges": 302873, "graph": "subgraphs of 100 sampled q", "llm": "openrouter/openai/gpt-6-luna", "max_turns": 10, "n": 20, "nodes": 103253, "sample": 100, "seed": 0, "split": "test"}`
- git: `82aed006bf2011562c7d5d9ba6ca1517dcb4cb92`
- run at: 2026-10-01T18:29:02.254259+00:00

| system | n | hits@1 | EM | F1 | p50 s | p95 s | $/query | $/1k q | Jev calls/q | LLM calls/q | in tok/q | status | linking |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| agent-search | 20 | 0.750 | 0.350 | 0.580 | 3.61 | 24.71 | 0.000451 | 0.451 | 0.00 | 4.80 | 4170 | ok=20 | 1.000 |
| jev | 20 | 0.550 | 0.400 | 0.554 | 1.13 | 4.68 | 0.000221 | 0.221 | 2.35 | 0.00 | 5255 | ok=20 | 1.000 |
| closedbook | 20 | 0.600 | 0.200 | 0.443 | 2.31 | 4.20 | 0.000065 | 0.065 | 0.00 | 1.00 | 198 | ok=20 | 1.000 |
| agent-graph | 20 | 0.750 | 0.450 | 0.669 | 5.86 | 30.31 | 0.000953 | 0.953 | 0.00 | 5.70 | 9723 | ok=20 | 1.000 |
| agent-walk | 20 | 0.750 | 0.600 | 0.776 | 4.68 | 21.39 | 0.000910 | 0.910 | 4.55 | 4.40 | 5241 | ok=20 | 1.000 |

## Systems

- **agent-search** (wall 321s): `{"system": "agent", "llm": "openrouter/openai/gpt-6-luna", "max_turns": 10}`
- **jev** (wall 9s): `{"system": "graphwalk", "linking": "given", "decision_model": "typesafe/jev-1.13", "embedder": "fastembed:BAAI/bge-small-en-v1.5", "traversal": {"strategy": "greedy", "beam_width": 3, "hop_mode": "relation", "direction": "both", "exclude_visited": true, "allow_stop_at_start": false, "budget": {"max_depth": 4, "max_decision_calls": 12, "max_input_tokens": null}, "length_alpha": 1.0, "prob_floor": 0.001, "temperature": 1.0, "top_p": 1.0, "n_samples": 5, "seed": 0, "beam_batching": "per_depth", "max_request_tokens": 64000, "max_state_plus_question_tokens": 32000, "token_safety_margin": 0.2, "label_style": "opaque", "include_summaries": true, "show_types": true, "stop_style": "literal", "relation_glosses": {}, "answer_type": "hint", "answer_type_gate_min_p": 0.6, "prefilter_threshold": 50, "prefilter_top_n": 30, "max_frontier": 2000, "frontier_preview": 5}}`
- **closedbook** (wall 65s): `{"system": "agent", "llm": "openrouter/openai/gpt-6-luna", "max_turns": 1}`
- **agent-graph** (wall 392s): `{"system": "agent", "llm": "openrouter/openai/gpt-6-luna", "max_turns": 10}`
- **agent-walk** (wall 292s): `{"system": "agent", "llm": "openrouter/openai/gpt-6-luna", "max_turns": 10}`

## Agent arms

| arm | n | F1 | hits@1 | answered | turns (mean) | agent in-tok (mean) | agent out-tok | $/q (agent) | $/q (tools) | latency p50 / p90 s | grounded | tool calls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| agent-search | 20 | 0.580 [0.41, 0.76] | 0.75 | 1.00 | 4.8 | 4,170 | 238 | 0.00045 | 0.00000 | 3.6 / 17.9 | 0.80 | answer 1.0, search 3.8 |
| jev | 20 | 0.554 [0.37, 0.75] | 0.55 | 1.00 | 0.0 | 5,255 | 495 | 0.00000 | 0.00022 | 1.1 / 4.6 | - | - |
| closedbook | 20 | 0.443 [0.27, 0.62] | 0.60 | 1.00 | 1.0 | 198 | 91 | 0.00007 | 0.00000 | 2.3 / 4.0 | 0.00 | answer 1.0 |
| agent-graph | 20 | 0.669 [0.49, 0.85] | 0.75 | 1.00 | 5.7 | 9,723 | 664 | 0.00095 | 0.00000 | 5.9 / 27.5 | 0.84 | answer 1.0, neighbors 6.0, relations 3.4 |
| agent-walk | 20 | 0.776 [0.61, 0.92] | 0.75 | 1.00 | 4.4 | 5,241 | 276 | 0.00047 | 0.00044 | 4.7 / 21.3 | 0.85 | answer 1.0, neighbors 1.6, relations 0.8, walk 1.7 |

## Paired F1

- agent-walk minus agent-graph: +0.107 [-0.04, +0.27] F1 (20 q)
- agent-walk minus agent-search: +0.196 [+0.06, +0.36] F1 (20 q)
- agent-graph minus agent-search: +0.089 [-0.10, +0.28] F1 (20 q)
- agent-walk minus jev: +0.222 [+0.04, +0.42] F1 (20 q)
- agent-walk minus closedbook: +0.333 [+0.18, +0.49] F1 (20 q)
