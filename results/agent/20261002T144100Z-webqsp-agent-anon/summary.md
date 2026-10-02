# webqsp-agent

- params: `{"anonymized": true, "arms": ["jev"], "concurrency": 4, "dataset": "webqsp", "edges": 302871, "graph": "subgraphs of 100 sampled q", "llm": "openrouter/openai/gpt-6-luna", "max_turns": 10, "n": 100, "nodes": 103194, "sample": 100, "seed": 0, "split": "test", "start": 0}`
- git: `3c7eb6b3bd89479d76b100b3f280a898d07e0c3f`
- run at: 2026-10-02T14:41:00.180358+00:00

| system | n | hits@1 | EM | F1 | p50 s | p95 s | $/query | $/1k q | Jev calls/q | LLM calls/q | in tok/q | status | linking |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| jev | 100 | 0.360 | 0.290 | 0.360 | 0.68 | 5.09 | 0.000244 | 0.244 | 2.43 | 0.00 | 5819 | ok=100 | 1.000 |

## Systems

- **jev** (wall 33s): `{"system": "graphwalk", "linking": "given", "decision_model": "typesafe/jev-1.13", "embedder": "fastembed:BAAI/bge-small-en-v1.5", "traversal": {"strategy": "greedy", "beam_width": 3, "hop_mode": "relation", "direction": "both", "exclude_visited": true, "allow_stop_at_start": false, "budget": {"max_depth": 4, "max_decision_calls": 12, "max_input_tokens": null}, "length_alpha": 1.0, "prob_floor": 0.001, "temperature": 1.0, "top_p": 1.0, "n_samples": 5, "seed": 0, "beam_batching": "per_depth", "max_request_tokens": 64000, "max_state_plus_question_tokens": 32000, "token_safety_margin": 0.2, "label_style": "opaque", "include_summaries": true, "show_types": true, "stop_style": "literal", "relation_glosses": {}, "answer_type": "hint", "answer_type_gate_min_p": 0.6, "prefilter_threshold": 50, "prefilter_top_n": 30, "max_frontier": 2000, "frontier_preview": 5}}`

## Agent arms

| arm | n | F1 | hits@1 | answered | turns (mean) | agent in-tok (mean) | agent out-tok | $/q (agent) | $/q (tools) | latency p50 / p90 s | grounded | tool calls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| jev | 100 | 0.360 [0.27, 0.45] | 0.36 | 1.00 | 0.0 | 5,819 | 528 | 0.00000 | 0.00024 | 0.7 / 2.1 | - | - |

## Paired F1


