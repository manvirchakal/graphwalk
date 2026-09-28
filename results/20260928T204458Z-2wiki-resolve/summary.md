# 2wiki-resolve

- params: `{"concurrency": 4, "dataset": "2wiki", "hops": null, "linking": "resolve", "n": 300, "rag_k": 5, "seed": 0, "total_questions": 6785}`
- git: `21163766ec298f95585180451806b2f936ebc3c7` (dirty)
- run at: 2026-09-28T20:44:58.711327+00:00

| system | n | hits@1 | EM | F1 | p50 s | p95 s | $/query | $/1k q | Jev calls/q | LLM calls/q | in tok/q | status | linking |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| graphwalk-greedy | 300 | 0.573 | 0.573 | 0.574 | 0.25 | 0.64 | 0.000049 | 0.049 | 1.42 | 0.00 | 1178 | no_entry=16, ok=284 | 0.553 |
| graphwalk-beam | 300 | 0.767 | 0.767 | 0.784 | 0.42 | 0.80 | 0.000071 | 0.071 | 1.85 | 0.00 | 1692 | no_entry=16, ok=284 | 0.553 |

## Systems

- **graphwalk-greedy** (wall 25s): `{"system": "graphwalk", "linking": "resolve", "decision_model": "typesafe/jev-1.13", "embedder": "fastembed:BAAI/bge-small-en-v1.5", "traversal": {"strategy": "greedy", "beam_width": 3, "hop_mode": "entity", "direction": "both", "exclude_visited": true, "allow_stop_at_start": false, "budget": {"max_depth": 4, "max_decision_calls": 12, "max_input_tokens": null}, "length_alpha": 1.0, "prob_floor": 0.001, "temperature": 1.0, "top_p": 1.0, "n_samples": 5, "seed": 0, "beam_batching": "per_depth", "max_request_tokens": 64000, "max_state_plus_question_tokens": 32000, "token_safety_margin": 0.2, "label_style": "opaque", "include_summaries": true, "prefilter_threshold": 50, "prefilter_top_n": 30, "max_frontier": 2000, "frontier_preview": 5}}`
- **graphwalk-beam** (wall 33s): `{"system": "graphwalk", "linking": "resolve", "decision_model": "typesafe/jev-1.13", "embedder": "fastembed:BAAI/bge-small-en-v1.5", "traversal": {"strategy": "beam", "beam_width": 3, "hop_mode": "entity", "direction": "both", "exclude_visited": true, "allow_stop_at_start": false, "budget": {"max_depth": 4, "max_decision_calls": 12, "max_input_tokens": null}, "length_alpha": 1.0, "prob_floor": 0.001, "temperature": 1.0, "top_p": 1.0, "n_samples": 5, "seed": 0, "beam_batching": "per_depth", "max_request_tokens": 64000, "max_state_plus_question_tokens": 32000, "token_safety_margin": 0.2, "label_style": "opaque", "include_summaries": true, "prefilter_threshold": 50, "prefilter_top_n": 30, "max_frontier": 2000, "frontier_preview": 5}}`
