# 2wiki-gold

- params: `{"concurrency": 4, "dataset": "2wiki", "hops": null, "linking": "gold", "n": 300, "rag_k": 5, "seed": 0, "total_questions": 6785}`
- git: `21163766ec298f95585180451806b2f936ebc3c7` (dirty)
- run at: 2026-09-28T20:43:54.562900+00:00

| system | n | hits@1 | EM | F1 | p50 s | p95 s | $/query | $/1k q | Jev calls/q | LLM calls/q | in tok/q | status | linking |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| graphwalk-greedy | 300 | 0.890 | 0.890 | 0.919 | 0.40 | 0.67 | 0.000048 | 0.048 | 1.51 | 0.00 | 1132 | ok=300 | 1.000 |
| graphwalk-beam | 300 | 0.890 | 0.890 | 0.919 | 0.44 | 0.79 | 0.000067 | 0.067 | 1.98 | 0.00 | 1607 | ok=300 | 1.000 |
| graphwalk-relation | 300 | 0.877 | 0.877 | 0.910 | 0.37 | 0.60 | 0.000036 | 0.036 | 1.53 | 0.00 | 854 | ok=300 | 1.000 |
| vector-rag | 300 | 0.600 | 0.600 | 0.600 | 1.57 | 2.83 | 0.000040 | 0.040 | 0.00 | 1.00 | 218 | ok=300 | n/a |

## Systems

- **graphwalk-greedy** (wall 27s): `{"system": "graphwalk", "linking": "gold", "decision_model": "typesafe/jev-1.13", "embedder": "fastembed:BAAI/bge-small-en-v1.5", "traversal": {"strategy": "greedy", "beam_width": 3, "hop_mode": "entity", "direction": "both", "exclude_visited": true, "allow_stop_at_start": false, "budget": {"max_depth": 4, "max_decision_calls": 12, "max_input_tokens": null}, "length_alpha": 1.0, "prob_floor": 0.001, "temperature": 1.0, "top_p": 1.0, "n_samples": 5, "seed": 0, "beam_batching": "per_depth", "max_request_tokens": 64000, "max_state_plus_question_tokens": 32000, "token_safety_margin": 0.2, "label_style": "opaque", "include_summaries": true, "prefilter_threshold": 50, "prefilter_top_n": 30, "max_frontier": 2000, "frontier_preview": 5}}`
- **graphwalk-beam** (wall 34s): `{"system": "graphwalk", "linking": "gold", "decision_model": "typesafe/jev-1.13", "embedder": "fastembed:BAAI/bge-small-en-v1.5", "traversal": {"strategy": "beam", "beam_width": 3, "hop_mode": "entity", "direction": "both", "exclude_visited": true, "allow_stop_at_start": false, "budget": {"max_depth": 4, "max_decision_calls": 12, "max_input_tokens": null}, "length_alpha": 1.0, "prob_floor": 0.001, "temperature": 1.0, "top_p": 1.0, "n_samples": 5, "seed": 0, "beam_batching": "per_depth", "max_request_tokens": 64000, "max_state_plus_question_tokens": 32000, "token_safety_margin": 0.2, "label_style": "opaque", "include_summaries": true, "prefilter_threshold": 50, "prefilter_top_n": 30, "max_frontier": 2000, "frontier_preview": 5}}`
- **graphwalk-relation** (wall 26s): `{"system": "graphwalk", "linking": "gold", "decision_model": "typesafe/jev-1.13", "embedder": "fastembed:BAAI/bge-small-en-v1.5", "traversal": {"strategy": "greedy", "beam_width": 3, "hop_mode": "relation", "direction": "both", "exclude_visited": true, "allow_stop_at_start": false, "budget": {"max_depth": 4, "max_decision_calls": 12, "max_input_tokens": null}, "length_alpha": 1.0, "prob_floor": 0.001, "temperature": 1.0, "top_p": 1.0, "n_samples": 5, "seed": 0, "beam_batching": "per_depth", "max_request_tokens": 64000, "max_state_plus_question_tokens": 32000, "token_safety_margin": 0.2, "label_style": "opaque", "include_summaries": true, "prefilter_threshold": 50, "prefilter_top_n": 30, "max_frontier": 2000, "frontier_preview": 5}}`
- **vector-rag** (wall 998s): `{"system": "vector-rag", "k": 5, "llm": "openrouter/openai/gpt-6-luna", "embedder": "fastembed:BAAI/bge-small-en-v1.5", "documents": 33091, "document": "one per entity: name, type, <= 40 incident triples"}`
