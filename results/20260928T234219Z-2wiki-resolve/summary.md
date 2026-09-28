# 2wiki-resolve

- params: `{"concurrency": 4, "dataset": "2wiki", "hops": null, "linking": "resolve", "n": 300, "rag_k": 5, "seed": 0, "split": "test", "total_questions": 6785, "variants": {"": {}}}`
- git: `13caa5bdfc7f61a8dfced0cbc395f37fda280b03`
- run at: 2026-09-28T23:42:19.669628+00:00

| system | n | hits@1 | EM | F1 | p50 s | p95 s | $/query | $/1k q | Jev calls/q | LLM calls/q | in tok/q | status | linking |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| graphwalk-greedy-v2 | 300 | 0.750 | 0.750 | 0.773 | 0.36 | 0.64 | 0.000050 | 0.050 | 1.46 | 0.00 | 1200 | no_entry=25, ok=275 | 0.737 |

## Systems

- **graphwalk-greedy-v2** (wall 25s): `{"system": "graphwalk", "linking": "resolve", "decision_model": "typesafe/jev-1.13", "embedder": "fastembed:BAAI/bge-small-en-v1.5", "traversal": {"strategy": "greedy", "beam_width": 3, "hop_mode": "entity", "direction": "both", "exclude_visited": true, "allow_stop_at_start": false, "budget": {"max_depth": 4, "max_decision_calls": 12, "max_input_tokens": null}, "length_alpha": 1.0, "prob_floor": 0.001, "temperature": 1.0, "top_p": 1.0, "n_samples": 5, "seed": 0, "beam_batching": "per_depth", "max_request_tokens": 64000, "max_state_plus_question_tokens": 32000, "token_safety_margin": 0.2, "label_style": "opaque", "include_summaries": true, "show_types": true, "stop_style": "literal", "relation_glosses": {}, "answer_type": "off", "answer_type_gate_min_p": 0.6, "prefilter_threshold": 50, "prefilter_top_n": 30, "max_frontier": 2000, "frontier_preview": 5}}`
