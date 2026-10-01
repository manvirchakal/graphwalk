# cwq

- params: `{"answer_in_graph": 40, "concurrency": 4, "dataset": "cwq", "max_depth": 4, "n": 50, "preset": "relation-v2", "seed": 0, "split": "test", "total_questions": 3531}`
- git: `0a5b2d4104b9ca076ea4720f20f5bb62cbece8f2`
- run at: 2026-10-01T16:03:24.455694+00:00

| system | n | hits@1 | EM | F1 | p50 s | p95 s | $/query | $/1k q | Jev calls/q | LLM calls/q | in tok/q | status | linking |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| graphwalk-relation-v2-jev | 50 | 0.280 | 0.200 | 0.315 | 1.66 | 3.25 | 0.000218 | 0.218 | 2.34 | 0.00 | 5186 | ok=50 | 1.000 |
| graphwalk-relation-v2-llm | 50 | 0.280 | 0.200 | 0.333 | 33.75 | 82.88 | 0.001139 | 1.139 | 2.58 | 0.00 | 4356 | error=5, ok=45 | 1.000 |
| llm-path-retry | 50 | 0.260 | 0.140 | 0.293 | 8.12 | 40.38 | 0.001227 | 1.227 | 0.00 | 1.70 | 8736 | ok=50 | 1.000 |

## Systems

- **graphwalk-relation-v2-jev** (wall 27s): `{"system": "graphwalk", "decision_model": "typesafe/jev-1.13", "embedder": "BAAI/bge-small-en-v1.5", "traversal": {"strategy": "greedy", "beam_width": 3, "hop_mode": "relation", "direction": "both", "exclude_visited": true, "allow_stop_at_start": false, "budget": {"max_depth": 4, "max_decision_calls": 12, "max_input_tokens": null}, "length_alpha": 1.0, "prob_floor": 0.001, "temperature": 1.0, "top_p": 1.0, "n_samples": 5, "seed": 0, "beam_batching": "per_depth", "max_request_tokens": 64000, "max_state_plus_question_tokens": 32000, "token_safety_margin": 0.2, "label_style": "opaque", "include_summaries": true, "show_types": true, "stop_style": "literal", "relation_glosses": {}, "answer_type": "hint", "answer_type_gate_min_p": 0.6, "prefilter_threshold": 50, "prefilter_top_n": 30, "max_frontier": 2000, "frontier_preview": 5}, "graph": "per question"}`
- **graphwalk-relation-v2-llm** (wall 499s): `{"system": "graphwalk", "decision_model": "llm-decider:openrouter/openai/gpt-6-luna", "embedder": "BAAI/bge-small-en-v1.5", "traversal": {"strategy": "greedy", "beam_width": 3, "hop_mode": "relation", "direction": "both", "exclude_visited": true, "allow_stop_at_start": false, "budget": {"max_depth": 4, "max_decision_calls": 12, "max_input_tokens": null}, "length_alpha": 1.0, "prob_floor": 0.001, "temperature": 1.0, "top_p": 1.0, "n_samples": 5, "seed": 0, "beam_batching": "per_depth", "max_request_tokens": 64000, "max_state_plus_question_tokens": 32000, "token_safety_margin": 0.2, "label_style": "opaque", "include_summaries": true, "show_types": true, "stop_style": "literal", "relation_glosses": {}, "answer_type": "hint", "answer_type_gate_min_p": 0.6, "prefilter_threshold": 50, "prefilter_top_n": 30, "max_frontier": 2000, "frontier_preview": 5}, "graph": "per question"}`
- **llm-path-retry** (wall 287s): `{"system": "llm-path", "llm": "openrouter/openai/gpt-6-luna", "retries": 2, "graph": "per question"}`

## Calibration of walk confidence

| system | n | mean confidence | EM | ECE | Brier | AUROC | EM, most confident half |
|---|---|---|---|---|---|---|---|
| graphwalk-relation-v2-jev | 50 | 0.671 | 0.200 | 0.471 | 0.366 | 0.710 | 0.280 |
| graphwalk-relation-v2-llm | 50 | 0.679 | 0.200 | 0.491 | 0.419 | 0.502 | 0.160 |
