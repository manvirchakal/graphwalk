# webqsp

- params: `{"answer_in_graph": 50, "concurrency": 4, "dataset": "webqsp", "max_depth": 4, "n": 50, "preset": "relation-v2", "seed": 0, "split": "test", "total_questions": 1628}`
- git: `deeb81e3adf4ae70a674bac31db3f06ed15306eb`
- run at: 2026-10-01T15:38:29.453282+00:00

| system | n | hits@1 | EM | F1 | p50 s | p95 s | $/query | $/1k q | Jev calls/q | LLM calls/q | in tok/q | status | linking |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| graphwalk-relation-v2-jev | 50 | 0.540 | 0.440 | 0.537 | 3.41 | 8.86 | 0.000207 | 0.207 | 2.18 | 0.00 | 4923 | ok=50 | 1.000 |
| graphwalk-relation-v2-llm | 50 | 0.660 | 0.560 | 0.694 | 26.72 | 75.31 | 0.000907 | 0.907 | 2.36 | 0.00 | 3880 | error=1, ok=49 | 1.000 |
| llm-path-retry | 50 | 0.740 | 0.560 | 0.758 | 2.58 | 10.59 | 0.000724 | 0.724 | 0.00 | 1.08 | 5558 | ok=50 | 1.000 |

## Systems

- **graphwalk-relation-v2-jev** (wall 49s): `{"system": "graphwalk", "decision_model": "typesafe/jev-1.13", "embedder": "BAAI/bge-small-en-v1.5", "traversal": {"strategy": "greedy", "beam_width": 3, "hop_mode": "relation", "direction": "both", "exclude_visited": true, "allow_stop_at_start": false, "budget": {"max_depth": 4, "max_decision_calls": 12, "max_input_tokens": null}, "length_alpha": 1.0, "prob_floor": 0.001, "temperature": 1.0, "top_p": 1.0, "n_samples": 5, "seed": 0, "beam_batching": "per_depth", "max_request_tokens": 64000, "max_state_plus_question_tokens": 32000, "token_safety_margin": 0.2, "label_style": "opaque", "include_summaries": true, "show_types": true, "stop_style": "literal", "relation_glosses": {}, "answer_type": "hint", "answer_type_gate_min_p": 0.6, "prefilter_threshold": 50, "prefilter_top_n": 30, "max_frontier": 2000, "frontier_preview": 5}, "graph": "per question"}`
- **graphwalk-relation-v2-llm** (wall 417s): `{"system": "graphwalk", "decision_model": "llm-decider:openrouter/openai/gpt-6-luna", "embedder": "BAAI/bge-small-en-v1.5", "traversal": {"strategy": "greedy", "beam_width": 3, "hop_mode": "relation", "direction": "both", "exclude_visited": true, "allow_stop_at_start": false, "budget": {"max_depth": 4, "max_decision_calls": 12, "max_input_tokens": null}, "length_alpha": 1.0, "prob_floor": 0.001, "temperature": 1.0, "top_p": 1.0, "n_samples": 5, "seed": 0, "beam_batching": "per_depth", "max_request_tokens": 64000, "max_state_plus_question_tokens": 32000, "token_safety_margin": 0.2, "label_style": "opaque", "include_summaries": true, "show_types": true, "stop_style": "literal", "relation_glosses": {}, "answer_type": "hint", "answer_type_gate_min_p": 0.6, "prefilter_threshold": 50, "prefilter_top_n": 30, "max_frontier": 2000, "frontier_preview": 5}, "graph": "per question"}`
- **llm-path-retry** (wall 178s): `{"system": "llm-path", "llm": "openrouter/openai/gpt-6-luna", "retries": 2, "graph": "per question"}`

## Calibration of walk confidence

| system | n | mean confidence | EM | ECE | Brier | AUROC | EM, most confident half |
|---|---|---|---|---|---|---|---|
| graphwalk-relation-v2-jev | 50 | 0.821 | 0.440 | 0.381 | 0.308 | 0.919 | 0.720 |
| graphwalk-relation-v2-llm | 50 | 0.714 | 0.560 | 0.224 | 0.258 | 0.627 | 0.600 |
