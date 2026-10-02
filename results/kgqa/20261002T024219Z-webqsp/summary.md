# webqsp

- params: `{"answer_in_graph": 50, "concurrency": 4, "dataset": "webqsp", "max_depth": 4, "n": 50, "preset": "relation-v2", "seed": 0, "split": "test", "total_questions": 1628}`
- git: `67896a922334df530d286bdf46c5a269809cc440` (dirty)
- run at: 2026-10-02T02:42:19.340954+00:00

| system | n | hits@1 | EM | F1 | p50 s | p95 s | $/query | $/1k q | Jev calls/q | LLM calls/q | in tok/q | status | linking |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| graphwalk-relation-v2-logprob | 50 | 0.620 | 0.480 | 0.642 | 5.61 | 12.99 | 0.000336 | 0.336 | 2.34 | 0.00 | 2638 | ok=50 | 1.000 |

## Systems

- **graphwalk-relation-v2-logprob** (wall 86s): `{"system": "graphwalk", "decision_model": "logprob:qwen/qwen3.8-27b", "embedder": "BAAI/bge-small-en-v1.5", "traversal": {"strategy": "greedy", "beam_width": 3, "hop_mode": "relation", "direction": "both", "exclude_visited": true, "allow_stop_at_start": false, "budget": {"max_depth": 4, "max_decision_calls": 12, "max_input_tokens": null}, "length_alpha": 1.0, "prob_floor": 0.001, "temperature": 1.0, "top_p": 1.0, "n_samples": 5, "seed": 0, "beam_batching": "per_depth", "max_request_tokens": 64000, "max_state_plus_question_tokens": 32000, "token_safety_margin": 0.2, "label_style": "opaque", "include_summaries": true, "show_types": true, "stop_style": "literal", "relation_glosses": {}, "answer_type": "hint", "answer_type_gate_min_p": 0.6, "prefilter_threshold": 50, "prefilter_top_n": 30, "max_frontier": 2000, "frontier_preview": 5}, "graph": "per question"}`

## Calibration of walk confidence

| system | n | mean confidence | EM | ECE | Brier | AUROC | EM, most confident half |
|---|---|---|---|---|---|---|---|
| graphwalk-relation-v2-logprob | 50 | 0.865 | 0.480 | 0.385 | 0.334 | 0.827 | 0.720 |
