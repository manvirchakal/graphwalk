# webqsp

- params: `{"answer_in_graph": 191, "concurrency": 4, "dataset": "webqsp", "max_depth": 4, "n": 200, "preset": "relation-v2", "seed": 0, "split": "test", "total_questions": 1628}`
- git: `bdfeebde970ccdf7da1556e282d67cdd45efaf5a`
- run at: 2026-10-03T00:18:51.540067+00:00

| system | n | hits@1 | EM | F1 | p50 s | p95 s | $/query | $/1k q | Jev calls/q | LLM calls/q | in tok/q | status | linking |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| graphwalk-relation-v2-logprob-shuf2 | 200 | 0.600 | 0.510 | 0.617 | 6.13 | 9.59 | 0.000283 | 0.283 | 2.23 | 0.00 | 2547 | ok=200 | 1.000 |

## Systems

- **graphwalk-relation-v2-logprob-shuf2** (wall 331s): `{"system": "graphwalk", "decision_model": "logprob:qwen/qwen3.8-27b", "embedder": "BAAI/bge-small-en-v1.5", "traversal": {"strategy": "greedy", "beam_width": 3, "hop_mode": "relation", "direction": "both", "exclude_visited": true, "allow_stop_at_start": false, "budget": {"max_depth": 4, "max_decision_calls": 12, "max_input_tokens": null}, "length_alpha": 1.0, "prob_floor": 0.001, "temperature": 1.0, "top_p": 1.0, "n_samples": 5, "seed": 0, "beam_batching": "per_depth", "max_request_tokens": 64000, "max_state_plus_question_tokens": 32000, "token_safety_margin": 0.2, "label_style": "opaque", "include_summaries": true, "show_types": true, "stop_style": "literal", "relation_glosses": {}, "answer_type": "hint", "answer_type_gate_min_p": 0.6, "prefilter_threshold": 50, "prefilter_top_n": 30, "max_frontier": 2000, "frontier_preview": 5}, "shuffle": 2, "graph": "per question"}`

## Calibration of walk confidence

| system | n | mean confidence | EM | ECE | Brier | AUROC | EM, most confident half |
|---|---|---|---|---|---|---|---|
| graphwalk-relation-v2-logprob-shuf2 | 200 | 0.817 | 0.510 | 0.307 | 0.292 | 0.791 | 0.710 |
