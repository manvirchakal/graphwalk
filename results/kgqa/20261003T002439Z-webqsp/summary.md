# webqsp

- params: `{"answer_in_graph": 191, "concurrency": 4, "dataset": "webqsp", "max_depth": 4, "n": 200, "preset": "relation-v2", "seed": 0, "split": "test", "total_questions": 1628}`
- git: `94769eacb7e65a2da4e6484de794b19cbe71bc6d`
- run at: 2026-10-03T00:24:39.872010+00:00

| system | n | hits@1 | EM | F1 | p50 s | p95 s | $/query | $/1k q | Jev calls/q | LLM calls/q | in tok/q | status | linking |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| graphwalk-relation-v2-logprob-shuf3 | 200 | 0.600 | 0.510 | 0.609 | 5.80 | 10.77 | 0.000277 | 0.277 | 2.25 | 0.00 | 2568 | ok=200 | 1.000 |

## Systems

- **graphwalk-relation-v2-logprob-shuf3** (wall 331s): `{"system": "graphwalk", "decision_model": "logprob:qwen/qwen3.8-27b", "embedder": "BAAI/bge-small-en-v1.5", "traversal": {"strategy": "greedy", "beam_width": 3, "hop_mode": "relation", "direction": "both", "exclude_visited": true, "allow_stop_at_start": false, "budget": {"max_depth": 4, "max_decision_calls": 12, "max_input_tokens": null}, "length_alpha": 1.0, "prob_floor": 0.001, "temperature": 1.0, "top_p": 1.0, "n_samples": 5, "seed": 0, "beam_batching": "per_depth", "max_request_tokens": 64000, "max_state_plus_question_tokens": 32000, "token_safety_margin": 0.2, "label_style": "opaque", "include_summaries": true, "show_types": true, "stop_style": "literal", "relation_glosses": {}, "answer_type": "hint", "answer_type_gate_min_p": 0.6, "prefilter_threshold": 50, "prefilter_top_n": 30, "max_frontier": 2000, "frontier_preview": 5}, "shuffle": 3, "graph": "per question"}`

## Calibration of walk confidence

| system | n | mean confidence | EM | ECE | Brier | AUROC | EM, most confident half |
|---|---|---|---|---|---|---|---|
| graphwalk-relation-v2-logprob-shuf3 | 200 | 0.808 | 0.510 | 0.298 | 0.285 | 0.785 | 0.700 |
