# 2wiki-gold

- params: `{"concurrency": 4, "dataset": "2wiki", "hops": null, "linking": "gold", "n": 300, "rag_k": 5, "seed": 0, "split": "test", "total_questions": 6785, "variants": {"-logprob": {}}}`
- git: `67896a922334df530d286bdf46c5a269809cc440`
- run at: 2026-10-02T02:40:30.401615+00:00

| system | n | hits@1 | EM | F1 | p50 s | p95 s | $/query | $/1k q | Jev calls/q | LLM calls/q | in tok/q | status | linking |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| graphwalk-greedy-v2-logprob | 300 | 0.853 | 0.853 | 0.888 | 2.55 | 6.37 | 0.000069 | 0.069 | 1.54 | 0.00 | 601 | ok=300 | 1.000 |

## Systems

- **graphwalk-greedy-v2-logprob** (wall 234s): `{"system": "graphwalk", "linking": "gold", "decision_model": "logprob:qwen/qwen3.8-27b", "embedder": "fastembed:BAAI/bge-small-en-v1.5", "traversal": {"strategy": "greedy", "beam_width": 3, "hop_mode": "entity", "direction": "both", "exclude_visited": true, "allow_stop_at_start": false, "budget": {"max_depth": 4, "max_decision_calls": 12, "max_input_tokens": null}, "length_alpha": 1.0, "prob_floor": 0.001, "temperature": 1.0, "top_p": 1.0, "n_samples": 5, "seed": 0, "beam_batching": "per_depth", "max_request_tokens": 64000, "max_state_plus_question_tokens": 32000, "token_safety_margin": 0.2, "label_style": "opaque", "include_summaries": true, "show_types": true, "stop_style": "literal", "relation_glosses": {}, "answer_type": "off", "answer_type_gate_min_p": 0.6, "prefilter_threshold": 50, "prefilter_top_n": 30, "max_frontier": 2000, "frontier_preview": 5}}`

## Calibration of walk confidence

| system | n | mean confidence | EM | ECE | Brier | AUROC | EM, most confident half |
|---|---|---|---|---|---|---|---|
| graphwalk-greedy-v2-logprob | 300 | 0.959 | 0.853 | 0.109 | 0.107 | 0.769 | 0.940 |
