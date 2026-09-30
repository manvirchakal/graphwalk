# 2wiki-gold

- params: `{"concurrency": 4, "dataset": "2wiki", "hops": null, "linking": "gold", "n": 300, "rag_k": 5, "seed": 0, "split": "test", "total_questions": 6785, "variants": {"-llm": {}}}`
- git: `cd0718e24536d4984f634a30ccb2c8c9dc34e17d`
- run at: 2026-09-30T16:19:21.888793+00:00

| system | n | hits@1 | EM | F1 | p50 s | p95 s | $/query | $/1k q | Jev calls/q | LLM calls/q | in tok/q | status | linking |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| graphwalk-greedy-v2-llm | 300 | 0.927 | 0.927 | 0.941 | 18.58 | 40.57 | 0.000164 | 0.164 | 1.52 | 0.00 | 793 | ok=300 | 1.000 |

## Systems

- **graphwalk-greedy-v2-llm** (wall 1538s): `{"system": "graphwalk", "linking": "gold", "decision_model": "llm-decider:openrouter/openai/gpt-6-luna", "embedder": "fastembed:BAAI/bge-small-en-v1.5", "traversal": {"strategy": "greedy", "beam_width": 3, "hop_mode": "entity", "direction": "both", "exclude_visited": true, "allow_stop_at_start": false, "budget": {"max_depth": 4, "max_decision_calls": 12, "max_input_tokens": null}, "length_alpha": 1.0, "prob_floor": 0.001, "temperature": 1.0, "top_p": 1.0, "n_samples": 5, "seed": 0, "beam_batching": "per_depth", "max_request_tokens": 64000, "max_state_plus_question_tokens": 32000, "token_safety_margin": 0.2, "label_style": "opaque", "include_summaries": true, "show_types": true, "stop_style": "literal", "relation_glosses": {}, "answer_type": "off", "answer_type_gate_min_p": 0.6, "prefilter_threshold": 50, "prefilter_top_n": 30, "max_frontier": 2000, "frontier_preview": 5}}`

## Calibration of walk confidence

| system | n | mean confidence | EM | ECE | Brier | AUROC | EM, most confident half |
|---|---|---|---|---|---|---|---|
| graphwalk-greedy-v2-llm | 300 | 0.982 | 0.927 | 0.056 | 0.055 | 0.687 | 0.960 |
