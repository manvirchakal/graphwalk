# webqsp

- params: `{"answer_in_graph": 477, "concurrency": 4, "dataset": "webqsp", "max_depth": 4, "n": 500, "preset": "relation-v2", "seed": 0, "split": "test", "total_questions": 1628}`
- git: `8c769dc7e29342bc62fddb4c2b8cb671647a4782`
- run at: 2026-10-02T03:12:06.393683+00:00

| system | n | hits@1 | EM | F1 | p50 s | p95 s | $/query | $/1k q | Jev calls/q | LLM calls/q | in tok/q | status | linking |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| graphwalk-relation-v2-jev | 500 | 0.544 | 0.446 | 0.542 | 1.84 | 3.12 | 0.000215 | 0.215 | 2.24 | 0.00 | 5129 | ok=500 | 1.000 |
| llm-path-retry | 500 | 0.684 | 0.560 | 0.692 | 2.38 | 15.81 | 0.000784 | 0.784 | 0.00 | 1.14 | 5770 | ok=500 | 1.000 |

## Systems

- **graphwalk-relation-v2-jev** (wall 266s): `{"system": "graphwalk", "decision_model": "typesafe/jev-1.13", "embedder": "BAAI/bge-small-en-v1.5", "traversal": {"strategy": "greedy", "beam_width": 3, "hop_mode": "relation", "direction": "both", "exclude_visited": true, "allow_stop_at_start": false, "budget": {"max_depth": 4, "max_decision_calls": 12, "max_input_tokens": null}, "length_alpha": 1.0, "prob_floor": 0.001, "temperature": 1.0, "top_p": 1.0, "n_samples": 5, "seed": 0, "beam_batching": "per_depth", "max_request_tokens": 64000, "max_state_plus_question_tokens": 32000, "token_safety_margin": 0.2, "label_style": "opaque", "include_summaries": true, "show_types": true, "stop_style": "literal", "relation_glosses": {}, "answer_type": "hint", "answer_type_gate_min_p": 0.6, "prefilter_threshold": 50, "prefilter_top_n": 30, "max_frontier": 2000, "frontier_preview": 5}, "graph": "per question"}`
- **llm-path-retry** (wall 663s): `{"system": "llm-path", "llm": "openrouter/openai/gpt-6-luna", "retries": 2, "graph": "per question"}`

## Calibration of walk confidence

| system | n | mean confidence | EM | ECE | Brier | AUROC | EM, most confident half |
|---|---|---|---|---|---|---|---|
| graphwalk-relation-v2-jev | 500 | 0.801 | 0.446 | 0.355 | 0.307 | 0.855 | 0.704 |
