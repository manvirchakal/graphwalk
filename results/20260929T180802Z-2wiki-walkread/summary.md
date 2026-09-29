# 2wiki-text

- params: `{"dataset": "2wiki", "embedder": "BAAI/bge-small-en-v1.5", "graph": "2wiki-330b8be64c4322b1.graph.json", "llm": "openrouter/openai/gpt-6-luna", "max_docs": 5, "n": 120, "per_type": 30, "preset": "greedy-v2", "seed": 0}`
- git: `53e516efd2f84cb2bce7f509757902498d25bd38`
- run at: 2026-09-29T18:21:36.134366+00:00

| system | n | hits@1 | EM | F1 | p50 s | p95 s | $/query | $/1k q | Jev calls/q | LLM calls/q | in tok/q | status | linking |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| graphwalk-schema | 120 | 0.167 | 0.167 | 0.251 | 0.86 | 1.57 | 0.000187 | 0.187 | 3.59 | 0.00 | 4455 | ok=120 | n/a |
| graphwalk-reader-source | 120 | 0.592 | 0.592 | 0.650 | 2.94 | 4.81 | 0.000337 | 0.337 | 4.71 | 1.00 | 6457 | ok=120 | n/a |
| graphwalk-reader-source-schema | 120 | 0.600 | 0.600 | 0.662 | 2.90 | 4.28 | 0.000349 | 0.349 | 4.92 | 1.00 | 6644 | ok=120 | n/a |

## Systems

- **graphwalk-schema** (wall 14s): `{"system": "graphwalk", "linking": "choice", "decision_model": "typesafe/jev-1.13", "embedder": "fastembed:BAAI/bge-small-en-v1.5", "traversal": {"strategy": "greedy", "beam_width": 3, "hop_mode": "entity", "direction": "both", "exclude_visited": true, "allow_stop_at_start": false, "budget": {"max_depth": 4, "max_decision_calls": 12, "max_input_tokens": null}, "length_alpha": 1.0, "prob_floor": 0.001, "temperature": 1.0, "top_p": 1.0, "n_samples": 5, "seed": 0, "beam_batching": "per_depth", "max_request_tokens": 64000, "max_state_plus_question_tokens": 32000, "token_safety_margin": 0.2, "label_style": "opaque", "include_summaries": true, "show_types": true, "stop_style": "literal", "relation_glosses": {}, "answer_type": "hint", "answer_type_gate_min_p": 0.6, "prefilter_threshold": 50, "prefilter_top_n": 30, "max_frontier": 2000, "frontier_preview": 5}}`
- **graphwalk-reader-source** (wall 400s): `{"system": "graphwalk-reader", "decision_model": "typesafe/jev-1.13", "llm": "openrouter/openai/gpt-6-luna", "embedder": "fastembed:BAAI/bge-small-en-v1.5", "max_entries": 2, "max_nodes": 20, "max_facts": 12, "context": "source paragraphs", "max_docs": 5, "traversal": {"strategy": "greedy", "beam_width": 3, "hop_mode": "entity", "direction": "both", "exclude_visited": true, "allow_stop_at_start": false, "budget": {"max_depth": 4, "max_decision_calls": 12, "max_input_tokens": null}, "length_alpha": 1.0, "prob_floor": 0.001, "temperature": 1.0, "top_p": 1.0, "n_samples": 5, "seed": 0, "beam_batching": "per_depth", "max_request_tokens": 64000, "max_state_plus_question_tokens": 32000, "token_safety_margin": 0.2, "label_style": "opaque", "include_summaries": true, "show_types": true, "stop_style": "literal", "relation_glosses": {}, "answer_type": "hint", "answer_type_gate_min_p": 0.6, "prefilter_threshold": 50, "prefilter_top_n": 30, "max_frontier": 2000, "frontier_preview": 5}}`
- **graphwalk-reader-source-schema** (wall 400s): `{"system": "graphwalk-reader", "decision_model": "typesafe/jev-1.13", "llm": "openrouter/openai/gpt-6-luna", "embedder": "fastembed:BAAI/bge-small-en-v1.5", "max_entries": 2, "max_nodes": 20, "max_facts": 12, "context": "source paragraphs", "max_docs": 5, "traversal": {"strategy": "greedy", "beam_width": 3, "hop_mode": "entity", "direction": "both", "exclude_visited": true, "allow_stop_at_start": false, "budget": {"max_depth": 4, "max_decision_calls": 12, "max_input_tokens": null}, "length_alpha": 1.0, "prob_floor": 0.001, "temperature": 1.0, "top_p": 1.0, "n_samples": 5, "seed": 0, "beam_batching": "per_depth", "max_request_tokens": 64000, "max_state_plus_question_tokens": 32000, "token_safety_margin": 0.2, "label_style": "opaque", "include_summaries": true, "show_types": true, "stop_style": "literal", "relation_glosses": {}, "answer_type": "hint", "answer_type_gate_min_p": 0.6, "prefilter_threshold": 50, "prefilter_top_n": 30, "max_frontier": 2000, "frontier_preview": 5}}`

## By question type

| system | bridge_comparison F1 (EM) | comparison F1 (EM) | compositional F1 (EM) | inference F1 (EM) |
|---|---|---|---|---|
| graphwalk-schema | 0.000 (0.000) n=30 | 0.029 (0.000) n=30 | 0.353 (0.233) n=30 | 0.622 (0.433) n=30 |
| graphwalk-reader-source | 0.733 (0.733) n=30 | 0.700 (0.700) n=30 | 0.328 (0.267) n=30 | 0.838 (0.667) n=30 |
| graphwalk-reader-source-schema | 0.752 (0.733) n=30 | 0.700 (0.700) n=30 | 0.350 (0.267) n=30 | 0.846 (0.700) n=30 |
