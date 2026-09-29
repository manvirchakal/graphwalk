# hotpotqa-text

- params: `{"dataset": "hotpotqa", "embedder": "BAAI/bge-small-en-v1.5", "graph": "hotpotqa-81b56da2fae89b0c.graph.json", "llm": "openrouter/openai/gpt-6-luna", "max_docs": 5, "n": 120, "per_type": 60, "preset": "greedy-v2", "seed": 0}`
- git: `53e516efd2f84cb2bce7f509757902498d25bd38` (dirty)
- run at: 2026-09-29T18:37:47.238359+00:00

| system | n | hits@1 | EM | F1 | p50 s | p95 s | $/query | $/1k q | Jev calls/q | LLM calls/q | in tok/q | status | linking |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| graphwalk-schema | 120 | 0.075 | 0.075 | 0.186 | 0.74 | 1.32 | 0.000171 | 0.171 | 3.35 | 0.00 | 4067 | ok=120 | n/a |
| graphwalk-reader-source | 120 | 0.567 | 0.567 | 0.668 | 3.09 | 4.44 | 0.000374 | 0.374 | 5.62 | 1.00 | 7425 | ok=120 | n/a |
| graphwalk-reader-source-schema | 120 | 0.558 | 0.558 | 0.665 | 3.09 | 4.79 | 0.000382 | 0.382 | 5.72 | 1.00 | 7549 | ok=120 | n/a |

## Systems

- **graphwalk-schema** (wall 13s): `{"system": "graphwalk", "linking": "choice", "decision_model": "typesafe/jev-1.13", "embedder": "fastembed:BAAI/bge-small-en-v1.5", "traversal": {"strategy": "greedy", "beam_width": 3, "hop_mode": "entity", "direction": "both", "exclude_visited": true, "allow_stop_at_start": false, "budget": {"max_depth": 4, "max_decision_calls": 12, "max_input_tokens": null}, "length_alpha": 1.0, "prob_floor": 0.001, "temperature": 1.0, "top_p": 1.0, "n_samples": 5, "seed": 0, "beam_batching": "per_depth", "max_request_tokens": 64000, "max_state_plus_question_tokens": 32000, "token_safety_margin": 0.2, "label_style": "opaque", "include_summaries": true, "show_types": true, "stop_style": "literal", "relation_glosses": {}, "answer_type": "hint", "answer_type_gate_min_p": 0.6, "prefilter_threshold": 50, "prefilter_top_n": 30, "max_frontier": 2000, "frontier_preview": 5}}`
- **graphwalk-reader-source** (wall 399s): `{"system": "graphwalk-reader", "decision_model": "typesafe/jev-1.13", "llm": "openrouter/openai/gpt-6-luna", "embedder": "fastembed:BAAI/bge-small-en-v1.5", "max_entries": 2, "max_nodes": 20, "max_facts": 12, "context": "source paragraphs", "max_docs": 5, "traversal": {"strategy": "greedy", "beam_width": 3, "hop_mode": "entity", "direction": "both", "exclude_visited": true, "allow_stop_at_start": false, "budget": {"max_depth": 4, "max_decision_calls": 12, "max_input_tokens": null}, "length_alpha": 1.0, "prob_floor": 0.001, "temperature": 1.0, "top_p": 1.0, "n_samples": 5, "seed": 0, "beam_batching": "per_depth", "max_request_tokens": 64000, "max_state_plus_question_tokens": 32000, "token_safety_margin": 0.2, "label_style": "opaque", "include_summaries": true, "show_types": true, "stop_style": "literal", "relation_glosses": {}, "answer_type": "hint", "answer_type_gate_min_p": 0.6, "prefilter_threshold": 50, "prefilter_top_n": 30, "max_frontier": 2000, "frontier_preview": 5}}`
- **graphwalk-reader-source-schema** (wall 400s): `{"system": "graphwalk-reader", "decision_model": "typesafe/jev-1.13", "llm": "openrouter/openai/gpt-6-luna", "embedder": "fastembed:BAAI/bge-small-en-v1.5", "max_entries": 2, "max_nodes": 20, "max_facts": 12, "context": "source paragraphs", "max_docs": 5, "traversal": {"strategy": "greedy", "beam_width": 3, "hop_mode": "entity", "direction": "both", "exclude_visited": true, "allow_stop_at_start": false, "budget": {"max_depth": 4, "max_decision_calls": 12, "max_input_tokens": null}, "length_alpha": 1.0, "prob_floor": 0.001, "temperature": 1.0, "top_p": 1.0, "n_samples": 5, "seed": 0, "beam_batching": "per_depth", "max_request_tokens": 64000, "max_state_plus_question_tokens": 32000, "token_safety_margin": 0.2, "label_style": "opaque", "include_summaries": true, "show_types": true, "stop_style": "literal", "relation_glosses": {}, "answer_type": "hint", "answer_type_gate_min_p": 0.6, "prefilter_threshold": 50, "prefilter_top_n": 30, "max_frontier": 2000, "frontier_preview": 5}}`

## By question type

| system | bridge F1 (EM) | comparison F1 (EM) |
|---|---|---|
| graphwalk-schema | 0.334 (0.150) n=60 | 0.038 (0.000) n=60 |
| graphwalk-reader-source | 0.667 (0.500) n=60 | 0.668 (0.633) n=60 |
| graphwalk-reader-source-schema | 0.664 (0.483) n=60 | 0.667 (0.633) n=60 |
