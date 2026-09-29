# 2wiki-text

- params: `{"dataset": "2wiki", "embedder": "BAAI/bge-small-en-v1.5", "escalation_llm": "openrouter/xiaomi/mimo-v2.6-pro", "graph": "2wiki-330b8be64c4322b1.graph.json", "ingest": {"concurrency": 8, "escalate": true, "max_candidates": 5, "max_chunk_chars": 2000, "max_relations": 8, "min_similarity": 0.75, "prune_missing": false, "route_concurrency": 4, "route_threshold": 0.6, "routing": "jev"}, "llm": "openrouter/openai/gpt-6-luna", "n": 120, "per_type": 30, "preset": "greedy-v2", "rag_k": 5, "seed": 0}`
- git: `6d29770e0b05afd5d02f61fabdec814c60503eef`
- run at: 2026-09-29T16:11:38.051712+00:00

| system | n | hits@1 | EM | F1 | p50 s | p95 s | $/query | $/1k q | Jev calls/q | LLM calls/q | in tok/q | status | linking |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| graphwalk | 120 | 0.167 | 0.167 | 0.254 | 0.87 | 1.44 | 0.000184 | 0.184 | 3.50 | 0.00 | 4388 | ok=120 | n/a |
| graphwalk-reader | 120 | 0.492 | 0.492 | 0.539 | 2.73 | 4.56 | 0.000333 | 0.333 | 4.74 | 1.00 | 6540 | ok=120 | n/a |
| text-rag | 120 | 0.325 | 0.325 | 0.368 | 2.01 | 4.37 | 0.000095 | 0.095 | 0.00 | 1.00 | 737 | ok=120 | n/a |
| text-iter-rag | 120 | 0.683 | 0.683 | 0.749 | 3.75 | 10.17 | n/a | n/a | 0.00 | 1.76 | 1651 | error=2, ok=118 | n/a |

## Systems

- **graphwalk** (wall 14s): `{"system": "graphwalk", "linking": "choice", "decision_model": "typesafe/jev-1.13", "embedder": "fastembed:BAAI/bge-small-en-v1.5", "traversal": {"strategy": "greedy", "beam_width": 3, "hop_mode": "entity", "direction": "both", "exclude_visited": true, "allow_stop_at_start": false, "budget": {"max_depth": 4, "max_decision_calls": 12, "max_input_tokens": null}, "length_alpha": 1.0, "prob_floor": 0.001, "temperature": 1.0, "top_p": 1.0, "n_samples": 5, "seed": 0, "beam_batching": "per_depth", "max_request_tokens": 64000, "max_state_plus_question_tokens": 32000, "token_safety_margin": 0.2, "label_style": "opaque", "include_summaries": true, "show_types": true, "stop_style": "literal", "relation_glosses": {}, "answer_type": "hint", "answer_type_gate_min_p": 0.6, "prefilter_threshold": 50, "prefilter_top_n": 30, "max_frontier": 2000, "frontier_preview": 5}}`
- **graphwalk-reader** (wall 334s): `{"system": "graphwalk-reader", "decision_model": "typesafe/jev-1.13", "llm": "openrouter/openai/gpt-6-luna", "embedder": "fastembed:BAAI/bge-small-en-v1.5", "max_entries": 2, "max_nodes": 20, "max_facts": 12, "traversal": {"strategy": "greedy", "beam_width": 3, "hop_mode": "entity", "direction": "both", "exclude_visited": true, "allow_stop_at_start": false, "budget": {"max_depth": 4, "max_decision_calls": 12, "max_input_tokens": null}, "length_alpha": 1.0, "prob_floor": 0.001, "temperature": 1.0, "top_p": 1.0, "n_samples": 5, "seed": 0, "beam_batching": "per_depth", "max_request_tokens": 64000, "max_state_plus_question_tokens": 32000, "token_safety_margin": 0.2, "label_style": "opaque", "include_summaries": true, "show_types": true, "stop_style": "literal", "relation_glosses": {}, "answer_type": "hint", "answer_type_gate_min_p": 0.6, "prefilter_threshold": 50, "prefilter_top_n": 30, "max_frontier": 2000, "frontier_preview": 5}}`
- **text-rag** (wall 359s): `{"system": "vector-rag", "k": 5, "llm": "openrouter/openai/gpt-6-luna", "embedder": "fastembed:BAAI/bge-small-en-v1.5", "documents": 970, "document": "one per paragraph: title and text"}`
- **text-iter-rag** (wall 606s): `{"system": "iter-rag", "k": 5, "k_search": 2, "max_steps": 5, "max_searches": 8, "max_docs": 40, "llm": "openrouter/openai/gpt-6-luna", "embedder": "fastembed:BAAI/bge-small-en-v1.5", "documents": 970, "document": "one per paragraph: title and text"}`

## By question type

| system | bridge_comparison F1 (EM) | comparison F1 (EM) | compositional F1 (EM) | inference F1 (EM) |
|---|---|---|---|---|
| graphwalk | 0.000 (0.000) n=30 | 0.029 (0.000) n=30 | 0.320 (0.200) n=30 | 0.668 (0.467) n=30 |
| graphwalk-reader | 0.533 (0.533) n=30 | 0.667 (0.667) n=30 | 0.211 (0.167) n=30 | 0.746 (0.600) n=30 |
| text-rag | 0.033 (0.033) n=30 | 0.867 (0.867) n=30 | 0.167 (0.167) n=30 | 0.404 (0.233) n=30 |
| text-iter-rag | 0.967 (0.967) n=30 | 0.900 (0.900) n=30 | 0.299 (0.233) n=30 | 0.832 (0.633) n=30 |
