# hotpotqa-text

- params: `{"dataset": "hotpotqa", "embedder": "BAAI/bge-small-en-v1.5", "escalation_llm": "openrouter/xiaomi/mimo-v2.6-pro", "graph": "hotpotqa-81b56da2fae89b0c.graph.json", "ingest": {"concurrency": 8, "escalate": true, "max_candidates": 5, "max_chunk_chars": 2000, "max_relations": 8, "min_similarity": 0.75, "prune_missing": false, "route_concurrency": 4, "route_threshold": 0.6, "routing": "jev"}, "llm": "openrouter/openai/gpt-6-luna", "n": 120, "per_type": 60, "preset": "greedy-v2", "rag_k": 5, "seed": 0}`
- git: `1d2925fe602c38f6599f64178023baf44843d85e`
- run at: 2026-09-29T17:41:29.159859+00:00

| system | n | hits@1 | EM | F1 | p50 s | p95 s | $/query | $/1k q | Jev calls/q | LLM calls/q | in tok/q | status | linking |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| graphwalk | 120 | 0.075 | 0.075 | 0.182 | 0.73 | 1.45 | 0.000174 | 0.174 | 3.38 | 0.00 | 4149 | ok=120 | n/a |
| graphwalk-reader | 120 | 0.458 | 0.458 | 0.550 | 3.10 | 5.45 | 0.000394 | 0.394 | 5.65 | 1.00 | 7576 | ok=120 | n/a |
| text-rag | 120 | 0.675 | 0.675 | 0.755 | 2.21 | 3.33 | 0.000092 | 0.092 | 0.00 | 1.00 | 659 | ok=120 | n/a |
| text-iter-rag | 120 | 0.700 | 0.700 | 0.817 | 2.25 | 5.31 | 0.000141 | 0.141 | 0.00 | 1.19 | 887 | ok=120 | n/a |

## Systems

- **graphwalk** (wall 14s): `{"system": "graphwalk", "linking": "choice", "decision_model": "typesafe/jev-1.13", "embedder": "fastembed:BAAI/bge-small-en-v1.5", "traversal": {"strategy": "greedy", "beam_width": 3, "hop_mode": "entity", "direction": "both", "exclude_visited": true, "allow_stop_at_start": false, "budget": {"max_depth": 4, "max_decision_calls": 12, "max_input_tokens": null}, "length_alpha": 1.0, "prob_floor": 0.001, "temperature": 1.0, "top_p": 1.0, "n_samples": 5, "seed": 0, "beam_batching": "per_depth", "max_request_tokens": 64000, "max_state_plus_question_tokens": 32000, "token_safety_margin": 0.2, "label_style": "opaque", "include_summaries": true, "show_types": true, "stop_style": "literal", "relation_glosses": {}, "answer_type": "hint", "answer_type_gate_min_p": 0.6, "prefilter_threshold": 50, "prefilter_top_n": 30, "max_frontier": 2000, "frontier_preview": 5}}`
- **graphwalk-reader** (wall 400s): `{"system": "graphwalk-reader", "decision_model": "typesafe/jev-1.13", "llm": "openrouter/openai/gpt-6-luna", "embedder": "fastembed:BAAI/bge-small-en-v1.5", "max_entries": 2, "max_nodes": 20, "max_facts": 12, "traversal": {"strategy": "greedy", "beam_width": 3, "hop_mode": "entity", "direction": "both", "exclude_visited": true, "allow_stop_at_start": false, "budget": {"max_depth": 4, "max_decision_calls": 12, "max_input_tokens": null}, "length_alpha": 1.0, "prob_floor": 0.001, "temperature": 1.0, "top_p": 1.0, "n_samples": 5, "seed": 0, "beam_batching": "per_depth", "max_request_tokens": 64000, "max_state_plus_question_tokens": 32000, "token_safety_margin": 0.2, "label_style": "opaque", "include_summaries": true, "show_types": true, "stop_style": "literal", "relation_glosses": {}, "answer_type": "hint", "answer_type_gate_min_p": 0.6, "prefilter_threshold": 50, "prefilter_top_n": 30, "max_frontier": 2000, "frontier_preview": 5}}`
- **text-rag** (wall 400s): `{"system": "vector-rag", "k": 5, "llm": "openrouter/openai/gpt-6-luna", "embedder": "fastembed:BAAI/bge-small-en-v1.5", "documents": 1187, "document": "one per paragraph: title and text"}`
- **text-iter-rag** (wall 481s): `{"system": "iter-rag", "k": 5, "k_search": 2, "max_steps": 5, "max_searches": 8, "max_docs": 40, "llm": "openrouter/openai/gpt-6-luna", "embedder": "fastembed:BAAI/bge-small-en-v1.5", "documents": 1187, "document": "one per paragraph: title and text"}`

## By question type

| system | bridge F1 (EM) | comparison F1 (EM) |
|---|---|---|
| graphwalk | 0.334 (0.150) n=60 | 0.030 (0.000) n=60 |
| graphwalk-reader | 0.539 (0.367) n=60 | 0.561 (0.550) n=60 |
| text-rag | 0.652 (0.533) n=60 | 0.857 (0.817) n=60 |
| text-iter-rag | 0.784 (0.600) n=60 | 0.849 (0.800) n=60 |

## Ingestion

1187 paragraphs, 1098 chunks, 6640 nodes created, 2335 merged, 8414 edges; 1176 LLM + 1216 decision calls (25 cached extractions), $0.7813, 3601s.
