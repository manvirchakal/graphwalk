# 2wiki-text

- params: `{"dataset": "2wiki", "embedder": "BAAI/bge-small-en-v1.5", "escalation_llm": "openrouter/xiaomi/mimo-v2.6-pro", "graph": "2wiki-3b5e6b05372aab3b.graph.json", "ingest": {"concurrency": 4, "escalate": true, "max_candidates": 5, "max_chunk_chars": 2000, "max_relations": 8, "min_similarity": 0.75, "prune_missing": false, "route_threshold": 0.6, "routing": "jev"}, "llm": "openrouter/openai/gpt-6-luna", "n": 8, "per_type": 2, "preset": "greedy-v2", "rag_k": 5, "seed": 7}`
- git: `4c872a9ab3e18d609a0c21903babce7886b5ec34` (dirty)
- run at: 2026-09-29T12:16:15.564824+00:00

| system | n | hits@1 | EM | F1 | p50 s | p95 s | $/query | $/1k q | Jev calls/q | LLM calls/q | in tok/q | status | linking |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| graphwalk | 8 | 0.375 | 0.375 | 0.375 | 1.08 | 2.21 | 0.000193 | 0.193 | 3.50 | 0.00 | 4598 | ok=8 | n/a |
| graphwalk-reader | 8 | 0.750 | 0.750 | 0.750 | 2.41 | 4.65 | 0.000334 | 0.334 | 4.38 | 1.00 | 6445 | ok=8 | n/a |
| text-rag | 8 | 0.500 | 0.500 | 0.583 | 1.79 | 2.61 | 0.000088 | 0.088 | 0.00 | 1.00 | 594 | ok=8 | n/a |
| text-iter-rag | 8 | 0.625 | 0.625 | 0.708 | 2.62 | 5.61 | 0.000161 | 0.161 | 0.00 | 1.50 | 995 | ok=8 | n/a |

## Systems

- **graphwalk** (wall 2s): `{"system": "graphwalk", "linking": "choice", "decision_model": "typesafe/jev-1.13", "embedder": "fastembed:BAAI/bge-small-en-v1.5", "traversal": {"strategy": "greedy", "beam_width": 3, "hop_mode": "entity", "direction": "both", "exclude_visited": true, "allow_stop_at_start": false, "budget": {"max_depth": 4, "max_decision_calls": 12, "max_input_tokens": null}, "length_alpha": 1.0, "prob_floor": 0.001, "temperature": 1.0, "top_p": 1.0, "n_samples": 5, "seed": 0, "beam_batching": "per_depth", "max_request_tokens": 64000, "max_state_plus_question_tokens": 32000, "token_safety_margin": 0.2, "label_style": "opaque", "include_summaries": true, "show_types": true, "stop_style": "literal", "relation_glosses": {}, "answer_type": "hint", "answer_type_gate_min_p": 0.6, "prefilter_threshold": 50, "prefilter_top_n": 30, "max_frontier": 2000, "frontier_preview": 5}}`
- **graphwalk-reader** (wall 10s): `{"system": "graphwalk-reader", "decision_model": "typesafe/jev-1.13", "llm": "openrouter/openai/gpt-6-luna", "embedder": "fastembed:BAAI/bge-small-en-v1.5", "max_entries": 2, "max_nodes": 20, "max_facts": 12, "traversal": {"strategy": "greedy", "beam_width": 3, "hop_mode": "entity", "direction": "both", "exclude_visited": true, "allow_stop_at_start": false, "budget": {"max_depth": 4, "max_decision_calls": 12, "max_input_tokens": null}, "length_alpha": 1.0, "prob_floor": 0.001, "temperature": 1.0, "top_p": 1.0, "n_samples": 5, "seed": 0, "beam_batching": "per_depth", "max_request_tokens": 64000, "max_state_plus_question_tokens": 32000, "token_safety_margin": 0.2, "label_style": "opaque", "include_summaries": true, "show_types": true, "stop_style": "literal", "relation_glosses": {}, "answer_type": "hint", "answer_type_gate_min_p": 0.6, "prefilter_threshold": 50, "prefilter_top_n": 30, "max_frontier": 2000, "frontier_preview": 5}}`
- **text-rag** (wall 9s): `{"system": "vector-rag", "k": 5, "llm": "openrouter/openai/gpt-6-luna", "embedder": "fastembed:BAAI/bge-small-en-v1.5", "documents": 72, "document": "one per paragraph: title and text"}`
- **text-iter-rag** (wall 13s): `{"system": "iter-rag", "k": 5, "k_search": 2, "max_steps": 5, "max_searches": 8, "max_docs": 40, "llm": "openrouter/openai/gpt-6-luna", "embedder": "fastembed:BAAI/bge-small-en-v1.5", "documents": 72, "document": "one per paragraph: title and text"}`

## By question type

| system | bridge_comparison F1 (EM) | comparison F1 (EM) | compositional F1 (EM) | inference F1 (EM) |
|---|---|---|---|---|
| graphwalk | 0.000 (0.000) n=2 | 0.000 (0.000) n=2 | 1.000 (1.000) n=2 | 0.500 (0.500) n=2 |
| graphwalk-reader | 0.500 (0.500) n=2 | 1.000 (1.000) n=2 | 0.500 (0.500) n=2 | 1.000 (1.000) n=2 |
| text-rag | 0.000 (0.000) n=2 | 1.000 (1.000) n=2 | 0.500 (0.500) n=2 | 0.833 (0.500) n=2 |
| text-iter-rag | 1.000 (1.000) n=2 | 1.000 (1.000) n=2 | 0.500 (0.500) n=2 | 0.333 (0.000) n=2 |

## Ingestion

72 paragraphs, 72 chunks, 331 nodes created, 36 merged, 315 edges; 58 LLM + 49 decision calls (17 cached extractions), $0.0261, 207s.
