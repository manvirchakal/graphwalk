# fanoutqa-text

- params: `{"corpus": "JinChao1022/fanoutqa-retrieval@03addfe298d5886dd87fa8e0592974b86538f2b2", "dev_sha256": "b62a9797732c716e6b17ba4086f277d154d747ce2fa01614cb76a3372e7fb88c", "embedder": "BAAI/bge-small-en-v1.5", "escalation_llm": "openrouter/xiaomi/mimo-v2.6-pro", "graph": "fanoutqa-745e982de686504d.graph.json", "llm": "openrouter/openai/gpt-6-luna", "max_chars": 8000, "max_docs": 20, "n": 40, "preset": "relation-v2", "rag_k": 10, "seed": 0}`
- git: `eae80fb194eb2d109c66d653a23eba42ae4cbec0` (dirty)
- run at: 2026-09-29T21:31:14.715432+00:00

| system | n | hits@1 | EM | F1 | p50 s | p95 s | $/query | $/1k q | Jev calls/q | LLM calls/q | in tok/q | status | linking |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| graphwalk-reader-facts | 40 | 0.000 | 0.000 | 0.182 | 6.04 | 10.14 | 0.001087 | 1.087 | 5.50 | 1.00 | 14210 | ok=40 | n/a |
| graphwalk-reader-source | 40 | 0.025 | 0.025 | 0.238 | 6.38 | 14.83 | 0.002640 | 2.640 | 5.45 | 1.00 | 26292 | ok=40 | n/a |
| text-rag | 40 | 0.025 | 0.025 | 0.257 | 4.68 | 12.28 | 0.002951 | 2.951 | 0.00 | 1.00 | 21995 | ok=40 | n/a |
| text-iter-rag | 40 | 0.000 | 0.000 | 0.229 | 6.52 | 20.11 | 0.006041 | 6.041 | 0.00 | 1.82 | 46073 | ok=40 | n/a |

## Systems

- **graphwalk-reader-facts** (wall 135s): `{"system": "graphwalk-reader", "decision_model": "typesafe/jev-1.13", "llm": "openrouter/openai/gpt-6-luna", "embedder": "fastembed:BAAI/bge-small-en-v1.5", "max_entries": 2, "max_nodes": 40, "max_facts": 12, "context": "facts", "max_docs": 5, "traversal": {"strategy": "greedy", "beam_width": 3, "hop_mode": "relation", "direction": "both", "exclude_visited": true, "allow_stop_at_start": false, "budget": {"max_depth": 4, "max_decision_calls": 12, "max_input_tokens": null}, "length_alpha": 1.0, "prob_floor": 0.001, "temperature": 1.0, "top_p": 1.0, "n_samples": 5, "seed": 0, "beam_batching": "per_depth", "max_request_tokens": 64000, "max_state_plus_question_tokens": 32000, "token_safety_margin": 0.2, "label_style": "opaque", "include_summaries": true, "show_types": true, "stop_style": "literal", "relation_glosses": {}, "answer_type": "hint", "answer_type_gate_min_p": 0.6, "prefilter_threshold": 50, "prefilter_top_n": 30, "max_frontier": 2000, "frontier_preview": 5}}`
- **graphwalk-reader-source** (wall 138s): `{"system": "graphwalk-reader", "decision_model": "typesafe/jev-1.13", "llm": "openrouter/openai/gpt-6-luna", "embedder": "fastembed:BAAI/bge-small-en-v1.5", "max_entries": 2, "max_nodes": 40, "max_facts": 12, "context": "source paragraphs", "max_docs": 20, "traversal": {"strategy": "greedy", "beam_width": 3, "hop_mode": "relation", "direction": "both", "exclude_visited": true, "allow_stop_at_start": false, "budget": {"max_depth": 4, "max_decision_calls": 12, "max_input_tokens": null}, "length_alpha": 1.0, "prob_floor": 0.001, "temperature": 1.0, "top_p": 1.0, "n_samples": 5, "seed": 0, "beam_batching": "per_depth", "max_request_tokens": 64000, "max_state_plus_question_tokens": 32000, "token_safety_margin": 0.2, "label_style": "opaque", "include_summaries": true, "show_types": true, "stop_style": "literal", "relation_glosses": {}, "answer_type": "hint", "answer_type_gate_min_p": 0.6, "prefilter_threshold": 50, "prefilter_top_n": 30, "max_frontier": 2000, "frontier_preview": 5}}`
- **text-rag** (wall 136s): `{"system": "vector-rag", "k": 10, "llm": "openrouter/openai/gpt-6-luna", "embedder": "fastembed:BAAI/bge-small-en-v1.5", "documents": 233, "document": "one per page: title and text"}`
- **text-iter-rag** (wall 268s): `{"system": "iter-rag", "k": 10, "k_search": 2, "max_steps": 5, "max_searches": 8, "max_docs": 20, "llm": "openrouter/openai/gpt-6-luna", "embedder": "fastembed:BAAI/bge-small-en-v1.5", "documents": 233, "document": "one per page: title and text"}`

## By question type

| system | fanout F1 (EM) |
|---|---|
| graphwalk-reader-facts | 0.182 (0.000) n=40 |
| graphwalk-reader-source | 0.238 (0.025) n=40 |
| text-rag | 0.257 (0.025) n=40 |
| text-iter-rag | 0.229 (0.000) n=40 |

## Ingestion

233 paragraphs, 424 chunks, 4092 nodes created, 3557 merged, 6586 edges; 412 LLM + 552 decision calls (135 cached extractions), $0.5963, 1196s (waiting on extraction 0s, routing 1062s, writing 129s; 120 escalations).

## FanOutQA accuracy

Ceiling (reference strings present in the truncated evidence): 0.894 loose.

| system | loose acc | strict acc | $/1k q | p50 s | LLM calls/q | Jev calls/q |
|---|---|---|---|---|---|---|
| graphwalk-reader-facts | 0.364 | 0.050 | 1.087 | 6.04 | 1.00 | 5.50 |
| graphwalk-reader-source | 0.482 | 0.050 | 2.640 | 6.38 | 1.00 | 5.45 |
| text-rag | 0.589 | 0.200 | 2.951 | 4.68 | 1.00 | 0.00 |
| text-iter-rag | 0.670 | 0.300 | 6.041 | 6.52 | 1.82 | 0.00 |
