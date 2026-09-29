# fanoutqa-text

- params: `{"corpus": "JinChao1022/fanoutqa-retrieval@03addfe298d5886dd87fa8e0592974b86538f2b2", "dev_sha256": "b62a9797732c716e6b17ba4086f277d154d747ce2fa01614cb76a3372e7fb88c", "embedder": "BAAI/bge-small-en-v1.5", "escalation_llm": "openrouter/xiaomi/mimo-v2.6-pro", "graph": "fanoutqa-3d0a3eecd6e98c42.graph.json", "llm": "openrouter/openai/gpt-6-luna", "max_chars": 8000, "max_docs": 20, "n": 2, "preset": "relation-v2", "rag_k": 10, "seed": 5}`
- git: `eae80fb194eb2d109c66d653a23eba42ae4cbec0`
- run at: 2026-09-29T20:04:15.030915+00:00

| system | n | hits@1 | EM | F1 | p50 s | p95 s | $/query | $/1k q | Jev calls/q | LLM calls/q | in tok/q | status | linking |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| graphwalk-reader-facts | 2 | 0.000 | 0.000 | 0.321 | 6.08 | 7.16 | 0.000746 | 0.746 | 5.00 | 1.00 | 9641 | ok=2 | n/a |
| graphwalk-reader-source | 2 | 0.000 | 0.000 | 0.321 | 5.15 | 7.02 | 0.001504 | 1.504 | 5.00 | 1.00 | 16100 | ok=2 | n/a |
| text-rag | 2 | 0.000 | 0.000 | 0.272 | 3.02 | 8.06 | 0.002759 | 2.759 | 0.00 | 1.00 | 20501 | ok=2 | n/a |
| text-iter-rag | 2 | 0.000 | 0.000 | 0.236 | 3.41 | 11.39 | 0.004121 | 4.121 | 0.00 | 1.50 | 30795 | ok=2 | n/a |

## Systems

- **graphwalk-reader-facts** (wall 9s): `{"system": "graphwalk-reader", "decision_model": "typesafe/jev-1.13", "llm": "openrouter/openai/gpt-6-luna", "embedder": "fastembed:BAAI/bge-small-en-v1.5", "max_entries": 2, "max_nodes": 40, "max_facts": 12, "context": "facts", "max_docs": 5, "traversal": {"strategy": "greedy", "beam_width": 3, "hop_mode": "relation", "direction": "both", "exclude_visited": true, "allow_stop_at_start": false, "budget": {"max_depth": 4, "max_decision_calls": 12, "max_input_tokens": null}, "length_alpha": 1.0, "prob_floor": 0.001, "temperature": 1.0, "top_p": 1.0, "n_samples": 5, "seed": 0, "beam_batching": "per_depth", "max_request_tokens": 64000, "max_state_plus_question_tokens": 32000, "token_safety_margin": 0.2, "label_style": "opaque", "include_summaries": true, "show_types": true, "stop_style": "literal", "relation_glosses": {}, "answer_type": "hint", "answer_type_gate_min_p": 0.6, "prefilter_threshold": 50, "prefilter_top_n": 30, "max_frontier": 2000, "frontier_preview": 5}}`
- **graphwalk-reader-source** (wall 8s): `{"system": "graphwalk-reader", "decision_model": "typesafe/jev-1.13", "llm": "openrouter/openai/gpt-6-luna", "embedder": "fastembed:BAAI/bge-small-en-v1.5", "max_entries": 2, "max_nodes": 40, "max_facts": 12, "context": "source paragraphs", "max_docs": 20, "traversal": {"strategy": "greedy", "beam_width": 3, "hop_mode": "relation", "direction": "both", "exclude_visited": true, "allow_stop_at_start": false, "budget": {"max_depth": 4, "max_decision_calls": 12, "max_input_tokens": null}, "length_alpha": 1.0, "prob_floor": 0.001, "temperature": 1.0, "top_p": 1.0, "n_samples": 5, "seed": 0, "beam_batching": "per_depth", "max_request_tokens": 64000, "max_state_plus_question_tokens": 32000, "token_safety_margin": 0.2, "label_style": "opaque", "include_summaries": true, "show_types": true, "stop_style": "literal", "relation_glosses": {}, "answer_type": "hint", "answer_type_gate_min_p": 0.6, "prefilter_threshold": 50, "prefilter_top_n": 30, "max_frontier": 2000, "frontier_preview": 5}}`
- **text-rag** (wall 11s): `{"system": "vector-rag", "k": 10, "llm": "openrouter/openai/gpt-6-luna", "embedder": "fastembed:BAAI/bge-small-en-v1.5", "documents": 11, "document": "one per page: title and text"}`
- **text-iter-rag** (wall 15s): `{"system": "iter-rag", "k": 10, "k_search": 2, "max_steps": 5, "max_searches": 8, "max_docs": 20, "llm": "openrouter/openai/gpt-6-luna", "embedder": "fastembed:BAAI/bge-small-en-v1.5", "documents": 11, "document": "one per page: title and text"}`

## By question type

| system | fanout F1 (EM) |
|---|---|
| graphwalk-reader-facts | 0.321 (0.000) n=2 |
| graphwalk-reader-source | 0.321 (0.000) n=2 |
| text-rag | 0.272 (0.000) n=2 |
| text-iter-rag | 0.236 (0.000) n=2 |

## Ingestion

11 paragraphs, 56 chunks, 737 nodes created, 350 merged, 1050 edges; 67 LLM + 76 decision calls (0 cached extractions), $0.0908, 218s (waiting on extraction 116s, routing 87s, writing 15s; 8 escalations).

## FanOutQA accuracy

Ceiling (reference strings present in the truncated evidence): 0.650 loose.

| system | loose acc | strict acc | $/1k q | p50 s | LLM calls/q | Jev calls/q |
|---|---|---|---|---|---|---|
| graphwalk-reader-facts | 0.450 | 0.000 | 0.746 | 6.08 | 1.00 | 5.00 |
| graphwalk-reader-source | 0.450 | 0.000 | 1.504 | 5.15 | 1.00 | 5.00 |
| text-rag | 0.650 | 0.000 | 2.759 | 3.02 | 1.00 | 0.00 |
| text-iter-rag | 0.600 | 0.000 | 4.121 | 3.41 | 1.50 | 0.00 |
