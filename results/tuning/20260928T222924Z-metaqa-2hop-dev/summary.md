# metaqa-2hop-dev

- params: `{"concurrency": 4, "dataset": "metaqa", "hops": 2, "linking": "given", "n": 200, "rag_k": 5, "seed": 0, "split": "dev", "total_questions": 14872, "variants": {"-soft": {"answer_type": "hint", "relation_glosses": "dataset", "show_types": true, "stop_style": "literal"}}}`
- git: `da2d7e7b2767b60ea7ffb6f5fbc33d67d458ee60` (dirty)
- run at: 2026-09-28T22:29:24.678411+00:00

| system | n | hits@1 | EM | F1 | p50 s | p95 s | $/query | $/1k q | Jev calls/q | LLM calls/q | in tok/q | status | linking |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| graphwalk-relation-beam-soft | 200 | 0.980 | 0.980 | 0.980 | 1.56 | 3.31 | 0.000201 | 0.201 | 3.90 | 0.00 | 4776 | ok=200 | 1.000 |

## Systems

- **graphwalk-relation-beam-soft** (wall 89s): `{"system": "graphwalk", "linking": "given", "decision_model": "typesafe/jev-1.13", "embedder": "fastembed:BAAI/bge-small-en-v1.5", "traversal": {"strategy": "beam", "beam_width": 3, "hop_mode": "relation", "direction": "both", "exclude_visited": true, "allow_stop_at_start": false, "budget": {"max_depth": 4, "max_decision_calls": 12, "max_input_tokens": null}, "length_alpha": 1.0, "prob_floor": 0.001, "temperature": 1.0, "top_p": 1.0, "n_samples": 5, "seed": 0, "beam_batching": "per_depth", "max_request_tokens": 64000, "max_state_plus_question_tokens": 32000, "token_safety_margin": 0.2, "label_style": "opaque", "include_summaries": true, "show_types": true, "stop_style": "literal", "relation_glosses": {"directed_by": "the director of the film", "written_by": "a writer of the film", "starred_actors": "an actor who starred in the film", "release_year": "the year the film was released", "in_language": "the language of the film", "has_genre": "the genre of the film", "has_tags": "keywords describing the film (topics, people, themes)", "has_imdb_rating": "the film's IMDb rating", "has_imdb_votes": "how many IMDb votes the film has"}, "answer_type": "hint", "answer_type_gate_min_p": 0.6, "prefilter_threshold": 50, "prefilter_top_n": 30, "max_frontier": 2000, "frontier_preview": 5}}`
