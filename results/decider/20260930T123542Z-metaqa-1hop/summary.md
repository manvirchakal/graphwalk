# metaqa-1hop

- params: `{"concurrency": 4, "dataset": "metaqa", "hops": 1, "linking": "given", "n": 200, "rag_k": 5, "seed": 0, "split": "test", "total_questions": 9947, "variants": {"-jev": {}}}`
- git: `ec8490f74e39dc596aa0b5b6100b5c7aa78ad52c` (dirty)
- run at: 2026-09-30T12:35:42.226201+00:00

| system | n | hits@1 | EM | F1 | p50 s | p95 s | $/query | $/1k q | Jev calls/q | LLM calls/q | in tok/q | status | linking |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| graphwalk-relation-v2-jev | 200 | 0.940 | 0.930 | 0.965 | 0.56 | 1.51 | 0.000064 | 0.064 | 1.97 | 0.00 | 1517 | ok=200 | 1.000 |

## Systems

- **graphwalk-relation-v2-jev** (wall 35s): `{"system": "graphwalk", "linking": "given", "decision_model": "typesafe/jev-1.13", "embedder": "fastembed:BAAI/bge-small-en-v1.5", "traversal": {"strategy": "greedy", "beam_width": 3, "hop_mode": "relation", "direction": "both", "exclude_visited": true, "allow_stop_at_start": false, "budget": {"max_depth": 4, "max_decision_calls": 12, "max_input_tokens": null}, "length_alpha": 1.0, "prob_floor": 0.001, "temperature": 1.0, "top_p": 1.0, "n_samples": 5, "seed": 0, "beam_batching": "per_depth", "max_request_tokens": 64000, "max_state_plus_question_tokens": 32000, "token_safety_margin": 0.2, "label_style": "opaque", "include_summaries": true, "show_types": true, "stop_style": "literal", "relation_glosses": {"directed_by": "the director of the film", "written_by": "a writer of the film", "starred_actors": "an actor who starred in the film", "release_year": "the year the film was released", "in_language": "the language of the film", "has_genre": "the genre of the film", "has_tags": "keywords describing the film (topics, people, themes)", "has_imdb_rating": "the film's IMDb rating", "has_imdb_votes": "how many IMDb votes the film has"}, "answer_type": "hint", "answer_type_gate_min_p": 0.6, "prefilter_threshold": 50, "prefilter_top_n": 30, "max_frontier": 2000, "frontier_preview": 5}}`

## Calibration of walk confidence

| system | n | mean confidence | EM | ECE | Brier | AUROC | EM, most confident half |
|---|---|---|---|---|---|---|---|
| graphwalk-relation-v2-jev | 200 | 0.966 | 0.930 | 0.038 | 0.064 | 0.635 | 0.950 |
