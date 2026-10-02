# metaqa-1hop

- params: `{"concurrency": 4, "dataset": "metaqa", "hops": 1, "linking": "given", "n": 200, "rag_k": 5, "seed": 0, "split": "test", "total_questions": 9947, "variants": {"-logprob": {}}}`
- git: `63b06aa6a310b4340f1f5f561b4512190e31b421` (dirty)
- run at: 2026-10-02T02:13:09.146965+00:00

| system | n | hits@1 | EM | F1 | p50 s | p95 s | $/query | $/1k q | Jev calls/q | LLM calls/q | in tok/q | status | linking |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| graphwalk-relation-v2-logprob | 200 | 0.945 | 0.935 | 0.970 | 5.53 | 7.89 | 0.000095 | 0.095 | 1.96 | 0.00 | 906 | ok=200 | 1.000 |

## Systems

- **graphwalk-relation-v2-logprob** (wall 275s): `{"system": "graphwalk", "linking": "given", "decision_model": "logprob:qwen/qwen3.8-27b", "embedder": "fastembed:BAAI/bge-small-en-v1.5", "traversal": {"strategy": "greedy", "beam_width": 3, "hop_mode": "relation", "direction": "both", "exclude_visited": true, "allow_stop_at_start": false, "budget": {"max_depth": 4, "max_decision_calls": 12, "max_input_tokens": null}, "length_alpha": 1.0, "prob_floor": 0.001, "temperature": 1.0, "top_p": 1.0, "n_samples": 5, "seed": 0, "beam_batching": "per_depth", "max_request_tokens": 64000, "max_state_plus_question_tokens": 32000, "token_safety_margin": 0.2, "label_style": "opaque", "include_summaries": true, "show_types": true, "stop_style": "literal", "relation_glosses": {"directed_by": "the director of the film", "written_by": "a writer of the film", "starred_actors": "an actor who starred in the film", "release_year": "the year the film was released", "in_language": "the language of the film", "has_genre": "the genre of the film", "has_tags": "keywords describing the film (topics, people, themes)", "has_imdb_rating": "the film's IMDb rating", "has_imdb_votes": "how many IMDb votes the film has"}, "answer_type": "hint", "answer_type_gate_min_p": 0.6, "prefilter_threshold": 50, "prefilter_top_n": 30, "max_frontier": 2000, "frontier_preview": 5}}`

## Calibration of walk confidence

| system | n | mean confidence | EM | ECE | Brier | AUROC | EM, most confident half |
|---|---|---|---|---|---|---|---|
| graphwalk-relation-v2-logprob | 200 | 0.970 | 0.935 | 0.055 | 0.066 | 0.582 | 0.960 |
