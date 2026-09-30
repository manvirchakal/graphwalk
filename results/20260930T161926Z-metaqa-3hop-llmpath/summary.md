# metaqa-3hop-llmpath

- params: `{"dataset": "metaqa", "hops": 3, "linking": "given", "llm": "openrouter/openai/gpt-6-luna", "n": 200, "seed": 1, "split": "test", "total_questions": 14274}`
- git: `c53a3dd7194260feb5223c160b8892ecef87952e` (dirty)
- run at: 2026-09-30T16:54:14.580582+00:00

| system | n | hits@1 | EM | F1 | p50 s | p95 s | $/query | $/1k q | Jev calls/q | LLM calls/q | in tok/q | status | linking |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| llm-path-retry | 200 | 0.980 | 0.800 | 0.960 | 2.52 | 20.02 | 0.000090 | 0.090 | 0.00 | 1.04 | 319 | ok=200 | 1.000 |

## Systems

- **llm-path-retry** (wall 718s): `{"system": "llm-path", "llm": "openrouter/openai/gpt-6-luna", "retries": 2, "relations": 9, "schema": "- film --directed_by--> person: the director of the film\n- film --has_genre--> genre: the genre of the film\n- film --has_imdb_rating--> rating: the film's IMDb rating\n- film --has_imdb_votes--> votes: how many IMDb votes the film has\n- film --has_tags--> tag: keywords describing the film (topics, people, themes)\n- film --in_language--> language: the language of the film\n- film --release_year--> year: the year the film was released\n- film --starred_actors--> person: an actor who starred in the film\n- film --written_by--> person: a writer of the film"}`
