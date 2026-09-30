# metaqa-3hop-llmpath-dev

- params: `{"dataset": "metaqa", "hops": 3, "linking": "given", "llm": "openrouter/openai/gpt-6-luna", "n": 20, "seed": 0, "split": "dev", "total_questions": 14274}`
- git: `ec8490f74e39dc596aa0b5b6100b5c7aa78ad52c` (dirty)
- run at: 2026-09-30T12:33:33.523102+00:00

| system | n | hits@1 | EM | F1 | p50 s | p95 s | $/query | $/1k q | Jev calls/q | LLM calls/q | in tok/q | status | linking |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| llm-path | 20 | 1.000 | 0.900 | 0.990 | 2.25 | 2.83 | 0.000081 | 0.081 | 0.00 | 1.00 | 302 | ok=20 | 1.000 |
| llm-path-retry | 20 | 1.000 | 0.900 | 0.990 | 2.36 | 2.79 | 0.000084 | 0.084 | 0.00 | 1.00 | 302 | ok=20 | 1.000 |

## Systems

- **llm-path** (wall 66s): `{"system": "llm-path", "llm": "openrouter/openai/gpt-6-luna", "retries": 0, "relations": 9, "schema": "- film --directed_by--> person: the director of the film\n- film --has_genre--> genre: the genre of the film\n- film --has_imdb_rating--> rating: the film's IMDb rating\n- film --has_imdb_votes--> votes: how many IMDb votes the film has\n- film --has_tags--> tag: keywords describing the film (topics, people, themes)\n- film --in_language--> language: the language of the film\n- film --release_year--> year: the year the film was released\n- film --starred_actors--> person: an actor who starred in the film\n- film --written_by--> person: a writer of the film"}`
- **llm-path-retry** (wall 66s): `{"system": "llm-path", "llm": "openrouter/openai/gpt-6-luna", "retries": 2, "relations": 9, "schema": "- film --directed_by--> person: the director of the film\n- film --has_genre--> genre: the genre of the film\n- film --has_imdb_rating--> rating: the film's IMDb rating\n- film --has_imdb_votes--> votes: how many IMDb votes the film has\n- film --has_tags--> tag: keywords describing the film (topics, people, themes)\n- film --in_language--> language: the language of the film\n- film --release_year--> year: the year the film was released\n- film --starred_actors--> person: an actor who starred in the film\n- film --written_by--> person: a writer of the film"}`
