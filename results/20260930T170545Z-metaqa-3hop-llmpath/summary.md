# metaqa-3hop-llmpath

- params: `{"dataset": "metaqa", "hops": 3, "linking": "given", "llm": "openrouter/openai/gpt-6-luna", "n": 200, "seed": 2, "split": "test", "total_questions": 14274}`
- git: `c11a457e3e8feac33826b6772ac6bab05281ebea`
- run at: 2026-09-30T17:40:25.795268+00:00

| system | n | hits@1 | EM | F1 | p50 s | p95 s | $/query | $/1k q | Jev calls/q | LLM calls/q | in tok/q | status | linking |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| llm-path-retry | 200 | 0.970 | 0.835 | 0.953 | 2.56 | 18.14 | 0.000097 | 0.097 | 0.00 | 1.07 | 330 | ok=200 | 1.000 |

## Systems

- **llm-path-retry** (wall 718s): `{"system": "llm-path", "llm": "openrouter/openai/gpt-6-luna", "retries": 2, "relations": 9, "schema": "- film --directed_by--> person: the director of the film\n- film --has_genre--> genre: the genre of the film\n- film --has_imdb_rating--> rating: the film's IMDb rating\n- film --has_imdb_votes--> votes: how many IMDb votes the film has\n- film --has_tags--> tag: keywords describing the film (topics, people, themes)\n- film --in_language--> language: the language of the film\n- film --release_year--> year: the year the film was released\n- film --starred_actors--> person: an actor who starred in the film\n- film --written_by--> person: a writer of the film"}`
