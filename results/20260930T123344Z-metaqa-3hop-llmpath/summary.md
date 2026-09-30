# metaqa-3hop-llmpath

- params: `{"dataset": "metaqa", "hops": 3, "linking": "given", "llm": "openrouter/openai/gpt-6-luna", "n": 200, "seed": 0, "split": "test", "total_questions": 14274}`
- git: `420192337f549e273f1ac28f514b0e647cf27879`
- run at: 2026-09-30T13:41:05.372964+00:00

| system | n | hits@1 | EM | F1 | p50 s | p95 s | $/query | $/1k q | Jev calls/q | LLM calls/q | in tok/q | status | linking |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| llm-path | 200 | 0.975 | 0.825 | 0.961 | 2.31 | 3.26 | 0.000080 | 0.080 | 0.00 | 1.00 | 302 | ok=200 | 1.000 |
| llm-path-retry | 200 | 0.965 | 0.825 | 0.952 | 2.38 | 3.81 | 0.000091 | 0.091 | 0.00 | 1.04 | 317 | ok=200 | 1.000 |

## Systems

- **llm-path** (wall 665s): `{"system": "llm-path", "llm": "openrouter/openai/gpt-6-luna", "retries": 0, "relations": 9, "schema": "- film --directed_by--> person: the director of the film\n- film --has_genre--> genre: the genre of the film\n- film --has_imdb_rating--> rating: the film's IMDb rating\n- film --has_imdb_votes--> votes: how many IMDb votes the film has\n- film --has_tags--> tag: keywords describing the film (topics, people, themes)\n- film --in_language--> language: the language of the film\n- film --release_year--> year: the year the film was released\n- film --starred_actors--> person: an actor who starred in the film\n- film --written_by--> person: a writer of the film"}`
- **llm-path-retry** (wall 694s): `{"system": "llm-path", "llm": "openrouter/openai/gpt-6-luna", "retries": 2, "relations": 9, "schema": "- film --directed_by--> person: the director of the film\n- film --has_genre--> genre: the genre of the film\n- film --has_imdb_rating--> rating: the film's IMDb rating\n- film --has_imdb_votes--> votes: how many IMDb votes the film has\n- film --has_tags--> tag: keywords describing the film (topics, people, themes)\n- film --in_language--> language: the language of the film\n- film --release_year--> year: the year the film was released\n- film --starred_actors--> person: an actor who starred in the film\n- film --written_by--> person: a writer of the film"}`
