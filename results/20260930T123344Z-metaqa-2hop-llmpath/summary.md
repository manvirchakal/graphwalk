# metaqa-2hop-llmpath

- params: `{"dataset": "metaqa", "hops": 2, "linking": "given", "llm": "openrouter/openai/gpt-6-luna", "n": 200, "seed": 0, "split": "test", "total_questions": 14872}`
- git: `b00becd556a5916ca669277993e8bf218532ff52`
- run at: 2026-09-30T13:18:20.592274+00:00

| system | n | hits@1 | EM | F1 | p50 s | p95 s | $/query | $/1k q | Jev calls/q | LLM calls/q | in tok/q | status | linking |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| llm-path | 200 | 1.000 | 1.000 | 1.000 | 1.84 | 3.48 | 0.000061 | 0.061 | 0.00 | 1.00 | 298 | ok=200 | 1.000 |
| llm-path-retry | 200 | 1.000 | 1.000 | 1.000 | 1.86 | 3.64 | 0.000061 | 0.061 | 0.00 | 1.00 | 298 | ok=200 | 1.000 |

## Systems

- **llm-path** (wall 665s): `{"system": "llm-path", "llm": "openrouter/openai/gpt-6-luna", "retries": 0, "relations": 9, "schema": "- film --directed_by--> person: the director of the film\n- film --has_genre--> genre: the genre of the film\n- film --has_imdb_rating--> rating: the film's IMDb rating\n- film --has_imdb_votes--> votes: how many IMDb votes the film has\n- film --has_tags--> tag: keywords describing the film (topics, people, themes)\n- film --in_language--> language: the language of the film\n- film --release_year--> year: the year the film was released\n- film --starred_actors--> person: an actor who starred in the film\n- film --written_by--> person: a writer of the film"}`
- **llm-path-retry** (wall 667s): `{"system": "llm-path", "llm": "openrouter/openai/gpt-6-luna", "retries": 2, "relations": 9, "schema": "- film --directed_by--> person: the director of the film\n- film --has_genre--> genre: the genre of the film\n- film --has_imdb_rating--> rating: the film's IMDb rating\n- film --has_imdb_votes--> votes: how many IMDb votes the film has\n- film --has_tags--> tag: keywords describing the film (topics, people, themes)\n- film --in_language--> language: the language of the film\n- film --release_year--> year: the year the film was released\n- film --starred_actors--> person: an actor who starred in the film\n- film --written_by--> person: a writer of the film"}`
