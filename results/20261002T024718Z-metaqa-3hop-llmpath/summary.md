# metaqa-3hop-llmpath

- params: `{"dataset": "metaqa", "hops": 3, "linking": "given", "llm": "openrouter/openai/gpt-6-luna", "n": 500, "seed": 0, "split": "test", "total_questions": 14274}`
- git: `ca64b4b58c76298b21777157409fcb2697a590c4` (dirty)
- run at: 2026-10-02T02:56:09.166650+00:00

| system | n | hits@1 | EM | F1 | p50 s | p95 s | $/query | $/1k q | Jev calls/q | LLM calls/q | in tok/q | status | linking |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| llm-path-retry | 500 | 0.980 | 0.834 | 0.964 | 2.19 | 3.05 | 0.000092 | 0.092 | 0.00 | 1.05 | 320 | ok=500 | 1.000 |

## Systems

- **llm-path-retry** (wall 526s): `{"system": "llm-path", "llm": "openrouter/openai/gpt-6-luna", "retries": 2, "relations": 9, "schema": "- film --directed_by--> person: the director of the film\n- film --has_genre--> genre: the genre of the film\n- film --has_imdb_rating--> rating: the film's IMDb rating\n- film --has_imdb_votes--> votes: how many IMDb votes the film has\n- film --has_tags--> tag: keywords describing the film (topics, people, themes)\n- film --in_language--> language: the language of the film\n- film --release_year--> year: the year the film was released\n- film --starred_actors--> person: an actor who starred in the film\n- film --written_by--> person: a writer of the film"}`
