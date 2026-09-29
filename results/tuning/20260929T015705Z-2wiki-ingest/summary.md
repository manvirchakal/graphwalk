# 2Wiki ingestion quality

- params: `{"embedder": "BAAI/bge-small-en-v1.5", "llm": "openrouter/openai/gpt-6-luna", "n": 2, "seed": 99}`
- git: `102921da4c20ba578047edf6a61b43614f3561a8` (dirty)
- run at: 2026-09-29T01:58:32.021762+00:00
- gold: 2 questions, 7 entities, 4 triples (0 with a literal object)

| variant | nodes | edges | entity recall | split | over-merged | triple recall (entity / literal) | complete chains | decision calls | escalations | LLM calls | cost $ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| jev+llm | 158 | 189 | 0.857 | 0.333 | 1 | 1.000 (1.000 / n/a) | 1.000 | 16 | 0 | 22 | 0.0151 |

- **jev+llm**: 20 documents, 22 chunks, 191 entities / 189 relations extracted, 158 nodes created, 33 merged, 0 cached extractions, 0 errors, 86s
