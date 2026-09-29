# 2Wiki ingestion quality

- params: `{"embedder": "BAAI/bge-small-en-v1.5", "llm": "openrouter/openai/gpt-6-luna", "n": 10, "seed": 99}`
- git: `102921da4c20ba578047edf6a61b43614f3561a8` (dirty)
- run at: 2026-09-29T02:04:39.734893+00:00
- gold: 10 questions, 32 entities, 20 triples (1 with a literal object)

| variant | nodes | edges | entity recall | split | over-merged | triple recall (entity / literal) | complete chains | decision calls | escalations | LLM calls | cost $ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| jev | 590 | 722 | 0.812 | 0.154 | 1 | 0.800 (0.789 / 1.000) | 0.600 | 85 | 0 | 77 | 0.0497 |

- **jev**: 95 documents, 99 chunks, 729 entities / 727 relations extracted, 590 nodes created, 139 merged, 22 cached extractions, 0 errors, 297s
