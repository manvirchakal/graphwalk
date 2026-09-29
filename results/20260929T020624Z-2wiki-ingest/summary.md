# 2Wiki ingestion quality

- params: `{"embedder": "BAAI/bge-small-en-v1.5", "llm": "openrouter/openai/gpt-6-luna", "n": 40, "seed": 0}`
- git: `92d339c8b265ad6ddc8464073efdd1f901cfb034`
- run at: 2026-09-29T02:28:36.592667+00:00
- gold: 40 questions, 117 entities, 80 triples (4 with a literal object)

| variant | nodes | edges | entity recall | split | over-merged | triple recall (entity / literal) | complete chains | decision calls | escalations | LLM calls | cost $ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| exact | 1653 | 1710 | 0.880 | 0.029 | 0 | 0.688 (0.684 / 0.750) | 0.425 | 0 | 0 | 300 | 0.1252 |
| jev | 1586 | 1695 | 0.880 | 0.058 | 1 | 0.775 (0.776 / 0.750) | 0.600 | 280 | 0 | 0 | 0.0370 |
| jev+llm | 1586 | 1695 | 0.880 | 0.049 | 0 | 0.775 (0.776 / 0.750) | 0.600 | 280 | 19 | 19 | 0.0401 |

- **exact**: 309 documents, 316 chunks, 1919 entities / 1726 relations extracted, 1653 nodes created, 266 merged, 16 cached extractions, 0 errors, 1056s
- **jev**: 309 documents, 316 chunks, 1919 entities / 1726 relations extracted, 1586 nodes created, 333 merged, 316 cached extractions, 0 errors, 109s
- **jev+llm**: 309 documents, 316 chunks, 1919 entities / 1726 relations extracted, 1586 nodes created, 333 merged, 316 cached extractions, 0 errors, 165s
