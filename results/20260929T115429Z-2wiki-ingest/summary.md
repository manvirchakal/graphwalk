# 2Wiki ingestion quality

- params: `{"embedder": "BAAI/bge-small-en-v1.5", "escalation_llm": "openrouter/xiaomi/mimo-v2.6-pro", "llm": "openrouter/openai/gpt-6-luna", "n": 40, "seed": 0}`
- git: `52557bd2648fc31b9addd9f99f369ca5b2c4a785`
- run at: 2026-09-29T12:00:57.974142+00:00
- gold: 40 questions, 117 entities, 80 triples (4 with a literal object)

| variant | nodes | edges | entity recall | split | over-merged | triple recall (entity / literal) | complete chains | decision calls | escalations | LLM calls | cost $ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| jev+llm | 1583 | 1695 | 0.880 | 0.039 | 0 | 0.775 (0.776 / 0.750) | 0.600 | 280 | 13 | 13 | 0.0502 |

- **jev+llm**: 309 documents, 316 chunks, 1919 entities / 1726 relations extracted, 1583 nodes created, 336 merged, 316 cached extractions, 0 errors, 388s
  - escalation to `openrouter/xiaomi/mimo-v2.6-pro`: 13 calls, 11486 input / 10094 output tokens, $0.0132
