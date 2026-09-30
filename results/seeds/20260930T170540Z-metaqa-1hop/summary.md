# metaqa-1hop

- params: `{"concurrency": 4, "dataset": "metaqa", "hops": 1, "linking": "given", "n": 200, "rag_k": 5, "seed": 1, "split": "test", "total_questions": 9947, "variants": {"": {}}}`
- git: `cc090a1b7d98f401dba10a87c01a9e75921149c8`
- run at: 2026-09-30T17:05:40.980079+00:00

| system | n | hits@1 | EM | F1 | p50 s | p95 s | $/query | $/1k q | Jev calls/q | LLM calls/q | in tok/q | status | linking |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| vector-rag | 200 | 0.895 | 0.855 | 0.899 | 1.62 | 15.44 | 0.000057 | 0.057 | 0.00 | 1.00 | 383 | ok=200 | n/a |

## Systems

- **vector-rag** (wall 666s): `{"system": "vector-rag", "k": 5, "llm": "openrouter/openai/gpt-6-luna", "embedder": "fastembed:BAAI/bge-small-en-v1.5", "documents": 43234, "document": "one per entity: name, type, <= 40 incident triples"}`
