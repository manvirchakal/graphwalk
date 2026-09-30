# metaqa-1hop

- params: `{"concurrency": 4, "dataset": "metaqa", "hops": 1, "linking": "given", "n": 200, "rag_k": 5, "seed": 2, "split": "test", "total_questions": 9947, "variants": {"": {}}}`
- git: `1b04ef0d2591ba392acc1dd395f8dbaa5d3315b6`
- run at: 2026-09-30T17:51:43.902738+00:00

| system | n | hits@1 | EM | F1 | p50 s | p95 s | $/query | $/1k q | Jev calls/q | LLM calls/q | in tok/q | status | linking |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| vector-rag | 200 | 0.875 | 0.810 | 0.859 | 1.57 | 3.01 | 0.000054 | 0.054 | 0.00 | 1.00 | 343 | ok=200 | n/a |

## Systems

- **vector-rag** (wall 665s): `{"system": "vector-rag", "k": 5, "llm": "openrouter/openai/gpt-6-luna", "embedder": "fastembed:BAAI/bge-small-en-v1.5", "documents": 43234, "document": "one per entity: name, type, <= 40 incident triples"}`
