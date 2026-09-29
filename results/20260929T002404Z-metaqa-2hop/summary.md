# metaqa-2hop

- params: `{"concurrency": 4, "dataset": "metaqa", "hops": 2, "linking": "given", "n": 200, "rag_k": 5, "seed": 0, "split": "test", "total_questions": 14872, "variants": {"": {}}}`
- git: `216023241137a15d15b4a4873f37114bc172a613` (dirty)
- run at: 2026-09-29T00:24:04.249728+00:00

| system | n | hits@1 | EM | F1 | p50 s | p95 s | $/query | $/1k q | Jev calls/q | LLM calls/q | in tok/q | status | linking |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| iter-rag | 200 | 0.830 | 0.760 | 0.810 | 4.07 | 11.75 | 0.000317 | 0.317 | 0.00 | 2.07 | 1580 | ok=200 | n/a |

## Systems

- **iter-rag** (wall 1380s): `{"system": "iter-rag", "k": 5, "k_search": 2, "max_steps": 5, "max_searches": 8, "max_docs": 40, "llm": "openrouter/openai/gpt-6-luna", "embedder": "fastembed:BAAI/bge-small-en-v1.5", "documents": 43234, "document": "one per entity: name, type, <= 40 incident triples"}`
