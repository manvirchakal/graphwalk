# 2wiki-gold

- params: `{"concurrency": 4, "dataset": "2wiki", "hops": null, "linking": "gold", "n": 300, "rag_k": 5, "seed": 0, "split": "test", "total_questions": 6785, "variants": {"": {}}}`
- git: `216023241137a15d15b4a4873f37114bc172a613` (dirty)
- run at: 2026-09-29T01:23:21.502182+00:00

| system | n | hits@1 | EM | F1 | p50 s | p95 s | $/query | $/1k q | Jev calls/q | LLM calls/q | in tok/q | status | linking |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| iter-rag | 300 | 0.917 | 0.917 | 0.921 | 1.86 | 5.27 | 0.000099 | 0.099 | 0.00 | 1.44 | 417 | ok=300 | n/a |

## Systems

- **iter-rag** (wall 1434s): `{"system": "iter-rag", "k": 5, "k_search": 2, "max_steps": 5, "max_searches": 8, "max_docs": 40, "llm": "openrouter/openai/gpt-6-luna", "embedder": "fastembed:BAAI/bge-small-en-v1.5", "documents": 33091, "document": "one per entity: name, type, <= 40 incident triples"}`
