# 2wiki-text

- params: `{"addon": "hybrid-choice locate from 20260930T124411Z-2wiki-evidence", "embedder": "BAAI/bge-small-en-v1.5", "k": 5, "llm": "openrouter/openai/gpt-6-luna", "n": 120, "seed": 0}`
- git: `eb16e103e2e891f39592906ac471f532b307aa94`
- run at: 2026-09-30T21:41:20.130088+00:00

| system | n | hits@1 | EM | F1 | p50 s | p95 s | $/query | $/1k q | Jev calls/q | LLM calls/q | in tok/q | status | linking |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| iter-rag-steps5 | 120 | 0.675 | 0.675 | 0.743 | 4.33 | 10.63 | 0.000278 | 0.278 | 0.00 | 1.77 | 1663 | ok=120 | n/a |
| iter-rag-steps5+graph | 120 | 0.692 | 0.692 | 0.757 | 2.56 | 8.89 | 0.000222 | 0.222 | 0.00 | 1.40 | 1372 | ok=120 | n/a |
| iter-rag-steps2 | 120 | 0.667 | 0.667 | 0.732 | 3.06 | 5.38 | 0.000219 | 0.219 | 0.00 | 1.57 | 1404 | ok=120 | n/a |
| iter-rag-steps2+graph | 120 | 0.708 | 0.708 | 0.782 | 2.59 | 6.16 | 0.000166 | 0.166 | 0.00 | 1.31 | 1248 | ok=120 | n/a |
| iter-rag-steps5-notitles | 120 | 0.708 | 0.708 | 0.780 | 3.63 | 13.33 | 0.000280 | 0.280 | 0.00 | 1.78 | 1685 | ok=120 | n/a |
| iter-rag-steps5-notitles+graph | 120 | 0.700 | 0.700 | 0.763 | 2.25 | 10.61 | 0.000252 | 0.252 | 0.00 | 1.47 | 1520 | ok=120 | n/a |
| iter-rag-steps2-notitles | 120 | 0.683 | 0.683 | 0.742 | 3.62 | 6.61 | 0.000230 | 0.230 | 0.00 | 1.59 | 1444 | ok=120 | n/a |
| iter-rag-steps2-notitles+graph | 120 | 0.675 | 0.675 | 0.747 | 2.35 | 5.55 | 0.000197 | 0.197 | 0.00 | 1.31 | 1251 | ok=120 | n/a |

## Systems

- **iter-rag-steps5** (wall 709s): `{"system": "iter-rag", "k": 5, "k_search": 2, "first": "question embedding", "title_search": true, "max_steps": 5, "max_searches": 8, "max_docs": 40, "llm": "openrouter/openai/gpt-6-luna", "embedder": "fastembed:BAAI/bge-small-en-v1.5", "documents": 970, "document": "one per paragraph: title and text"}`
- **iter-rag-steps5+graph** (wall 561s): `{"system": "iter-rag", "k": 5, "k_search": 2, "first": "custom", "title_search": true, "max_steps": 5, "max_searches": 8, "max_docs": 40, "llm": "openrouter/openai/gpt-6-luna", "embedder": "fastembed:BAAI/bge-small-en-v1.5", "documents": 970, "document": "one per paragraph: title and text"}`
- **iter-rag-steps2** (wall 629s): `{"system": "iter-rag", "k": 5, "k_search": 2, "first": "question embedding", "title_search": true, "max_steps": 2, "max_searches": 8, "max_docs": 40, "llm": "openrouter/openai/gpt-6-luna", "embedder": "fastembed:BAAI/bge-small-en-v1.5", "documents": 970, "document": "one per paragraph: title and text"}`
- **iter-rag-steps2+graph** (wall 524s): `{"system": "iter-rag", "k": 5, "k_search": 2, "first": "custom", "title_search": true, "max_steps": 2, "max_searches": 8, "max_docs": 40, "llm": "openrouter/openai/gpt-6-luna", "embedder": "fastembed:BAAI/bge-small-en-v1.5", "documents": 970, "document": "one per paragraph: title and text"}`
- **iter-rag-steps5-notitles** (wall 715s): `{"system": "iter-rag", "k": 5, "k_search": 2, "first": "question embedding", "title_search": false, "max_steps": 5, "max_searches": 8, "max_docs": 40, "llm": "openrouter/openai/gpt-6-luna", "embedder": "fastembed:BAAI/bge-small-en-v1.5", "documents": 970, "document": "one per paragraph: title and text"}`
- **iter-rag-steps5-notitles+graph** (wall 586s): `{"system": "iter-rag", "k": 5, "k_search": 2, "first": "custom", "title_search": false, "max_steps": 5, "max_searches": 8, "max_docs": 40, "llm": "openrouter/openai/gpt-6-luna", "embedder": "fastembed:BAAI/bge-small-en-v1.5", "documents": 970, "document": "one per paragraph: title and text"}`
- **iter-rag-steps2-notitles** (wall 637s): `{"system": "iter-rag", "k": 5, "k_search": 2, "first": "question embedding", "title_search": false, "max_steps": 2, "max_searches": 8, "max_docs": 40, "llm": "openrouter/openai/gpt-6-luna", "embedder": "fastembed:BAAI/bge-small-en-v1.5", "documents": 970, "document": "one per paragraph: title and text"}`
- **iter-rag-steps2-notitles+graph** (wall 524s): `{"system": "iter-rag", "k": 5, "k_search": 2, "first": "custom", "title_search": false, "max_steps": 2, "max_searches": 8, "max_docs": 40, "llm": "openrouter/openai/gpt-6-luna", "embedder": "fastembed:BAAI/bge-small-en-v1.5", "documents": 970, "document": "one per paragraph: title and text"}`

## By question type

| system | bridge_comparison F1 (EM) | comparison F1 (EM) | compositional F1 (EM) | inference F1 (EM) |
|---|---|---|---|---|
| iter-rag-steps5 | 0.967 (0.967) n=30 | 0.900 (0.900) n=30 | 0.322 (0.233) n=30 | 0.786 (0.600) n=30 |
| iter-rag-steps5+graph | 0.967 (0.967) n=30 | 0.900 (0.900) n=30 | 0.322 (0.233) n=30 | 0.839 (0.667) n=30 |
| iter-rag-steps2 | 0.900 (0.900) n=30 | 0.900 (0.900) n=30 | 0.338 (0.267) n=30 | 0.791 (0.600) n=30 |
| iter-rag-steps2+graph | 1.000 (1.000) n=30 | 0.900 (0.900) n=30 | 0.360 (0.267) n=30 | 0.868 (0.667) n=30 |
| iter-rag-steps5-notitles | 0.989 (0.967) n=30 | 0.900 (0.900) n=30 | 0.422 (0.333) n=30 | 0.810 (0.633) n=30 |
| iter-rag-steps5-notitles+graph | 0.967 (0.967) n=30 | 0.900 (0.900) n=30 | 0.328 (0.267) n=30 | 0.858 (0.667) n=30 |
| iter-rag-steps2-notitles | 0.967 (0.967) n=30 | 0.867 (0.867) n=30 | 0.361 (0.267) n=30 | 0.775 (0.633) n=30 |
| iter-rag-steps2-notitles+graph | 0.900 (0.900) n=30 | 0.867 (0.867) n=30 | 0.407 (0.300) n=30 | 0.815 (0.633) n=30 |
