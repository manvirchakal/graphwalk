# Text ingestion (experimental)

graphwalk can build a graph from documents and use it to find passages. **It does not
answer questions over documents better than RAG**, and we don't recommend it for that:
on 2Wiki, HotpotQA, and FanOutQA, multi-step RAG beat walking an extracted graph by
10–19 F1. Extraction drops the dates, order, and qualifiers that questions need
([M7](results-m7.md)). What does work is using the graph to *locate* passages when
questions chain through named entities.

## Ingest

```bash
pip install 'graphwalk[llm,embeddings]'
graphwalk ingest docs/ --graph docs.db --report ingest-report.json
graphwalk locate docs.db "Who directed ...?" -k 5 --context 200
```

`ingest` reads `.txt`/`.md` files (one document each) and `.json`/`.jsonl`/`.csv`
records (one document per record). An LLM extracts entities and relations, and Jev
decides for each entity whether it is an existing node or a new one; low-confidence
decisions go to the LLM. Re-running is idempotent: unchanged documents are skipped,
changed ones are retracted and re-ingested, and `--prune` retracts deleted ones.
Ingestion costs money (one LLM call per chunk); the report records it.

## `locate` and `read`

`locate` walks from the entities a question names and returns the source spans behind
the walk (each extracted fact records the sentence it came from), with the path that
reached them. `read` returns the text.

```python
from graphwalk import Index

async with Index.open("docs.db") as index:  # inside async code
    await index.ingest("docs/")
    for location in await index.locate("Who directed ...?", k=5, mode="hybrid"):
        passage = await index.read(location, context=200)
        print(location.path, passage.text)  # answer from the text, not the graph
```

| Mode | What it does | When ([E1](results-phase4.md)) |
|---|---|---|
| `graph` | Walk, return the spans behind the walk | Rarely best alone |
| `dense` | Embedding similarity over text chunks | Broad or list-style questions (FanOutQA: hybrid −5 recall) |
| `hybrid` | Fuses both (`dense` and `hybrid` need the `embeddings` extra) | Entity chains (2Wiki: +18 recall over dense; ties on HotpotQA) |

`read` detects documents that changed since `locate` ran and flags the passage as
`stale` (or refuses, with `on_stale="refuse"`). By default the text is kept in the
store; `IngestConfig(store_text=False)` with `FileDocuments` re-reads it from disk.
