---
name: graphwalk
description: Answer questions over an existing knowledge graph (triples, RDF, CSV/JSONL edge lists) by walking it hop by hop with a fast classifier, getting entity answers with a confidence and the path taken; escalate low-confidence queries to an LLM. Use when a user has a knowledge graph and wants cheap, fast multi-hop lookups with a per-answer confidence, and can accept lower accuracy than an LLM writing the query. Not for best-accuracy KG-QA (have an LLM write the query from the relations near the entity), QA over documents (use RAG), or constrained/superlative questions.
---

# graphwalk

Run `graphwalk guide` (or `python -c "import graphwalk; print(graphwalk.guide())"`)
for the full guide: when to use it, the measured trade-offs, recipes, and API.

Short version:

1. Check fit first. graphwalk answers from the graph's nodes. It is 6× cheaper and
   ~4× faster than an LLM writing the query, and gives an informative confidence, but
   is ~15 F1 points less accurate on curated graphs of any schema size. It loses to
   multi-step RAG on documents. It does no filtering, ranking, or set intersection.
2. Import: `graphwalk import kg.nt --graph kg.db` (also `.csv`, `.tsv`, `.jsonl`). No
   key is needed. Give nodes readable names and types.
3. Query from Python:

   ```python
   from graphwalk import Index

   # walks with TraversalConfig.kgqa() by default
   async with Index.open("kg.db", node_types=[...], escalate_below=0.9) as index:
       result = await index.walk(question)
       result.best.names, result.best.path, result.confidence, result.escalated
   ```

4. Treat `result.confidence` as a ranking signal. Validate the threshold on 50–100
   labeled questions from the user's own graph before trusting it.
5. Keys: `OPENROUTER_API_KEY` (Jev decisions; the LLM for escalation). Without Jev,
   use `GRAPHWALK_DECISION_FALLBACK=llm`; its confidence is not calibrated.
