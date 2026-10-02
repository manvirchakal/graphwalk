# Quickstart

## Install

```bash
pip install graphwalk              # import, walk, the CLI
pip install 'graphwalk[mcp]'       # + the MCP server
pip install 'graphwalk[llm]'       # + escalation and the LLM decider (LiteLLM)
```

Python 3.12 or newer. Other extras: `embeddings` (a local embedder for the type
prefilter and dense `locate`) and `eval` (benchmark loaders).

## Keys

Each hop is decided by a **decider**. The default is **Jev**, a fast classification
model:

```bash
export OPENROUTER_API_KEY=sk-or-...        # model typesafe/jev-1.13 on OpenRouter
# or: export TYPESAFE_API_KEY=... GRAPHWALK_DECISION_PROVIDER=typesafe
```

Or decide with an open-weights model's token probabilities, through OpenRouter or your
own server (as accurate, its confidence as informative, about 10× slower through
OpenRouter; see [deciders](deciders.md)):

```bash
export GRAPHWALK_DECIDER=logprob           # default model qwen/qwen3.8-27b on OpenRouter
# your own server: GRAPHWALK_DECIDER_BASE_URL=http://localhost:8000/v1 GRAPHWALK_DECIDER_MODEL=...
```

Escalation needs an LLM key too (the same OpenRouter key works, or OpenAI, Anthropic,
or x.ai; see [configuration](configuration.md)). Importing a graph needs no key.

## Import a graph

From triples: CSV/TSV with a header, JSONL, or N-Triples.

```csv
subject,relation,object,subject_type,object_type
Inception,directed_by,Christopher Nolan,film,person
Christopher Nolan,born_in,London,person,place
```

```bash
graphwalk import movies.csv --graph movies.db
```

Header aliases: `head`/`source`/`s`, `predicate`/`type`/`p`, `tail`/`target`/`o`, and
optional `subject_name`/`object_name`. In N-Triples, `rdfs:label` becomes the name and
`rdf:type` the type. **Give nodes readable names and types:** the decider reads names,
relation names, and types, not opaque ids. Imports are idempotent and run at about 9k
triples/s into SQLite.

## Ask

```bash
graphwalk query movies.db "Where was the director of Inception born?"
graphwalk query movies.db "Where was the director of Inception born?" --escalate-below 0.9
```

```python
import asyncio
from graphwalk import Index


async def main() -> None:
    async with Index.open(
        "movies.db",
        node_types=["film", "person", "place"],  # your graph's types: an answer-type hint
        escalate_below=0.9,                      # optional; needs the llm extra
    ) as index:
        result = await index.walk("Where was the director of Inception born?")
        if result.best is None:
            print("no answer")  # no entity linked, or nothing reached
        else:
            print(result.best.names)      # a set of nodes
            print(result.best.path)       # the hops taken: the justification
            print(result.confidence, result.escalated, result.cost_usd)


asyncio.run(main())
```

A runnable version over a small movie graph:
[`examples/kg/walk.py`](../examples/kg/walk.py).

## Reading the confidence

`result.confidence` is the length-normalized probability of the path the walk took.
On the benchmarks it ranks answers well (AUROC 0.92 on WebQSP, 0.97 on MetaQA 2–3 hop),
and **0.9** was a good escalation threshold. On your own graph, check it against
50–100 labeled questions before relying on it: the threshold does not transfer
automatically. The LLM-decider fallback's confidence barely is informative: it is the
model's stated scores, not probabilities.

## What it can't do

- **Constraints.** "The earliest...", "both A and B", superlatives: a walk follows one
  relation path and does not filter, rank, or intersect. Do that in your code, or let
  an agent do it with the result.
- **Facts the graph lacks.** Answers are nodes. Dates or qualifiers that live in text
  or attributes are out of reach.
