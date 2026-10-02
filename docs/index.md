# graphwalk

**Cheap, confidence-scored question answering over a knowledge graph you already
have.** A Python library, a CLI, and an MCP server for agents.

graphwalk answers a question by walking the graph from the entities it names. Each hop
is a **constrained classification decision**: the current node's relations, plus
`STOP`, are the options, and a fast decision model (Jev) returns a probability for each.
Every answer comes with the path that reached it and a **confidence**. Unsure answers
can be escalated to a stronger model, or checked by an agent.

```bash
pip install graphwalk
graphwalk import kg.nt --graph kg.db
graphwalk query kg.db "Where was the director of Inception born?" --escalate-below 0.9
```

## When to use it

Every row is a measured result; the [evidence page](evidence.md) has the tables,
confidence intervals, and costs. Samples are 30–300 questions per setting, mostly one
seed: read them as directions.

| Situation | Use | What we measured |
|---|---|---|
| An **agent** exploring a KG with graph tools | Give it `walk` too ([MCP](agents.md)) | 17% cheaper (95% CI 8–27%) at equal F1, strong agent, WebQSP, n=100 |
| Existing KG, **cost or latency** first, or a per-answer confidence | `walk` | ~6× cheaper, ~4× faster than an LLM writing the query; AUROC of confidence 0.92 (WebQSP) to 0.97 (MetaQA) |
| Existing KG, **best accuracy** | An LLM writing the query | 0.66 vs 0.49 F1 on a 5,419-relation Freebase graph |
| A cheap first pass that **knows when it's wrong** | `walk` with `escalate_below` | Modest savings: the LLM path writer's accuracy for ~19% less (Freebase) |
| Questions with **constraints or superlatives** | Not graphwalk | Every system tried scored about 0.3 on CWQ |
| **Questions over documents** | Multi-step RAG | RAG wins by 10–19 F1; graphwalk's text ingestion is [experimental](text.md) |

The short version: graphwalk is not more accurate than an LLM. It gives a cheap first
answer with a confidence you can act on, and it saves an agent tokens.

## How a walk works

1. **Link**: find the graph nodes the question names.
2. **Decide**: at each node, the options are its relations (grouped, with counts and
   example targets) and `STOP`. Jev returns a distribution over them; the walk follows
   the best (greedy) or the top few (beam).
3. **Answer**: the nodes where the walk stops, with the path and the confidence (the
   length-normalized path probability).
4. **Escalate** (optional): below a threshold, re-walk with an LLM decider.

Read next: the [quickstart](quickstart.md), or [agents and MCP](agents.md).

!!! note "Status"
    Alpha (v0.1). Jev is a hosted model from TypeSafe; this project has no affiliation
    with TypeSafe. Without Jev, `GRAPHWALK_DECISION_FALLBACK=llm` makes any chat model
    the decider (slower, and its confidence is not calibrated).
