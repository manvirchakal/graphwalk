# Evidence

Every claim in these docs comes from an experiment below. Each result page names the
script that regenerates its table and the run directory under
[`results/`](../results/README.md) that holds the committed summary and per-question
scores. Intervals are 95% bootstraps over questions.

**Read the numbers as directions, not guarantees.** Samples are 30–300 questions per
setting, most settings ran with one seed, and the decision model (Jev) is a hosted
model whose versions can change.

## Summary

| Question | Answer | Where |
|---|---|---|
| Does `walk` make an agent cheaper? | **Yes, with an expensive agent:** −25% cost (95% CI 7–45%) at equal F1, strong agent, n=30. With a cheap agent, the cost was a wash. | [A8, A8b](results-phase7.md) |
| Does `walk` make an agent more accurate? | **No.** 0.72 vs 0.74 F1 (cheap agent, n=100); 0.75 vs 0.71 (strong agent, n=30); neither CI excludes 0. | [A8, A8b](results-phase7.md) |
| Is the walk's confidence informative? | **Yes:** AUROC 0.92–0.97 on WebQSP and MetaQA 2–3 hop; weaker (0.64–0.71) on MetaQA 1-hop, 2Wiki, and CWQ. The LLM decider's confidence barely is (0.50–0.69). | [E3](results-phase4.md), [A2](results-phase7.md) |
| Can we predict before walking whether it will work? | **Barely**, within a graph (AUROC 0.55–0.68). Walk first, read the confidence. | [A7](results-phase7.md) |
| Walk vs an LLM writing the query, curated graphs | The LLM is **more accurate**: MetaQA 3-hop 0.955 vs 0.887; Freebase (5,419 relations) 0.66 vs 0.49. Walking is ~6× cheaper and ~4× faster. | [E2](results-phase4.md), [A6](results-phase7.md) |
| Walk vs multi-step RAG, curated graph (MetaQA) | Walking **wins**: 0.97/0.99/0.89 vs 0.89/0.81/0.41 F1 (1/2/3 hop) at 3–5× lower cost. | [M5](results-m5.md) |
| Escalating unsure walks | Works, modest savings: the LLM decider's accuracy at about half its cost on MetaQA; ~19% savings on Freebase. | [A1, A6](results-phase7.md) |
| Constraint questions (CWQ) | Every system tried: hits@1 ≈ 0.28, far below published methods (0.63–0.69). | [A2](results-phase7.md) |
| QA over text (graph built from documents) | Multi-step RAG **wins** by 10–19 F1, even with a 20× pricier extractor. | [M7](results-m7.md), [E4, E5](results-phase4.md) |
| The graph as a retriever over text | Hybrid beats dense by 18 recall on entity chains (2Wiki); ties or loses elsewhere. | [E1](results-phase4.md) |
| Scale | SQLite, 1.4M edges: walks add ~8 ms per walk on typical nodes, 0.1–0.2 s from a 20k-neighbor hub. | [A5](results-phase7.md) |

## Reproducing

Each table has one command. The paper tables read the committed scores and need no API
key:

```bash
uv sync --extra eval
uv run python scripts/paper/table_curated.py      # E2/E3/E4
uv run python scripts/paper/table_escalation.py   # A1
uv run python scripts/paper/figure_calibration.py # E3 calibration
```

Rerunning an experiment from scratch needs `OPENROUTER_API_KEY` and the extras its
script names in its docstring, for example:

```bash
uv sync --extra eval --extra llm --extra embeddings
uv run python scripts/eval/agent_arms.py --n 100 --arms graph walk search
```

Datasets are not redistributed: the loaders download MetaQA, WebQSP and CWQ (the
RoG subgraphs), 2WikiMultiHopQA, HotpotQA, and FanOutQA from their public sources
and cache them locally.

## Pages

- [Phase 7](results-phase7.md): escalation (A1), Freebase KG-QA (A2, A6), scale (A5),
  routing (A7), graphwalk as an agent's tool (A8).
- [Phase 4](results-phase4.md): retrieval (E1), LLM-written queries (E2), Jev vs an LLM
  decider (E3), seeds (E4), extractors (E5), cost (E6), RAG add-on (E7).
- [M7](results-m7.md): QA over graphs built from text.
- [M6](results-m6.md): ingestion quality.
- [M5](results-m5.md): curated graphs vs RAG.
