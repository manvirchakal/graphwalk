# M7 results: QA over graphs built from text

**Question.** M5 showed graphwalk beating RAG on MetaQA, where the graph is a clean,
curated knowledge base. Does that hold when the graph comes from ingesting text (M6)?

**Answer: no.** On both text datasets, RAG over the original paragraphs beats every
graphwalk variant, and costs less. The graph-only walk is far behind. Adding an LLM
reader closes about half the gap, but it still trails multi-step RAG by 21–27 F1
points.

## Setup

- **Questions.** Seeded, type-stratified samples (seed 0):
  - 2WikiMultiHopQA dev: 30 each of compositional, inference, comparison, and
    bridge-comparison.
  - HotpotQA distractor validation: 60 each of bridge and comparison.
- **Knowledge.** Every paragraph of the sampled questions (gold and distractor,
  de-duplicated by title) is pooled: 970 paragraphs for 2Wiki, 1,187 for HotpotQA.
  Every system reads this same pool.
  - graphwalk reads it as a graph ingested by the M6 pipeline: `gpt-6-luna`
    extraction, Jev routing, escalation to `mimo-v2.6-pro`.
  - The RAG baselines read the paragraphs as-is: one document per paragraph,
    bge-small, top-5.
- **Systems.** All use `gpt-6-luna` as the LLM, with a short-answer prompt.
  - `graphwalk`: graph-only walk (`greedy-v2` preset, choice linking). The answer is
    the name of the node reached.
  - `graphwalk-reader`: walks from up to two entities named in the question. The LLM
    reads the walked paths plus each visited node's summary, attributes, and up to 12
    facts.
  - `text-rag`: one-shot RAG over the paragraphs.
  - `text-iter-rag`: multi-step RAG (up to 5 steps; searches by title or embedding).
- **Metrics.** SQuAD-style EM and token F1 against the single gold answer.
- **Runs.** `results/20260929T154958Z-2wiki-text/` and
  `results/20260929T171954Z-hotpotqa-text/`. Script: `scripts/text_qa.py`.

## Results

### 2WikiMultiHopQA (120 questions)

| system | F1 | EM | $/1k q | p50 s |
|---|---|---|---|---|
| graphwalk (graph only) | 0.254 | 0.167 | 0.18 | 0.9 |
| graphwalk + reader | 0.539 | 0.492 | 0.33 | 2.7 |
| text RAG | 0.368 | 0.325 | 0.10 | 2.0 |
| **text multi-step RAG** | **0.749** | **0.683** | ≈0.15* | 3.8 |

\*Two questions errored on OpenRouter's 20-requests/min limit, so the run reports no
total cost. The per-query cost is in line with HotpotQA.

| system (F1) | compositional | inference | comparison | bridge-comparison |
|---|---|---|---|---|
| graphwalk (graph only) | 0.320 | 0.668 | 0.029 | 0.000 |
| graphwalk + reader | 0.211 | 0.746 | 0.667 | 0.533 |
| text RAG | 0.167 | 0.404 | 0.867 | 0.033 |
| text multi-step RAG | 0.299 | **0.832** | **0.900** | **0.967** |

### HotpotQA (120 questions)

| system | F1 | EM | $/1k q | p50 s |
|---|---|---|---|---|
| graphwalk (graph only) | 0.182 | 0.075 | 0.17 | 0.7 |
| graphwalk + reader | 0.550 | 0.458 | 0.39 | 3.1 |
| text RAG | 0.755 | 0.675 | 0.09 | 2.2 |
| **text multi-step RAG** | **0.817** | **0.700** | 0.14 | 2.3 |

| system (F1) | bridge | comparison |
|---|---|---|
| graphwalk (graph only) | 0.334 | 0.030 |
| graphwalk + reader | 0.539 | 0.561 |
| text RAG | 0.652 | 0.857 |
| text multi-step RAG | **0.784** | **0.849** |

Ingestion (not included in the per-query costs above):

| | 2Wiki | HotpotQA |
|---|---|---|
| Paragraphs / chunks | 970 / 1,011 | 1,187 / 1,098 |
| Graph | 5,151 nodes, 5,972 edges | 6,640 nodes, 8,414 edges |
| Cost | ≈$0.60† | $0.78 |
| Time | 19 min (extractions cached) | 60 min (bound by 18 req/min extraction) |

†Measured from the extraction runs before the container restarts; see the
reproducibility notes.

## Why graphwalk loses here

1. **The walk, more than missing answers.** On 2Wiki the gold answer is usually in the
   graph. For compositional questions it is a node in 19/30 and an attribute in 5/30,
   and absent in 6/30. Yet graph-only graphwalk gets 20% EM there. Walks go wrong on
   noisy, free-form relation names:
   - `legally_acknowledged_as_son` instead of stepfather;
   - `succeeded` → `parent_of` → `spouse_of` → `daughter_of` for a paternal
     grandmother.

   They also stop early when the next link is missing. For "birthplace of the director
   of *Woman Without a Face*", the walk ends at the director: the birthplace was never
   extracted as an edge. M6 measured only about 60% of 2Wiki reasoning chains complete
   in the graph, and every miss is fatal to a walk.
2. **Comparisons and yes/no questions are out of reach for graph-only walking**
   (0.00–0.03 F1), as the design expected. The reader variant fixes that (0.53–0.67),
   but only when both entities and the compared fact made it into the graph.
3. **RAG reads the source sentences directly.** Anything extraction dropped or
   mangled is still in the paragraph. Multi-step RAG also retrieves by title, which
   suits Wikipedia-style corpora where every entity has its own paragraph.
4. **Granularity mismatches** hurt every system about equally, e.g. "Rupert III of
   the Palatinate" vs. gold "Rupert", and "Switzerland" vs. "Genève".

One-shot RAG is weak on 2Wiki bridge-comparison (0.03) and compositional questions,
where the second hop's paragraph is rarely in the top 5 for the question. Multi-step
retrieval fixes exactly that. On HotpotQA, one-shot RAG is already strong, because the
distractor setting keeps each pool small and on-topic.

## What this means for graphwalk

- **Where it wins:** curated, well-typed graphs with set-valued multi-hop answers
  (MetaQA, M5). There, walking beats both RAG variants and costs less.
- **Where it loses:** graphs extracted from text by a small LLM. Compared with text
  RAG, graphwalk is less accurate (−21 to −27 F1 against multi-step RAG), costs 2–3×
  more per query with the reader, and needs a costly ingestion step (about $0.0006 per
  paragraph, rate-limited). Its only advantage is graph-only latency (0.7–0.9 s vs.
  2–4 s), and at 0.18–0.25 F1 that advantage doesn't matter.
- The practical position is **"walk a real knowledge graph"**, not **"turn documents
  into a graph and walk it."** Text-to-graph extraction is the weak link. Improving it
  (a stronger extractor, a fixed relation schema, keeping literal facts as edges) is
  the only path that could change this result, and it would cost more per paragraph.

## Caveats

- Small samples (120 questions per dataset) and one seed. Per-type cells have 30–60
  questions each, so treat differences under about 10 points as noise.
- One small, cheap model (`gpt-6-luna`) does extraction and reading. A stronger
  extractor would help graphwalk more than RAG, since RAG never depends on extraction.
- graphwalk's preset was tuned on MetaQA, not on text-derived graphs, and nothing was
  tuned here. RAG wasn't tuned either (k = 5 throughout).
- The 2Wiki run hit OpenRouter's new-account limit of 20 requests/min per model: two
  multi-step RAG questions errored and scored 0. That changes its F1 by at most
  0.017.

## Reproducibility notes

- The container restarted three times during these runs (idle sessions are
  reclaimed). That led to three changes: per-system checkpoints (a re-run reuses
  finished systems), ingestion checkpoints (a partial graph plus its ledger, every 25
  windows), and an extraction cache written every 25 entries.
- Ingested graphs are cached as `~/.cache/graphwalk/graphs/<dataset>-<hash>.graph.json`,
  keyed by the pooled paragraphs and ingestion config.
- A first attempt failed every graphwalk traversal. The extracted graph has 337 node
  types, and the answer-type question offered all of them, which is over Jev's
  255-option limit. The engine now turns that question off in this case, and the
  harness offers only the 50 most common types.
