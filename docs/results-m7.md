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
  `results/20260929T171954Z-hotpotqa-text/`. Script: `scripts/eval/text_qa.py`.

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

## Follow-up: walk, then read the source; and a fixed relation schema

Two changes aimed at the failure analysis above, evaluated on the **same** 120 + 120
questions, pooled paragraphs, and cached graphs
(`results/20260929T180802Z-2wiki-walkread/`,
`results/20260929T182414Z-hotpotqa-walkread/`; script `scripts/eval/walk_read.py`).

- **Walk, then read the source.** The reader no longer reads extracted facts. The walk
  chooses which paragraphs to read, and the reader reads their original text. It takes
  up to 5 paragraphs, the same budget one-shot RAG gets, in this order: each question
  entity's own paragraph, then the paragraphs supporting each walked edge (from edge
  provenance), then walked entities' own paragraphs.
- **A fixed relation schema** (`ingest/normalize.py`):
  - 38 general relations, chosen a priori from common Wikidata properties, each
    offered as-is or reversed.
  - Every distinct extracted relation type is mapped onto the schema with a batched
    Jev choice question, shown three example edges.
  - Entity-valued attributes become edges.
  - Results: 2Wiki mapped 501 of 1,476 types, rewriting 3,246 edges and adding 94
    attribute edges, in 53 Jev calls for $0.13. HotpotQA mapped 721 of 2,545 types
    (3,545 edges, 124 attribute edges) in 91 calls for $0.22.

| F1 (EM) | 2Wiki | HotpotQA |
|---|---|---|
| graphwalk + reader, extracted facts (M7) | 0.539 (0.492) | 0.550 (0.458) |
| **graphwalk + reader, source paragraphs** | **0.650 (0.592)** | **0.668 (0.567)** |
| … on the schema-normalized graph | 0.662 (0.600) | 0.665 (0.558) |
| graphwalk graph-only (M7) | 0.254 (0.167) | 0.182 (0.075) |
| … on the schema-normalized graph | 0.251 (0.167) | 0.186 (0.075) |
| text RAG (M7) | 0.368 (0.325) | 0.755 (0.675) |
| text multi-step RAG (M7) | 0.749 (0.683) | 0.817 (0.700) |

| 2Wiki F1 by type | compositional | inference | comparison | bridge-comparison |
|---|---|---|---|---|
| reader, source paragraphs | 0.328 | 0.838 | 0.700 | 0.733 |
| text multi-step RAG | 0.299 | 0.832 | 0.900 | 0.967 |

| HotpotQA F1 by type | bridge | comparison |
|---|---|---|
| reader, source paragraphs | 0.667 | 0.668 |
| text RAG | 0.652 | 0.857 |
| text multi-step RAG | 0.784 | 0.849 |

**What changed**

- **Reading source text is a real gain: +11 to +12 F1 on both datasets** at the same
  cost ($0.34–0.37 per 1k queries). It confirms the diagnosis: extraction, not the
  walk's choice of entities, was losing the answers. On 2Wiki's single-chain
  questions the walk now picks paragraphs as well as multi-step RAG does (inference
  0.838 vs. 0.832; compositional 0.328 vs. 0.299).
- **The relation schema did nothing** (+1.2 / −0.3 F1, within noise), for graph-only
  walks or the reader. Only a third of relation types mapped, although those cover
  half the edges, and walks on 2Wiki were rarely lost *only* because of relation
  names. Missing links and wrong entry points matter more. It isn't worth its
  complexity here.
- **Still behind multi-step RAG, by 9–15 F1.** The gap is almost entirely comparison
  questions (−20 to −23 F1 on both datasets) plus HotpotQA bridge questions (−12). For
  comparisons, both entities are usually linked, but the 5 walk-chosen paragraphs
  miss the compared fact more often than retrieval does. Multi-step RAG searches
  again for whatever is still missing; a walk can't. Parity was not reached.

**Cost of the follow-up:** $0.56 of API usage (key usage $5.54 → $6.10), including the
two normalization passes.

**Where this leaves graphwalk on text:** the graph is useful as a *navigator*,
choosing which source paragraphs to read, and matches multi-step RAG on single-chain
questions. It doesn't replace retrieval for comparisons or loosely connected
questions. The obvious next experiment is a hybrid: combine the walk's paragraphs with
retrieved ones, and let the reader search again when the walk runs out. At that point
graphwalk is a component of a RAG system rather than an alternative to it, and it
should be evaluated as one.

## Step 2: FanOutQA, where walking should win on text

FanOutQA (dev set) asks fan-out questions over Wikipedia whose answers are sets or
item -> value maps, e.g. "the batting hand of each of the first five picks in the 1998
MLB draft". If walking a text-derived graph beats retrieval anywhere, it should be
here: a relation-mode hop takes *all* targets of a relation at once, which is what
beat RAG on MetaQA.

**Setup** (`scripts/eval/fanout.py`; run `results/20260929T211957Z-fanoutqa/`):

- 40 dev questions (seed 0) and their 233 evidence pages, from a pinned community
  mirror of the FanOutQA corpus. Wikipedia's API now rate-limits bulk fetches: 11 of
  233 pages arrived in 9 minutes. Page ids match all 1,562 dev evidence pages;
  revisions can't be checked.
- Pages are truncated to 8,000 characters. That keeps 89.4% of reference strings,
  versus 92.4% for full pages, whose median length is 98k characters. Every system
  reads the same truncated pages, pooled into one graph (1,220 chunks) and one
  page-level RAG index.
- Systems, all on `gpt-6-luna` with a list-style answer prompt:
  - graphwalk + reader over extracted facts, and over source pages (up to 20), both
    with relation-mode walks (`relation-v2`);
  - one-shot RAG (top 10 pages);
  - multi-step RAG (up to 20 pages, title search).
- Metric: FanOutQA's accuracy (the share of reference strings, keys and values, found
  in the answer; strict = all found), without its lemmatizer.

| system | loose acc | strict acc | $/1k q | p50 s |
|---|---|---|---|---|
| graphwalk + reader, extracted facts | 0.364 | 0.050 | 1.09 | 6.0 |
| graphwalk + reader, source pages | 0.482 | 0.050 | 2.64 | 6.4 |
| text RAG (top 10 pages) | 0.589 | 0.200 | 2.95 | 4.7 |
| **text multi-step RAG** | **0.670** | **0.300** | 6.04 | 6.5 |
| ceiling (strings present in the truncated pages) | 0.894 | | | |

Head to head, per question: multi-step RAG is better on 20, tied on 15, and worse on 5
than graphwalk + reader over source pages.

**Why walking still loses here**

- **The walks do find sets.** For example, Supreme Court ← `associate_justice_of` ←
  all the justices; IBM Award ← `won` ← nine players; Solar System ← `planet_of` ←
  all planets.
- **But most FanOutQA sets are ranked or time-bound:** "the most recent four
  justices", "the top 5 highest-grossing films", "the 5 most recent Olympics". The
  extracted graph holds the set but not the order (ranks, dates), so the walk returns
  a superset and the reader has to guess. Multi-step RAG reads the ordered table on
  the list page, then fetches each item's page by title. That is exactly the fan-out
  strategy, and on Wikipedia every item has its own page.
- **Entry linking and noisy walks still cost questions.** The Olympics question started
  at "Low Countries" and scored 0; another walk meandered Avatar → `music_by` → James
  Horner → Titanic instead of reading the ranked list.
- **Cost is graphwalk's one advantage:** about 44% of multi-step RAG's per-query cost
  over source pages, 18% over facts. That's before ingestion, which cost about $2 for
  233 pages. Multi-step RAG needs none.

**Cost of step 2:** $2.69 of API usage (key usage $6.20 → $8.89). That's above my
$1.90 estimate, because the pages split into 1,220 chunks rather than about 920.

## Verdict on graphwalk over text

Across 2Wiki, HotpotQA and FanOutQA, walking a graph built from text by a small
extractor does not beat multi-step RAG over the same text. It loses 9–15 F1 on
single-answer questions (after the walk-then-read-source fix) and 19 points of loose
accuracy on fan-out questions. The test designed to favor walking (set-valued
answers) did not change that. The failure is not the walk's decision-making: walks
reach the right sets. The problem is that text-to-graph extraction drops exactly what
these questions need (orderings, dates, qualifiers), while retrieval keeps the source.

graphwalk's demonstrated advantage remains curated knowledge graphs (MetaQA, M5), and
that claim still needs the strongest competitor there: an LLM writing a graph query.
