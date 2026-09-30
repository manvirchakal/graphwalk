# Phase 4 results: experiments for the paper

Status on 2026-09-30: E1–E7 and E2b are complete. Every number below comes from a
committed run and can be regenerated with the `scripts/paper/` script named in its
heading. Intervals are 95% percentile bootstraps over questions (2,000 resamples). The
paired Δ columns resample per-question differences.

## Where graphwalk helps (summary)

| setting | verdict | evidence |
|---|---|---|
| QA over text, against multi-step RAG | **loses**, by 10–19 points | E4 (CIs exclude 0), M7 |
| …with a 20× pricier extractor | still loses; no gain from extraction | E5 |
| small, clean curated schema (MetaQA) | **loses** to an LLM writing the query (−0.07 F1 at 3 hops); beats RAG | E2, E4 |
| large, noisy schema (text-derived graphs) | **wins** against an LLM writing the query: 3–5× the F1 at 1/20–1/35 the cost | E2b |
| one-shot evidence retrieval, entity chains (2Wiki) | **wins**: hybrid +18 points recall over dense | E1 |
| retrieval elsewhere (HotpotQA, FanOutQA) | ties dense, or costs about 5 points | E1 |
| helping multi-step RAG | no reliable accuracy gain; 16–21% fewer LLM rounds, but the Jev calls cost more than the LLM savings | E7, E1 analysis |
| decider: Jev vs an LLM | the LLM is as accurate or better (+0.06 at 3 hops); Jev is 3–4× cheaper and 7–10× faster, and its confidence tells right from wrong on the harder walks (AUROC 0.94–0.97 vs 0.54–0.64) | E3 |

## E1: Is the graph a good retriever? (`table_evidence.py`)

Each method returns documents from the same pooled text: M7's 2Wiki and HotpotQA
paragraphs, or FanOutQA's evidence pages. Each list is scored against the gold
evidence: supporting-fact paragraphs, or each question's evidence pages. No answers
are generated.

| method | 2Wiki recall@5 | HotpotQA recall@5 | FanOutQA recall@10 |
|---|---|---|---|
| dense (bge-small, titled chunks) | 0.758 [0.717, 0.800] | 0.921 [0.883, 0.954] | 0.884 [0.831, 0.930] |
| control: titles named in the question, then dense | 0.758 | 0.925 | 0.879 |
| graph `locate` (name linking, as shipped) | 0.854 [0.808, 0.894] | 0.713 [0.650, 0.771] | 0.436 [0.347, 0.526] |
| graph `locate` (Jev linking) | 0.892 [0.856, 0.923] | 0.758 [0.700, 0.812] | 0.465 [0.376, 0.553] |
| hybrid (name linking) | 0.927 [0.896, 0.954] | 0.938 [0.904, 0.967] | 0.827 [0.762, 0.888] |
| hybrid (Jev linking) | **0.935** [0.906, 0.960] | 0.929 [0.896, 0.963] | 0.837 [0.778, 0.894] |
| multi-step RAG's retrievals, in order (M7) | 0.740 [0.696, 0.783] | 0.921 [0.883, 0.954] | 0.770 [0.686, 0.853] |

Paired hybrid (Jev linking) − dense:
- 2Wiki recall@5: **+0.177 [+0.135, +0.219]**;
- HotpotQA recall@5: +0.008 [−0.017, +0.033];
- FanOutQA recall@10: −0.047 [−0.098, +0.001] (at recall@5, −0.060 [−0.118, −0.005]).

**What it says.**
- **Where the graph helps: 2Wiki's chains.** Graph `locate` alone beats dense by 10–13
  points, and hybrid by 18, with complete evidence found for 82% of questions against
  48%. The gain is on compositional, inference and bridge-comparison questions: the
  walk reaches the second-hop paragraph, which the question's embedding doesn't point
  to.
- **It isn't title matching.** The control ranks paragraphs whose title the question
  names, then fills with dense. It adds nothing over dense.
- **Where the graph hurts.**
  - On HotpotQA, graph alone loses to dense by 16–21 points. The distractor pools are
    small and on-topic, so dense is already near its ceiling.
  - On FanOutQA, graph alone finds under half the evidence pages. The walks reach
    only part of each set, and extraction dropped much of it.
- **Default for `locate`: hybrid.** It is the only method that is never far behind:
  best on 2Wiki, tied on HotpotQA, about 5 points under dense on FanOutQA. Graph-only
  `locate` should not be the default.
- **Caveat.** The choice was made on the same seed-0 samples reported here. No
  separate tuning split existed for E1.
- **Retrieval isn't the bottleneck for multi-step RAG.** Over everything it read (6
  paragraphs on average), its recall on 2Wiki is 0.985, and 0.958 on HotpotQA. So the
  M7 QA gap between graphwalk and multi-step RAG is about reading and comparing, not
  about finding the evidence.

**Can the graph help multi-step RAG?** Hardly, on these benchmarks. Of multi-step
RAG's wrong answers (28 on 2Wiki, 17 on HotpotQA, 9 on FanOutQA), only 2, 5 and 3
missed gold evidence. Hybrid `locate` held the missing evidence for 2, 2 and 2 of those.
Its failures are reasoning and answer granularity, not retrieval. The pooled corpora are
small and every entity has a titled paragraph it can look up. Settings where retrieval
should fail (no title lookup, tight step budgets, large corpora, longer chains) are
tested in E7.

## E2: Does an LLM writing the graph query beat walking? (`table_curated.py`)

An LLM (the same `gpt-6-luna`) sees MetaQA's schema: 9 relations, their endpoint
types, and one-line glosses. It also sees the start entity and the question. It writes
a relation path, which code executes the way a relation-mode walk does (all targets per
hop, visited nodes excluded). The questions are the same 200 per hop count as
graphwalk's.

| MetaQA | graphwalk (Jev), F1 over 3 seeds | LLM writes the path (2 retries), F1 over 3 seeds | Δ on seed 0 [95% CI] | $/1k q (Jev → path) |
|---|---|---|---|---|
| 1-hop | 0.955 ± 0.017 | 0.980 ± 0.001 | +0.015 [+0.000, +0.035] | 0.064 → 0.049 |
| 2-hop | 0.963 ± 0.018 | 0.998 ± 0.003 | +0.020 [+0.005, +0.040] | 0.096 → 0.061 |
| 3-hop | 0.887 ± 0.029 | **0.955 ± 0.005** | **+0.055 [+0.027, +0.089]** | 0.154 → 0.091 |

Without retries the path scores 0.961 at 3 hops. On a small, clean schema, planning the
whole path from the schema beats choosing one hop at a time, costs less, and varies
less across seeds. graphwalk keeps only latency (1.4–2× faster). Replies left empty
by the 512-token output cap (the reasoning model used it all) were 6–7 of about 210
attempts per 3-hop run and none at 1–2 hops. They count against the LLM, so if
anything they understate it.

### E2b: the same on large, noisy schemas (`scripts/eval/query_writer_text.py`)

The M7 graphs extracted from text, with each relation's most common endpoint types
and edge count as the schema. The start entity is linked by the same Jev linker.
Output cap 4,096 tokens: a first 2Wiki run at 512 left a third of the replies empty
and was discarded.

| text-derived graph | relation types | graphwalk (graph only), F1 | LLM writes the path, F1 | Δ walk − path [95% CI] | $/1k q (walk vs path) |
|---|---|---|---|---|---|
| 2Wiki | 1,447 | **0.254** | 0.079 | **+0.176 [+0.107, +0.254]** | 0.18 vs 3.63 |
| HotpotQA | 2,507 | **0.182** | 0.040 | **+0.142 [+0.086, +0.199]** | 0.17 vs 6.09 |

The LLM writes plausible paths (`spouse_of` → `son_of`), but on an extracted graph the
relation usually exists under another name or in the other direction. 81% of its 2Wiki
attempts returned nothing. Walking chooses among the edges that exist. So the curated-
graph claim narrows to schemas too large or noisy to plan over: there, walking wins on
accuracy and costs 1/20th as much. Both are weak in absolute terms on these graphs;
multi-step RAG over the text reaches 0.75–0.82.

## E3: Jev vs an LLM as the decider (`table_curated.py`, `figure_calibration.py`)

Same traversal, only the decision backend swapped. The LLM decider asks `gpt-6-luna`
to score every option 0–100 and normalizes the scores. Seed 0.

| dataset | F1, Jev → LLM | Δ [95% CI] | $/1k q, Jev → LLM | decision s (p50), Jev → LLM | AUROC, Jev vs LLM | EM of the most confident half, Jev vs LLM |
|---|---|---|---|---|---|---|
| MetaQA 1-hop | 0.965 → 0.980 | +0.015 [+0.000, +0.035] | 0.064 → 0.192 | 0.55 → 4.8 | 0.64 vs 0.66 | 0.95 vs 0.97 |
| MetaQA 2-hop | 0.980 → 0.985 | +0.005 [−0.020, +0.030] | 0.096 → 0.327 | 0.84 → 7.6 | **0.94 vs 0.54** | 1.00 vs 0.99 |
| MetaQA 3-hop | 0.897 → **0.956** | **+0.060 [+0.027, +0.097]** | 0.154 → 0.548 | 1.3 → 10.8 | **0.97 vs 0.64** | **0.99 vs 0.86** |
| 2Wiki gold graph | 0.915 → **0.941** | **+0.026 [+0.010, +0.045]** | 0.048 → 0.164 | 0.46 → 3.1 | 0.67 vs 0.69 | 0.93 vs 0.96 |

- **The calibrated classifier is not what makes walks accurate.** A small LLM decides
  as well or better.
- **What Jev buys is cost, speed, and informative confidence.** It is 3–4× cheaper and
  7–10× faster per decision. On the harder walks, its confidence separates right from
  wrong answers, which the LLM's does not.
- **The paper's claim is "cheap, fast, and knows when it's wrong"**, not "more
  accurate". Wall latency for the LLM decider (p50 19–53 s) mostly waits for
  rate-limit slots (20 requests/min); the decision column is time spent in calls.

## E4: Seeds and confidence intervals

- **graphwalk (Jev), three seeded subsets** (F1 mean ± sd):
  - MetaQA 1-hop 0.955 ± 0.017, 2-hop 0.963 ± 0.018, 3-hop 0.887 ± 0.029.
  - 2Wiki gold 0.912 ± 0.007.
  - Seed 0 reproduces M5 within 1 point.
- **LLM-written path, three seeds:** 0.980 ± 0.001, 0.998 ± 0.003, 0.955 ± 0.005.
- **Vector RAG on MetaQA 1-hop, three seeds:** 0.881 ± 0.020. At 2–3 hops the seed-0
  gaps to graphwalk (−0.71, −0.79) make more seeds pointless.
- **Text-derived graphs** (`table_text_qa.py`, `table_fanout.py`), paired against
  multi-step RAG:
  - graphwalk + reader over source paragraphs: −0.100 [−0.172, −0.027] (2Wiki) and
    −0.149 [−0.225, −0.081] (HotpotQA);
  - FanOutQA loose accuracy −0.188 [−0.300, −0.085].

  No extra text seeds were run: each needs a new ingestion, and every gap already
  excludes 0.

## E5: Would a stronger extractor flip the text results?

A 60-question 2Wiki slice (the first 15 of each type from the M7 sample; 516
paragraphs), ingested twice with the same pipeline. Only the extraction model differs
(`results/20260930T181133Z-2wiki-extractor/`).

| | `gpt-6-luna` | `gpt-6.1-sol` (20× the price) | Δ [95% CI] |
|---|---|---|---|
| gold triples recalled / chains complete | 0.702 / 0.450 | 0.715 / 0.500 | |
| hybrid (Jev linking) recall@5 | 0.958 | 0.950 | |
| graphwalk, graph only, F1 | 0.335 | 0.281 | −0.054 [−0.128, +0.014] |
| graphwalk + reader (extracted facts), F1 | 0.563 | 0.521 | −0.042 [−0.131, +0.050] |
| graphwalk + reader (source), F1 | 0.630 | 0.607 | −0.023 [−0.077, +0.031] |
| multi-step RAG, F1 (no graph) | 0.764 | 0.764 | |
| ingestion cost | ~$0.08 (cached extractions) | $2.75 | |

No. The stronger extractor recovers slightly more gold facts, but retrieval and QA do
not improve (if anything, they slip, within noise), and the best graphwalk variant
stays behind multi-step RAG: −0.158 [−0.275, −0.039]. What loses the text benchmarks
is what the graph cannot represent (comparisons, order, qualifiers), not extraction
errors. The M7 diagnosis "extraction is the bottleneck" does not survive.

## E7: Does the graph help multi-step RAG? (`scripts/eval/rag_addon.py`)

Multi-step RAG on the M7 2Wiki sample (120 questions). With the add-on, its first
retrieval (top 5 paragraphs) comes from graphwalk's hybrid `locate` (Jev linking; E1's
checkpointed results), not from the question's embedding. Everything else is the same.
The stressors are title lookup on or off, and 5 or 2 LLM rounds
(`results/20260930T205134Z-2wiki-ragaddon/`).

| title lookup | rounds | F1 without → with add-on | Δ [95% CI] | LLM calls/q | LLM $/1k q |
|---|---|---|---|---|---|
| on | 5 | 0.743 → 0.757 | +0.013 [−0.015, +0.043] | 1.77 → 1.40 | 0.278 → 0.222 |
| on | 2 | 0.732 → **0.782** | **+0.050 [+0.003, +0.100]** | 1.57 → 1.31 | 0.219 → 0.166 |
| off | 5 | 0.780 → 0.763 | −0.017 [−0.062, +0.026] | 1.78 → 1.47 | 0.280 → 0.252 |
| off | 2 | 0.742 → 0.747 | +0.005 [−0.041, +0.052] | 1.59 → 1.31 | 0.230 → 0.197 |

The add-on itself costs $0.245 per 1k questions in Jev calls, and 0.84 s at the median.

- **Accuracy: no reliable gain.** One of four comparisons clears zero, barely (+0.05
  at 2 rounds with title lookup). Its twin without title lookup shows +0.005. With four
  comparisons at 95%, one borderline hit is about what chance would give. The honest
  claim is "no accuracy loss, possibly a small gain under tight round budgets"; that
  needs more questions to confirm.
- **Rounds: a consistent drop of 16–21%.** Every setting answers in fewer LLM rounds,
  because the graph's first retrieval already holds the second-hop paragraph more
  often. But in dollars, the LLM savings ($0.03–0.06 per 1k) are smaller than the
  add-on's Jev cost ($0.245 per 1k). The add-on trades cheap LLM calls for pricier
  classifier calls. It is not a cost saving at `gpt-6-luna` prices; it would be with a
  pricier reader model.
- **The "no title lookup" stressor doesn't work on 2Wiki.** Paragraphs are embedded with
  their titles, so plain embedding search already finds entity pages (0.780 without
  title lookup vs 0.743 with). Testing corpora without one page per entity needs such a
  corpus, not a switch.

**Verdict.** On these benchmarks, graphwalk is not a meaningful add-on to multi-step RAG.
It cuts LLM rounds but not accuracy or dollars. The settings where it might matter (a
large corpus, entities not titled per page, longer chains, an expensive reader model)
remain untested.

## E6: Cost and latency, ingestion amortized (`table_cost.py`)

- **On 2Wiki and HotpotQA, graphwalk never breaks even.** Its per-query cost is
  higher than multi-step RAG's ($0.34–0.37 vs $0.14–0.28 per 1k questions), and it
  also pays for ingestion ($0.60–0.78 per pool).
- **On FanOutQA, its per-query cost is lower** ($2.64 vs $6.04 per 1k). Ingestion
  (about $2) pays for itself after about 600 questions, but at 19 points lower
  accuracy.
- **On curated graphs, there is no ingestion.** graphwalk costs $0.05–0.15 per 1k
  questions. That is below multi-step RAG, but above the LLM-written path at 2 and 3
  hops.

## Budget

The roadmap's "about $41" was the first API key's spending cap. That account held $10
of credits, which ran out mid-phase: requests were refused with `402` from about 13:45
UTC. All jobs were stopped, and no committed run contains a refused request. The rest
ran on a second key. `scripts/eval/credits.py` now checks the account balance, not a
key's limit, before each job. Phase 4 spent about $5.90 on API calls in total ($1.08 on the first key, $4.83 on the second); E5's
`gpt-6.1-sol` extraction ($2.75) was the largest item.
