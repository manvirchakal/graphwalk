# Phase 4 results: experiments for the paper (in progress)

Status on 2026-09-30: E1 and E2 are complete. E3, E4 and E5 are partly run. They
stopped when the OpenRouter account ran out of credits (see the last section). Every
number below comes from a committed run and can be regenerated with the
`scripts/paper/` script named in its heading. Intervals are 95% percentile bootstraps
over questions (2,000 resamples). The paired Δ columns resample per-question
differences.

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

## E2: Does an LLM writing the graph query beat walking? (`table_curated.py`)

An LLM (the same `gpt-6-luna`) sees MetaQA's schema: 9 relations, their endpoint
types, and one-line glosses. It also sees the start entity and the question. It writes
a relation path, which code executes the way a relation-mode walk does (all targets
per hop, visited nodes excluded). The questions are the same 200 per hop count (seed
0) as graphwalk's.

| MetaQA | graphwalk (Jev) F1 | LLM writes the path F1 | Δ [95% CI] | $/1k q (Jev → path) | p50 s (Jev → path) |
|---|---|---|---|---|---|
| 1-hop | 0.965 | 0.980 | +0.015 [+0.000, +0.035] | 0.064 → 0.048 | 0.56 → 1.23 |
| 2-hop | 0.980 | **1.000** | +0.020 [+0.005, +0.040] | 0.096 → 0.061 | 0.88 → 1.84 |
| 3-hop | 0.897 | **0.961** | **+0.065 [+0.033, +0.100]** | 0.154 → 0.080 | 1.63 → 2.31 |

Letting the LLM retry an invalid or empty path changed nothing (0.952 at 3-hop).

**What it says.** On a small, clean, curated schema, the strongest practical
competitor wins. Planning the whole path up front from the schema is more accurate
than choosing one hop at a time, and costs about half as much per query. graphwalk
keeps only latency (1.4–2× faster). The M5 claim "graphwalk beats RAG on curated
graphs" still holds against RAG: the vector and multi-step RAG rows in the table
stand. It does not hold against text-to-query. As the roadmap planned, the claim
narrows to where writing the query should break: large, noisy, or fuzzy schemas.

E2b tests that on the M7 graphs extracted from text (1,447 and 2,545 free-form relation
types; `scripts/eval/query_writer_text.py`). It has not run yet.

## E3: Jev vs an LLM as the decider (`table_curated.py`, `figure_calibration.py`)

The traversal is the same and only the decision backend changes. The LLM decider asks
`gpt-6-luna` to score every option 0–100 and normalizes the scores. Only MetaQA 1-hop
finished before credits ran out:

| MetaQA 1-hop | F1 | $/1k q | ECE | Brier | AUROC | EM, most confident half |
|---|---|---|---|---|---|---|
| Jev | 0.965 | 0.064 | 0.038 | 0.064 | 0.635 | 0.950 |
| LLM decider | 0.980 | 0.192 | 0.045 | 0.053 | 0.658 | 0.970 |

At 1 hop the LLM decides as well as Jev and is about as well calibrated, at 3× the
cost. Its measured latency (p50 26 s) was taken while other jobs shared the same
model's 20-requests/min limit, and may include provider-side retries. It is not
comparable until re-measured alone. 2-hop, 3-hop and the 2Wiki gold graph are still
to run; 3-hop is the informative one.

Jev's own calibration on all four curated sets (seed 0): the most confident half of
answers is 93–100% right, with AUROC 0.94–0.97 at 2 and 3 hops. So Jev's confidence
is usable for abstaining or escalating, although its level is overconfident at 3 hops
(ECE 0.14).

## E4: Seeds and confidence intervals

- **Done.**
  - graphwalk with Jev on three seeded MetaQA and 2Wiki-gold subsets. F1 mean ± sd:
    - MetaQA 1-hop: 0.955 ± 0.017;
    - MetaQA 2-hop: 0.963 ± 0.018;
    - MetaQA 3-hop: 0.887 ± 0.029;
    - 2Wiki gold: 0.912 ± 0.007.
  - Seed 0 reproduces M5 within 1 point.
  - Bootstrap CIs and paired differences for every headline table, from the committed
    per-question `scores.jsonl`.
- **Text-derived graphs** (`table_text_qa.py`, `table_fanout.py`). The M7 gaps are
  real. Paired F1 against multi-step RAG:
  - graphwalk + reader over source paragraphs: −0.100 [−0.172, −0.027] on 2Wiki and
    −0.149 [−0.225, −0.081] on HotpotQA;
  - on FanOutQA, loose accuracy −0.188 [−0.300, −0.085].
- **Not run yet.**
  - Extra seeds for the LLM-written path, and for RAG on MetaQA 1-hop.
  - A held-out 2Wiki text seed. It would also re-check E1's hybrid default on fresh
    questions.

## E5: Would a stronger extractor flip the text results?

A 60-question 2Wiki slice (516 paragraphs).

| extractor | ingestion cost | nodes | edges | entities found | gold triples recalled | reasoning chains complete |
|---|---|---|---|---|---|---|
| `gpt-6-luna` (M7's) | $0.08 (extractions cached) | 2,706 | 3,038 | 0.780 | 0.702 | 0.450 |
| `gpt-6.1-sol` | stopped at about a quarter of the slice | | | | | |

A 4-question probe cost $0.15 with `gpt-6.1-sol` and recalled more gold triples (0.70
vs 0.60), with the same chain completeness (0.50). That is too small to mean anything.
The full slice is about $2 more.

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

## Budget: what happened

The roadmap's "about $41" was the API key's spending limit ($50 cap, $9.97 used). The
account itself had $10 of credits. With $0.03 left, OpenRouter began refusing requests
(`402: requires more credits`) at about 13:45 UTC. `gpt-6-luna` calls still fit for a
while, which is why the stall first showed up only in E5's larger `gpt-6.1-sol`
requests. All jobs were then stopped. No committed result contains a failed request:
every run above is `ok` on all questions. Finishing the plan needs about $3–4 of
credits:

| remaining | est. cost |
|---|---|
| E3: LLM decider on MetaQA 2/3-hop and 2Wiki gold | ~$0.4 |
| E5: finish the `gpt-6.1-sol` slice, then E1 + QA on both graphs | ~$2.0 |
| E2b: LLM-written paths on the two text graphs | ~$0.7 |
| E4: extra seeds (LLM path, RAG 1-hop) | ~$0.2 |
