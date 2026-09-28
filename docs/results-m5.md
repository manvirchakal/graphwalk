# M5 results: traversal-only evals vs vector RAG

> **Update:** after a tuning round on MetaQA dev, the `-v2` presets reach test F1 0.965 / 0.990 / 0.887 (1/2/3-hop), at 14–39% more cost than v1. See "Tuning round" below. The tables that follow first are the untuned v1 results.

Run on 2026-09-28 against live APIs. Raw per-dataset summaries are in `results/*/summary.md`.
Reproduce with `graphwalk eval metaqa --hops 1 --hops 2 --hops 3 --n 200 --system ...` and
`graphwalk eval 2wiki --n 300 --linking gold|resolve --system ...` (see §6 "As built in M5" in
`design.md` for the setup).

**Setup, briefly.**
- **Decisions:** Jev `typesafe/jev-1.13` via OpenRouter.
- **Prefilter and RAG retrieval:** bge-small (fastembed, local CPU).
- **RAG reader:** `openai/gpt-6-luna` via OpenRouter, top-5 per-entity documents built from
  the *same* graph.
- **Sampling and concurrency:** seeded subsets (seed 0), concurrency 4.
- **Cost:** provider-reported, per 1,000 queries.
- **Latency:** per-question wall time in seconds.

With n = 200, a 95% interval on a score near 0.8 is about ±0.055, so differences under about
5 points are noise.

## MetaQA (answers are entity *sets*; F1 over the set)

| hops | system | F1 | EM | hits@1 | p50 s | p95 s | $/1k q | Jev calls/q |
|---|---|---|---|---|---|---|---|---|
| 1 | graphwalk relation (greedy) | **0.914** | 0.885 | 0.895 | **0.45** | 1.45 | **0.046** | 1.7 |
| 1 | graphwalk relation-beam | 0.909 | 0.880 | 0.890 | 2.49 | 7.95 | 0.161 | 3.6 |
| 1 | graphwalk entity greedy | 0.763 | 0.625 | 0.910 | 0.47 | 1.32 | 0.086 | 1.8 |
| 1 | graphwalk entity beam | 0.762 | 0.625 | 0.905 | 1.00 | 1.86 | 0.232 | 3.1 |
| 1 | vector RAG | 0.886 | 0.840 | 0.865 | 1.44 | 3.25 | 0.056 | — |
| 2 | graphwalk relation (greedy) | 0.790 | 0.790 | 0.790 | **0.75** | 2.02 | **0.081** | 2.6 |
| 2 | graphwalk relation-beam | **0.855** | 0.855 | 0.855 | 2.11 | 4.36 | 0.165 | 3.4 |
| 2 | graphwalk entity greedy | 0.581 | 0.370 | 0.915 | 0.62 | 1.21 | 0.118 | 2.6 |
| 2 | graphwalk entity beam | 0.608 | 0.385 | 0.955 | 0.93 | 1.72 | 0.266 | 3.3 |
| 2 | vector RAG | 0.271 | 0.210 | 0.340 | 2.03 | 4.10 | 0.085 | — |
| 3 | graphwalk relation (greedy) | 0.811 | 0.765 | 0.820 | **1.37** | 3.03 | **0.121** | 3.9 |
| 3 | graphwalk relation-beam | **0.893** | 0.820 | 0.900 | 2.44 | 4.48 | 0.197 | 4.0 |
| 3 | graphwalk entity greedy | 0.240 | 0.060 | 0.745 | 0.90 | 1.55 | 0.185 | 3.7 |
| 3 | graphwalk entity beam | 0.262 | 0.070 | 0.785 | 1.13 | 2.21 | 0.404 | 3.9 |
| 3 | vector RAG | 0.108 | 0.030 | 0.235 | 2.93 | 5.62 | 0.118 | — |

## 2WikiMultiHopQA (compositional + inference, 2-hop; pooled evidence graph; n = 300)

| entry nodes | system | F1 | EM | p50 s | p95 s | $/1k q | linking acc. |
|---|---|---|---|---|---|---|---|
| gold | graphwalk entity greedy | **0.919** | 0.890 | 0.40 | 0.67 | 0.048 | 1.00 |
| gold | graphwalk entity beam | **0.919** | 0.890 | 0.44 | 0.79 | 0.067 | 1.00 |
| gold | graphwalk relation (greedy) | 0.910 | 0.877 | **0.37** | 0.60 | **0.036** | 1.00 |
| — | vector RAG | 0.600 | 0.600 | 1.57 | 2.83 | 0.040 | — |
| resolved | graphwalk entity greedy | 0.574 | 0.573 | 0.25 | 0.64 | 0.049 | 0.55 |
| resolved | graphwalk entity beam | 0.784 | 0.767 | 0.42 | 0.80 | 0.071 | 0.55 |

## Tuning round (MetaQA dev → test once)

**Method.**
- Four knobs, all off in v1:
  1. `show_types`: node types in relation-mode prompts.
  2. `stop_style="literal"`: state the STOP condition first, literally.
  3. `relation_glosses`: one-line relation meanings.
  4. `answer_type`: a speculative "what type of thing does the query ask for?" question,
     batched into the first call. `hint` shows the prediction in later questions; `gate`
     withholds STOP from beams of the wrong type.
- Ablated on seeded **dev** subsets (n = 200 per hop count) with relation-greedy
  (`scripts/tune_metaqa_dev.py`; summaries in `results/tuning/`).
- The winner was frozen as the `-v2` presets and run **once** on the same test subsets
  as v1.

**Dev ablation (relation-greedy, F1).**

| variant | 1-hop | 2-hop | 3-hop |
|---|---|---|---|
| v1 | 0.928 | 0.810 | 0.815 |
| + types | 0.933 | 0.800 | 0.844 |
| + literal STOP | 0.928 | 0.890 | 0.848 |
| + glosses | 0.928 | 0.800 | 0.812 |
| + answer-type hint | 0.938 | 0.850 | 0.856 |
| + answer-type gate | 0.905 | 0.835 | 0.855 |
| **all four soft (chosen)** | **0.953** | **0.985** | **0.918** |
| soft + gate | 0.928 | 0.960 | 0.915 |

The knobs interact. Types and glosses do nothing alone, but together with literal STOP
and the answer-type hint they take 2-hop from 0.81 to 0.985. The hard gate never helps.
Relation-beam with the soft knobs scored 0.980 (2-hop) and 0.937 (3-hop) on dev, so
there was no reason to tune beam separately.

**Test, before → after (same 200 questions per hop count; RAG unchanged).**

| hops | system | F1 v1 → v2 | EM v1 → v2 | p50 s | $/1k q v1 → v2 | vector RAG F1 / $/1k |
|---|---|---|---|---|---|---|
| 1 | relation-greedy | 0.914 → **0.965** | 0.885 → 0.930 | 0.43 | 0.046 → 0.064 | 0.886 / 0.056 |
| 1 | relation-beam | 0.909 → 0.965 | 0.880 → 0.930 | 2.32 | 0.161 → 0.212 | |
| 2 | relation-greedy | 0.790 → **0.990** | 0.790 → 0.990 | 0.69 | 0.081 → 0.095 | 0.271 / 0.085 |
| 2 | relation-beam | 0.855 → 0.990 | 0.855 → 0.990 | 1.74 | 0.165 → 0.203 | |
| 3 | relation-greedy | 0.811 → **0.887** | 0.765 → 0.800 | 1.31 | 0.121 → 0.153 | 0.108 / 0.118 |
| 3 | relation-beam | 0.893 → 0.916 | 0.820 → 0.815 | 2.20 | 0.197 → 0.241 | |

On 2Wiki (gold entry), v2 is unchanged within noise:
- greedy: 0.919 → 0.918;
- relation: 0.910 → 0.901.

2Wiki's graph is untyped, so the answer-type hint switches itself off there, and 2Wiki's
misses were not STOP failures.

**Takeaways.**
- **The STOP fix is real and transfers from dev to test.** MetaQA 2-hop goes from 0.79 to
  0.99, and every hop count improves.
- **Beam is now mostly redundant.** After tuning, greedy matches beam at 1–2 hops and
  trails by about 3 points at 3 hops (within noise), at half the cost and 1/2–1/5 the
  latency. Greedy relation-v2 is the default recommendation.
- **Cost moved the wrong way for the pitch.** The extra prompt content and the
  answer-type question add 14–39% tokens. Greedy-v2 now costs more than RAG at 1 hop
  ($0.064 vs $0.056 per 1k) and 3 hops ($0.153 vs $0.118), and about the same at 2 hops.
  On MetaQA the accuracy gap now dwarfs the cost gap, but "cheaper" is not a claim this
  data supports.
- **Honesty notes.**
  - I wrote the relation glosses after reading v1 *test* error analysis (the
    `has_tags`/`has_genre` confusion). They describe the schema, not any question. Glosses
    alone did nothing on dev.
  - The answer-type hint needs a typed graph.
  - The answer-type question sometimes costs its own call when the first step needs no
    decision (1.97 vs 1.74 calls per query at 1 hop).

## What this says about the bet

**Accuracy on multi-hop: strongly for graphwalk, with a big caveat about the baseline.**
- Relation-mode graphwalk holds F1 0.81–0.91 from 1 to 3 hops. Single-shot vector RAG
  collapses from 0.89 to 0.27 to 0.11.
- That baseline is deliberately simple: one retrieval, k = 5, entity documents. It is exactly
  the kind of RAG that is known to fail at multi-hop retrieval. An iterative or agentic RAG
  loop, or a larger k and a bigger model, would close part of the gap at higher cost and
  latency. The honest claim is "beats naive RAG on the same KG", not "beats graph RAG".
  GraphRAG/LightRAG comparisons are still to come.

**Latency: a real win, about 2–3x.**
- At p50, greedy graphwalk takes 0.4–1.4 s vs 1.4–2.9 s for RAG.
- Beam loses most of that edge (2.1–2.5 s p50 in relation mode).

**Cost: roughly at parity, not the big win the pitch implies.**
- Jev is only about 2.4x cheaper per input token than the cheapest current small LLM. A walk
  makes 2–4 calls of about 500–1,000 tokens each, while RAG makes one call of about 450
  tokens.
- Greedy relation mode is 10–20% cheaper than RAG on MetaQA and at parity on 2Wiki.
  Everything else in graphwalk costs more than RAG.
- In absolute terms all of it is tiny: $0.04–0.40 per 1,000 queries.

**Beam vs greedy.** Beam doesn't help at 1 hop and helps 6–8 points at 2–3 hops in relation
mode, for about 2x cost and latency. With resolved entry nodes it also recovers much of the
entity-linking damage, because each candidate start becomes its own beam.

**Entity vs relation hops.** On set-answer datasets, entity hops get the first answer right
(hits@1 0.75–0.96) but can't return the whole set (EM ≤ 0.63). Relation mode, added in Q3, is
what makes MetaQA work.

## Failure analysis

- **STOP is the weak decision.**
  - All 42 relation-greedy misses on MetaQA 2-hop reached the correct answer set and then did
    not stop. For example, at `{1995}` the walk followed `release_year` back out to 500
    films.
  - No correct answers came from walks cut off at `max_depth`, so the depth limit is not
    doing hidden work.
  - Likely fixes: show node types in relation-mode prompts, and phrase STOP more literally.
    These should be tuned on MetaQA's dev split and then re-measured on test.
- **Relation naming.** "What words describe X" questions route to `has_genre` instead of
  `has_tags`: the bare relation name doesn't say "keywords". Relation descriptions in
  options would likely fix this.
- **Dataset ambiguity.** MetaQA merges same-titled films into one node (e.g. several "Les
  Misérables"), and gold lists one version's answers. 18 of the 23 relation errors at
  1 hop are shared with RAG.
- **Entity linking.** Name matching finds exactly the gold start only 55% of the time on
  2Wiki. Titles in questions differ from graph names, and extra mentions also match. This
  is now the largest loss end to end.

## Caveats

- 2Wiki's pooled evidence graph is sparse (average degree ≈ 1.6) and every question is
  2 hops, so its gold-linking numbers flatter traversal. MetaQA, with 4,000-edge hubs, is the
  meaningful test.
- Latency is measured at concurrency 4 on 4 CPUs: the local prefilter and store run in
  Python and share the CPU. RAG latency excludes client-side rate-limit waits, because
  OpenRouter caps new accounts at 20 LLM requests/min.
- Prompts are v1 and untuned. Nothing was tuned on these test subsets.
- The seeded subsets are 200 (MetaQA) and 300 (2Wiki) questions, not the full test sets.
