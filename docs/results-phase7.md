# Phase 7 results: large curated graphs and escalation

Plan and kill criterion: `roadmap.md`, Phase 7. Each section names the script that
regenerates its table.

## A1: Escalation (`scripts/paper/table_escalation.py`)

Re-walk with the LLM decider when Jev's best-path confidence (`exp(score)`) is below a
threshold. Simulated offline from the E3 runs (seed 0): both deciders answered the same
questions, and an escalated question pays for both walks. No new API calls.

| dataset | system | n | escalated | F1 | F1 vs LLM decider | EM | EM vs LLM decider | $ / 1k queries |
|---|---|---|---|---|---|---|---|---|
| MetaQA 1-hop | Jev only | 200 | 0% | 0.965 [0.943, 0.984] | -0.015 [-0.035, +0.000] | 0.930 [0.890, 0.965] | -0.015 [-0.035, +0.000] | 0.064 |
| MetaQA 1-hop | LLM decider only | 200 | 100% | 0.980 [0.965, 0.991] | — | 0.945 [0.910, 0.975] | — | 0.192 |
| MetaQA 1-hop | escalate below 0.8 | 200 | 3% | 0.975 [0.957, 0.990] | -0.005 [-0.015, +0.000] | 0.940 [0.905, 0.970] | -0.005 [-0.015, +0.000] | 0.070 |
| MetaQA 1-hop | escalate below 0.9 | 200 | 13% | 0.980 [0.965, 0.991] | +0.000 [+0.000, +0.000] | 0.945 [0.910, 0.975] | +0.000 [+0.000, +0.000] | 0.090 |
| MetaQA 1-hop | escalate below 0.95 | 200 | 22% | 0.980 [0.965, 0.991] | +0.000 [+0.000, +0.000] | 0.945 [0.910, 0.975] | +0.000 [+0.000, +0.000] | 0.110 |
| MetaQA 1-hop | escalate below 0.99 | 200 | 38% | 0.980 [0.965, 0.991] | +0.000 [+0.000, +0.000] | 0.945 [0.910, 0.975] | +0.000 [+0.000, +0.000] | 0.142 |
| MetaQA 2-hop | Jev only | 200 | 0% | 0.980 [0.960, 0.995] | -0.005 [-0.030, +0.020] | 0.980 [0.960, 0.995] | -0.005 [-0.030, +0.020] | 0.096 |
| MetaQA 2-hop | LLM decider only | 200 | 100% | 0.985 [0.965, 1.000] | — | 0.985 [0.965, 1.000] | — | 0.327 |
| MetaQA 2-hop | escalate below 0.8 | 200 | 14% | 0.995 [0.985, 1.000] | +0.010 [+0.000, +0.025] | 0.995 [0.985, 1.000] | +0.010 [+0.000, +0.025] | 0.144 |
| MetaQA 2-hop | escalate below 0.9 | 200 | 30% | 0.990 [0.975, 1.000] | +0.005 [+0.000, +0.015] | 0.990 [0.975, 1.000] | +0.005 [+0.000, +0.015] | 0.198 |
| MetaQA 2-hop | escalate below 0.95 | 200 | 49% | 0.985 [0.965, 1.000] | +0.000 [+0.000, +0.000] | 0.985 [0.965, 1.000] | +0.000 [+0.000, +0.000] | 0.257 |
| MetaQA 2-hop | escalate below 0.99 | 200 | 88% | 0.985 [0.965, 1.000] | +0.000 [+0.000, +0.000] | 0.985 [0.965, 1.000] | +0.000 [+0.000, +0.000] | 0.383 |
| MetaQA 3-hop | Jev only | 200 | 0% | 0.897 [0.855, 0.934] | -0.060 [-0.097, -0.024] | 0.805 [0.750, 0.860] | -0.015 [-0.040, +0.005] | 0.154 |
| MetaQA 3-hop | LLM decider only | 200 | 100% | 0.956 [0.931, 0.978] | — | 0.820 [0.765, 0.875] | — | 0.548 |
| MetaQA 3-hop | escalate below 0.8 | 200 | 11% | 0.945 [0.915, 0.970] | -0.012 [-0.031, +0.006] | 0.820 [0.765, 0.875] | +0.000 [-0.015, +0.015] | 0.224 |
| MetaQA 3-hop | escalate below 0.9 | 200 | 25% | 0.956 [0.931, 0.978] | +0.000 [+0.000, +0.000] | 0.820 [0.765, 0.875] | +0.000 [+0.000, +0.000] | 0.303 |
| MetaQA 3-hop | escalate below 0.95 | 200 | 42% | 0.956 [0.931, 0.978] | +0.000 [+0.000, +0.000] | 0.820 [0.765, 0.875] | +0.000 [+0.000, +0.000] | 0.392 |
| MetaQA 3-hop | escalate below 0.99 | 200 | 94% | 0.956 [0.931, 0.978] | +0.000 [+0.000, +0.000] | 0.820 [0.765, 0.875] | +0.000 [+0.000, +0.000] | 0.671 |
| 2Wiki gold-evidence graph | Jev only | 300 | 0% | 0.915 [0.886, 0.942] | -0.026 [-0.045, -0.010] | 0.883 [0.847, 0.920] | -0.043 [-0.070, -0.020] | 0.048 |
| 2Wiki gold-evidence graph | LLM decider only | 300 | 100% | 0.941 [0.916, 0.965] | — | 0.927 [0.897, 0.953] | — | 0.164 |
| 2Wiki gold-evidence graph | escalate below 0.8 | 300 | 7% | 0.927 [0.900, 0.953] | -0.015 [-0.029, -0.002] | 0.897 [0.863, 0.930] | -0.030 [-0.057, -0.007] | 0.069 |
| 2Wiki gold-evidence graph | escalate below 0.9 | 300 | 18% | 0.934 [0.908, 0.959] | -0.007 [-0.019, +0.002] | 0.917 [0.883, 0.947] | -0.010 [-0.027, +0.003] | 0.098 |
| 2Wiki gold-evidence graph | escalate below 0.95 | 300 | 34% | 0.942 [0.918, 0.966] | +0.001 [-0.003, +0.007] | 0.927 [0.897, 0.957] | +0.000 [-0.010, +0.010] | 0.126 |
| 2Wiki gold-evidence graph | escalate below 0.99 | 300 | 93% | 0.941 [0.916, 0.965] | +0.000 [+0.000, +0.000] | 0.927 [0.897, 0.953] | +0.000 [+0.000, +0.000] | 0.209 |

- **Escalating below 0.9 recovers all of the LLM decider's accuracy on MetaQA at
  47–61% of its cost.** On 3 hops, where the LLM decider's F1 lead is real (+0.060
  [+0.024, +0.097]), escalating 25% of queries closes it completely, at $0.30 vs $0.55
  per 1k queries.
- **On the 2Wiki gold graph it takes 0.95** (34% escalated, $0.13 vs $0.16) to close
  the gap: Jev's confidence is less informative there (AUROC 0.67 in E3, vs 0.94–0.97
  on MetaQA 2/3-hop).
- **0.9 is the recommended default**, and should be checked, not assumed, on each new
  graph: the A2 pilot measures it on Freebase.
- **Caveats.** One seed and 200–300 questions per dataset. The threshold grid was chosen
  after seeing that Jev's confidences cluster near 1, but it was not tuned per dataset.
  On text-derived graphs escalation does not help (walks fail for lack of facts, not
  for lack of decisiveness; see `docs/results-phase4.md`).

**In the library:** `Index(..., escalate_below=0.9)` (fallback: the LLM-as-decider on
the escalation model, or `fallback_decider=`), `Index.walk(query)` for the graph's own
answers with their confidence, and `EscalatingTraverser` for direct use.

## A5: Scale (`scripts/eval/scale.py`, `results/20261001T153517Z-scale/`)

A synthetic SQLite graph: 200k nodes and 1.4M edges, with 10 hubs of 20k neighbors
each (like a country or a profession in Freebase). Walks use a scripted decider with
no I/O, so the timings are graphwalk's own overhead per walk, excluding model latency.
The WebQSP pilot ran on the same 4-core machine at the same time, so the numbers are
somewhat pessimistic.

| operation | p50 | p95 | notes |
|---|---|---|---|
| import | 7,900 triples/s | | 1.4M triples in 176 s; 1.1 GB file; peak RSS 0.9 GB |
| `neighbors`, typical node (degree 13) | 0.4 ms | 0.6 ms | |
| `neighbors`, hub (degree 20k) | 1.16 s | 1.42 s | parses 20k edges and 20k nodes from JSON |
| `degree`, hub | 0.49 s | 0.72 s | measured before `degree` moved to SQL `COUNT` (commit after this run) |
| name linking | <0.1 ms | <0.1 ms | index build 6.6 s; right node linked 50/50 |
| walk, entity hops, avoiding hubs | 9 ms | — | p95 1.3–1.6 s, from walks that reach a hub |
| walk, entity hops, from a hub | 1.3–1.6 s | 1.9 s | |
| walk, relation hops, from a hub | 3.1–3.2 s | 3.3 s | p95 of walks from typical nodes: 6.0–6.3 s |

- **Away from hubs the engine is fast:** about 10 ms of overhead per 3-decision walk,
  well under a single Jev call (~0.3–0.5 s).
- **Hubs are the bottleneck, and relation mode suffers most:** every step parses all
  of a hub's neighbors, so one hub costs 1–3 s per walk, more than the decisions do.
  Making pydantic skip its Python-level checks on rows the store wrote itself did not
  help (the time is in core JSON parsing), so that change was reverted.
- **The fix is structural** (Phase 8, hardening): a relation-mode step should get
  per-relation counts and a few preview targets from SQL (`GROUP BY type`) and load
  only the chosen relation's targets. Entity mode at a hub needs the prefilter
  computed from names and types in SQL, not full models.
- Until then, the "fast" claim holds for graphs without very high-degree nodes, or
  with hubs capped at import.
- **Found and fixed on the way:** the embedding cache could drop rows when one call
  overflowed it, so the prefilter crashed on a 20k-option step (now fixed, with a
  regression test). No earlier result was affected: any occurrence would have raised
  an error, and none was recorded.

## A2 pilot: WebQSP, 50 questions (`scripts/eval/kgqa.py`, `results/kgqa/20261001T153829Z-webqsp/`)

Freebase per-question subgraphs (RoG release; median ~4,400 triples), topic entities
given, seed 0, n = 50 (every sampled question has its answer in its subgraph).

| system | hits@1 | F1 | $ / 1k q | p50 s | AUROC of confidence |
|---|---|---|---|---|---|
| graphwalk, Jev | 0.54 | 0.54 | 0.21 | 3.4 | **0.92** |
| graphwalk, LLM decider | 0.66 | 0.69 | 0.91 | 26.9 | 0.63 |
| LLM writes the relation path (2 retries) | **0.74** | **0.76** | 0.72 | 2.6 | — |

Escalation, simulated offline (an escalated question pays for both systems):

| cascade | escalated | F1 | $ / 1k q | F1 vs path writer |
|---|---|---|---|---|
| Jev → LLM decider, below 0.9 | 52% | 0.68 | 0.77 | −0.08 [−0.18, +0.01] |
| Jev → path writer, below 0.8 | 36% | 0.68 | 0.49 | −0.08 [−0.17, +0.00] |
| Jev → path writer, below 0.9 | 52% | 0.72 | 0.61 | −0.04 [−0.10, +0.00] |
| Jev → path writer, below 0.95 | 66% | 0.76 | 0.70 | +0.00 |

- **Confidence is informative here too** (AUROC 0.92, vs 0.63 for the LLM decider),
  so that half of the kill criterion passes.
- **Escalation saves almost no money at equal accuracy**, so the other half fails.
  The path writer is strongest: Freebase relation names are self-describing
  (`people.person.place_of_birth`) and each subgraph's schema is small, so one LLM call
  usually writes the right path. The questions Jev answers confidently are the easy
  ones the path writer also gets (top 20 by confidence: both EM 0.80). Matching the
  path writer's accuracy means escalating two thirds of the questions, at 96% of its
  cost.
- **Caveats:** n = 50, one seed, wide CIs. Jev's 3.4 s p50 is mostly not decisions
  (0.58 s): it is the per-question graph build (0.33 s) and embedding each
  question's relation options on CPU with a cold cache, on a machine also running A5.
- **Verdict, per the Phase 7 rule:** stop before the full A2 run. The one cheap check
  left is CWQ (compositional questions, up to four hops), where writing the whole path
  in one go should be hardest; E2b found walking wins on large, noisy schemas. If CWQ
  also favors the path writer, the paper's KG-QA claim narrows to "cheap first pass
  with informative confidence", and the release story to the cost and latency of
  walking.

## A2 pilot: CWQ, 50 questions (`results/kgqa/20261001T160324Z-cwq/`)

Same setup as WebQSP. CWQ composes WebQSP questions with constraints, conjunctions,
comparisons, and superlatives, up to four hops. Only 40 of the 50 sampled questions
have their answer inside their subgraph.

| system | hits@1 | F1 | $ / 1k q | p50 s | AUROC of confidence |
|---|---|---|---|---|---|
| graphwalk, Jev | 0.28 | 0.32 | **0.22** | **1.7** | 0.71 |
| graphwalk, LLM decider | 0.28 | 0.33 | 1.14 | 33.8 | 0.50 |
| LLM writes the relation path (2 retries) | 0.26 | 0.29 | 1.23 | 8.1 | — |

F1 gap, Jev − path writer: +0.02 [−0.07, +0.11]. Every cascade escalates 60–92% of
questions and gains nothing measurable.

- **Here Jev ties the alternatives at about a fifth of the cost and latency.** The
  path writer's advantage on WebQSP disappears when the path is long and
  compositional (it needs 1.7 calls per question and retries more often).
- **But every system is weak.** Published methods report hits@1 of about 0.63 (RoG)
  to 0.69 (Think-on-Graph with GPT-4) on CWQ. Our three systems are one greedy relation
  path from the topic entity, or one written path, with a small model. None handles
  CWQ's constraints ("the earliest", "that also ..."), and a fifth of the sample has
  no answer in its subgraph. Parity at 0.3 is not evidence the method is good. It only
  shows that nothing in this comparison is.
- **Confidence is weaker here** (AUROC 0.71) but still informative; the LLM decider's
  is not (0.50).

## A2 verdict (both pilots, 100 questions, about $0.20)

- **The full A2 run is not worth its $20–35 as designed.** On WebQSP the LLM path
  writer wins outright. On CWQ everything ties at a low level, far below published
  numbers. More seeds would firm up those two statements without changing them.
- **What survives:** Jev's confidence is informative (AUROC 0.92 / 0.71, vs 0.63 /
  0.50 for the LLM decider), and Jev walks are 4–5× cheaper and 2–15× faster.
  Escalation pays off on curated graphs with a small schema (MetaQA, A1), not on
  Freebase, where one LLM call can write the path.
- **To compete on CWQ** would need beam search over multiple topic entities plus
  constraint handling (filters and superlatives on the reached set). That is a
  research project, not a run.
