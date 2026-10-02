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
- Fixed since: see "A5 after the hub fix" below.
- **Found and fixed on the way:** the embedding cache could drop rows when one call
  overflowed it, so the prefilter crashed on a 20k-option step (now fixed, with a
  regression test). No earlier result was affected: any occurrence would have raised
  an error, and none was recorded.

### A5 after the hub fix (`results/20261001T164752Z-scale/`)

Two changes (roadmap Phase 8, hardening): stores can list a node's edges as ids only
(`AdjacencyStore`; SQLite reads them straight from its indexes), and the engine loads
full nodes only for the options it keeps. Profiling then found a quadratic loop: relation
hops deduplicated each relation's targets with a list scan, 1.8 s of a 1.9 s walk from
a hub. Same graph and machine (now otherwise idle), same scripted decider:

| operation | before | after | speedup |
|---|---|---|---|
| list a hub's edges (`neighbors` → `adjacency`) | 1,157 ms | 56 ms | 21× |
| walk, relation hops, from a hub | 3,069 ms | 113 ms | 27× |
| walk, relation hops, typical nodes, p95 | 6,048 ms | 176 ms | 34× |
| walk, entity hops, from a hub | 1,307 ms | 191 ms | 7× |
| walk, entity hops, from a hub, with prefilter | 1,624 ms | 539 ms | 3× |
| walk, typical nodes, p50 | 9–10 ms | 7–8 ms | |

- **Hubs no longer dominate a walk:** the overhead is now below one Jev call
  (~0.3–0.5 s) in every configuration.
- **Results are unchanged.** A parity test walks a random graph with hubs and
  self-loops through both paths, in both hop modes, with and without the prefilter,
  and compares every step. The loader order is the same, so earlier results stand.
- **The quadratic loop was in the shared code path,** so earlier relation-mode
  latencies on graphs with large fan-outs (MetaQA's genres and years) were somewhat
  inflated. Accuracy and cost were not affected.
- **Still open:** with the prefilter, a hub's 20k neighbor names are embedded and
  ranked. The benchmark's fake embedder makes that cheap. A real local embedder needs a
  few seconds the first time it sees a hub, then hits its cache. Entity mode at hubs
  is best run without the prefilter or with relation hops.

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

## A6: one large graph, 5,419 relations (`scripts/eval/kgqa_global.py`, `results/kgqa/20261001T172038Z-webqsp-global/`)

The test of graphwalk's remaining pitch: a curated graph whose schema is too large to
curate per question. All 1,628 WebQSP test subgraphs merged into one SQLite graph:
781k nodes, 2.28M edges, 5,419 Freebase relations (each A2 question saw a median of
288). Same topic entities as A2; 100 questions, seed 0; gpt-6-luna for every LLM.

| system | n | hits@1 | F1 | $/1k q | p50 s | in tok/q |
|---|---|---|---|---|---|---|
| graphwalk, Jev, `kgqa()` | 100 | 0.50 | 0.49 | 0.24 | 0.8 | 5.6k |
| LLM path, schema within 2 hops of the topic entity | 100 | 0.65 | 0.66 | 1.48 | 3.0 | 12k |
| LLM path, the whole 5,419-relation schema | 50 | 0.64 | 0.66 | 12.95 | 3.6 | 132k |

Paired bootstrap (5,000 resamples), F1:
- local-schema path − Jev: **+0.16 [+0.07, +0.25]** (100 q). Jev is better on 9
  questions, worse on 30.
- whole-schema path − local-schema path: −0.02 [−0.09, +0.05] (the first 50 q).
- whole-schema path − Jev: +0.14 [+0.01, +0.27] (50 q).

Cascade, computed offline (Jev; below the threshold, the local-schema path writer):
0.59 F1 at $0.91/1k (t = 0.8, 42% escalated), 0.64 at $1.08 (t = 0.9), 0.66 at $1.20
(t = 0.95, 68% escalated): the path writer's accuracy for 19% less.

- **The large-schema claim fails on a curated graph.** A schema too large to read is
  not a problem for an LLM writing the query: fetching the relations around the topic
  entity (one adjacency query) gives it a few hundred, and it beats walking by 16 F1
  points. Even the whole 5,419-relation schema fits a 1M-token context and does as
  well, only at 9× the cost.
- **What remains for walking is cost and speed:** 6× cheaper and ~4× faster at p50
  than the local-schema path writer, at −0.16 F1. A confidence cascade recovers the
  accuracy but saves only ~19% of the cost.
- **E2b's win is therefore a statement about noisy, text-extracted graphs** (where
  every method scores 0.18–0.25), not about large schemas.

## A7: can we tell before walking? (`scripts/eval/router_analysis.py`, `results/router/20261001T174239Z/`)

No API spend: the Jev walk logs above (MetaQA 1–3 hop decider runs, the WebQSP and CWQ
pilots, A6) joined with their questions. Label: walk F1 ≥ 0.5. Pre-walk features: question
length, wh-word, superlative/temporal/conjunction/counting cues, number of topic
entities, degree and relation counts around the start. 5-fold cross-validated logistic
regression; AUROC with 95% bootstrap CIs.

| data | n | walk right | pre-walk | confidence | pre + confidence |
|---|---|---|---|---|---|
| MetaQA 1–3 hop | 1,760 | 0.94 | 0.68 [0.62, 0.73] | 0.96 [0.94, 0.98] | 0.97 [0.96, 0.98] |
| WebQSP | 50 | 0.54 | 0.55 [0.38, 0.72] | 0.89 [0.78, 0.97] | 0.86 [0.75, 0.95] |
| CWQ | 50 | 0.32 | 0.41 [0.23, 0.60] | 0.70 [0.52, 0.87] | 0.50 [0.32, 0.68] |
| WebQSP, merged graph | 100 | 0.53 | 0.62 [0.50, 0.73] | 0.83 [0.74, 0.91] | 0.76 [0.67, 0.85] |
| Freebase, pooled | 200 | 0.48 | 0.63 [0.55, 0.70] | 0.83 [0.77, 0.88] | 0.78 [0.72, 0.84] |

- **Within a graph, the question barely predicts whether the walk will work** (0.55–0.68;
  CWQ is at chance), and adding these features to confidence does not help (it overfits
  on the small Freebase sets). The superlative/constraint cue we expected to matter
  never ranks among the useful features at this sample size.
- **What does predict it is which graph you are on** (pooled over all data, pre-walk
  AUROC rises to 0.81, almost all of it MetaQA vs Freebase), and a deployment already
  knows that: the guide's regime map is the "router".
- **So the routing rule is: walk first, read the confidence.** A walk costs about
  $0.0002 and a second; its confidence separates right from wrong far better than
  anything visible beforehand. Caveat: 200 Freebase questions, one hand-made feature
  set; an LLM judging the question might do better, but it would cost more than the walk.

## A8: graphwalk as an agent's tool (`scripts/eval/agent_arms.py`, `results/agent/20261001T195538Z-webqsp-agent/`)

One plain tool-calling loop (`graphwalk.eval.agent`), the same model, prompt, and
10-turn budget for every arm; only the tools differ. 100 WebQSP test questions over one
graph merging their subgraphs (103k nodes, 303k edges; the search index embeds the same
facts as text, record nodes grouped into one passage each). Topic entities given.
Agent model: a cheap one (gpt-6-luna). Total spend ≈ $0.30 (pilot of 20 included).

| arm | F1 [95% CI] | hits@1 | turns | agent input tok/q | agent $/q | tool $/q | grounded |
|---|---|---|---|---|---|---|---|
| walk alone (no agent) | 0.51 [0.42, 0.60] | 0.52 | 0 | – | – | 0.00022 | – |
| closed book (no tools) | 0.42 [0.34, 0.50] | 0.55 | 1 | 197 | 0.00009 | – | 0.00 |
| agent + graph tools | **0.74** [0.67, 0.81] | 0.83 | 5.5 | 9,421 | 0.00088 | – | 0.91 |
| agent + graph tools + walk | 0.72 [0.64, 0.80] | 0.76 | 4.8 | 5,979 | 0.00062 | 0.00039 | 0.94 |
| agent + search (RAG) | 0.68 [0.60, 0.75] | 0.81 | 4.1 | 4,377 | 0.00044 | – | 0.92 |

Paired F1: walk − graph −0.02 [−0.09, +0.05]; graph − search +0.06 [+0.00, +0.12];
walk − search +0.05 [−0.02, +0.11]; agent+walk − walk alone +0.21 [+0.12, +0.30].
("Grounded": share of answers that appeared in some tool output; the closed-book arm
shows the model knows ~40% of WebQSP by heart.)

Split by the first walk's confidence (walk arm vs graph-only arm, same questions):

| first walk | q | F1 walk / graph | turns walk / graph | input tok walk / graph |
|---|---|---|---|---|
| ≥ 0.9 | 38 | 0.90 / 0.90 | 2.6 / 4.2 | 1,580 / 5,364 |
| < 0.9 | 62 | 0.61 / 0.64 | 6.1 / 6.3 | 8,675 / 11,908 |

- **The walk tool does not make the agent more accurate** (the pilot's +0.11 on 20
  questions did not survive 100). An agent with plain `relations`/`neighbors` finds the
  same answers.
- **It makes the agent cheaper when the walk is confident:** on 38% of questions the
  agent took the walk's answer, with no loss in accuracy, in 2.6 turns instead of 4.2
  and 70% fewer input tokens. On the rest it is pure overhead (one more call, same
  exploration afterwards).
- **Whether that pays depends on the agent's price.** With this cheap agent the walk's
  own decision cost (~$0.0004) eats the savings (≈ $0.0012/q either way, list prices).
  Repriced at $3/$15 per M tokens (a frontier-class agent), the walk arm costs ≈ $0.025/q
  vs $0.038/q for graph tools alone (−34%), at equal accuracy.
- **Search (RAG) is cheapest in tokens and a little less accurate** (−0.06 F1 vs graph
  tools, CI touching 0): over a graph whose facts are short triples, text search is a
  reasonable baseline, not a straw man.
- **The agent loop is what lifts accuracy** (+0.21 over the walk alone): it applies the
  constraints and checks that walks skip.

Limits: one dataset, one cheap agent model, topic entities given; a stronger agent
might explore more efficiently (narrowing the walk's token savings) or trust walks
differently. Turns and tokens are logged per question, so other prices can be applied
to `results.json` without rerunning.

### A8b: a strong agent model (`results/agent/20261001T214718Z-webqsp-agent/`)

!!! note "Superseded by P4"
    Over all 100 questions ([P4](results-paper.md)), the saving is 17% (95% CI 8–27%),
    not 25%: these first 30 questions overstated it.

The same harness, the first 30 questions, with a frontier-class agent (Gemini 3.1 Pro,
$2/$12 per M tokens; actual OpenRouter charges). ≈ $3.30.

| arm | F1 [95% CI] | hits@1 | turns | agent in / out tok | $/q (incl. tools) | p50 latency s |
|---|---|---|---|---|---|---|
| closed book | 0.57 [0.43, 0.71] | 0.77 | 1 | 177 / 886 | 0.011 | 7 |
| graph tools | 0.71 [0.57, 0.85] | 0.77 | 6.4 | 9,839 / 1,882 | 0.042 | 30 |
| graph tools + walk | **0.75** [0.62, 0.88] | 0.80 | 6.4 | 8,804 / 1,142 | 0.032 | 24 |
| search (RAG) | 0.72 [0.57, 0.85] | 0.83 | 6.9 | 7,171 / 1,136 | **0.028** | 40 |

Paired: walk − graph F1 +0.04 [−0.01, +0.12], cost **−$0.011/q [−0.019, −0.003] (−25%)**;
walk − search F1 +0.03 [−0.03, +0.11], cost +$0.004/q [−0.002, +0.009].

| first walk | q | F1 walk / graph | turns walk / graph | $/q walk / graph |
|---|---|---|---|---|
| ≥ 0.9 | 11 | 0.97 / 0.97 | 5.4 / 5.0 | 0.024 / 0.033 |
| < 0.9 | 15 | 0.57 / 0.52 | 7.7 / 7.8 | 0.041 / 0.055 |
| not called | 4 | 0.82 / 0.74 | 4.5 / 4.8 | 0.017 / 0.020 |

- **The cost saving is real with an expensive agent** (−25%, CI excludes 0), close to
  the −34% predicted by repricing A8's cheap-agent tokens, at equal or slightly better
  accuracy.
- **But not by the mechanism expected.** The strong agent does not stop at a confident
  walk: it verifies (5.4 turns vs 5.0). The saving is in tokens: fewer `neighbors`/
  `relations` calls per turn and ~40% less output (reasoning) per question. A strong
  agent treats the walk as a map, not an answer.
- **Search (RAG) is as accurate and slightly cheaper** than either graph arm on this
  benchmark, but the slowest (p50 40 s, more rounds). Over short Freebase triples,
  dense search is a strong baseline for a strong agent.
- **The strong model knows WebQSP better** (closed book 0.57), which compresses every
  arm's margin; 30 questions cannot separate the three tool sets on accuracy.
