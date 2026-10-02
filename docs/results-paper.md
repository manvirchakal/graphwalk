# Paper experiments (P0–P5)

The experiments planned in [`paper/PLAN.md`](../paper/PLAN.md) to answer a reviewer's
first questions. Same caveats as the other pages: small samples, 95% bootstrap CIs
where shown, hosted models whose versions can change.

## P0: selective prediction (`scripts/paper/figure_selective.py`)

If the walk answers only when confident, how accurate is it? Questions are sorted by
the walk's confidence and answered down to a coverage level. AURC is the area under
the risk–coverage curve (risk = 1 − accuracy; lower is better); "random" is the AURC of
a random order (the error rate), "perfect" the floor for that accuracy. Correct means EM
on MetaQA and 2Wiki, hits@1 on WebQSP and CWQ. Curve points:
`paper/figures/selective.csv`. No new model calls: scores from E3, A2, A6 and P1.

| dataset | decider | n | AUROC | AURC | random | perfect | acc @25% | @50% | @100% |
|---|---|---|---|---|---|---|---|---|---|
| MetaQA 1-hop | Jev | 200 | 0.64 | 0.040 | 0.070 | 0.003 | 0.98 | 0.95 | 0.93 |
| | LLM, stated scores | 200 | 0.66 | 0.030 | 0.055 | 0.002 | 0.96 | 0.97 | 0.95 |
| | open model, token probs | 200 | 0.58 | 0.043 | 0.065 | 0.002 | 0.96 | 0.96 | 0.94 |
| MetaQA 2-hop | Jev | 200 | 0.94 | 0.002 | 0.020 | 0.000 | 1.00 | 1.00 | 0.98 |
| | LLM, stated scores | 200 | 0.54 | 0.008 | 0.015 | 0.000 | 1.00 | 0.99 | 0.99 |
| | open model, token probs | 200 | **0.97** | 0.001 | 0.030 | 0.001 | 1.00 | 1.00 | 0.97 |
| MetaQA 3-hop | Jev | 200 | **0.97** | 0.028 | 0.195 | 0.021 | 1.00 | 0.99 | 0.81 |
| | LLM, stated scores | 200 | 0.64 | 0.149 | 0.180 | 0.018 | 0.88 | 0.86 | 0.82 |
| | open model, token probs | 200 | 0.96 | 0.032 | 0.200 | 0.022 | 1.00 | 0.99 | 0.80 |
| 2Wiki (gold-evidence graph) | Jev | 300 | 0.67 | 0.141 | 0.117 | 0.007 | 0.88 | 0.93 | 0.88 |
| | LLM, stated scores | 300 | 0.69 | 0.037 | 0.073 | 0.003 | 0.96 | 0.96 | 0.93 |
| | open model, token probs | 300 | **0.77** | 0.139 | 0.147 | 0.012 | 0.88 | 0.94 | 0.85 |
| WebQSP | Jev | 50 | **0.87** | 0.189 | 0.460 | 0.132 | 1.00 | 0.80 | 0.54 |
| | LLM, stated scores | 50 | 0.64 | 0.238 | 0.340 | 0.069 | 0.75 | 0.72 | 0.66 |
| | open model, token probs | 50 | 0.84 | 0.154 | 0.380 | 0.087 | 1.00 | 0.84 | 0.62 |
| CWQ | Jev | 50 | 0.59 | 0.611 | 0.720 | 0.371 | 0.50 | 0.28 | 0.28 |
| | LLM, stated scores | 50 | 0.54 | 0.750 | 0.720 | 0.371 | 0.25 | 0.24 | 0.28 |
| | open model, token probs | 50 | **0.73** | 0.607 | 0.720 | 0.371 | 0.50 | 0.44 | 0.28 |
| WebQSP, one graph (A6) | Jev | 100 | 0.83 | 0.268 | 0.500 | 0.156 | 0.92 | 0.74 | 0.50 |

- **Where the confidence works, it works for selective answering:** on MetaQA 3-hop,
  answering the most confident half is 99% accurate against 81% overall; on WebQSP,
  the most confident quarter is 100% (n=50) against 54%.
- **On 2Wiki, sorting by Jev's confidence is worse than random order** (AURC 0.141 vs
  0.117) despite AUROC 0.67: a few confident wrong answers sit at the top. Calibration
  is not uniform across graphs; the paper must say so.
- AUROC on WebQSP/CWQ here uses hits@1; with EM (as in A2) Jev's is 0.92 / 0.71.

## P1: is the confidence Jev's, or the framing's? (`graphwalk.decisions.logprob`)

The same walks, with the decision made by an open-weights model (Qwen3.8-27B via
OpenRouter): options are lettered, the model answers with one letter, and the
distribution is read from that token's probabilities (`max_tokens=1`, temperature 0,
reasoning off). Runs: `results/decider/20261002T02*` and `results/kgqa/20261002T024*`.

| dataset | n | EM Jev / open | AUROC Jev / open | p50 s Jev / open | $ / 1k q Jev / open |
|---|---|---|---|---|---|
| MetaQA 1-hop | 200 | 0.93 / 0.94 | 0.64 / 0.58 | – / 5.5 | – / 0.10 |
| MetaQA 2-hop | 200 | 0.98 / 0.97 | 0.94 / 0.97 | 0.7 / 6.8 | 0.10 / 0.14 |
| MetaQA 3-hop | 200 | 0.80 / 0.80 | 0.97 / 0.96 | 1.1 / 9.9 | 0.15 / 0.26 |
| 2Wiki gold graph | 300 | 0.88 / 0.85 | 0.67 / 0.77 | – / 2.6 | – / 0.07 |
| WebQSP (hits@1) | 50 | 0.54 / 0.62 | 0.87 / 0.84 | – / 5.6 | – / 0.34 |
| CWQ (hits@1) | 50 | 0.28 / 0.28 | 0.59 / 0.73 | – / 7.0 | – / 0.28 |

(Jev latency and cost on MetaQA 2/3-hop are from P5's 500-question runs; "–" where the
Jev run predates per-run latency logging in this form; see E3 for those.)

- **The confidence comes from the framing, not from Jev.** An open model's token
  probabilities rank answers as well as Jev's on MetaQA 2–3 hop and WebQSP, and better
  on 2Wiki and CWQ, with the same accuracy. The weak LLM-decider confidence in E3
  (0.50–0.69) was the model's *stated* scores, not its probabilities.
- **What Jev buys is latency:** ~10× faster per walk than the open model through
  OpenRouter, and somewhat cheaper. It is no longer the source of the calibration
  claim.
- **Caveat:** found while setting this up, the engine cut options over the decider's
  limit in arbitrary order when the count was under `prefilter_threshold`; it now
  ranks them with the embedder. The 2Wiki, WebQSP and CWQ runs above were made after
  the fix; MetaQA never hit it. Jev's limit (255) is rarely reached.

## P4: the strong agent on 100 questions (`scripts/paper/table_agent.py`)

A8b ran the strong agent (Gemini 3.1 Pro) on the first 30 of the 100-question WebQSP
sample; P4 ran the other 70 with the graph-tools and walk arms only (same harness, one
arm per run; `results/agent/20261002T025127Z-*` and `20261002T030706Z-*`). ≈ $5.

| arm | n | F1 [95% CI] | hits@1 | $/q | p50 s |
|---|---|---|---|---|---|
| graph tools | 100 | 0.747 [0.672, 0.823] | 0.79 | 0.0376 | 21.6 |
| graph tools + walk | 100 | 0.755 [0.683, 0.824] | 0.84 | 0.0311 | 19.4 |

Paired, walk arm minus graph-tools arm:

| questions | n | F1 | $/q | relative | latency s |
|---|---|---|---|---|---|
| all | 100 | +0.008 [−0.038, +0.057] | −0.0065 [−0.0101, −0.0030] | **−17%** | −0.1 [−2.4, +2.1] |
| A8b (first 30) | 30 | +0.035 [−0.013, +0.116] | −0.0107 [−0.0192, −0.0029] | −25% | +0.4 [−5.2, +6.2] |
| P4 (other 70) | 70 | −0.004 [−0.063, +0.058] | −0.0047 [−0.0088, −0.0012] | −13% | −0.4 [−2.7, +1.6] |

- **The saving holds but is smaller than the pilot's:** −17% (CI −8% to −27%) at equal
  accuracy. The first 30 questions overstated it (−25%); the other 70 alone give −13%.
- **No accuracy effect** (+0.01 F1, CI ±0.05) and no latency effect.

### P1b: does it hold for other open models? (MetaQA 3-hop)

The same decider and walks with two more open-weights models
(`results/decider/20261002T190949Z-metaqa-3hop`, DeepSeek V4 Flash;
`…191835Z-metaqa-3hop`, Gemma 4 31B). Jev from P5 (n = 500, same seed-0 sample).

| decider | n | EM | F1 | AUROC | ECE | p50 s | $ / 1k q |
|---|---|---|---|---|---|---|---|
| Jev | 500 | 0.80 | 0.88 | **0.95** | 0.11 | **1.1** | 0.15 |
| Qwen3.8-27B | 200 | 0.80 | 0.90 | **0.96** | 0.13 | 9.9 | 0.26 |
| Gemma 4 31B | 200 | **0.82** | **0.93** | 0.82 | 0.18 | 9.7 | 0.30 |
| DeepSeek V4 Flash | 200 | 0.44 | 0.49 | 0.84 | 0.36 | 10.5 | 0.09 |

- **The confidence is informative with every model tried** (AUROC 0.82–0.96), but its
  quality varies: Gemma's is overconfident (mean confidence 0.995 against 0.82 EM).
- **Accuracy depends on the model.** Gemma matched or beat Jev; DeepSeek V4 Flash
  collapsed (it rarely chose `STOP`: 150 of 200 walks used all four decisions). The
  claim is "a decider that reads probabilities", not "any model": check the model on
  labeled questions first.
- Only Qwen3.8-27B matched Jev on both accuracy and confidence.

## P3: how much is memory? Entity names replaced by aliases (`scripts/paper/table_anon.py`)

A8's cheap-agent setup (gpt-6-luna, the same 100 WebQSP questions, one graph merging
their subgraphs), rerun with every entity name replaced by a meaningless alias
(`Entity 3fa9c1`) in the graph, the gold answers, and the question text
(`rog.anonymize`; relations, compound-value ids, numbers and dates kept). The topic
entity's name was found and replaced in 70 of 100 questions; in the rest the question
names it another way ("usa"). Runs: `results/agent/20261002T1*-webqsp-agent-anon`.
≈ $0.70.

| arm | F1, real names (A8) | F1, aliases | change | $/q real | $/q aliases |
|---|---|---|---|---|---|
| closed book | 0.42 [0.34, 0.50] | 0.01 [0.00, 0.01] | −0.42 | 0.00009 | 0.00012 |
| walk alone (no agent) | 0.51 [0.42, 0.60] | 0.36 [0.27, 0.45] | −0.15 | 0.00022 | 0.00024 |
| agent + graph tools | 0.74 [0.67, 0.81] | 0.45 [0.36, 0.54] | −0.29 | 0.00088 | 0.00186 |
| agent + graph tools + walk | 0.72 [0.64, 0.80] | **0.52** [0.43, 0.61] | −0.20 | 0.00100 | 0.00227 |
| agent + search (RAG) | 0.68 [0.60, 0.75] | 0.20 [0.13, 0.27] | **−0.48** | 0.00044 | 0.00179 |

Paired on the aliased graph:

| comparison | F1 | $/q |
|---|---|---|
| graph tools + walk − graph tools | +0.073 [−0.014, +0.162] | +0.00042 [+0.00016, +0.00068] |
| graph tools + walk − search | **+0.322** [+0.226, +0.417] | +0.00049 [+0.00020, +0.00077] |
| graph tools − search | **+0.249** [+0.157, +0.345] | +0.00007 [−0.00017, +0.00031] |
| graph tools + walk − walk alone | +0.161 [+0.066, +0.257] | +0.00203 [+0.00175, +0.00231] |

- **The renaming works as a memory control:** closed book falls to 0.01.
- **Every arm leaned on names, search the most.** Without them, an agent exploring
  the graph beats one searching the same facts as text by 0.25–0.32 F1. A8/A8b's
  "search ties graph tools" holds only on a graph whose names the model knows. Part of
  this is dense retrieval losing meaningful names to embed, which a private graph with
  real but unfamiliar names would not suffer as much; the aliased graph is a lower
  bound for search, not a model of a private KG.
- **`walk` may help accuracy when memory can't:** +0.07 F1 over graph tools alone, but
  the CI includes 0 (n=100). With this cheap agent it costs a little more, as in A8.
- **The walk alone loses 0.15 F1:** Jev reads entity names in the options it scores,
  so its relation choices also used what the names mean.
- Agents explore about twice as much on aliases (15–18k vs 6–9k input tokens per
  question), so every agent arm costs roughly twice as much.

, 500 questions (MetaQA)

Same seed-0 samples at n = 500 (`results/decider/20261002T021023Z-metaqa-2hop`,
`…021310Z-metaqa-3hop`, `results/20261002T023850Z-metaqa-2hop-llmpath`,
`…024718Z-metaqa-3hop-llmpath`).

| setting | system | F1 | EM | $ / 1k q | p50 s |
|---|---|---|---|---|---|
| MetaQA 2-hop | walk (Jev) | 0.985 | 0.982 | 0.096 | 0.67 |
| | LLM writes the path, 2 retries | **0.999** | 0.996 | **0.061** | 1.44 |
| MetaQA 3-hop | walk (Jev) | 0.884 | 0.804 | 0.152 | 1.08 |
| | LLM writes the path, 2 retries | **0.964** | 0.834 | **0.092** | 2.19 |

- **Confirms E2 at 2.5× the sample:** on a small clean schema the LLM path writer is
  more accurate *and* cheaper; the walk is about 2× faster. The case for walking here is
  latency and the per-answer confidence, not cost or accuracy.

WebQSP, 500 test questions, per-question subgraphs (`results/kgqa/20261002T031206Z-webqsp`):

| system | hits@1 | F1 | $ / 1k q | p50 / p95 s | AUROC (EM) |
|---|---|---|---|---|---|
| walk (Jev) | 0.54 | 0.54 | **0.22** | **1.8 / 3.1** | 0.86 |
| LLM writes the path, 2 retries | **0.68** | **0.69** | 0.78 | 2.4 / 15.8 | – |

- On Freebase's larger per-question schemas the walk is **3.6× cheaper** with a much
  shorter tail (p95 3 s vs 16 s), and 15 F1 points less accurate, as in A2.
- Its confidence stays informative at 10× the A2 sample (AUROC 0.86 with EM, vs 0.92
  on the 50-question pilot).
