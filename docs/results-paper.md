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

## P1: is the confidence Jev's, or the framing's? (`graphwalk.eval.logprob_decider`)

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

## P5: walk vs the LLM path writer, 500 questions (MetaQA)

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
