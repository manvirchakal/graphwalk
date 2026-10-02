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

## P4, P5

Running; results to follow.
