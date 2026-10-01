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
