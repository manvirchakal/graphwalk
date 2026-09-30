# fanoutqa evidence retrieval (E1)

- params: `{"dataset": "fanoutqa", "decision_model": "typesafe/jev-1.13", "embedder": "BAAI/bge-small-en-v1.5", "graph": "fanoutqa-745e982de686504d.graph.json", "ks": [5, 10, 20], "multi_step_run": "fanout-fanoutqa-745e982de686504d.graph", "n": 40, "preset": "relation-v2", "seed": 0}`
- git: `6fc07c720c0192c6fb60f0c249c9127830442d05`
- environment: `{"git_dirty": false, "git_sha": "6fc07c720c0192c6fb60f0c249c9127830442d05", "packages": {"fastembed": "0.8.1", "graphwalk": "0.1.0", "litellm": "1.103.0", "networkx": "3.7", "numpy": "2.5.3", "pydantic": "2.13.5", "typesafe-sdk": "0.7.2"}, "python": "3.12.3", "timestamp": "2026-09-30T12:54:15.021421+00:00"}`
- run at: 2026-09-30T12:54:15.021464+00:00

## Recall of gold evidence documents

| method | recall@5 | recall@10 | recall@20 | 95% CI @10 | complete@10 | docs returned | Jev calls/q | $/1k q |
|---|---|---|---|---|---|---|---|---|
| dense | 0.645 | 0.884 | 0.965 | [0.831, 0.930] | 0.525 | 20.0 | 0.00 | 0.000 |
| title+dense | 0.643 | 0.879 | 0.965 | [0.826, 0.926] | 0.500 | 20.0 | 0.00 | 0.000 |
| graph | 0.371 | 0.436 | 0.478 | [0.347, 0.526] | 0.075 | 6.1 | 3.67 | 0.296 |
| hybrid | 0.587 | 0.827 | 0.967 | [0.762, 0.888] | 0.475 | 20.0 | 3.67 | 0.296 |
| graph-choice | 0.389 | 0.465 | 0.511 | [0.376, 0.553] | 0.075 | 6.5 | 5.55 | 0.431 |
| hybrid-choice | 0.585 | 0.837 | 0.970 | [0.778, 0.894] | 0.475 | 20.0 | 5.55 | 0.431 |
| multi-step | 0.575 | 0.770 | 0.941 | [0.686, 0.853] | 0.425 | 11.7 | 0.00 | 6.041 |

multi-step, over everything it read (mean 11.7 docs): recall 0.941. Its cost is the M7 run's (LLM calls, including answering).

## Paired differences (recall@k, 95% bootstrap CI)

| a - b | Δ recall@5 | Δ recall@10 | Δ recall@20 |
|---|---|---|---|
| hybrid - dense | -0.058 [-0.110, -0.005] | -0.057 [-0.106, -0.011] | +0.002 [-0.025, +0.036] |
| hybrid-choice - dense | -0.060 [-0.118, -0.005] | -0.047 [-0.098, +0.001] | +0.004 [-0.021, +0.039] |
| graph-choice - dense | -0.256 [-0.349, -0.171] | -0.419 [-0.514, -0.321] | -0.455 [-0.554, -0.352] |
| graph-choice - graph | +0.018 [-0.029, +0.064] | +0.029 [-0.026, +0.084] | +0.033 [-0.020, +0.087] |
| graph-choice - title+dense | -0.255 [-0.350, -0.169] | -0.414 [-0.509, -0.316] | -0.455 [-0.554, -0.352] |
| hybrid-choice - title+dense | -0.058 [-0.116, -0.004] | -0.042 [-0.090, +0.003] | +0.004 [-0.021, +0.039] |
| multi-step - hybrid-choice | -0.010 [-0.070, +0.053] | -0.067 [-0.143, +0.014] | -0.029 [-0.078, +0.015] |

## recall@10 by question type

| method | fanout (n) |
|---|---|
| dense | 0.884 (40) |
| title+dense | 0.879 (40) |
| graph | 0.436 (40) |
| hybrid | 0.827 (40) |
| graph-choice | 0.465 (40) |
| hybrid-choice | 0.837 (40) |
| multi-step | 0.770 (40) |
