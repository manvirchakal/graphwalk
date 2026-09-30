# 2wiki evidence retrieval (E1)

- params: `{"dataset": "2wiki", "decision_model": "typesafe/jev-1.13", "embedder": "BAAI/bge-small-en-v1.5", "graph": "2wiki-330b8be64c4322b1.graph.json", "ks": [2, 5, 10], "multi_step_run": "2wiki-330b8be64c4322b1.graph-534f8fe7", "n": 120, "preset": "greedy-v2", "seed": 0}`
- git: `ec8490f74e39dc596aa0b5b6100b5c7aa78ad52c`
- environment: `{"git_dirty": true, "git_sha": "ec8490f74e39dc596aa0b5b6100b5c7aa78ad52c", "packages": {"fastembed": "0.8.1", "graphwalk": "0.1.0", "litellm": "1.103.0", "networkx": "3.7", "numpy": "2.5.3", "pydantic": "2.13.5", "typesafe-sdk": "0.7.2"}, "python": "3.12.3", "timestamp": "2026-09-30T12:44:10.645912+00:00"}`
- run at: 2026-09-30T12:44:10.645958+00:00

## Recall of gold evidence documents

| method | recall@2 | recall@5 | recall@10 | 95% CI @5 | complete@5 | docs returned | Jev calls/q | $/1k q |
|---|---|---|---|---|---|---|---|---|
| dense | 0.673 | 0.758 | 0.810 | [0.717, 0.800] | 0.475 | 10.0 | 0.00 | 0.000 |
| title+dense | 0.648 | 0.758 | 0.808 | [0.717, 0.800] | 0.475 | 10.0 | 0.00 | 0.000 |
| graph | 0.725 | 0.854 | 0.860 | [0.808, 0.894] | 0.692 | 3.3 | 3.52 | 0.183 |
| hybrid | 0.756 | 0.927 | 0.946 | [0.896, 0.954] | 0.800 | 10.0 | 3.52 | 0.183 |
| graph-choice | 0.750 | 0.892 | 0.894 | [0.856, 0.923] | 0.742 | 3.4 | 4.70 | 0.245 |
| hybrid-choice | 0.781 | 0.935 | 0.954 | [0.906, 0.960] | 0.817 | 10.0 | 4.70 | 0.245 |
| multi-step | 0.652 | 0.740 | 0.985 | [0.696, 0.783] | 0.458 | 5.9 | 0.00 | n/a |

multi-step, over everything it read (mean 5.9 docs): recall 0.985. Its cost is the M7 run's (LLM calls, including answering).

## Paired differences (recall@k, 95% bootstrap CI)

| a − b | Δ recall@2 | Δ recall@5 | Δ recall@10 |
|---|---|---|---|
| hybrid − dense | +0.083 [+0.033, +0.129] | +0.169 [+0.127, +0.210] | +0.135 [+0.100, +0.173] |
| hybrid-choice − dense | +0.108 [+0.065, +0.150] | +0.177 [+0.135, +0.219] | +0.144 [+0.108, +0.181] |
| graph-choice − dense | +0.077 [+0.023, +0.129] | +0.133 [+0.079, +0.185] | +0.083 [+0.031, +0.135] |
| graph-choice − graph | +0.025 [+0.008, +0.046] | +0.037 [+0.013, +0.071] | +0.033 [+0.008, +0.067] |
| graph-choice − title+dense | +0.102 [+0.050, +0.152] | +0.133 [+0.079, +0.185] | +0.085 [+0.033, +0.138] |
| hybrid-choice − title+dense | +0.133 [+0.087, +0.177] | +0.177 [+0.135, +0.221] | +0.146 [+0.108, +0.185] |
| multi-step − hybrid-choice | -0.129 [-0.173, -0.085] | -0.196 [-0.240, -0.154] | +0.031 [+0.004, +0.056] |

## recall@5 by question type

| method | bridge_comparison (n) | comparison (n) | compositional (n) | inference (n) |
|---|---|---|---|---|
| dense | 0.583 (30) | 1.000 (30) | 0.700 (30) | 0.750 (30) |
| title+dense | 0.583 (30) | 1.000 (30) | 0.700 (30) | 0.750 (30) |
| graph | 0.817 (30) | 0.850 (30) | 0.783 (30) | 0.967 (30) |
| hybrid | 0.858 (30) | 1.000 (30) | 0.850 (30) | 1.000 (30) |
| graph-choice | 0.850 (30) | 0.900 (30) | 0.850 (30) | 0.967 (30) |
| hybrid-choice | 0.875 (30) | 1.000 (30) | 0.867 (30) | 1.000 (30) |
| multi-step | 0.542 (30) | 1.000 (30) | 0.683 (30) | 0.733 (30) |

Errors (scored as empty): {'multi-step': 2}
