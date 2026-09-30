# hotpotqa evidence retrieval (E1)

- params: `{"dataset": "hotpotqa", "decision_model": "typesafe/jev-1.13", "embedder": "BAAI/bge-small-en-v1.5", "graph": "hotpotqa-81b56da2fae89b0c.graph.json", "ks": [2, 5, 10], "multi_step_run": "hotpotqa-81b56da2fae89b0c.graph-534f8fe7", "n": 120, "preset": "greedy-v2", "seed": 0}`
- git: `ec8490f74e39dc596aa0b5b6100b5c7aa78ad52c`
- environment: `{"git_dirty": true, "git_sha": "ec8490f74e39dc596aa0b5b6100b5c7aa78ad52c", "packages": {"fastembed": "0.8.1", "graphwalk": "0.1.0", "litellm": "1.103.0", "networkx": "3.7", "numpy": "2.5.3", "pydantic": "2.13.5", "typesafe-sdk": "0.7.2"}, "python": "3.12.3", "timestamp": "2026-09-30T12:47:42.705309+00:00"}`
- run at: 2026-09-30T12:47:42.705350+00:00

## Recall of gold evidence documents

| method | recall@2 | recall@5 | recall@10 | 95% CI @5 | complete@5 | docs returned | Jev calls/q | $/1k q |
|---|---|---|---|---|---|---|---|---|
| dense | 0.804 | 0.921 | 0.967 | [0.883, 0.954] | 0.850 | 10.0 | 0.00 | 0.000 |
| title+dense | 0.771 | 0.925 | 0.967 | [0.887, 0.958] | 0.858 | 10.0 | 0.00 | 0.000 |
| graph | 0.554 | 0.713 | 0.733 | [0.650, 0.771] | 0.525 | 4.5 | 4.47 | 0.229 |
| hybrid | 0.671 | 0.938 | 0.979 | [0.904, 0.967] | 0.883 | 10.0 | 4.47 | 0.229 |
| graph-choice | 0.596 | 0.758 | 0.779 | [0.700, 0.812] | 0.583 | 4.8 | 5.67 | 0.292 |
| hybrid-choice | 0.708 | 0.929 | 0.979 | [0.896, 0.963] | 0.867 | 10.0 | 5.67 | 0.292 |
| multi-step | 0.800 | 0.921 | 0.958 | [0.883, 0.954] | 0.850 | 5.1 | 0.00 | 0.141 |

multi-step, over everything it read (mean 5.1 docs): recall 0.958. Its cost is the M7 run's (LLM calls, including answering).

## Paired differences (recall@k, 95% bootstrap CI)

| a − b | Δ recall@2 | Δ recall@5 | Δ recall@10 |
|---|---|---|---|
| hybrid − dense | -0.133 [-0.188, -0.079] | +0.017 [-0.013, +0.046] | +0.013 [-0.004, +0.033] |
| hybrid-choice − dense | -0.096 [-0.154, -0.037] | +0.008 [-0.017, +0.033] | +0.013 [-0.004, +0.033] |
| graph-choice − dense | -0.208 [-0.279, -0.138] | -0.163 [-0.225, -0.104] | -0.188 [-0.242, -0.133] |
| graph-choice − graph | +0.042 [+0.000, +0.079] | +0.046 [+0.004, +0.092] | +0.046 [+0.004, +0.092] |
| graph-choice − title+dense | -0.175 [-0.250, -0.104] | -0.167 [-0.225, -0.108] | -0.188 [-0.242, -0.133] |
| hybrid-choice − title+dense | -0.062 [-0.125, -0.004] | +0.004 [-0.025, +0.033] | +0.013 [-0.004, +0.033] |
| multi-step − hybrid-choice | +0.092 [+0.037, +0.150] | -0.008 [-0.042, +0.021] | -0.021 [-0.042, -0.004] |

## recall@5 by question type

| method | bridge (n) | comparison (n) |
|---|---|---|
| dense | 0.850 (60) | 0.992 (60) |
| title+dense | 0.850 (60) | 1.000 (60) |
| graph | 0.750 (60) | 0.675 (60) |
| hybrid | 0.892 (60) | 0.983 (60) |
| graph-choice | 0.783 (60) | 0.733 (60) |
| hybrid-choice | 0.875 (60) | 0.983 (60) |
| multi-step | 0.850 (60) | 0.992 (60) |
