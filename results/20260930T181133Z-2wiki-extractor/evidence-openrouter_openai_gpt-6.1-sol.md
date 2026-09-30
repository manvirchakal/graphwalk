# 2wiki evidence retrieval (E1)

- params: `{"extractor": "openrouter/openai/gpt-6.1-sol", "n": 60}`
- git: `a6ce2306d1d50a268541eac86ab1ad140efe6a3e`
- environment: `{"git_dirty": true, "git_sha": "a6ce2306d1d50a268541eac86ab1ad140efe6a3e", "packages": {"fastembed": "0.8.1", "graphwalk": "0.1.0", "litellm": "1.103.0", "networkx": "3.7", "numpy": "2.5.3", "pydantic": "2.13.5", "typesafe-sdk": "0.7.2"}, "python": "3.12.3", "timestamp": "2026-09-30T18:14:31.035978+00:00"}`
- run at: 2026-09-30T18:14:31.036023+00:00

## Recall of gold evidence documents

| method | recall@2 | recall@5 | recall@10 | 95% CI @5 | complete@5 | docs returned | Jev calls/q | $/1k q |
|---|---|---|---|---|---|---|---|---|
| dense | 0.683 | 0.733 | 0.812 | [0.675, 0.792] | 0.417 | 10.0 | 0.00 | 0.000 |
| title+dense | 0.658 | 0.733 | 0.812 | [0.675, 0.792] | 0.417 | 10.0 | 0.00 | 0.000 |
| graph | 0.738 | 0.854 | 0.854 | [0.792, 0.908] | 0.667 | 3.0 | 3.38 | 0.177 |
| hybrid | 0.779 | 0.938 | 0.946 | [0.904, 0.967] | 0.800 | 9.9 | 3.38 | 0.177 |
| graph-choice | 0.758 | 0.887 | 0.887 | [0.838, 0.933] | 0.717 | 3.2 | 4.53 | 0.232 |
| hybrid-choice | 0.800 | 0.950 | 0.958 | [0.917, 0.975] | 0.833 | 9.9 | 4.53 | 0.232 |

## Paired differences (recall@k, 95% bootstrap CI)

| a - b | Δ recall@2 | Δ recall@5 | Δ recall@10 |
|---|---|---|---|
| hybrid - dense | +0.096 [+0.033, +0.167] | +0.204 [+0.142, +0.267] | +0.133 [+0.079, +0.188] |
| hybrid-choice - dense | +0.117 [+0.050, +0.183] | +0.217 [+0.158, +0.275] | +0.146 [+0.092, +0.200] |
| graph-choice - dense | +0.075 [+0.000, +0.150] | +0.154 [+0.067, +0.233] | +0.075 [-0.008, +0.154] |
| graph-choice - graph | +0.021 [+0.000, +0.050] | +0.033 [+0.000, +0.075] | +0.033 [+0.000, +0.075] |
| graph-choice - title+dense | +0.100 [+0.029, +0.171] | +0.154 [+0.067, +0.233] | +0.075 [-0.008, +0.154] |
| hybrid-choice - title+dense | +0.142 [+0.079, +0.204] | +0.217 [+0.158, +0.275] | +0.146 [+0.092, +0.200] |

## recall@5 by question type

| method | bridge_comparison (n) | comparison (n) | compositional (n) | inference (n) |
|---|---|---|---|---|
| dense | 0.600 (15) | 1.000 (15) | 0.667 (15) | 0.667 (15) |
| title+dense | 0.600 (15) | 1.000 (15) | 0.667 (15) | 0.667 (15) |
| graph | 0.817 (15) | 0.833 (15) | 0.900 (15) | 0.867 (15) |
| hybrid | 0.850 (15) | 1.000 (15) | 0.967 (15) | 0.933 (15) |
| graph-choice | 0.850 (15) | 0.867 (15) | 0.900 (15) | 0.933 (15) |
| hybrid-choice | 0.867 (15) | 1.000 (15) | 0.967 (15) | 0.967 (15) |
