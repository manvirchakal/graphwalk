# 2wiki evidence retrieval (E1)

- params: `{"extractor": "openrouter/openai/gpt-6-luna", "n": 60}`
- git: `a6ce2306d1d50a268541eac86ab1ad140efe6a3e`
- environment: `{"git_dirty": true, "git_sha": "a6ce2306d1d50a268541eac86ab1ad140efe6a3e", "packages": {"fastembed": "0.8.1", "graphwalk": "0.1.0", "litellm": "1.103.0", "networkx": "3.7", "numpy": "2.5.3", "pydantic": "2.13.5", "typesafe-sdk": "0.7.2"}, "python": "3.12.3", "timestamp": "2026-09-30T18:13:32.949951+00:00"}`
- run at: 2026-09-30T18:13:32.949998+00:00

## Recall of gold evidence documents

| method | recall@2 | recall@5 | recall@10 | 95% CI @5 | complete@5 | docs returned | Jev calls/q | $/1k q |
|---|---|---|---|---|---|---|---|---|
| dense | 0.683 | 0.733 | 0.812 | [0.675, 0.792] | 0.417 | 10.0 | 0.00 | 0.000 |
| title+dense | 0.658 | 0.733 | 0.812 | [0.675, 0.792] | 0.417 | 10.0 | 0.00 | 0.000 |
| graph | 0.733 | 0.871 | 0.887 | [0.812, 0.925] | 0.717 | 3.2 | 3.53 | 0.184 |
| hybrid | 0.775 | 0.946 | 0.963 | [0.912, 0.975] | 0.833 | 9.9 | 3.53 | 0.184 |
| graph-choice | 0.762 | 0.908 | 0.917 | [0.863, 0.950] | 0.783 | 3.3 | 4.68 | 0.241 |
| hybrid-choice | 0.796 | 0.958 | 0.975 | [0.929, 0.983] | 0.867 | 9.9 | 4.68 | 0.241 |

## Paired differences (recall@k, 95% bootstrap CI)

| a - b | Δ recall@2 | Δ recall@5 | Δ recall@10 |
|---|---|---|---|
| hybrid - dense | +0.092 [+0.029, +0.158] | +0.212 [+0.154, +0.271] | +0.150 [+0.096, +0.208] |
| hybrid-choice - dense | +0.113 [+0.050, +0.175] | +0.225 [+0.167, +0.283] | +0.163 [+0.108, +0.217] |
| graph-choice - dense | +0.079 [+0.008, +0.150] | +0.175 [+0.096, +0.254] | +0.104 [+0.025, +0.179] |
| graph-choice - graph | +0.029 [+0.004, +0.062] | +0.037 [+0.000, +0.083] | +0.029 [-0.004, +0.075] |
| graph-choice - title+dense | +0.104 [+0.042, +0.171] | +0.175 [+0.096, +0.254] | +0.104 [+0.025, +0.179] |
| hybrid-choice - title+dense | +0.138 [+0.079, +0.200] | +0.225 [+0.167, +0.283] | +0.163 [+0.108, +0.217] |

## recall@5 by question type

| method | bridge_comparison (n) | comparison (n) | compositional (n) | inference (n) |
|---|---|---|---|---|
| dense | 0.600 (15) | 1.000 (15) | 0.667 (15) | 0.667 (15) |
| title+dense | 0.600 (15) | 1.000 (15) | 0.667 (15) | 0.667 (15) |
| graph | 0.850 (15) | 0.800 (15) | 0.933 (15) | 0.900 (15) |
| hybrid | 0.883 (15) | 1.000 (15) | 0.933 (15) | 0.967 (15) |
| graph-choice | 0.867 (15) | 0.867 (15) | 0.933 (15) | 0.967 (15) |
| hybrid-choice | 0.900 (15) | 1.000 (15) | 0.933 (15) | 1.000 (15) |
