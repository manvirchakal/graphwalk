# Results index

Each run directory holds a committed `summary.md` (and, for FanOutQA, the
`reference.json` answers). The full per-question `results.json` files are large and
gitignored; rerunning the listed command regenerates them. Runs are named
`<UTC timestamp>-<dataset>-<variant>`. When a milestone was rerun, the later run
supersedes the earlier one, and the results doc says which is current.

| Run | Produced by | Commit | Reported in | Status |
| --- | --- | --- | --- | --- |
| `20260928T192614Z-metaqa-{1,2,3}hop` (3 runs to `200403Z`) | `graphwalk eval` | `2116376` | `docs/results-m5.md` | superseded (untuned presets) |
| `20260928T204354Z-2wiki-gold`, `204458Z-2wiki-resolve` | `graphwalk eval` | `84f77d1` | `docs/results-m5.md` | superseded |
| `tuning/20260928T2211..2231Z-metaqa-*-dev` | `scripts/eval/tune_metaqa_dev.py` | `79be712` | `docs/results-m5.md` (tuning) | dev-set only |
| `20260928T223503Z-metaqa-{1,2,3}hop`, `224239Z-2wiki-gold` | `graphwalk eval` | `79be712` | `docs/results-m5.md` | superseded by the RAG-baseline runs |
| `20260928T234219Z-2wiki-resolve`, `-resolve-best`, `-choice` | `graphwalk eval` | `4bb6c13` | `docs/results-m5.md` (linking) | current |
| `heldout/20260928T2350..2352Z-2wiki-*` | `graphwalk eval` (held-out seed) | `2160232` | `docs/results-m5.md` (linking) | current |
| `20260929T000051Z-metaqa-{1,2,3}hop` (to `005913Z`), `012321Z-2wiki-gold` | `graphwalk eval` | `a9e768b` | `docs/results-m5.md` | **current M5** |
| `tuning/20260929T0157..0159Z-2wiki-ingest` | `scripts/eval/ingest_2wiki.py` (seed 99) | `25c8bc5` | `docs/results-m6.md` (prompt tuning) | tuning only |
| `20260929T020624Z-2wiki-ingest` | `scripts/eval/ingest_2wiki.py` | `25c8bc5` | `docs/results-m6.md` | **current M6** (exact, Jev, Jev + first escalation model) |
| `20260929T115429Z-2wiki-ingest` | `scripts/eval/ingest_2wiki.py` | `4c872a9` | `docs/results-m6.md` | **current M6** (Jev + `mimo-v2.6-pro` escalation row) |
| `tuning/20260929T121541Z-2wiki-text` | `scripts/eval/text_qa.py` | `a5a1c11` | `docs/results-m7.md` (smoke run) | tuning only |
| `20260929T154958Z-2wiki-text`, `171954Z-hotpotqa-text` | `scripts/eval/text_qa.py` | `1d2925f`, `34d3854` | `docs/results-m7.md` | **current M7** |
| `20260929T180802Z-2wiki-walkread`, `182414Z-hotpotqa-walkread` | `scripts/eval/walk_read.py` | `d4439c7` | `docs/results-m7.md` (step 1) | **current** |
| `tuning/20260929T200331Z-fanoutqa` | `scripts/eval/fanout.py` (dev slice) | `3c3e0db` | `docs/results-m7.md` (step 2) | tuning only |
| `20260929T211957Z-fanoutqa` | `scripts/eval/fanout.py` | `3c3e0db` | `docs/results-m7.md` (step 2) | **current** |
| `20260930T12*-{2wiki,hotpotqa,fanoutqa}-evidence` | `scripts/eval/evidence.py` | phase 4 | `docs/results-phase4.md` (E1) | **current** |
| `20260930T123344Z-metaqa-{1,2,3}hop-llmpath` | `scripts/eval/query_writer.py` | phase 4 | `docs/results-phase4.md` (E2) | **current** |
| `tuning/20260930T122636Z-metaqa-*-llmpath-dev` | `scripts/eval/query_writer.py --split dev --n 20` | phase 4 | E2 prompt check | dev only |
| `decider/20260930T1235..1249Z-*` | `scripts/eval/decider.py --deciders jev --seed 0/1/2` | phase 4 | `docs/results-phase4.md` (E3/E4) | **current** |
| `decider/20260930T140255Z-metaqa-1hop` | `scripts/eval/decider.py --deciders llm` | phase 4 | `docs/results-phase4.md` (E3) | **current** (1-hop only) |
| `decider/20260930T15*..16*Z-*` (LLM decider, 2–3 hop and 2Wiki) | `scripts/eval/decider.py --deciders llm` | phase 4 | `docs/results-phase4.md` (E3) | **current** |
| `2026093*-*-llmpath`, `*-extractor`, `*-ragaddon` | `scripts/eval/query_writer*.py`, `extractor.py`, `rag_addon.py` | phase 4 | `docs/results-phase4.md` (E2, E2b, E5, E7; each heading names its run) | **current** (later timestamp wins) |
| `20261001T153517Z-scale`, `164752Z-scale` | `scripts/eval/scale.py` | phase 7 | `docs/results-phase7.md` (A5; before and after the hub fix) | **current** |
| `kgqa/20261001T153829Z-webqsp`, `160324Z-cwq` | `scripts/eval/kgqa.py` | phase 7 | `docs/results-phase7.md` (A2 pilots) | **current** |
| `kgqa/20261001T172038Z-webqsp-global` | `scripts/eval/kgqa_global.py` | phase 7 | `docs/results-phase7.md` (A6) | **current** |
| `router/20261001T174239Z` | `scripts/eval/router_analysis.py` (no API calls) | phase 7 | `docs/results-phase7.md` (A7) | **current** |
| `agent/20261001T182902Z-webqsp-agent` | `scripts/eval/agent_arms.py --pilot 20` | phase 7 | `docs/results-phase7.md` (A8 pilot) | superseded |
| `agent/20261001T195538Z-webqsp-agent` | `scripts/eval/agent_arms.py` (cheap agent, 100 q) | phase 7 | `docs/results-phase7.md` (A8) | **current** |
| `agent/20261001T214718Z-webqsp-agent` | `scripts/eval/agent_arms.py --pilot 30` (strong agent) | phase 7 | `docs/results-phase7.md` (A8b) | **current** |

Each run also has a committed `scores.jsonl` (one row per system and question), which
the paper scripts in `scripts/paper/` read; older runs got theirs from
`scripts/paper/export_scores.py`.

Scripts ran from the repo root; the commits above are where each summary was added.
Each section heading in the results docs names the script and run behind its table.
