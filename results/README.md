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

Scripts ran from the repo root; the commits above are where each summary was added.
The script paths are as of Phase 0 (they lived directly in `scripts/` before).
Paper tables will get their own one-command scripts under `scripts/paper/` (roadmap
Phase 4).
