# M6 results: ingestion quality on 2Wiki paragraphs

**Setup.** 40 walkable 2Wiki dev questions (seed 0). All their context paragraphs,
gold and distractor, are de-duplicated by title: 309 documents, 316 chunks. They are
ingested into an empty graph by `scripts/eval/ingest_2wiki.py`. Extraction uses
`gpt-6-luna`, run once and cached, so all three variants share identical extractions
and differ only in routing. The graph is then scored against the 40 questions' gold
evidence triples (definitions are in `graphwalk/eval/ingest_eval.py`). Run:
`results/20260929T020624Z-2wiki-ingest/`.

The routing prompt was revised once, on a separate seed-99 sample of 10 questions
(`results/tuning/`). The first prompt ("a similar name is not enough") made Jev pick
NEW for 7 mentions whose name was identical to a candidate's, e.g. "Gabriel Range"
at p = 0.9. The revised prompt says an equivalent name with a compatible type is the
same entity unless something contradicts it. That cut split entities from 15% to 12%
on the tuning sample. Seed 0 was not looked at until the prompt was fixed.

| routing | nodes | entity recall | split entities | over-merged | triple recall | complete chains | routing calls | routing cost |
|---|---|---|---|---|---|---|---|---|
| exact name | 1,653 | 0.880 | 0.029 | 0 | 0.688 | 0.425 (17/40) | 0 | $0 |
| Jev | 1,586 | 0.880 | 0.058 | 1 | **0.775** | **0.600 (24/40)** | 280 Jev | $0.037 |
| Jev + escalation to `gpt-6-luna` (p < 0.6) | 1,586 | 0.880 | 0.049 | 0 | **0.775** | **0.600 (24/40)** | 280 Jev + 19 LLM | $0.040 |
| Jev + escalation to `mimo-v2.6-pro` (p < 0.6) | 1,583 | 0.880 | 0.039 | 0 | **0.775** | **0.600 (24/40)** | 280 Jev + 13 LLM | $0.050 |

The last row is a separate run
(`results/20260929T115429Z-2wiki-ingest/`) with the same cached extractions. It
escalates to `xiaomi/mimo-v2.6-pro` on OpenRouter ($0.435/$0.87 per M tokens,
Artificial Analysis Intelligence Index 46 vs. 21–37 for `gpt-6-luna`), which was the
best price-per-intelligence option listed on 2026-09-29.

Extraction for all 316 chunks cost $0.125: 300 calls, the rest cached from the tuning
runs. That is about $0.0004 per paragraph and 17 minutes at 18 requests per minute.
Extraction dominates cost: Jev routing adds about 30% on top.

## What this says

- **Jev routing is worth it.** It merges name variants that exact matching can't:
  "Gustav Adolph, Count of Nassau-Saarbrücken" = "Count Gustav Adolf of
  Nassau-Saarbrücken", "Catherine II" = "Catherine the Great of Russia", "Toronto Film
  Festival" = "Toronto International Film Festival". That raises triple recall by 9
  points and complete reasoning chains from 17 to 24 of 40 questions, which is what
  multi-hop QA over this graph (M7) needs. Cost is about $0.00012 per paragraph.
- **The cost is a few more duplicates.** 5.8% of found gold entities have more than
  one node, versus 2.9% for exact matching, because Jev still sometimes answers NEW
  for an identically named candidate. The metric favours exact matching by
  construction: it only counts duplicates with the *same* normalized name. Exact
  matching's own misses (different spellings) don't show up here, but they do show
  up as 67 more nodes and lower triple recall.
- **Escalation helps judgment, not the headline metrics.** Originally, escalation went
  to the *extraction* model (`gpt-6-luna`), because the pipeline had one LLM slot. That
  was a design flaw: a weaker general model was second-guessing Jev. The escalation
  model is now configurable (`--escalation-model`). With `mimo-v2.6-pro`, the 13
  escalated decisions look better on inspection:
  - It merged "YouTube" and "Viscount Northland" into their identically named nodes.
    Jev had leaned NEW on both, and Viscount Northland was one of the missed triples.
  - It kept "Princess Louise Caroline of Hesse-Kassel" (1789–1867) apart from "Louise
    of Hesse-Kassel", her daughter, the Queen of Denmark, whom `gpt-6-luna` had wrongly
    merged her into.
  - It is still debatable on titles versus houses: it folded "Duke of
    Schleswig-Holstein-Sonderburg-Glücksburg" into the house node.

  Split entities fell from 4.9% to 3.9%, but triple recall and complete chains did not
  move. Cost: $0.0132 for 13 calls, about $0.001 per call, since it spends about 780
  reasoning tokens per answer. That adds about 35% to routing cost. Latency is the
  bigger cost: about 17 s per call, sequential with routing, so this run took 388 s
  vs. 165 s with `gpt-6-luna`. Use it when quality matters more than ingestion time.
- **Jev is not deterministic near the threshold.** The two escalation runs sent
  different decisions to the LLM (19 vs. 13) from identical inputs, because Jev's
  probabilities for borderline cases vary from run to run. Differences of a few
  duplicates between single runs are within that noise.
- **Over-merge detection is weak.** Only merges of two gold entities known to be
  distinct (both ends of one triple) are counted. Spot checks on the tuning sample
  found plausible-but-wrong merges this can't see: "Venice" → "Republic of Venice",
  "County of Saarbrücken" → "Nassau-Saarbrücken", and "Robert Dundas of Arniston" →
  "…, the younger" (several men had that name). A precision audit needs hand labels,
  which I haven't done.

## Where the missing 22% of triples go (Jev variant, 18 misses)

- **Object missing (8):** mostly gold naming, not extraction. "West Orange" vs "West
  Orange, New Jersey", "America" vs "United States", "Henry I" vs "Henry the Fowler",
  and "Mission" vs "Mission, British Columbia". The scorer is strict, so true recall is
  somewhat higher.
- **Both entities found, no edge (6):** the relation was not extracted, or was attached
  to a duplicate node. Examples: Donna Summer → Naples (place of death), Mariamne →
  Herod the Great (spouse).
- **Subject missing (4):** the entity is under another name or was never extracted.

## Caveats

- One small sample (40 questions, 309 paragraphs), a small extraction model, and one
  seed. The differences between variants are 7 chains out of 40 and 9 points of
  triple recall. They are consistent across the tuning and main samples, but they
  are not tight estimates.
- There is no QA here: whether 60% complete chains turns into answers is M7.
- Jev itself is not perfectly deterministic. A re-run of the Jev variant from the same
  cache reproduced these counts (18 misses), but individual close decisions can flip
  (see the escalation counts above).
- Cost check: the OpenRouter key's usage went from $2.673 to $2.766 around the MiMo
  run, which is $0.093. The run reported $0.050, and a smoke-test call $0.00006. The
  difference most likely comes from a Jev-only diagnostic re-run (about $0.04)
  finishing just before the first reading and being billed late. I have not verified
  that.
