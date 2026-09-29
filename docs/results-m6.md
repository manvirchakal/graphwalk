# M6 results: ingestion quality on 2Wiki paragraphs

**Setup.** 40 walkable 2Wiki dev questions (seed 0). All their context paragraphs,
gold and distractor, are de-duplicated by title: 309 documents, 316 chunks. They are
ingested into an empty graph by `scripts/ingest_2wiki.py`. Extraction uses
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
| Jev + LLM escalation (p < 0.6) | 1,586 | 0.880 | 0.049 | 0 | **0.775** | **0.600 (24/40)** | 280 Jev + 19 LLM | $0.040 |

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
- **LLM escalation is marginal.** 19 of 280 decisions went to the LLM. That removed
  the one detected wrong merge and a few duplicates, and changed recall not at all.
  It's cheap enough to keep as the default, but it's not where the gains are.
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
  cache reproduced these counts (18 misses), but individual close decisions can flip.
