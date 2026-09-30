"""Which run (a directory under results/) feeds which paper table. The one place to
update when a run is superseded."""

# M7 text-derived graphs (seed 0): graph-only, readers, and the text baselines.
TEXT_QA = {
    "2WikiMultiHopQA": ["20260929T154958Z-2wiki-text", "20260929T180802Z-2wiki-walkread"],
    "HotpotQA": ["20260929T171954Z-hotpotqa-text", "20260929T182414Z-hotpotqa-walkread"],
}
FANOUT = "20260929T211957Z-fanoutqa"

# One-off cost of building each text-derived graph (extraction, routing, escalation),
# as measured in M7 (docs/results-m7.md). The 2Wiki figure is approximate: its run was
# resumed across container restarts, and the cost was measured from key usage.
INGEST_COST = {"2WikiMultiHopQA": 0.60, "HotpotQA": 0.78, "FanOutQA": 2.0}

# E6: (label, run, system, reads a text-derived graph). The first row per dataset is
# the reference for break-even.
COST_ROWS = {
    "2WikiMultiHopQA": [
        ("multi-step RAG", "20260929T154958Z-2wiki-text", "text-iter-rag", False),
        ("RAG", "20260929T154958Z-2wiki-text", "text-rag", False),
        ("graphwalk + reader (source)", "20260929T180802Z-2wiki-walkread",
         "graphwalk-reader-source", True),
        ("graphwalk, graph only", "20260929T154958Z-2wiki-text", "graphwalk", True),
    ],
    "HotpotQA": [
        ("multi-step RAG", "20260929T171954Z-hotpotqa-text", "text-iter-rag", False),
        ("RAG", "20260929T171954Z-hotpotqa-text", "text-rag", False),
        ("graphwalk + reader (source)", "20260929T182414Z-hotpotqa-walkread",
         "graphwalk-reader-source", True),
        ("graphwalk, graph only", "20260929T171954Z-hotpotqa-text", "graphwalk", True),
    ],
    "FanOutQA": [
        ("multi-step RAG", FANOUT, "text-iter-rag", False),
        ("RAG", FANOUT, "text-rag", False),
        ("graphwalk + reader (source)", FANOUT, "graphwalk-reader-source", True),
        ("graphwalk + reader (facts)", FANOUT, "graphwalk-reader-facts", True),
    ],
}  # fmt: skip
