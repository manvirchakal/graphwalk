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

# E1: evidence retrieval runs and the k each is reported at.
EVIDENCE = {
    "2Wiki": ("20260930T124411Z-2wiki-evidence", 5),
    "HotpotQA": ("20260930T124743Z-hotpotqa-evidence", 5),
}

# Curated graphs: label -> [(run, system) per seed]; the first label is the reference.
_JEV = "graphwalk-relation-v2-jev"
CURATED = {
    "MetaQA 1-hop": {
        "graphwalk (Jev)": [("decider/20260930T123542Z-metaqa-1hop", _JEV),
                            ("decider/20260930T124219Z-metaqa-1hop", _JEV),
                            ("decider/20260930T124616Z-metaqa-1hop", _JEV)],
        "vector RAG": [("20260928T192614Z-metaqa-1hop", "vector-rag")],
        "multi-step RAG": [("20260929T000051Z-metaqa-1hop", "iter-rag")],
    },
    "MetaQA 2-hop": {
        "graphwalk (Jev)": [("decider/20260930T123641Z-metaqa-2hop", _JEV),
                            ("decider/20260930T124319Z-metaqa-2hop", _JEV),
                            ("decider/20260930T124715Z-metaqa-2hop", _JEV)],
        "vector RAG": [("20260928T194721Z-metaqa-2hop", "vector-rag")],
        "multi-step RAG": [("20260929T002404Z-metaqa-2hop", "iter-rag")],
    },
    "MetaQA 3-hop": {
        "graphwalk (Jev)": [("decider/20260930T123821Z-metaqa-3hop", _JEV),
                            ("decider/20260930T124452Z-metaqa-3hop", _JEV),
                            ("decider/20260930T124852Z-metaqa-3hop", _JEV)],
        "vector RAG": [("20260928T200403Z-metaqa-3hop", "vector-rag")],
        "multi-step RAG": [("20260929T005913Z-metaqa-3hop", "iter-rag")],
    },
    "2Wiki gold-evidence graph": {
        "graphwalk (Jev)": [("decider/20260930T123900Z-2wiki-gold", "graphwalk-greedy-v2-jev"),
                            ("decider/20260930T124528Z-2wiki-gold", "graphwalk-greedy-v2-jev"),
                            ("decider/20260930T124928Z-2wiki-gold", "graphwalk-greedy-v2-jev")],
        "vector RAG": [("20260928T204354Z-2wiki-gold", "vector-rag")],
        "multi-step RAG": [("20260929T012321Z-2wiki-gold", "iter-rag")],
    },
}  # fmt: skip

# E3: decider -> (run, system) per dataset, seed 0.
CALIBRATION = {
    dataset: {"Jev": systems["graphwalk (Jev)"][0]} for dataset, systems in CURATED.items()
}
