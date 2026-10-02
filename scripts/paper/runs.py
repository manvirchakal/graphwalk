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
    "FanOutQA": ("20260930T125415Z-fanoutqa-evidence", 10),
}

# Curated graphs: label -> [(run, system) per seed]; the first label is the reference.
_JEV = "graphwalk-relation-v2-jev"
CURATED = {
    "MetaQA 1-hop": {
        "graphwalk (Jev)": [("decider/20260930T123542Z-metaqa-1hop", _JEV),
                            ("decider/20260930T124219Z-metaqa-1hop", _JEV),
                            ("decider/20260930T124616Z-metaqa-1hop", _JEV)],
        "graphwalk (LLM decider)": [("decider/20260930T140255Z-metaqa-1hop",
                                     "graphwalk-relation-v2-llm")],
        "LLM writes the path": [("20260930T123344Z-metaqa-1hop-llmpath", "llm-path")],
        "LLM writes the path, 2 retries": [
            ("20260930T123344Z-metaqa-1hop-llmpath", "llm-path-retry"),
            ("20260930T161926Z-metaqa-1hop-llmpath", "llm-path-retry"),
            ("20260930T170545Z-metaqa-1hop-llmpath", "llm-path-retry"),
        ],
        "vector RAG": [
            ("20260928T192614Z-metaqa-1hop", "vector-rag"),
            ("seeds/20260930T170540Z-metaqa-1hop", "vector-rag"),
            ("seeds/20260930T175143Z-metaqa-1hop", "vector-rag"),
        ],
        "multi-step RAG": [("20260929T000051Z-metaqa-1hop", "iter-rag")],
    },
    "MetaQA 2-hop": {
        "graphwalk (Jev)": [("decider/20260930T123641Z-metaqa-2hop", _JEV),
                            ("decider/20260930T124319Z-metaqa-2hop", _JEV),
                            ("decider/20260930T124715Z-metaqa-2hop", _JEV)],
        "LLM writes the path": [("20260930T123344Z-metaqa-2hop-llmpath", "llm-path")],
        "LLM writes the path, 2 retries": [
            ("20260930T123344Z-metaqa-2hop-llmpath", "llm-path-retry"),
            ("20260930T161926Z-metaqa-2hop-llmpath", "llm-path-retry"),
            ("20260930T170545Z-metaqa-2hop-llmpath", "llm-path-retry"),
        ],
        "graphwalk (LLM decider)": [
            ("decider/20260930T150921Z-metaqa-2hop", "graphwalk-relation-v2-llm")
        ],
        "vector RAG": [("20260928T194721Z-metaqa-2hop", "vector-rag")],
        "multi-step RAG": [("20260929T002404Z-metaqa-2hop", "iter-rag")],
    },
    "MetaQA 3-hop": {
        "graphwalk (Jev)": [("decider/20260930T123821Z-metaqa-3hop", _JEV),
                            ("decider/20260930T124452Z-metaqa-3hop", _JEV),
                            ("decider/20260930T124852Z-metaqa-3hop", _JEV)],
        "LLM writes the path": [("20260930T123344Z-metaqa-3hop-llmpath", "llm-path")],
        "LLM writes the path, 2 retries": [
            ("20260930T123344Z-metaqa-3hop-llmpath", "llm-path-retry"),
            ("20260930T161926Z-metaqa-3hop-llmpath", "llm-path-retry"),
            ("20260930T170545Z-metaqa-3hop-llmpath", "llm-path-retry"),
        ],
        "graphwalk (LLM decider)": [
            ("decider/20260930T155339Z-metaqa-3hop", "graphwalk-relation-v2-llm")
        ],
        "vector RAG": [("20260928T200403Z-metaqa-3hop", "vector-rag")],
        "multi-step RAG": [("20260929T005913Z-metaqa-3hop", "iter-rag")],
    },
    "2Wiki gold-evidence graph": {
        "graphwalk (Jev)": [("decider/20260930T123900Z-2wiki-gold", "graphwalk-greedy-v2-jev"),
                            ("decider/20260930T124528Z-2wiki-gold", "graphwalk-greedy-v2-jev"),
                            ("decider/20260930T124928Z-2wiki-gold", "graphwalk-greedy-v2-jev")],
        "graphwalk (LLM decider)": [
            ("decider/20260930T161921Z-2wiki-gold", "graphwalk-greedy-v2-llm")
        ],
        "vector RAG": [("20260928T204354Z-2wiki-gold", "vector-rag")],
        "multi-step RAG": [("20260929T012321Z-2wiki-gold", "iter-rag")],
    },
}  # fmt: skip

# E3: decider -> (run, system) per dataset, seed 0.
CALIBRATION = {
    dataset: {
        label.removeprefix("graphwalk (").removesuffix(")"): systems[label][0]
        for label in ("graphwalk (Jev)", "graphwalk (LLM decider)")
        if label in systems
    }
    for dataset, systems in CURATED.items()
}

# Freebase KG-QA (A2 pilots, A6 global graph): label -> decider -> (run, system).
KGQA_CALIBRATION = {
    "WebQSP": {"Jev": ("kgqa/20261001T153829Z-webqsp", "graphwalk-relation-v2-jev"),
               "LLM decider": ("kgqa/20261001T153829Z-webqsp", "graphwalk-relation-v2-llm")},
    "CWQ": {"Jev": ("kgqa/20261001T160324Z-cwq", "graphwalk-relation-v2-jev"),
            "LLM decider": ("kgqa/20261001T160324Z-cwq", "graphwalk-relation-v2-llm")},
    "WebQSP, one graph": {"Jev": ("kgqa/20261001T172038Z-webqsp-global", "graphwalk-kgqa-jev")},
}  # fmt: skip

# P1: the same walks with an open-weights model's token probabilities as the decider
# (graphwalk.eval.logprob_decider). label -> (run, system, correctness metric).
_LP = "graphwalk-relation-v2-logprob"
LOGPROB_CALIBRATION = {
    "MetaQA 1-hop": ("decider/20261002T021309Z-metaqa-1hop", _LP, "em"),
    "MetaQA 2-hop": ("decider/20261002T021917Z-metaqa-2hop", _LP, "em"),
    "MetaQA 3-hop": ("decider/20261002T022745Z-metaqa-3hop", _LP, "em"),
    "2Wiki gold-evidence graph": ("decider/20261002T024030Z-2wiki-gold",
                                  "graphwalk-greedy-v2-logprob", "em"),
    "WebQSP": ("kgqa/20261002T024219Z-webqsp", _LP, "hits1"),
    "CWQ": ("kgqa/20261002T024438Z-cwq", _LP, "hits1"),
}  # fmt: skip

# A8b + P4: strong agent (Gemini 3.1 Pro), WebQSP agent graph, n=100 sample (seed 0).
# A8b ran the first 30 questions; P4 the other 70, one arm per run.
STRONG_AGENT = {
    "graph tools": [("agent/20261001T214718Z-webqsp-agent", "agent-graph"),
                    ("agent/20261002T025127Z-webqsp-agent", "agent-graph")],
    "graph tools + walk": [("agent/20261001T214718Z-webqsp-agent", "agent-walk"),
                           ("agent/20261002T030706Z-webqsp-agent", "agent-walk")],
}  # fmt: skip

# A8 (cheap agent, real names) vs P3 (the same 100 questions, entity names replaced by
# aliases): arm -> (A8 run, P3 run), system name as in both.
_A8 = "agent/20261001T195538Z-webqsp-agent"
CHEAP_AGENT_ANON = {
    "walk alone (no agent)": ("jev", _A8, "agent/20261002T144100Z-webqsp-agent-anon"),
    "closed book": ("closedbook", _A8, "agent/20261002T144313Z-webqsp-agent-anon"),
    "graph tools": ("agent-graph", _A8, "agent/20261002T145824Z-webqsp-agent-anon"),
    "graph tools + walk": ("agent-walk", _A8, "agent/20261002T151303Z-webqsp-agent-anon"),
    "search (RAG)": ("agent-search", _A8, "agent/20261002T154650Z-webqsp-agent-anon"),
}
