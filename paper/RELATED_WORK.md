# Related work and novelty check

Status: literature search, 2026-10-02. Every entry was located at the URL given; entries
marked *(abstract)* were checked from the abstract only, *(unverified)* not checked.
Re-check OpenReview and recent workshop papers just before submission: the area moves
fast (several of the closest papers are from 2026).

## Verdict per claim

| Our claim | Status | Closest prior | How to frame it |
|---|---|---|---|
| Hop = classification over the node's relations + STOP, path probability = product of step probabilities | **Not novel** | SR (Zhang 2022): END relation, p(path) = ∏ p(r_t); MINERVA (Das 2018): softmax over outgoing edges, NO_OP stay action, beam ranked by path probability | "Building on SR and MINERVA". Our difference: the decider is zero-shot and swappable (hosted classifier or any LLM's letter log-probabilities), not trained per dataset. |
| A small/cheap model per hop instead of an LLM writing the path | **Not novel** | EffiQA, KG-CoT, ToG's BM25/SBERT pruning ablation, Shrestha & Kim 2025 (6.7M edge scorer), RouterKGQA (1.15 LLM calls/q) | Ours is a measured cost/accuracy trade-off against an LLM path writer, including where it loses. |
| Path probability as selective-prediction confidence (AUROC, AURC, risk–coverage) | **Not found published; crowded neighbourhood** | Ca2KG (WWW 2026): selective accuracy–coverage on MetaQA/WebQSP, but LLM-prompted confidence; CPR (ICML 2026): path-level scores + conformal sets, learned scorer; UAG (AAAI 2025); DoublyCal (IJCAI 2026) | "To our knowledge, among the first to evaluate per-hop path probability as a selective-prediction signal for KGQA." Never "first uncertainty for KGQA". |
| Escalation on low confidence | **Not novel as a mechanism** | SkewRoute (Findings EMNLP 2025): small→large LLM routing in KG-RAG on retrieval-score skew; Dekoninck 2025 cascades | Ours is a different signal (decider path confidence); compare to SkewRoute if we keep the escalation section. |
| Letter log-probabilities beat stated scores as confidence | **Expected, partly contested** | Kadavath 2022, Xiong 2024, Kapoor 2024 agree; Tian 2023 finds the opposite for RLHF API models; Kim & Kang 2026: the winner flips with the protocol | A confirming result in a new setting (per-hop, ≤20 options); state the protocol exactly. |
| Open LLM via letter log-probs matches a dedicated classifier; model choice matters (Gemma overconfident, DeepSeek inaccurate) | **Plausibly new in this setting** | Zhuang 2024 (label log-probs for reranking) for the mechanism | Present as the cross-decider comparison; needs the controls below. |
| Renamed entities: closed book 0.42→0.01; graph tools beat search by 0.25+ F1 only without real names | **Partly known** | ARoG (AAAI 2026): anonymized WebQSP/CWQ breaks ToG (privacy framing); Mandarapu 2026: KG grounding helps mostly out-of-training; Gashkov 2025: masked URIs for SPARQL memorization; Longpre 2021: entity substitution | New part: graph tools vs search for the **same agent**, tied with names, separated without. No such comparison found. Address KG-LLM-Bench (pseudonyms change accuracy 0.2%, but in-context subgraphs, not memorizable questions). |
| Adding a walk tool cuts a strong agent's cost 17% at equal F1 | **New in its specifics; principle known** | AWO meta-tools (2026): composite tools cut calls ≤11.9%; Middleware (Gu 2024): KG tools for accuracy; KGVoyager 2026 | A tool ablation with a CI; workshop-level evidence (n=100, one benchmark). |

**Overall:** the paper is an empirical study, not a method paper. Its defensible
contributions are (1) the selective-prediction evaluation of path confidence across
deciders and datasets, including where it fails (2Wiki); (2) the cross-decider
comparison (hosted classifier, open LLM log-probs, stated scores); (3) the memorization
control for agents, graph tools vs search; (4) honest cost/accuracy trade-offs; (5) the
library. The working title's "hops as calibrated classification" reads as a method claim
and should change.

## What reviewers will now ask for

Ordered by how likely they are to sink the paper.

1. **Option-order control.** Our options are presented in a fixed order (sorted by
   direction and relation name, STOP in a fixed slot). Letter/position bias in
   multiple-choice LLM answers is well documented (Zheng 2024; Pezeshkpour 2024), and
   worst exactly where the model is torn, i.e. where confidence matters. Run 3–5
   random orderings per step on MetaQA 3-hop and WebQSP; report accuracy and AUROC
   spread, and optionally PriDe debiasing. Cheap (~$1).
2. **Letter-mass diagnostics.** How much first-token probability falls outside the
   offered letters, how often the top token isn't a valid letter (we count `no_letter`
   already; report it). Wang 2024 shows first-token answers can disagree with text
   answers.
3. **Calibration, not only ranking.** ECE, Brier and reliability diagrams for every
   decider, plus temperature scaling fit on a held-out split, since AUROC ignores scale
   and Tian 2023 / Kim & Kang 2026 measure with ECE.
4. **A fairer stated-confidence baseline.** One confidence for the chosen option, or
   verbal confidence, or sampling consistency, not only 0–100 per option. Otherwise the
   log-prob vs stated gap looks like a strawman.
5. **Ca2KG-style baseline** on MetaQA/WebQSP selective accuracy, or at least their
   numbers side by side where the setups match.
6. **Cite and contrast** SR and MINERVA in the method section, CPR and Ca2KG in the
   confidence section, SkewRoute for escalation, ARoG for anonymization.

## References by topic

### Stepwise relation walkers (pre-LLM)
- Zhang et al. **SR: Subgraph Retrieval Enhanced Model for Multi-hop KBQA.** ACL 2022. https://aclanthology.org/2022.acl-long.396/ — END relation, p(r|q)=σ(s(q,r)−s(q,END)), path probability as product, beam search. SR+NSM WebQSP/CWQ Hits@1 68.9/50.2.
- Das et al. **Go for a Walk and Arrive at the Answer (MINERVA).** ICLR 2018. https://arxiv.org/abs/1711.05851 — RL policy over outgoing edges, NO_OP, beam ranked by path probability.
- Xiong et al. **DeepPath.** EMNLP 2017. https://arxiv.org/abs/1707.06690
- Qiu et al. **Stepwise Reasoning Network (SRN).** WSDM 2020. https://doi.org/10.1145/3336191.3371812
- Zhou et al. **IRN.** COLING 2018. https://aclanthology.org/C18-1171/
- Shi et al. **TransferNet.** EMNLP 2021. https://aclanthology.org/2021.emnlp-main.341
- He et al. **NSM.** WSDM 2021. https://arxiv.org/abs/2101.03737
- Jiang et al. **UniKGQA.** ICLR 2023. https://arxiv.org/abs/2212.00959
- Sun et al. **PullNet.** EMNLP 2019. https://aclanthology.org/D19-1242/ ; **GraftNet.** EMNLP 2018. https://arxiv.org/abs/1809.00782

### LLM-era traversal, path generation, KG agents
- Sun et al. **Think-on-Graph.** ICLR 2024. https://arxiv.org/abs/2307.07697 — ≤2ND+D+1 LLM calls; WebQSP/CWQ 76.2/58.9 (ChatGPT), 82.6/69.5 (GPT-4).
- Ma et al. **Think-on-Graph 2.0.** ICLR 2025. https://arxiv.org/abs/2407.10805
- Luo et al. **Reasoning on Graphs (RoG).** ICLR 2024. https://arxiv.org/abs/2310.01061 — 85.7/62.6 in its own table (GNN-RAG's table lists 80.0/57.8; cite the source used).
- Jiang et al. **StructGPT.** EMNLP 2023. https://arxiv.org/abs/2305.09645 — 72.6/54.3 (as reported by KG-Agent).
- Jiang et al. **KG-Agent.** ACL 2025. https://aclanthology.org/2025.acl-long.468/ — 83.3/72.2.
- Mavromatis & Karypis. **GNN-RAG.** arXiv 2024. https://arxiv.org/abs/2405.20139 — 80.6/61.7; Table 4 gives LLM calls and input tokens per question.
- Gu et al. **Pangu: Don't Generate, Discriminate.** ACL 2023. https://aclanthology.org/2023.acl-long.270
- Li et al. **KB-BINDER.** ACL 2023. https://arxiv.org/abs/2305.01750
- Luo et al. **ChatKBQA.** Findings ACL 2024. https://arxiv.org/abs/2310.08975
- Xiong et al. **Interactive-KBQA.** ACL 2024. https://arxiv.org/abs/2402.15131
- Tan et al. **Paths-over-Graph.** arXiv 2024. https://arxiv.org/abs/2410.14211
- Sui et al. **FiDeLiS.** Findings ACL 2025. https://arxiv.org/abs/2405.13873
- Wang et al. **ReKnoS.** ICLR 2025. https://arxiv.org/abs/2503.22166
- Dong et al. **EffiQA.** COLING 2025. https://arxiv.org/abs/2406.01238 — small plug-in model prunes per hop.
- Zhao et al. **KG-CoT.** IJCAI 2024. https://ijcai.org/proceedings/2024/734 — small stepwise reasoner, fewer API calls.
- Liu et al. **KELP.** Findings ACL 2024. https://arxiv.org/abs/2406.13862
- Ma et al. **Debate on Graph.** AAAI 2025. https://arxiv.org/abs/2409.03155
- Jin et al. **Graph-CoT.** Findings ACL 2024. https://arxiv.org/abs/2404.07103
- Shrestha & Kim. **Efficient Multi-Hop QA over KGs via LLM Planning and Embedding-Guided Search.** arXiv 2025. https://arxiv.org/abs/2511.19648 — closest cost competitor (MetaQA).
- Yuan et al. **RouterKGQA.** arXiv 2026. https://arxiv.org/abs/2603.20017
- Huang et al. **Less is More: smaller LMs as subgraph retrievers.** Findings EMNLP 2024. https://arxiv.org/abs/2410.06121
- Sun et al. **Search-on-Graph.** KDD 2026. https://arxiv.org/abs/2510.08825 — fewer LLM calls than ToG but more input tokens.
- Gu et al. **Middleware for LLMs.** EMNLP 2024. https://arxiv.org/abs/2402.14672
- Liu et al. **AgentBench.** ICLR 2024. https://arxiv.org/abs/2308.03688 (KG environment)
- **BYOKG-RAG.** EMNLP 2025. https://aclanthology.org/2025.emnlp-main.1417.pdf *(abstract)*
- Wisam et al. **KGVoyager.** arXiv 2026. https://arxiv.org/abs/2609.01780
- Abuzakuk et al. **Optimizing Agentic Workflows using Meta-tools (AWO).** arXiv 2026. https://arxiv.org/abs/2601.22037

### Confidence, selective prediction, routing in KGQA
- Ren et al. **Ca2KG: When to Trust.** WWW 2026. https://arxiv.org/abs/2601.09241 — closest: selective accuracy–coverage on MetaQA/WebQSP.
- Lin et al. **Conformal Path Reasoning.** ICML 2026. https://arxiv.org/abs/2605.08077
- Ni et al. **UAG: Towards Trustworthy KG Reasoning.** AAAI 2025. https://arxiv.org/abs/2410.08985
- Lu et al. **Double-Calibration (DoublyCal).** IJCAI 2026. https://arxiv.org/abs/2601.11956
- Wang et al. **SkewRoute.** Findings EMNLP 2025. https://aclanthology.org/2025.findings-emnlp.606
- Dekoninck et al. **A Unified Approach to Routing and Cascading for LLMs.** ICML 2025. https://arxiv.org/abs/2410.10347 *(first author unverified)*
- Zhao et al. **SAUP.** ACL 2025. https://arxiv.org/abs/2412.01033 — per-step uncertainty along agent trajectories.
- Hou et al. **Selective Temporal KG Reasoning.** arXiv 2024. https://arxiv.org/abs/2404.01695
- Patidar et al. **GrailQAbility.** arXiv 2022. https://arxiv.org/abs/2212.10189 *(first author unverified)*
- Kamath, Jia & Liang. **Selective QA under Domain Shift.** ACL 2020. https://aclanthology.org/2020.acl-main.503
- Geifman & El-Yaniv. **Selective Classification for Deep Neural Networks.** NeurIPS 2017. https://arxiv.org/abs/1705.08500 ; Geifman et al. ICLR 2019 https://arxiv.org/abs/1805.08206 (AURC; check the PDF before citing for it)

### LLM confidence: token probabilities vs stated, multiple-choice bias
- Kadavath et al. **Language Models (Mostly) Know What They Know.** arXiv 2022. https://arxiv.org/abs/2207.05221
- Tian et al. **Just Ask for Calibration.** EMNLP 2023. https://arxiv.org/abs/2305.14975 — contrary result for RLHF models.
- Xiong et al. **Can LLMs Express Their Uncertainty?** ICLR 2024. https://arxiv.org/abs/2306.13063
- Kapoor et al. **LLMs Must Be Taught to Know What They Don't Know.** NeurIPS 2024. https://arxiv.org/abs/2406.08391
- Tao et al. **Revisiting Uncertainty Estimation and Calibration of LLMs.** arXiv 2025. https://arxiv.org/abs/2505.23854
- Kim & Kang. **Same Answer, Different Confidence.** arXiv 2026. https://arxiv.org/abs/2605.27752
- Zheng et al. **LLMs Are Not Robust Multiple Choice Selectors.** ICLR 2024. https://arxiv.org/abs/2309.03882
- Pezeshkpour & Hruschka. **LLMs Sensitivity to the Order of Options.** Findings NAACL 2024. https://arxiv.org/abs/2308.11483
- Wang et al. **"My Answer is C".** Findings ACL 2024. https://arxiv.org/abs/2402.14499
- Guda et al. **Quantifying and Mitigating Selection Bias in LLMs.** arXiv 2025. https://arxiv.org/abs/2511.21709
- Zhao et al. **Calibrate Before Use.** ICML 2021. https://arxiv.org/abs/2102.09690
- Zhuang et al. **Beyond Yes and No.** NAACL 2024. https://arxiv.org/abs/2310.14122
- OpenAI. **GPT-4 Technical Report.** 2023. https://arxiv.org/abs/2303.08774 — post-training reduces calibration.
- Mei et al. **Do Reasoning Models Know When They Don't Know?** arXiv 2025. https://arxiv.org/abs/2506.18183
- Surveys: Geng et al. NAACL 2024 https://aclanthology.org/2024.naacl-long.366 ; Shorinwa et al. 2024 https://arxiv.org/abs/2412.05563 ; Liu et al. 2025 https://arxiv.org/abs/2503.15850

### Memorization and anonymized KGs
- Ning et al. **Privacy-protected RAG for KGQA (ARoG).** AAAI 2026. https://arxiv.org/abs/2508.08785 — closest to P3.
- Tan et al. **PrivGemo.** arXiv 2026. https://arxiv.org/abs/2601.08739
- Gashkov et al. **SPARQL generation: measuring training-data memorization.** ICWE 2025. https://arxiv.org/abs/2507.13859
- Markowitz et al. **KG-LLM-Bench.** KnowledgeNLP @ NAACL 2025. https://arxiv.org/abs/2504.07087 — counterpoint: pseudonyms barely matter in-context.
- Mandarapu & Kunkunuru. **KG Grounding Helps LLMs Only for Out-of-Training Knowledge.** arXiv 2026. https://arxiv.org/abs/2606.22419
- Longpre et al. **Entity-Based Knowledge Conflicts in QA.** EMNLP 2021. https://aclanthology.org/2021.emnlp-main.565
- Zhang et al. **Diagnosing Pitfalls in KG-RAG Datasets (KGQAGen).** NeurIPS 2025 D&B. https://arxiv.org/abs/2505.23495 — ~57% of WebQSP/CWQ-style answers verified correct.
- Dammu et al. **Dynamic-KGQA.** arXiv 2025. https://arxiv.org/abs/2503.05049
- **CoLoTa.** arXiv 2025. https://arxiv.org/abs/2504.14462
- Sui et al. **OKGQA.** ACL 2025. https://aclanthology.org/2025.acl-long.622/
