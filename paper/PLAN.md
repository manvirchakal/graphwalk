# Paper plan

Status: draft plan, 2026-10-02. Nothing here has been run yet.

> **Update after the literature search ([RELATED_WORK.md](RELATED_WORK.md)):** the hop
> framing and path probability are prior art (SR 2022, MINERVA 2018), and the working
> title below overclaims. The paper is an empirical study: path confidence as selective
> prediction across deciders (not found published, but Ca2KG and CPR are close), the
> memorization control for agents, and cost trade-offs. Before writing, add the controls
> listed there: option-order shuffling, letter-mass diagnostics, ECE/temperature
> scaling, a fairer stated-confidence baseline.

## The claim

Working title: *Hops as calibrated classification: cheap, confidence-scored question
answering over knowledge graphs, and what it does not buy.*

One sentence: if each hop of a KG walk is a constrained classification over the current
node's relations (plus STOP), made by a small calibrated classifier, the walk is 4–6×
cheaper than an LLM writing the query and its path probability tells right answers from
wrong ones (AUROC 0.92–0.97). It is **less accurate** than the LLM, and as an agent tool
it saves a strong agent about 25% of its cost without changing accuracy.

This is an empirical paper with an open-source system and honest negative results. It
is not a state-of-the-art KG-QA paper, and should not try to be one: on WebQSP and CWQ we
sit well below RoG/ToG-style methods.

## What a reviewer will attack, and what answers it

| Attack | Current answer | Needed |
|---|---|---|
| "The contribution is just Jev, a closed hosted model." | None. Every Jev number depends on one vendor. | **P1**: the same walk with an open decider, to show the classification framing (not Jev) gives the calibration. |
| "The LLM decider's confidence is unfair: it states scores, it doesn't use log-probabilities." | True: `llm_decider.py` normalizes stated 0–100 scores. | **P2**: an LLM decider that reads option-letter log-probabilities. If it is as calibrated as Jev, the calibration claim shrinks to "any token-probability decider". We must know before a reviewer does. |
| "WebQSP/MetaQA are memorized." | Closed book scores 0.42 (cheap agent) and 0.57 (strong agent) F1. | **P3**: the same graph with entity names replaced, so neither the agent nor the walk can rely on memory. |
| "n=30 for the agent result." | CI on cost excludes 0, but F1 is unresolved. | **P4**: A8b on 100 questions, the walk and graph-tool arms only. |
| "One seed, 50–300 questions." | Bootstrap CIs. | **P5**: walk vs LLM path writer on ≥ 500 questions per setting (walks cost ~$0.0002 each), 3 seeds where the decider is sampled. |
| "Where is the comparison to published KG-QA?" | Cited numbers (CWQ 0.63–0.69). | Cite RoG, ToG, KG-Agent numbers in one table, with our cost per question next to theirs where they report it. No reimplementation. |
| "Calibration claims need more than AUROC." | ECE, Brier exist in `figure_calibration.py`. | **P0**: risk–coverage curves (selective accuracy), AURC, reliability diagrams. Free: from committed scores. |

## Experiments, in order

| id | what | cost | blocks |
|---|---|---|---|
| P0 | Risk–coverage, AURC, reliability diagrams from existing runs (E3, A2, A6). | $0 | nothing |
| P1 | Open decider: a local cross-encoder reranker scores each (question, path-so-far, relation) option; softmax with one temperature fit on MetaQA dev. MetaQA 1–3 hop and WebQSP, 300 q each. | $0 (CPU) | a small backend class |
| P2 | Log-prob LLM decider: options as letters, one token, `top_logprobs`, through an OpenRouter model that returns them. Same sets as P1. | ~$1–2 | checking which OpenRouter models return logprobs |
| P5 | Walk vs LLM path writer on 500 q per setting (MetaQA 2/3-hop, WebQSP), plus 2 more seeds of the walk. | ~$1–2 | nothing |
| P3 | Renamed-entity WebQSP graph (names → stable random tokens, relations kept): closed book, walk, graph-tool agent, walk-tool agent, cheap agent, 100 q. | ~$1 cheap agent; +$4 with the strong agent on 30 q | an anonymizer for the graph and gold answers |
| P4 | Strong agent, graph vs graph+walk, 70 more questions (total 100). | ~$5–6 | budget |

Total ≈ $10–13. Credits are $7.84 with a $1 floor, so P4 needs a top-up (about $10 to
run everything with margin). P0, P1 and P5 fit now.

Decision rules, set before running so the results can't steer the story:

- If P1's open decider reaches AUROC ≥ 0.85 on MetaQA 2–3 hop, the paper's claim is the
  framing, with Jev as the fastest instance. If it doesn't, the claim narrows to "a
  calibrated classifier as decider", and Jev's role is stated plainly as a dependency.
- If P2's log-prob LLM decider matches Jev's AUROC, the calibration advantage claim is
  dropped; the paper keeps cost and latency (7–10× faster per decision).
- If P3 erases the agent-tool cost saving, the agent section is reported as a negative
  result.
- P4: report whatever comes out, including if the −25% shrinks.

## Paper outline (8 pages + appendix, ACL format)

1. Introduction: KG-QA with LLMs is accurate and expensive; confidence is the missing
   piece for routing. Contributions: the framing, the calibration finding, the agent-tool
   finding, the negative results, the library.
2. Related work: KG-QA (semantic parsing; RoG, ToG, KG-Agent, StructGPT), LLM
   calibration and selective prediction, tool-using agents over KGs, GraphRAG.
3. Method: hop as classification; relation hops vs entity hops; path probability as
   confidence; escalation.
4. Setup: datasets (MetaQA, WebQSP, CWQ, Freebase global graph), deciders (Jev, open
   reranker, LLM stated, LLM log-prob), baselines (LLM path writer, multi-step RAG,
   agents), metrics (F1, hits@1, AUROC, AURC, $/q, latency).
5. Results: (a) accuracy and cost vs the LLM path writer; (b) calibration and selective
   prediction; (c) escalation; (d) as an agent's tool, including the renamed graph.
6. Negative results: constraints (CWQ), text-built graphs (RAG wins), RAG add-on, router.
7. Limitations: hosted decider, sample sizes, Freebase-only for large graphs, model
   versions drift.
Appendix: prompts, per-setting tables, costs, the reproduction commands.

All tables are generated by `scripts/paper/*` from committed scores, as today.

## Venue and process

- **Target:** an ACL-family workshop (knowledge-graph, structured-data, or
  efficiency/trustworthiness workshops of EMNLP/ACL/NAACL) or the main conference
  through ACL Rolling Review if P1–P4 come out well. Workshops are the realistic first
  target: they welcome negative results and systems with careful evaluation.
- **arXiv:** post at submission time. A first-time cs.CL submitter usually needs an
  endorsement; a workshop acceptance or a co-author with arXiv history solves it.
- **Software:** a JOSS paper for the library is a separate, short track and suits it.
- **Authorship and disclosure:** one human author. State in the acknowledgements that
  the code and experiments were developed with an AI coding assistant, as ACL's policy
  on AI writing assistance asks. No affiliation is needed ("Independent researcher").

## Timeline

| week | work |
|---|---|
| 1 | P0, P1, P5; anonymizer for P3 |
| 2 | P2, P3, P4 (after top-up); freeze numbers |
| 3 | draft sections 1–6 in LaTeX under `paper/`; figures from scripts |
| 4 | related work pass, limitations, a read by someone outside the project, submit |
