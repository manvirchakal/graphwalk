# Documentation

What is here now, and the plan for user docs (roadmap Phase 5).

## Now: the internal record

| File | What it is |
| --- | --- |
| [`design.md`](design.md) | The design record: architecture, protocols, and "as built" notes per milestone. Kept as the internal reference; it is not user documentation. |
| [`results-m5.md`](results-m5.md) | Traversal over curated graphs (MetaQA, 2Wiki) vs vector and multi-step RAG. |
| [`results-m6.md`](results-m6.md) | Ingestion quality on 2Wiki paragraphs. |
| [`results-m7.md`](results-m7.md) | QA over ingested text graphs (2Wiki, HotpotQA), walk-then-read-source, and FanOutQA. |
| [`../results/README.md`](../results/README.md) | Which run backs which number. |

## Plan: user docs for v0.1

A small MkDocs Material site, built in CI and published to GitHub Pages. Pages, in
the order a new user needs them:

1. **What graphwalk is and when to use it.** The graph as an index over text:
   `locate` returns positions and `read` returns the text. It states the measured
   trade-offs plainly, including where multi-step RAG wins.
2. **Quickstart.** Library: index a folder, `locate`, `read`. MCP: stdio in five
   lines of client config.
3. **Providers and keys.** One table of OpenRouter, TypeSafe, OpenAI, Anthropic,
   and x.ai by role (decisions, LLM, embeddings), with environment variables,
   headers, base-URL overrides, and the LLM-as-decider fallback.
4. **MCP server.** stdio and remote HTTP, tool reference, server auth (none, bearer,
   OAuth), the base-URL allowlist, request limits, and Docker.
5. **Ingestion.** Sources, incremental re-ingestion, costs, and checkpoints.
6. **Python API reference**, generated from docstrings.
7. **Reproducing the paper.** One command per table.

`design.md` and the results files stay in the repo, linked from an "Internals"
section rather than rewritten.
