"""Table: QA over graphs built from text (2Wiki, HotpotQA; M7 and its follow-up).

    uv run python scripts/paper/table_text_qa.py

F1 with 95% bootstrap CIs over questions, and each system's paired F1 difference
from multi-step RAG (the strongest baseline). FanOutQA is in table_fanout.py.
"""

from common import by_system, cost_per_1k, p50, paired, scores, table, with_ci
from runs import TEXT_QA

LABELS = {
    "graphwalk": "graphwalk, graph only",
    "graphwalk-reader": "graphwalk + reader (extracted facts)",
    "graphwalk-reader-source": "graphwalk + reader (source paragraphs)",
    "text-rag": "RAG (top 5)",
    "text-iter-rag": "multi-step RAG",
}
REFERENCE = "text-iter-rag"


def main() -> None:
    for dataset, runs in TEXT_QA.items():
        systems = by_system(row for run in runs for row in scores(run))
        reference = systems[REFERENCE]
        rows = [
            [
                label,
                with_ci([r["f1"] for r in systems[name]]),
                paired(systems[name], reference, lambda r: r["f1"]) if name != REFERENCE else "—",
                cost_per_1k(systems[name]),
                p50(systems[name]),
            ]
            for name, label in LABELS.items()
            if name in systems
        ]
        print(f"### {dataset} (n = {len(reference)})\n")  # noqa: T201
        print(table(["system", "F1 [95% CI]", "Δ F1 vs multi-step RAG", "$/1k q", "p50 s"],  # noqa: T201
                    rows))  # fmt: skip
        print()  # noqa: T201


if __name__ == "__main__":
    main()
