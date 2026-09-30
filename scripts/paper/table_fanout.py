"""Table: FanOutQA (set- and map-valued answers over Wikipedia pages; M7 step 2).

    uv run python scripts/paper/table_fanout.py

FanOutQA's accuracy (share of reference strings found in the answer; strict = all
found), with 95% bootstrap CIs and the paired difference from multi-step RAG.
"""

import json

from common import RESULTS, by_system, cost_per_1k, p50, paired, scores, table, with_ci
from runs import FANOUT

from graphwalk.eval.datasets import fanoutqa

LABELS = {
    "graphwalk-reader-facts": "graphwalk + reader (extracted facts)",
    "graphwalk-reader-source": "graphwalk + reader (source pages)",
    "text-rag": "RAG (top 10 pages)",
    "text-iter-rag": "multi-step RAG",
}


def main() -> None:
    reference = json.loads((RESULTS / FANOUT / "reference.json").read_text(encoding="utf-8"))
    systems = by_system(scores(FANOUT))
    for rows in systems.values():
        for row in rows:
            loose, strict = fanoutqa.answer_in_text(
                reference[row["question_id"]], " | ".join(row["answers"])
            )
            row["loose"], row["strict"] = loose, float(strict)
    base = systems["text-iter-rag"]
    out = [
        [label, with_ci([r["loose"] for r in systems[name]]),
         with_ci([r["strict"] for r in systems[name]]),
         paired(systems[name], base, lambda r: r["loose"]) if name != "text-iter-rag" else "—",
         cost_per_1k(systems[name]), p50(systems[name])]
        for name, label in LABELS.items()
    ]  # fmt: skip
    print(f"### FanOutQA dev (n = {len(base)})\n")  # noqa: T201
    print(table(["system", "loose acc [95% CI]", "strict acc [95% CI]",  # noqa: T201
                 "Δ loose vs multi-step RAG", "$/1k q", "p50 s"], out))  # fmt: skip


if __name__ == "__main__":
    main()
