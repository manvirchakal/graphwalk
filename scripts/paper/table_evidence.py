"""Table (E1): evidence retrieval. Recall of gold evidence documents per method.

    uv run python scripts/paper/table_evidence.py

Reads each E1 run's rows.jsonl (see scripts/eval/evidence.py). recall@k with 95%
bootstrap CIs at the dataset's middle k (5 paragraphs; 10 pages for FanOutQA), and
the paired difference of hybrid-choice from dense.
"""

import sys

from common import RESULTS, mean, table
from runs import EVIDENCE

from graphwalk.eval.evidence import EvidenceRow, complete_at, recall_at
from graphwalk.eval.metrics import bootstrap_ci

METHODS = {
    "dense": "dense (bge-small, titled chunks)",
    "title+dense": "control: named titles, then dense",
    "graph": "graph locate (name linking)",
    "graph-choice": "graph locate (Jev linking)",
    "hybrid": "hybrid (name linking)",
    "hybrid-choice": "hybrid (Jev linking)",
    "multi-step": "multi-step RAG's retrievals (M7)",
}


def ci(values: list[float]) -> str:
    lo, hi = bootstrap_ci(values)
    return f"{mean(values):.3f} [{lo:.3f}, {hi:.3f}]"


def main() -> None:
    columns: dict[str, dict[str, str]] = {m: {} for m in METHODS}
    header = ["method"]
    for dataset, (run, k) in EVIDENCE.items():
        lines = (RESULTS / run / "rows.jsonl").read_text(encoding="utf-8").splitlines()
        rows = [EvidenceRow.model_validate_json(line) for line in lines]
        header += [f"{dataset} recall@{k}", f"complete@{k}"]
        dense = {r.question_id: r for r in rows if r.method == "dense"}
        for method in METHODS:
            group = [r for r in rows if r.method == method]
            if not group:
                columns[method][dataset] = "n/a | n/a"
                continue
            recall = [recall_at(r.gold, r.at(k), k) for r in group]
            complete = mean([complete_at(r.gold, r.at(k), k) for r in group])
            columns[method][dataset] = f"{ci(recall)} | {complete:.3f}"
        diffs = [
            recall_at(r.gold, r.at(k), k)
            - recall_at(dense[r.question_id].gold, dense[r.question_id].at(k), k)
            for r in sorted(rows, key=lambda r: r.question_id)  # as paired_table orders them
            if r.method == "hybrid-choice"
        ]
        lo, hi = bootstrap_ci(diffs)
        delta = f"{mean(diffs):+.3f} [{lo:+.3f}, {hi:+.3f}]"
        print(f"{dataset}: hybrid-choice - dense, recall@{k}: {delta} (n = {len(dense)})",  # noqa: T201
              file=sys.stderr)  # fmt: skip
    body = [[label, *" | ".join(columns[m][d] for d in EVIDENCE).split(" | ")]
            for m, label in METHODS.items()]  # fmt: skip
    print(table(header, body))  # noqa: T201


if __name__ == "__main__":
    main()
