"""A8b + P4: does a walk tool make a strong agent cheaper? Pooled over all 100 questions.

    uv run python scripts/paper/table_agent.py

Per arm: F1, hits@1, cost and latency per question; then paired differences (walk arm
minus graph-tools arm, same questions) with 95% bootstrap CIs, overall and per batch
(A8b's first 30 questions, P4's other 70), since the two batches ran a day apart.
"""

from common import by_system, mean, scores, table, with_ci
from runs import STRONG_AGENT

from graphwalk.eval.metrics import bootstrap_ci, percentile

type Rows = dict[str, dict[str, float]]


def arm(runs: list[tuple[str, str]]) -> tuple[Rows, dict[str, int]]:
    """question id -> metrics, and question id -> batch index."""
    rows: Rows = {}
    batch: dict[str, int] = {}
    for i, (run, system) in enumerate(runs):
        for r in by_system(scores(run))[system]:
            keys = ("f1", "hits1", "cost_usd", "latency_s")
            rows[r["question_id"]] = {k: float(r[k] or 0.0) for k in keys}
            batch[r["question_id"]] = i
    return rows, batch


def paired(a: Rows, b: Rows, ids: list[str], key: str, digits: int) -> str:
    diffs = [a[q][key] - b[q][key] for q in ids]
    lo, hi = bootstrap_ci(diffs)
    return f"{mean(diffs):+.{digits}f} [{lo:+.{digits}f}, {hi:+.{digits}f}]"


def main() -> None:
    arms = {label: arm(runs) for label, runs in STRONG_AGENT.items()}
    rows = []
    for label, (data, _) in arms.items():
        v = list(data.values())
        rows.append([label, str(len(v)), with_ci([x["f1"] for x in v]),
                     f"{mean([x['hits1'] for x in v]):.2f}",
                     f"{mean([x['cost_usd'] for x in v]):.4f}",
                     f"{percentile([x['latency_s'] for x in v], 50):.1f}"])  # fmt: skip
    print(table(["arm", "n", "F1 [95% CI]", "hits@1", "$/q", "p50 s"], rows))  # noqa: T201
    (graph, batch), (walk, _) = arms["graph tools"], arms["graph tools + walk"]
    common = sorted(set(graph) & set(walk))
    out = []
    for name, ids in (("all", common), ("A8b (first 30)", [q for q in common if batch[q] == 0]),
                      ("P4 (other 70)", [q for q in common if batch[q] == 1])):  # fmt: skip
        base = mean([graph[q]["cost_usd"] for q in ids])
        cost = mean([walk[q]["cost_usd"] - graph[q]["cost_usd"] for q in ids])
        out.append([name, str(len(ids)), paired(walk, graph, ids, "f1", 3),
                    paired(walk, graph, ids, "cost_usd", 4), f"{cost / base:+.0%}",
                    paired(walk, graph, ids, "latency_s", 1)])  # fmt: skip
    print("\nPaired, walk arm minus graph-tools arm\n")  # noqa: T201
    print(table(["questions", "n", "F1", "$/q", "$/q relative", "latency s"], out))  # noqa: T201


if __name__ == "__main__":
    main()
