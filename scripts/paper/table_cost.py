"""Table (E6): cost and latency, with ingestion amortized over N queries. No new calls.

    uv run python scripts/paper/table_cost.py

Per-query cost is the provider-reported mean over each run's questions; ingestion is
the one-off cost of building the graph from text (curated graphs have none). Total cost
of answering N queries = ingestion + N * per-query cost. "Break-even" is the N above
which graphwalk's total is lower than the reference system's, when its per-query cost
is lower at all.
"""

import json
import math

from common import RESULTS, by_system, mean, p50, scores, table
from runs import COST_ROWS, FANOUT, INGEST_COST

from graphwalk.eval.datasets import fanoutqa

NS = (1_000, 10_000, 100_000, 1_000_000)


def per_query(run: str, system: str) -> tuple[float, float, str]:
    """(mean $ per query, mean accuracy, p50 latency). Accuracy is F1, or FanOutQA's
    loose accuracy for the FanOutQA run."""
    rows = by_system(scores(run))[system]
    ok = [r for r in rows if r["status"] != "error" and r["cost_usd"] is not None]
    if run == FANOUT:
        reference = json.loads((RESULTS / FANOUT / "reference.json").read_text("utf-8"))
        accuracy = [fanoutqa.answer_in_text(reference[r["question_id"]],
                                            " | ".join(r["answers"]))[0] for r in rows]  # fmt: skip
    else:
        accuracy = [r["f1"] for r in rows]
    return mean([r["cost_usd"] for r in ok]), mean(accuracy), p50(rows)


def main() -> None:
    out: list[list[str]] = []
    for dataset, entries in COST_ROWS.items():
        ingest = INGEST_COST.get(dataset)
        reference = None
        for label, run, system, graph in entries:
            cost, f1, latency = per_query(run, system)
            fixed = (ingest or 0.0) if graph else 0.0
            totals = [f"${fixed + n * cost:,.2f}" for n in NS]
            if reference is None:
                reference = (cost, fixed)
                even = "reference"
            else:
                saving = reference[0] - cost
                if saving <= 0:
                    even = "never"
                elif fixed <= reference[1]:
                    even = "always"
                else:
                    even = f"{math.ceil((fixed - reference[1]) / saving):,}"
            out.append(
                [dataset, label, f"{1000 * cost:.3f}", f"{f1:.3f}", latency,
                 "—" if not fixed else f"${fixed:.2f}", *totals, even]
            )  # fmt: skip
    header = ["dataset", "system", "$/1k q", "F1 / loose acc", "p50 s", "ingestion",
              *(f"total @ {n:,} q" for n in NS), "break-even N vs reference"]  # fmt: skip
    print(table(header, out))  # noqa: T201


if __name__ == "__main__":
    main()
