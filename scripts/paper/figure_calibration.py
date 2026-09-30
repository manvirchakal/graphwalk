"""Figure data (E3): is a walk's confidence calibrated? Jev vs an LLM decider.

    uv run python scripts/paper/figure_calibration.py

A walk's confidence is the probability of its best path. Per dataset and decider:
ECE, Brier score, AUROC for telling right from wrong answers (EM), and accuracy on the
most confident half. Then reliability-diagram bins pooled over the curated datasets
(the data behind the paper's calibration figure).
"""

from common import by_system, mean, scores, table
from runs import CALIBRATION

from graphwalk.eval.calibration import auroc, ece

BINS = 10


def pairs(run: str, system: str) -> list[tuple[float, float]]:
    """(confidence, EM) per question; walks without an answer count as confidence 0."""
    return [(r["confidence"] or 0.0, float(r["em"])) for r in by_system(scores(run))[system]]


def main() -> None:
    summary: list[list[str]] = []
    pooled: dict[str, list[tuple[float, float]]] = {}
    for dataset, deciders in CALIBRATION.items():
        for decider, (run, system) in deciders.items():
            data = pairs(run, system)
            pooled.setdefault(decider, []).extend(data)
            brier = mean([(c - y) ** 2 for c, y in data])
            top = sorted(data, key=lambda p: -p[0])[: len(data) // 2]
            summary.append(
                [dataset, decider, str(len(data)), f"{mean([y for _, y in data]):.3f}",
                 f"{mean([c for c, _ in data]):.3f}", f"{ece(data):.3f}", f"{brier:.3f}",
                 f"{auroc(data):.3f}", f"{mean([y for _, y in top]):.3f}"]
            )  # fmt: skip
    print(table(["dataset", "decider", "n", "EM", "mean confidence", "ECE", "Brier",  # noqa: T201
                 "AUROC", "EM, most confident half"], summary))  # fmt: skip
    print("\nReliability bins, pooled over datasets (confidence range: n, mean confidence, EM)\n")  # noqa: T201
    rows: list[list[str]] = []
    for b in range(BINS):
        lo, hi = b / BINS, (b + 1) / BINS
        cells = [f"{lo:.1f}-{hi:.1f}"]
        for data in pooled.values():
            members = [(c, y) for c, y in data if lo <= c < hi or (hi == 1.0 and c == 1.0)]
            cells.append(
                f"{len(members)}: {mean([c for c, _ in members]):.2f} / "
                f"{mean([y for _, y in members]):.2f}"
                if members
                else "0"
            )
        rows.append(cells)
    print(table(["confidence", *pooled], rows))  # noqa: T201


if __name__ == "__main__":
    main()
