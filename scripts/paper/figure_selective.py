"""P0: selective prediction. If a walk answers only when confident, how accurate is it?

    uv run python scripts/paper/figure_selective.py

Per dataset and decider, questions are sorted by confidence (most confident first) and
answered down to a coverage level. Reported: AUROC; AURC, the area under the
risk-coverage curve (risk = 1 - accuracy; lower is better), next to the AURC of a random
order (the error rate) and of a perfect order (the floor for this accuracy); and
accuracy at 25/50/75/100% coverage. Walks without an answer count as confidence 0.

Correct means EM on MetaQA (the answer set exactly) and hits@1 on WebQSP/CWQ (the
standard there). The curve points go to ``paper/figures/selective.csv`` for plotting.
"""

import csv

from common import ROOT, by_system, mean, scores, table
from runs import CALIBRATION, KGQA_CALIBRATION

from graphwalk.eval.calibration import auroc

COVERAGES = (0.25, 0.5, 0.75, 1.0)


def pairs(run: str, system: str, metric: str) -> list[tuple[float, float]]:
    return [(r["confidence"] or 0.0, float(r[metric])) for r in by_system(scores(run))[system]]


def risks(data: list[tuple[float, float]]) -> list[float]:
    """Risk at each coverage k/n, k = 1..n, most confident first (ties keep run order)."""
    ordered = sorted(data, key=lambda p: -p[0])
    out: list[float] = []
    wrong = 0.0
    for k, (_, y) in enumerate(ordered, start=1):
        wrong += 1.0 - y
        out.append(wrong / k)
    return out


def oracle_aurc(data: list[tuple[float, float]]) -> float:
    return mean(risks([(y, y) for _, y in data]))


def main() -> None:
    sources = [
        *((dataset, decider, run, system, "em")
          for dataset, deciders in CALIBRATION.items()
          for decider, (run, system) in deciders.items()),
        *((dataset, decider, run, system, "hits1")
          for dataset, deciders in KGQA_CALIBRATION.items()
          for decider, (run, system) in deciders.items()),
    ]  # fmt: skip
    rows: list[list[str]] = []
    curve: list[list[str]] = []
    for dataset, decider, run, system, metric in sources:
        data = pairs(run, system, metric)
        rs = risks(data)
        n = len(data)
        at = [1.0 - rs[max(1, round(c * n)) - 1] for c in COVERAGES]
        rows.append(
            [dataset, decider, str(n), f"{auroc(data):.3f}", f"{mean(rs):.3f}",
             f"{1.0 - mean([y for _, y in data]):.3f}", f"{oracle_aurc(data):.3f}",
             *(f"{a:.3f}" for a in at)]
        )  # fmt: skip
        curve.extend([dataset, decider, f"{k / n:.4f}", f"{r:.4f}"] for k, r in enumerate(rs, 1))
    print(table(["dataset", "decider", "n", "AUROC", "AURC", "AURC random",  # noqa: T201
                 "AURC perfect", *(f"acc @{c:.0%}" for c in COVERAGES)], rows))  # fmt: skip
    out = ROOT / "paper" / "figures" / "selective.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["dataset", "decider", "coverage", "risk"])
        writer.writerows(curve)


if __name__ == "__main__":
    main()
