"""P7: calibration error, before and after recalibration.

    uv run python scripts/paper/table_recalibration.py

AUROC (P0, P1) says whether confidence *ranks* answers; it ignores scale. This table
says whether the numbers mean what they say. Per dataset and decider, from the committed
scores of P0/P1 (plus the larger WebQSP runs of P5 and P6):

* ECE (10 equal-width bins) and Brier score of the raw confidence;
* the same after Platt scaling (a logistic fit of correctness on log-confidence),
  cross-fitted: fit on half the questions, scored on the other half, both ways, averaged
  over 10 random splits. It is monotone, so AUROC is unchanged;
* the Brier score of always predicting the base rate (no skill), and the Brier skill
  score after scaling (1 - Brier / base Brier; 0 = no better than the base rate).

Correct means EM on MetaQA and 2Wiki, hits@1 on WebQSP/CWQ, as in P0. Walks without an
answer count as confidence 0 (log-confidence clamped at 1e-6).
"""

import math
import random

from common import by_system, mean, scores, table
from runs import CALIBRATION, KGQA_CALIBRATION, LARGE_CALIBRATION, LOGPROB_CALIBRATION

from graphwalk.eval.calibration import ece

SPLITS = 10
FLOOR = 1e-6


def brier(pairs: list[tuple[float, float]]) -> float:
    return mean([(c - y) ** 2 for c, y in pairs])


def platt(train: list[tuple[float, float]]) -> tuple[float, float]:
    """Fit p = sigmoid(a * log c + b) by Newton's method, with a small ridge on ``a`` and
    ``b`` so a perfectly separable split stays finite."""
    a, b, ridge = 1.0, 0.0, 1e-2
    xs = [math.log(max(c, FLOOR)) for c, _ in train]
    ys = [y for _, y in train]
    for _ in range(50):
        ga = gb = haa = hab = hbb = 0.0
        for x, y in zip(xs, ys, strict=True):
            p = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, a * x + b))))
            w = p * (1.0 - p)
            ga += (p - y) * x
            gb += p - y
            haa += w * x * x
            hab += w * x
            hbb += w
        ga += ridge * a
        gb += ridge * b
        haa += ridge
        hbb += ridge
        det = haa * hbb - hab * hab
        if det <= 0:
            break
        da = (hbb * ga - hab * gb) / det
        db = (haa * gb - hab * ga) / det
        a, b = a - da, b - db
        if abs(da) + abs(db) < 1e-9:  # noqa: PLR2004
            break
    return a, b


def apply(
    params: tuple[float, float], pairs: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    a, b = params
    return [
        (1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, a * math.log(max(c, FLOOR)) + b)))), y)
        for c, y in pairs
    ]


def cross_fit(pairs: list[tuple[float, float]]) -> tuple[float, float]:
    """Mean (ECE, Brier) of Platt-scaled confidence on held-out halves."""
    eces: list[float] = []
    briers: list[float] = []
    for split in range(SPLITS):
        order = list(range(len(pairs)))
        random.Random(split).shuffle(order)  # noqa: S311
        halves = [[pairs[i] for i in order[::2]], [pairs[i] for i in order[1::2]]]
        for fit, held in (halves, halves[::-1]):
            scaled = apply(platt(fit), held)
            eces.append(ece(scaled))
            briers.append(brier(scaled))
    return mean(eces), mean(briers)


def main() -> None:
    sources = [
        *((dataset, decider, run, system, "em")
          for dataset, deciders in CALIBRATION.items()
          for decider, (run, system) in deciders.items()),
        *((dataset, decider, run, system, "hits1")
          for dataset, deciders in KGQA_CALIBRATION.items()
          for decider, (run, system) in deciders.items()),
        *((dataset, "open model, token probabilities", run, system, metric)
          for dataset, (run, system, metric) in LOGPROB_CALIBRATION.items()),
        *((dataset, decider, run, system, metric)
          for dataset, deciders in LARGE_CALIBRATION.items()
          for decider, (run, system, metric) in deciders.items()),
    ]  # fmt: skip
    order = list(dict.fromkeys(dataset for dataset, *_ in sources))
    sources.sort(key=lambda source: order.index(source[0]))
    rows: list[list[str]] = []
    for dataset, decider, run, system, metric in sources:
        data = [(r["confidence"] or 0.0, float(r[metric])) for r in by_system(scores(run))[system]]
        rate = mean([y for _, y in data])
        base = rate * (1.0 - rate)
        scaled_ece, scaled_brier = cross_fit(data)
        rows.append(
            [dataset, decider, str(len(data)), f"{rate:.3f}",
             f"{mean([c for c, _ in data]):.3f}", f"{ece(data):.3f}", f"{scaled_ece:.3f}",
             f"{brier(data):.3f}", f"{scaled_brier:.3f}", f"{base:.3f}",
             f"{1.0 - scaled_brier / base:+.2f}" if base else "n/a"]
        )  # fmt: skip
    print(table(["dataset", "decider", "n", "accuracy", "mean conf.", "ECE raw",  # noqa: T201
                 "ECE scaled", "Brier raw", "Brier scaled", "Brier base rate",
                 "skill scaled"], rows))  # fmt: skip


if __name__ == "__main__":
    main()
