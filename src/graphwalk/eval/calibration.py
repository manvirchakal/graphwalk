"""Is a walk's confidence worth anything? Calibration of answer confidence.

A walk's confidence is the probability of its best path (``exp(score)``, the product of
its decisions' probabilities). A calibrated decider's confidence should track how often
the answer is right, so it can be used to abstain or escalate. Measured per run:

* ECE: expected calibration error over equal-width bins (lower is better);
* Brier score (lower is better);
* AUROC of confidence for telling right answers from wrong ones (0.5 = useless);
* accuracy on the most confident half, vs. overall (selective answering).
"""

import math
from collections.abc import Sequence

from graphwalk.eval.runner import Record


def confidence(record: Record) -> float | None:
    """``exp(score)`` of the best path, or ``None`` if the walk produced no answer."""
    score = record.answer.detail.get("score")
    if not isinstance(score, int | float):
        return None
    return math.exp(float(score))


def ece(pairs: Sequence[tuple[float, float]], bins: int = 10) -> float:
    """``pairs`` of (confidence, correct in [0, 1])."""
    if not pairs:
        return math.nan
    total = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        members = [(c, y) for c, y in pairs if lo <= c < hi or (b == bins - 1 and c == 1.0)]
        if members:
            gap = abs(math.fsum(c for c, _ in members) - math.fsum(y for _, y in members))
            total += gap
    return total / len(pairs)


def auroc(pairs: Sequence[tuple[float, float]]) -> float:
    """Probability that a random right answer is more confident than a random wrong
    one (ties count half). ``correct`` is thresholded at 0.5."""
    pos = [c for c, y in pairs if y >= 0.5]  # noqa: PLR2004
    neg = [c for c, y in pairs if y < 0.5]  # noqa: PLR2004
    if not pos or not neg:
        return math.nan
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def calibration_row(system: str, records: Sequence[Record]) -> str:
    """One markdown row; correctness is set EM (MetaQA) or EM (text answers). Walks
    with no answer count as confidence 0."""
    pairs = [(confidence(r) or 0.0, r.score.em) for r in records]
    if not pairs:
        return f"| {system} | 0 | n/a | n/a | n/a | n/a | n/a |"
    brier = math.fsum((c - y) ** 2 for c, y in pairs) / len(pairs)
    ordered = sorted(pairs, key=lambda p: -p[0])
    top = ordered[: max(1, len(ordered) // 2)]
    overall = math.fsum(y for _, y in pairs) / len(pairs)
    top_acc = math.fsum(y for _, y in top) / len(top)
    mean_conf = math.fsum(c for c, _ in pairs) / len(pairs)
    return (
        f"| {system} | {len(pairs)} | {mean_conf:.3f} | {overall:.3f} | {ece(pairs):.3f} "
        f"| {brier:.3f} | {auroc(pairs):.3f} | {top_acc:.3f} |"
    )


CALIBRATION_HEADER = (
    "| system | n | mean confidence | EM | ECE | Brier | AUROC | EM, most confident half |\n"
    "|---|---|---|---|---|---|---|---|"
)
