"""Answer metrics.

Strings are compared after SQuAD-style normalization (lowercase, drop punctuation and
articles, collapse whitespace).

* ``hits@1``: the top prediction matches any gold answer.
* Set questions (MetaQA): exact match = the predicted set equals the gold set; P/R/F1
  over the two sets.
* Text questions (2Wiki): exact match = the top prediction equals the gold answer;
  F1 = SQuAD token F1 of the top prediction.
"""

import math
import random
import re
import string
from collections import Counter
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict

from graphwalk.eval.types import AnswerKind

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = str.maketrans("", "", string.punctuation)


def normalize(text: str) -> str:
    text = text.casefold().translate(_PUNCT)
    return " ".join(_ARTICLES.sub(" ", text).split())


class Score(BaseModel):
    model_config = ConfigDict(frozen=True)

    hits1: float
    em: float
    precision: float
    recall: float
    f1: float


def _token_f1(pred: str, gold: str) -> tuple[float, float, float]:
    p, g = normalize(pred).split(), normalize(gold).split()
    common = sum((Counter(p) & Counter(g)).values())
    if not p or not g or common == 0:
        return 0.0, 0.0, float(p == g)
    precision, recall = common / len(p), common / len(g)
    return precision, recall, 2 * precision * recall / (precision + recall)


def score(
    predictions: Sequence[str], answer_set: Sequence[str], gold: Sequence[str], kind: AnswerKind
) -> Score:
    gold_norm = {normalize(g) for g in gold}
    top = normalize(predictions[0]) if predictions else ""
    hits1 = float(bool(top) and top in gold_norm)
    if kind == "text":
        if not predictions:
            return Score(hits1=0.0, em=0.0, precision=0.0, recall=0.0, f1=0.0)
        best = max((_token_f1(predictions[0], g) for g in gold), key=lambda t: t[2])
        return Score(hits1=hits1, em=hits1, precision=best[0], recall=best[1], f1=best[2])
    pred_norm = {normalize(p) for p in answer_set} - {""}
    hit = len(pred_norm & gold_norm)
    precision = hit / len(pred_norm) if pred_norm else 0.0
    recall = hit / len(gold_norm) if gold_norm else 0.0
    f1 = 2 * precision * recall / (precision + recall) if hit else 0.0
    return Score(
        hits1=hits1, em=float(pred_norm == gold_norm), precision=precision, recall=recall, f1=f1
    )


def percentile(values: Sequence[float], q: float) -> float:
    """Nearest-rank percentile (``q`` in [0, 100]); NaN for no values."""
    if not values:
        return math.nan
    ordered = sorted(values)
    rank = max(1, math.ceil(q / 100 * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


def bootstrap_ci(
    values: Sequence[float], *, samples: int = 2000, level: float = 0.95, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for the mean of ``values``.

    For a paired comparison of two systems, pass the per-question differences.
    """
    if not values:
        return math.nan, math.nan
    rng = random.Random(seed)  # noqa: S311 - reproducible resampling
    n = len(values)
    means = sorted(math.fsum(rng.choices(values, k=n)) / n for _ in range(samples))
    tail = (1 - level) / 2
    lo = means[int(tail * samples)]
    hi = means[min(samples - 1, int((1 - tail) * samples))]
    return lo, hi
