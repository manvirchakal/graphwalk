"""Can we tell before walking whether a walk will answer? (no API spend)

    uv run --extra eval python scripts/eval/router_analysis.py

Uses the per-question Jev walk logs already in ``results/`` (MetaQA 1-3 hop, WebQSP,
CWQ, WebQSP over the merged graph) and joins them with the questions. Label: walk F1 >=
0.5. Compares, by cross-validated AUROC:

* ``pre``: a logistic regression over features known before the walk (question text,
  number of topic entities, degree and relation count around the start);
* ``conf``: the walk's own confidence (known after a ~$0.0002 walk);
* ``pre+conf``: both.

If ``pre`` is near ``conf`` a router could skip doomed walks; if it adds nothing on top
of ``conf``, "walk first, read the confidence" is the whole routing rule.
"""

import json
import math
import re
from collections import defaultdict
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import numpy.typing as npt

from graphwalk.eval.calibration import auroc
from graphwalk.eval.datasets import metaqa, rog
from graphwalk.eval.types import EvalQuestion

type Triple = tuple[str, str, str]
type Matrix = npt.NDArray[np.float64]

RUNS = {
    "metaqa": [
        "results/decider/20260930T123542Z-metaqa-1hop",
        "results/decider/20260930T123641Z-metaqa-2hop",
        "results/decider/20260930T123821Z-metaqa-3hop",
        "results/decider/20260930T124219Z-metaqa-1hop",
        "results/decider/20260930T124319Z-metaqa-2hop",
        "results/decider/20260930T124452Z-metaqa-3hop",
        "results/decider/20260930T124616Z-metaqa-1hop",
        "results/decider/20260930T124715Z-metaqa-2hop",
        "results/decider/20260930T124852Z-metaqa-3hop",
    ],
    "webqsp": ["results/kgqa/20261001T153829Z-webqsp"],
    "cwq": ["results/kgqa/20261001T160324Z-cwq"],
    "webqsp-global": ["results/kgqa/20261001T172038Z-webqsp-global"],
}

SUPERLATIVE = re.compile(
    r"\b(first|last|most|least|largest|smallest|biggest|oldest|youngest|latest|earliest|"
    r"highest|lowest|longest|shortest|best|top|\w+est)\b",
    re.IGNORECASE,
)
TEMPORAL = re.compile(r"\b(when|before|after|during|since|until|in \d{3,4}|\d{4})\b", re.I)
CONJUNCTION = re.compile(r"\b(and|both|also|as well as|but|that|which|whose)\b", re.I)
COUNTING = re.compile(r"\b(how many|number of|count)\b", re.I)
WH = ("who", "what", "where", "when", "which", "how")
FEATURES = (
    "words", "entities", "superlative", "temporal", "conjunction", "counting",
    *(f"wh_{w}" for w in WH), "log_degree", "relations_1hop", "log_relations_2hop",
)  # fmt: skip


def walk_rows(paths: Iterable[str]) -> dict[str, tuple[float, float]]:
    """question id -> (f1, confidence) for the Jev walk system (last run wins)."""
    rows: dict[str, tuple[float, float]] = {}
    for path in paths:
        for line in (Path(path) / "scores.jsonl").read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if "jev" in row["system"]:
                rows[row["question_id"]] = (float(row["f1"]), float(row["confidence"] or 0.0))
    return rows


class Adjacency:
    def __init__(self, triples: Iterable[Triple]) -> None:
        self.out: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for h, r, t in triples:
            self.out[h].append((r, t))
            self.out[t].append((r, h))

    def features(self, start: Sequence[str]) -> tuple[float, float, float]:
        degree = sum(len(self.out.get(s, ())) for s in start)
        one = {r for s in start for r, _ in self.out.get(s, ())}
        frontier = {n for s in start for _, n in self.out.get(s, ())}
        two = set(one)
        for node in list(frontier)[:2000]:
            two.update(r for r, _ in self.out.get(node, ()))
        return math.log1p(degree), float(len(one)), math.log1p(len(two))


def text_features(question: EvalQuestion) -> list[float]:
    text = question.question
    lowered = text.lower()
    return [
        float(len(text.split())),
        float(len(question.start or ())),
        float(bool(SUPERLATIVE.search(text))),
        float(bool(TEMPORAL.search(text))),
        float(bool(CONJUNCTION.search(text))),
        float(bool(COUNTING.search(text))),
        *(float(lowered.startswith(w) or f" {w} " in f" {lowered[:30]} ") for w in WH),
    ]


def dataset_rows(name: str) -> tuple[list[list[float]], list[float], list[float]]:
    walks = {k: v for path in RUNS[name] for k, v in walk_rows([path]).items()}
    features: list[list[float]] = []
    labels: list[float] = []
    confidences: list[float] = []
    if name == "metaqa":
        kb, _ = metaqa.download(1)
        adjacency = Adjacency(metaqa.read_triples(kb))
        questions = [
            q for hops in (1, 2, 3) for q in metaqa.read_questions(metaqa.download(hops)[1], hops)
        ]
        lookup = {q.id: (q, adjacency) for q in questions if q.id in walks}
    else:
        source = "webqsp" if name.startswith("webqsp") else "cwq"
        questions, graphs = rog.load(source, "test")
        lookup = {q.id: (q, Adjacency(graphs[q.id])) for q in questions if q.id in walks}
    for qid, (f1, conf) in walks.items():
        question, adjacency = lookup[qid]
        features.append([*text_features(question), *adjacency.features(question.start or ())])
        labels.append(float(f1 >= 0.5))  # noqa: PLR2004
        confidences.append(conf)
    return features, labels, confidences


def fit_logistic(x: Matrix, y: Matrix, l2: float = 1.0, steps: int = 500) -> Matrix:
    w = np.zeros(x.shape[1])
    for _ in range(steps):  # Newton steps would be overkill; plain gradient descent
        p = 1.0 / (1.0 + np.exp(-(x @ w)))
        grad = x.T @ (p - y) / len(y) + l2 * np.r_[0.0, w[1:]] / len(y)
        w -= 0.5 * grad
    return w


def cv_scores(x: Matrix, y: Matrix, folds: int = 5, seed: int = 0) -> Matrix:
    """Out-of-fold predicted probabilities (features standardised within each fold)."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(y))
    out = np.zeros(len(y))
    for k in range(folds):
        test = order[k::folds]
        train = np.setdiff1d(order, test)
        mu, sd = x[train].mean(0), x[train].std(0) + 1e-9
        design = np.c_[np.ones(len(train)), (x[train] - mu) / sd]
        w = fit_logistic(design, y[train])
        out[test] = np.c_[np.ones(len(test)), (x[test] - mu) / sd] @ w
    return out


def auc(scores: Sequence[float] | Matrix, labels: Sequence[float] | Matrix) -> float:
    return auroc([(float(s), float(y)) for s, y in zip(scores, labels, strict=True)])


def auc_ci(scores: Matrix, labels: Matrix, samples: int = 1000) -> tuple[float, float]:
    rng = np.random.default_rng(0)
    values: list[float] = []
    for _ in range(samples):
        i = rng.integers(0, len(labels), len(labels))
        if not math.isnan(v := auc(scores[i], labels[i])):
            values.append(v)
    values.sort()
    return values[int(0.025 * len(values))], values[int(0.975 * len(values)) - 1]


def report(name: str, x: Matrix, y: Matrix, conf: Matrix) -> list[str]:
    pre = cv_scores(x, y)
    both = cv_scores(np.c_[x, conf], y)
    lines = [f"| {name} | {len(y)} | {y.mean():.2f} |"]
    for scores in (pre, conf, both):
        lo, hi = auc_ci(scores, y)
        lines[0] += f" {auc(scores, y):.2f} [{lo:.2f}, {hi:.2f}] |"
    singles = sorted(
        ((abs(auc(x[:, j], y) - 0.5), FEATURES[j], auc(x[:, j], y)) for j in range(x.shape[1])),
        reverse=True,
    )
    top = ", ".join(f"{f} {a:.2f}" for gap, f, a in singles[:4] if not math.isnan(gap))
    return [*lines, f"|  | top single features: {top} | | | | |"]


def main() -> None:
    data = {name: dataset_rows(name) for name in RUNS}
    header = [
        "| data | n | walk right | pre-walk AUROC | confidence AUROC | pre+conf AUROC |",
        "|---|---|---|---|---|---|",
    ]
    lines = list(header)
    pooled: list[tuple[list[float], float, float]] = []
    for name, (feats, labels, confs) in data.items():
        x, y, c = np.array(feats), np.array(labels), np.array(confs)
        lines += report(name, x, y, c)
        pooled += zip(feats, labels, confs, strict=True)
    freebase = [
        r for name in ("webqsp", "cwq", "webqsp-global") for r in zip(*data[name], strict=True)
    ]
    for name, rows in (("freebase (pooled)", freebase), ("all (pooled)", pooled)):
        lines += report(
            name,
            np.array([r[0] for r in rows]),
            np.array([r[1] for r in rows]),
            np.array([r[2] for r in rows]),
        )
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = Path("results") / "router" / stamp
    out.mkdir(parents=True, exist_ok=True)
    body = "\n".join(
        [
            "# Router analysis: predicting walk success before the walk",
            "",
            "Label: walk F1 >= 0.5. AUROC from 5-fold cross-validated logistic regression"
            " (pre-walk features; + confidence) or confidence alone; 95% bootstrap CIs.",
            f"Features: {', '.join(FEATURES)}.",
            "",
            *lines,
        ]
    )
    (out / "summary.md").write_text(body + "\n", encoding="utf-8")
    print(body)  # noqa: T201


if __name__ == "__main__":
    main()
