"""Shared helpers for the paper's table scripts.

Tables are computed from the committed per-question ``scores.jsonl`` of each run (see
``graphwalk.eval.runner.score_rows``); ``runs.py`` says which run feeds which table.
Confidence intervals are 95% percentile bootstraps over questions.
"""

import json
import math
import sys
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
sys.path.insert(0, str(ROOT / "src"))

from graphwalk.eval.metrics import bootstrap_ci  # noqa: E402

type Row = dict[str, Any]


def scores(run: str) -> list[Row]:
    """The per-question rows of a run directory under results/."""
    path = RESULTS / run / "scores.jsonl"
    if not path.exists():
        msg = f"{path} missing (run scripts/paper/export_scores.py {run} if results.json exists)"
        raise SystemExit(msg)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def by_system(rows: Iterable[Row]) -> dict[str, list[Row]]:
    grouped: dict[str, list[Row]] = defaultdict(list)
    for row in rows:
        grouped[row["system"]].append(row)
    return grouped


def mean(values: Sequence[float]) -> float:
    return math.fsum(values) / len(values) if values else math.nan


def with_ci(values: Sequence[float], digits: int = 3) -> str:
    """``mean [lo, hi]``."""
    lo, hi = bootstrap_ci(values)
    return f"{mean(values):.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def paired(a: Sequence[Row], b: Sequence[Row], metric: Callable[[Row], float]) -> str:
    """Mean difference ``a - b`` over shared questions, with a paired bootstrap CI."""
    right = {r["question_id"]: r for r in b}
    diffs = [metric(r) - metric(right[r["question_id"]]) for r in a if r["question_id"] in right]
    lo, hi = bootstrap_ci(diffs)
    return f"{mean(diffs):+.3f} [{lo:+.3f}, {hi:+.3f}]"


def cost_per_1k(rows: Sequence[Row]) -> str:
    """Mean provider-reported cost per 1,000 queries; ``n/a`` unless every query
    reported one (errors excluded)."""
    costs = [r["cost_usd"] for r in rows if r["status"] != "error"]
    if not costs or any(c is None for c in costs):
        return "n/a"
    return f"{1000 * mean(costs):.3f}"


def p50(rows: Sequence[Row]) -> str:
    values = sorted(r["latency_s"] for r in rows)
    if not values:
        return "n/a"
    return f"{values[max(0, math.ceil(len(values) / 2) - 1)]:.2f}"


def decision_p50(rows: Sequence[Row]) -> str:
    """Median time spent in decision calls per question (walks only): unlike wall
    latency, it excludes waiting for a rate-limit slot."""
    values = sorted(
        r["decision_latency_s"] for r in rows if r.get("decision_latency_s") is not None
    )
    if not values:
        return "—"
    return f"{values[max(0, math.ceil(len(values) / 2) - 1)]:.2f}"


def table(header: Sequence[str], rows: Iterable[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)
