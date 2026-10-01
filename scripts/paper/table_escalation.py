"""Table A1: escalation. Re-walk with the LLM decider when Jev's confidence is low.

    uv run python scripts/paper/table_escalation.py

Simulated offline from the E3 runs: both deciders answered the same questions, so a
question "escalated" at threshold ``t`` (Jev's best-path confidence below ``t``, or no
answer) takes the LLM decider's answer and pays for both walks. Thresholds are fixed
on a grid, not tuned per dataset (the grid sits high because Jev's confidences cluster
near 1). F1 and EM are shown with 95% bootstrap CIs, and their gaps to the LLM
decider alone with paired CIs. Only seed 0 is used: the other Jev seeds drew
different question samples, which the single LLM-decider run does not cover.
"""

from common import Row, by_system, mean, scores, table, with_ci
from runs import CALIBRATION

from graphwalk.eval.metrics import bootstrap_ci

THRESHOLDS = (0.8, 0.9, 0.95, 0.99)


def _rows(run: str, system: str) -> dict[str, Row]:
    return {r["question_id"]: r for r in by_system(scores(run))[system]}


def _cost(row: Row) -> float:
    return row.get("cost_usd") or 0.0


def escalate(jev: dict[str, Row], llm: dict[str, Row], threshold: float) -> list[Row]:
    """Per shared question: Jev's row, or the LLM decider's (with both costs) if
    Jev's confidence is below ``threshold``."""
    out: list[Row] = []
    for qid in sorted(jev.keys() & llm.keys()):
        j = jev[qid]
        if (j["confidence"] or 0.0) < threshold:
            out.append({**llm[qid], "cost_usd": _cost(j) + _cost(llm[qid]), "escalated": True})
        else:
            out.append({**j, "escalated": False})
    return out


def _gap(a: list[Row], b: dict[str, Row], metric: str) -> str:
    diffs = [float(r[metric]) - float(b[r["question_id"]][metric]) for r in a]
    lo, hi = bootstrap_ci(diffs)
    return f"{mean(diffs):+.3f} [{lo:+.3f}, {hi:+.3f}]"


def main() -> None:
    body: list[list[str]] = []
    for dataset, deciders in CALIBRATION.items():
        if "LLM decider" not in deciders:
            continue
        jev, llm = _rows(*deciders["Jev"]), _rows(*deciders["LLM decider"])
        shared = sorted(jev.keys() & llm.keys())
        lines: list[tuple[str, list[Row]]] = [
            ("Jev only", [jev[q] for q in shared]),
            ("LLM decider only", [{**llm[q], "escalated": True} for q in shared]),
        ]
        lines += [(f"escalate below {t}", escalate(jev, llm, t)) for t in THRESHOLDS]
        for label, rows in lines:
            esc = mean([1.0 if r.get("escalated") else 0.0 for r in rows])
            body.append(
                [dataset, label, str(len(rows)), f"{esc:.0%}",
                 with_ci([float(r["f1"]) for r in rows]),
                 _gap(rows, llm, "f1") if label != "LLM decider only" else "—",
                 with_ci([float(r["em"]) for r in rows]),
                 _gap(rows, llm, "em") if label != "LLM decider only" else "—",
                 f"{1000 * mean([_cost(r) for r in rows]):.3f}"]
            )  # fmt: skip
    header = ["dataset", "system", "n", "escalated", "F1", "F1 vs LLM decider", "EM",
              "EM vs LLM decider", "$ / 1k queries"]  # fmt: skip
    print(table(header, body))  # noqa: T201


if __name__ == "__main__":
    main()
