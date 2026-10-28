"""P9: walks the decider never answered, before and after scoring them lowest.

    uv run python scripts/paper/table_nochoice.py

A walk of forced moves (one legal option each) that ends at a dead end made no decision,
but the dead end counted as a certain STOP (p = 1), so its confidence was 1.0, the top
of the ranking. Since d6dff66 such walks score log(prob_floor) (``Answer.n_choices``).
This table rescores the committed runs both ways: the walks are the same, only the
confidence of the zero-choice ones changes (1.0 before, 10^-3 after).

Seeds 0-2 are the 2Wiki runs where the problem was found. Seed 3 was run after the rule
was committed; its last row keeps only questions absent from seeds 0-2. Zero-choice
walks are read from each run's results.json (``n_choices`` = 0, or for runs before the
field existed, no decider confidence along the path; runs that recorded neither count
none). The check at the end counts them in every other run behind the confidence tables.

AURC: mean risk over coverages (lower is better), against a random order (the error
rate). ECE after Platt scaling is cross-fitted as in P7. Delta AUROC: paired bootstrap
over questions, 95% percentile interval.
"""

import json
import random

from common import RESULTS, Row, by_system, mean, scores, table
from figure_selective import risks
from runs import CALIBRATION, KGQA_CALIBRATION, LOGPROB_CALIBRATION, NOCHOICE
from table_recalibration import cross_fit

from graphwalk.eval.calibration import auroc

FLOOR = 1e-3
RESAMPLES = 1000
HELD_OUT = "Jev, seed 3 (held out)"


def zero_choice(run: str, system: str) -> set[str]:
    """Question ids whose walk (of ``system``) finished without a decision the decider
    answered."""
    data = json.loads((RESULTS / run / "results.json").read_text(encoding="utf-8"))
    ids: set[str] = set()
    for system_run in data["runs"]:
        if system_run["system"] != system:
            continue
        for record in system_run["records"]:
            answer = record["answer"]
            detail = answer.get("detail") or {}
            if answer.get("status") != "ok" or detail.get("score") is None:
                continue
            if "n_choices" in detail:
                zero = detail["n_choices"] == 0
            else:  # older runs: no decider confidence along the path, if it was recorded
                zero = "min_confidence" in detail and detail["min_confidence"] is None
            if zero:
                ids.add(record["question"]["id"])
    return ids


def aurc(pairs: list[tuple[float, float]]) -> float:
    return mean(risks(pairs))


def delta_auroc(before: list[tuple[float, float]], after: list[tuple[float, float]]) -> str:
    rng = random.Random(0)  # noqa: S311
    n = len(before)
    diffs: list[float] = []
    for _ in range(RESAMPLES):
        idx = [rng.randrange(n) for _ in range(n)]
        b = [before[i] for i in idx]
        a = [after[i] for i in idx]
        if 0 < sum(y for _, y in b) < n:
            diffs.append(auroc(a) - auroc(b))
    diffs.sort()
    lo, hi = diffs[int(0.025 * len(diffs))], diffs[int(0.975 * len(diffs)) - 1]
    return f"{auroc(after) - auroc(before):+.2f} [{lo:+.2f}, {hi:+.2f}]"


def row(label: str, records: list[Row], zero: set[str]) -> list[str]:
    flagged = [r["question_id"] in zero for r in records]
    em = [float(r["em"]) for r in records]
    raw = [float(r["confidence"] or 0.0) for r in records]
    before = [(1.0 if z else c, y) for z, c, y in zip(flagged, raw, em, strict=True)]
    after = [(FLOOR if z else c, y) for z, c, y in zip(flagged, raw, em, strict=True)]
    wrong = sum(z and not y for z, y in zip(flagged, em, strict=True))
    return [
        label, str(len(records)), f"{mean(em):.3f}", f"{sum(flagged)} ({wrong})",
        f"{auroc(before):.2f} → {auroc(after):.2f}", delta_auroc(before, after),
        f"{aurc(before):.3f} → {aurc(after):.3f}", f"{1.0 - mean(em):.3f}",
        f"{cross_fit(before)[0]:.3f} → {cross_fit(after)[0]:.3f}",
    ]  # fmt: skip


def main() -> None:
    rows: list[list[str]] = []
    seen: set[str] = set()
    for label, (run, system) in NOCHOICE.items():
        records = by_system(scores(run))[system]
        zero = zero_choice(run, system)
        rows.append(row(label, records, zero))
        if label == HELD_OUT:
            unseen = [r for r in records if r["question_id"] not in seen]
            rows.append(row(f"{label}, unseen questions", unseen, zero))
        elif label.startswith("Jev"):
            seen |= {r["question_id"] for r in records}
    print(table(["2Wiki walks", "n", "EM", "zero-choice (wrong)", "AUROC",  # noqa: T201
                 "Δ AUROC", "AURC", "AURC random", "ECE after Platt"], rows))  # fmt: skip

    others = {
        f"{dataset} ({decider})": source
        for dataset, deciders in (*CALIBRATION.items(), *KGQA_CALIBRATION.items())
        for decider, source in deciders.items()
    } | {f"{dataset} (token probabilities)": (run, system)
         for dataset, (run, system, _) in LOGPROB_CALIBRATION.items()}  # fmt: skip
    counts = [[label, str(len(zero_choice(*source)))] for label, source in others.items()]
    print("\nZero-choice walks in the other confidence runs:\n")  # noqa: T201
    print(table(["run", "zero-choice walks"], counts))  # noqa: T201


if __name__ == "__main__":
    main()
