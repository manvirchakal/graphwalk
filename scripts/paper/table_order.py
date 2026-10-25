"""P6: does the order the options are shown in change the walk's answers or confidence?

    uv run python scripts/paper/table_order.py

Each setting reruns the same questions with the options in the original order (a plain
rerun, to measure run-to-run noise at temperature 0) and in shuffled orders
(``--shuffle K``, graphwalk.eval.shuffle). Per run: accuracy, AUROC of the confidence,
and, against the reference run, the share of questions with the same answer set and
the rank correlation of confidence. The last row per setting scores the reference
answers by how many of the runs agree with them (a consistency signal, for comparison).
Then the logprob decider's format diagnostics: the mean probability its first token put
on the offered letters, and how often the top token was not a letter.
"""

import json
import math

from common import RESULTS, by_system, mean, scores, table
from runs import OPTION_ORDER

from graphwalk.eval.calibration import auroc


def ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            out[order[k]] = (i + j) / 2
        i = j + 1
    return out


def spearman(a: list[float], b: list[float]) -> float:
    ra, rb = ranks(a), ranks(b)
    ma, mb = mean(ra), mean(rb)
    cov = math.fsum((x - ma) * (y - mb) for x, y in zip(ra, rb, strict=True))
    var = math.sqrt(math.fsum((x - ma) ** 2 for x in ra) * math.fsum((y - mb) ** 2 for y in rb))
    return cov / var if var else math.nan


def main() -> None:
    rows: list[list[str]] = []
    diagnostics: list[list[str]] = []
    for setting, (metric, variants) in OPTION_ORDER.items():
        runs = {
            label: {r["question_id"]: r for r in by_system(scores(run))[system]}
            for label, (run, system) in variants.items()
        }
        reference = runs[next(iter(variants))]
        for label, data in runs.items():
            shared = [q for q in reference if q in data]
            pairs = [(data[q]["confidence"] or 0.0, float(data[q][metric])) for q in shared]
            same = [sorted(data[q]["answers"]) == sorted(reference[q]["answers"]) for q in shared]
            rho = spearman(
                [data[q]["confidence"] or 0.0 for q in shared],
                [reference[q]["confidence"] or 0.0 for q in shared],
            )
            rows.append(
                [setting, label, str(len(shared)), f"{mean([y for _, y in pairs]):.3f}",
                 f"{auroc(pairs):.3f}", f"{mean([float(s) for s in same]):.3f}",
                 f"{rho:.3f}"]
            )  # fmt: skip
            path = RESULTS / variants[label][0] / "diagnostics.json"
            if path.exists():
                for model, stats in json.loads(path.read_text(encoding="utf-8")).items():
                    diagnostics.append(
                        [setting, label, model, str(stats["asked"]),
                         f"{stats['mean_letter_mass']:.4f}",
                         f"{stats['top_not_letter'] / max(1, stats['asked']):.4f}",
                         str(stats["no_letter"])]
                    )  # fmt: skip
        agreement = [
            (
                mean([float(sorted(d[q]["answers"]) == sorted(reference[q]["answers"]))
                      for d in runs.values() if q in d]),
                float(reference[q][metric]),
            )
            for q in reference
        ]  # fmt: skip
        rows.append(
            [setting, f"agreement of {len(runs)} runs", str(len(agreement)),
             f"{mean([y for _, y in agreement]):.3f}", f"{auroc(agreement):.3f}", "", ""]
        )  # fmt: skip
    print(table(["setting", "run", "n", "accuracy", "AUROC", "same answer",  # noqa: T201
                 "conf. rank corr."], rows))  # fmt: skip
    print()  # noqa: T201
    print(table(["setting", "run", "decider", "questions", "mass on letters",  # noqa: T201
                 "top token not a letter", "no letter"], diagnostics))  # fmt: skip


if __name__ == "__main__":
    main()
