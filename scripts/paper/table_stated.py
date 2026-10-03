"""P8: token probabilities vs stated confidence, from the same model and prompt.

    uv run python scripts/paper/table_stated.py

The ``llm`` decider of E3/P1 differs from the ``logprob`` decider in two ways: another
model, and confidence it states rather than token probabilities. Here the model, the
endpoint and the lettered prompt are fixed (Qwen3.8-27B, graphwalk.eval.stated_decider)
and only the confidence signal changes: token probabilities (``logprob``), a 0-100
score per option, one letter with a 0-100 confidence, or the vote share of 5 samples at
temperature 1. Per setting: accuracy, AUROC, AURC (P0), Brier skill after Platt scaling
(P7), and cost per 1,000 questions.
"""

from common import by_system, cost_per_1k, mean, scores, table
from figure_selective import risks
from runs import STATED
from table_recalibration import cross_fit

from graphwalk.eval.calibration import auroc


def main() -> None:
    rows: list[list[str]] = []
    for setting, (metric, variants) in STATED.items():
        for label, (run, system) in variants.items():
            records = by_system(scores(run))[system]
            data = [(r["confidence"] or 0.0, float(r[metric])) for r in records]
            rate = mean([y for _, y in data])
            base = rate * (1.0 - rate)
            _, scaled_brier = cross_fit(data)
            rows.append(
                [setting, label, str(len(data)), f"{rate:.3f}", f"{auroc(data):.3f}",
                 f"{mean(risks(data)):.3f}", f"{1.0 - scaled_brier / base:+.2f}",
                 cost_per_1k(records)]
            )  # fmt: skip
    print(table(["setting", "confidence signal", "n", "accuracy", "AUROC", "AURC",  # noqa: T201
                 "Brier skill (scaled)", "$ / 1k q"], rows))  # fmt: skip


if __name__ == "__main__":
    main()
