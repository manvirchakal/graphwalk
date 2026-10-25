"""Data files for the LaTeX paper's risk-coverage figure (paper/latex/figures/).

    uv run python scripts/paper/figure_tex.py

One whitespace-separated ``coverage risk`` file per (setting, confidence signal) of P8
(runs.STATED), the same model and prompt with only the confidence signal changed,
for pgfplots to read. Risk at coverage k/n as in figure_selective.risks.
"""

import re

from common import ROOT, by_system, scores
from figure_selective import risks
from runs import STATED

OUT = ROOT / "paper" / "latex" / "figures"
SIGNALS = {
    "token probabilities": "token",
    "stated, 0-100 per option": "scores",
    "stated, top-1 + confidence": "top1",
    "vote share, 5 samples": "vote",
    "context: Jev": "jev",
}


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for setting, (metric, variants) in STATED.items():
        for label, (run, system) in variants.items():
            name = next((v for k, v in SIGNALS.items() if label.startswith(k)), None)
            if name is None or "rerun" in label:
                continue
            records = by_system(scores(run))[system]
            curve = risks([(r["confidence"] or 0.0, float(r[metric])) for r in records])
            lines = ["coverage risk"] + [
                f"{k / len(curve):.4f} {risk:.4f}" for k, risk in enumerate(curve, start=1)
            ]
            path = OUT / f"selective-{slug(setting)}-{name}.dat"
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            print(path.relative_to(ROOT))  # noqa: T201


if __name__ == "__main__":
    main()
