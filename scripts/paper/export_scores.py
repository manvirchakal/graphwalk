"""Write ``scores.jsonl`` next to a run's ``results.json`` (for runs made before the
runner wrote it).

    uv run python scripts/paper/export_scores.py 20260929T154958Z-2wiki-text [...]
"""

import argparse
import json

from common import RESULTS  # scripts/paper sibling

from graphwalk.eval.runner import SCORES_FILE, SystemRun, write_scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", help="run directories under results/")
    for run in parser.parse_args().runs:
        path = RESULTS / run / "results.json"
        document = json.loads(path.read_text(encoding="utf-8"))
        runs = [SystemRun.model_validate(raw) for raw in document["runs"]]
        write_scores(RESULTS / run / SCORES_FILE, runs)
        print(f"{run}: {sum(len(r.records) for r in runs)} rows")  # noqa: T201


if __name__ == "__main__":
    main()
