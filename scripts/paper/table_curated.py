"""Table: curated graphs (MetaQA 1/2/3-hop; 2Wiki's pooled gold-evidence graph).

    uv run python scripts/paper/table_curated.py

For each system: F1 on the seed-0 subset with a 95% bootstrap CI; F1 on every seed run
(each seed is a different seeded subset of the same test set); the paired F1 difference
from graphwalk with Jev on seed 0; cost and latency on seed 0. Wall latency (p50 s)
includes waiting for a rate-limit slot, which inflates the LLM-decider rows (every call
waits under 20 requests/min); the decision column is time in decision calls only.
Systems: graphwalk with Jev (E4 seeds), graphwalk with an LLM decider (E3), an LLM
writing the relation path (E2), vector RAG and multi-step RAG (M5).
"""

import statistics

from common import (
    by_system,
    cost_per_1k,
    decision_p50,
    mean,
    p50,
    paired,
    scores,
    table,
    with_ci,
)
from runs import CURATED


def main() -> None:
    for dataset, systems in CURATED.items():
        loaded = {
            label: [by_system(scores(run))[system] for run, system in seeds]
            for label, seeds in systems.items()
            if seeds
        }
        reference = next(iter(loaded.values()))[0]
        out: list[list[str]] = []
        for label, seeds in loaded.items():
            first = seeds[0]
            per_seed = [mean([r["f1"] for r in rows]) for rows in seeds]
            spread = (
                f"{statistics.mean(per_seed):.3f} ± {statistics.stdev(per_seed):.3f} "
                f"({len(per_seed)} seeds)"
                if len(per_seed) > 1
                else "—"
            )
            out.append(
                [label, with_ci([r["f1"] for r in first]), spread,
                 f"{mean([r['em'] for r in first]):.3f}",
                 "—" if first is reference else paired(first, reference, lambda r: r["f1"]),
                 cost_per_1k(first), p50(first), decision_p50(first)]
            )  # fmt: skip
        print(f"### {dataset} (n = {len(reference)} per seed)\n")  # noqa: T201
        print(table(["system", "F1, seed 0 [95% CI]", "F1 across seeds (mean ± sd)", "EM",  # noqa: T201
                     "Δ F1 vs graphwalk (Jev)", "$/1k q", "p50 s",
                     "p50 decision s"], out))  # fmt: skip
        print()  # noqa: T201


if __name__ == "__main__":
    main()
