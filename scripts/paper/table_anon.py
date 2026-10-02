"""P3: how much of the WebQSP results is the model's memory? Real names vs aliases.

    uv run python scripts/paper/table_anon.py

The A8 cheap-agent sample (100 WebQSP test questions, seed 0) rerun with every entity
name replaced by a meaningless alias (``rog.anonymize``): per arm, F1 and cost with real
names and with aliases, then paired differences between arms on the aliased graph.
"""

from common import by_system, mean, scores, table, with_ci
from runs import CHEAP_AGENT_ANON

from graphwalk.eval.metrics import bootstrap_ci

type Rows = dict[str, dict[str, float]]


def rows(run: str, system: str) -> Rows:
    return {
        r["question_id"]: {k: float(r[k] or 0.0) for k in ("f1", "cost_usd", "input_tokens")}
        for r in by_system(scores(run))[system]
    }


def diff(a: Rows, b: Rows, key: str, digits: int) -> str:
    ids = sorted(set(a) & set(b))
    values = [a[q][key] - b[q][key] for q in ids]
    lo, hi = bootstrap_ci(values)
    return f"{mean(values):+.{digits}f} [{lo:+.{digits}f}, {hi:+.{digits}f}]"


def main() -> None:
    real: dict[str, Rows] = {}
    anon: dict[str, Rows] = {}
    out = []
    for label, (system, real_run, anon_run) in CHEAP_AGENT_ANON.items():
        real[label], anon[label] = rows(real_run, system), rows(anon_run, system)
        r, a = list(real[label].values()), list(anon[label].values())
        out.append([label, with_ci([x["f1"] for x in r], 2), with_ci([x["f1"] for x in a], 2),
                    diff(anon[label], real[label], "f1", 2),
                    f"{mean([x['cost_usd'] for x in r]):.5f}",
                    f"{mean([x['cost_usd'] for x in a]):.5f}",
                    f"{mean([x['input_tokens'] for x in a]):,.0f}"])  # fmt: skip
    print(table(["arm", "F1 real names", "F1 aliases", "aliases - real", "$/q real",  # noqa: T201
                 "$/q aliases", "agent in tok/q aliases"], out))  # fmt: skip
    walk = "graph tools + walk"
    pairs = [(walk, "graph tools"), (walk, "search (RAG)"), ("graph tools", "search (RAG)"),
             (walk, "walk alone (no agent)")]  # fmt: skip
    print("\nPaired on the aliased graph\n")  # noqa: T201
    print(table(["comparison", "F1", "$/q"],  # noqa: T201
                [[f"{x} - {y}", diff(anon[x], anon[y], "f1", 3),
                  diff(anon[x], anon[y], "cost_usd", 5)] for x, y in pairs]))  # fmt: skip


if __name__ == "__main__":
    main()
