"""E1: evidence retrieval. Does ``locate`` find the passages that answer a question?

Every method returns a ranked list of documents (paragraphs or pages) from the same
pooled text, scored against the gold evidence documents:

* 2Wiki and HotpotQA: the paragraphs named in ``supporting_facts``;
* FanOutQA: the question's evidence pages.

``recall@k`` is the share of gold documents among the first ``k`` distinct documents a
method returns; ``complete@k`` is whether all of them are. Methods may return fewer
than ``k`` (a walk that reaches little returns little); that counts against them.
"""

import asyncio
import math
import re
import time
from collections import Counter, defaultdict
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from graphwalk.decisions.base import DecisionBackend
from graphwalk.embeddings.base import Embedder
from graphwalk.eval.metrics import bootstrap_ci
from graphwalk.locate.dense import DenseLocator, fuse
from graphwalk.locate.documents import StoredDocuments
from graphwalk.locate.graph import GraphLocator, LocateResult
from graphwalk.locate.model import Location
from graphwalk.stores.networkx_store import NetworkXStore
from graphwalk.traversal import ChoiceEntryResolver, NameEntryResolver, TraversalConfig, Traverser
from graphwalk.traversal.prompts import node_text

_FROZEN = ConfigDict(frozen=True, extra="forbid")


def gold_titles(record: Mapping[str, Any]) -> list[str]:
    """Titles of the gold evidence documents: the supporting-fact paragraphs if the
    record has them, else every context document (FanOutQA's evidence pages)."""
    facts = record.get("supporting_facts")
    if facts:
        return list(dict.fromkeys(str(title) for title, _ in facts))
    return list(dict.fromkeys(str(title) for title, _ in record["context"]))


def distinct_keys(locations: Sequence[Location]) -> list[str]:
    return list(dict.fromkeys(loc.key for loc in locations))


def first_per_document(locations: Sequence[Location]) -> list[Location]:
    """The best-ranked location of each document, in rank order."""
    seen: set[str] = set()
    out: list[Location] = []
    for location in locations:
        if location.key not in seen:
            seen.add(location.key)
            out.append(location)
    return out


def hybrid_keys(graph: Sequence[Location], dense: Sequence[Location], k: int) -> list[str]:
    """Documents from fusing the first ``k`` of each list, as ``locate(mode="hybrid")``
    does; ``dense`` should already be one location per document."""
    return distinct_keys(fuse([graph[:k], dense[:k]], k))


def recall_at(gold: Sequence[str], ranked: Sequence[str], k: int) -> float:
    if not gold:
        return math.nan
    top = set(ranked[:k])
    return sum(g in top for g in gold) / len(gold)


def complete_at(gold: Sequence[str], ranked: Sequence[str], k: int) -> float:
    top = set(ranked[:k])
    return float(all(g in top for g in gold))


class EvidenceRow(BaseModel):
    """One method's ranking for one question."""

    model_config = _FROZEN

    question_id: str
    type: str
    method: str
    gold: tuple[str, ...]
    ranked: tuple[str, ...]
    """Distinct document keys, best first (for hybrid: per ``k``, see ``by_k``)."""
    by_k: dict[int, tuple[str, ...]] = {}
    """Methods whose ranking depends on ``k`` (hybrid): the ranking at each ``k``."""
    locations: tuple[Location, ...] = ()
    """Graph methods: the locations themselves, so hybrids can be fused after a resume."""
    decision_calls: int = 0
    cost_usd: float | None = None
    latency_s: float = 0.0
    error: str | None = None

    def at(self, k: int) -> tuple[str, ...]:
        return self.by_k.get(k, self.ranked)


def _mean(values: Sequence[float]) -> float:
    return math.fsum(values) / len(values) if values else math.nan


def _methods(rows: Sequence[EvidenceRow]) -> dict[str, list[EvidenceRow]]:
    grouped: dict[str, list[EvidenceRow]] = defaultdict(list)
    for row in rows:
        grouped[row.method].append(row)
    return grouped


def recall_table(rows: Sequence[EvidenceRow], ks: Sequence[int], ci_k: int) -> str:
    """Per method: recall@k (95% bootstrap CI at ``ci_k``), complete@``ci_k``, mean
    documents returned, Jev calls and cost per query."""
    header = (
        "| method | "
        + " | ".join(f"recall@{k}" for k in ks)
        + f" | 95% CI @{ci_k} | complete@{ci_k} | docs returned | Jev calls/q | $/1k q |"
    )
    lines = [header, "|---" * (len(ks) + 6) + "|"]
    for method, group in _methods(rows).items():
        recalls = {k: [recall_at(r.gold, r.at(k), k) for r in group] for k in ks}
        lo, hi = bootstrap_ci(recalls[ci_k])
        complete = _mean([complete_at(r.gold, r.at(ci_k), ci_k) for r in group])
        returned = _mean([float(len(r.at(max(ks)))) for r in group])
        calls = _mean([float(r.decision_calls) for r in group])
        costs = [r.cost_usd for r in group]
        cost = (
            "n/a"
            if any(c is None for c in costs)
            else f"{1000 * _mean([c or 0.0 for c in costs]):.3f}"
        )
        cells = " | ".join(f"{_mean(recalls[k]):.3f}" for k in ks)
        lines.append(
            f"| {method} | {cells} | [{lo:.3f}, {hi:.3f}] | {complete:.3f} | {returned:.1f} "
            f"| {calls:.2f} | {cost} |"
        )
    return "\n".join(lines)


def type_table(rows: Sequence[EvidenceRow], k: int) -> str:
    """recall@k per question type, per method."""
    types = sorted({r.type for r in rows})
    lines = [
        "| method | " + " | ".join(f"{t} (n)" for t in types) + " |",
        "|---" * (len(types) + 1) + "|",
    ]
    for method, group in _methods(rows).items():
        cells: list[str] = []
        for type_ in types:
            values = [recall_at(r.gold, r.at(k), k) for r in group if r.type == type_]
            cells.append(f"{_mean(values):.3f} ({len(values)})" if values else "n/a")
        lines.append(f"| {method} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def paired_table(
    rows: Sequence[EvidenceRow], pairs: Sequence[tuple[str, str]], ks: Sequence[int]
) -> str:
    """Mean per-question recall difference ``a - b`` with a paired 95% bootstrap CI."""
    grouped = {m: {r.question_id: r for r in g} for m, g in _methods(rows).items()}
    lines = ["| a - b | " + " | ".join(f"Δ recall@{k}" for k in ks) + " |",
             "|---" * (len(ks) + 1) + "|"]  # fmt: skip
    for a, b in pairs:
        if a not in grouped or b not in grouped:
            continue
        shared = sorted(grouped[a].keys() & grouped[b].keys())
        cells: list[str] = []
        for k in ks:
            diffs = [
                recall_at(grouped[a][q].gold, grouped[a][q].at(k), k)
                - recall_at(grouped[b][q].gold, grouped[b][q].at(k), k)
                for q in shared
            ]
            lo, hi = bootstrap_ci(diffs)
            cells.append(f"{_mean(diffs):+.3f} [{lo:+.3f}, {hi:+.3f}]")
        lines.append(f"| {a} - {b} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


# -- running the retrievers ------------------------------------------------------------

MAX_ANSWER_TYPES = 50
"""LLM-extracted graphs have hundreds of free-form types: the answer-type question is
offered only the most common ones (as in the M7 harness)."""


@dataclass(frozen=True)
class EvidenceQuestion:
    id: str
    text: str
    type: str
    gold: tuple[str, ...]
    """Gold document keys."""


def names_title(query: str, title: str) -> bool:
    """Whether ``query`` names ``title``, ignoring case and a trailing disambiguator
    ("Polish-Russian War (film)" is named by "Polish-Russian War")."""
    bare = re.sub(r"\s*\([^)]*\)\s*$", "", title).strip()
    return len(bare) >= 3 and bool(  # noqa: PLR2004
        re.search(rf"(?<!\w){re.escape(bare)}(?!\w)", query, re.IGNORECASE)
    )


def load_rows(path: Path) -> list[EvidenceRow] | None:
    if not path.exists():
        return None
    lines = path.read_text(encoding="utf-8").splitlines()
    return [EvidenceRow.model_validate_json(line) for line in lines]


def save_rows(path: Path, rows: Sequence[EvidenceRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text("".join(r.model_dump_json() + "\n" for r in rows), encoding="utf-8")
    tmp.replace(path)


def _cost(result: LocateResult) -> float | None:
    costs = [w.trace.totals.cost_usd for w in result.walks]
    if result.link is not None and result.link.decision_calls:
        costs.append(result.link.cost_usd)
    return None if any(c is None for c in costs) else math.fsum(c or 0.0 for c in costs)


async def retrieval_rows(
    store: NetworkXStore,
    questions: Sequence[EvidenceQuestion],
    titles: Mapping[str, str],
    *,
    decider: DecisionBackend,
    embedder: Embedder,
    config: TraversalConfig,
    ks: Sequence[int],
    concurrency: int = 8,
    checkpoints: Path | None = None,
    on_progress: Callable[[str, int, int], None] | None = None,
) -> list[EvidenceRow]:
    """Rows for dense, title+dense, graph, graph-choice, hybrid, and hybrid-choice.

    ``store`` must hold the documents (``put_document``); ``titles`` maps document title
    to key, for the title-match control. With ``checkpoints``, each method's rows are
    saved there and reused on a re-run (only the graph methods cost anything).
    """
    max_k = max(ks)
    type_counts = Counter([n.type async for n in store.iter_nodes()])
    traverser = Traverser(
        store,
        decider,
        embedder=embedder,
        config=config,
        node_types=sorted(t for t, _ in type_counts.most_common(MAX_ANSWER_TYPES)),
    )
    texts = sorted({node_text(n) async for n in store.iter_nodes()})
    if texts:
        traverser.preload_embeddings(texts, await embedder.embed(texts))
    docs = StoredDocuments(store)
    locators = {
        # as the library ships it: entities linked by name
        "graph": GraphLocator(
            store, traverser, names=NameEntryResolver(store, embedder=embedder), documents=docs
        ),
        # the entry entity chosen by Jev among the name matches (the M7 reader's linking)
        "graph-choice": GraphLocator(
            store,
            traverser,
            resolver=ChoiceEntryResolver(store, decider),
            names=NameEntryResolver(store),
            documents=docs,
        ),
    }
    dense = DenseLocator(store, docs, embedder)
    limit = asyncio.Semaphore(concurrency)

    async def per_question(
        method: str, run: Callable[[EvidenceQuestion], Awaitable[EvidenceRow]]
    ) -> list[EvidenceRow]:
        path = None if checkpoints is None else checkpoints / f"{method}.jsonl"
        saved = None if path is None else load_rows(path)
        if saved is not None:
            return saved
        done = 0

        async def one(question: EvidenceQuestion) -> EvidenceRow:
            nonlocal done
            async with limit:
                row = await run(question)
            done += 1
            if on_progress is not None:
                on_progress(method, done, len(questions))
            return row

        rows = list(await asyncio.gather(*(one(q) for q in questions)))
        if path is not None:
            save_rows(path, rows)
        return rows

    def row(question: EvidenceQuestion, method: str, **fields: Any) -> EvidenceRow:
        return EvidenceRow(
            question_id=question.id, type=question.type, gold=question.gold, method=method,
            **fields,
        )  # fmt: skip

    # Dense chunks are ranked individually; enough of them to cover max_k documents.
    dense_found: dict[str, list[Location]] = {}
    for question in questions:  # local and cheap; kept in memory for the hybrids
        dense_found[question.id] = first_per_document(
            await dense.locate(question.text, k=max_k * 8)
        )[:max_k]

    async def run_dense(question: EvidenceQuestion) -> EvidenceRow:
        return row(question, "dense", ranked=tuple(distinct_keys(dense_found[question.id])),
                   cost_usd=0.0)  # fmt: skip

    rows = await per_question("dense", run_dense)
    # Control: is the graph just matching document titles named in the question?
    by_length = sorted(titles.items(), key=lambda t: -len(t[0]))
    for question in questions:
        named = [key for title, key in by_length if names_title(question.text, title)]
        ranked = list(dict.fromkeys([*named, *distinct_keys(dense_found[question.id])]))
        rows.append(row(question, "title+dense", ranked=tuple(ranked[:max_k]), cost_usd=0.0))

    for method, locator in locators.items():

        async def run_graph(
            question: EvidenceQuestion, locator: GraphLocator = locator, method: str = method
        ) -> EvidenceRow:
            started = time.perf_counter()
            try:
                result = await locator.locate(question.text, k=max_k)
            except Exception as error:  # noqa: BLE001 - recorded, scored as empty
                return row(question, method, ranked=(), error=repr(error))
            return row(
                question,
                method,
                ranked=tuple(distinct_keys(result.locations)),
                locations=tuple(result.locations),
                decision_calls=result.decision_calls,
                cost_usd=_cost(result),
                latency_s=time.perf_counter() - started,
            )

        graph_rows = await per_question(method, run_graph)
        rows += [r.model_copy(update={"locations": ()}) for r in graph_rows]
        for graph_row, question in zip(graph_rows, questions, strict=True):
            fused = {
                k: tuple(hybrid_keys(graph_row.locations, dense_found[question.id], k)) for k in ks
            }
            rows.append(
                row(question, method.replace("graph", "hybrid"), ranked=fused[max_k], by_k=fused,
                    decision_calls=graph_row.decision_calls, cost_usd=graph_row.cost_usd)
            )  # fmt: skip
    return rows
