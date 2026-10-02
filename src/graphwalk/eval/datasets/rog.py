"""WebQSP and CWQ over Freebase, as per-question subgraphs (Luo et al., 2024, "Reasoning
on Graphs"; the release used by RoG and by later KG-QA work).

Each question comes with its own subgraph: the Freebase triples within a few hops of
its topic entities (median about 4,400 for WebQSP). Entities are named; compound value
type nodes (Freebase's reified n-ary facts, e.g. a marriage with its dates) keep their
``m.``/``g.`` ids as names. The topic entities are given, as in the papers this
release is used for. A few percent of questions have no answer inside their subgraph;
they are kept (and counted), so scores are comparable to published ones.

Source: ``huggingface.co/datasets/rmanluo/RoG-webqsp`` and ``.../RoG-cwq`` (parquet),
pinned below. Needs the ``eval`` extra (pyarrow).
"""

import importlib
import json
import re
import urllib.request
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, cast

from graphwalk.core.hashing import content_hash
from graphwalk.core.model import Edge, Node, Provenance, utc_now
from graphwalk.eval.datasets.cache import cache_dir, fetch
from graphwalk.eval.types import EvalQuestion
from graphwalk.stores.networkx_store import NetworkXStore
from graphwalk.stores.sqlite_store import SQLiteStore

type KGQADataset = Literal["webqsp", "cwq"]
type Triple = tuple[str, str, str]

REPOS: dict[KGQADataset, tuple[str, str]] = {
    "webqsp": ("rmanluo/RoG-webqsp", "c0632533135a06f8c5d536b420deec5fcb5c58f3"),
    "cwq": ("rmanluo/RoG-cwq", "b0f6275586286312c4e99ef4b0adc65463c91f8e"),
}
CVT = re.compile(r"^[mg]\.[0-9a-z_]+$")
CVT_TYPE = "cvt"
ENTITY_TYPE = "entity"
CVT_SUMMARY = "a compound value: an event or relationship that links several entities"


def _shards(repo: str, revision: str, split: str) -> list[str]:
    url = f"https://huggingface.co/api/datasets/{repo}/tree/{revision}/data"
    request = urllib.request.Request(url, headers={"User-Agent": "graphwalk-eval/0.1"})
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
        listing = cast("list[dict[str, object]]", json.load(response))
    return sorted(
        str(item["path"])
        for item in listing
        if str(item["path"]).startswith(f"data/{split}-") and str(item["path"]).endswith(".parquet")
    )


def download(dataset: KGQADataset, split: str = "test") -> list[Path]:
    """The split's parquet shards, downloaded once into the cache."""
    repo, revision = REPOS[dataset]
    root = cache_dir() / "rog" / dataset / revision
    manifest = root / f"{split}.json"
    if manifest.exists():
        names = cast("list[str]", json.loads(manifest.read_text(encoding="utf-8")))
    else:
        names = _shards(repo, revision, split)
        if not names:
            msg = f"{repo}@{revision}: no {split} shards"
            raise ValueError(msg)
    paths = [
        fetch(
            f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{name}",
            root / Path(name).name,
        )
        for name in names
    ]
    manifest.write_text(json.dumps(names), encoding="utf-8")
    return paths


def load(
    dataset: KGQADataset, split: str = "test"
) -> tuple[list[EvalQuestion], dict[str, list[Triple]]]:
    """Questions (topic entities as ``start``) and each question's subgraph by id."""
    # The eval extra's pyarrow ships no type stubs.
    pq: Any = importlib.import_module("pyarrow.parquet")

    questions: list[EvalQuestion] = []
    graphs: dict[str, list[Triple]] = {}
    for path in download(dataset, split):
        for record in cast("list[dict[str, object]]", pq.read_table(path).to_pylist()):
            qid = str(record["id"])
            answers = tuple(dict.fromkeys(str(a) for a in cast("list[str]", record["answer"])))
            if not answers:
                continue
            triples = [
                (str(h), str(r), str(t))
                for h, r, t in cast("list[list[str]]", record["graph"])
                if h and r and t
            ]
            graphs[qid] = list(dict.fromkeys(triples))
            names = {x for h, _, t in graphs[qid] for x in (h, t)}
            start = tuple(str(e) for e in cast("list[str]", record["q_entity"]) if e in names)
            questions.append(
                EvalQuestion(
                    id=qid,
                    dataset=dataset,
                    question=str(record["question"]),
                    answers=answers,
                    kind="set",
                    start=start,
                    gold_start=start,
                    meta={
                        "answer_in_graph": any(a in names for a in answers),
                        "graph_triples": len(graphs[qid]),
                    },
                )
            )
    return questions, graphs


def node_type(node_id: str) -> str:
    return CVT_TYPE if CVT.match(node_id) else ENTITY_TYPE


async def build_store(triples: Sequence[Triple], source_id: str) -> NetworkXStore:
    """One question's subgraph. Node ids and names are the release's entity strings."""
    provenance = (
        Provenance(
            source_id=source_id,
            ingested_at=utc_now(),
            confidence=1.0,
            content_hash=content_hash(source_id),
        ),
    )
    store = NetworkXStore()
    for name in sorted({x for h, _, t in triples for x in (h, t)}):
        type_ = node_type(name)
        await store.upsert_node(
            Node(
                id=name,
                type=type_,
                name=name,
                summary=CVT_SUMMARY if type_ == CVT_TYPE else None,
                provenance=provenance,
            )
        )
    for head, relation, tail in triples:
        await store.upsert_edge(
            Edge(source=head, target=tail, type=relation, provenance=provenance)
        )
    return store


async def build_global_store(
    graphs: Iterable[Sequence[Triple]], path: Path, *, source_id: str, batch_size: int = 5000
) -> SQLiteStore:
    """Every question's subgraph merged into one SQLite graph at ``path`` (built once;
    reopened if a finished build is there). The merged schema is the union of the
    subgraphs' relations: thousands, too many to curate per question."""
    store = SQLiteStore(path)
    if await store.get_metadata("rog_global") == source_id:
        return store
    await store.clear()
    triples = list(dict.fromkeys(t for g in graphs for t in g))
    provenance = (
        Provenance(
            source_id=source_id,
            ingested_at=utc_now(),
            confidence=1.0,
            content_hash=content_hash(source_id),
        ),
    )
    names = sorted({x for h, _, t in triples for x in (h, t)})
    for start in range(0, len(names), batch_size):
        nodes: list[Node] = []
        for name in names[start : start + batch_size]:
            type_ = node_type(name)
            nodes.append(
                Node(
                    id=name,
                    type=type_,
                    name=name,
                    summary=CVT_SUMMARY if type_ == CVT_TYPE else None,
                    provenance=provenance,
                )
            )
        await store.upsert_many(nodes, ())
    for start in range(0, len(triples), batch_size):
        edges = [
            Edge(source=h, target=t, type=r, provenance=provenance)
            for h, r, t in triples[start : start + batch_size]
        ]
        await store.upsert_many((), edges)
    await store.set_metadata("rog_global", source_id)
    return store


def relation_signature(triples: Sequence[Triple]) -> Mapping[str, tuple[str, str]]:
    """Relation -> (subject type, object type), for the LLM-written-path baseline."""
    out: dict[str, tuple[str, str]] = {}
    for head, relation, tail in triples:
        out.setdefault(relation, (node_type(head), node_type(tail)))
    return out


LITERAL = re.compile(r"^[-+]?\d[\d,.:\-TZ ]*$")
"""Numbers and dates: kept as they are by :func:`anonymize`."""


MIN_MENTION = 4
"""Shortest other entity name replaced where it appears in a question."""


def alias(name: str, salt: str = "") -> str:
    return f"Entity {content_hash(salt + name).removeprefix('sha256:')[:6]}"


def anonymize(
    questions: Sequence[EvalQuestion], graphs: Mapping[str, Sequence[Triple]], salt: str = ""
) -> tuple[list[EvalQuestion], dict[str, list[Triple]]]:
    """Replace every entity name with a stable meaningless alias (``Entity 3fa9c1``):
    in the triples, the gold answers, the topic entities, and the topic entities'
    names where they appear in the question (case-insensitive substrings, so "jamaican"
    loses "jamaica" too), plus any other name from the question's subgraph that appears
    in the question as whole words (at least 4 characters). Relations, compound
    value ids, numbers, and dates are kept.

    A model can then not answer from memory: the answer strings no longer exist outside
    the graph. ``meta["topic_in_question"]`` records whether every topic entity's name
    was found in the question (when not, the question still names it some other way,
    so the model may still know what it is about).
    """

    def rename(x: str) -> str:
        return x if CVT.match(x) or LITERAL.match(x) else alias(x, salt)

    out_graphs = {
        qid: [(rename(h), r, rename(t)) for h, r, t in triples] for qid, triples in graphs.items()
    }
    out_questions: list[EvalQuestion] = []
    for q in questions:
        text = q.question
        found = True
        for name in sorted(q.start or (), key=len, reverse=True):
            pattern = re.compile(re.escape(name), re.IGNORECASE)
            text, count = pattern.subn(rename(name), text)
            found = found and count > 0
        lowered = text.lower()
        others = {
            x
            for h, _, t in graphs.get(q.id, ())
            for x in (h, t)
            if len(x) >= MIN_MENTION and x.lower() in lowered and rename(x) != x
        }
        for name in sorted(others, key=len, reverse=True):
            pattern = re.compile(rf"\b{re.escape(name)}\b", re.IGNORECASE)
            text = pattern.sub(rename(name), text)
        out_questions.append(
            q.model_copy(
                update={
                    "question": text,
                    "answers": tuple(rename(a) for a in q.answers),
                    "start": None if q.start is None else tuple(rename(s) for s in q.start),
                    "gold_start": None
                    if q.gold_start is None
                    else tuple(rename(s) for s in q.gold_start),
                    "meta": {**q.meta, "topic_in_question": found},
                }
            )
        )
    return out_questions, out_graphs
