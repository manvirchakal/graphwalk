"""2WikiMultiHopQA (Ho et al., 2020), dev split, as one pooled evidence graph.

Each question ships gold evidence triples. A graph built from one question's triples
*is* its reasoning chain, which would make traversal trivial, so the graph pools the
evidence triples of **every** dev question (33k nodes, 31k triples) and each question
is walked on that. Only ``compositional`` and ``inference`` questions are used: their
answer is the node at the end of a 2-hop walk. ``comparison`` and
``bridge_comparison`` need a comparison computed in code, which graph-only traversal
cannot answer.

Source: ``huggingface.co/datasets/voidful/2WikiMultihopQA`` (original JSON), pinned.
"""

import json
from pathlib import Path
from typing import Any, cast

from graphwalk.eval.datasets.cache import cache_dir, fetch
from graphwalk.eval.datasets.metaqa import build_store
from graphwalk.eval.types import EvalQuestion
from graphwalk.stores.networkx_store import NetworkXStore

REVISION = "16852fde9d85cba158cf7e6517e7a3f9415a28c0"
URL = f"https://huggingface.co/datasets/voidful/2WikiMultihopQA/resolve/{REVISION}/dev.json"
WALKABLE_TYPES = ("compositional", "inference")


def download() -> Path:
    return fetch(URL, cache_dir() / "2wiki" / "dev.json")


def read_records(path: Path) -> list[dict[str, Any]]:
    return cast("list[dict[str, Any]]", json.loads(path.read_text(encoding="utf-8")))


def pooled_triples(records: list[dict[str, Any]]) -> list[tuple[str, str, str]]:
    triples: list[tuple[str, str, str]] = []
    for record in records:
        for subject, relation, obj in record.get("evidences", []):
            if subject and relation and obj:
                triples.append((str(subject), str(relation), str(obj)))
    return list(dict.fromkeys(triples))


async def build_graph(records: list[dict[str, Any]]) -> NetworkXStore:
    # Subjects get the "film" type in build_store; relabel for 2Wiki's mixed subjects.
    store = await build_store(pooled_triples(records), source_id="2wiki:dev:evidences")
    async for node in store.iter_nodes():
        await store.upsert_node(node.model_copy(update={"type": "entity"}))
    return store


def read_questions(records: list[dict[str, Any]]) -> list[EvalQuestion]:
    questions: list[EvalQuestion] = []
    for record in records:
        if record.get("type") not in WALKABLE_TYPES or not record.get("evidences"):
            continue
        evidences = record["evidences"]
        start = str(evidences[0][0])
        questions.append(
            EvalQuestion(
                id=f"2wiki-{record['_id']}",
                dataset="2wiki",
                question=str(record["question"]),
                answers=(str(record["answer"]),),
                kind="text",
                start=None,
                gold_start=(start,),
                meta={"type": str(record["type"]), "hops": len(evidences)},
            )
        )
    return questions
