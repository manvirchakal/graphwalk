"""MetaQA (Zhang et al., 2018): movie KB with 1/2/3-hop questions and set answers.

Source: the original files mirrored at ``huggingface.co/datasets/camazlucas/MetaQA``,
pinned below. ``kb.txt`` has 134,741 ``subject|relation|object`` triples; node ids are
entity names (as in the original). The topic entity is the ``[bracketed]`` span.
"""

import re
from pathlib import Path

from graphwalk.core.hashing import content_hash
from graphwalk.core.model import Edge, Node, Provenance, utc_now
from graphwalk.eval.datasets.cache import cache_dir, fetch
from graphwalk.eval.types import EvalQuestion
from graphwalk.stores.networkx_store import NetworkXStore

REVISION = "f83854092893610686ab19f2038c039afa9bc643"
BASE_URL = f"https://huggingface.co/datasets/camazlucas/MetaQA/resolve/{REVISION}"

OBJECT_TYPES = {
    "directed_by": "person",
    "written_by": "person",
    "starred_actors": "person",
    "release_year": "year",
    "in_language": "language",
    "has_genre": "genre",
    "has_tags": "tag",
    "has_imdb_rating": "rating",
    "has_imdb_votes": "votes",
}
RELATION_GLOSSES = {
    "directed_by": "the director of the film",
    "written_by": "a writer of the film",
    "starred_actors": "an actor who starred in the film",
    "release_year": "the year the film was released",
    "in_language": "the language of the film",
    "has_genre": "the genre of the film",
    "has_tags": "keywords describing the film (topics, people, themes)",
    "has_imdb_rating": "the film's IMDb rating",
    "has_imdb_votes": "how many IMDb votes the film has",
}
"""Schema documentation for ``TraversalConfig.relation_glosses`` (written for the tuning
round in M5; see docs/results-m5.md)."""

_TOPIC = re.compile(r"\[([^\]]+)\]")


def download(hops: int, split: str = "test") -> tuple[Path, Path]:
    root = cache_dir() / "metaqa"
    kb = fetch(f"{BASE_URL}/kb/kb.txt", root / "kb.txt")
    qa = fetch(f"{BASE_URL}/{hops}-hop/vanilla/qa_{split}.txt", root / f"{hops}-hop_qa_{split}.txt")
    return kb, qa


def read_triples(kb_path: Path) -> list[tuple[str, str, str]]:
    triples: list[tuple[str, str, str]] = []
    for line in kb_path.read_text(encoding="utf-8").splitlines():
        parts = line.split("|")
        if len(parts) == 3 and all(parts):  # noqa: PLR2004 - subject|relation|object
            triples.append((parts[0], parts[1], parts[2]))
    return list(dict.fromkeys(triples))


async def build_store(triples: list[tuple[str, str, str]], source_id: str) -> NetworkXStore:
    """Films are subjects; object types follow the relation. A name that is both a
    subject and an object keeps the ``film`` type."""
    provenance = (
        Provenance(
            source_id=source_id,
            ingested_at=utc_now(),
            confidence=1.0,
            content_hash=content_hash(source_id),
        ),
    )
    types: dict[str, str] = {}
    for subject, _, _ in triples:
        types[subject] = "film"
    for _, relation, obj in triples:
        types.setdefault(obj, OBJECT_TYPES.get(relation, "entity"))
    store = NetworkXStore()
    for name, type_ in sorted(types.items()):
        await store.upsert_node(Node(id=name, type=type_, name=name, provenance=provenance))
    for subject, relation, obj in triples:
        await store.upsert_edge(
            Edge(source=subject, target=obj, type=relation, provenance=provenance)
        )
    return store


def read_questions(qa_path: Path, hops: int, split: str = "test") -> list[EvalQuestion]:
    questions: list[EvalQuestion] = []
    lines = qa_path.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        if "\t" not in line:
            continue
        text, answers = line.split("\t", 1)
        topic = _TOPIC.search(text)
        if topic is None:
            continue
        gold = tuple(a for a in answers.split("|") if a)
        if not gold:
            continue
        questions.append(
            EvalQuestion(
                id=f"metaqa-{hops}hop-{split}-{index}",
                dataset=f"metaqa-{hops}hop",
                question=_TOPIC.sub(r"\1", text),
                answers=gold,
                kind="set",
                start=(topic.group(1),),
                gold_start=(topic.group(1),),
                meta={"hops": hops},
            )
        )
    return questions
