"""FanOutQA (Zhu et al., 2024), dev set: fan-out questions over English Wikipedia.

Each question needs facts from many pages (about 7 on average), and most answers are
sets or entity -> value maps, e.g. "the batting hand of each of the first five picks
in the 1998 MLB draft". The evidence pages come from a pinned mirror of the dev corpus
(:func:`load_corpus`), since Wikipedia's API rate-limits bulk fetching.

The official accuracy metric (``fanoutqa.eval.string.answer_in_text``) is implemented
in :func:`answer_in_text`, without its spaCy lemmatization step, so this is slightly
stricter than the official score.
"""

import hashlib
import importlib
import itertools
import json
import re
import unicodedata
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from graphwalk.eval.datasets.cache import cache_dir, fetch

URL = (
    "https://raw.githubusercontent.com/zhudotexe/fanoutqa/main/fanoutqa/data/fanout-final-dev.json"
)


def download() -> Path:
    return fetch(URL, cache_dir() / "fanoutqa" / "fanout-final-dev.json")


def dev_sha256(path: Path) -> str:
    """Recorded with results: the upstream file is not pinned to a commit."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_questions(path: Path) -> list[dict[str, Any]]:
    return cast("list[dict[str, Any]]", json.loads(path.read_text(encoding="utf-8")))


@dataclass(frozen=True)
class Evidence:
    pageid: int
    revid: int
    title: str


def evidence(question: Mapping[str, Any]) -> list[Evidence]:
    """The pages the human decomposition used, de-duplicated, in order."""
    out: dict[int, Evidence] = {}

    def walk(subs: Sequence[Mapping[str, Any]]) -> None:
        for sub in subs:
            ev: Mapping[str, Any] = sub.get("evidence") or {}
            # A few upstream entries have placeholder ids ("###TBD###"): no page.
            ids = (str(ev.get("pageid", "")), str(ev.get("revid", "")))
            if all(i.isdigit() for i in ids):
                page = Evidence(int(ev["pageid"]), int(ev["revid"]), str(ev["title"]))
                out.setdefault(page.pageid, page)
            walk(sub.get("decomposition") or [])

    walk(question.get("decomposition") or [])
    return list(out.values())


# -- the official accuracy metric, minus lemmatization --------------------------------------


def normalize(text: object) -> str:
    """Lowercase, unicode-normalize, drop thousands separators and ``,.?!:;``."""
    value = unicodedata.normalize("NFKC", str(text).lower())
    value = re.sub(r"(\d+,)+\d+(\.\d+)?", lambda m: m[0].replace(",", ""), value)
    value = re.sub(r"[,.?!:;]", "", value)
    return re.sub(r"\s+", " ", value).strip()


def _leaves(reference: object) -> Iterator[str]:
    if isinstance(reference, list):
        for item in cast("list[object]", reference):
            yield from _leaves(item)
    elif isinstance(reference, dict):
        ref = cast("dict[object, object]", reference)
        for item in itertools.chain(ref.keys(), ref.values()):
            yield from _leaves(item)
    elif isinstance(reference, bool):
        yield "yes" if reference else "no"
    else:
        yield str(reference)


def answer_in_text(reference: object, candidate: str) -> tuple[float, bool]:
    """``(loose, strict)``: the share of reference strings (keys and values for a dict)
    found in ``candidate`` with word boundaries, and whether all were."""
    leaves = list(_leaves(reference))
    if not leaves:
        return 0.0, False
    text = normalize(candidate)
    found = sum(bool(re.search(rf"\b{re.escape(normalize(leaf))}\b", text)) for leaf in leaves)
    return found / len(leaves), found == len(leaves)


# -- the evidence corpus --------------------------------------------------------------------

CORPUS_DATASET = "JinChao1022/fanoutqa-retrieval"
CORPUS_REVISION = "03addfe298d5886dd87fa8e0592974b86538f2b2"
CORPUS_URL = (
    f"https://huggingface.co/datasets/{CORPUS_DATASET}/resolve/{CORPUS_REVISION}"
    "/corpus/data-00000-of-00001.arrow"
)


def load_corpus() -> dict[str, dict[str, str]]:
    """Page id -> ``{"title", "text"}`` (Markdown) for every dev evidence page.

    Wikipedia's API rate-limits bulk fetching, so the pages come from a community
    mirror of the FanOutQA dev corpus, pinned to a revision. Its page ids match all
    1,562 dev evidence pages; revisions cannot be checked (the mirror stores none), so
    answers may occasionally differ from the page text.
    """
    path = cache_dir() / "fanoutqa" / "corpus.json"
    if path.exists():
        return cast("dict[str, dict[str, str]]", json.loads(path.read_text(encoding="utf-8")))
    # The eval extra's pyarrow ships no type stubs; only needed once, to convert.
    pa: Any = importlib.import_module("pyarrow")

    arrow = fetch(CORPUS_URL, cache_dir() / "fanoutqa" / "corpus.arrow")
    with pa.memory_map(str(arrow)) as source:
        rows = cast("list[dict[str, str]]", pa.ipc.open_stream(source).read_all().to_pylist())
    corpus = {str(r["doc_id"]): {"title": r["title"], "text": r["text"]} for r in rows}
    path.write_text(json.dumps(corpus, ensure_ascii=False), encoding="utf-8")
    return corpus


def records(
    questions: Sequence[Mapping[str, Any]], corpus: Mapping[str, Mapping[str, str]], max_chars: int
) -> list[dict[str, Any]]:
    """Questions in the harness's record shape: ``context`` holds each evidence page,
    truncated to ``max_chars``; ``answer`` keeps the structured reference."""
    out: list[dict[str, Any]] = []
    for q in questions:
        context = [
            [corpus[str(p.pageid)]["title"], [corpus[str(p.pageid)]["text"][:max_chars]]]
            for p in evidence(q)
            if str(p.pageid) in corpus
        ]
        out.append(
            {
                "_id": q["id"],
                "question": q["question"],
                "answer": q["answer"],
                "type": "fanout",
                "categories": q.get("categories", []),
                "context": context,
            }
        )
    return out
