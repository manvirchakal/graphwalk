"""FanOutQA (Zhu et al., 2024), dev set: fan-out questions over English Wikipedia.

Each question needs facts from many pages (about 7 on average), and most answers are
sets or entity -> value maps, e.g. "the batting hand of each of the first five picks
in the 1998 MLB draft". The evidence pages are fetched at their pinned revisions from
the Wikipedia API, converted to plain text (tables and infoboxes become one line per
row), and cached.

The official accuracy metric (``fanoutqa.eval.string.answer_in_text``) is implemented
in :func:`answer_in_text`, without its spaCy lemmatization step, so this is slightly
stricter than the official score.
"""

import hashlib
import importlib
import itertools
import json
import re
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, cast

from graphwalk.eval.datasets.cache import cache_dir, fetch

URL = (
    "https://raw.githubusercontent.com/zhudotexe/fanoutqa/main/fanoutqa/data/fanout-final-dev.json"
)
WIKI_API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "graphwalk-eval/0.1 (research benchmark; https://github.com/manvirchakal/graphwalk)"


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


# -- Wikipedia HTML -> text ---------------------------------------------------------------

_SKIP = {"script", "style", "sup", "math", "figure", "figcaption"}
_SKIP_CLASSES = ("reference", "mw-editsection", "navbox", "reflist", "hatnote", "noprint")
_STOP_SECTIONS = {"references", "external links", "see also", "notes", "further reading",
                  "bibliography", "sources", "citations", "footnotes"}  # fmt: skip
_BLOCK = {"p", "li", "h1", "h2", "h3", "h4", "h5", "h6", "dd", "dt", "div", "br", "caption"}


class _TextParser(HTMLParser):
    """Paragraphs and headings as lines; each table row as ``cell | cell | ...``."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.lines: list[str] = []
        self._buf: list[str] = []
        self._skip = 0
        self._stack: list[tuple[str, bool]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None
        self._heading: str | None = None
        self.stopped = False

    def finish(self) -> None:
        """Emit the trailing paragraph."""
        self._flush()

    def _flush(self) -> None:
        text = " ".join("".join(self._buf).split())
        if text:
            self.lines.append(text)
        self._buf = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if self.stopped:
            return
        classes = next((v or "" for k, v in attrs if k == "class"), "")
        skip = tag in _SKIP or any(c in classes for c in _SKIP_CLASSES)
        if tag in {"br", "img", "hr", "meta", "link", "input", "wbr"}:
            if tag == "br":
                self._text(" ")
            return
        self._stack.append((tag, skip))
        if skip:
            self._skip += 1
            return
        if self._skip:
            return
        if tag == "tr":
            self._flush()
            self._row = []
        elif tag in {"td", "th"} and self._row is not None:
            self._cell = []
        elif tag in {"h2", "h3", "h4"}:
            self._flush()
            self._heading = ""
        elif tag in _BLOCK and self._cell is None:
            self._flush()

    def handle_endtag(self, tag: str) -> None:
        if self.stopped:
            return
        while self._stack:
            open_tag, skip = self._stack.pop()
            if skip:
                self._skip -= 1
            if open_tag == tag:
                break
        if self._skip:
            return
        if tag in {"td", "th"} and self._row is not None and self._cell is not None:
            self._row.append(" ".join("".join(self._cell).split()))
            self._cell = None
        elif tag == "tr" and self._row is not None:
            cells = [c for c in self._row if c]
            if cells:
                self.lines.append(" | ".join(cells))
            self._row = None
        elif tag in {"h2", "h3", "h4"} and self._heading is not None:
            heading = " ".join(self._heading.split())
            self._heading = None
            if heading.lower() in _STOP_SECTIONS:
                self.stopped = True
                return
            if heading:
                self.lines.append(f"## {heading}")
        elif tag in _BLOCK and self._cell is None:
            self._flush()

    def _text(self, data: str) -> None:
        if self._heading is not None:
            self._heading += data
        elif self._cell is not None:
            self._cell.append(data)
        else:
            self._buf.append(data)

    def handle_data(self, data: str) -> None:
        if not self.stopped and not self._skip:
            self._text(data)


def html_to_text(html: str) -> str:
    parser = _TextParser()
    parser.feed(html)
    parser.close()
    parser.finish()
    return "\n".join(parser.lines)


def _get_json(params: Mapping[str, str], attempts: int = 5) -> Any:
    url = f"{WIKI_API}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})  # noqa: S310
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
                return json.load(response)
        except urllib.error.HTTPError as error:
            if error.code not in (429, 500, 502, 503, 504) or attempt == attempts - 1:
                raise
            time.sleep(5 * 2**attempt)
    msg = "unreachable"
    raise AssertionError(msg)


def page_text(page: Evidence, *, pause_s: float = 0.5) -> str:
    """The page's text at its pinned revision (cached)."""
    path = cache_dir() / "fanoutqa" / "pages" / f"{page.pageid}-{page.revid}.txt"
    if path.exists():
        return path.read_text(encoding="utf-8")
    data = _get_json(
        {
            "action": "parse",
            "oldid": str(page.revid),
            "prop": "text",
            "format": "json",
            "formatversion": "2",
        }
    )
    text = html_to_text(str(data["parse"]["text"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    time.sleep(pause_s)  # be polite to the API
    return text


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

    Wikipedia's API now rate-limits bulk fetching (see :func:`page_text`), so the pages
    come from a community mirror of the FanOutQA dev corpus, pinned to a revision. Its
    page ids match all 1,562 dev evidence pages; revisions cannot be checked (the mirror
    stores none), so answers may occasionally differ from the page text.
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
