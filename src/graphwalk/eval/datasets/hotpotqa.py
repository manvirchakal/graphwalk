"""HotpotQA (Yang et al., 2018), distractor setting, validation split.

Each question comes with 10 paragraphs (2 gold, 8 distractors). Types are ``bridge``
(multi-hop through a bridge entity) and ``comparison``. Answers are short spans,
often ``yes``/``no``.

Rows come from the Hugging Face datasets-server API, which serves the dataset's
current revision; the revision seen at download time is saved next to the cache and
must equal :data:`REVISION`, so a changed upstream fails loudly instead of silently.
"""

import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, cast

from graphwalk.eval.datasets.cache import cache_dir

DATASET = "hotpotqa/hotpot_qa"
REVISION = "1908d6afbbead072334abe2965f91bd2709910ab"
ROWS = "https://datasets-server.huggingface.co/rows"
INFO = f"https://huggingface.co/api/datasets/{DATASET}"
PAGE = 100
TYPES = ("bridge", "comparison")


def _get(url: str, attempts: int = 5) -> Any:
    """GET JSON, retrying server errors (the rows API returns transient 5xx)."""
    request = urllib.request.Request(url, headers={"User-Agent": "graphwalk-eval/0.1"})  # noqa: S310
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310
                return json.load(response)
        except urllib.error.HTTPError as error:
            if error.code < 500 or attempt == attempts - 1:  # noqa: PLR2004
                raise
            time.sleep(2**attempt)
    msg = "unreachable"
    raise AssertionError(msg)


def download() -> Path:
    """The validation split as JSON records (cached)."""
    path = cache_dir() / "hotpotqa" / "validation-distractor.json"
    if path.exists():
        return path
    revision = _get(INFO).get("sha")
    if revision != REVISION:
        msg = f"{DATASET} is at revision {revision}, expected {REVISION}"
        raise RuntimeError(msg)
    rows: list[dict[str, Any]] = []
    offset = 0
    while True:
        url = (
            f"{ROWS}?dataset={DATASET}&config=distractor&split=validation"
            f"&offset={offset}&length={PAGE}"
        )
        page = _get(url)
        rows.extend(r["row"] for r in page["rows"])
        offset += PAGE
        if offset >= page["num_rows_total"]:
            break
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    tmp.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)
    (path.parent / "REVISION").write_text(REVISION, encoding="utf-8")
    return path


def read_records(path: Path) -> list[dict[str, Any]]:
    """Records with ``context`` as ``[[title, [sentences...]], ...]`` (the 2Wiki shape)."""
    records: list[dict[str, Any]] = []
    for row in cast("list[dict[str, Any]]", json.loads(path.read_text(encoding="utf-8"))):
        context = row["context"]
        records.append(
            {
                "_id": row["id"],
                "question": row["question"],
                "answer": row["answer"],
                "type": row["type"],
                "level": row.get("level"),
                "context": [
                    [title, sentences]
                    for title, sentences in zip(context["title"], context["sentences"], strict=True)
                ],
            }
        )
    return records
