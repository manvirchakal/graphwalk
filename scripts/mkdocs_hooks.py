"""MkDocs hook: links from docs to files outside ``docs/`` point at GitHub instead.

The results pages link to run directories, scripts, and the roadmap; those files are
not part of the site, so a relative link would 404 there.
"""

import posixpath
import re
from pathlib import Path
from typing import Any

REPO = "https://github.com/manvirchakal/graphwalk"
ROOT = Path(__file__).resolve().parent.parent
LINK = re.compile(r"(\]\()([^)\s#]+)(#[^)\s]*)?(\))")


def on_page_markdown(markdown: str, page: Any, **_: Any) -> str:
    source = posixpath.dirname(f"docs/{page.file.src_uri}")

    def fix(match: re.Match[str]) -> str:
        target = match.group(2)
        if re.match(r"^[a-z]+:", target) or target.startswith("/"):
            return match.group(0)
        path = posixpath.normpath(posixpath.join(source, target))
        if path.startswith("docs/") or path.startswith(".."):
            return match.group(0)
        kind = "tree" if (ROOT / path).is_dir() else "blob"
        anchor = match.group(3) or ""
        return f"{match.group(1)}{REPO}/{kind}/main/{path}{anchor}{match.group(4)}"

    return LINK.sub(fix, markdown)
