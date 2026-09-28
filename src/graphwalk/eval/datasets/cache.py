"""Dataset cache and downloads. Downloads are pinned to exact revisions."""

import os
import shutil
import urllib.request
from pathlib import Path


def cache_dir() -> Path:
    """``$GRAPHWALK_CACHE_DIR`` or ``~/.cache/graphwalk``, plus ``datasets``."""
    root = os.environ.get("GRAPHWALK_CACHE_DIR")
    base = Path(root) if root else Path.home() / ".cache" / "graphwalk"
    return base / "datasets"


def fetch(url: str, dest: Path) -> Path:
    """Download ``url`` to ``dest`` once (atomic rename). Honors proxy env vars."""
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "graphwalk-eval/0.1"})  # noqa: S310
    with urllib.request.urlopen(request, timeout=300) as response, tmp.open("wb") as out:  # noqa: S310
        shutil.copyfileobj(response, out)
    tmp.replace(dest)
    return dest
