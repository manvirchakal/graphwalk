"""A JSON-file-backed extraction cache that survives crashes."""

import json
from collections.abc import Iterator, MutableMapping
from pathlib import Path
from typing import cast

from pydantic import JsonValue


class JsonFileCache(MutableMapping[str, JsonValue]):
    """Chunk key -> extraction, persisted to ``path``.

    Every ``flush_every`` new entries are written out (atomically: temp file + rename),
    so an interrupted ingestion keeps the extractions it already paid for. Call
    :meth:`flush` at the end to write the rest.
    """

    def __init__(self, path: str | Path, *, flush_every: int = 25) -> None:
        self.path = Path(path)
        self._flush_every = max(1, flush_every)
        self._unsaved = 0
        self._data: dict[str, JsonValue] = {}
        if self.path.exists():
            self._data = cast(
                "dict[str, JsonValue]", json.loads(self.path.read_text(encoding="utf-8"))
            )

    def __getitem__(self, key: str) -> JsonValue:
        return self._data[key]

    def __setitem__(self, key: str, value: JsonValue) -> None:
        self._data[key] = value
        self._unsaved += 1
        if self._unsaved >= self._flush_every:
            self.flush()

    def __delitem__(self, key: str) -> None:
        del self._data[key]
        self._unsaved += 1

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def flush(self) -> None:
        if not self._unsaved and self.path.exists():
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(f".{self.path.name}.tmp")
        tmp.write_text(json.dumps(self._data, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.path)
        self._unsaved = 0
