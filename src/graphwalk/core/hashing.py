"""Stable content hashing, used for provenance and ingestion idempotency."""

import hashlib
import json

from pydantic import JsonValue

HASH_PREFIX = "sha256:"


def content_hash(content: str | bytes | JsonValue) -> str:
    """Return ``sha256:<hex>`` for text, bytes, or any JSON value.

    JSON values are canonicalized (sorted keys, no insignificant whitespace), so two
    structurally equal values always hash the same regardless of key order. Strings are
    hashed as their UTF-8 bytes, not as a JSON string literal, so hashing a file's text
    matches hashing its bytes.
    """
    if isinstance(content, bytes):
        data = content
    elif isinstance(content, str):
        data = content.encode("utf-8")
    else:
        data = json.dumps(
            content, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode("utf-8")
    return HASH_PREFIX + hashlib.sha256(data).hexdigest()
