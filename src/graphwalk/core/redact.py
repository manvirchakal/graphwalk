"""Keep secrets out of error messages and logs."""

from collections.abc import Iterable

MASK = "***"
_MIN_SECRET = 6
"""Shorter strings are not masked: they would mangle ordinary text."""


def redact(text: str, secrets: Iterable[str | None]) -> str:
    """``text`` with every occurrence of each secret (and its last 8 characters, which
    some providers echo in error messages) replaced by ``***``."""
    for secret in secrets:
        if not secret or len(secret) < _MIN_SECRET:
            continue
        text = text.replace(secret, MASK)
        tail = secret[-8:]
        if len(secret) > 12 and tail in text:  # noqa: PLR2004 - keep short keys whole
            text = text.replace(tail, MASK)
    return text
