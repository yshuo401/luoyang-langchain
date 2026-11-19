"""Dependency-free ULID generation for internal request and job IDs."""

from __future__ import annotations

import secrets
import time

_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def _encode(value: int, length: int) -> str:
    chars = ["0"] * length
    for index in range(length - 1, -1, -1):
        chars[index] = _ALPHABET[value & 31]
        value >>= 5
    return "".join(chars)


def new_ulid(timestamp_ms: int | None = None) -> str:
    """Return a 26-character Crockford Base32 ULID."""

    timestamp = int(time.time() * 1000) if timestamp_ms is None else timestamp_ms
    return _encode(timestamp, 10) + _encode(secrets.randbits(80), 16)
