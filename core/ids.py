"""ULID generation.

Primary keys are ULIDs rather than UUIDv4 because almost every query in this
system is "the most recent N", and a sortable key makes that an index scan
instead of a sort. See docs/03-data-model.md §1.

Format: 26 characters of Crockford base32 encoding
  48 bits  millisecond timestamp
  80 bits  randomness

Implemented here rather than pulled from a package: it is twenty lines, it
removes a dependency, and the encoding is worth being able to read.
"""

from __future__ import annotations

import secrets
import time

# Crockford base32: no I, L, O or U, so a ULID cannot be misread aloud or
# mistyped into one of those lookalikes.
_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
_ENCODED_LENGTH = 26


def new_id(*, timestamp_ms: int | None = None) -> str:
    """Return a new ULID.

    Sortable across milliseconds. Within a single millisecond the ordering is
    random, which is fine for our purposes — nothing depends on sub-millisecond
    ordering, and `turns` carries an explicit `seq` where order matters.
    """
    ts = timestamp_ms if timestamp_ms is not None else int(time.time() * 1000)
    value = (ts << 80) | secrets.randbits(80)

    chars = []
    for _ in range(_ENCODED_LENGTH):
        chars.append(_ALPHABET[value & 0x1F])
        value >>= 5
    return "".join(reversed(chars))


def timestamp_of(ulid: str) -> int:
    """Extract the millisecond timestamp from a ULID. Useful when debugging."""
    if len(ulid) != _ENCODED_LENGTH:
        raise ValueError(f"not a ULID: {ulid!r}")
    value = 0
    for char in ulid:
        index = _ALPHABET.find(char.upper())
        if index < 0:
            raise ValueError(f"invalid ULID character {char!r} in {ulid!r}")
        value = (value << 5) | index
    return value >> 80
