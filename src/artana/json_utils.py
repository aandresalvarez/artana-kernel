from __future__ import annotations

import hashlib

from artana.canonicalization import (
    CANONICAL_JSON_PROFILE,
    canonical_json_dumps,
    canonicalize_json_object,
)


def sha256_hex(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


__all__ = [
    "CANONICAL_JSON_PROFILE",
    "canonical_json_dumps",
    "canonicalize_json_object",
    "sha256_hex",
]
