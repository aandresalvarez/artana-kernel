from __future__ import annotations

import json
from typing import Final

# Replay-critical canonicalization profile. Bump only with an explicit migration.
CANONICAL_JSON_PROFILE: Final[str] = "v1"


def canonical_json_dumps(value: object) -> str:
    """Serialize JSON using a stable canonical form.

    This format is used for replay-sensitive hashing and comparisons.
    """
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def canonicalize_json_object(value: str) -> str:
    """Canonicalize a JSON object string.

    Raises ValueError when the input is valid JSON but not an object.
    """
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("Expected a JSON object.")
    return canonical_json_dumps(parsed)


def canonicalize_json_object_or_original(value: str) -> str:
    """Canonicalize a JSON object string, or return the original input.

    This is used when we need best-effort normalization without rejecting legacy
    malformed/non-object payloads during replay matching.
    """
    try:
        return canonicalize_json_object(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return value


__all__ = [
    "CANONICAL_JSON_PROFILE",
    "canonical_json_dumps",
    "canonicalize_json_object",
    "canonicalize_json_object_or_original",
]
