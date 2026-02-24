from __future__ import annotations

import pytest

from artana._kernel.tool_execution import derive_idempotency_key
from artana.canonicalization import (
    CANONICAL_JSON_PROFILE,
    canonical_json_dumps,
    canonicalize_json_object,
)


def test_canonical_json_profile_is_pinned() -> None:
    assert CANONICAL_JSON_PROFILE == "v1"


def test_canonical_json_dumps_is_stable() -> None:
    payload = {
        "b": 1,
        "a": {
            "d": 2,
            "c": [{"y": 0, "x": 1}],
        },
    }
    assert canonical_json_dumps(payload) == '{"a":{"c":[{"x":1,"y":0}],"d":2},"b":1}'


def test_canonicalize_json_object_requires_object() -> None:
    with pytest.raises(ValueError, match="Expected a JSON object"):
        canonicalize_json_object("[1,2,3]")


def test_idempotency_key_is_stable_for_same_sequence() -> None:
    key_one = derive_idempotency_key(
        run_id="run_1",
        tool_name="submit_transfer",
        seq=2,
    )
    key_two = derive_idempotency_key(
        run_id="run_1",
        tool_name="submit_transfer",
        seq=2,
    )

    assert key_one == key_two


def test_idempotency_key_changes_for_different_sequence() -> None:
    key_one = derive_idempotency_key(
        run_id="run_1",
        tool_name="submit_transfer",
        seq=2,
    )
    key_two = derive_idempotency_key(
        run_id="run_1",
        tool_name="submit_transfer",
        seq=3,
    )

    assert key_one != key_two
