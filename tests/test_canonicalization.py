from __future__ import annotations

import json

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


def test_idempotency_key_is_invariant_to_json_key_order() -> None:
    key_one = derive_idempotency_key(
        run_id="run_1",
        tool_name="submit_transfer",
        arguments_json='{"b":2,"a":1}',
        step_key="transfer",
    )
    key_two = derive_idempotency_key(
        run_id="run_1",
        tool_name="submit_transfer",
        arguments_json='{"a":1,"b":2}',
        step_key="transfer",
    )

    assert key_one == key_two


def test_idempotency_key_rejects_invalid_json_arguments() -> None:
    with pytest.raises((ValueError, json.JSONDecodeError)):
        derive_idempotency_key(
            run_id="run_1",
            tool_name="submit_transfer",
            arguments_json='{"a":1',
            step_key="transfer",
        )
