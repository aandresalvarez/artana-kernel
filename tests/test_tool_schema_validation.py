from __future__ import annotations

import json
from decimal import Decimal
from enum import Enum
from typing import Optional

import pytest
from pydantic import BaseModel

from artana.ports.tool import LocalToolRegistry, ToolExecutionContext


class Mode(str, Enum):
    SAFE = "safe"
    FAST = "fast"


class NestedConfig(BaseModel):
    retries: int
    tags: list[str]


def _resolve_ref(schema: dict[str, object], maybe_ref: dict[str, object]) -> dict[str, object]:
    ref = maybe_ref.get("$ref")
    if not isinstance(ref, str):
        return maybe_ref
    key = ref.rsplit("/", maxsplit=1)[-1]
    defs = schema.get("$defs")
    if not isinstance(defs, dict):
        return maybe_ref
    resolved = defs.get(key)
    if not isinstance(resolved, dict):
        return maybe_ref
    return resolved


def _has_type(schema_obj: dict[str, object], expected_type: str) -> bool:
    direct_type = schema_obj.get("type")
    if direct_type == expected_type:
        return True
    any_of = schema_obj.get("anyOf")
    if not isinstance(any_of, list):
        return False
    for candidate in any_of:
        if isinstance(candidate, dict) and candidate.get("type") == expected_type:
            return True
    return False


def test_tool_schema_generation_reflects_annotations() -> None:
    registry = LocalToolRegistry()

    async def typed_tool(
        count: int,
        ratio: float,
        enabled: bool,
        amount: Decimal,
        mode: Mode,
        note: Optional[int] = None,
        config: NestedConfig | None = None,
        values: list[int] | None = None,
    ) -> str:
        return '{"ok":true}'

    registry.register(typed_tool)
    definitions = registry.to_all_tool_definitions()
    assert len(definitions) == 1

    schema_obj = json.loads(definitions[0].arguments_schema_json)
    assert isinstance(schema_obj, dict)
    properties = schema_obj.get("properties")
    assert isinstance(properties, dict)

    count_schema = properties.get("count")
    ratio_schema = properties.get("ratio")
    enabled_schema = properties.get("enabled")
    amount_schema = properties.get("amount")
    mode_schema = properties.get("mode")
    note_schema = properties.get("note")
    config_schema = properties.get("config")
    values_schema = properties.get("values")

    assert isinstance(count_schema, dict)
    assert isinstance(ratio_schema, dict)
    assert isinstance(enabled_schema, dict)
    assert isinstance(amount_schema, dict)
    assert isinstance(mode_schema, dict)
    assert isinstance(note_schema, dict)
    assert isinstance(config_schema, dict)
    assert isinstance(values_schema, dict)

    assert _has_type(count_schema, "integer")
    assert _has_type(ratio_schema, "number")
    assert _has_type(enabled_schema, "boolean")
    assert _has_type(amount_schema, "number")
    assert _has_type(note_schema, "integer")
    assert _has_type(note_schema, "null")
    assert _has_type(config_schema, "null")
    assert _has_type(values_schema, "array")
    assert _has_type(values_schema, "null")

    config_any_of = config_schema.get("anyOf")
    assert isinstance(config_any_of, list)
    config_obj_candidate = next(
        (
            candidate
            for candidate in config_any_of
            if isinstance(candidate, dict) and candidate.get("type") != "null"
        ),
        None,
    )
    assert isinstance(config_obj_candidate, dict)
    resolved_config_schema = _resolve_ref(schema_obj, config_obj_candidate)
    assert _has_type(resolved_config_schema, "object")

    resolved_mode_schema = _resolve_ref(schema_obj, mode_schema)
    assert _has_type(resolved_mode_schema, "string")
    enum_values = resolved_mode_schema.get("enum")
    assert enum_values == ["safe", "fast"]

    required = schema_obj.get("required")
    assert isinstance(required, list)
    assert set(required) == {"count", "ratio", "enabled", "amount", "mode"}
    assert "note" not in required


@pytest.mark.asyncio
async def test_tool_argument_validation_rejects_invalid_payloads() -> None:
    registry = LocalToolRegistry()
    calls = 0

    async def validate_me(count: int, enabled: bool) -> str:
        nonlocal calls
        calls += 1
        return '{"ok":true}'

    registry.register(validate_me)
    context = ToolExecutionContext(
        run_id="run_tool_validation",
        tenant_id="org_validation",
        idempotency_key="idemp_validation",
        request_event_id=None,
        tool_version="1.0.0",
        schema_version="1",
    )

    with pytest.raises(ValueError, match="Field required"):
        await registry.call(
            "validate_me",
            '{"enabled":true}',
            context=context,
        )
    with pytest.raises(ValueError, match="extra_forbidden"):
        await registry.call(
            "validate_me",
            '{"count":1,"enabled":true,"unexpected":1}',
            context=context,
        )
    with pytest.raises(ValueError, match="valid integer"):
        await registry.call(
            "validate_me",
            '{"count":"1","enabled":true}',
            context=context,
        )

    assert calls == 0
    success = await registry.call(
        "validate_me",
        '{"count":1,"enabled":true}',
        context=context,
    )
    assert success.outcome == "success"
    assert calls == 1
