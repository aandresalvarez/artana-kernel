from __future__ import annotations

import json

from pydantic import BaseModel, RootModel


class _ToolArgumentsModel(RootModel[dict[str, object]]):
    pass


def model_from_tool_arguments_json(arguments_json: str) -> BaseModel:
    try:
        parsed = json.loads(arguments_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Tool arguments must be valid JSON: {arguments_json!r}.") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"Tool arguments JSON must be an object, got {type(parsed)!r}.")

    return _ToolArgumentsModel(parsed)


__all__ = ["model_from_tool_arguments_json"]
