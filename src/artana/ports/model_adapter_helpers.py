from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import cast

from pydantic import BaseModel

from artana.events import ChatMessage
from artana.json_utils import canonicalize_json_object
from artana.ports.model_types import (
    ModelUsage,
    SupportsModelDump,
    ToolCall,
    ToolDefinition,
)


def serialize_tools(tools: Sequence[ToolDefinition]) -> list[dict[str, object]]:
    serialized: list[dict[str, object]] = []
    for tool in tools:
        schema_obj = json.loads(tool.arguments_schema_json)
        if not isinstance(schema_obj, Mapping):
            raise TypeError(
                f"Tool schema for {tool.name} must be a JSON object, got {type(schema_obj)!r}."
            )
        serialized.append(
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": dict(schema_obj),
                },
            }
        )
    return serialized


def serialize_tools_for_responses(
    tools: Sequence[ToolDefinition],
) -> list[dict[str, object]]:
    serialized: list[dict[str, object]] = []
    for tool in tools:
        schema_obj = json.loads(tool.arguments_schema_json)
        if not isinstance(schema_obj, Mapping):
            raise TypeError(
                f"Tool schema for {tool.name} must be a JSON object, got {type(schema_obj)!r}."
            )
        serialized.append(
            {
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "parameters": dict(schema_obj),
            }
        )
    return serialized


def serialize_messages(
    messages: Sequence[ChatMessage], *, fallback_prompt: str
) -> list[dict[str, object]]:
    if len(messages) == 0:
        return [{"role": "user", "content": fallback_prompt}]
    serialized: list[dict[str, object]] = []
    for message in messages:
        payload: dict[str, object] = {"role": message.role, "content": message.content}
        if message.role == "tool":
            if message.tool_call_id is None or message.name is None:
                raise ValueError(
                    "Tool chat messages require both tool_call_id and name."
                )
        if message.tool_call_id is not None:
            payload["tool_call_id"] = message.tool_call_id
        if message.name is not None:
            payload["name"] = message.name
        if message.tool_calls is not None:
            payload["tool_calls"] = [
                {
                    "id": tool_call.id,
                    "type": tool_call.type,
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                    },
                }
                for tool_call in message.tool_calls
            ]
        serialized.append(payload)
    return serialized


def serialize_messages_for_responses(
    messages: Sequence[ChatMessage], *, fallback_prompt: str
) -> list[dict[str, object]]:
    if len(messages) == 0:
        return [{"role": "user", "content": fallback_prompt}]

    serialized: list[dict[str, object]] = []
    for message in messages:
        if message.role == "tool":
            if message.tool_call_id is None or message.name is None:
                raise ValueError(
                    "Tool chat messages require both tool_call_id and name."
                )
            serialized.append(
                {
                    "type": "function_call_output",
                    "call_id": message.tool_call_id,
                    "output": message.content,
                }
            )
            continue

        if message.role == "assistant" and message.tool_calls:
            if message.content != "":
                serialized.append(
                    {
                        "role": "assistant",
                        "content": message.content,
                    }
                )
            for tool_call in message.tool_calls:
                serialized.append(
                    {
                        "type": "function_call",
                        "call_id": tool_call.id,
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                    }
                )
            continue

        serialized.append({"role": message.role, "content": message.content})
    return serialized


def normalize_response(response_obj: object) -> Mapping[str, object]:
    if isinstance(response_obj, Mapping):
        return response_obj
    if isinstance(response_obj, SupportsModelDump):
        return response_obj.model_dump()
    raise TypeError(f"Unsupported LiteLLM response object: {type(response_obj)!r}.")


def extract_output_json(response: Mapping[str, object]) -> str | None:
    choice = first_choice(response)
    message_obj = choice.get("message")
    if not isinstance(message_obj, Mapping):
        raise ValueError("LiteLLM response missing message object in first choice.")

    parsed_obj = message_obj.get("parsed")
    if isinstance(parsed_obj, BaseModel):
        return parsed_obj.model_dump_json()
    if isinstance(parsed_obj, Mapping):
        return json.dumps(dict(parsed_obj))

    content_obj = message_obj.get("content")
    if isinstance(content_obj, str):
        return content_obj
    if isinstance(content_obj, Sequence):
        for item in content_obj:
            if isinstance(item, Mapping):
                text_obj = item.get("text")
                if isinstance(text_obj, str):
                    return text_obj
    return None


def extract_output_json_from_responses(response: Mapping[str, object]) -> str | None:
    output_text_obj = response.get("output_text")
    if isinstance(output_text_obj, str):
        return output_text_obj

    output_obj = response.get("output")
    if not isinstance(output_obj, Sequence):
        return None

    for item in output_obj:
        if not isinstance(item, Mapping):
            continue

        item_type = item.get("type")
        if item_type == "output_text":
            text_obj = item.get("text")
            if isinstance(text_obj, str):
                return text_obj
            value_obj = item.get("value")
            if isinstance(value_obj, str):
                return value_obj

        if item_type != "message":
            continue

        content_obj = item.get("content")
        if not isinstance(content_obj, Sequence):
            continue
        for part in content_obj:
            if not isinstance(part, Mapping):
                continue
            part_text = part.get("text")
            if isinstance(part_text, str):
                return part_text
            part_value = part.get("value")
            if isinstance(part_value, str):
                return part_value
    return None


def extract_usage(response: Mapping[str, object]) -> ModelUsage:
    usage_obj = response.get("usage")
    if not isinstance(usage_obj, Mapping):
        return ModelUsage(prompt_tokens=0, completion_tokens=0, cost_usd=0.0)

    prompt_tokens = as_int(usage_obj.get("prompt_tokens"))
    if prompt_tokens == 0:
        prompt_tokens = as_int(usage_obj.get("input_tokens"))
    completion_tokens = as_int(usage_obj.get("completion_tokens"))
    if completion_tokens == 0:
        completion_tokens = as_int(usage_obj.get("output_tokens"))
    cost_usd = as_float(response.get("_response_cost"))
    if cost_usd == 0.0:
        cost_usd = as_float(response.get("response_cost"))
    if cost_usd == 0.0:
        computed_cost = compute_litellm_cost(response)
        if computed_cost is not None:
            cost_usd = computed_cost
    return ModelUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cost_usd=cost_usd,
    )


def extract_tool_calls(response: Mapping[str, object]) -> tuple[ToolCall, ...]:
    choice = first_choice(response)
    message_obj = choice.get("message")
    if not isinstance(message_obj, Mapping):
        return ()

    tool_calls_obj = message_obj.get("tool_calls")
    if not isinstance(tool_calls_obj, Sequence):
        return ()

    parsed: list[ToolCall] = []
    for tool_call_obj in tool_calls_obj:
        if not isinstance(tool_call_obj, Mapping):
            continue
        function_obj = tool_call_obj.get("function")
        if not isinstance(function_obj, Mapping):
            continue
        tool_call_id_obj = tool_call_obj.get("id")
        tool_name = function_obj.get("name")
        arguments_json = function_obj.get("arguments")
        if not isinstance(tool_name, str):
            continue
        if not isinstance(arguments_json, str):
            continue
        try:
            canonical_arguments_json = canonicalize_json_object(arguments_json)
        except Exception as exc:
            raise ValueError(
                f"Tool call {tool_name!r} returned malformed arguments JSON."
            ) from exc
        parsed.append(
            ToolCall(
                tool_name=tool_name,
                arguments_json=canonical_arguments_json,
                tool_call_id=tool_call_id_obj
                if isinstance(tool_call_id_obj, str)
                else None,
            )
        )
    return tuple(parsed)


def extract_tool_calls_from_responses(
    response: Mapping[str, object],
) -> tuple[ToolCall, ...]:
    output_obj = response.get("output")
    if not isinstance(output_obj, Sequence):
        return ()

    parsed: list[ToolCall] = []
    for item in output_obj:
        if not isinstance(item, Mapping):
            continue
        item_type = item.get("type")
        if item_type != "function_call":
            continue
        tool_name_obj = item.get("name")
        arguments_obj = item.get("arguments")
        tool_call_id_obj = item.get("call_id")
        if not isinstance(tool_name_obj, str):
            continue
        if isinstance(arguments_obj, str):
            arguments_json = arguments_obj
        elif isinstance(arguments_obj, Mapping):
            arguments_json = json.dumps(dict(arguments_obj))
        else:
            continue
        try:
            canonical_arguments_json = canonicalize_json_object(arguments_json)
        except Exception as exc:
            raise ValueError(
                f"Tool call {tool_name_obj!r} returned malformed arguments JSON."
            ) from exc
        parsed.append(
            ToolCall(
                tool_name=tool_name_obj,
                arguments_json=canonical_arguments_json,
                tool_call_id=tool_call_id_obj
                if isinstance(tool_call_id_obj, str)
                else None,
            )
        )
    return tuple(parsed)


def extract_response_id(response: Mapping[str, object]) -> str | None:
    response_id = response.get("id")
    if isinstance(response_id, str):
        return response_id
    return None


def extract_responses_output_items(
    response: Mapping[str, object],
) -> tuple[dict[str, object], ...]:
    output_obj = response.get("output")
    if not isinstance(output_obj, Sequence):
        return ()

    normalized: list[dict[str, object]] = []
    for item in output_obj:
        if not isinstance(item, Mapping):
            continue
        canonical_item = _canonicalize_mapping(item)
        normalized.append(canonical_item)
    return tuple(normalized)


def is_responses_unsupported_exception(exc: Exception) -> bool:
    status_code = status_code_from_exception(exc)
    message = str(exc).lower()
    unsupported_patterns = (
        "responses api",
        "/v1/responses",
        "unsupported endpoint",
        "does not support responses",
        "not supported for this model",
        "unsupported parameter",
        "unknown parameter",
        "unrecognized request argument",
        "unknown argument",
    )
    if status_code in {404, 405, 501}:
        return True
    if status_code in {400, 422} and any(pattern in message for pattern in unsupported_patterns):
        return True
    return any(pattern in message for pattern in unsupported_patterns)


def is_transient_exception(exc: Exception) -> bool:
    status_code = status_code_from_exception(exc)
    if status_code is not None:
        return status_code in {408, 409, 429, 500, 502, 503, 504}
    message = str(exc).lower()
    return "rate limit" in message or "temporar" in message


def has_tokens(usage: ModelUsage) -> bool:
    return usage.prompt_tokens > 0 or usage.completion_tokens > 0


def first_choice(response: Mapping[str, object]) -> Mapping[str, object]:
    choices_obj = response.get("choices")
    if not isinstance(choices_obj, Sequence) or len(choices_obj) == 0:
        raise ValueError("LiteLLM response does not include choices.")
    first = choices_obj[0]
    if not isinstance(first, Mapping):
        raise ValueError("LiteLLM response first choice must be an object.")
    return first


def as_int(value: object) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return 0


def as_float(value: object) -> float:
    if isinstance(value, int):
        return float(value)
    if isinstance(value, float):
        return value
    return 0.0


def status_code_from_exception(exc: Exception) -> int | None:
    direct = getattr(exc, "status_code", None)
    if isinstance(direct, int):
        return direct
    response = getattr(exc, "response", None)
    if response is None:
        return None
    response_status = getattr(response, "status_code", None)
    if isinstance(response_status, int):
        return response_status
    return None


def compute_litellm_cost(response: Mapping[str, object]) -> float | None:
    try:
        from litellm import completion_cost
    except Exception:
        return None

    try:
        computed = completion_cost(completion_response=dict(response))
    except Exception:
        return None

    if isinstance(computed, int):
        return float(computed)
    if isinstance(computed, float):
        return computed
    return None


def _canonicalize_value(value: object) -> object:
    if isinstance(value, Mapping):
        return _canonicalize_mapping(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_canonicalize_value(item) for item in value]
    return value


def _canonicalize_mapping(value: Mapping[object, object]) -> dict[str, object]:
    normalized = {
        str(key): _canonicalize_value(nested)
        for key, nested in value.items()
    }
    return cast(
        dict[str, object],
        json.loads(json.dumps(normalized, sort_keys=True, separators=(",", ":"))),
    )
