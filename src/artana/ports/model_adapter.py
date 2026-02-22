from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping, Sequence
from typing import cast

from pydantic import BaseModel

from artana.ports.model_types import (
    LiteLLMCompletionFn,
    ModelPermanentError,
    ModelRequest,
    ModelResult,
    ModelTimeoutError,
    ModelTransientError,
    ModelUsage,
    OutputT,
    SupportsModelDump,
    ToolCall,
    ToolDefinition,
)


class LiteLLMAdapter:
    def __init__(
        self,
        completion_fn: LiteLLMCompletionFn | None = None,
        *,
        timeout_seconds: float = 30.0,
        max_retries: int = 2,
        initial_backoff_seconds: float = 0.25,
        max_backoff_seconds: float = 2.0,
        fail_on_unknown_cost: bool = False,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be > 0")
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        if initial_backoff_seconds <= 0:
            raise ValueError("initial_backoff_seconds must be > 0")
        if max_backoff_seconds <= 0:
            raise ValueError("max_backoff_seconds must be > 0")

        self._timeout_seconds = timeout_seconds
        self._max_retries = max_retries
        self._initial_backoff_seconds = initial_backoff_seconds
        self._max_backoff_seconds = max_backoff_seconds
        self._fail_on_unknown_cost = fail_on_unknown_cost

        if completion_fn is not None:
            self._completion_fn = completion_fn
            return

        from litellm import acompletion

        self._completion_fn = cast(LiteLLMCompletionFn, acompletion)

    async def complete(self, request: ModelRequest[OutputT]) -> ModelResult[OutputT]:
        tools_payload = _serialize_tools(request.allowed_tools)
        response_dict = await self._call_with_retry(
            model=request.model,
            prompt=request.prompt,
            response_format=request.output_schema,
            tools_payload=tools_payload if tools_payload else None,
        )

        raw_output = _extract_output_json(response_dict)
        output = request.output_schema.model_validate_json(raw_output)
        usage = _extract_usage(response_dict)
        if self._fail_on_unknown_cost and _has_tokens(usage) and usage.cost_usd <= 0.0:
            raise ModelPermanentError(
                "LiteLLM response cost is unknown for a tokenized response. "
                "Configure model pricing or disable fail_on_unknown_cost."
            )
        tool_calls = _extract_tool_calls(response_dict)

        return ModelResult(
            output=output,
            usage=usage,
            tool_calls=tool_calls,
            raw_output=raw_output,
        )

    async def _call_with_retry(
        self,
        *,
        model: str,
        prompt: str,
        response_format: type[BaseModel],
        tools_payload: list[dict[str, object]] | None,
    ) -> Mapping[str, object]:
        attempt = 0
        while True:
            try:
                response_obj = await asyncio.wait_for(
                    self._completion_fn(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        response_format=response_format,
                        tools=tools_payload,
                    ),
                    timeout=self._timeout_seconds,
                )
                return _normalize_response(response_obj)
            except asyncio.TimeoutError as exc:
                if attempt >= self._max_retries:
                    raise ModelTimeoutError(
                        f"LiteLLM timed out after {attempt + 1} attempts."
                    ) from exc
                await asyncio.sleep(self._retry_backoff(attempt))
                attempt += 1
            except Exception as exc:
                if _is_transient_exception(exc):
                    if attempt >= self._max_retries:
                        raise ModelTransientError(
                            f"LiteLLM transient failure after {attempt + 1} attempts: {exc}"
                        ) from exc
                    await asyncio.sleep(self._retry_backoff(attempt))
                    attempt += 1
                    continue
                raise ModelPermanentError(f"LiteLLM permanent failure: {exc}") from exc

    def _retry_backoff(self, attempt: int) -> float:
        backoff = float(self._initial_backoff_seconds * (2**attempt))
        if backoff > self._max_backoff_seconds:
            return float(self._max_backoff_seconds)
        return float(backoff)


def _serialize_tools(tools: Sequence[ToolDefinition]) -> list[dict[str, object]]:
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


def _normalize_response(response_obj: object) -> Mapping[str, object]:
    if isinstance(response_obj, Mapping):
        return response_obj
    if isinstance(response_obj, SupportsModelDump):
        dumped = response_obj.model_dump()
        return dumped
    raise TypeError(f"Unsupported LiteLLM response object: {type(response_obj)!r}.")


def _extract_output_json(response: Mapping[str, object]) -> str:
    choice = _first_choice(response)
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

    raise ValueError("Could not extract structured output from LiteLLM response.")


def _extract_usage(response: Mapping[str, object]) -> ModelUsage:
    usage_obj = response.get("usage")
    if not isinstance(usage_obj, Mapping):
        return ModelUsage(prompt_tokens=0, completion_tokens=0, cost_usd=0.0)

    prompt_tokens = _as_int(usage_obj.get("prompt_tokens"))
    completion_tokens = _as_int(usage_obj.get("completion_tokens"))
    cost_usd = _as_float(response.get("_response_cost"))

    if cost_usd == 0.0:
        cost_usd = _as_float(response.get("response_cost"))
    if cost_usd == 0.0:
        computed_cost = _compute_litellm_cost(response)
        if computed_cost is not None:
            cost_usd = computed_cost

    return ModelUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cost_usd=cost_usd,
    )


def _extract_tool_calls(response: Mapping[str, object]) -> tuple[ToolCall, ...]:
    choice = _first_choice(response)
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
        tool_name = function_obj.get("name")
        arguments_json = function_obj.get("arguments")
        if not isinstance(tool_name, str):
            continue
        if not isinstance(arguments_json, str):
            continue
        parsed.append(ToolCall(tool_name=tool_name, arguments_json=arguments_json))

    return tuple(parsed)


def _first_choice(response: Mapping[str, object]) -> Mapping[str, object]:
    choices_obj = response.get("choices")
    if not isinstance(choices_obj, Sequence) or len(choices_obj) == 0:
        raise ValueError("LiteLLM response does not include choices.")

    first = choices_obj[0]
    if not isinstance(first, Mapping):
        raise ValueError("LiteLLM response first choice must be an object.")
    return first


def _as_int(value: object) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return 0


def _as_float(value: object) -> float:
    if isinstance(value, int):
        return float(value)
    if isinstance(value, float):
        return value
    return 0.0


def _is_transient_exception(exc: Exception) -> bool:
    status_code = _status_code_from_exception(exc)
    if status_code is not None:
        return status_code in {408, 409, 429, 500, 502, 503, 504}
    message = str(exc).lower()
    return "rate limit" in message or "temporar" in message


def _status_code_from_exception(exc: Exception) -> int | None:
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


def _compute_litellm_cost(response: Mapping[str, object]) -> float | None:
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


def _has_tokens(usage: ModelUsage) -> bool:
    return usage.prompt_tokens > 0 or usage.completion_tokens > 0
