from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import cast

from pydantic import BaseModel

from artana.ports.model_adapter_helpers import (
    extract_output_json,
    extract_tool_calls,
    extract_usage,
    has_tokens,
    is_transient_exception,
    normalize_response,
    serialize_messages,
    serialize_tools,
)
from artana.ports.model_types import (
    LiteLLMCompletionFn,
    ModelPermanentError,
    ModelRequest,
    ModelResult,
    ModelTimeoutError,
    ModelTransientError,
    OutputT,
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
        tools_payload = serialize_tools(request.allowed_tools)
        response_dict = await self._call_with_retry(
            model=request.model,
            messages_payload=serialize_messages(
                request.messages, fallback_prompt=request.prompt
            ),
            response_format=request.output_schema,
            tools_payload=tools_payload if tools_payload else None,
        )

        raw_output = extract_output_json(response_dict)
        tool_calls = extract_tool_calls(response_dict)
        if raw_output is None:
            if not tool_calls:
                raise ValueError(
                    "Could not extract structured output from LiteLLM response."
                )
            raw_output = "{}"

        output = request.output_schema.model_validate_json(raw_output)
        usage = extract_usage(response_dict)
        if self._fail_on_unknown_cost and has_tokens(usage) and usage.cost_usd <= 0.0:
            raise ModelPermanentError(
                "LiteLLM response cost is unknown for a tokenized response. "
                "Configure model pricing or disable fail_on_unknown_cost."
            )

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
        messages_payload: list[dict[str, object]],
        response_format: type[BaseModel],
        tools_payload: list[dict[str, object]] | None,
    ) -> Mapping[str, object]:
        attempt = 0
        while True:
            try:
                response_obj = await asyncio.wait_for(
                    self._completion_fn(
                        model=model,
                        messages=messages_payload,
                        response_format=response_format,
                        tools=tools_payload,
                    ),
                    timeout=self._timeout_seconds,
                )
                return normalize_response(response_obj)
            except asyncio.TimeoutError as exc:
                if attempt >= self._max_retries:
                    raise ModelTimeoutError(
                        f"LiteLLM timed out after {attempt + 1} attempts."
                    ) from exc
                await asyncio.sleep(self._retry_backoff(attempt))
                attempt += 1
            except Exception as exc:
                if is_transient_exception(exc):
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
