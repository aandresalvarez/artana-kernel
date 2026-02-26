from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import cast

from pydantic import BaseModel

from artana.ports.model_adapter_helpers import (
    extract_output_json,
    extract_output_json_from_responses,
    extract_response_id,
    extract_responses_output_items,
    extract_tool_calls,
    extract_tool_calls_from_responses,
    extract_usage,
    has_tokens,
    is_responses_unsupported_exception,
    is_transient_exception,
    normalize_response,
    serialize_messages,
    serialize_messages_for_responses,
    serialize_tools,
    serialize_tools_for_responses,
)
from artana.ports.model_types import (
    LiteLLMCompletionFn,
    LiteLLMResponsesFn,
    ModelAPIModeUsed,
    ModelPermanentError,
    ModelRequest,
    ModelResult,
    ModelTimeoutError,
    ModelTransientError,
    OutputT,
)


class _ResponsesUnsupported(RuntimeError):
    pass


class LiteLLMAdapter:
    _RESPONSES_PREFIXES: frozenset[str] = frozenset(
        {"openai", "azure", "anthropic", "vertex", "vertex_ai"}
    )

    def __init__(
        self,
        completion_fn: LiteLLMCompletionFn | None = None,
        responses_fn: LiteLLMResponsesFn | None = None,
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
        else:
            from litellm import acompletion

            self._completion_fn = cast(LiteLLMCompletionFn, acompletion)

        if responses_fn is not None:
            self._responses_fn = responses_fn
        else:
            from litellm import aresponses

            self._responses_fn = cast(LiteLLMResponsesFn, aresponses)

    async def complete(self, request: ModelRequest[OutputT]) -> ModelResult[OutputT]:
        tools_payload = serialize_tools(request.allowed_tools)
        tools_payload_responses = serialize_tools_for_responses(request.allowed_tools)
        messages_payload = serialize_messages(
            request.messages, fallback_prompt=request.prompt
        )
        responses_input = serialize_messages_for_responses(
            request.messages, fallback_prompt=request.prompt
        )
        selected_mode = self._select_mode(request.model, request.model_options.api_mode)
        response_dict: Mapping[str, object]
        api_mode_used: ModelAPIModeUsed

        if selected_mode == "responses":
            try:
                response_dict = await self._call_responses_with_retry(
                    model=request.model,
                    input_payload=responses_input,
                    response_format=request.output_schema,
                    tools_payload=tools_payload_responses if tools_payload_responses else None,
                    previous_response_id=request.model_options.previous_response_id,
                    reasoning_effort=request.model_options.reasoning_effort,
                    verbosity=request.model_options.verbosity,
                )
                api_mode_used = "responses"
            except _ResponsesUnsupported as exc:
                if request.model_options.api_mode != "auto":
                    raise ModelPermanentError(
                        f"LiteLLM responses unsupported for model {request.model!r}: {exc}"
                    ) from exc
                response_dict = await self._call_chat_with_retry(
                    model=request.model,
                    messages_payload=messages_payload,
                    response_format=request.output_schema,
                    tools_payload=tools_payload if tools_payload else None,
                )
                api_mode_used = "chat"
        else:
            response_dict = await self._call_chat_with_retry(
                model=request.model,
                messages_payload=messages_payload,
                response_format=request.output_schema,
                tools_payload=tools_payload if tools_payload else None,
            )
            api_mode_used = "chat"

        if api_mode_used == "responses":
            raw_output = extract_output_json_from_responses(response_dict)
            tool_calls = extract_tool_calls_from_responses(response_dict)
            response_id = extract_response_id(response_dict)
            response_output_items = extract_responses_output_items(response_dict)
        else:
            raw_output = extract_output_json(response_dict)
            tool_calls = extract_tool_calls(response_dict)
            response_id = None
            response_output_items = ()

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
            api_mode_used=api_mode_used,
            response_id=response_id,
            response_output_items=response_output_items,
        )

    async def _call_chat_with_retry(
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

    async def _call_responses_with_retry(
        self,
        *,
        model: str,
        input_payload: list[dict[str, object]],
        response_format: type[BaseModel],
        tools_payload: list[dict[str, object]] | None,
        previous_response_id: str | None,
        reasoning_effort: str | None,
        verbosity: str | None,
    ) -> Mapping[str, object]:
        reasoning_payload: dict[str, object] | None = None
        if reasoning_effort is not None:
            reasoning_payload = {"effort": reasoning_effort}
        text_payload: dict[str, object] | None = None
        if verbosity is not None:
            text_payload = {"verbosity": verbosity}

        attempt = 0
        while True:
            try:
                response_obj = await asyncio.wait_for(
                    self._responses_fn(
                        model=model,
                        input=input_payload,
                        previous_response_id=previous_response_id,
                        reasoning=reasoning_payload,
                        text=text_payload,
                        text_format=response_format,
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
                if is_responses_unsupported_exception(exc):
                    raise _ResponsesUnsupported(str(exc)) from exc
                if is_transient_exception(exc):
                    if attempt >= self._max_retries:
                        raise ModelTransientError(
                            f"LiteLLM transient failure after {attempt + 1} attempts: {exc}"
                        ) from exc
                    await asyncio.sleep(self._retry_backoff(attempt))
                    attempt += 1
                    continue
                raise ModelPermanentError(f"LiteLLM permanent failure: {exc}") from exc

    def _select_mode(self, model: str, requested_mode: str) -> str:
        if requested_mode == "chat":
            return "chat"
        if requested_mode == "responses":
            return "responses"
        if requested_mode != "auto":
            return "chat"
        return "responses" if self._supports_responses(model) else "chat"

    def _supports_responses(self, model: str) -> bool:
        if "/" in model:
            prefix, _sep, _name = model.partition("/")
            return prefix.lower() in self._RESPONSES_PREFIXES
        return False

    def _retry_backoff(self, attempt: int) -> float:
        backoff = float(self._initial_backoff_seconds * (2**attempt))
        if backoff > self._max_backoff_seconds:
            return float(self._max_backoff_seconds)
        return float(backoff)
