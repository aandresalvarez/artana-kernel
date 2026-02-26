from __future__ import annotations

import asyncio
from typing import TypeVar

import pytest
from pydantic import BaseModel

from artana.events import ChatMessage, ToolCallMessage, ToolFunctionCall
from artana.ports.model import (
    LiteLLMAdapter,
    ModelCallOptions,
    ModelPermanentError,
    ModelRequest,
    ModelTimeoutError,
)

OutputModelT = TypeVar("OutputModelT", bound=BaseModel)


class Decision(BaseModel):
    approved: bool
    reason: str


class _TransientLiteLLMError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"transient status {status_code}")
        self.status_code = status_code


class _PermanentLiteLLMError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"permanent status {status_code}")
        self.status_code = status_code


class _ResponsesUnsupportedLiteLLMError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__("unsupported endpoint: /v1/responses")
        self.status_code = status_code


@pytest.mark.asyncio
async def test_litellm_adapter_retries_transient_error_then_succeeds() -> None:
    calls = 0

    async def completion_fn(
        *,
        model: str,
        messages: list[dict[str, object]],
        response_format: type[BaseModel],
        tools: list[dict[str, object]] | None = None,
    ) -> object:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise _TransientLiteLLMError(status_code=429)
        return {
            "choices": [{"message": {"content": '{"approved": true, "reason": "ok"}'}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            "_response_cost": 0.001,
        }

    adapter = LiteLLMAdapter(
        completion_fn=completion_fn,
        timeout_seconds=1.0,
        max_retries=2,
        initial_backoff_seconds=0.001,
        max_backoff_seconds=0.001,
    )
    request = ModelRequest(
        run_id="run_model_retry",
        model="gpt-4o-mini",
        prompt="hello",
        messages=(ChatMessage(role="user", content="hello"),),
        output_schema=Decision,
        allowed_tools=(),
    )

    result = await adapter.complete(request)
    assert calls == 2
    assert result.output.approved is True


@pytest.mark.asyncio
async def test_litellm_adapter_raises_timeout_error_after_retries() -> None:
    calls = 0

    async def completion_fn(
        *,
        model: str,
        messages: list[dict[str, object]],
        response_format: type[BaseModel],
        tools: list[dict[str, object]] | None = None,
    ) -> object:
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.05)
        return {
            "choices": [{"message": {"content": '{"approved": true, "reason": "ok"}'}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            "_response_cost": 0.001,
        }

    adapter = LiteLLMAdapter(
        completion_fn=completion_fn,
        timeout_seconds=0.001,
        max_retries=1,
        initial_backoff_seconds=0.001,
        max_backoff_seconds=0.001,
    )
    request = ModelRequest(
        run_id="run_model_timeout",
        model="gpt-4o-mini",
        prompt="hello",
        messages=(ChatMessage(role="user", content="hello"),),
        output_schema=Decision,
        allowed_tools=(),
    )

    with pytest.raises(ModelTimeoutError):
        await adapter.complete(request)
    assert calls == 2


@pytest.mark.asyncio
async def test_litellm_adapter_raises_permanent_error_without_retry() -> None:
    calls = 0

    async def completion_fn(
        *,
        model: str,
        messages: list[dict[str, object]],
        response_format: type[BaseModel],
        tools: list[dict[str, object]] | None = None,
    ) -> object:
        nonlocal calls
        calls += 1
        raise _PermanentLiteLLMError(status_code=400)

    adapter = LiteLLMAdapter(
        completion_fn=completion_fn,
        timeout_seconds=1.0,
        max_retries=2,
        initial_backoff_seconds=0.001,
        max_backoff_seconds=0.001,
    )
    request = ModelRequest(
        run_id="run_model_permanent",
        model="gpt-4o-mini",
        prompt="hello",
        messages=(ChatMessage(role="user", content="hello"),),
        output_schema=Decision,
        allowed_tools=(),
    )

    with pytest.raises(ModelPermanentError):
        await adapter.complete(request)
    assert calls == 1


@pytest.mark.asyncio
async def test_litellm_adapter_fails_when_cost_unknown_in_strict_mode() -> None:
    async def completion_fn(
        *,
        model: str,
        messages: list[dict[str, object]],
        response_format: type[BaseModel],
        tools: list[dict[str, object]] | None = None,
    ) -> object:
        return {
            "choices": [{"message": {"content": '{"approved": true, "reason": "ok"}'}}],
            "usage": {"prompt_tokens": 4, "completion_tokens": 2},
        }

    adapter = LiteLLMAdapter(
        completion_fn=completion_fn,
        timeout_seconds=1.0,
        max_retries=0,
        fail_on_unknown_cost=True,
    )
    request = ModelRequest(
        run_id="run_model_cost_unknown",
        model="gpt-4o-mini",
        prompt="hello",
        messages=(ChatMessage(role="user", content="hello"),),
        output_schema=Decision,
        allowed_tools=(),
    )

    with pytest.raises(ModelPermanentError, match="cost is unknown"):
        await adapter.complete(request)


@pytest.mark.asyncio
async def test_litellm_adapter_sends_full_message_history() -> None:
    captured_messages: list[list[dict[str, object]]] = []

    async def completion_fn(
        *,
        model: str,
        messages: list[dict[str, object]],
        response_format: type[BaseModel],
        tools: list[dict[str, object]] | None = None,
    ) -> object:
        captured_messages.append(messages)
        return {
            "choices": [{"message": {"content": '{"approved": true, "reason": "ok"}'}}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 1},
            "_response_cost": 0.001,
        }

    adapter = LiteLLMAdapter(
        completion_fn=completion_fn,
        timeout_seconds=1.0,
        max_retries=0,
    )
    request = ModelRequest(
        run_id="run_model_messages",
        model="gpt-4o-mini",
        prompt="latest user turn",
        messages=(
            ChatMessage(role="system", content="You are concise."),
            ChatMessage(role="user", content="Summarize the run."),
            ChatMessage(role="assistant", content="Need one more input."),
        ),
        output_schema=Decision,
        allowed_tools=(),
    )

    result = await adapter.complete(request)
    assert result.output.approved is True
    assert captured_messages == [
        [
            {"role": "system", "content": "You are concise."},
            {"role": "user", "content": "Summarize the run."},
            {"role": "assistant", "content": "Need one more input."},
        ]
    ]


@pytest.mark.asyncio
async def test_litellm_adapter_serializes_tool_protocol_messages() -> None:
    captured_messages: list[list[dict[str, object]]] = []

    async def completion_fn(
        *,
        model: str,
        messages: list[dict[str, object]],
        response_format: type[BaseModel],
        tools: list[dict[str, object]] | None = None,
    ) -> object:
        captured_messages.append(messages)
        return {
            "choices": [{"message": {"content": '{"approved": true, "reason": "ok"}'}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1},
            "_response_cost": 0.001,
        }

    adapter = LiteLLMAdapter(
        completion_fn=completion_fn,
        timeout_seconds=1.0,
        max_retries=0,
    )
    request = ModelRequest(
        run_id="run_model_tool_messages",
        model="gpt-4o-mini",
        prompt="ignored",
        messages=(
            ChatMessage(
                role="assistant",
                content="",
                tool_calls=[
                    ToolCallMessage(
                        id="call_abc",
                        function=ToolFunctionCall(
                            name="lookup_weather",
                            arguments='{"city":"NYC"}',
                        ),
                    )
                ],
            ),
            ChatMessage(
                role="tool",
                content='{"temp_c":21}',
                tool_call_id="call_abc",
                name="lookup_weather",
            ),
        ),
        output_schema=Decision,
        allowed_tools=(),
    )

    result = await adapter.complete(request)
    assert result.output.approved is True
    assert captured_messages == [
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_abc",
                        "type": "function",
                        "function": {
                            "name": "lookup_weather",
                            "arguments": '{"city":"NYC"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "content": '{"temp_c":21}',
                "tool_call_id": "call_abc",
                "name": "lookup_weather",
            },
        ]
    ]


@pytest.mark.asyncio
async def test_litellm_adapter_extracts_tool_call_ids() -> None:
    async def completion_fn(
        *,
        model: str,
        messages: list[dict[str, object]],
        response_format: type[BaseModel],
        tools: list[dict[str, object]] | None = None,
    ) -> object:
        return {
            "choices": [
                {
                    "message": {
                        "content": '{"approved": false, "reason": "needs tool"}',
                        "tool_calls": [
                            {
                                "id": "call_xyz",
                                "type": "function",
                                "function": {
                                    "name": "lookup_weather",
                                    "arguments": '{"city":"SF"}',
                                },
                            }
                        ],
                    }
                }
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2},
            "_response_cost": 0.001,
        }

    adapter = LiteLLMAdapter(
        completion_fn=completion_fn,
        timeout_seconds=1.0,
        max_retries=0,
    )
    request = ModelRequest(
        run_id="run_model_tool_call_id",
        model="gpt-4o-mini",
        prompt="hello",
        messages=(ChatMessage(role="user", content="hello"),),
        output_schema=Decision,
        allowed_tools=(),
    )

    result = await adapter.complete(request)
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].tool_name == "lookup_weather"
    assert result.tool_calls[0].tool_call_id == "call_xyz"


@pytest.mark.asyncio
async def test_litellm_adapter_raises_on_malformed_tool_call_arguments() -> None:
    async def completion_fn(
        *,
        model: str,
        messages: list[dict[str, object]],
        response_format: type[BaseModel],
        tools: list[dict[str, object]] | None = None,
    ) -> object:
        return {
            "choices": [
                {
                    "message": {
                        "content": '{"approved": true, "reason": "ok"}',
                        "tool_calls": [
                            {
                                "id": "call_bad",
                                "type": "function",
                                "function": {
                                    "name": "lookup_weather",
                                    "arguments": '{"city":"SF"',
                                },
                            }
                        ],
                    }
                }
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2},
            "_response_cost": 0.001,
        }

    adapter = LiteLLMAdapter(
        completion_fn=completion_fn,
        timeout_seconds=1.0,
        max_retries=0,
    )
    request = ModelRequest(
        run_id="run_model_bad_tool_args",
        model="gpt-4o-mini",
        prompt="hello",
        messages=(ChatMessage(role="user", content="hello"),),
        output_schema=Decision,
        allowed_tools=(),
    )

    with pytest.raises(ValueError, match="malformed arguments JSON"):
        await adapter.complete(request)


@pytest.mark.asyncio
async def test_litellm_adapter_auto_uses_responses_for_supported_prefix() -> None:
    captured: list[dict[str, object]] = []

    async def completion_fn(
        *,
        model: str,
        messages: list[dict[str, object]],
        response_format: type[BaseModel],
        tools: list[dict[str, object]] | None = None,
    ) -> object:
        raise AssertionError("chat completion should not be used for openai/ in auto mode")

    async def responses_fn(
        *,
        input: str | list[dict[str, object]],
        model: str,
        previous_response_id: str | None = None,
        reasoning: dict[str, object] | None = None,
        text: dict[str, object] | None = None,
        text_format: type[BaseModel] | dict[str, object] | None = None,
        tools: list[dict[str, object]] | None = None,
    ) -> object:
        captured.append(
            {
                "input": input,
                "model": model,
                "previous_response_id": previous_response_id,
                "reasoning": reasoning,
                "text": text,
                "text_format": text_format,
                "tools": tools,
            }
        )
        return {
            "id": "resp_123",
            "output_text": '{"approved": true, "reason": "ok"}',
            "output": [],
            "usage": {"input_tokens": 5, "output_tokens": 3},
            "_response_cost": 0.002,
        }

    adapter = LiteLLMAdapter(
        completion_fn=completion_fn,
        responses_fn=responses_fn,
        timeout_seconds=1.0,
        max_retries=0,
    )
    request = ModelRequest(
        run_id="run_model_responses_auto",
        model="openai/gpt-5.3-codex",
        prompt="hello",
        messages=(ChatMessage(role="user", content="hello"),),
        output_schema=Decision,
        allowed_tools=(),
        model_options=ModelCallOptions(
            api_mode="auto",
            reasoning_effort="high",
            verbosity="low",
            previous_response_id="resp_prev",
        ),
    )

    result = await adapter.complete(request)
    assert result.output.approved is True
    assert result.api_mode_used == "responses"
    assert result.response_id == "resp_123"
    assert len(captured) == 1
    assert captured[0]["previous_response_id"] == "resp_prev"
    assert captured[0]["reasoning"] == {"effort": "high"}
    assert captured[0]["text"] == {"verbosity": "low"}


@pytest.mark.asyncio
async def test_litellm_adapter_auto_falls_back_to_chat_on_responses_unsupported() -> None:
    chat_calls = 0
    responses_calls = 0

    async def completion_fn(
        *,
        model: str,
        messages: list[dict[str, object]],
        response_format: type[BaseModel],
        tools: list[dict[str, object]] | None = None,
    ) -> object:
        nonlocal chat_calls
        chat_calls += 1
        return {
            "choices": [{"message": {"content": '{"approved": true, "reason": "chat"}'}}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 1},
            "_response_cost": 0.001,
        }

    async def responses_fn(
        *,
        input: str | list[dict[str, object]],
        model: str,
        previous_response_id: str | None = None,
        reasoning: dict[str, object] | None = None,
        text: dict[str, object] | None = None,
        text_format: type[BaseModel] | dict[str, object] | None = None,
        tools: list[dict[str, object]] | None = None,
    ) -> object:
        nonlocal responses_calls
        responses_calls += 1
        raise _ResponsesUnsupportedLiteLLMError(status_code=404)

    adapter = LiteLLMAdapter(
        completion_fn=completion_fn,
        responses_fn=responses_fn,
        timeout_seconds=1.0,
        max_retries=0,
    )
    request = ModelRequest(
        run_id="run_model_responses_fallback",
        model="openai/gpt-5.3-codex",
        prompt="hello",
        messages=(ChatMessage(role="user", content="hello"),),
        output_schema=Decision,
        allowed_tools=(),
        model_options=ModelCallOptions(api_mode="auto"),
    )

    result = await adapter.complete(request)
    assert result.api_mode_used == "chat"
    assert result.output.reason == "chat"
    assert responses_calls == 1
    assert chat_calls == 1


@pytest.mark.asyncio
async def test_litellm_adapter_responses_mode_is_strict_on_unsupported() -> None:
    async def completion_fn(
        *,
        model: str,
        messages: list[dict[str, object]],
        response_format: type[BaseModel],
        tools: list[dict[str, object]] | None = None,
    ) -> object:
        raise AssertionError("chat completion should not run in strict responses mode")

    async def responses_fn(
        *,
        input: str | list[dict[str, object]],
        model: str,
        previous_response_id: str | None = None,
        reasoning: dict[str, object] | None = None,
        text: dict[str, object] | None = None,
        text_format: type[BaseModel] | dict[str, object] | None = None,
        tools: list[dict[str, object]] | None = None,
    ) -> object:
        raise _ResponsesUnsupportedLiteLLMError(status_code=404)

    adapter = LiteLLMAdapter(
        completion_fn=completion_fn,
        responses_fn=responses_fn,
        timeout_seconds=1.0,
        max_retries=0,
    )
    request = ModelRequest(
        run_id="run_model_responses_strict",
        model="openai/gpt-5.3-codex",
        prompt="hello",
        messages=(ChatMessage(role="user", content="hello"),),
        output_schema=Decision,
        allowed_tools=(),
        model_options=ModelCallOptions(api_mode="responses"),
    )

    with pytest.raises(ModelPermanentError, match="responses unsupported"):
        await adapter.complete(request)


@pytest.mark.asyncio
async def test_litellm_adapter_extracts_tool_calls_from_responses_output_items() -> None:
    async def completion_fn(
        *,
        model: str,
        messages: list[dict[str, object]],
        response_format: type[BaseModel],
        tools: list[dict[str, object]] | None = None,
    ) -> object:
        raise AssertionError("chat completion should not run for this test")

    async def responses_fn(
        *,
        input: str | list[dict[str, object]],
        model: str,
        previous_response_id: str | None = None,
        reasoning: dict[str, object] | None = None,
        text: dict[str, object] | None = None,
        text_format: type[BaseModel] | dict[str, object] | None = None,
        tools: list[dict[str, object]] | None = None,
    ) -> object:
        return {
            "id": "resp_tool_1",
            "output_text": '{"approved": false, "reason": "needs tool"}',
            "output": [
                {
                    "type": "function_call",
                    "name": "lookup_weather",
                    "arguments": '{"city":"SF"}',
                    "call_id": "call_resp_1",
                }
            ],
            "usage": {"input_tokens": 3, "output_tokens": 2},
            "_response_cost": 0.001,
        }

    adapter = LiteLLMAdapter(
        completion_fn=completion_fn,
        responses_fn=responses_fn,
        timeout_seconds=1.0,
        max_retries=0,
    )
    request = ModelRequest(
        run_id="run_model_responses_tool_call",
        model="openai/gpt-5.3-codex",
        prompt="hello",
        messages=(ChatMessage(role="user", content="hello"),),
        output_schema=Decision,
        allowed_tools=(),
        model_options=ModelCallOptions(api_mode="responses"),
    )

    result = await adapter.complete(request)
    assert result.api_mode_used == "responses"
    assert result.response_id == "resp_tool_1"
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].tool_name == "lookup_weather"
    assert result.tool_calls[0].tool_call_id == "call_resp_1"
    assert len(result.response_output_items) == 1
