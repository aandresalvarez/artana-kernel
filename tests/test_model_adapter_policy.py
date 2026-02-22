from __future__ import annotations

import asyncio
from typing import TypeVar

import pytest
from pydantic import BaseModel

from artana.ports.model import (
    LiteLLMAdapter,
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


@pytest.mark.asyncio
async def test_litellm_adapter_retries_transient_error_then_succeeds() -> None:
    calls = 0

    async def completion_fn(
        *,
        model: str,
        messages: list[dict[str, str]],
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
        messages: list[dict[str, str]],
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
        messages: list[dict[str, str]],
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
        messages: list[dict[str, str]],
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
        output_schema=Decision,
        allowed_tools=(),
    )

    with pytest.raises(ModelPermanentError, match="cost is unknown"):
        await adapter.complete(request)
