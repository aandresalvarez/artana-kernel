from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TypeVar, cast

from pydantic import BaseModel

from artana.ports.model import (
    ModelAPIModeUsed,
    ModelRequest,
    ModelResult,
    ModelUsage,
    ToolCall,
)

OutputT = TypeVar("OutputT", bound=BaseModel)


class MockModelPort:
    def __init__(
        self,
        *,
        output: Mapping[str, object] | None = None,
        output_factory: Callable[[ModelRequest[BaseModel]], Mapping[str, object]] | None = None,
        usage: ModelUsage | None = None,
        tool_calls: tuple[ToolCall, ...] = (),
        api_mode_used: ModelAPIModeUsed = "chat",
        response_id: str | None = None,
        response_output_items: tuple[dict[str, object], ...] = (),
    ) -> None:
        if output is None and output_factory is None:
            raise ValueError("MockModelPort requires output or output_factory.")
        self._output = dict(output) if output is not None else None
        self._output_factory = output_factory
        self._usage = usage or ModelUsage(
            prompt_tokens=1,
            completion_tokens=1,
            cost_usd=0.0,
        )
        self._tool_calls = tool_calls
        self._api_mode_used = api_mode_used
        self._response_id = response_id
        self._response_output_items = response_output_items
        self.calls = 0

    async def complete(self, request: ModelRequest[OutputT]) -> ModelResult[OutputT]:
        self.calls += 1
        payload = self._resolve_output(request)
        output = request.output_schema.model_validate(payload)
        return ModelResult(
            output=output,
            usage=self._usage,
            tool_calls=self._tool_calls,
            api_mode_used=self._api_mode_used,
            response_id=self._response_id,
            response_output_items=self._response_output_items,
        )

    def _resolve_output(self, request: ModelRequest[OutputT]) -> dict[str, object]:
        if self._output_factory is not None:
            generated = self._output_factory(cast(ModelRequest[BaseModel], request))
            return dict(generated)
        if self._output is None:
            raise RuntimeError("MockModelPort is missing configured output.")
        return dict(self._output)


__all__ = ["MockModelPort"]
