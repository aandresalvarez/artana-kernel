from __future__ import annotations

from dataclasses import asdict
from typing import Any, TypeVar

import json
from pydantic import ConfigDict
from pydantic import BaseModel, create_model

from artana.events import ChatMessage
from artana.kernel import ArtanaKernel, ModelInput, StepModelResult
from artana.models import TenantContext

OutputT = TypeVar("OutputT", bound=BaseModel)


class KernelModelClient:
    def __init__(self, *, kernel: ArtanaKernel) -> None:
        self._kernel = kernel

    async def chat(
        self,
        *,
        run_id: str,
        tenant: TenantContext,
        model: str,
        prompt: str,
        output_schema: type[OutputT],
        step_key: str | None = None,
    ) -> StepModelResult[OutputT]:
        try:
            await self._kernel.load_run(run_id=run_id)
        except ValueError:
            await self._kernel.start_run(tenant=tenant, run_id=run_id)
        return await self._kernel.step_model(
            run_id=run_id,
            tenant=tenant,
            model=model,
            input=ModelInput.from_prompt(prompt),
            output_schema=output_schema,
            step_key=step_key,
        )


class AutonomousAgent:
    """
    The Agent Runtime.
    Wraps ArtanaKernel with a durable Model -> Tool -> Model orchestration loop.
    """

    def __init__(self, *, kernel: ArtanaKernel) -> None:
        self._kernel = kernel

    async def run(
        self,
        *,
        run_id: str,
        tenant: TenantContext,
        model: str,
        system_prompt: str = "You are a helpful autonomous agent.",
        prompt: str,
        output_schema: type[OutputT],
        max_iterations: int = 15,
    ) -> OutputT:
        try:
            await self._kernel.load_run(run_id=run_id)
            messages = [
                ChatMessage(role="system", content=system_prompt),
                ChatMessage(role="user", content=prompt),
            ]
        except ValueError:
            await self._kernel.start_run(tenant=tenant, run_id=run_id)
            messages = [
                ChatMessage(role="system", content=system_prompt),
                ChatMessage(role="user", content=prompt),
            ]

        if max_iterations <= 0:
            raise ValueError("max_iterations must be >= 1.")

        iteration = 0
        while iteration < max_iterations:
            iteration += 1

            model_result = await self._kernel.step_model(
                run_id=run_id,
                tenant=tenant,
                model=model,
                input=ModelInput.from_messages(messages),
                output_schema=output_schema,
                step_key=f"turn_{iteration}_model",
            )

            if not model_result.tool_calls:
                return model_result.output

            tool_calls_str = json.dumps([asdict(tool_call) for tool_call in model_result.tool_calls])
            messages.append(
                ChatMessage(role="assistant", content=f"Action requested: {tool_calls_str}")
            )

            for tool_call in model_result.tool_calls:
                arguments = _model_from_tool_arguments_json(tool_call.arguments_json)
                tool_result = await self._kernel.step_tool(
                    run_id=run_id,
                    tenant=tenant,
                    tool_name=tool_call.tool_name,
                    arguments=arguments,
                    step_key=f"turn_{iteration}_tool_{tool_call.tool_name}",
                )
                messages.append(
                    ChatMessage(
                        role="tool",
                        content=f"Result from {tool_call.tool_name}: {tool_result.result_json}",
                    )
                )

        raise RuntimeError(
            f"Agent exceeded max iterations ({max_iterations}) without reaching an answer."
        )

ChatClient = KernelModelClient


def _model_from_tool_arguments_json(arguments_json: str) -> BaseModel:
    try:
        arguments = json.loads(arguments_json)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Tool arguments must be valid JSON, got {arguments_json!r}."
        ) from exc

    if not isinstance(arguments, dict):
        raise ValueError(
            f"Tool arguments JSON must be an object, got {type(arguments)!r}."
        )

    fields: dict[str, tuple[Any, ...]] = {}
    for name in arguments:
        fields[name] = (Any, ...)

    ToolArguments = create_model(
        "ToolArguments",
        __config__=ConfigDict(extra="forbid"),
        **fields,
    )
    return ToolArguments(**arguments)
