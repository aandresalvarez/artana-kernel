from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar

from pydantic import BaseModel

from artana.events import ChatMessage
from artana.kernel import ArtanaKernel, ModelInput, StepModelResult
from artana.models import TenantContext
from artana.ports.model import ModelCallOptions

OutputT = TypeVar("OutputT", bound=BaseModel)


@dataclass(frozen=True, slots=True)
class AgentRuntimeState:
    messages: tuple[ChatMessage, ...]
    turn_index: int = 0


@dataclass(frozen=True, slots=True)
class AgentRuntimeResult(Generic[OutputT]):
    run_id: str
    state: AgentRuntimeState
    last_step: StepModelResult[OutputT]


class AgentRuntime:
    """Agent loop layer on top of ArtanaKernel execution primitives."""

    def __init__(self, *, kernel: ArtanaKernel) -> None:
        self._kernel = kernel

    async def run_turn(
        self,
        *,
        run_id: str,
        tenant: TenantContext,
        model: str,
        output_schema: type[OutputT],
        state: AgentRuntimeState,
        step_key: str | None = None,
        model_options: ModelCallOptions | None = None,
    ) -> AgentRuntimeResult[OutputT]:
        if len(state.messages) == 0:
            raise ValueError("AgentRuntimeState.messages cannot be empty.")

        turn_key = (
            step_key if step_key is not None else f"agent_turn_{state.turn_index}"
        )
        step = await self._kernel.step_model(
            run_id=run_id,
            tenant=tenant,
            model=model,
            input=ModelInput.from_messages(state.messages),
            output_schema=output_schema,
            step_key=turn_key,
            model_options=model_options,
        )
        assistant_message = ChatMessage(
            role="assistant",
            content=step.output.model_dump_json(),
        )
        next_state = AgentRuntimeState(
            messages=state.messages + (assistant_message,),
            turn_index=state.turn_index + 1,
        )
        return AgentRuntimeResult(run_id=run_id, state=next_state, last_step=step)

    async def run_until(
        self,
        *,
        run_id: str,
        tenant: TenantContext,
        model: str,
        output_schema: type[OutputT],
        initial_messages: Sequence[ChatMessage],
        should_continue: Callable[[AgentRuntimeResult[OutputT]], bool],
        max_turns: int = 4,
        step_key_prefix: str = "agent_turn",
        model_options: ModelCallOptions | None = None,
    ) -> AgentRuntimeResult[OutputT]:
        if max_turns <= 0:
            raise ValueError("max_turns must be >= 1.")

        current_state = AgentRuntimeState(messages=tuple(initial_messages), turn_index=0)
        last_result: AgentRuntimeResult[OutputT] | None = None
        for _ in range(max_turns):
            result = await self.run_turn(
                run_id=run_id,
                tenant=tenant,
                model=model,
                output_schema=output_schema,
                state=current_state,
                step_key=f"{step_key_prefix}_{current_state.turn_index}",
                model_options=model_options,
            )
            last_result = result
            current_state = result.state
            if not should_continue(result):
                break

        if last_result is None:
            raise RuntimeError("Agent runtime did not execute any turns.")
        return last_result
