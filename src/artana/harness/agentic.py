from __future__ import annotations

from typing import TypeVar

from pydantic import BaseModel

from artana.acceptance import AcceptanceSpec
from artana.agent.context import ContextBuilder
from artana.agent.loop import DraftVerifyLoopConfig
from artana.agent.memory import MemoryStore
from artana.harness.strong_model import StrongModelHarness
from artana.kernel import ArtanaKernel, ReplayPolicy
from artana.models import TenantContext

OutputT = TypeVar("OutputT", bound=BaseModel)


class StrongModelAgentHarness(StrongModelHarness):
    __test__ = False

    def __init__(
        self,
        *,
        kernel: ArtanaKernel,
        tenant: TenantContext | None = None,
        default_model: str = "gpt-5.4",
        draft_model: str = "gpt-5-mini",
        verify_model: str = "gpt-5.4",
        replay_policy: ReplayPolicy = "allow_prompt_drift",
        context_builder: ContextBuilder | None = None,
        loop: DraftVerifyLoopConfig | None = None,
        acceptance: AcceptanceSpec | None = None,
        memory_store: MemoryStore | None = None,
        auto_reflect: bool = False,
        reflection_model: str = "gpt-5-mini",
        agent_system_prompt: str = "You are a strong-model autonomous agent.",
        max_iterations: int = 8,
    ) -> None:
        super().__init__(
            kernel=kernel,
            tenant=tenant,
            default_model=default_model,
            draft_model=draft_model,
            verify_model=verify_model,
            replay_policy=replay_policy,
        )
        self._agent_context_builder = context_builder
        self._agent_loop = loop
        self._agent_acceptance = acceptance
        self._agent_memory_store = memory_store
        self._agent_auto_reflect = auto_reflect
        self._agent_reflection_model = reflection_model
        self._agent_system_prompt = agent_system_prompt
        self._agent_max_iterations = max_iterations

    async def run_agent(
        self,
        *,
        prompt: str,
        output_schema: type[OutputT],
        run_id: str | None = None,
        system_prompt: str | None = None,
        max_iterations: int | None = None,
        acceptance: AcceptanceSpec | None = None,
        context_builder: ContextBuilder | None = None,
        loop: DraftVerifyLoopConfig | None = None,
        workspace_aware: bool = True,
        ) -> OutputT:
        return await self.run_autonomous_agent(
            run_id=run_id,
            prompt=prompt,
            output_schema=output_schema,
            system_prompt=(
                self._agent_system_prompt if system_prompt is None else system_prompt
            ),
            max_iterations=(
                self._agent_max_iterations if max_iterations is None else max_iterations
            ),
            context_builder=(
                self._agent_context_builder if context_builder is None else context_builder
            ),
            loop=self._agent_loop if loop is None else loop,
            memory_store=self._agent_memory_store,
            acceptance=(
                self._agent_acceptance if acceptance is None else acceptance
            ),
            auto_reflect=self._agent_auto_reflect,
            reflection_model=self._agent_reflection_model,
            workspace_aware=workspace_aware,
        )


__all__ = ["StrongModelAgentHarness"]
