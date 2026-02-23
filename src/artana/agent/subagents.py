from __future__ import annotations

import json
from typing import TypeVar

from pydantic import BaseModel

from artana._kernel.types import ToolCallable
from artana.agent.autonomous import AutonomousAgent
from artana.agent.compaction import CompactionStrategy
from artana.agent.context import ContextBuilder
from artana.agent.memory import MemoryStore
from artana.kernel import ArtanaKernel
from artana.models import TenantContext
from artana.ports.tool import ToolExecutionContext

OutputT = TypeVar("OutputT", bound=BaseModel)


class SubAgentFactory:
    def __init__(
        self,
        *,
        kernel: ArtanaKernel,
        tenant: TenantContext,
        context_builder: ContextBuilder | None = None,
        compaction: CompactionStrategy | None = None,
        memory_store: MemoryStore | None = None,
    ) -> None:
        self._kernel = kernel
        self._tenant = tenant
        self._context_builder = context_builder
        self._compaction = compaction
        self._memory_store = memory_store

    def create(
        self,
        *,
        name: str,
        output_schema: type[OutputT],
        model: str,
        system_prompt: str,
        max_iterations: int = 10,
        requires_capability: str | None = None,
    ) -> ToolCallable:
        if max_iterations <= 0:
            raise ValueError("max_iterations must be >= 1.")

        async def _sub_agent(task: str, artana_context: ToolExecutionContext) -> str:
            child_agent = AutonomousAgent(
                kernel=self._kernel,
                context_builder=self._context_builder,
                compaction=self._compaction,
                memory_store=self._memory_store,
            )
            child_run_id = f"{artana_context.run_id}::sub_agent::{artana_context.idempotency_key}"
            result = await child_agent.run(
                run_id=child_run_id,
                tenant=self._tenant,
                model=model,
                system_prompt=system_prompt,
                prompt=task,
                output_schema=output_schema,
                max_iterations=max_iterations,
            )
            return json.dumps(
                {
                    "run_id": child_run_id,
                    "result": result.model_dump(mode="json"),
                },
                ensure_ascii=False,
            )

        _sub_agent.__name__ = name
        if requires_capability is not None:
            return self._kernel.tool(requires_capability=requires_capability)(_sub_agent)
        return self._kernel.tool()(_sub_agent)


__all__ = ["SubAgentFactory"]
