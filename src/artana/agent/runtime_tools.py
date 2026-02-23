from __future__ import annotations

import json

from artana.agent.memory import MemoryStore
from artana.kernel import ArtanaKernel
from artana.ports.tool import ToolExecutionContext


class RuntimeToolManager:
    def __init__(
        self,
        *,
        kernel: ArtanaKernel,
        memory_store: MemoryStore,
        progressive_skills: bool,
        load_skill_name: str,
        core_memory_append: str,
        core_memory_replace: str,
        core_memory_search: str,
    ) -> None:
        self._kernel = kernel
        self._memory_store = memory_store
        self._progressive_skills = progressive_skills
        self._load_skill_name = load_skill_name
        self._core_memory_append = core_memory_append
        self._core_memory_replace = core_memory_replace
        self._core_memory_search = core_memory_search
        self._registered = False

    def ensure_registered(self) -> None:
        if self._registered:
            return
        self._registered = True

        @self._kernel.tool()
        async def load_skill(skill_name: str) -> str:
            return self._tool_description(skill_name)

        @self._kernel.tool()
        async def core_memory_append(content: str, artana_context: ToolExecutionContext) -> str:
            await self._memory_store.append(run_id=artana_context.run_id, text=content)
            return json.dumps({"status": "appended", "run_id": artana_context.run_id})

        @self._kernel.tool()
        async def core_memory_replace(content: str, artana_context: ToolExecutionContext) -> str:
            await self._memory_store.replace(run_id=artana_context.run_id, content=content)
            return json.dumps({"status": "replaced", "run_id": artana_context.run_id})

        @self._kernel.tool()
        async def core_memory_search(query: str, artana_context: ToolExecutionContext) -> str:
            return await self._memory_store.search(run_id=artana_context.run_id, query=query)

        load_skill.__name__ = self._load_skill_name
        core_memory_append.__name__ = self._core_memory_append
        core_memory_replace.__name__ = self._core_memory_replace
        core_memory_search.__name__ = self._core_memory_search

    def visible_tool_names(self, *, loaded_skills: set[str]) -> set[str] | None:
        if not self._progressive_skills:
            return None
        core_tools = {
            self._load_skill_name,
            self._core_memory_append,
            self._core_memory_replace,
            self._core_memory_search,
        }
        core_tools.update(loaded_skills)
        return {
            tool.name
            for tool in self._kernel._tool_port.to_all_tool_definitions()
            if tool.name in core_tools or tool.name in loaded_skills
        }

    def available_skill_summaries(self) -> dict[str, str]:
        summaries: dict[str, str] = {}
        runtime = {
            self._load_skill_name,
            self._core_memory_append,
            self._core_memory_replace,
            self._core_memory_search,
        }
        for tool in self._kernel._tool_port.to_all_tool_definitions():
            if tool.name in runtime:
                continue
            summaries[tool.name] = tool.description or "no description"
        return summaries

    def _tool_description(self, skill_name: str) -> str:
        tools = {tool.name: tool for tool in self._kernel._tool_port.to_all_tool_definitions()}
        tool = tools.get(skill_name)
        if tool is None:
            return json.dumps(
                {
                    "name": skill_name,
                    "loaded": False,
                    "error": "unknown_skill",
                    "available": sorted(tools.keys()),
                },
                ensure_ascii=False,
            )
        try:
            arguments_schema = json.loads(tool.arguments_schema_json)
        except json.JSONDecodeError:
            arguments_schema = {}
        return json.dumps(
            {
                "name": tool.name,
                "loaded": True,
                "description": tool.description,
                "arguments_schema": arguments_schema,
                "usage_examples": [f"{tool.name}(...)"],
            },
            ensure_ascii=False,
        )


def extract_loaded_skill_name(payload_json: str) -> str | None:
    try:
        payload = json.loads(payload_json)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("loaded") is not True:
        return None
    loaded_name = payload.get("name")
    if not isinstance(loaded_name, str):
        return None
    return loaded_name


__all__ = ["RuntimeToolManager", "extract_loaded_skill_name"]
