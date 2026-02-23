from __future__ import annotations

import json

from artana.agent.memory import MemoryStore
from artana.kernel import ArtanaKernel
from artana.ports.tool import ToolCallable, ToolExecutionContext


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
        query_event_history: str,
    ) -> None:
        self._kernel = kernel
        self._memory_store = memory_store
        self._progressive_skills = progressive_skills
        self._load_skill_name = load_skill_name
        self._core_memory_append = core_memory_append
        self._core_memory_replace = core_memory_replace
        self._core_memory_search = core_memory_search
        self._query_event_history = query_event_history
        self._registered = False

    def ensure_registered(self) -> None:
        if self._registered:
            return

        async def load_skill(skill_name: str, artana_context: ToolExecutionContext) -> str:
            return self._tool_description(
                skill_name=skill_name,
                tenant_capabilities=artana_context.tenant_capabilities,
            )
        self._register_runtime_tool(
            name=self._load_skill_name,
            function=load_skill,
        )

        async def core_memory_append(content: str, artana_context: ToolExecutionContext) -> str:
            await self._memory_store.append(run_id=artana_context.run_id, text=content)
            return json.dumps({"status": "appended", "run_id": artana_context.run_id})
        self._register_runtime_tool(
            name=self._core_memory_append,
            function=core_memory_append,
        )

        async def core_memory_replace(content: str, artana_context: ToolExecutionContext) -> str:
            await self._memory_store.replace(run_id=artana_context.run_id, content=content)
            return json.dumps({"status": "replaced", "run_id": artana_context.run_id})
        self._register_runtime_tool(
            name=self._core_memory_replace,
            function=core_memory_replace,
        )

        async def core_memory_search(query: str, artana_context: ToolExecutionContext) -> str:
            return await self._memory_store.search(run_id=artana_context.run_id, query=query)
        self._register_runtime_tool(
            name=self._core_memory_search,
            function=core_memory_search,
        )

        async def query_event_history(
            limit: int,
            event_type: str,
            artana_context: ToolExecutionContext,
        ) -> str:
            if limit <= 0:
                return json.dumps(
                    {
                        "ok": False,
                        "error": "invalid_limit",
                        "detail": "limit must be >= 1",
                    },
                    ensure_ascii=False,
                )
            events = await self._kernel.get_events(run_id=artana_context.run_id)
            normalized_event_type = event_type.strip().lower()
            if normalized_event_type in {"", "*", "all"}:
                filtered_events = list(events)
            else:
                filtered_events = [
                    event
                    for event in events
                    if event.event_type.value == normalized_event_type
                ]
            selected = filtered_events[-limit:]
            return json.dumps(
                {
                    "ok": True,
                    "run_id": artana_context.run_id,
                    "event_type": normalized_event_type or "all",
                    "returned": len(selected),
                    "events": [
                        {
                            "seq": event.seq,
                            "event_id": event.event_id,
                            "event_type": event.event_type.value,
                            "timestamp": event.timestamp.isoformat(),
                            "payload": event.payload.model_dump(mode="json"),
                        }
                        for event in selected
                    ],
                },
                ensure_ascii=False,
            )
        self._register_runtime_tool(
            name=self._query_event_history,
            function=query_event_history,
            requires_capability="self_reflection",
        )

        self._registered = True

    def visible_tool_names(
        self,
        *,
        loaded_skills: set[str],
        tenant_capabilities: frozenset[str],
    ) -> set[str] | None:
        if not self._progressive_skills:
            return None
        runtime_tools = self._runtime_tool_names()
        allowed_loaded_skills = {
            tool_name
            for tool_name in loaded_skills
            if self._is_tool_allowed_for_capabilities(
                tool_name=tool_name,
                tenant_capabilities=tenant_capabilities,
            )
        }
        runtime_tools.update(allowed_loaded_skills)
        return {
            tool.name
            for tool in self._kernel.list_registered_tools()
            if tool.name in runtime_tools
        }

    def available_skill_summaries(
        self, *, tenant_capabilities: frozenset[str]
    ) -> dict[str, str]:
        summaries: dict[str, str] = {}
        runtime = self._runtime_tool_names()
        for tool in self._kernel.list_registered_tools():
            if tool.name in runtime:
                continue
            if not self._is_tool_allowed_for_capabilities(
                tool_name=tool.name,
                tenant_capabilities=tenant_capabilities,
            ):
                continue
            summaries[tool.name] = tool.description or "no description"
        return summaries

    def _tool_description(self, *, skill_name: str, tenant_capabilities: frozenset[str]) -> str:
        tools = {tool.name: tool for tool in self._kernel.list_registered_tools()}
        runtime_tools = self._runtime_tool_names()
        visible_skill_names = sorted(
            tool_name
            for tool_name in tools.keys()
            if tool_name not in runtime_tools
            and self._is_tool_allowed_for_capabilities(
                tool_name=tool_name,
                tenant_capabilities=tenant_capabilities,
            )
        )
        tool = tools.get(skill_name)
        if tool is None:
            return json.dumps(
                {
                    "name": skill_name,
                    "loaded": False,
                    "error": "unknown_skill",
                    "available": visible_skill_names,
                },
                ensure_ascii=False,
            )
        if tool.name in runtime_tools or not self._is_tool_allowed_for_capabilities(
            tool_name=tool.name,
            tenant_capabilities=tenant_capabilities,
        ):
            return json.dumps(
                {
                    "name": skill_name,
                    "loaded": False,
                    "error": "forbidden_skill",
                    "available": visible_skill_names,
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

    def _runtime_tool_names(self) -> set[str]:
        return {
            self._load_skill_name,
            self._core_memory_append,
            self._core_memory_replace,
            self._core_memory_search,
            self._query_event_history,
        }

    def _register_runtime_tool(
        self,
        *,
        name: str,
        function: ToolCallable,
        requires_capability: str | None = None,
    ) -> None:
        function.__name__ = name
        self._kernel.tool(requires_capability=requires_capability)(function)

    def _is_tool_allowed_for_capabilities(
        self,
        *,
        tool_name: str,
        tenant_capabilities: frozenset[str],
    ) -> bool:
        capability_map = self._kernel.tool_capability_map()
        required_capability = capability_map.get(tool_name)
        if required_capability is None:
            return tool_name in capability_map
        return required_capability in tenant_capabilities


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
