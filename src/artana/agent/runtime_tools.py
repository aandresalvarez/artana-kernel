from __future__ import annotations

import json
from collections.abc import Iterable

from artana.agent.memory import MemoryStore
from artana.canonicalization import canonical_json_dumps
from artana.json_utils import sha256_hex
from artana.kernel import ArtanaKernel
from artana.models import TenantContext
from artana.ports.tool import ToolCallable, ToolExecutionContext
from artana.safety import IntentPlanRecord
from artana.skills import SkillDefinition, SkillRegistry


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
        record_intent_plan: str = "record_intent_plan",
        skill_registry: SkillRegistry | None = None,
        allowed_skill_names: frozenset[str] | None = None,
    ) -> None:
        self._kernel = kernel
        self._memory_store = memory_store
        self._progressive_skills = progressive_skills
        self._load_skill_name = load_skill_name
        self._core_memory_append = core_memory_append
        self._core_memory_replace = core_memory_replace
        self._core_memory_search = core_memory_search
        self._query_event_history = query_event_history
        self._record_intent_plan = record_intent_plan
        self._skill_registry = skill_registry
        self._allowed_skill_names = allowed_skill_names
        self._registered = False

    def ensure_registered(self) -> None:
        if self._registered:
            return

        async def load_skill(skill_name: str, artana_context: ToolExecutionContext) -> str:
            return self._load_skill_payload(
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
            events = await self._kernel.get_events(
                run_id=artana_context.run_id,
                tenant=TenantContext(
                    tenant_id=artana_context.tenant_id,
                    capabilities=artana_context.tenant_capabilities,
                    budget_usd_limit=artana_context.tenant_budget_usd_limit or 1.0,
                ),
            )
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

        async def record_intent_plan(
            goal: str,
            why: str,
            success_criteria: str,
            assumed_state: str,
            applies_to_tools: list[str] | None,
            intent_id: str | None,
            artana_context: ToolExecutionContext,
        ) -> str:
            payload = {
                "goal": goal,
                "why": why,
                "success_criteria": success_criteria,
                "assumed_state": assumed_state,
                "applies_to_tools": applies_to_tools or [],
            }
            resolved_intent_id = intent_id
            if resolved_intent_id is None:
                resolved_intent_id = sha256_hex(canonical_json_dumps(payload))
            tenant_budget = artana_context.tenant_budget_usd_limit
            if tenant_budget is None:
                return json.dumps(
                    {
                        "ok": False,
                        "error": "missing_tenant_budget",
                        "detail": "tenant_budget_usd_limit missing in ToolExecutionContext",
                    },
                    ensure_ascii=False,
                )
            await self._kernel.record_intent_plan(
                run_id=artana_context.run_id,
                tenant=TenantContext(
                    tenant_id=artana_context.tenant_id,
                    capabilities=artana_context.tenant_capabilities,
                    budget_usd_limit=tenant_budget,
                ),
                intent=IntentPlanRecord(
                    intent_id=resolved_intent_id,
                    goal=goal,
                    why=why,
                    success_criteria=success_criteria,
                    assumed_state=assumed_state,
                    applies_to_tools=tuple(applies_to_tools or []),
                ),
            )
            return json.dumps(
                {"ok": True, "intent_id": resolved_intent_id},
                ensure_ascii=False,
            )

        self._register_runtime_tool(
            name=self._record_intent_plan,
            function=record_intent_plan,
        )

        self._registered = True

    def visible_tool_names(
        self,
        *,
        loaded_skills: set[str],
        tenant_capabilities: frozenset[str],
        active_registry_skills: set[str] | None = None,
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
        runtime_tools.update(
            self._active_registry_skill_tool_names(
                active_registry_skills=active_registry_skills or set(),
                tenant_capabilities=tenant_capabilities,
            )
        )
        return {
            tool.name
            for tool in self._kernel.list_registered_tools()
            if tool.name in runtime_tools
        }

    def available_skill_summaries(
        self, *, tenant_capabilities: frozenset[str]
    ) -> dict[str, str]:
        summaries: dict[str, str] = {}
        for skill in self._visible_registry_skills(tenant_capabilities=tenant_capabilities):
            summaries[skill.name] = skill.summary
        runtime = self._runtime_tool_names()
        for tool in self._kernel.list_registered_tools():
            if tool.name in runtime:
                continue
            if tool.name in summaries:
                continue
            if not self._is_tool_allowed_for_capabilities(
                tool_name=tool.name,
                tenant_capabilities=tenant_capabilities,
            ):
                continue
            summaries[tool.name] = tool.description or "no description"
        return summaries

    def active_registry_skill_definitions(
        self,
        *,
        active_skill_names: Iterable[str],
        tenant_capabilities: frozenset[str],
    ) -> tuple[SkillDefinition, ...]:
        if self._skill_registry is None:
            return ()
        active_names = set(active_skill_names)
        definitions: list[SkillDefinition] = []
        for skill_name in sorted(active_names):
            skill = self._skill_registry.get_skill(skill_name)
            if skill is None:
                continue
            if self._registry_skill_load_error(
                skill=skill,
                tenant_capabilities=tenant_capabilities,
            ) is not None:
                continue
            definitions.append(skill)
        return tuple(definitions)

    def resolve_preloaded_registry_skills(
        self,
        *,
        preload_skill_names: frozenset[str],
        tenant_capabilities: frozenset[str],
    ) -> set[str]:
        if self._skill_registry is None or not preload_skill_names:
            return set()
        active_skills: set[str] = set()
        for skill_name in sorted(preload_skill_names):
            skill = self._skill_registry.get_skill(skill_name)
            if skill is None:
                raise ValueError(f"Unknown preloaded skill: {skill_name}")
            error = self._registry_skill_load_error(
                skill=skill,
                tenant_capabilities=tenant_capabilities,
            )
            if error is None:
                active_skills.add(skill_name)
                continue
            if error == "forbidden_skill":
                continue
            raise RuntimeError(f"Preloaded skill {skill_name!r} is invalid and cannot load.")
        return active_skills

    def _load_skill_payload(self, *, skill_name: str, tenant_capabilities: frozenset[str]) -> str:
        if self._skill_registry is not None:
            skill = self._skill_registry.get_skill(skill_name)
            if skill is not None:
                return self._registry_skill_payload(
                    skill=skill,
                    tenant_capabilities=tenant_capabilities,
                )
        return self._legacy_tool_description(
            skill_name=skill_name,
            tenant_capabilities=tenant_capabilities,
        )

    def _legacy_tool_description(
        self,
        *,
        skill_name: str,
        tenant_capabilities: frozenset[str],
    ) -> str:
        tools = {tool.name: tool for tool in self._kernel.list_registered_tools()}
        runtime_tools = self._runtime_tool_names()
        visible_skill_names = self._visible_available_skill_names(
            tenant_capabilities=tenant_capabilities
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

    def _registry_skill_payload(
        self,
        *,
        skill: SkillDefinition,
        tenant_capabilities: frozenset[str],
    ) -> str:
        visible_skill_names = self._visible_available_skill_names(
            tenant_capabilities=tenant_capabilities
        )
        error = self._registry_skill_load_error(
            skill=skill,
            tenant_capabilities=tenant_capabilities,
        )
        if error is not None:
            payload: dict[str, object] = {
                "name": skill.name,
                "kind": "registry_skill",
                "loaded": False,
                "error": error,
                "available": visible_skill_names,
            }
            missing_tools = self._missing_skill_tools(skill)
            if missing_tools:
                payload["missing_tools"] = missing_tools
            return json.dumps(payload, ensure_ascii=False)
        return json.dumps(
            {
                "name": skill.name,
                "kind": "registry_skill",
                "loaded": True,
                "summary": skill.summary,
                "instructions_markdown": skill.instructions_markdown,
                "tool_names": list(skill.tools),
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
            self._record_intent_plan,
        }

    def _visible_available_skill_names(
        self,
        *,
        tenant_capabilities: frozenset[str],
    ) -> list[str]:
        return sorted(self.available_skill_summaries(tenant_capabilities=tenant_capabilities))

    def _visible_registry_skills(
        self,
        *,
        tenant_capabilities: frozenset[str],
    ) -> tuple[SkillDefinition, ...]:
        if self._skill_registry is None:
            return ()
        visible_skills = [
            skill
            for skill in self._skill_registry.list_skills()
            if self._registry_skill_is_visible(
                skill=skill,
                tenant_capabilities=tenant_capabilities,
            )
        ]
        return tuple(sorted(visible_skills, key=lambda skill: skill.name))

    def _active_registry_skill_tool_names(
        self,
        *,
        active_registry_skills: set[str],
        tenant_capabilities: frozenset[str],
    ) -> set[str]:
        tool_names: set[str] = set()
        for skill in self.active_registry_skill_definitions(
            active_skill_names=active_registry_skills,
            tenant_capabilities=tenant_capabilities,
        ):
            tool_names.update(skill.tools)
        return tool_names

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

    def _is_registry_skill_allowed(self, skill_name: str) -> bool:
        if self._allowed_skill_names is None:
            return True
        return skill_name in self._allowed_skill_names

    def _registry_skill_is_visible(
        self,
        *,
        skill: SkillDefinition,
        tenant_capabilities: frozenset[str],
    ) -> bool:
        if not self._is_registry_skill_allowed(skill.name):
            return False
        if any(
            capability not in tenant_capabilities
            for capability in skill.requires_capabilities
        ):
            return False
        registered_tool_names = {tool.name for tool in self._kernel.list_registered_tools()}
        capability_map = self._kernel.tool_capability_map()
        for tool_name in skill.tools:
            if tool_name not in registered_tool_names:
                continue
            required_capability = capability_map.get(tool_name)
            if (
                required_capability is not None
                and required_capability not in tenant_capabilities
            ):
                return False
        return True

    def _registry_skill_load_error(
        self,
        *,
        skill: SkillDefinition,
        tenant_capabilities: frozenset[str],
    ) -> str | None:
        if not self._is_registry_skill_allowed(skill.name):
            return "forbidden_skill"
        if any(
            capability not in tenant_capabilities
            for capability in skill.requires_capabilities
        ):
            return "forbidden_skill"
        missing_tools = self._missing_skill_tools(skill)
        if missing_tools:
            return "invalid_skill"
        capability_map = self._kernel.tool_capability_map()
        for tool_name in skill.tools:
            required_capability = capability_map.get(tool_name)
            if (
                required_capability is not None
                and required_capability not in tenant_capabilities
            ):
                return "forbidden_skill"
        return None

    def _missing_skill_tools(self, skill: SkillDefinition) -> list[str]:
        registered_tool_names = {tool.name for tool in self._kernel.list_registered_tools()}
        return [tool_name for tool_name in skill.tools if tool_name not in registered_tool_names]


def extract_loaded_skill_name(payload_json: str) -> str | None:
    loaded_skill = extract_loaded_skill(payload_json)
    if loaded_skill is None:
        return None
    return loaded_skill[0]


def extract_loaded_skill(payload_json: str) -> tuple[str, str] | None:
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
    kind = payload.get("kind")
    if kind == "registry_skill":
        return loaded_name, kind
    return loaded_name, "legacy_tool"


__all__ = ["RuntimeToolManager", "extract_loaded_skill", "extract_loaded_skill_name"]
