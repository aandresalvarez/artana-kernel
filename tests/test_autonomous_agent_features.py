from __future__ import annotations

import json
from pathlib import Path
from typing import TypeVar

import pytest
from pydantic import BaseModel

from artana.agent import AutonomousAgent, CompactionStrategy, ContextBuilder, SubAgentFactory
from artana.agent.experience import ExperienceRule, RuleType, SQLiteExperienceStore
from artana.agent.memory import InMemoryMemoryStore
from artana.events import ChatMessage, EventType, ModelRequestedPayload
from artana.kernel import ArtanaKernel
from artana.models import TenantContext
from artana.ports.model import ModelRequest, ModelResult, ModelUsage, ToolCall
from artana.store import SQLiteStore

OutputModelT = TypeVar("OutputModelT", bound=BaseModel)


class AgentResult(BaseModel):
    done: bool


class ChildResult(BaseModel):
    answer: str


def _tenant() -> TenantContext:
    return TenantContext(
        tenant_id="org_agent_features",
        capabilities=frozenset(),
        budget_usd_limit=3.0,
    )


class CompactionModelPort:
    def __init__(self) -> None:
        self.calls = 0

    async def complete(
        self, request: ModelRequest[OutputModelT]
    ) -> ModelResult[OutputModelT]:
        self.calls += 1
        if "summary" in request.output_schema.model_fields:
            output = request.output_schema.model_validate({"summary": "condensed history"})
        else:
            output = request.output_schema.model_validate({"done": True})
        return ModelResult(
            output=output,
            usage=ModelUsage(prompt_tokens=8, completion_tokens=3, cost_usd=0.01),
        )


class ProgressiveSkillsModelPort:
    def __init__(self) -> None:
        self.calls = 0
        self.allowed_tool_batches: list[list[str]] = []

    async def complete(
        self, request: ModelRequest[OutputModelT]
    ) -> ModelResult[OutputModelT]:
        self.calls += 1
        self.allowed_tool_batches.append([tool.name for tool in request.allowed_tools])

        if self.calls == 1:
            tool_calls = (
                ToolCall(
                    tool_name="load_skill",
                    arguments_json='{"skill_name":"secret_tool"}',
                ),
            )
            output = request.output_schema.model_validate({"done": False})
        elif self.calls == 2:
            tool_calls = (
                ToolCall(
                    tool_name="secret_tool",
                    arguments_json='{"value":"x"}',
                ),
            )
            output = request.output_schema.model_validate({"done": False})
        else:
            tool_calls = ()
            output = request.output_schema.model_validate({"done": True})

        return ModelResult(
            output=output,
            usage=ModelUsage(prompt_tokens=12, completion_tokens=6, cost_usd=0.01),
            tool_calls=tool_calls,
        )


class MemoryModelPort:
    def __init__(self) -> None:
        self.calls = 0
        self.system_messages: list[str] = []

    async def complete(
        self, request: ModelRequest[OutputModelT]
    ) -> ModelResult[OutputModelT]:
        self.calls += 1
        if len(request.messages) > 0:
            self.system_messages.append(request.messages[0].content)
        else:
            self.system_messages.append("")

        if self.calls == 1:
            tool_calls = (
                ToolCall(
                    tool_name="core_memory_append",
                    arguments_json='{"content":"prefers-python"}',
                ),
            )
            output = request.output_schema.model_validate({"done": False})
        else:
            tool_calls = ()
            output = request.output_schema.model_validate({"done": True})
        return ModelResult(
            output=output,
            usage=ModelUsage(prompt_tokens=10, completion_tokens=4, cost_usd=0.01),
            tool_calls=tool_calls,
        )


class AutoReflectModelPort:
    def __init__(self) -> None:
        self.calls = 0
        self.reflection_calls = 0

    async def complete(
        self, request: ModelRequest[OutputModelT]
    ) -> ModelResult[OutputModelT]:
        self.calls += 1
        if "extracted_rules" in request.output_schema.model_fields:
            self.reflection_calls += 1
            output = request.output_schema.model_validate(
                {
                    "extracted_rules": [
                        {
                            "rule_id": "rule_iso_date",
                            "tenant_id": "placeholder",
                            "task_category": "placeholder",
                            "rule_type": RuleType.WIN_PATTERN.value,
                            "content": "Always format dates as YYYY-MM-DD.",
                            "success_count": 1,
                            "fail_count": 0,
                        }
                    ]
                }
            )
            return ModelResult(
                output=output,
                usage=ModelUsage(prompt_tokens=6, completion_tokens=4, cost_usd=0.01),
            )

        output = request.output_schema.model_validate({"done": True})
        return ModelResult(
            output=output,
            usage=ModelUsage(prompt_tokens=6, completion_tokens=3, cost_usd=0.01),
        )


class ChildModelPort:
    def __init__(self) -> None:
        self.calls = 0

    async def complete(
        self, request: ModelRequest[OutputModelT]
    ) -> ModelResult[OutputModelT]:
        self.calls += 1
        output = request.output_schema.model_validate({"answer": "delegated"})
        return ModelResult(
            output=output,
            usage=ModelUsage(prompt_tokens=7, completion_tokens=3, cost_usd=0.01),
        )


class ContextCaptureModelPort:
    def __init__(self) -> None:
        self.first_system_content: str | None = None

    async def complete(
        self, request: ModelRequest[OutputModelT]
    ) -> ModelResult[OutputModelT]:
        if self.first_system_content is None and len(request.messages) > 0:
            self.first_system_content = request.messages[0].content
        output = request.output_schema.model_validate({"done": True})
        return ModelResult(
            output=output,
            usage=ModelUsage(prompt_tokens=3, completion_tokens=2, cost_usd=0.01),
        )


class SkillGuardModelPort:
    def __init__(self) -> None:
        self.calls = 0
        self.allowed_tool_batches: list[list[str]] = []
        self.first_system_content: str | None = None

    async def complete(
        self, request: ModelRequest[OutputModelT]
    ) -> ModelResult[OutputModelT]:
        self.calls += 1
        self.allowed_tool_batches.append([tool.name for tool in request.allowed_tools])
        if self.first_system_content is None and len(request.messages) > 0:
            self.first_system_content = request.messages[0].content

        if self.calls == 1:
            tool_calls = (
                ToolCall(
                    tool_name="load_skill",
                    arguments_json='{"skill_name":"admin_only_tool"}',
                ),
            )
            output = request.output_schema.model_validate({"done": False})
        else:
            tool_calls = ()
            output = request.output_schema.model_validate({"done": True})

        return ModelResult(
            output=output,
            usage=ModelUsage(prompt_tokens=6, completion_tokens=3, cost_usd=0.01),
            tool_calls=tool_calls,
        )


class SubAgentInheritanceModelPort:
    def __init__(self) -> None:
        self.child_runs_seen: set[str] = set()

    async def complete(
        self, request: ModelRequest[OutputModelT]
    ) -> ModelResult[OutputModelT]:
        if "::sub_agent::" in request.run_id:
            self.child_runs_seen.add(request.run_id)
            if len(request.messages) <= 2:
                tool_calls = (
                    ToolCall(tool_name="child_secret_tool", arguments_json="{}"),
                )
                output = request.output_schema.model_validate({"done": False})
            else:
                tool_calls = ()
                output = request.output_schema.model_validate({"done": True})
            return ModelResult(
                output=output,
                usage=ModelUsage(prompt_tokens=5, completion_tokens=2, cost_usd=0.01),
                tool_calls=tool_calls,
            )

        output = request.output_schema.model_validate({"done": True})
        return ModelResult(
            output=output,
            usage=ModelUsage(prompt_tokens=3, completion_tokens=1, cost_usd=0.01),
        )


class CustomContextBuilder(ContextBuilder):
    async def build_messages(
        self,
        *,
        run_id: str,
        tenant: TenantContext,
        short_term_messages: tuple[ChatMessage, ...],
        system_prompt: str,
        active_skills: frozenset[str],
        available_skill_summaries: dict[str, str] | None,
        memory_text: str | None,
    ) -> tuple[ChatMessage, ...]:
        return (ChatMessage(role="system", content="CUSTOM CONTEXT"),) + short_term_messages


class ChildTaskArgs(BaseModel):
    task: str


@pytest.mark.asyncio
async def test_autonomous_agent_compaction_records_compact_step(tmp_path: Path) -> None:
    store = SQLiteStore(str(tmp_path / "state.db"))
    model_port = CompactionModelPort()
    kernel = ArtanaKernel(store=store, model_port=model_port)
    agent = AutonomousAgent(
        kernel=kernel,
        compaction=CompactionStrategy(
            trigger_at_messages=1,
            keep_recent_messages=0,
            summarize_with_model="gpt-4o-mini",
        ),
    )

    try:
        result = await agent.run(
            run_id="run_compaction_feature",
            tenant=_tenant(),
            model="gpt-4o-mini",
            prompt="compress this history",
            output_schema=AgentResult,
            max_iterations=2,
        )
        assert result.done is True
        assert model_port.calls == 2

        events = await store.get_events_for_run("run_compaction_feature")
        step_keys: list[str | None] = []
        for event in events:
            if event.event_type != EventType.MODEL_REQUESTED:
                continue
            payload = event.payload
            if isinstance(payload, ModelRequestedPayload):
                step_keys.append(payload.step_key)
        assert "turn_1_compact" in step_keys
        assert "turn_1_model" in step_keys
    finally:
        await kernel.close()


@pytest.mark.asyncio
async def test_autonomous_agent_progressive_skill_loading(tmp_path: Path) -> None:
    store = SQLiteStore(str(tmp_path / "state.db"))
    model_port = ProgressiveSkillsModelPort()
    kernel = ArtanaKernel(store=store, model_port=model_port)
    tool_calls = 0

    @kernel.tool()
    async def secret_tool(value: str) -> str:
        nonlocal tool_calls
        tool_calls += 1
        return json.dumps({"echo": value})

    agent = AutonomousAgent(kernel=kernel)

    try:
        result = await agent.run(
            run_id="run_progressive_skills",
            tenant=_tenant(),
            model="gpt-4o-mini",
            prompt="use the hidden tool",
            output_schema=AgentResult,
            max_iterations=5,
        )
        assert result.done is True
        assert tool_calls == 1
        assert "secret_tool" not in model_port.allowed_tool_batches[0]
        assert "load_skill" in model_port.allowed_tool_batches[0]
        assert any("secret_tool" in batch for batch in model_port.allowed_tool_batches[1:])
    finally:
        await kernel.close()


@pytest.mark.asyncio
async def test_autonomous_agent_memory_tool_injects_long_term_memory(tmp_path: Path) -> None:
    store = SQLiteStore(str(tmp_path / "state.db"))
    model_port = MemoryModelPort()
    kernel = ArtanaKernel(store=store, model_port=model_port)
    memory_store = InMemoryMemoryStore()
    context_builder = ContextBuilder(memory_store=memory_store)
    agent = AutonomousAgent(
        kernel=kernel,
        context_builder=context_builder,
        memory_store=memory_store,
    )

    try:
        result = await agent.run(
            run_id="run_long_term_memory",
            tenant=_tenant(),
            model="gpt-4o-mini",
            prompt="remember my language preference",
            output_schema=AgentResult,
            max_iterations=4,
        )
        assert result.done is True
        assert model_port.calls >= 2
        assert "Long-Term Memory:\nprefers-python" in model_port.system_messages[1]
        assert await memory_store.load("run_long_term_memory") == "prefers-python"
    finally:
        await kernel.close()


@pytest.mark.asyncio
async def test_context_builder_injects_experience_rules(tmp_path: Path) -> None:
    store = SQLiteStore(str(tmp_path / "state.db"))
    model_port = ContextCaptureModelPort()
    kernel = ArtanaKernel(store=store, model_port=model_port)
    experience_store = SQLiteExperienceStore(str(tmp_path / "experience.db"))
    tenant = _tenant()

    await experience_store.save_rules(
        [
            ExperienceRule(
                rule_id="rule_for_finance_task",
                tenant_id=tenant.tenant_id,
                task_category="Financial_Reporting",
                rule_type=RuleType.WIN_PATTERN,
                content="Always format dates as YYYY-MM-DD.",
                success_count=4,
                fail_count=0,
            ),
            ExperienceRule(
                rule_id="rule_other_tenant",
                tenant_id="another_tenant",
                task_category="Financial_Reporting",
                rule_type=RuleType.ANTI_PATTERN,
                content="Should never be visible to this tenant.",
                success_count=3,
                fail_count=0,
            ),
        ]
    )

    agent = AutonomousAgent(
        kernel=kernel,
        context_builder=ContextBuilder(
            experience_store=experience_store,
            task_category="Financial_Reporting",
            progressive_skills=False,
        ),
    )

    try:
        result = await agent.run(
            run_id="run_experience_context",
            tenant=tenant,
            model="gpt-4o-mini",
            prompt="run extraction",
            output_schema=AgentResult,
            max_iterations=2,
        )
        assert result.done is True
        assert model_port.first_system_content is not None
        assert "[PAST LEARNINGS FOR THIS TASK]" in model_port.first_system_content
        assert (
            "WIN_PATTERN: Always format dates as YYYY-MM-DD."
            in model_port.first_system_content
        )
        assert "Should never be visible to this tenant." not in model_port.first_system_content
    finally:
        await kernel.close()
        await experience_store.close()


@pytest.mark.asyncio
async def test_autonomous_agent_auto_reflect_persists_experience_rules(tmp_path: Path) -> None:
    store = SQLiteStore(str(tmp_path / "state.db"))
    model_port = AutoReflectModelPort()
    kernel = ArtanaKernel(store=store, model_port=model_port)
    experience_store = SQLiteExperienceStore(str(tmp_path / "experience.db"))
    tenant = _tenant()
    task_category = "Financial_Reporting"

    agent = AutonomousAgent(
        kernel=kernel,
        context_builder=ContextBuilder(
            experience_store=experience_store,
            task_category=task_category,
            progressive_skills=False,
        ),
        auto_reflect=True,
        reflection_model="gpt-4o-mini",
    )

    try:
        result = await agent.run(
            run_id="run_auto_reflect",
            tenant=tenant,
            model="gpt-4o-mini",
            prompt="extract the weekly report",
            output_schema=AgentResult,
            max_iterations=2,
        )
        assert result.done is True
        assert model_port.reflection_calls == 1

        saved_rules = await experience_store.get_rules(
            tenant_id=tenant.tenant_id,
            task_category=task_category,
        )
        assert len(saved_rules) == 1
        assert saved_rules[0].rule_id == "rule_iso_date"
        assert saved_rules[0].tenant_id == tenant.tenant_id
        assert saved_rules[0].task_category == task_category

        events = await store.get_events_for_run("run_auto_reflect")
        step_keys: list[str | None] = []
        for event in events:
            if event.event_type != EventType.MODEL_REQUESTED:
                continue
            payload = event.payload
            if isinstance(payload, ModelRequestedPayload):
                step_keys.append(payload.step_key)
        assert "turn_1_model" in step_keys
        assert "turn_1_reflection" in step_keys
    finally:
        await kernel.close()
        await experience_store.close()


@pytest.mark.asyncio
async def test_auto_reflect_requires_experience_configuration(tmp_path: Path) -> None:
    store = SQLiteStore(str(tmp_path / "state.db"))
    kernel = ArtanaKernel(store=store, model_port=ContextCaptureModelPort())
    try:
        with pytest.raises(
            ValueError,
            match="auto_reflect requires context_builder.experience_store",
        ):
            AutonomousAgent(kernel=kernel, auto_reflect=True)
    finally:
        await kernel.close()


@pytest.mark.asyncio
async def test_sub_agent_factory_spawns_child_run_with_lineage(tmp_path: Path) -> None:
    store = SQLiteStore(str(tmp_path / "state.db"))
    model_port = ChildModelPort()
    kernel = ArtanaKernel(store=store, model_port=model_port)
    tenant = _tenant()
    factory = SubAgentFactory(kernel=kernel, tenant=tenant)
    factory.create(
        name="run_child_agent",
        output_schema=ChildResult,
        model="gpt-4o-mini",
        system_prompt="You are a child specialist.",
    )

    try:
        parent = await kernel.start_run(tenant=tenant, run_id="parent_run")
        tool_result = await kernel.step_tool(
            run_id=parent.run_id,
            tenant=tenant,
            tool_name="run_child_agent",
            arguments=ChildTaskArgs(task="analyze this"),
            step_key="spawn_child",
        )
        payload = json.loads(tool_result.result_json)
        assert isinstance(payload, dict)
        child_run_id = payload.get("run_id")
        assert isinstance(child_run_id, str)
        assert child_run_id.startswith("parent_run::sub_agent::")
        assert payload.get("result") == {"answer": "delegated"}

        child_events = await store.get_events_for_run(child_run_id)
        assert len(child_events) >= 1
        assert child_events[0].event_type == EventType.RUN_STARTED
        assert child_events[0].tenant_id == tenant.tenant_id
    finally:
        await kernel.close()


@pytest.mark.asyncio
async def test_progressive_skills_hides_and_blocks_forbidden_tools(tmp_path: Path) -> None:
    store = SQLiteStore(str(tmp_path / "state.db"))
    model_port = SkillGuardModelPort()
    kernel = ArtanaKernel(store=store, model_port=model_port)

    @kernel.tool(requires_capability="admin:tool")
    async def admin_only_tool() -> str:
        return json.dumps({"ok": True})

    agent = AutonomousAgent(kernel=kernel)
    tenant = TenantContext(
        tenant_id="org_no_admin",
        capabilities=frozenset(),
        budget_usd_limit=3.0,
    )

    try:
        result = await agent.run(
            run_id="run_progressive_skill_guard",
            tenant=tenant,
            model="gpt-4o-mini",
            prompt="try loading admin tool",
            output_schema=AgentResult,
            max_iterations=4,
        )
        assert result.done is True
        assert model_port.first_system_content is not None
        assert "admin_only_tool" not in model_port.first_system_content
        assert all(
            "admin_only_tool" not in batch for batch in model_port.allowed_tool_batches
        )

        events = await store.get_events_for_run("run_progressive_skill_guard")
        load_skill_payloads: list[dict[str, object]] = []
        for event in events:
            if event.event_type != EventType.TOOL_COMPLETED:
                continue
            payload = event.payload
            if payload.kind != "tool_completed":
                continue
            if payload.tool_name != "load_skill":
                continue
            parsed = json.loads(payload.result_json)
            if isinstance(parsed, dict):
                load_skill_payloads.append(parsed)
        assert load_skill_payloads
        assert load_skill_payloads[0].get("loaded") is False
        assert load_skill_payloads[0].get("error") == "forbidden_skill"
    finally:
        await kernel.close()


@pytest.mark.asyncio
async def test_sub_agent_factory_inherits_parent_tenant_context(tmp_path: Path) -> None:
    store = SQLiteStore(str(tmp_path / "state.db"))
    model_port = SubAgentInheritanceModelPort()
    kernel = ArtanaKernel(store=store, model_port=model_port)
    tenant = TenantContext(
        tenant_id="org_subagent_inherit",
        capabilities=frozenset({"child:execute"}),
        budget_usd_limit=3.0,
    )
    factory = SubAgentFactory(
        kernel=kernel,
        context_builder=ContextBuilder(progressive_skills=False),
    )

    @kernel.tool(requires_capability="child:execute")
    async def child_secret_tool() -> str:
        return json.dumps({"ok": True})

    factory.create(
        name="run_child_with_inherited_tenant",
        output_schema=AgentResult,
        model="gpt-4o-mini",
        system_prompt="You are a delegated child agent.",
        max_iterations=3,
    )

    try:
        parent = await kernel.start_run(tenant=tenant, run_id="parent_inherit")
        tool_result = await kernel.step_tool(
            run_id=parent.run_id,
            tenant=tenant,
            tool_name="run_child_with_inherited_tenant",
            arguments=ChildTaskArgs(task="use child secret tool"),
            step_key="spawn_child",
        )
        payload = json.loads(tool_result.result_json)
        assert isinstance(payload, dict)
        assert payload.get("result") == {"done": True}
        assert model_port.child_runs_seen
    finally:
        await kernel.close()


@pytest.mark.asyncio
async def test_context_builder_override_is_used_by_autonomous_agent(tmp_path: Path) -> None:
    store = SQLiteStore(str(tmp_path / "state.db"))
    model_port = ContextCaptureModelPort()
    kernel = ArtanaKernel(store=store, model_port=model_port)
    agent = AutonomousAgent(
        kernel=kernel,
        context_builder=CustomContextBuilder(progressive_skills=False),
    )

    try:
        result = await agent.run(
            run_id="run_custom_context",
            tenant=_tenant(),
            model="gpt-4o-mini",
            prompt="hello",
            output_schema=AgentResult,
            max_iterations=2,
        )
        assert result.done is True
        assert model_port.first_system_content == "CUSTOM CONTEXT"
    finally:
        await kernel.close()
