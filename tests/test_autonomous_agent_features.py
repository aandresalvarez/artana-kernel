from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import TypeVar

import pytest
from pydantic import BaseModel

from artana.agent import (
    AgentRunFailed,
    AutonomousAgent,
    CompactionStrategy,
    ContextBuilder,
    MaxIterationsExceeded,
    SubAgentFactory,
)
from artana.agent.experience import ExperienceRule, RuleType, SQLiteExperienceStore
from artana.agent.memory import InMemoryMemoryStore
from artana.agent.runtime_tools import RuntimeToolManager
from artana.events import ChatMessage, EventType, ModelRequestedPayload, RunSummaryPayload
from artana.kernel import ArtanaKernel
from artana.middleware import QuotaMiddleware
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
            tool_calls: tuple[ToolCall, ...]
            tool_calls = (
                ToolCall(
                    tool_name="load_skill",
                    arguments_json='{"skill_name":"secret_tool"}',
                    tool_call_id="call_load_skill_1",
                ),
            )
            output = request.output_schema.model_validate({"done": False})
        elif self.calls == 2:
            tool_calls = (
                ToolCall(
                    tool_name="secret_tool",
                    arguments_json='{"value":"x"}',
                    tool_call_id="call_secret_tool_2",
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
            tool_calls: tuple[ToolCall, ...]
            tool_calls = (
                ToolCall(
                    tool_name="core_memory_append",
                    arguments_json='{"content":"prefers-python"}',
                    tool_call_id="call_memory_append_1",
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
            tool_calls: tuple[ToolCall, ...]
            tool_calls = (
                ToolCall(
                    tool_name="load_skill",
                    arguments_json='{"skill_name":"admin_only_tool"}',
                    tool_call_id="call_load_admin_skill_1",
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
            tool_calls: tuple[ToolCall, ...]
            if len(request.messages) <= 2:
                tool_calls = (
                    ToolCall(
                        tool_name="child_secret_tool",
                        arguments_json="{}",
                        tool_call_id="call_child_secret_tool_1",
                    ),
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


class ToolProtocolCaptureModelPort:
    def __init__(self) -> None:
        self.calls = 0
        self.messages_second_call: tuple[ChatMessage, ...] = ()

    async def complete(
        self, request: ModelRequest[OutputModelT]
    ) -> ModelResult[OutputModelT]:
        self.calls += 1
        if self.calls == 1:
            output = request.output_schema.model_validate({"done": False})
            return ModelResult(
                output=output,
                usage=ModelUsage(prompt_tokens=6, completion_tokens=2, cost_usd=0.01),
                tool_calls=(
                    ToolCall(
                        tool_name="lookup_weather",
                        arguments_json='{"city":"SF"}',
                        tool_call_id="call_weather_1",
                    ),
                ),
            )
        self.messages_second_call = tuple(request.messages)
        output = request.output_schema.model_validate({"done": True})
        return ModelResult(
            output=output,
            usage=ModelUsage(prompt_tokens=6, completion_tokens=2, cost_usd=0.01),
        )


class MissingToolCallIdModelPort:
    async def complete(
        self, request: ModelRequest[OutputModelT]
    ) -> ModelResult[OutputModelT]:
        output = request.output_schema.model_validate({"done": False})
        return ModelResult(
            output=output,
            usage=ModelUsage(prompt_tokens=4, completion_tokens=2, cost_usd=0.01),
            tool_calls=(
                ToolCall(
                    tool_name="any_tool",
                    arguments_json="{}",
                ),
            ),
        )


class BudgetExceededModelPort:
    async def complete(
        self, request: ModelRequest[OutputModelT]
    ) -> ModelResult[OutputModelT]:
        output = request.output_schema.model_validate({"done": True})
        return ModelResult(
            output=output,
            usage=ModelUsage(prompt_tokens=2, completion_tokens=1, cost_usd=1.0),
        )


class EndlessToolLoopModelPort:
    async def complete(
        self, request: ModelRequest[OutputModelT]
    ) -> ModelResult[OutputModelT]:
        output = request.output_schema.model_validate({"done": False})
        return ModelResult(
            output=output,
            usage=ModelUsage(prompt_tokens=4, completion_tokens=2, cost_usd=0.01),
            tool_calls=(
                ToolCall(
                    tool_name="spin",
                    arguments_json="{}",
                    tool_call_id="call_spin_loop",
                ),
            ),
        )


class QueryHistoryModelPort:
    def __init__(self) -> None:
        self.calls = 0

    async def complete(
        self, request: ModelRequest[OutputModelT]
    ) -> ModelResult[OutputModelT]:
        self.calls += 1
        if self.calls == 1:
            output = request.output_schema.model_validate({"done": False})
            return ModelResult(
                output=output,
                usage=ModelUsage(prompt_tokens=5, completion_tokens=2, cost_usd=0.01),
                tool_calls=(
                    ToolCall(
                        tool_name="query_event_history",
                        arguments_json='{"limit":5,"event_type":"all"}',
                        tool_call_id="call_history_1",
                    ),
                ),
            )
        output = request.output_schema.model_validate({"done": True})
        return ModelResult(
            output=output,
            usage=ModelUsage(prompt_tokens=5, completion_tokens=2, cost_usd=0.01),
        )


class RecordIntentPlanModelPort:
    def __init__(self) -> None:
        self.calls = 0

    async def complete(
        self, request: ModelRequest[OutputModelT]
    ) -> ModelResult[OutputModelT]:
        self.calls += 1
        if self.calls == 1:
            output = request.output_schema.model_validate({"done": False})
            return ModelResult(
                output=output,
                usage=ModelUsage(prompt_tokens=5, completion_tokens=2, cost_usd=0.01),
                tool_calls=(
                    ToolCall(
                        tool_name="record_intent_plan",
                        arguments_json=(
                            '{"goal":"Send invoice","why":"Billing cycle closed",'
                            '"success_criteria":"Invoice dispatched exactly once",'
                            '"assumed_state":"Customer is active",'
                            '"applies_to_tools":["send_invoice"],'
                            '"intent_id":"intent_runtime_1"}'
                        ),
                        tool_call_id="call_intent_1",
                    ),
                ),
            )
        output = request.output_schema.model_validate({"done": True})
        return ModelResult(
            output=output,
            usage=ModelUsage(prompt_tokens=5, completion_tokens=2, cost_usd=0.01),
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
        available_skill_summaries: Mapping[str, str] | None,
        memory_text: str | None,
    ) -> tuple[ChatMessage, ...]:
        return (ChatMessage(role="system", content="CUSTOM CONTEXT"),) + short_term_messages


class ChildTaskArgs(BaseModel):
    task: str


class MemoryAppendArgs(BaseModel):
    content: str


class EventHistoryArgs(BaseModel):
    limit: int
    event_type: str


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
        run_summaries = [
            event.payload
            for event in events
            if event.event_type == EventType.RUN_SUMMARY
            and isinstance(event.payload, RunSummaryPayload)
        ]
        assert any(
            payload.summary_type == AutonomousAgent.COMPACTION_ARTIFACT_SUMMARY_TYPE
            for payload in run_summaries
        )
    finally:
        await kernel.close()


@pytest.mark.asyncio
async def test_autonomous_agent_compaction_reuses_artifact_for_identical_window(
    tmp_path: Path,
) -> None:
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
    tenant = _tenant()
    run_id = "run_compaction_artifact_reuse"

    try:
        await kernel.start_run(tenant=tenant, run_id=run_id)
        short_term_messages = (ChatMessage(role="user", content="compress this history"),)

        first = await agent._compact_if_needed(
            run_id=run_id,
            tenant=tenant,
            model="gpt-4o-mini",
            iteration=1,
            short_term_messages=short_term_messages,
        )
        second = await agent._compact_if_needed(
            run_id=run_id,
            tenant=tenant,
            model="gpt-4o-mini",
            iteration=2,
            short_term_messages=short_term_messages,
        )

        assert first == second
        assert model_port.calls == 1

        artifact_summary = await kernel.get_latest_run_summary(
            run_id=run_id,
            summary_type=AutonomousAgent.COMPACTION_ARTIFACT_SUMMARY_TYPE,
        )
        assert artifact_summary is not None
        artifact_payload = json.loads(artifact_summary.summary_json)
        assert artifact_payload["summary"] == "condensed history"
        assert len(artifact_payload["window_hash"]) == 64
        assert len(artifact_payload["summary_hash"]) == 64

        events = await store.get_events_for_run(run_id)
        compact_model_steps: list[str | None] = []
        for event in events:
            if event.event_type != EventType.MODEL_REQUESTED:
                continue
            payload = event.payload
            if isinstance(payload, ModelRequestedPayload):
                compact_model_steps.append(payload.step_key)
        assert "turn_1_compact" in compact_model_steps
        assert "turn_2_compact" not in compact_model_steps
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
async def test_context_builder_injects_workspace_context_file(tmp_path: Path) -> None:
    workspace_context = tmp_path / "ACTIVE_PLAN.md"
    workspace_context.write_text("1. Fix tests\n2. Run lint", encoding="utf-8")
    context_builder = ContextBuilder(
        progressive_skills=False,
        workspace_context_path=str(workspace_context),
    )

    messages = await context_builder.build_messages(
        run_id="run_workspace_context",
        tenant=_tenant(),
        short_term_messages=(ChatMessage(role="user", content="continue"),),
        system_prompt="You are the agent.",
        active_skills=frozenset(),
        available_skill_summaries=None,
        memory_text=None,
    )

    assert messages[0].role == "system"
    assert (
        "Workspace Context / Active Plan:\n1. Fix tests\n2. Run lint"
        in messages[0].content
    )


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
    factory = SubAgentFactory(kernel=kernel)
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
async def test_autonomous_agent_emits_openai_tool_protocol_messages(tmp_path: Path) -> None:
    store = SQLiteStore(str(tmp_path / "state.db"))
    model_port = ToolProtocolCaptureModelPort()
    kernel = ArtanaKernel(store=store, model_port=model_port)

    @kernel.tool()
    async def lookup_weather(city: str) -> str:
        return json.dumps({"city": city, "temp_c": 20})

    agent = AutonomousAgent(
        kernel=kernel,
        context_builder=ContextBuilder(progressive_skills=False),
    )

    try:
        result = await agent.run(
            run_id="run_tool_protocol",
            tenant=_tenant(),
            model="gpt-4o-mini",
            prompt="get weather",
            output_schema=AgentResult,
            max_iterations=3,
        )
        assert result.done is True
        assert len(model_port.messages_second_call) >= 4
        assistant_tool_message = model_port.messages_second_call[-2]
        tool_result_message = model_port.messages_second_call[-1]
        assert assistant_tool_message.role == "assistant"
        assert assistant_tool_message.tool_calls is not None
        assert assistant_tool_message.tool_calls[0].id == "call_weather_1"
        assert assistant_tool_message.tool_calls[0].function.name == "lookup_weather"
        assert tool_result_message.role == "tool"
        assert tool_result_message.tool_call_id == "call_weather_1"
        assert tool_result_message.name == "lookup_weather"
        parsed_tool_content = json.loads(tool_result_message.content)
        assert parsed_tool_content["city"] == "SF"
    finally:
        await kernel.close()


@pytest.mark.asyncio
async def test_autonomous_agent_requires_tool_call_id_for_tool_execution(
    tmp_path: Path,
) -> None:
    store = SQLiteStore(str(tmp_path / "state.db"))
    kernel = ArtanaKernel(store=store, model_port=MissingToolCallIdModelPort())
    agent = AutonomousAgent(
        kernel=kernel,
        context_builder=ContextBuilder(progressive_skills=False),
    )

    try:
        with pytest.raises(ValueError, match="missing tool_call_id"):
            await agent.run(
                run_id="run_missing_tool_call_id",
                tenant=_tenant(),
                model="gpt-4o-mini",
                prompt="use any tool",
                output_schema=AgentResult,
                max_iterations=2,
            )
    finally:
        await kernel.close()


@pytest.mark.asyncio
async def test_autonomous_agent_budget_exceeded_raises_failed_status(tmp_path: Path) -> None:
    store = SQLiteStore(str(tmp_path / "state.db"))
    kernel = ArtanaKernel(
        store=store,
        model_port=BudgetExceededModelPort(),
        middleware=[QuotaMiddleware()],
    )
    agent = AutonomousAgent(kernel=kernel)
    tenant = TenantContext(
        tenant_id="org_budget_fail",
        capabilities=frozenset(),
        budget_usd_limit=0.5,
    )

    try:
        with pytest.raises(AgentRunFailed) as exc_info:
            await agent.run(
                run_id="run_budget_fail",
                tenant=tenant,
                model="gpt-4o-mini",
                prompt="do work",
                output_schema=AgentResult,
                max_iterations=2,
            )
        assert exc_info.value.status == "failed_budget_exceeded"
    finally:
        await kernel.close()


@pytest.mark.asyncio
async def test_autonomous_agent_raises_max_iterations_exceeded(tmp_path: Path) -> None:
    store = SQLiteStore(str(tmp_path / "state.db"))
    model_port = EndlessToolLoopModelPort()
    kernel = ArtanaKernel(store=store, model_port=model_port)

    @kernel.tool()
    async def spin() -> str:
        return json.dumps({"spun": True})

    agent = AutonomousAgent(
        kernel=kernel,
        context_builder=ContextBuilder(progressive_skills=False),
    )

    try:
        with pytest.raises(MaxIterationsExceeded):
            await agent.run(
                run_id="run_max_iterations",
                tenant=_tenant(),
                model="gpt-4o-mini",
                prompt="keep spinning",
                output_schema=AgentResult,
                max_iterations=2,
            )
    finally:
        await kernel.close()


@pytest.mark.asyncio
async def test_runtime_query_event_history_tool_is_callable(tmp_path: Path) -> None:
    store = SQLiteStore(str(tmp_path / "state.db"))
    model_port = QueryHistoryModelPort()
    kernel = ArtanaKernel(store=store, model_port=model_port)
    tenant = TenantContext(
        tenant_id="org_reflection",
        capabilities=frozenset({"self_reflection"}),
        budget_usd_limit=3.0,
    )
    agent = AutonomousAgent(
        kernel=kernel,
        context_builder=ContextBuilder(progressive_skills=False),
    )

    try:
        result = await agent.run(
            run_id="run_query_history",
            tenant=tenant,
            model="gpt-4o-mini",
            prompt="inspect history",
            output_schema=AgentResult,
            max_iterations=3,
        )
        assert result.done is True

        events = await store.get_events_for_run("run_query_history")
        query_results: list[dict[str, object]] = []
        for event in events:
            if event.event_type != EventType.TOOL_COMPLETED:
                continue
            payload = event.payload
            if payload.kind != "tool_completed":
                continue
            if payload.tool_name != "query_event_history":
                continue
            parsed = json.loads(payload.result_json)
            if isinstance(parsed, dict):
                query_results.append(parsed)
        assert query_results
        assert query_results[0].get("ok") is True
        assert isinstance(query_results[0].get("events"), list)
    finally:
        await kernel.close()


@pytest.mark.asyncio
async def test_runtime_record_intent_plan_tool_is_callable(tmp_path: Path) -> None:
    store = SQLiteStore(str(tmp_path / "state.db"))
    model_port = RecordIntentPlanModelPort()
    kernel = ArtanaKernel(store=store, model_port=model_port)
    agent = AutonomousAgent(
        kernel=kernel,
        context_builder=ContextBuilder(progressive_skills=False),
    )

    try:
        result = await agent.run(
            run_id="run_record_intent",
            tenant=_tenant(),
            model="gpt-4o-mini",
            prompt="prepare intent plan",
            output_schema=AgentResult,
            max_iterations=3,
        )
        assert result.done is True

        summary = await kernel.get_latest_run_summary(
            run_id="run_record_intent",
            summary_type="policy::intent_plan",
        )
        assert summary is not None
        payload = json.loads(summary.summary_json)
        assert payload.get("intent_id") == "intent_runtime_1"
        assert payload.get("goal") == "Send invoice"
        assert payload.get("applies_to_tools") == ["send_invoice"]
    finally:
        await kernel.close()


@pytest.mark.asyncio
async def test_runtime_tools_register_with_configured_names(tmp_path: Path) -> None:
    store = SQLiteStore(str(tmp_path / "state.db"))
    kernel = ArtanaKernel(store=store, model_port=QueryHistoryModelPort())
    tenant = TenantContext(
        tenant_id="org_runtime_names",
        capabilities=frozenset({"self_reflection"}),
        budget_usd_limit=3.0,
    )
    runtime_tools = RuntimeToolManager(
        kernel=kernel,
        memory_store=InMemoryMemoryStore(),
        progressive_skills=True,
        load_skill_name="load_skill_v2",
        core_memory_append="memory_append_v2",
        core_memory_replace="memory_replace_v2",
        core_memory_search="memory_search_v2",
        query_event_history="query_history_v2",
    )
    runtime_tools.ensure_registered()

    configured_runtime_names = {
        "load_skill_v2",
        "memory_append_v2",
        "memory_replace_v2",
        "memory_search_v2",
        "query_history_v2",
    }

    try:
        registered = {tool.name for tool in kernel.list_registered_tools()}
        assert configured_runtime_names.issubset(registered)
        assert "load_skill" not in registered
        assert "query_event_history" not in registered

        visible = runtime_tools.visible_tool_names(
            loaded_skills=set(),
            tenant_capabilities=tenant.capabilities,
        )
        assert visible is not None
        assert configured_runtime_names.issubset(visible)

        await kernel.start_run(tenant=tenant, run_id="run_runtime_names")
        append_result = await kernel.step_tool(
            run_id="run_runtime_names",
            tenant=tenant,
            tool_name="memory_append_v2",
            arguments=MemoryAppendArgs(content="hello memory"),
            step_key="append",
        )
        append_payload = json.loads(append_result.result_json)
        assert append_payload.get("status") == "appended"

        history_result = await kernel.step_tool(
            run_id="run_runtime_names",
            tenant=tenant,
            tool_name="query_history_v2",
            arguments=EventHistoryArgs(limit=10, event_type="all"),
            step_key="query_history",
        )
        history_payload = json.loads(history_result.result_json)
        assert history_payload.get("ok") is True
        assert isinstance(history_payload.get("events"), list)
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
