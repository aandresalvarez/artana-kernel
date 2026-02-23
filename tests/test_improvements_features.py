from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TypeVar

import pytest
from pydantic import BaseModel

from artana import AutonomousAgent, ContextBuilder, KernelModelClient
from artana.events import (
    EventType,
    ModelRequestedPayload,
    ReplayedWithDriftPayload,
    RunSummaryPayload,
    ToolCompletedPayload,
    ToolRequestedPayload,
)
from artana.kernel import ArtanaKernel, ModelInput, ReplayConsistencyError
from artana.middleware import CapabilityGuardMiddleware, PIIScrubberMiddleware
from artana.models import TenantContext
from artana.ports.model import ModelRequest, ModelResult, ModelUsage
from artana.store import SQLiteStore

OutputModelT = TypeVar("OutputModelT", bound=BaseModel)


class Decision(BaseModel):
    approved: bool
    reason: str


class AgentResult(BaseModel):
    done: bool


class CountingModelPort:
    def __init__(self) -> None:
        self.calls = 0

    async def complete(
        self, request: ModelRequest[OutputModelT]
    ) -> ModelResult[OutputModelT]:
        self.calls += 1
        output = request.output_schema.model_validate(
            {"approved": True, "reason": f"call-{self.calls}"}
        )
        return ModelResult(
            output=output,
            usage=ModelUsage(prompt_tokens=3, completion_tokens=2, cost_usd=0.01),
        )


class AlwaysDoneModelPort:
    async def complete(
        self, request: ModelRequest[OutputModelT]
    ) -> ModelResult[OutputModelT]:
        output = request.output_schema.model_validate({"done": True})
        return ModelResult(
            output=output,
            usage=ModelUsage(prompt_tokens=4, completion_tokens=2, cost_usd=0.01),
        )


class EchoModelPort:
    async def complete(
        self, request: ModelRequest[OutputModelT]
    ) -> ModelResult[OutputModelT]:
        output = request.output_schema.model_validate({"approved": True, "reason": "ok"})
        return ModelResult(
            output=output,
            usage=ModelUsage(prompt_tokens=2, completion_tokens=1, cost_usd=0.01),
        )


class LegacyModelOnlyMiddleware:
    async def prepare_model(self, invocation: object) -> object:
        return invocation

    async def before_model(self, *, run_id: str, tenant: TenantContext) -> None:
        return None

    async def after_model(
        self,
        *,
        run_id: str,
        tenant: TenantContext,
        usage: ModelUsage,
    ) -> None:
        return None


class EchoArgs(BaseModel):
    email: str


def _tenant() -> TenantContext:
    return TenantContext(
        tenant_id="org_improvements",
        capabilities=frozenset(),
        budget_usd_limit=2.0,
    )


@pytest.mark.asyncio
async def test_replay_policy_allow_prompt_drift_replays_completed_model(tmp_path: Path) -> None:
    store = SQLiteStore(str(tmp_path / "state.db"))
    model_port = CountingModelPort()
    kernel = ArtanaKernel(store=store, model_port=model_port)
    client = KernelModelClient(kernel=kernel)

    try:
        first = await client.step(
            run_id="run_allow_drift",
            tenant=_tenant(),
            model="gpt-4o-mini",
            prompt="first prompt",
            output_schema=Decision,
            step_key="decision",
        )
        second = await client.step(
            run_id="run_allow_drift",
            tenant=_tenant(),
            model="gpt-4o-mini",
            prompt="second prompt with drift",
            output_schema=Decision,
            step_key="decision",
            replay_policy="allow_prompt_drift",
        )

        assert first.replayed is False
        assert second.replayed is True
        assert second.replayed_with_drift is True
        assert model_port.calls == 1

        events = await store.get_events_for_run("run_allow_drift")
        drift_events = [
            event
            for event in events
            if event.event_type == EventType.REPLAYED_WITH_DRIFT
        ]
        assert len(drift_events) == 1
        assert isinstance(drift_events[0].payload, ReplayedWithDriftPayload)
        assert drift_events[0].payload.replay_policy == "allow_prompt_drift"
    finally:
        await kernel.close()


@pytest.mark.asyncio
async def test_replay_policy_fork_on_drift_forks_run(tmp_path: Path) -> None:
    store = SQLiteStore(str(tmp_path / "state.db"))
    model_port = CountingModelPort()
    kernel = ArtanaKernel(store=store, model_port=model_port)
    tenant = _tenant()

    try:
        await kernel.start_run(tenant=tenant, run_id="run_fork")
        first = await kernel.step_model(
            run_id="run_fork",
            tenant=tenant,
            model="gpt-4o-mini",
            input=ModelInput.from_prompt("stable prompt"),
            output_schema=Decision,
            step_key="decision",
        )
        second = await kernel.step_model(
            run_id="run_fork",
            tenant=tenant,
            model="gpt-4o-mini",
            input=ModelInput.from_prompt("changed prompt"),
            output_schema=Decision,
            step_key="decision",
            replay_policy="fork_on_drift",
        )

        assert first.run_id == "run_fork"
        assert second.run_id.startswith("run_fork::fork::")
        assert second.forked_from_run_id == "run_fork"
        assert second.replayed is False
        assert model_port.calls == 2

        original_events = await store.get_events_for_run("run_fork")
        fork_events = await store.get_events_for_run(second.run_id)
        assert any(
            event.event_type == EventType.REPLAYED_WITH_DRIFT
            for event in original_events
        )
        assert [event.event_type for event in fork_events[:2]] == [
            EventType.RUN_STARTED,
            EventType.MODEL_REQUESTED,
        ]
    finally:
        await kernel.close()


@pytest.mark.asyncio
async def test_replay_rejects_tool_signature_schema_drift(tmp_path: Path) -> None:
    database_path = tmp_path / "state.db"

    first_kernel = ArtanaKernel(
        store=SQLiteStore(str(database_path)),
        model_port=CountingModelPort(),
    )

    @first_kernel.tool()
    async def submit_transfer(account_id: str) -> str:
        return '{"ok":true}'

    try:
        await KernelModelClient(kernel=first_kernel).step(
            run_id="run_tool_schema_drift",
            tenant=_tenant(),
            model="gpt-4o-mini",
            prompt="schema check",
            output_schema=Decision,
            step_key="decision",
        )
    finally:
        await first_kernel.close()

    second_kernel = ArtanaKernel(
        store=SQLiteStore(str(database_path)),
        model_port=CountingModelPort(),
    )

    @second_kernel.tool()
    async def submit_transfer_v2(account_id: str, amount: str) -> str:
        return '{"ok":true}'

    try:
        with pytest.raises(ReplayConsistencyError, match="allowed tool signatures"):
            await KernelModelClient(kernel=second_kernel).step(
                run_id="run_tool_schema_drift",
                tenant=_tenant(),
                model="gpt-4o-mini",
                prompt="schema check",
                output_schema=Decision,
                step_key="decision",
            )
    finally:
        await second_kernel.close()


@pytest.mark.asyncio
async def test_pii_scrubber_applies_to_tool_request_and_result(tmp_path: Path) -> None:
    store = SQLiteStore(str(tmp_path / "state.db"))
    kernel = ArtanaKernel(
        store=store,
        model_port=EchoModelPort(),
        middleware=[PIIScrubberMiddleware()],
    )
    tenant = _tenant()

    @kernel.tool()
    async def echo_private(email: str) -> str:
        return json.dumps({"echo": email, "status": "ok"})

    try:
        await kernel.start_run(tenant=tenant, run_id="run_tool_pii")
        result = await kernel.step_tool(
            run_id="run_tool_pii",
            tenant=tenant,
            tool_name="echo_private",
            arguments=EchoArgs(email="alice@example.com"),
            step_key="tool_step",
        )
        assert "alice@example.com" not in result.result_json
        assert "[REDACTED_EMAIL]" in result.result_json

        events = await store.get_events_for_run("run_tool_pii")
        requested_payload = events[1].payload
        completed_payload = events[2].payload
        assert isinstance(requested_payload, ToolRequestedPayload)
        assert isinstance(completed_payload, ToolCompletedPayload)
        assert "alice@example.com" not in requested_payload.arguments_json
        assert "[REDACTED_EMAIL]" in requested_payload.arguments_json
        assert "alice@example.com" not in completed_payload.result_json
        assert "[REDACTED_EMAIL]" in completed_payload.result_json
    finally:
        await kernel.close()


@pytest.mark.asyncio
async def test_legacy_middleware_without_tool_hooks_does_not_break_tools(
    tmp_path: Path,
) -> None:
    store = SQLiteStore(str(tmp_path / "state.db"))
    kernel = ArtanaKernel(
        store=store,
        model_port=EchoModelPort(),
        middleware=[LegacyModelOnlyMiddleware()],
    )
    tenant = _tenant()

    @kernel.tool()
    async def echo_private(email: str) -> str:
        return json.dumps({"echo": email, "status": "ok"})

    try:
        await kernel.start_run(tenant=tenant, run_id="run_legacy_middleware_tool")
        result = await kernel.step_tool(
            run_id="run_legacy_middleware_tool",
            tenant=tenant,
            tool_name="echo_private",
            arguments=EchoArgs(email="alice@example.com"),
            step_key="tool_step",
        )
        payload = json.loads(result.result_json)
        assert payload["echo"] == "alice@example.com"
        assert payload["status"] == "ok"
    finally:
        await kernel.close()


@pytest.mark.asyncio
async def test_autonomous_agent_emits_run_summary_event(tmp_path: Path) -> None:
    store = SQLiteStore(str(tmp_path / "state.db"))
    kernel = ArtanaKernel(store=store, model_port=AlwaysDoneModelPort())
    agent = AutonomousAgent(
        kernel=kernel,
        context_builder=ContextBuilder(progressive_skills=False),
    )

    try:
        result = await agent.run(
            run_id="run_summary_events",
            tenant=_tenant(),
            model="gpt-4o-mini",
            prompt="finish quickly",
            output_schema=AgentResult,
            max_iterations=2,
        )
        assert result.done is True

        events = await store.get_events_for_run("run_summary_events")
        assert any(
            event.event_type == EventType.RUN_SUMMARY
            and isinstance(event.payload, RunSummaryPayload)
            and event.payload.summary_type == "agent_model_step"
            for event in events
        )
    finally:
        await kernel.close()


@pytest.mark.asyncio
async def test_model_requested_always_records_context_version(tmp_path: Path) -> None:
    store = SQLiteStore(str(tmp_path / "state.db"))
    kernel = ArtanaKernel(store=store, model_port=CountingModelPort())
    client = KernelModelClient(kernel=kernel)

    try:
        prompt = "approve transfer"
        await client.step(
            run_id="run_context_version_defaults",
            tenant=_tenant(),
            model="gpt-4o-mini",
            prompt=prompt,
            output_schema=Decision,
            step_key="decision",
        )

        events = await store.get_events_for_run("run_context_version_defaults")
        model_requested = next(
            event.payload
            for event in events
            if event.event_type == EventType.MODEL_REQUESTED
        )
        assert isinstance(model_requested, ModelRequestedPayload)
        assert model_requested.context_version is not None
        assert model_requested.context_version.system_prompt_hash == hashlib.sha256(
            prompt.encode("utf-8")
        ).hexdigest()
        assert model_requested.context_version.context_builder_version == "unknown"
        assert model_requested.context_version.compaction_version == "unknown"
    finally:
        await kernel.close()


@pytest.mark.asyncio
async def test_step_model_emits_capability_decision_summary(tmp_path: Path) -> None:
    store = SQLiteStore(str(tmp_path / "state.db"))
    kernel = ArtanaKernel(
        store=store,
        model_port=CountingModelPort(),
        middleware=[CapabilityGuardMiddleware()],
    )
    tenant = _tenant()

    @kernel.tool()
    async def public_lookup(account_id: str) -> str:
        return json.dumps({"ok": True, "account_id": account_id})

    @kernel.tool(requires_capability="payments")
    async def restricted_transfer(account_id: str) -> str:
        return json.dumps({"ok": True, "account_id": account_id})

    try:
        await kernel.start_run(tenant=tenant, run_id="run_capability_decisions")
        await kernel.step_model(
            run_id="run_capability_decisions",
            tenant=tenant,
            model="gpt-4o-mini",
            input=ModelInput.from_prompt("summarize account state"),
            output_schema=Decision,
            step_key="decision",
        )

        events = await store.get_events_for_run("run_capability_decisions")
        summaries = [
            event.payload
            for event in events
            if event.event_type == EventType.RUN_SUMMARY
            and isinstance(event.payload, RunSummaryPayload)
            and event.payload.summary_type == "capability_decision"
        ]
        assert len(summaries) == 1
        summary_payload = json.loads(summaries[0].summary_json)
        decisions = {
            decision["tool_name"]: decision
            for decision in summary_payload["decisions"]
        }
        assert decisions["public_lookup"]["decision"] == "allowed"
        assert decisions["public_lookup"]["reason"] == "allowed_no_capability_required"
        assert decisions["restricted_transfer"]["decision"] == "filtered"
        assert decisions["restricted_transfer"]["reason"] == "filtered_missing_capability"
        assert "public_lookup" in summary_payload["final_allowed_tools"]
        assert "restricted_transfer" not in summary_payload["final_allowed_tools"]
    finally:
        await kernel.close()
