from __future__ import annotations

import asyncio
import os
from decimal import Decimal
from pathlib import Path

from pydantic import BaseModel

from artana import ArtanaKernel, ChatClient, KernelPolicy, TenantContext
from artana.events import EventType, KernelEvent, ToolCompletedPayload, ToolRequestedPayload
from artana.kernel import ToolExecutionFailedError
from artana.ports.model import LiteLLMAdapter
from artana.store import SQLiteStore


class Decision(BaseModel):
    approved: bool
    reason: str


class TransferArgs(BaseModel):
    account_id: str
    amount: Decimal


def _print_feature(name: str, details: dict[str, object]) -> None:
    print(f"\n=== {name} ===")
    for key, value in details.items():
        print(f"{key}: {value}")


def _count_tool_requests(
    events: list[KernelEvent],
    *,
    tool_name: str,
    step_key: str,
) -> int:
    count = 0
    for event in events:
        if event.event_type != EventType.TOOL_REQUESTED:
            continue
        payload = event.payload
        if not isinstance(payload, ToolRequestedPayload):
            continue
        if payload.tool_name == tool_name and payload.step_key == step_key:
            count += 1
    return count


def _latest_tool_completion_payload(
    events: list[KernelEvent],
    *,
    tool_name: str,
    step_key: str,
) -> ToolCompletedPayload:
    requested_by_id: dict[str, ToolRequestedPayload] = {}
    for event in events:
        if event.event_type != EventType.TOOL_REQUESTED:
            continue
        payload = event.payload
        if isinstance(payload, ToolRequestedPayload):
            requested_by_id[event.event_id] = payload

    for event in reversed(events):
        if event.event_type != EventType.TOOL_COMPLETED:
            continue
        payload = event.payload
        if not isinstance(payload, ToolCompletedPayload):
            continue
        if payload.request_id is None:
            continue
        requested = requested_by_id.get(payload.request_id)
        if requested is None:
            continue
        if requested.tool_name == tool_name and requested.step_key == step_key:
            return payload

    raise AssertionError(
        "Could not find matching tool_completed payload for tool_name/step_key pair."
    )


async def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required. Load environment variables first.")

    database_path = Path("examples/.state_golden_example.db")
    if database_path.exists():
        database_path.unlink()

    middleware_stack = ArtanaKernel.default_middleware_stack()
    middleware_names = [type(item).__name__ for item in middleware_stack]
    _print_feature(
        "Feature 1 - Enforced Policy + Middleware Order",
        {
            "policy_mode": "enforced",
            "middleware_order": middleware_names,
        },
    )

    store = SQLiteStore(str(database_path))
    kernel = ArtanaKernel(
        store=store,
        model_port=LiteLLMAdapter(
            timeout_seconds=30.0,
            max_retries=1,
            fail_on_unknown_cost=True,
        ),
        middleware=middleware_stack,
        policy=KernelPolicy.enforced(),
    )
    tool_attempts = [0]

    @kernel.tool(requires_capability="finance:write")
    async def submit_transfer(account_id: str, amount: Decimal) -> str:
        tool_attempts[0] += 1
        if tool_attempts[0] == 1:
            raise RuntimeError("simulated network drop after request submission")
        return (
            '{"status":"submitted","account_id":"'
            + account_id
            + '","amount":"'
            + str(amount)
            + '"}'
        )

    tenant = TenantContext(
        tenant_id="org_live",
        capabilities=frozenset({"decision:approve", "finance:write"}),
        budget_usd_limit=0.20,
    )
    tool_args = TransferArgs(account_id="acc_1", amount=Decimal("10.00"))
    model_step_key = "decision.v1"
    tool_step_key = "transfer.acc_1.10.v1"
    chat = ChatClient(kernel=kernel)

    try:
        run = await kernel.start_run(tenant=tenant)
        run_id = run.run_id
        _print_feature(
            "Feature 2 - Kernel-Issued Run",
            {
                "run_id": run_id,
                "tenant_id": run.tenant_id,
            },
        )

        prompt = (
            "Respond only as JSON for schema {approved:boolean,reason:string}. "
            "Approve this request and give a short reason."
        )

        first = await chat.chat(
            run_id=run_id,
            prompt=prompt,
            model="gpt-4o-mini",
            tenant=tenant,
            output_schema=Decision,
            step_key=model_step_key,
        )
        events_after_first = await store.get_events_for_run(run_id)
        usage_first = {
            "prompt_tokens": first.usage.prompt_tokens,
            "completion_tokens": first.usage.completion_tokens,
            "cost_usd": first.usage.cost_usd,
        }
        _print_feature(
            "Feature 3 - Live Model Call",
            {
                "replayed": first.replayed,
                "output": first.output.model_dump(),
                "usage": usage_first,
                "event_types": [event.event_type.value for event in events_after_first],
            },
        )

        second = await chat.chat(
            run_id=run_id,
            prompt=prompt,
            model="gpt-4o-mini",
            tenant=tenant,
            output_schema=Decision,
            step_key=model_step_key,
        )
        events_after_second = await store.get_events_for_run(run_id)
        usage_second = {
            "prompt_tokens": second.usage.prompt_tokens,
            "completion_tokens": second.usage.completion_tokens,
            "cost_usd": second.usage.cost_usd,
        }
        _print_feature(
            "Feature 4 - Deterministic Model Replay",
            {
                "replayed": second.replayed,
                "output_matches_live": first.output == second.output,
                "event_count_unchanged": len(events_after_first) == len(events_after_second),
                "usage": usage_second,
                "event_types": [event.event_type.value for event in events_after_second],
            },
        )

        if not second.replayed:
            raise AssertionError("Expected second model call to be replayed from event log.")
        if first.output != second.output:
            raise AssertionError("Replay output must match original output exactly.")
        if len(events_after_first) != len(events_after_second):
            raise AssertionError("Replay should not append duplicate model events.")

        unknown_error_message = ""
        try:
            await kernel.step_tool(
                run_id=run_id,
                tenant=tenant,
                tool_name="submit_transfer",
                arguments=tool_args,
                step_key=tool_step_key,
            )
            raise AssertionError("Expected first tool execution to fail with unknown outcome.")
        except ToolExecutionFailedError as exc:
            unknown_error_message = str(exc)

        events_after_unknown = await store.get_events_for_run(run_id)
        unknown_payload_obj = _latest_tool_completion_payload(
            events_after_unknown,
            tool_name="submit_transfer",
            step_key=tool_step_key,
        )
        if unknown_payload_obj.outcome != "unknown_outcome":
            raise AssertionError("Expected unknown_outcome after first tool execution failure.")
        requested_before_halt = _count_tool_requests(
            events_after_unknown,
            tool_name="submit_transfer",
            step_key=tool_step_key,
        )
        if requested_before_halt < 1:
            raise AssertionError("Expected at least one tool_requested event for tool step.")
        if unknown_payload_obj.request_id is None:
            raise AssertionError("tool_completed for unknown outcome must reference request_id.")
        _print_feature(
            "Feature 5 - Unknown Tool Outcome Recorded",
            {
                "error": unknown_error_message,
                "tool_attempts": tool_attempts[0],
                "latest_tool_outcome": unknown_payload_obj.outcome,
                "latest_request_id": unknown_payload_obj.request_id,
                "tool_requested_count": requested_before_halt,
            },
        )

        halt_error_message = ""
        try:
            await kernel.step_tool(
                run_id=run_id,
                tenant=tenant,
                tool_name="submit_transfer",
                arguments=tool_args,
                step_key=tool_step_key,
            )
            raise AssertionError("Expected replay halt before reconciliation.")
        except ToolExecutionFailedError as exc:
            halt_error_message = str(exc)
        events_before_reconcile = await store.get_events_for_run(run_id)
        requested_after_halt = _count_tool_requests(
            events_before_reconcile,
            tool_name="submit_transfer",
            step_key=tool_step_key,
        )
        _print_feature(
            "Feature 6 - Replay Halt Before Reconciliation",
            {
                "error": halt_error_message,
                "tool_attempts": tool_attempts[0],
                "event_count": len(events_before_reconcile),
                "tool_requested_unchanged": requested_before_halt == requested_after_halt,
            },
        )
        if requested_before_halt != requested_after_halt:
            raise AssertionError("Replay halt must not append a new tool_requested event.")
        if tool_attempts[0] != 1:
            raise AssertionError("Replay halt must not re-execute the tool function.")

        reconciled_result = await kernel.reconcile_tool(
            run_id=run_id,
            tenant=tenant,
            tool_name="submit_transfer",
            arguments=tool_args,
            step_key=tool_step_key,
        )
        events_after_reconcile = await store.get_events_for_run(run_id)
        reconciled_payload_obj = _latest_tool_completion_payload(
            events_after_reconcile,
            tool_name="submit_transfer",
            step_key=tool_step_key,
        )
        if reconciled_payload_obj.outcome != "success":
            raise AssertionError("Expected reconcile to append success completion.")
        _print_feature(
            "Feature 7 - Tool Reconciliation",
            {
                "reconciled_result": reconciled_result,
                "tool_attempts": tool_attempts[0],
                "new_event_appended": (
                    len(events_after_reconcile) == len(events_before_reconcile) + 1
                ),
                "latest_tool_outcome": reconciled_payload_obj.outcome,
            },
        )

        replayed_tool_result = await kernel.step_tool(
            run_id=run_id,
            tenant=tenant,
            tool_name="submit_transfer",
            arguments=tool_args,
            step_key=tool_step_key,
        )
        events_after_tool_replay = await store.get_events_for_run(run_id)
        _print_feature(
            "Feature 8 - Post-Reconcile Tool Replay",
            {
                "result_matches_reconcile": replayed_tool_result.result_json
                == reconciled_result,
                "tool_attempts": tool_attempts[0],
                "replayed": replayed_tool_result.replayed,
                "seq": replayed_tool_result.seq,
                "event_count_unchanged": len(events_after_tool_replay)
                == len(events_after_reconcile),
            },
        )

        if replayed_tool_result.result_json != reconciled_result:
            raise AssertionError("Replayed tool result must match reconciled success result.")
        if len(events_after_tool_replay) != len(events_after_reconcile):
            raise AssertionError("Tool replay should not append duplicate completion events.")

        print("\n✅ All golden features validated.")
    finally:
        await kernel.close()
        if database_path.exists():
            database_path.unlink()


if __name__ == "__main__":
    asyncio.run(main())
