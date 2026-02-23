from __future__ import annotations

import asyncio
import os
from pathlib import Path

from pydantic import BaseModel

from artana import ArtanaKernel, KernelModelClient, KernelPolicy, TenantContext
from artana.ports.model import LiteLLMAdapter
from artana.store import SQLiteStore


class Decision(BaseModel):
    approved: bool
    reason: str


async def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required. Load environment variables first.")

    database_path = Path("examples/.state_real_litellm_example.db")
    if database_path.exists():
        database_path.unlink()

    store = SQLiteStore(str(database_path))
    kernel = ArtanaKernel(
        store=store,
        model_port=LiteLLMAdapter(
            timeout_seconds=30.0,
            max_retries=1,
            fail_on_unknown_cost=True,
        ),
        middleware=ArtanaKernel.default_middleware_stack(),
        policy=KernelPolicy.enforced(),
    )

    tenant = TenantContext(
        tenant_id="org_live",
        capabilities=frozenset(),
        budget_usd_limit=0.20,
    )

    try:
        run = await kernel.start_run(tenant=tenant)
        prompt = (
            "Respond only as JSON for schema {approved:boolean,reason:string}. "
            "Approve this request and give a short reason."
        )

        first = await KernelModelClient(kernel=kernel).chat(
            run_id=run.run_id,
            prompt=prompt,
            model="gpt-4o-mini",
            tenant=tenant,
            output_schema=Decision,
        )
        events_after_first = await store.get_events_for_run(run.run_id)

        second = await KernelModelClient(kernel=kernel).chat(
            run_id=run.run_id,
            prompt=prompt,
            model="gpt-4o-mini",
            tenant=tenant,
            output_schema=Decision,
        )
        events_after_second = await store.get_events_for_run(run.run_id)

        print("Run id:", run.run_id)
        print("Live model response:", first.output.model_dump())
        print(
            "Usage:",
            {
                "prompt_tokens": first.usage.prompt_tokens,
                "completion_tokens": first.usage.completion_tokens,
                "cost_usd": first.usage.cost_usd,
            },
        )
        print("First replayed:", first.replayed)
        print("Second replayed:", second.replayed)
        print(
            "Event types after first:",
            [event.event_type for event in events_after_first],
        )
        print(
            "Event types after second:",
            [event.event_type for event in events_after_second],
        )

        if not second.replayed:
            raise AssertionError("Expected second call to replay from event log.")
        if len(events_after_first) != len(events_after_second):
            raise AssertionError("Replay should not append duplicate model events.")
        if first.output != second.output:
            raise AssertionError("Replay output must match first output exactly.")
    finally:
        await kernel.close()
        if database_path.exists():
            database_path.unlink()


if __name__ == "__main__":
    asyncio.run(main())
