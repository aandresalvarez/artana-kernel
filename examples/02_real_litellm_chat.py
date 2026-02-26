from __future__ import annotations

import asyncio
from pathlib import Path

from _live_example_utils import (
    friendly_exit,
    print_example_header,
    print_summary,
    require_openai_api_key,
    resolve_model,
)
from pydantic import BaseModel

from artana import ArtanaKernel, KernelModelClient, KernelPolicy, TenantContext
from artana.ports.model import LiteLLMAdapter
from artana.store import SQLiteStore


class Decision(BaseModel):
    approved: bool
    reason: str


async def main() -> None:
    require_openai_api_key(script_name="02_real_litellm_chat.py")
    model_name = resolve_model(env_var="ARTANA_MODEL", default="gpt-4o-mini")
    print_example_header(
        title="02 - Real LiteLLM Chat (OpenAI)",
        models={"primary": model_name},
    )

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

        first = await KernelModelClient(kernel).step(
            run_id=run.run_id,
            prompt=prompt,
            model=model_name,
            tenant=tenant,
            output_schema=Decision,
        )
        events_after_first = await store.get_events_for_run(run.run_id)

        second = await KernelModelClient(kernel).step(
            run_id=run.run_id,
            prompt=prompt,
            model=model_name,
            tenant=tenant,
            output_schema=Decision,
        )
        events_after_second = await store.get_events_for_run(run.run_id)

        if not second.replayed:
            raise AssertionError("Expected second call to replay from event log.")
        if len(events_after_first) != len(events_after_second):
            raise AssertionError("Replay should not append duplicate model events.")
        if first.output != second.output:
            raise AssertionError("Replay output must match first output exactly.")

        print_summary(
            payload={
                "run_id": run.run_id,
                "model": model_name,
                "first_replayed": first.replayed,
                "second_replayed": second.replayed,
                "output": first.output.model_dump(),
                "usage": {
                    "prompt_tokens": first.usage.prompt_tokens,
                    "completion_tokens": first.usage.completion_tokens,
                    "cost_usd": first.usage.cost_usd,
                },
                "event_count": len(events_after_second),
            }
        )
    finally:
        await kernel.close()
        if database_path.exists():
            database_path.unlink()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as exc:
        raise friendly_exit(script_name="02_real_litellm_chat.py", error=exc) from exc
