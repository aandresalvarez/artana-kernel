from __future__ import annotations

import asyncio
import os
from pathlib import Path

from pydantic import BaseModel

from artana import ArtanaKernel, KernelModelClient, ModelCallOptions, TenantContext
from artana.ports.model import LiteLLMAdapter
from artana.store import SQLiteStore


class Decision(BaseModel):
    approved: bool
    reason: str


async def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required. Load environment variables first.")

    database_path = Path("examples/.state_responses_mode.db")
    if database_path.exists():
        database_path.unlink()

    store = SQLiteStore(str(database_path))
    kernel = ArtanaKernel(
        store=store,
        model_port=LiteLLMAdapter(timeout_seconds=30.0, max_retries=1),
        middleware=ArtanaKernel.default_middleware_stack(),
    )

    tenant = TenantContext(
        tenant_id="org_responses",
        capabilities=frozenset(),
        budget_usd_limit=0.50,
    )

    try:
        run = await kernel.start_run(tenant=tenant)

        first = await KernelModelClient(kernel=kernel).step(
            run_id=run.run_id,
            prompt=(
                "Respond only as JSON for schema {approved:boolean,reason:string}. "
                "Approve this request and give a short reason."
            ),
            model="openai/gpt-5.3-codex",
            tenant=tenant,
            output_schema=Decision,
            model_options=ModelCallOptions(
                api_mode="responses",
                reasoning_effort="high",
                verbosity="medium",
            ),
        )

        second = await KernelModelClient(kernel=kernel).step(
            run_id=run.run_id,
            prompt=(
                "Respond only as JSON for schema {approved:boolean,reason:string}. "
                "Keep the approval and make the reason at most five words."
            ),
            model="openai/gpt-5.3-codex",
            tenant=tenant,
            output_schema=Decision,
            model_options=ModelCallOptions(
                api_mode="responses",
                previous_response_id=first.response_id,
            ),
        )

        print("Run id:", run.run_id)
        print("First api_mode_used:", first.api_mode_used)
        print("First response_id:", first.response_id)
        print("Second api_mode_used:", second.api_mode_used)
        print("Second response_id:", second.response_id)
        print("Second output:", second.output.model_dump())
    finally:
        await kernel.close()
        if database_path.exists():
            database_path.unlink()


if __name__ == "__main__":
    asyncio.run(main())
