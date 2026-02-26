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

from artana import ArtanaKernel, KernelModelClient, ModelCallOptions, TenantContext
from artana.ports.model import LiteLLMAdapter
from artana.store import SQLiteStore


class Decision(BaseModel):
    approved: bool
    reason: str


async def main() -> None:
    require_openai_api_key(script_name="08_responses_mode.py")
    model_name = resolve_model(env_var="ARTANA_RESPONSES_MODEL", default="openai/gpt-5.3-codex")
    print_example_header(
        title="08 - Responses Mode (OpenAI Responses API)",
        models={"responses": model_name},
    )

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

        first = await KernelModelClient(kernel).step(
            run_id=run.run_id,
            prompt=(
                "Respond only as JSON for schema {approved:boolean,reason:string}. "
                "Approve this request and give a short reason."
            ),
            model=model_name,
            tenant=tenant,
            output_schema=Decision,
            model_options=ModelCallOptions(
                api_mode="responses",
                reasoning_effort="high",
                verbosity="medium",
            ),
        )

        second = await KernelModelClient(kernel).step(
            run_id=run.run_id,
            prompt=(
                "Respond only as JSON for schema {approved:boolean,reason:string}. "
                "Keep the approval and make the reason at most five words."
            ),
            model=model_name,
            tenant=tenant,
            output_schema=Decision,
            model_options=ModelCallOptions(
                api_mode="responses",
                previous_response_id=first.response_id,
            ),
        )

        print_summary(
            payload={
                "run_id": run.run_id,
                "model": model_name,
                "first": {
                    "api_mode_used": first.api_mode_used,
                    "response_id": first.response_id,
                    "output": first.output.model_dump(),
                },
                "second": {
                    "api_mode_used": second.api_mode_used,
                    "response_id": second.response_id,
                    "output": second.output.model_dump(),
                },
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
        raise friendly_exit(script_name="08_responses_mode.py", error=exc) from exc
