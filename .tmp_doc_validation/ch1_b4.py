import asyncio
from artana.kernel import ArtanaKernel, WorkflowContext, json_step_serde
from artana.models import TenantContext
from artana.store import SQLiteStore


async def main():
    kernel = ArtanaKernel(
        store=SQLiteStore("workflow.db"),
        model_port=None,  # not needed here
    )

    tenant = TenantContext(
        tenant_id="workflow_user",
        capabilities=frozenset(),
        budget_usd_limit=1.0,
    )

    async def my_workflow(ctx: WorkflowContext):
        step1 = await ctx.step(
            name="compute_value",
            action=lambda: asyncio.sleep(0, result=42),
            serde=json_step_serde(),
        )

        if step1 == 42:
            await ctx.pause(reason="Confirm value before proceeding")

        return "Finished"

    first = await kernel.run_workflow(
        run_id="workflow_run",
        tenant=tenant,
        workflow=my_workflow,
    )

    print("status:", first.status)
    await kernel.close()


asyncio.run(main())