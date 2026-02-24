import asyncio
from artana.harness import IncrementalTaskHarness, TaskUnit
from artana.kernel import ArtanaKernel
from artana.models import TenantContext
from artana.store import SQLiteStore


class DataPipelineHarness(IncrementalTaskHarness):

    async def define_tasks(self):
        return [
            TaskUnit(id="ingest", description="Ingest data"),
            TaskUnit(id="transform", description="Transform data"),
            TaskUnit(id="validate", description="Validate results"),
        ]

    async def work_on(self, task: TaskUnit):
        print("Executing:", task.id)


async def main():
    kernel = ArtanaKernel(
        store=SQLiteStore("chapter2_step2.db"),
        model_port=None,
    )

    tenant = TenantContext(
        tenant_id="pipeline_team",
        capabilities=frozenset(),
        budget_usd_limit=5.0,
    )

    harness = DataPipelineHarness(kernel=kernel, tenant=tenant)

    progress = await harness.run("pipeline_run_001")
    print("Progress snapshot:", progress)

    await kernel.close()


asyncio.run(main())