import asyncio
from artana.harness import IncrementalTaskHarness, TaskUnit
from artana.kernel import ArtanaKernel
from artana.models import TenantContext
from artana.store import SQLiteStore


class ResearchHarness(IncrementalTaskHarness):

    async def define_tasks(self):
        return [
            TaskUnit(id="collect", description="Collect data"),
            TaskUnit(id="analyze", description="Analyze data"),
            TaskUnit(id="summarize", description="Write summary"),
        ]

    async def work_on(self, task: TaskUnit):
        print("Working on:", task.id)


async def main():
    kernel = ArtanaKernel(
        store=SQLiteStore("harness.db"),
        model_port=None,
    )

    tenant = TenantContext(
        tenant_id="research_team",
        capabilities=frozenset(),
        budget_usd_limit=1.0,
    )

    harness = ResearchHarness(kernel=kernel, tenant=tenant)

    progress = await harness.run("research_run")

    print("Task states:", progress)
    await kernel.close()


asyncio.run(main())