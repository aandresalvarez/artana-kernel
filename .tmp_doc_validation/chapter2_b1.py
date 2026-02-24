import asyncio
from pydantic import BaseModel

from artana.harness import IncrementalTaskHarness, SupervisorHarness, TaskUnit
from artana.kernel import ArtanaKernel
from artana.models import TenantContext
from artana.store import SQLiteStore


class ResearchHarness(IncrementalTaskHarness):

    async def define_tasks(self):
        return [
            TaskUnit(id="fact", description="Provide a historical fact"),
        ]

    async def work_on(self, task: TaskUnit):
        print("Research task executed:", task.id)


async def main():
    kernel = ArtanaKernel(
        store=SQLiteStore("chapter2_step1.db"),
        model_port=None,
    )

    tenant = TenantContext(
        tenant_id="manager",
        capabilities=frozenset(),
        budget_usd_limit=5.0,
    )

    supervisor = SupervisorHarness(kernel=kernel, tenant=tenant)
    child_harness = ResearchHarness(kernel=kernel, tenant=tenant)

    result = await supervisor.run_child(
        harness=child_harness,
        run_id="swarm_run_01"
    )

    print("Child task states:", result)
    await kernel.close()


asyncio.run(main())