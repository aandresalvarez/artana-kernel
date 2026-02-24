import asyncio
from artana.harness import IncrementalTaskHarness, SupervisorHarness, TaskUnit
from artana.kernel import ArtanaKernel
from artana.models import TenantContext
from artana.store import SQLiteStore


class DeploymentHarness(IncrementalTaskHarness):

    async def define_tasks(self):
        return [
            TaskUnit(id="build", description="Build artifacts"),
            TaskUnit(id="deploy", description="Deploy services"),
        ]

    async def work_on(self, task: TaskUnit):
        print("Executing:", task.id)


async def main():
    kernel = ArtanaKernel(
        store=SQLiteStore("chapter4_step4.db"),
        model_port=None,
    )

    tenant = TenantContext(
        tenant_id="ops",
        capabilities=frozenset(),
        budget_usd_limit=5.0,
    )

    supervisor = SupervisorHarness(kernel=kernel, tenant=tenant)
    deployment = DeploymentHarness(kernel=kernel, tenant=tenant)

    result = await supervisor.run_child(
        harness=deployment,
        run_id="deployment_run",
    )

    print("Deployment state:", result)
    await kernel.close()


asyncio.run(main())