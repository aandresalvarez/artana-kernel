import asyncio
from artana.harness import IncrementalTaskHarness, TaskUnit
from artana.kernel import ArtanaKernel
from artana.models import TenantContext
from artana.store import SQLiteStore


class MigrationHarness(IncrementalTaskHarness):

    async def define_tasks(self):
        return [
            TaskUnit(id="backup", description="Backup DB"),
            TaskUnit(id="migrate", description="Run migrations"),
            TaskUnit(id="verify", description="Verify schema"),
        ]

    async def work_on(self, task: TaskUnit):
        print("Executing:", task.id)


async def main():
    kernel = ArtanaKernel(
        store=SQLiteStore("chapter3_step3.db"),
        model_port=None,
    )

    tenant = TenantContext(
        tenant_id="ops",
        capabilities=frozenset(),
        budget_usd_limit=5.0,
    )

    harness = MigrationHarness(kernel=kernel, tenant=tenant)

    await harness.run("migration_run")

    # Simulate restart:
    await harness.run("migration_run")

    await kernel.close()


asyncio.run(main())