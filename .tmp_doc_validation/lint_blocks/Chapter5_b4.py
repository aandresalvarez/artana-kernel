import asyncio

task_queue = asyncio.Queue()

async def worker():
    while True:
        run_id, tenant = await task_queue.get()
        worker_id = "worker-1"
        leased = await kernel.acquire_run_lease(
            run_id=run_id,
            worker_id=worker_id,
            ttl_seconds=30,
        )
        if not leased:
            task_queue.task_done()
            continue
        harness = DeploymentHarness(kernel=kernel, tenant=tenant)
        try:
            await harness.run(run_id)
        finally:
            await kernel.release_run_lease(run_id=run_id, worker_id=worker_id)
            task_queue.task_done()