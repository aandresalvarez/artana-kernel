status = await kernel.get_run_status(run_id="billing_run")
resume_point = await kernel.resume_point(run_id="billing_run")
active_runs = await kernel.list_active_runs(tenant_id=tenant.tenant_id)

await kernel.acquire_run_lease(
    run_id="billing_run",
    worker_id="worker_a",
    ttl_seconds=30,
)