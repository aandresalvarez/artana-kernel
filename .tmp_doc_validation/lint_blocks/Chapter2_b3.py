from artana.kernel import ReplayPolicy

harness = DataPipelineHarness(
    kernel=kernel,
    tenant=tenant,
)

# Strict mode (default safety)
await harness.run("run_strict")

# Allow minor prompt drift
harness = DataPipelineHarness(
    kernel=kernel,
    tenant=tenant,
    replay_policy="allow_prompt_drift",
)

await harness.run("run_drift_safe")