from artana.kernel import ReplayPolicy

# Strict replay (default safety)
await kernel.step_model(..., replay_policy="strict")

# Allow prompt drift while preserving prior outputs
await kernel.step_model(..., replay_policy="allow_prompt_drift")

# Fork run automatically if prompt changed
await kernel.step_model(..., replay_policy="fork_on_drift")