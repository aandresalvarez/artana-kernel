result = await kernel.step_model(
    run_id="long_run",
    tenant=tenant,
    model="demo-model",
    input=ModelInput.from_prompt("New improved prompt"),
    output_schema=Decision,
    step_key="analysis_step",
    replay_policy="fork_on_drift",
)